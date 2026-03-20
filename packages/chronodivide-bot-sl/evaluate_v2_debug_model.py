from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model_lib.batch import collate_model_samples
from model_lib.dataset import ModelShardRecord, RA2SLSectionDataset, discover_model_shards
from model_lib.losses_v2 import compute_ra2_sl_v2_free_running_metrics
from model_lib.model_v2 import RA2SLV2DebugModel
from train_sl_model import move_batch_to_device
from train_sl_model_v2_debug import TrainConfigV2Debug, infer_v2_model_config
from transform_lib.label_layout_v2 import LABEL_LAYOUT_V2_ACTION_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a V2 debug checkpoint and report action-family confusion.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_records(run_dir: Path) -> tuple[list[ModelShardRecord], list[ModelShardRecord], Path]:
    data_split = load_json(run_dir / "data_split.json")
    tensor_dir = Path(str(data_split["config"]["tensor_dir"]))
    val_stems = {str(stem) for stem in data_split.get("valShards", [])}
    train_stems = {str(stem) for stem in data_split.get("trainShards", [])}
    all_records = discover_model_shards(tensor_dir)
    val_records = [record for record in all_records if record.stem in val_stems]
    split_records = [record for record in all_records if record.stem in train_stems or record.stem in val_stems]
    if not val_records:
        raise ValueError(f"No V2 validation shards found for run dir: {run_dir}")
    if not split_records:
        raise ValueError(f"No V2 train/val shards found for run dir: {run_dir}")
    return val_records, split_records, tensor_dir


def confusion_rows_to_report(confusion: dict[tuple[int, int], int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (target_id, predicted_id), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "targetId": target_id,
                "targetName": LABEL_LAYOUT_V2_ACTION_FAMILIES[target_id],
                "predictedId": predicted_id,
                "predictedName": LABEL_LAYOUT_V2_ACTION_FAMILIES[predicted_id],
                "count": count,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()

    val_records, split_records, tensor_dir = load_split_records(run_dir)
    dataset = RA2SLSectionDataset(val_records, artifact_variant="v2", cache_size=2)
    data_loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    train_config = TrainConfigV2Debug(
        tensor_dir=Path(str(config_dict["tensor_dir"])),
        output_dir=Path(str(config_dict["output_dir"])),
        manifest_path=None if config_dict.get("manifest_path") is None else Path(str(config_dict["manifest_path"])),
        map_names=config_dict.get("map_names"),
        player_countries=config_dict.get("player_countries"),
        player_names=config_dict.get("player_names"),
        game_ids=config_dict.get("game_ids"),
        side_ids=config_dict.get("side_ids"),
        max_shards=config_dict.get("max_shards"),
        max_train_samples=config_dict.get("max_train_samples"),
        max_val_samples=config_dict.get("max_val_samples"),
        teacher_forcing_mode=str(config_dict.get("teacher_forcing_mode", "full")),
        action_family_weighting=str(config_dict.get("action_family_weighting", "none")),
        action_family_weight_min=float(config_dict.get("action_family_weight_min", 0.25)),
        action_family_weight_max=float(config_dict.get("action_family_weight_max", 4.0)),
        family_balanced_sampling=bool(config_dict.get("family_balanced_sampling", False)),
        batch_size=int(config_dict.get("batch_size", 8)),
        num_workers=int(config_dict.get("num_workers", 0)),
        epochs=int(config_dict.get("epochs", 1)),
        learning_rate=float(config_dict.get("learning_rate", 1e-3)),
        weight_decay=float(config_dict.get("weight_decay", 1e-4)),
        grad_clip_norm=float(config_dict.get("grad_clip_norm", 1.0)),
        val_ratio=float(config_dict.get("val_ratio", 0.2)),
        seed=int(config_dict.get("seed", 0)),
        cache_size=int(config_dict.get("cache_size", 2)),
        window_size=int(config_dict.get("window_size", 1)),
        window_stride=int(config_dict.get("window_stride", 1)),
        use_lstm_core=bool(config_dict.get("use_lstm_core", False)),
        lstm_num_layers=int(config_dict.get("lstm_num_layers", 1)),
        checkpoint_dir=Path(str(config_dict["checkpoint_dir"])),
        device=str(config_dict.get("device")) if config_dict.get("device") is not None else None,
    )
    model = RA2SLV2DebugModel(infer_v2_model_config(split_records, train_config))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    confusion_counter: Counter[tuple[int, int]] = Counter()
    target_counter: Counter[int] = Counter()
    predicted_counter: Counter[int] = Counter()
    metric_sums: dict[str, float] = {}
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["model_inputs"],
                teacher_forcing_targets=None,
                teacher_forcing_masks=None,
                teacher_forcing_mode="none",
            )
            metrics = compute_ra2_sl_v2_free_running_metrics(outputs, batch)
            batch_count += 1
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.detach().item())

            target_ids = torch.argmax(batch["training_targets"]["actionFamilyOneHot"], dim=-1).reshape(-1)
            predicted_ids = torch.argmax(outputs["actionFamilyLogits"], dim=-1).reshape(-1)
            target_mask = batch["training_masks"]["actionFamilyLossMask"].reshape(-1) > 0

            for target_id, predicted_id, active in zip(target_ids.tolist(), predicted_ids.tolist(), target_mask.tolist()):
                if not active:
                    continue
                confusion_counter[(int(target_id), int(predicted_id))] += 1
                target_counter[int(target_id)] += 1
                predicted_counter[int(predicted_id)] += 1

    metrics_report = {
        name: (value / max(1, batch_count))
        for name, value in metric_sums.items()
    }

    report = {
        "runDir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "tensorDir": str(tensor_dir),
        "valShardCount": len(val_records),
        "valSampleCount": len(dataset),
        "freeRunningMetrics": metrics_report,
        "targetFamilyCounts": {
            LABEL_LAYOUT_V2_ACTION_FAMILIES[action_family_id]: count
            for action_family_id, count in sorted(target_counter.items())
        },
        "predictedFamilyCounts": {
            LABEL_LAYOUT_V2_ACTION_FAMILIES[action_family_id]: count
            for action_family_id, count in sorted(predicted_counter.items())
        },
        "confusionRows": confusion_rows_to_report(dict(confusion_counter)),
        "topConfusions": confusion_rows_to_report(dict(confusion_counter))[:20],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
