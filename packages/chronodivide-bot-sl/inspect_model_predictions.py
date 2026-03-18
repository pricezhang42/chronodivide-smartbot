from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from action_dict import ACTION_TYPE_ID_TO_NAME
from model_lib.batch import collate_model_samples
from model_lib.dataset import ModelShardFilter, ModelShardRecord, RA2SLSectionDataset, discover_model_shards
from model_lib.model import RA2SLBaselineConfig, RA2SLBaselineModel


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RA2 SL model predictions on real saved tensor samples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--examples-per-motif", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("shardStems"), list):
        raise ValueError(f"Manifest is missing shardStems: {path}")
    return manifest


def build_shard_filter(args: argparse.Namespace) -> ModelShardFilter:
    return ModelShardFilter.create(
        map_names=parse_csv(args.map_name),
        player_country_names=parse_csv(args.player_country),
        player_names=parse_csv(args.player_name),
        replay_game_ids=parse_csv(args.game_id),
        player_side_ids=[args.side_id] if args.side_id is not None else None,
    )


def classify_action_motif(action_name: str) -> str | None:
    if action_name.startswith("Queue::Add::") or action_name.startswith("PlaceBuilding::") or action_name.startswith("Order::Deploy"):
        return "build_order"
    if action_name.startswith("Queue::Cancel::") or action_name.startswith("Queue::Hold::"):
        return "queue_heavy"
    if (
        action_name.startswith("Order::Attack::")
        or action_name.startswith("Order::ForceAttack::")
        or action_name.startswith("Order::AttackMove::")
        or action_name.startswith("Order::Guard::")
    ):
        return "combat"
    if action_name.startswith("ActivateSuperWeapon::"):
        return "superweapon"
    return None


def move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    return value


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_tensor_dir(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        manifest = load_manifest(args.manifest.resolve())
        tensor_dir = manifest.get("tensorDir")
        if not tensor_dir:
            raise ValueError(f"Manifest is missing tensorDir: {args.manifest}")
        return Path(tensor_dir).resolve()
    if args.tensor_dir is None:
        raise ValueError("Either --tensor-dir or --manifest is required.")
    return args.tensor_dir.resolve()


def resolve_records(args: argparse.Namespace, tensor_dir: Path) -> list[ModelShardRecord]:
    records = discover_model_shards(tensor_dir, shard_filter=build_shard_filter(args))
    if args.manifest is None:
        return records
    manifest = load_manifest(args.manifest.resolve())
    shard_stems = {str(stem) for stem in manifest["shardStems"]}
    filtered = [record for record in records if record.stem in shard_stems]
    if not filtered:
        raise ValueError("Manifest/filter combination matched zero shards.")
    return filtered


def infer_entity_name_vocab_size(records: list[ModelShardRecord]) -> int:
    max_size = 0
    for record in records:
        id_to_name = record.metadata["schema"]["sharedNameVocabulary"]["idToName"]
        max_size = max(max_size, int(len(id_to_name)))
    if max_size <= 0:
        raise ValueError("Could not infer entity-name vocabulary size from the selected records.")
    return max_size


def decode_entity_name(record: ModelShardRecord, token_id: int) -> str | None:
    if token_id < 0:
        return None
    id_to_name = record.metadata["schema"]["sharedNameVocabulary"]["idToName"]
    if token_id >= len(id_to_name):
        return None
    return str(id_to_name[token_id])


def decode_topk_action_types(logits: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits, dim=-1)
    k = min(int(top_k), int(probabilities.shape[-1]))
    values, indices = torch.topk(probabilities, k=k, dim=-1)
    return [
        {
            "actionTypeId": int(index.item()),
            "actionTypeName": ACTION_TYPE_ID_TO_NAME[int(index.item())],
            "probability": float(value.item()),
        }
        for value, index in zip(values, indices, strict=False)
    ]


def decode_grid_argmax(grid: torch.Tensor) -> dict[str, int]:
    height = int(grid.shape[-2])
    width = int(grid.shape[-1])
    flat_index = int(torch.argmax(grid.reshape(-1)).item())
    row = flat_index // width
    col = flat_index % width
    return {
        "row": row,
        "col": col,
    }


def inspect_sample(
    *,
    dataset: RA2SLSectionDataset,
    index: int,
    model: RA2SLBaselineModel,
    device: torch.device,
    top_k: int,
) -> dict[str, Any]:
    sample = dataset[index]
    batch = collate_model_samples([sample])
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        outputs = model(batch["model_inputs"])

    metadata = sample["metadata"]
    record = dataset.shard_records[int(metadata["shard_index"])]

    gold_action_id = int(torch.argmax(sample["training_targets"]["actionTypeOneHot"]).item())
    predicted_action_id = int(torch.argmax(outputs["actionTypeLogits"][0]).item())
    report: dict[str, Any] = {
        "globalIndex": int(index),
        "shardStem": str(metadata["shard_stem"]),
        "replayGameId": str(metadata["replay_game_id"]),
        "playerName": str(metadata["player_name"]),
        "playerCountryName": metadata["player_country_name"],
        "tick": int(sample["sample_context"]["ticks"].item()),
        "goldActionTypeId": gold_action_id,
        "goldActionTypeName": ACTION_TYPE_ID_TO_NAME[gold_action_id],
        "predictedActionTypeId": predicted_action_id,
        "predictedActionTypeName": ACTION_TYPE_ID_TO_NAME[predicted_action_id],
        "goldActionAvailable": int(sample["feature_sections"]["availableActionMask"][gold_action_id].item()) == 1,
        "topKActionTypes": decode_topk_action_types(outputs["actionTypeLogits"][0].detach().cpu(), top_k),
    }

    queue_active = int(sample["training_masks"]["queueLossMask"].item()) == 1
    if queue_active:
        gold_queue = int(torch.argmax(sample["training_targets"]["queueOneHot"]).item())
        pred_queue = int(torch.argmax(outputs["queueLogits"][0]).item())
        report["queue"] = {
            "gold": gold_queue,
            "predicted": pred_queue,
        }

    target_entity_active = int(sample["training_masks"]["targetEntityLossMask"].item()) == 1
    if target_entity_active:
        gold_entity_index = int(torch.argmax(sample["training_targets"]["targetEntityOneHot"]).item())
        pred_entity_index = int(torch.argmax(outputs["targetEntityLogits"][0]).item())
        gold_token = int(sample["feature_sections"]["entityNameTokens"][gold_entity_index].item())
        pred_token = int(sample["feature_sections"]["entityNameTokens"][pred_entity_index].item())
        report["targetEntity"] = {
            "goldIndex": gold_entity_index,
            "goldName": decode_entity_name(record, gold_token),
            "predictedIndex": pred_entity_index,
            "predictedName": decode_entity_name(record, pred_token),
        }

    target_location_active = int(sample["training_masks"]["targetLocationLossMask"].item()) == 1
    if target_location_active:
        gold_grid = decode_grid_argmax(sample["training_targets"]["targetLocationOneHot"])
        pred_grid = decode_grid_argmax(outputs["targetLocationLogits"][0].detach().cpu())
        report["targetLocation"] = {
            "gold": gold_grid,
            "predicted": pred_grid,
        }

    target_location2_active = int(sample["training_masks"]["targetLocation2LossMask"].item()) == 1
    if target_location2_active:
        gold_grid = decode_grid_argmax(sample["training_targets"]["targetLocation2OneHot"])
        pred_grid = decode_grid_argmax(outputs["targetLocation2Logits"][0].detach().cpu())
        report["targetLocation2"] = {
            "gold": gold_grid,
            "predicted": pred_grid,
        }

    quantity_active = int(sample["training_masks"]["quantityLossMask"].item()) == 1
    if quantity_active:
        gold_quantity = int(sample["training_targets"]["quantityValue"].item())
        pred_quantity = float(outputs["quantityPred"][0].item())
        report["quantity"] = {
            "gold": gold_quantity,
            "predicted": pred_quantity,
            "predictedRounded": int(round(pred_quantity)),
        }

    return report


def main() -> None:
    args = parse_args()
    tensor_dir = resolve_tensor_dir(args)
    records = resolve_records(args, tensor_dir)
    dataset = RA2SLSectionDataset(records, cache_size=1)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    model = RA2SLBaselineModel(
        RA2SLBaselineConfig(
            entity_name_vocab_size=infer_entity_name_vocab_size(records),
        )
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    selected_indices: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        action_id = int(torch.argmax(sample["training_targets"]["actionTypeOneHot"]).item())
        action_name = ACTION_TYPE_ID_TO_NAME[action_id]
        motif = classify_action_motif(action_name)
        if motif is None:
            continue
        if len(selected_indices[motif]) >= int(args.examples_per_motif):
            continue
        selected_indices[motif].append(index)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tensorDir": str(tensor_dir),
        "manifest": str(args.manifest.resolve()) if args.manifest is not None else None,
        "device": str(device),
        "examplesPerMotif": int(args.examples_per_motif),
        "topK": int(args.top_k),
        "motifs": {},
    }
    for motif in ("build_order", "combat", "queue_heavy", "superweapon"):
        indices = selected_indices.get(motif, [])
        report["motifs"][motif] = {
            "sampleCount": len(indices),
            "samples": [
                inspect_sample(
                    dataset=dataset,
                    index=index,
                    model=model,
                    device=device,
                    top_k=args.top_k,
                )
                for index in indices
            ],
        }

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
