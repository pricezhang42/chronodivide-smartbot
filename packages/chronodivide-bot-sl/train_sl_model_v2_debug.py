from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model_lib.batch import collate_model_samples
from model_lib.dataset import (
    ModelShardRecord,
    RA2SLSectionDataset,
    RA2SLSequenceWindowDataset,
    discover_model_shards,
)
from model_lib.losses_v2 import compute_ra2_sl_v2_free_running_metrics, compute_ra2_sl_v2_loss
from model_lib.model_v2 import RA2SLV2DebugConfig, RA2SLV2DebugModel
from post_train_arena_eval import DEFAULT_DRIVER_DIR, run_post_train_arena_eval
from transform_lib.label_layout_v2 import LABEL_LAYOUT_V2_ACTION_FAMILIES
from train_sl_model import (
    build_loader,
    build_shard_filter,
    filter_records_by_manifest,
    limit_dataset,
    load_training_manifest,
    move_batch_to_device,
    resolve_device,
    save_json,
    set_random_seed,
    split_shards,
    utc_now_iso,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TENSOR_DIR = PACKAGE_ROOT / "generated_tensors" / "label_layout_v2_canonical_smoke_20260319"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "model_runs"


@dataclass
class TrainConfigV2Debug:
    tensor_dir: Path
    output_dir: Path
    manifest_path: Path | None
    map_names: list[str] | None
    player_countries: list[str] | None
    player_names: list[str] | None
    game_ids: list[str] | None
    side_ids: list[int] | None
    max_shards: int | None
    max_train_samples: int | None
    max_val_samples: int | None
    teacher_forcing_mode: str
    action_family_weighting: str
    action_family_weight_min: float
    action_family_weight_max: float
    place_building_weight_multiplier: float
    family_balanced_sampling: bool
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    val_ratio: float
    seed: int
    cache_size: int
    window_size: int
    window_stride: int
    use_lstm_core: bool
    lstm_num_layers: int
    checkpoint_dir: Path
    device: str | None
    post_train_arena_eval: bool
    post_train_arena_eval_driver_dir: Path
    post_train_arena_eval_match_count: int
    post_train_arena_eval_map_name: str
    post_train_arena_eval_max_ticks: int
    post_train_arena_eval_sample_interval_ticks: int
    post_train_arena_eval_candidate_mode: str
    post_train_arena_eval_candidate_country: str
    post_train_arena_eval_opponent_mode: str
    post_train_arena_eval_opponent_country: str
    post_train_arena_eval_mix_dir: Path | None


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_args() -> TrainConfigV2Debug:
    parser = argparse.ArgumentParser(description="Smoke-train the parallel V2 hierarchical label path.")
    parser.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "v2_debug")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--teacher-forcing-mode",
        type=str,
        choices=("none", "action_family", "full"),
        default="full",
    )
    parser.add_argument(
        "--action-family-weighting",
        type=str,
        choices=("none", "sqrt_inverse_frequency", "inverse_frequency"),
        default="sqrt_inverse_frequency",
    )
    parser.add_argument("--action-family-weight-min", type=float, default=0.25)
    parser.add_argument("--action-family-weight-max", type=float, default=4.0)
    parser.add_argument("--place-building-weight-multiplier", type=float, default=1.75)
    parser.add_argument("--family-balanced-sampling", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--use-lstm-core", action="store_true")
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--post-train-arena-eval", action="store_true")
    parser.add_argument("--post-train-arena-eval-driver-dir", type=Path, default=DEFAULT_DRIVER_DIR)
    parser.add_argument("--post-train-arena-eval-match-count", type=int, default=3)
    parser.add_argument("--post-train-arena-eval-map-name", type=str, default="2_pinch_point_le.map")
    parser.add_argument("--post-train-arena-eval-max-ticks", type=int, default=12000)
    parser.add_argument("--post-train-arena-eval-sample-interval-ticks", type=int, default=15)
    parser.add_argument(
        "--post-train-arena-eval-candidate-mode",
        type=str,
        choices=("baseline", "advisor"),
        default="advisor",
    )
    parser.add_argument("--post-train-arena-eval-candidate-country", type=str, default="IRAQ")
    parser.add_argument(
        "--post-train-arena-eval-opponent-mode",
        type=str,
        choices=("baseline", "advisor"),
        default="baseline",
    )
    parser.add_argument("--post-train-arena-eval-opponent-country", type=str, default="IRAQ")
    parser.add_argument("--post-train-arena-eval-mix-dir", type=Path, default=None)

    args = parser.parse_args()
    if args.window_size <= 0:
        raise ValueError(f"--window-size must be positive, got {args.window_size}.")
    if args.window_stride <= 0:
        raise ValueError(f"--window-stride must be positive, got {args.window_stride}.")

    output_dir = args.output_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else output_dir / "checkpoints"
    side_ids = [int(args.side_id)] if args.side_id is not None else None
    return TrainConfigV2Debug(
        tensor_dir=args.tensor_dir.resolve(),
        output_dir=output_dir,
        manifest_path=args.manifest.resolve() if args.manifest is not None else None,
        map_names=parse_csv(args.map_name),
        player_countries=parse_csv(args.player_country),
        player_names=parse_csv(args.player_name),
        game_ids=parse_csv(args.game_id),
        side_ids=side_ids,
        max_shards=args.max_shards,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        teacher_forcing_mode=args.teacher_forcing_mode,
        action_family_weighting=args.action_family_weighting,
        action_family_weight_min=args.action_family_weight_min,
        action_family_weight_max=args.action_family_weight_max,
        place_building_weight_multiplier=args.place_building_weight_multiplier,
        family_balanced_sampling=bool(args.family_balanced_sampling),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_size=args.cache_size,
        window_size=args.window_size,
        window_stride=args.window_stride,
        use_lstm_core=bool(args.use_lstm_core),
        lstm_num_layers=args.lstm_num_layers,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        post_train_arena_eval=bool(args.post_train_arena_eval),
        post_train_arena_eval_driver_dir=args.post_train_arena_eval_driver_dir.resolve(),
        post_train_arena_eval_match_count=args.post_train_arena_eval_match_count,
        post_train_arena_eval_map_name=args.post_train_arena_eval_map_name,
        post_train_arena_eval_max_ticks=args.post_train_arena_eval_max_ticks,
        post_train_arena_eval_sample_interval_ticks=args.post_train_arena_eval_sample_interval_ticks,
        post_train_arena_eval_candidate_mode=args.post_train_arena_eval_candidate_mode,
        post_train_arena_eval_candidate_country=args.post_train_arena_eval_candidate_country,
        post_train_arena_eval_opponent_mode=args.post_train_arena_eval_opponent_mode,
        post_train_arena_eval_opponent_country=args.post_train_arena_eval_opponent_country,
        post_train_arena_eval_mix_dir=(
            args.post_train_arena_eval_mix_dir.resolve()
            if args.post_train_arena_eval_mix_dir is not None
            else None
        ),
    )


def build_v2_dataset(records: list[ModelShardRecord], config: TrainConfigV2Debug) -> Dataset[Any]:
    if config.window_size > 1:
        return RA2SLSequenceWindowDataset(
            records,
            window_size=config.window_size,
            window_stride=config.window_stride,
            artifact_variant="v2",
            cache_size=config.cache_size,
        )
    return RA2SLSectionDataset(records, artifact_variant="v2", cache_size=config.cache_size)


def infer_entity_name_vocab_size(records: list[ModelShardRecord]) -> int:
    max_size = 0
    for record in records:
        vocab_size = len(record.metadata["schema"]["sharedNameVocabulary"]["idToName"])
        max_size = max(max_size, int(vocab_size))
    if max_size <= 0:
        raise ValueError("Could not infer a valid entity-name vocabulary size from the selected V2 shards.")
    return max_size


def infer_v2_model_config(records: list[ModelShardRecord], config: TrainConfigV2Debug) -> RA2SLV2DebugConfig:
    entity_name_vocab_size = infer_entity_name_vocab_size(records)
    max_action_family_count = 0
    max_order_type_count = 0
    max_target_mode_count = 0
    max_queue_update_type_count = 0
    max_buildable_object_vocab_size = 0
    max_super_weapon_type_count = 0
    max_commanded_units = 0
    max_entities = 0
    max_spatial_size = 0

    for record in records:
        schema_v2 = record.metadata.get("trainingTargetsV2")
        if not isinstance(schema_v2, dict):
            raise ValueError(f"{record.stem} is missing trainingTargetsV2 metadata.")
        max_action_family_count = max(max_action_family_count, int(schema_v2["actionFamilyCount"]))
        max_order_type_count = max(max_order_type_count, int(schema_v2["orderTypeCount"]))
        max_target_mode_count = max(max_target_mode_count, int(schema_v2["targetModeCount"]))
        max_queue_update_type_count = max(max_queue_update_type_count, int(schema_v2["queueUpdateTypeCount"]))
        max_buildable_object_vocab_size = max(
            max_buildable_object_vocab_size,
            int(schema_v2["buildableObjectVocabularySize"]),
        )
        max_super_weapon_type_count = max(
            max_super_weapon_type_count,
            int(schema_v2["superWeaponTypeCount"]),
        )
        max_commanded_units = max(max_commanded_units, int(schema_v2["maxCommandedUnits"]))
        max_entities = max(max_entities, int(schema_v2["maxEntities"]))
        max_spatial_size = max(max_spatial_size, int(schema_v2["spatialSize"]))

    return RA2SLV2DebugConfig(
        entity_name_vocab_size=entity_name_vocab_size,
        action_family_count=max_action_family_count,
        order_type_count=max_order_type_count,
        target_mode_count=max_target_mode_count,
        queue_update_type_count=max_queue_update_type_count,
        buildable_object_vocab_size=max_buildable_object_vocab_size,
        super_weapon_type_count=max_super_weapon_type_count,
        max_commanded_units=max_commanded_units,
        max_entities=max_entities,
        spatial_size=max_spatial_size,
        use_lstm_core=config.use_lstm_core,
        lstm_num_layers=config.lstm_num_layers,
    )


def build_action_family_weighting(
    dataset: Dataset[Any],
    config: TrainConfigV2Debug,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    class_count = len(LABEL_LAYOUT_V2_ACTION_FAMILIES)
    if config.action_family_weighting == "none":
        return None, {
            "mode": "none",
            "weightMin": config.action_family_weight_min,
            "weightMax": config.action_family_weight_max,
            "classCount": class_count,
        }

    counts = torch.zeros(class_count, dtype=torch.float64)
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        action_family_mask = sample["training_masks"]["actionFamilyLossMask"].reshape(-1) > 0
        if not bool(action_family_mask.any()):
            continue
        action_family_ids = torch.argmax(
            sample["training_targets"]["actionFamilyOneHot"].reshape(-1, class_count),
            dim=-1,
        )
        valid_action_family_ids = action_family_ids[action_family_mask]
        if valid_action_family_ids.numel() == 0:
            continue
        counts += torch.bincount(valid_action_family_ids, minlength=class_count).to(torch.float64)

    seen_mask = counts > 0
    weights = torch.ones(class_count, dtype=torch.float32)
    if not bool(seen_mask.any()):
        return weights, {
            "mode": config.action_family_weighting,
            "weightMin": config.action_family_weight_min,
            "weightMax": config.action_family_weight_max,
            "classCount": class_count,
            "seenClassCount": 0,
            "sampleCount": 0,
            "topWeighted": [],
            "topFrequent": [],
        }

    seen_counts = counts[seen_mask]
    mean_count = float(seen_counts.mean().item())
    if config.action_family_weighting == "inverse_frequency":
        raw_seen_weights = mean_count / seen_counts
    elif config.action_family_weighting == "sqrt_inverse_frequency":
        raw_seen_weights = torch.sqrt(torch.tensor(mean_count, dtype=torch.float64) / seen_counts)
    else:
        raise ValueError(f"Unsupported action-family weighting mode: {config.action_family_weighting}")

    clamped_seen_weights = raw_seen_weights.clamp(
        min=config.action_family_weight_min,
        max=config.action_family_weight_max,
    )
    normalized_seen_weights = clamped_seen_weights / clamped_seen_weights.mean().clamp(min=1e-9)
    weights[seen_mask] = normalized_seen_weights.to(torch.float32)

    place_building_family_id = LABEL_LAYOUT_V2_ACTION_FAMILIES.index("PlaceBuilding")
    if seen_mask[place_building_family_id] and config.place_building_weight_multiplier != 1.0:
        boosted_weight = (
            weights[place_building_family_id] * float(config.place_building_weight_multiplier)
        ).clamp(min=config.action_family_weight_min, max=config.action_family_weight_max)
        weights[place_building_family_id] = boosted_weight

    seen_indices = torch.nonzero(seen_mask, as_tuple=False).reshape(-1)
    ranked_by_weight = sorted(
        ((int(index.item()), float(weights[int(index.item())].item())) for index in seen_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    top_weighted = [
        {
            "actionFamilyId": action_family_id,
            "actionFamilyName": LABEL_LAYOUT_V2_ACTION_FAMILIES[action_family_id],
            "count": int(counts[action_family_id].item()),
            "weight": weight,
        }
        for action_family_id, weight in ranked_by_weight[:20]
    ]

    ranked_by_count = sorted(
        ((int(index.item()), int(counts[int(index.item())].item())) for index in seen_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    top_frequent = [
        {
            "actionFamilyId": action_family_id,
            "actionFamilyName": LABEL_LAYOUT_V2_ACTION_FAMILIES[action_family_id],
            "count": count,
            "weight": float(weights[action_family_id].item()),
        }
        for action_family_id, count in ranked_by_count[:20]
    ]

    payload = {
        "mode": config.action_family_weighting,
        "weightMin": config.action_family_weight_min,
        "weightMax": config.action_family_weight_max,
        "placeBuildingWeightMultiplier": config.place_building_weight_multiplier,
        "classCount": class_count,
        "seenClassCount": int(seen_mask.sum().item()),
        "sampleCount": int(counts.sum().item()),
        "topWeighted": top_weighted,
        "topFrequent": top_frequent,
    }
    return weights, payload


def build_action_family_sampling_weights(
    dataset: Dataset[Any],
    class_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if class_weights is None:
        return None

    sample_weights = torch.ones(len(dataset), dtype=torch.double)
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        action_family_mask = sample["training_masks"]["actionFamilyLossMask"].reshape(-1) > 0
        if not bool(action_family_mask.any()):
            continue
        action_family_ids = torch.argmax(
            sample["training_targets"]["actionFamilyOneHot"].reshape(-1, class_weights.shape[0]),
            dim=-1,
        )
        valid_action_family_ids = action_family_ids[action_family_mask]
        if valid_action_family_ids.numel() == 0:
            continue
        sample_weights[sample_index] = float(class_weights[valid_action_family_ids].mean().item())
    return sample_weights


def build_v2_train_loader(
    dataset: Dataset[Any],
    config: TrainConfigV2Debug,
    action_family_class_weights: torch.Tensor | None,
) -> DataLoader:
    batch_size = min(config.batch_size, len(dataset))
    if config.family_balanced_sampling:
        sampling_weights = build_action_family_sampling_weights(dataset, action_family_class_weights)
        if sampling_weights is None:
            raise ValueError("Family-balanced sampling requires non-null action-family class weights.")
        sampler = WeightedRandomSampler(
            weights=sampling_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=collate_model_samples,
        )
    return build_loader(dataset, config, shuffle=True)


def init_metrics_accumulator() -> dict[str, float]:
    return {"total_loss": 0.0, "batch_count": 0.0, "sample_count": 0.0}


def accumulate_metric_group(accumulator: dict[str, float], prefix: str, metric_group: dict[str, torch.Tensor]) -> None:
    for name, value in metric_group.items():
        accumulator[f"{prefix}.{name}"] = accumulator.get(f"{prefix}.{name}", 0.0) + float(value.detach().item())


def finalize_metrics(accumulator: dict[str, float]) -> dict[str, float]:
    batch_count = max(1.0, accumulator.get("batch_count", 1.0))
    finalized: dict[str, float] = {}
    for name, value in accumulator.items():
        if name in {"batch_count", "sample_count"}:
            continue
        finalized[name] = value / batch_count
    return finalized


def _count_supervised_steps(batch: dict[str, Any]) -> int:
    action_family_targets = batch["training_targets"]["actionFamilyOneHot"]
    return int(action_family_targets.reshape(-1, action_family_targets.shape[-1]).shape[0])


def run_epoch_v2(
    *,
    model: RA2SLV2DebugModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
    teacher_forcing_mode: str,
    action_family_class_weights: torch.Tensor | None,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    accumulator = init_metrics_accumulator()
    wall_clock_start = time.perf_counter()

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(
                batch["model_inputs"],
                teacher_forcing_targets=batch["training_targets"],
                teacher_forcing_masks=batch["training_masks"],
                teacher_forcing_mode=teacher_forcing_mode,
            )
            loss_output = compute_ra2_sl_v2_loss(
                outputs,
                batch,
                action_family_class_weights=action_family_class_weights,
            )
            if training:
                loss_output.total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        accumulator["total_loss"] += float(loss_output.total_loss.detach().item())
        accumulator["batch_count"] += 1.0
        accumulator["sample_count"] += float(_count_supervised_steps(batch))
        accumulate_metric_group(accumulator, "loss", loss_output.loss_by_head)
        accumulate_metric_group(accumulator, "metric", loss_output.metrics)

    elapsed_seconds = max(1e-6, time.perf_counter() - wall_clock_start)
    metrics = finalize_metrics(accumulator)
    sample_count = accumulator.get("sample_count", 0.0)
    metrics["epochSeconds"] = elapsed_seconds
    metrics["samplesPerSecond"] = sample_count / elapsed_seconds
    metrics["sampleCount"] = sample_count
    return metrics


def run_free_running_eval_epoch_v2(
    *,
    model: RA2SLV2DebugModel,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = init_metrics_accumulator()
    wall_clock_start = time.perf_counter()
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
            accumulator["batch_count"] += 1.0
            accumulator["sample_count"] += float(_count_supervised_steps(batch))
            accumulate_metric_group(accumulator, "metric", metrics)

    elapsed_seconds = max(1e-6, time.perf_counter() - wall_clock_start)
    finalized = finalize_metrics(accumulator)
    sample_count = accumulator.get("sample_count", 0.0)
    finalized["epochSeconds"] = elapsed_seconds
    finalized["samplesPerSecond"] = sample_count / elapsed_seconds
    finalized["sampleCount"] = sample_count
    return finalized


def save_checkpoint_v2(
    *,
    path: Path,
    model: RA2SLV2DebugModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    val_free_metrics: dict[str, float] | None,
    config: TrainConfigV2Debug,
    best_val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "val_free_metrics": val_free_metrics,
            "config": asdict(config),
            "best_val_loss": best_val_loss,
            "savedAt": utc_now_iso(),
        },
        path,
    )


def main() -> None:
    config = parse_args()
    set_random_seed(config.seed)
    manifest: dict[str, Any] | None = None
    if config.manifest_path is not None:
        manifest = load_training_manifest(config.manifest_path)
        manifest_tensor_dir = manifest.get("tensorDir")
        if manifest_tensor_dir:
            config.tensor_dir = Path(manifest_tensor_dir).resolve()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shard_filter = build_shard_filter(config)
    records = discover_model_shards(config.tensor_dir, shard_filter=shard_filter)
    if manifest is not None:
        records = filter_records_by_manifest(records, manifest)
    records = [record for record in records if record.sections_path_v2 is not None and record.training_path_v2 is not None]
    if config.max_shards is not None:
        records = records[: config.max_shards]
    if not records:
        raise ValueError("No V2 tensor shards matched the requested filters.")

    train_records, val_records = split_shards(records, config.val_ratio, config.seed)
    if not train_records:
        raise ValueError("Training split is empty after shard splitting.")

    train_dataset = build_v2_dataset(train_records, config)
    val_dataset = build_v2_dataset(val_records, config) if val_records else None
    train_dataset = limit_dataset(train_dataset, config.max_train_samples)
    if val_dataset is not None:
        val_dataset = limit_dataset(val_dataset, config.max_val_samples)
    action_family_class_weights, action_family_weight_payload = build_action_family_weighting(
        train_dataset,
        config,
    )

    train_loader = build_v2_train_loader(
        train_dataset,
        config,
        action_family_class_weights,
    )
    val_loader = build_loader(val_dataset, config, shuffle=False) if val_dataset is not None else None

    model = RA2SLV2DebugModel(infer_v2_model_config(records, config))
    device = resolve_device(config)
    if action_family_class_weights is not None:
        action_family_class_weights = action_family_class_weights.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

    split_payload = {
        "createdAt": utc_now_iso(),
        "config": asdict(config),
        "manifestPath": str(config.manifest_path) if config.manifest_path is not None else None,
        "device": str(device),
        "artifactVariant": "v2",
        "trainShardCount": len(train_records),
        "valShardCount": len(val_records),
        "trainDatasetItemCount": len(train_dataset),
        "valDatasetItemCount": len(val_dataset) if val_dataset is not None else 0,
        "trainShards": [record.stem for record in train_records],
        "valShards": [record.stem for record in val_records],
        "actionFamilyWeighting": action_family_weight_payload,
    }
    save_json(config.output_dir / "data_split.json", split_payload)
    if action_family_weight_payload is not None:
        save_json(config.output_dir / "action_family_weighting.json", action_family_weight_payload)

    history_path = config.output_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    best_val_loss: float | None = None
    last_train_metrics: dict[str, float] | None = None
    last_val_metrics: dict[str, float] | None = None
    last_val_free_metrics: dict[str, float] | None = None

    for epoch in range(config.epochs):
        train_metrics = run_epoch_v2(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
            teacher_forcing_mode=config.teacher_forcing_mode,
            action_family_class_weights=action_family_class_weights,
        )
        val_metrics = None
        val_free_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch_v2(
                model=model,
                data_loader=val_loader,
                optimizer=None,
                device=device,
                grad_clip_norm=config.grad_clip_norm,
                teacher_forcing_mode=config.teacher_forcing_mode,
                action_family_class_weights=action_family_class_weights,
            )
            val_free_metrics = run_free_running_eval_epoch_v2(
                model=model,
                data_loader=val_loader,
                device=device,
            )

        scheduler.step()
        last_train_metrics = train_metrics
        last_val_metrics = val_metrics
        last_val_free_metrics = val_free_metrics

        summary = {
            "epoch": epoch,
            "timestamp": utc_now_iso(),
            "train": train_metrics,
            "val": val_metrics,
            "valFree": val_free_metrics,
            "learningRate": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")

        print(
            f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.6f}"
            + f" train_sps={train_metrics['samplesPerSecond']:.2f}"
            + (
                f" val_loss={val_metrics['total_loss']:.6f} val_sps={val_metrics['samplesPerSecond']:.2f}"
                if val_metrics is not None
                else ""
            )
            + (
                f" val_free_family={val_free_metrics['metric.actionFamilyAccuracy']:.4f}"
                f" val_free_specific={val_free_metrics['metric.specificActionTypeAccuracy']:.4f}"
                f" val_free_full={val_free_metrics['metric.fullCommandExactMatch']:.4f}"
                if val_free_metrics is not None
                else ""
            )
        )

        candidate_loss = train_metrics["total_loss"] if val_metrics is None else val_metrics["total_loss"]
        is_best = best_val_loss is None or candidate_loss < best_val_loss
        if is_best:
            best_val_loss = float(candidate_loss)
            save_checkpoint_v2(
                path=config.checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_free_metrics=val_free_metrics,
                config=config,
                best_val_loss=best_val_loss,
            )

        save_checkpoint_v2(
            path=config.checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            val_free_metrics=val_free_metrics,
            config=config,
            best_val_loss=best_val_loss,
        )

    final_summary = {
        "createdAt": utc_now_iso(),
        "artifactVariant": "v2",
        "device": str(device),
        "trainShardCount": len(train_records),
        "valShardCount": len(val_records),
        "trainDatasetItemCount": len(train_dataset),
        "valDatasetItemCount": len(val_dataset) if val_dataset is not None else 0,
        "epochs": config.epochs,
        "checkpointDir": str(config.checkpoint_dir),
        "bestValLoss": best_val_loss,
        "actionFamilyWeighting": action_family_weight_payload,
        "finalTrainMetrics": last_train_metrics,
        "finalValMetrics": last_val_metrics,
        "finalValFreeMetrics": last_val_free_metrics,
    }
    training_summary_path = config.output_dir / "training_summary.json"
    save_json(training_summary_path, final_summary)

    if config.post_train_arena_eval:
        try:
            arena_eval_summary = run_post_train_arena_eval(
                enabled=True,
                driver_dir=config.post_train_arena_eval_driver_dir,
                output_dir=config.output_dir,
                checkpoint_dir=config.checkpoint_dir,
                preferred_checkpoint_names=["best.pt", "latest.pt"],
                match_count=config.post_train_arena_eval_match_count,
                map_name=config.post_train_arena_eval_map_name,
                max_ticks=config.post_train_arena_eval_max_ticks,
                sample_interval_ticks=config.post_train_arena_eval_sample_interval_ticks,
                candidate_mode=config.post_train_arena_eval_candidate_mode,
                candidate_country=config.post_train_arena_eval_candidate_country,
                opponent_mode=config.post_train_arena_eval_opponent_mode,
                opponent_country=config.post_train_arena_eval_opponent_country,
                mix_dir=config.post_train_arena_eval_mix_dir,
            )
            final_summary["postTrainArenaEval"] = arena_eval_summary
        except Exception as exc:
            final_summary["postTrainArenaEvalError"] = str(exc)
            print(f"Post-train arena evaluation failed: {exc}")
        save_json(training_summary_path, final_summary)


if __name__ == "__main__":
    main()
