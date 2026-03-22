from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from action_dict import ACTION_INFO_MASK, ACTION_TYPE_ID_TO_NAME
from model_lib.batch import collate_model_samples
from model_lib.dataset import (
    ModelShardFilter,
    ModelShardRecord,
    RA2SLSectionDataset,
    RA2SLSequenceWindowDataset,
    discover_model_shards,
)
from model_lib.losses import compute_ra2_sl_free_running_metrics, compute_ra2_sl_loss
from model_lib.model import RA2SLBaselineConfig, RA2SLBaselineModel
from post_train_arena_eval import DEFAULT_DRIVER_DIR, run_post_train_arena_eval


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TENSOR_DIR = PACKAGE_ROOT / "generated_tensors" / "feature_layout_v1_validation_5replays_fix_20260318"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "model_runs"


@dataclass
class TrainConfig:
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
    action_type_weighting: str
    action_type_weight_min: float
    action_type_weight_max: float
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
    resume: Path | None
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
    # Pseudo-reward settings.
    pseudo_reward_enabled: bool
    pseudo_reward_production_boost: float
    pseudo_reward_non_noop_boost: float
    composition_aux_scale: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the baseline RA2 supervised-learning model.")
    parser.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        choices=("none", "action_type", "action_type_queue", "full"),
        default="full",
    )
    parser.add_argument(
        "--action-type-weighting",
        type=str,
        choices=("none", "sqrt_inverse_frequency", "inverse_frequency"),
        default="sqrt_inverse_frequency",
    )
    parser.add_argument("--action-type-weight-min", type=float, default=0.25)
    parser.add_argument("--action-type-weight-max", type=float, default=4.0)
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
    parser.add_argument("--resume", type=Path, default=None)
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
    # Pseudo-reward arguments.
    parser.add_argument(
        "--pseudo-reward-enabled",
        action="store_true",
        help="Enable pseudo-reward sample importance weighting (upweight production/non-Noop samples).",
    )
    parser.add_argument("--pseudo-reward-production-boost", type=float, default=3.0,
                        help="Weight multiplier for Queue/PlaceBuilding samples.")
    parser.add_argument("--pseudo-reward-non-noop-boost", type=float, default=1.5,
                        help="Weight multiplier for non-Noop samples.")
    parser.add_argument("--composition-aux-scale", type=float, default=0.0,
                        help="Scale for composition prediction auxiliary loss (0 = disabled).")

    args = parser.parse_args()
    if args.window_size <= 0:
        raise ValueError(f"--window-size must be positive, got {args.window_size}.")
    if args.window_stride <= 0:
        raise ValueError(f"--window-stride must be positive, got {args.window_stride}.")
    output_dir = args.output_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else output_dir / "checkpoints"
    side_ids = [int(args.side_id)] if args.side_id is not None else None
    return TrainConfig(
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
        action_type_weighting=args.action_type_weighting,
        action_type_weight_min=args.action_type_weight_min,
        action_type_weight_max=args.action_type_weight_max,
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
        resume=args.resume.resolve() if args.resume is not None else None,
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
        pseudo_reward_enabled=bool(args.pseudo_reward_enabled),
        pseudo_reward_production_boost=args.pseudo_reward_production_boost,
        pseudo_reward_non_noop_boost=args.pseudo_reward_non_noop_boost,
        composition_aux_scale=args.composition_aux_scale,
    )


def build_shard_filter(config: TrainConfig) -> ModelShardFilter:
    return ModelShardFilter.create(
        map_names=config.map_names,
        player_country_names=config.player_countries,
        player_names=config.player_names,
        replay_game_ids=config.game_ids,
        player_side_ids=config.side_ids,
    )


def load_training_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shard_stems = manifest.get("shardStems")
    if not isinstance(shard_stems, list) or not shard_stems:
        raise ValueError(f"Training manifest is missing a non-empty shardStems list: {path}")
    return manifest


def filter_records_by_manifest(records: list[ModelShardRecord], manifest: dict[str, Any]) -> list[ModelShardRecord]:
    shard_stems = {str(stem) for stem in manifest["shardStems"]}
    filtered = [record for record in records if record.stem in shard_stems]
    if not filtered:
        raise ValueError("Training manifest matched zero shards in the selected tensor directory.")
    missing = sorted(shard_stems.difference(record.stem for record in filtered))
    if missing:
        raise ValueError(f"Training manifest referenced missing shard stems: {missing[:5]}")
    return filtered


def split_shards(records: list[ModelShardRecord], val_ratio: float, seed: int) -> tuple[list[ModelShardRecord], list[ModelShardRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or val_ratio <= 0.0:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def infer_entity_name_vocab_size(records: list[ModelShardRecord]) -> int:
    max_size = 0
    for record in records:
        vocab_size = len(record.metadata["schema"]["sharedNameVocabulary"]["idToName"])
        max_size = max(max_size, int(vocab_size))
    if max_size <= 0:
        raise ValueError("Could not infer a valid entity-name vocabulary size from the selected shards.")
    return max_size


def move_batch_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    return value


def limit_dataset(dataset: Dataset[Any], max_samples: int | None) -> Dataset[Any]:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive when provided, got {max_samples}.")
    return Subset(dataset, list(range(max_samples)))


def build_loader(dataset: Dataset[Any], config: TrainConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_model_samples,
    )


def build_section_dataset(records: list[ModelShardRecord], config: TrainConfig) -> Dataset[Any]:
    if config.window_size > 1:
        return RA2SLSequenceWindowDataset(
            records,
            window_size=config.window_size,
            window_stride=config.window_stride,
            cache_size=config.cache_size,
        )
    return RA2SLSectionDataset(records, cache_size=config.cache_size)


def build_action_type_weighting(
    dataset: Dataset[Any],
    config: TrainConfig,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if config.action_type_weighting == "none":
        return None, {
            "mode": "none",
            "weightMin": config.action_type_weight_min,
            "weightMax": config.action_type_weight_max,
            "classCount": len(ACTION_TYPE_ID_TO_NAME),
        }

    class_count = len(ACTION_TYPE_ID_TO_NAME)
    counts = torch.zeros(class_count, dtype=torch.float64)
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        action_type_mask = sample["training_masks"]["actionTypeLossMask"].reshape(-1) > 0
        if not bool(action_type_mask.any()):
            continue
        action_type_ids = torch.argmax(
            sample["training_targets"]["actionTypeOneHot"].reshape(-1, class_count),
            dim=-1,
        )
        valid_action_type_ids = action_type_ids[action_type_mask]
        if valid_action_type_ids.numel() == 0:
            continue
        counts += torch.bincount(valid_action_type_ids, minlength=class_count).to(torch.float64)

    seen_mask = counts > 0
    weights = torch.ones(class_count, dtype=torch.float32)
    if not bool(seen_mask.any()):
        return weights, {
            "mode": config.action_type_weighting,
            "weightMin": config.action_type_weight_min,
            "weightMax": config.action_type_weight_max,
            "classCount": class_count,
            "seenClassCount": 0,
            "sampleCount": 0,
            "topWeighted": [],
            "topFrequent": [],
            "familyCounts": {},
        }

    seen_counts = counts[seen_mask]
    mean_count = float(seen_counts.mean().item())
    if config.action_type_weighting == "inverse_frequency":
        raw_seen_weights = mean_count / seen_counts
    elif config.action_type_weighting == "sqrt_inverse_frequency":
        raw_seen_weights = torch.sqrt(torch.tensor(mean_count, dtype=torch.float64) / seen_counts)
    else:
        raise ValueError(f"Unsupported action-type weighting mode: {config.action_type_weighting}")

    clamped_seen_weights = raw_seen_weights.clamp(min=config.action_type_weight_min, max=config.action_type_weight_max)
    normalized_seen_weights = clamped_seen_weights / clamped_seen_weights.mean().clamp(min=1e-9)
    weights[seen_mask] = normalized_seen_weights.to(torch.float32)

    top_weighted: list[dict[str, Any]] = []
    seen_indices = torch.nonzero(seen_mask, as_tuple=False).reshape(-1)
    ranked_by_weight = sorted(
        ((int(index.item()), float(weights[int(index.item())].item())) for index in seen_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    for action_type_id, weight in ranked_by_weight[:20]:
        action_info = ACTION_INFO_MASK[action_type_id]
        top_weighted.append(
            {
                "actionTypeId": action_type_id,
                "actionTypeName": ACTION_TYPE_ID_TO_NAME[action_type_id],
                "family": action_info["family"],
                "count": int(counts[action_type_id].item()),
                "weight": weight,
            }
        )

    ranked_by_count = sorted(
        ((int(index.item()), int(counts[int(index.item())].item())) for index in seen_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    top_frequent: list[dict[str, Any]] = []
    for action_type_id, count in ranked_by_count[:20]:
        action_info = ACTION_INFO_MASK[action_type_id]
        top_frequent.append(
            {
                "actionTypeId": action_type_id,
                "actionTypeName": ACTION_TYPE_ID_TO_NAME[action_type_id],
                "family": action_info["family"],
                "count": count,
                "weight": float(weights[action_type_id].item()),
            }
        )

    family_counts: dict[str, int] = {}
    for action_type_id, count in ranked_by_count:
        family = str(ACTION_INFO_MASK[action_type_id]["family"])
        family_counts[family] = family_counts.get(family, 0) + count

    payload = {
        "mode": config.action_type_weighting,
        "weightMin": config.action_type_weight_min,
        "weightMax": config.action_type_weight_max,
        "classCount": class_count,
        "seenClassCount": int(seen_mask.sum().item()),
        "sampleCount": int(counts.sum().item()),
        "topWeighted": top_weighted,
        "topFrequent": top_frequent,
        "familyCounts": dict(sorted(family_counts.items())),
    }
    return weights, payload


def init_metrics_accumulator() -> dict[str, float]:
    return {
        "total_loss": 0.0,
        "batch_count": 0.0,
        "sample_count": 0.0,
    }


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
    action_type_targets = batch["training_targets"]["actionTypeOneHot"]
    return int(action_type_targets.reshape(-1, action_type_targets.shape[-1]).shape[0])


def run_epoch(
    *,
    model: RA2SLBaselineModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
    action_type_class_weights: torch.Tensor | None,
    teacher_forcing_mode: str,
    pseudo_reward_config: "PseudoRewardConfig | None" = None,
    composition_aux_scale: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    accumulator = init_metrics_accumulator()
    start_time = time.perf_counter()
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
            loss_output = compute_ra2_sl_loss(
                outputs,
                batch,
                action_type_class_weights=action_type_class_weights,
                pseudo_reward_config=pseudo_reward_config,
                composition_aux_scale=composition_aux_scale,
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

    metrics = finalize_metrics(accumulator)
    elapsed_seconds = max(1e-6, time.perf_counter() - start_time)
    sample_count = accumulator.get("sample_count", 0.0)
    metrics["epochSeconds"] = elapsed_seconds
    metrics["samplesPerSecond"] = sample_count / elapsed_seconds
    metrics["sampleCount"] = sample_count
    return metrics


def run_free_running_eval_epoch(
    *,
    model: RA2SLBaselineModel,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = init_metrics_accumulator()
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["model_inputs"],
                teacher_forcing_targets=None,
                teacher_forcing_masks=None,
                teacher_forcing_mode="none",
            )
            metrics = compute_ra2_sl_free_running_metrics(outputs, batch)
            accumulator["batch_count"] += 1.0
            accumulator["sample_count"] += float(_count_supervised_steps(batch))
            accumulate_metric_group(accumulator, "metric", metrics)

    finalized = finalize_metrics(accumulator)
    elapsed_seconds = max(1e-6, time.perf_counter() - start_time)
    sample_count = accumulator.get("sample_count", 0.0)
    finalized["epochSeconds"] = elapsed_seconds
    finalized["samplesPerSecond"] = sample_count / elapsed_seconds
    finalized["sampleCount"] = sample_count
    return finalized


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def save_checkpoint(
    *,
    path: Path,
    model: RA2SLBaselineModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    val_free_metrics: dict[str, float] | None,
    config: TrainConfig,
    best_val_loss: float | None,
    best_checkpoint_metrics: dict[str, dict[str, float | int | None]],
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
            "best_checkpoint_metrics": best_checkpoint_metrics,
            "savedAt": utc_now_iso(),
        },
        path,
    )


def resolve_device(config: TrainConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    if config.max_shards is not None:
        records = records[: config.max_shards]
    if not records:
        raise ValueError("No tensor shards matched the requested training filters.")

    train_records, val_records = split_shards(records, config.val_ratio, config.seed)
    if not train_records:
        raise ValueError("Training split is empty after shard splitting.")

    train_dataset = build_section_dataset(train_records, config)
    val_dataset = build_section_dataset(val_records, config) if val_records else None
    train_dataset = limit_dataset(train_dataset, config.max_train_samples)
    if val_dataset is not None:
        val_dataset = limit_dataset(val_dataset, config.max_val_samples)
    action_type_class_weights, action_type_weight_payload = build_action_type_weighting(train_dataset, config)
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False) if val_dataset is not None else None

    entity_name_vocab_size = infer_entity_name_vocab_size(records)
    device = resolve_device(config)
    if action_type_class_weights is not None:
        action_type_class_weights = action_type_class_weights.to(device)
    # Determine composition vocabulary size for the auxiliary head.
    composition_aux_size = 0
    if config.composition_aux_scale > 0:
        from transform_lib.feature_layout import OWNED_COMPOSITION_VOCABULARY
        composition_aux_size = len(OWNED_COMPOSITION_VOCABULARY)

    model = RA2SLBaselineModel(
        RA2SLBaselineConfig(
            entity_name_vocab_size=entity_name_vocab_size,
            use_lstm_core=config.use_lstm_core,
            lstm_num_layers=config.lstm_num_layers,
            composition_aux_size=composition_aux_size,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

    start_epoch = 0
    best_val_loss: float | None = None
    best_checkpoint_metrics: dict[str, dict[str, float | int | None]] = {
        "valLoss": {"value": None, "epoch": None},
        "valFreeActionAccuracy": {"value": None, "epoch": None},
        "valFreeFullActionExactMatch": {"value": None, "epoch": None},
    }
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = checkpoint.get("best_val_loss")
        saved_best_checkpoint_metrics = checkpoint.get("best_checkpoint_metrics")
        if isinstance(saved_best_checkpoint_metrics, dict):
            for key, value in saved_best_checkpoint_metrics.items():
                if key in best_checkpoint_metrics and isinstance(value, dict):
                    best_checkpoint_metrics[key] = {
                        "value": value.get("value"),
                        "epoch": value.get("epoch"),
                    }
    if best_val_loss is not None:
        best_checkpoint_metrics["valLoss"] = {
            "value": float(best_val_loss),
            "epoch": best_checkpoint_metrics["valLoss"].get("epoch"),
        }

    split_payload = {
        "createdAt": utc_now_iso(),
        "config": asdict(config),
        "manifestPath": str(config.manifest_path) if config.manifest_path is not None else None,
        "device": str(device),
        "trainShardCount": len(train_records),
        "valShardCount": len(val_records),
        "trainSampleCount": len(train_dataset),
        "valSampleCount": len(val_dataset) if val_dataset is not None else 0,
        "trainDatasetItemCount": len(train_dataset),
        "valDatasetItemCount": len(val_dataset) if val_dataset is not None else 0,
        "windowSize": config.window_size,
        "windowStride": config.window_stride,
        "useLstmCore": config.use_lstm_core,
        "lstmNumLayers": config.lstm_num_layers,
        "teacherForcingMode": config.teacher_forcing_mode,
        "actionTypeWeighting": action_type_weight_payload,
        "trainShards": [record.stem for record in train_records],
        "valShards": [record.stem for record in val_records],
    }
    save_json(config.output_dir / "data_split.json", split_payload)
    if action_type_weight_payload is not None:
        save_json(config.output_dir / "action_type_weighting.json", action_type_weight_payload)

    history_path = config.output_dir / "history.jsonl"
    if start_epoch == 0 and history_path.exists():
        history_path.unlink()

    # Build pseudo-reward config if enabled.
    pseudo_reward_config = None
    if config.pseudo_reward_enabled:
        from model_lib.pseudo_reward import PseudoRewardConfig
        pseudo_reward_config = PseudoRewardConfig(
            enabled=True,
            production_action_boost=config.pseudo_reward_production_boost,
            non_noop_boost=config.pseudo_reward_non_noop_boost,
        )

    last_train_metrics: dict[str, float] | None = None
    last_val_metrics: dict[str, float] | None = None
    last_val_free_metrics: dict[str, float] | None = None
    for epoch in range(start_epoch, config.epochs):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
            action_type_class_weights=action_type_class_weights,
            teacher_forcing_mode=config.teacher_forcing_mode,
            pseudo_reward_config=pseudo_reward_config,
            composition_aux_scale=config.composition_aux_scale,
        )
        val_metrics = None
        val_free_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                data_loader=val_loader,
                optimizer=None,
                device=device,
                grad_clip_norm=config.grad_clip_norm,
                action_type_class_weights=action_type_class_weights,
                teacher_forcing_mode=config.teacher_forcing_mode,
                pseudo_reward_config=pseudo_reward_config,
                composition_aux_scale=config.composition_aux_scale,
            )
            val_free_metrics = run_free_running_eval_epoch(
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
                f" val_free_action={val_free_metrics['metric.actionTypeAccuracy']:.4f}"
                f" val_free_full={val_free_metrics['metric.fullActionExactMatch']:.4f}"
                if val_free_metrics is not None
                else ""
            )
        )

        candidate_loss = train_metrics["total_loss"] if val_metrics is None else val_metrics["total_loss"]
        is_best_val_loss = best_val_loss is None or candidate_loss < best_val_loss
        if is_best_val_loss:
            best_val_loss = candidate_loss
            best_checkpoint_metrics["valLoss"] = {
                "value": float(candidate_loss),
                "epoch": epoch,
            }

        best_val_free_action_value = best_checkpoint_metrics["valFreeActionAccuracy"].get("value")
        is_best_val_free_action = (
            val_free_metrics is not None
            and (
                best_val_free_action_value is None
                or float(val_free_metrics["metric.actionTypeAccuracy"]) > float(best_val_free_action_value)
            )
        )
        if is_best_val_free_action and val_free_metrics is not None:
            best_checkpoint_metrics["valFreeActionAccuracy"] = {
                "value": float(val_free_metrics["metric.actionTypeAccuracy"]),
                "epoch": epoch,
            }

        best_val_free_full_value = best_checkpoint_metrics["valFreeFullActionExactMatch"].get("value")
        is_best_val_free_full = (
            val_free_metrics is not None
            and (
                best_val_free_full_value is None
                or float(val_free_metrics["metric.fullActionExactMatch"]) > float(best_val_free_full_value)
            )
        )
        if is_best_val_free_full and val_free_metrics is not None:
            best_checkpoint_metrics["valFreeFullActionExactMatch"] = {
                "value": float(val_free_metrics["metric.fullActionExactMatch"]),
                "epoch": epoch,
            }

        save_checkpoint(
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
            best_checkpoint_metrics=best_checkpoint_metrics,
        )
        if is_best_val_loss:
            save_checkpoint(
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
                best_checkpoint_metrics=best_checkpoint_metrics,
            )
            save_checkpoint(
                path=config.checkpoint_dir / "best_val_loss.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_free_metrics=val_free_metrics,
                config=config,
                best_val_loss=best_val_loss,
                best_checkpoint_metrics=best_checkpoint_metrics,
            )
        if is_best_val_free_action:
            save_checkpoint(
                path=config.checkpoint_dir / "best_val_free_action.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_free_metrics=val_free_metrics,
                config=config,
                best_val_loss=best_val_loss,
                best_checkpoint_metrics=best_checkpoint_metrics,
            )
        if is_best_val_free_full:
            save_checkpoint(
                path=config.checkpoint_dir / "best_val_free_full.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_free_metrics=val_free_metrics,
                config=config,
                best_val_loss=best_val_loss,
                best_checkpoint_metrics=best_checkpoint_metrics,
            )

    final_summary = {
        "createdAt": utc_now_iso(),
        "device": str(device),
        "trainShardCount": len(train_records),
        "valShardCount": len(val_records),
        "trainSampleCount": len(train_dataset),
        "valSampleCount": len(val_dataset) if val_dataset is not None else 0,
        "trainDatasetItemCount": len(train_dataset),
        "valDatasetItemCount": len(val_dataset) if val_dataset is not None else 0,
        "entityNameVocabSize": entity_name_vocab_size,
        "epochs": config.epochs,
        "checkpointDir": str(config.checkpoint_dir),
        "bestValLoss": best_val_loss,
        "bestCheckpointMetrics": best_checkpoint_metrics,
        "windowSize": config.window_size,
        "windowStride": config.window_stride,
        "useLstmCore": config.use_lstm_core,
        "lstmNumLayers": config.lstm_num_layers,
        "teacherForcingMode": config.teacher_forcing_mode,
        "actionTypeWeighting": action_type_weight_payload,
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
                preferred_checkpoint_names=[
                    "best_val_free_action.pt",
                    "best_val_loss.pt",
                    "best.pt",
                    "latest.pt",
                ],
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
