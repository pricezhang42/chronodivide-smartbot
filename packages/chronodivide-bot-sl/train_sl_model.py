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

from model_lib.batch import collate_model_samples
from model_lib.dataset import ModelShardFilter, ModelShardRecord, RA2SLSectionDataset, discover_model_shards
from model_lib.losses import compute_ra2_sl_loss
from model_lib.model import RA2SLBaselineConfig, RA2SLBaselineModel


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
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    val_ratio: float
    seed: int
    cache_size: int
    checkpoint_dir: Path
    device: str | None
    resume: Path | None


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)

    args = parser.parse_args()
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_size=args.cache_size,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        resume=args.resume.resolve() if args.resume is not None else None,
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


def run_epoch(
    *,
    model: RA2SLBaselineModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
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
            outputs = model(batch["model_inputs"])
            loss_output = compute_ra2_sl_loss(outputs, batch)
            if training:
                loss_output.total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        accumulator["total_loss"] += float(loss_output.total_loss.detach().item())
        accumulator["batch_count"] += 1.0
        accumulator["sample_count"] += float(batch["feature_sections"]["scalar"].shape[0])
        accumulate_metric_group(accumulator, "loss", loss_output.loss_by_head)
        accumulate_metric_group(accumulator, "metric", loss_output.metrics)

    metrics = finalize_metrics(accumulator)
    elapsed_seconds = max(1e-6, time.perf_counter() - start_time)
    sample_count = accumulator.get("sample_count", 0.0)
    metrics["epochSeconds"] = elapsed_seconds
    metrics["samplesPerSecond"] = sample_count / elapsed_seconds
    metrics["sampleCount"] = sample_count
    return metrics


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
    config: TrainConfig,
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
            "config": asdict(config),
            "best_val_loss": best_val_loss,
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

    train_dataset = RA2SLSectionDataset(train_records, cache_size=config.cache_size)
    val_dataset = RA2SLSectionDataset(val_records, cache_size=config.cache_size) if val_records else None
    train_dataset = limit_dataset(train_dataset, config.max_train_samples)
    if val_dataset is not None:
        val_dataset = limit_dataset(val_dataset, config.max_val_samples)
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False) if val_dataset is not None else None

    entity_name_vocab_size = infer_entity_name_vocab_size(records)
    device = resolve_device(config)
    model = RA2SLBaselineModel(
        RA2SLBaselineConfig(
            entity_name_vocab_size=entity_name_vocab_size,
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
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = checkpoint.get("best_val_loss")

    split_payload = {
        "createdAt": utc_now_iso(),
        "config": asdict(config),
        "manifestPath": str(config.manifest_path) if config.manifest_path is not None else None,
        "device": str(device),
        "trainShardCount": len(train_records),
        "valShardCount": len(val_records),
        "trainSampleCount": len(train_dataset),
        "valSampleCount": len(val_dataset) if val_dataset is not None else 0,
        "trainShards": [record.stem for record in train_records],
        "valShards": [record.stem for record in val_records],
    }
    save_json(config.output_dir / "data_split.json", split_payload)

    history_path = config.output_dir / "history.jsonl"
    if start_epoch == 0 and history_path.exists():
        history_path.unlink()

    last_train_metrics: dict[str, float] | None = None
    last_val_metrics: dict[str, float] | None = None
    for epoch in range(start_epoch, config.epochs):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                data_loader=val_loader,
                optimizer=None,
                device=device,
                grad_clip_norm=config.grad_clip_norm,
            )

        scheduler.step()
        last_train_metrics = train_metrics
        last_val_metrics = val_metrics

        summary = {
            "epoch": epoch,
            "timestamp": utc_now_iso(),
            "train": train_metrics,
            "val": val_metrics,
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
        )

        candidate_loss = train_metrics["total_loss"] if val_metrics is None else val_metrics["total_loss"]
        if best_val_loss is None or candidate_loss < best_val_loss:
            best_val_loss = candidate_loss

        save_checkpoint(
            path=config.checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            config=config,
            best_val_loss=best_val_loss,
        )
        if candidate_loss == best_val_loss:
            save_checkpoint(
                path=config.checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                config=config,
                best_val_loss=best_val_loss,
            )

    final_summary = {
        "createdAt": utc_now_iso(),
        "device": str(device),
        "trainShardCount": len(train_records),
        "valShardCount": len(val_records),
        "trainSampleCount": len(train_dataset),
        "valSampleCount": len(val_dataset) if val_dataset is not None else 0,
        "entityNameVocabSize": entity_name_vocab_size,
        "epochs": config.epochs,
        "checkpointDir": str(config.checkpoint_dir),
        "bestValLoss": best_val_loss,
        "finalTrainMetrics": last_train_metrics,
        "finalValMetrics": last_val_metrics,
    }
    save_json(config.output_dir / "training_summary.json", final_summary)


if __name__ == "__main__":
    main()
