from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from model_lib.batch import collate_model_samples
from model_lib.dataset import ModelShardFilter, RA2SLSectionDataset, summarize_model_shards


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    return items or None


def _print_tensor_dict(title: str, tensor_dict: dict[str, object]) -> None:
    print(title)
    for name, tensor in tensor_dict.items():
        shape = tuple(int(dimension) for dimension in tensor.shape)
        print(f"  - {name}: shape={shape} dtype={tensor.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect structured RA2 SL tensor shards for model-building.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_filter = ModelShardFilter.create(
        map_names=_parse_csv(args.map_name),
        player_country_names=_parse_csv(args.player_country),
        player_names=_parse_csv(args.player_name),
        replay_game_ids=_parse_csv(args.game_id),
        player_side_ids=[args.side_id] if args.side_id is not None else None,
    )

    dataset = RA2SLSectionDataset.from_directory(
        args.tensor_dir,
        shard_filter=shard_filter,
        cache_size=args.cache_size,
    )
    summary = summarize_model_shards(dataset.shard_records)
    print("Shard summary:")
    print(json.dumps(summary, indent=2))
    print()

    sample = dataset[0]
    print("First sample metadata:")
    print(json.dumps(sample["metadata"], indent=2))
    print()
    _print_tensor_dict("Feature sections:", sample["feature_sections"])
    print()
    _print_tensor_dict("Label sections:", sample["label_sections"])
    print()
    _print_tensor_dict("Training targets:", sample["training_targets"])
    print()
    _print_tensor_dict("Training masks:", sample["training_masks"])
    print()
    _print_tensor_dict("Sample context:", sample["sample_context"])
    print()

    batch_size = min(int(args.batch_size), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    batch = next(iter(loader))
    print(f"First batch size: {batch_size}")
    _print_tensor_dict("Batched feature sections:", batch["feature_sections"])
    print()
    _print_tensor_dict("Batched training targets:", batch["training_targets"])
    print()
    _print_tensor_dict("Batched training masks:", batch["training_masks"])
    print()
    print("Model input groups:")
    _print_tensor_dict("  Scalar sections:", batch["model_inputs"]["scalar_sections"])
    _print_tensor_dict("  Entity inputs:", batch["model_inputs"]["entity"])
    _print_tensor_dict("  Spatial inputs:", batch["model_inputs"]["spatial"])
    print()
    print(f"Inspected {min(len(dataset), int(args.max_samples))} sample(s) from {len(dataset)} total samples.")


if __name__ == "__main__":
    main()
