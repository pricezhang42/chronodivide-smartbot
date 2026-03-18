from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_lib.batch import collate_model_samples
from model_lib.dataset import ModelShardFilter, RA2SLSectionDataset, summarize_model_shards
from model_lib.model import RA2SLCoreConfig, RA2SLCoreModel


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward-pass smoke test for the RA2 SL model core.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=2)
    return parser.parse_args()


def _print_tensor_shape(name: str, tensor: torch.Tensor) -> None:
    print(f"  - {name}: shape={tuple(int(dimension) for dimension in tensor.shape)} dtype={tensor.dtype}")


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

    first_record = dataset.shard_records[0]
    shared_vocab_size = len(first_record.metadata["schema"]["sharedNameVocabulary"]["idToName"])
    model = RA2SLCoreModel(
        RA2SLCoreConfig(
            entity_name_vocab_size=shared_vocab_size,
        )
    )
    model.eval()

    batch_size = min(int(args.batch_size), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    batch = next(iter(loader))
    with torch.no_grad():
        outputs = model(batch["model_inputs"])

    print(f"Forward batch size: {batch_size}")
    print("Model outputs:")
    for name, tensor in outputs.items():
        _print_tensor_shape(name, tensor)


if __name__ == "__main__":
    main()
