from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_lib.batch import collate_model_samples
from model_lib.dataset import ModelShardFilter, RA2SLSectionDataset, summarize_model_shards
from model_lib.losses import compute_ra2_sl_loss
from model_lib.model import RA2SLBaselineConfig, RA2SLBaselineModel


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-batch training-step smoke test for the RA2 SL baseline model.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=2)
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

    first_record = dataset.shard_records[0]
    shared_vocab_size = len(first_record.metadata["schema"]["sharedNameVocabulary"]["idToName"])
    model = RA2SLBaselineModel(
        RA2SLBaselineConfig(
            entity_name_vocab_size=shared_vocab_size,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    batch_size = min(int(args.batch_size), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_model_samples,
    )
    batch = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["model_inputs"])
    loss_output = compute_ra2_sl_loss(outputs, batch)
    loss_output.total_loss.backward()
    optimizer.step()

    print(f"Batch size: {batch_size}")
    print(f"Total loss: {float(loss_output.total_loss.item()):.6f}")
    print("Per-head losses:")
    for name, value in loss_output.loss_by_head.items():
        print(f"  - {name}: {float(value.item()):.6f}")
    print("Metrics:")
    for name, value in loss_output.metrics.items():
        print(f"  - {name}: {float(value.item()):.6f}")

    grad_norm_sum = 0.0
    param_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm_sum += float(parameter.grad.norm().item())
        param_count += 1
    print(f"Parameters with gradients: {param_count}")
    print(f"Sum of gradient norms: {grad_norm_sum:.6f}")


if __name__ == "__main__":
    main()
