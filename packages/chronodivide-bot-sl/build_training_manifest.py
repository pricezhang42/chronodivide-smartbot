from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from model_lib.dataset import ModelShardFilter, discover_model_shards, summarize_model_shards


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible shard manifest for RA2 SL training.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor_dir = args.tensor_dir.resolve()
    output_path = args.output.resolve()
    shard_filter = ModelShardFilter.create(
        map_names=parse_csv(args.map_name),
        player_country_names=parse_csv(args.player_country),
        player_names=parse_csv(args.player_name),
        replay_game_ids=parse_csv(args.game_id),
        player_side_ids=[int(args.side_id)] if args.side_id is not None else None,
    )
    records = discover_model_shards(tensor_dir, shard_filter=shard_filter)
    if args.max_shards is not None:
        records = records[: args.max_shards]
    if not records:
        raise ValueError("No shards matched the requested manifest filters.")

    summary = summarize_model_shards(records)
    payload = {
        "createdAt": utc_now_iso(),
        "label": args.label,
        "tensorDir": str(tensor_dir),
        "filters": {
            "mapNames": parse_csv(args.map_name),
            "playerCountries": parse_csv(args.player_country),
            "playerNames": parse_csv(args.player_name),
            "gameIds": parse_csv(args.game_id),
            "sideIds": [int(args.side_id)] if args.side_id is not None else None,
            "maxShards": args.max_shards,
        },
        "summary": summary,
        "shardStems": [record.stem for record in records],
        "shards": [
            {
                "stem": record.stem,
                "gameId": record.replay_game_id,
                "mapName": record.map_name,
                "playerName": record.player_name,
                "playerCountryName": record.player_country_name,
                "playerSideId": record.player_side_id,
                "sampleCount": record.sample_count,
                "metaPath": str(record.meta_path),
            }
            for record in records
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
