from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_lib.dataset import ModelShardFilter, ModelShardRecord, discover_model_shards, summarize_model_shards


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("d:/workspace/ra2-headless-mix")
DEFAULT_JS_SCRIPT = PROJECT_ROOT / "packages" / "py-chronodivide" / "extract_action_labels.mjs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    parts = [part for part in parts if part]
    return parts or None


def parse_replay_ids(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    replay_section = text
    if "Replays:" in replay_section:
        replay_section = replay_section.split("Replays:", 1)[1]
    if "Ambiguous Pinch Point replays" in replay_section:
        replay_section = replay_section.split("Ambiguous Pinch Point replays", 1)[0]
    replay_ids = re.findall(r"- `([0-9a-fA-F-]{36})\.rpl`", replay_section)
    if not replay_ids:
        raise ValueError(f"No replay ids found in winner list: {markdown_path}")
    return replay_ids


def build_node_command(
    *,
    replay_path: Path,
    data_dir: Path,
    script_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        "node",
        str(script_path),
        "--replay",
        str(replay_path),
        "--data-dir",
        str(data_dir),
        "--player",
        "all",
        "--include-ui-actions",
        "true",
        "--output",
        str(output_path),
    ]


def extract_action_labels(
    *,
    replay_path: Path,
    data_dir: Path,
    script_path: Path,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="winner_manifest_", suffix=".json", delete=False) as handle:
        temp_output_path = Path(handle.name)
    try:
        command = build_node_command(
            replay_path=replay_path,
            data_dir=data_dir,
            script_path=script_path,
            output_path=temp_output_path,
        )
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(PACKAGE_ROOT),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            stdout_text = (completed.stdout or "").strip()
            details = stderr_text or stdout_text or f"Action-label extractor exited with status {completed.returncode}."
            raise RuntimeError(details)
        with temp_output_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        temp_output_path.unlink(missing_ok=True)


def infer_winner_player_name(payload: dict[str, Any]) -> str:
    replay_players = [str(player.get("name")) for player in payload.get("replay", {}).get("players", []) if player.get("name")]
    if len(replay_players) != 2:
        raise ValueError(f"Expected exactly 2 replay players, got {replay_players!r}")

    terminal_actions = [
        action
        for action in payload.get("actions", [])
        if str(action.get("rawActionName")) in {"ResignGameAction", "DropPlayerAction"}
    ]
    if not terminal_actions:
        raise ValueError("Could not infer a winner: no resign/drop action present.")

    terminal_actions.sort(key=lambda action: (int(action.get("tick", -1)), int(action.get("playerId", -1))))
    loser_name = str(terminal_actions[-1].get("playerName"))
    winner_candidates = [player_name for player_name in replay_players if player_name != loser_name]
    if len(winner_candidates) != 1:
        raise ValueError(
            f"Could not infer a unique winner from replay players {replay_players!r} and loser {loser_name!r}."
        )
    return winner_candidates[0]


def build_winner_record_map(
    records: list[ModelShardRecord],
    *,
    data_dir: Path,
    script_path: Path,
) -> dict[str, ModelShardRecord]:
    records_by_game_id: dict[str, list[ModelShardRecord]] = {}
    for record in records:
        records_by_game_id.setdefault(record.replay_game_id, []).append(record)

    winner_records: dict[str, ModelShardRecord] = {}
    for game_id, replay_records in records_by_game_id.items():
        if not replay_records:
            continue
        replay_path = Path(replay_records[0].replay_path)
        payload = extract_action_labels(
            replay_path=replay_path,
            data_dir=data_dir,
            script_path=script_path,
        )
        winner_name = infer_winner_player_name(payload)
        matching = [record for record in replay_records if record.player_name == winner_name]
        if len(matching) != 1:
            raise ValueError(
                f"Expected exactly one winning shard for replay {game_id}, winner {winner_name!r}, found {len(matching)}."
            )
        winner_records[game_id] = matching[0]
    return winner_records


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a winner-only manifest from an existing tensor corpus.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--winner-replay-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--py-chronodivide-script", type=Path, default=DEFAULT_JS_SCRIPT)
    parser.add_argument("--map-name", type=str, default=None)
    parser.add_argument("--player-country", type=str, default=None)
    parser.add_argument("--player-name", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--side-id", type=int, default=None)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tensor_dir = args.tensor_dir.resolve()
    output_path = args.output.resolve()
    winner_replay_list = args.winner_replay_list.resolve()
    data_dir = args.data_dir.resolve()
    script_path = args.py_chronodivide_script.resolve()

    replay_ids = set(parse_replay_ids(winner_replay_list))
    shard_filter = ModelShardFilter.create(
        map_names=parse_csv(args.map_name),
        player_country_names=parse_csv(args.player_country),
        player_names=parse_csv(args.player_name),
        replay_game_ids=sorted(replay_ids if args.game_id is None else set(parse_csv(args.game_id) or []) & replay_ids),
        player_side_ids=[int(args.side_id)] if args.side_id is not None else None,
    )
    records = discover_model_shards(tensor_dir, shard_filter=shard_filter)
    if not records:
        raise ValueError("No shards matched the requested winner-only manifest filters.")

    winner_records_by_game_id = build_winner_record_map(
        records,
        data_dir=data_dir,
        script_path=script_path,
    )
    selected_records = [winner_records_by_game_id[game_id] for game_id in sorted(winner_records_by_game_id)]
    summary = summarize_model_shards(selected_records)

    payload = {
        "createdAt": utc_now_iso(),
        "label": args.label,
        "tensorDir": str(tensor_dir),
        "winnerReplayListPath": str(winner_replay_list),
        "dataDir": str(data_dir),
        "pyChronodivideScript": str(script_path),
        "filters": {
            "mapNames": parse_csv(args.map_name),
            "playerCountries": parse_csv(args.player_country),
            "playerNames": parse_csv(args.player_name),
            "gameIds": sorted(replay_ids if args.game_id is None else set(parse_csv(args.game_id) or []) & replay_ids),
            "sideIds": [int(args.side_id)] if args.side_id is not None else None,
            "winnerOnly": True,
        },
        "summary": summary,
        "shardStems": [record.stem for record in selected_records],
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
            for record in selected_records
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
