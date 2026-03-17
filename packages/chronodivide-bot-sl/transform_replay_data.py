#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transform Chronodivide replays into replay-level (features, labels) torch shards.

This is the SL-transformer-side bridge that mirrors the role of
mini-AlphaStar's `transform_replay_data.py`, while delegating replay parsing,
feature extraction, and action-label decoding to the reusable `py-chronodivide`
package.

Current behavior:
- iterate replay files
- call `py-chronodivide/extract_sl_tensors.mjs` with flat tensors enabled
- group samples by player perspective
- save replay-player `.pt` shards as `(features, labels)`
- save sidecar metadata and a run manifest

Notes:
- feature tensors are stored as `float32`
- label tensors are stored as `int64`
- per-replay vocabularies still come from `py-chronodivide`, so schema metadata
  is saved per shard instead of globally
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional convenience dependency
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_REPLAY_DIR = PACKAGE_ROOT / "ladder_replays_top50"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "tensor_shards"
DEFAULT_PY_CHRONODIVIDE_SCRIPT = PROJECT_ROOT / "packages" / "py-chronodivide" / "extract_sl_tensors.mjs"
DEFAULT_DATA_DIR = Path(os.environ.get("CHRONODIVIDE_DATA_DIR", "d:/workspace/ra2-headless-mix"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._") or "player"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def feature_dtype() -> torch.dtype:
    return torch.float32


def label_dtype() -> torch.dtype:
    return torch.int64


@dataclass
class TransformConfig:
    replay_dir: Path
    output_dir: Path
    data_dir: Path
    py_chronodivide_script: Path
    replay_glob: str
    replay_start: int
    max_replays: int | None
    player: str
    include_no_action: bool
    include_ui_actions: bool
    max_actions: int | None
    max_tick: int | None
    max_entities: int
    max_selected_units: int
    spatial_size: int
    minimap_size: int
    overwrite: bool
    fail_fast: bool


def parse_args(argv: list[str]) -> TransformConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--py-chronodivide-script", type=Path, default=DEFAULT_PY_CHRONODIVIDE_SCRIPT)
    parser.add_argument("--replay-glob", default="*.rpl")
    parser.add_argument("--replay-start", type=int, default=0)
    parser.add_argument("--max-replays", type=int, default=None)
    parser.add_argument(
        "--player",
        default="first",
        help='Player selection mode: "first", "all", or an explicit player name.',
    )
    parser.add_argument("--include-no-action", action="store_true")
    parser.add_argument("--include-ui-actions", action="store_true")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-tick", type=int, default=None)
    parser.add_argument("--max-entities", type=int, default=128)
    parser.add_argument("--max-selected-units", type=int, default=64)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--minimap-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    args = parser.parse_args(argv)
    return TransformConfig(
        replay_dir=args.replay_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        py_chronodivide_script=args.py_chronodivide_script.resolve(),
        replay_glob=args.replay_glob,
        replay_start=max(0, args.replay_start),
        max_replays=args.max_replays,
        player=str(args.player),
        include_no_action=bool(args.include_no_action),
        include_ui_actions=bool(args.include_ui_actions),
        max_actions=args.max_actions,
        max_tick=args.max_tick,
        max_entities=args.max_entities,
        max_selected_units=args.max_selected_units,
        spatial_size=args.spatial_size,
        minimap_size=args.minimap_size,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
    )


def validate_config(config: TransformConfig) -> None:
    if not config.replay_dir.exists():
        raise FileNotFoundError(f"Replay directory does not exist: {config.replay_dir}")
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Chronodivide data directory does not exist: {config.data_dir}")
    if not config.py_chronodivide_script.exists():
        raise FileNotFoundError(f"py-chronodivide extractor does not exist: {config.py_chronodivide_script}")


def list_replays(config: TransformConfig) -> list[Path]:
    replay_paths = sorted(config.replay_dir.glob(config.replay_glob))
    replay_paths = replay_paths[config.replay_start :]
    if config.max_replays is not None:
        replay_paths = replay_paths[: config.max_replays]
    return replay_paths


def iter_progress(items: list[Path], description: str) -> Any:
    if tqdm is None:
        return items
    return tqdm(items, desc=description)


def node_player_arg(player: str) -> str | None:
    normalized = player.strip()
    if normalized.lower() == "first":
        return None
    return normalized


def build_node_command(config: TransformConfig, replay_path: Path, output_path: Path) -> list[str]:
    command = [
        "node",
        str(config.py_chronodivide_script),
        "--replay",
        str(replay_path),
        "--data-dir",
        str(config.data_dir),
        "--max-entities",
        str(config.max_entities),
        "--max-selected-units",
        str(config.max_selected_units),
        "--spatial-size",
        str(config.spatial_size),
        "--minimap-size",
        str(config.minimap_size),
        "--include-flat",
        "true",
        "--output",
        str(output_path),
    ]

    player_arg = node_player_arg(config.player)
    if player_arg:
        command.extend(["--player", player_arg])
    if config.include_no_action:
        command.extend(["--include-no-action", "true"])
    if config.include_ui_actions:
        command.extend(["--include-ui-actions", "true"])
    if config.max_actions is not None:
        command.extend(["--max-actions", str(config.max_actions)])
    if config.max_tick is not None:
        command.extend(["--max-tick", str(config.max_tick)])
    return command


def run_py_chronodivide_extract(config: TransformConfig, replay_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="chronodivide_sl_") as temp_dir:
        temp_output = Path(temp_dir) / "dataset.json"
        command = build_node_command(config, replay_path, temp_output)
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    [
                        f"py-chronodivide extraction failed for {replay_path.name}",
                        f"command: {' '.join(command)}",
                        f"stdout tail: {completed.stdout[-4000:]}",
                        f"stderr tail: {completed.stderr[-4000:]}",
                    ]
                )
            )
        if not temp_output.exists():
            raise RuntimeError(f"Expected extractor output file was not created: {temp_output}")
        with temp_output.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def group_samples_by_player(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        player_name = sample["playerName"]
        grouped.setdefault(player_name, []).append(sample)
    return grouped


def validate_flat_tensor_lengths(samples: list[dict[str, Any]], schema: dict[str, Any], replay_name: str, player_name: str) -> None:
    expected_feature_length = int(schema["flatFeatureLength"])
    expected_label_length = int(schema["flatLabelLength"])

    for index, sample in enumerate(samples):
        feature_length = len(sample.get("flatFeatureTensor", []))
        label_length = len(sample.get("flatLabelTensor", []))
        if feature_length != expected_feature_length:
            raise ValueError(
                f"{replay_name}/{player_name} sample {index} has flat feature length {feature_length}, "
                f"expected {expected_feature_length}."
            )
        if label_length != expected_label_length:
            raise ValueError(
                f"{replay_name}/{player_name} sample {index} has flat label length {label_length}, "
                f"expected {expected_label_length}."
            )


def build_tensors(samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows = [sample["flatFeatureTensor"] for sample in samples]
    label_rows = [sample["flatLabelTensor"] for sample in samples]
    features = torch.tensor(feature_rows, dtype=feature_dtype())
    labels = torch.tensor(label_rows, dtype=label_dtype())
    return features, labels


def player_output_stem(replay_path: Path, player_name: str) -> str:
    return f"{replay_path.stem}__{sanitize_filename(player_name)}"


def write_player_shard(
    config: TransformConfig,
    replay_path: Path,
    dataset: dict[str, Any],
    player_name: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    output_stem = player_output_stem(replay_path, player_name)
    tensor_path = config.output_dir / f"{output_stem}.pt"
    metadata_path = config.output_dir / f"{output_stem}.meta.json"

    if tensor_path.exists() and metadata_path.exists() and not config.overwrite:
        return {
            "status": "skipped",
            "replay": str(replay_path),
            "playerName": player_name,
            "tensorPath": str(tensor_path),
            "metadataPath": str(metadata_path),
            "reason": "existing shard",
        }

    validate_flat_tensor_lengths(samples, dataset["schema"], replay_path.name, player_name)
    features, labels = build_tensors(samples)

    ensure_parent(tensor_path)
    torch.save((features, labels), tensor_path)

    metadata = {
        "createdAt": utc_now_iso(),
        "replay": dataset["replay"],
        "playerName": player_name,
        "sampleCount": len(samples),
        "featureShape": list(features.shape),
        "labelShape": list(labels.shape),
        "featureDType": str(features.dtype),
        "labelDType": str(labels.dtype),
        "schema": dataset["schema"],
        "sourceOptions": dataset["options"],
        "sourceCounts": dataset["counts"],
        "tensorPath": str(tensor_path),
    }
    ensure_parent(metadata_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "status": "saved",
        "replay": str(replay_path),
        "playerName": player_name,
        "sampleCount": len(samples),
        "featureShape": list(features.shape),
        "labelShape": list(labels.shape),
        "tensorPath": str(tensor_path),
        "metadataPath": str(metadata_path),
    }


def transform_single_replay(config: TransformConfig, replay_path: Path) -> list[dict[str, Any]]:
    dataset = run_py_chronodivide_extract(config, replay_path)
    grouped_samples = group_samples_by_player(dataset.get("samples", []))
    if not grouped_samples:
        return [
            {
                "status": "empty",
                "replay": str(replay_path),
                "reason": "extractor returned no samples",
            }
        ]

    shard_results = []
    for player_name, player_samples in grouped_samples.items():
        shard_results.append(write_player_shard(config, replay_path, dataset, player_name, player_samples))
    return shard_results


def write_manifest(config: TransformConfig, replay_paths: list[Path], results: list[dict[str, Any]], errors: list[dict[str, Any]]) -> Path:
    config_dict = asdict(config)
    config_dict["replay_dir"] = str(config.replay_dir)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["data_dir"] = str(config.data_dir)
    config_dict["py_chronodivide_script"] = str(config.py_chronodivide_script)

    manifest = {
        "createdAt": utc_now_iso(),
        "config": config_dict,
        "replayCount": len(replay_paths),
        "savedShardCount": sum(1 for result in results if result["status"] == "saved"),
        "skippedShardCount": sum(1 for result in results if result["status"] == "skipped"),
        "emptyReplayCount": sum(1 for result in results if result["status"] == "empty"),
        "errorCount": len(errors),
        "results": results,
        "errors": errors,
    }
    manifest_path = config.output_dir / "manifest.json"
    ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    replay_paths = list_replays(config)
    if not replay_paths:
        raise FileNotFoundError(
            f"No replay files matching {config.replay_glob!r} were found in {config.replay_dir}."
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for replay_path in iter_progress(replay_paths, "Transform replays"):
        try:
            results.extend(transform_single_replay(config, replay_path))
        except Exception as exc:  # pragma: no cover - failure path exercised by real runs
            error_record = {
                "replay": str(replay_path),
                "errorType": exc.__class__.__name__,
                "error": str(exc),
            }
            errors.append(error_record)
            if config.fail_fast:
                raise

    manifest_path = write_manifest(config, replay_paths, results, errors)
    print(f"Processed {len(replay_paths)} replay(s).")
    print(f"Saved {sum(1 for result in results if result['status'] == 'saved')} shard(s).")
    print(f"Manifest: {manifest_path}")

    if errors:
        print(f"Encountered {len(errors)} replay error(s). See manifest for details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
