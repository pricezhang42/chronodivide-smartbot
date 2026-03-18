from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional convenience dependency
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PACKAGE_ROOT / "ladder_replays_top50"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "tensor_shards"
DEFAULT_PY_CHRONODIVIDE_SCRIPT = PROJECT_ROOT / "packages" / "py-chronodivide" / "extract_sl_tensors.mjs"
DEFAULT_DATA_DIR = Path(os.environ.get("CHRONODIVIDE_DATA_DIR", "d:/workspace/ra2-headless-mix"))

LABEL_LAYOUT_V1_VERSION = "v1"
LABEL_LAYOUT_V1_DELAY_BINS = 128
LABEL_LAYOUT_V1_MISSING_INT = -1
LABEL_LAYOUT_V1_QUANTITY_POLICY = {
    "canonicalStorage": "raw_integer",
    "missingValue": LABEL_LAYOUT_V1_MISSING_INT,
    "derivedTrainingTarget": "quantityValue int32",
    "bucketing": "not used in V1",
    "notes": [
        "Canonical V1 stores replay quantity as-is when used and -1 when unused.",
        "Any quantity bucketing is deferred to a later label version or model-specific derived target.",
    ],
}
LABEL_LAYOUT_V1_SUPERVISION_POLICY = {
    "semanticMaskMeaning": "whether a head is conceptually used by the action type",
    "resolutionMaskMeaning": "whether replay-time alignment produced a valid supervision target",
    "queue": "supervise only when the action type uses queue and queue is in {0,1}",
    "units": "supervise only positions where the action type uses units, unitsMask=1, and unitsResolvedMask=1",
    "targetEntity": "supervise only when the action type uses target_entity and targetEntityResolved=1",
    "targetLocation": "supervise when the action type uses target_location and targetLocationValid=1",
    "targetLocation2": "supervise when the action type uses target_location_2 and targetLocation2Valid=1",
    "quantity": "supervise only when the action type uses quantity and quantity >= 0",
    "notes": [
        "Entity-resolution failure suppresses target-entity supervision but does not suppress valid location supervision.",
        "Units are supervised position-by-position; unresolved selected units do not contribute loss.",
    ],
}
LABEL_LAYOUT_V1_ACTION_VOCAB_SCOPE = "per_replay_extract"
LABEL_LAYOUT_V1_CORE_LABEL_SECTIONS = [
    "actionTypeId",
    "delayBin",
    "queue",
    "unitsIndices",
    "unitsMask",
    "unitsResolvedMask",
    "targetEntityIndex",
    "targetEntityResolved",
    "targetLocation",
    "targetLocationValid",
    "targetLocation2",
    "targetLocation2Valid",
    "quantity",
]

FEATURE_CONTEXT_V1_VERSION = "v1"
TRAINING_TARGETS_V1_VERSION = "v1"


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


def schema_dtype_to_torch(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported schema dtype: {dtype_name}") from exc


def validate_probability(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")
    return value


@dataclass
class TransformConfig:
    replay_dir: Path
    output_dir: Path
    data_dir: Path
    py_chronodivide_script: Path
    extract_cache_dir: Path | None
    refresh_extract_cache: bool
    replay_glob: str
    replay_start: int
    max_replays: int | None
    player: str
    include_no_action: bool
    include_ui_actions: bool
    action_filter_profile: str
    filter_seed: int
    no_action_keep_prob: float
    select_units_keep_prob: float
    move_order_keep_prob: float
    gather_order_keep_prob: float
    attack_order_keep_prob: float
    max_actions: int | None
    max_tick: int | None
    max_entities: int
    max_selected_units: int
    spatial_size: int
    minimap_size: int
    overwrite: bool
    fail_fast: bool


@dataclass
class TransformRunState:
    action_type_counts: dict[str, int] = field(default_factory=dict)
    unseen_action_type_counts: dict[str, int] = field(default_factory=dict)


def parse_args(argv: list[str]) -> TransformConfig:
    parser = argparse.ArgumentParser(description="Transform Chronodivide replays into replay-level torch shards.")
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--py-chronodivide-script", type=Path, default=DEFAULT_PY_CHRONODIVIDE_SCRIPT)
    parser.add_argument("--extract-cache-dir", type=Path, default=None)
    parser.add_argument("--no-extract-cache", action="store_true")
    parser.add_argument("--refresh-extract-cache", action="store_true")
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
    parser.add_argument("--action-filter-profile", choices=["none", "mas"], default="mas")
    parser.add_argument("--filter-seed", type=int, default=0)
    parser.add_argument("--no-action-keep-prob", type=float, default=0.0)
    parser.add_argument("--select-units-keep-prob", type=float, default=0.2)
    parser.add_argument("--move-order-keep-prob", type=float, default=0.1)
    parser.add_argument("--gather-order-keep-prob", type=float, default=0.1)
    parser.add_argument("--attack-order-keep-prob", type=float, default=0.5)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-tick", type=int, default=None)
    parser.add_argument("--max-entities", type=int, default=128)
    parser.add_argument("--max-selected-units", type=int, default=64)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--minimap-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    args = parser.parse_args(argv)
    resolved_output_dir = args.output_dir.resolve()
    extract_cache_dir = None
    if not args.no_extract_cache:
        extract_cache_dir = (
            args.extract_cache_dir.resolve()
            if args.extract_cache_dir is not None
            else resolved_output_dir / "_extract_cache"
        )
    return TransformConfig(
        replay_dir=args.replay_dir.resolve(),
        output_dir=resolved_output_dir,
        data_dir=args.data_dir.resolve(),
        py_chronodivide_script=args.py_chronodivide_script.resolve(),
        extract_cache_dir=extract_cache_dir,
        refresh_extract_cache=bool(args.refresh_extract_cache),
        replay_glob=args.replay_glob,
        replay_start=max(0, args.replay_start),
        max_replays=args.max_replays,
        player=str(args.player),
        include_no_action=bool(args.include_no_action),
        include_ui_actions=bool(args.include_ui_actions),
        action_filter_profile=str(args.action_filter_profile),
        filter_seed=int(args.filter_seed),
        no_action_keep_prob=validate_probability("no_action_keep_prob", float(args.no_action_keep_prob)),
        select_units_keep_prob=validate_probability("select_units_keep_prob", float(args.select_units_keep_prob)),
        move_order_keep_prob=validate_probability("move_order_keep_prob", float(args.move_order_keep_prob)),
        gather_order_keep_prob=validate_probability("gather_order_keep_prob", float(args.gather_order_keep_prob)),
        attack_order_keep_prob=validate_probability("attack_order_keep_prob", float(args.attack_order_keep_prob)),
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


def player_output_stem(replay_path: Path, player_name: str) -> str:
    return f"{replay_path.stem}__{sanitize_filename(player_name)}"
