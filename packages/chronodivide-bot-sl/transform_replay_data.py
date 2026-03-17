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
- derive replay-local RA2 SL V1 action-type metadata
- apply mAS-style action filtering/downsampling on the action-aligned sample stream
- save replay-player `.pt` shards as `(features, labels)`
- save replay-player `.sections.pt` sidecars with structured feature/label tensors
- save sidecar metadata and a run manifest

Notes:
- feature tensors are stored as `float32`
- label tensors are stored as `int64`
- structured section tensors preserve the schema dtypes from `py-chronodivide`
- per-replay vocabularies still come from `py-chronodivide`, so schema metadata
  is saved per shard instead of globally
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
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
LABEL_LAYOUT_V1_VERSION = "v1"
LABEL_LAYOUT_V1_DELAY_BINS = 128
LABEL_LAYOUT_V1_MISSING_INT = -1
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
    action_type_name_to_global_id: dict[str, int] = field(default_factory=dict)
    action_type_counts: dict[str, int] = field(default_factory=dict)


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


def get_schema_name_by_id(names: dict[str, Any], value: int | None) -> str | None:
    if value is None:
        return None
    return names.get(str(int(value)))


def get_legacy_label_tensors(sample: dict[str, Any]) -> dict[str, Any]:
    return sample.get("legacyLabelTensors", sample["labelTensors"])


def get_schema_list_name(names: list[Any], value: int | None) -> str | None:
    if value is None:
        return None
    numeric_value = int(value)
    if 0 <= numeric_value < len(names):
        return str(names[numeric_value])
    return None


def normalize_action_type_component(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    normalized = normalized.strip("_")
    return normalized or fallback


def get_shared_name_from_token(dataset: dict[str, Any], token: int | None) -> str | None:
    if token is None:
        return None
    numeric_token = int(token)
    if numeric_token < 0:
        return None
    vocabulary = dataset["schema"].get("sharedNameVocabulary", {})
    id_to_name = vocabulary.get("idToName", [])
    if 0 <= numeric_token < len(id_to_name):
        name = id_to_name[numeric_token]
        if isinstance(name, str):
            return name
    return None


def get_raw_action_name(sample: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_id = int(legacy_label_tensors["rawActionId"][0])
    return get_schema_name_by_id(dataset["schema"]["action"]["rawActionIdNames"], raw_action_id)


def get_action_family_name(sample: dict[str, Any], dataset: dict[str, Any]) -> str:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    action_family_id = int(legacy_label_tensors["actionFamilyId"][0])
    action_families = dataset["schema"]["action"]["actionFamilies"]
    if 0 <= action_family_id < len(action_families):
        return str(action_families[action_family_id])
    return "unknown"


def get_order_type_name(sample: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    order_type_id = int(legacy_label_tensors["orderTypeId"][0])
    if order_type_id < 0:
        return None
    return get_schema_name_by_id(dataset["schema"]["action"]["orderTypeNames"], order_type_id)


def get_target_mode_name(sample: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    target_mode_id = int(legacy_label_tensors["targetModeId"][0])
    if target_mode_id < 0:
        return None
    return get_schema_list_name(dataset["schema"]["action"]["targetModes"], target_mode_id)


def get_queue_update_type_name(sample: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    queue_update_type_id = int(legacy_label_tensors["queueUpdateTypeId"][0])
    if queue_update_type_id < 0:
        return None
    return get_schema_name_by_id(dataset["schema"]["action"]["queueUpdateTypeNames"], queue_update_type_id)


def get_super_weapon_type_name(sample: dict[str, Any], dataset: dict[str, Any]) -> str | None:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    super_weapon_type_id = int(legacy_label_tensors["superWeaponTypeId"][0])
    if super_weapon_type_id < 0:
        return None
    return get_schema_name_by_id(dataset["schema"]["action"]["superWeaponTypeNames"], super_weapon_type_id)


def action_type_name_v1(sample: dict[str, Any], dataset: dict[str, Any]) -> str:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_id = int(legacy_label_tensors["rawActionId"][0])
    raw_action_name = get_raw_action_name(sample, dataset)

    if raw_action_name == "SelectUnitsAction":
        return "SelectUnits"

    if raw_action_name == "OrderUnitsAction":
        order_type_name = normalize_action_type_component(get_order_type_name(sample, dataset), "UnknownOrder")
        target_mode_name = normalize_action_type_component(get_target_mode_name(sample, dataset), "none")
        return f"Order::{order_type_name}::{target_mode_name}"

    if raw_action_name == "UpdateQueueAction":
        queue_update_type_name = normalize_action_type_component(
            get_queue_update_type_name(sample, dataset),
            "UnknownQueueUpdate",
        )
        item_name = normalize_action_type_component(
            get_shared_name_from_token(dataset, int(legacy_label_tensors["itemNameToken"][0])),
            "UnknownItem",
        )
        return f"Queue::{queue_update_type_name}::{item_name}"

    if raw_action_name == "PlaceBuildingAction":
        building_name = normalize_action_type_component(
            get_shared_name_from_token(dataset, int(legacy_label_tensors["buildingNameToken"][0])),
            "UnknownBuilding",
        )
        return f"PlaceBuilding::{building_name}"

    if raw_action_name == "ActivateSuperWeaponAction":
        super_weapon_name = normalize_action_type_component(
            get_super_weapon_type_name(sample, dataset),
            "UnknownSuperWeapon",
        )
        return f"ActivateSuperWeapon::{super_weapon_name}"

    if raw_action_name == "SellObjectAction":
        return "SellObject"

    if raw_action_name == "ToggleRepairAction":
        return "ToggleRepair"

    if raw_action_name == "ResignGameAction":
        return "ResignGame"

    if raw_action_name == "NoAction":
        return "NoAction"

    if raw_action_name == "DropPlayerAction":
        return "DropPlayer"

    if raw_action_name == "PingLocationAction":
        return "PingLocation"

    return normalize_action_type_component(raw_action_name, f"RawAction_{raw_action_id}")


def derive_action_type_semantic_mask(sample: dict[str, Any], dataset: dict[str, Any]) -> dict[str, bool]:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_name = get_raw_action_name(sample, dataset)
    target_mode_name = get_target_mode_name(sample, dataset)
    queue_value = int(legacy_label_tensors["queue"][0])
    target_location = legacy_label_tensors["targetTile"]
    building_tile = legacy_label_tensors["buildingTile"]
    super_weapon_tile = legacy_label_tensors["superWeaponTile"]
    super_weapon_tile2 = legacy_label_tensors["superWeaponTile2"]

    if raw_action_name == "SelectUnitsAction":
        return {
            "usesQueue": False,
            "usesUnits": True,
            "usesTargetEntity": False,
            "usesTargetLocation": False,
            "usesTargetLocation2": False,
            "usesQuantity": False,
        }

    if raw_action_name == "OrderUnitsAction":
        return {
            "usesQueue": queue_value >= 0,
            "usesUnits": True,
            "usesTargetEntity": target_mode_name == "object",
            "usesTargetLocation": target_mode_name in {"tile", "ore_tile"},
            "usesTargetLocation2": False,
            "usesQuantity": False,
        }

    if raw_action_name == "UpdateQueueAction":
        return {
            "usesQueue": False,
            "usesUnits": False,
            "usesTargetEntity": False,
            "usesTargetLocation": False,
            "usesTargetLocation2": False,
            "usesQuantity": True,
        }

    if raw_action_name == "PlaceBuildingAction":
        has_target_location = any(int(value) >= 0 for value in building_tile)
        return {
            "usesQueue": False,
            "usesUnits": False,
            "usesTargetEntity": False,
            "usesTargetLocation": has_target_location,
            "usesTargetLocation2": False,
            "usesQuantity": False,
        }

    if raw_action_name == "ActivateSuperWeaponAction":
        return {
            "usesQueue": False,
            "usesUnits": False,
            "usesTargetEntity": False,
            "usesTargetLocation": any(int(value) >= 0 for value in super_weapon_tile),
            "usesTargetLocation2": any(int(value) >= 0 for value in super_weapon_tile2),
            "usesQuantity": False,
        }

    if raw_action_name in {"SellObjectAction", "ToggleRepairAction"}:
        return {
            "usesQueue": False,
            "usesUnits": False,
            "usesTargetEntity": True,
            "usesTargetLocation": False,
            "usesTargetLocation2": False,
            "usesQuantity": False,
        }

    return {
        "usesQueue": False,
        "usesUnits": False,
        "usesTargetEntity": False,
        "usesTargetLocation": False,
        "usesTargetLocation2": False,
        "usesQuantity": False,
    }


def compute_flat_length(schema_sections: list[dict[str, Any]]) -> int:
    total = 0
    for section in schema_sections:
        length = 1
        for dimension in section["shape"]:
            length *= int(dimension)
        total += length
    return total


def append_schema_section(schema_sections: list[dict[str, Any]], *, name: str, shape: list[int], dtype: str) -> None:
    if any(section["name"] == name for section in schema_sections):
        return
    schema_sections.append({"name": name, "shape": shape, "dtype": dtype})


def get_section_shape(schema_sections: list[dict[str, Any]], section_name: str) -> list[int]:
    for section in schema_sections:
        if section["name"] == section_name:
            return list(section["shape"])
    raise KeyError(f"Section not found in schema: {section_name}")


def valid_tile(tile: list[int]) -> bool:
    return len(tile) == 2 and all(int(value) >= 0 for value in tile)


def canonical_target_entity_index(sample: dict[str, Any], dataset: dict[str, Any]) -> int:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_name = get_raw_action_name(sample, dataset)
    if raw_action_name in {"SellObjectAction", "ToggleRepairAction"}:
        return int(legacy_label_tensors["objectEntityIndex"][0])
    return int(legacy_label_tensors["targetEntityIndex"][0])


def canonical_target_location(sample: dict[str, Any], dataset: dict[str, Any]) -> list[int]:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_name = get_raw_action_name(sample, dataset)
    if raw_action_name == "PlaceBuildingAction":
        return [int(value) for value in legacy_label_tensors["buildingTile"]]
    if raw_action_name == "ActivateSuperWeaponAction":
        return [int(value) for value in legacy_label_tensors["superWeaponTile"]]
    return [int(value) for value in legacy_label_tensors["targetTile"]]


def canonical_target_location_2(sample: dict[str, Any], dataset: dict[str, Any]) -> list[int]:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_name = get_raw_action_name(sample, dataset)
    if raw_action_name == "ActivateSuperWeaponAction":
        return [int(value) for value in legacy_label_tensors["superWeaponTile2"]]
    return [LABEL_LAYOUT_V1_MISSING_INT, LABEL_LAYOUT_V1_MISSING_INT]


def build_v1_label_sections(legacy_label_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units_shape = get_section_shape(legacy_label_sections, "actionSelectedUnitIndices")
    return [
        {"name": "actionTypeId", "shape": [1], "dtype": "int32"},
        {"name": "delayBin", "shape": [1], "dtype": "int32"},
        {"name": "queue", "shape": [1], "dtype": "int32"},
        {"name": "unitsIndices", "shape": units_shape, "dtype": "int32"},
        {"name": "unitsMask", "shape": units_shape, "dtype": "int32"},
        {"name": "unitsResolvedMask", "shape": units_shape, "dtype": "int32"},
        {"name": "targetEntityIndex", "shape": [1], "dtype": "int32"},
        {"name": "targetEntityResolved", "shape": [1], "dtype": "int32"},
        {"name": "targetLocation", "shape": [2], "dtype": "int32"},
        {"name": "targetLocationValid", "shape": [1], "dtype": "int32"},
        {"name": "targetLocation2", "shape": [2], "dtype": "int32"},
        {"name": "targetLocation2Valid", "shape": [1], "dtype": "int32"},
        {"name": "quantity", "shape": [1], "dtype": "int32"},
    ]


def build_canonical_label_tensors_v1(sample: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    derived_metadata = sample.get("derivedLabelMetadata", {})
    semantic_mask = derived_metadata["semanticMaskV1"]
    action_type_name = derived_metadata["actionTypeNameV1"]
    action_type_id = int(dataset["labelLayoutV1"]["actionTypeNameToId"][action_type_name])

    delay_to_next = int(legacy_label_tensors["delayToNextAction"][0])
    delay_bin = LABEL_LAYOUT_V1_MISSING_INT
    if delay_to_next >= 0:
        delay_bin = min(delay_to_next, LABEL_LAYOUT_V1_DELAY_BINS - 1)

    queue_value = int(legacy_label_tensors["queue"][0]) if semantic_mask["usesQueue"] else LABEL_LAYOUT_V1_MISSING_INT

    units_length = len(legacy_label_tensors["actionSelectedUnitIndices"])
    if semantic_mask["usesUnits"]:
        units_indices = [int(value) for value in legacy_label_tensors["actionSelectedUnitIndices"]]
        units_mask = [int(value) for value in legacy_label_tensors["actionSelectedUnitMask"]]
        units_resolved_mask = [int(value) for value in legacy_label_tensors["actionSelectedUnitResolvedMask"]]
    else:
        units_indices = [LABEL_LAYOUT_V1_MISSING_INT] * units_length
        units_mask = [0] * units_length
        units_resolved_mask = [0] * units_length

    target_entity_index = LABEL_LAYOUT_V1_MISSING_INT
    target_entity_resolved = 0
    if semantic_mask["usesTargetEntity"]:
        target_entity_index = canonical_target_entity_index(sample, dataset)
        target_entity_resolved = int(target_entity_index >= 0)

    target_location = [LABEL_LAYOUT_V1_MISSING_INT, LABEL_LAYOUT_V1_MISSING_INT]
    target_location_valid = 0
    if semantic_mask["usesTargetLocation"]:
        target_location = canonical_target_location(sample, dataset)
        target_location_valid = int(valid_tile(target_location))

    target_location_2 = [LABEL_LAYOUT_V1_MISSING_INT, LABEL_LAYOUT_V1_MISSING_INT]
    target_location_2_valid = 0
    if semantic_mask["usesTargetLocation2"]:
        target_location_2 = canonical_target_location_2(sample, dataset)
        target_location_2_valid = int(valid_tile(target_location_2))

    quantity = LABEL_LAYOUT_V1_MISSING_INT
    if semantic_mask["usesQuantity"]:
        quantity = int(legacy_label_tensors["quantity"][0])

    return {
        "actionTypeId": [action_type_id],
        "delayBin": [delay_bin],
        "queue": [queue_value],
        "unitsIndices": units_indices,
        "unitsMask": units_mask,
        "unitsResolvedMask": units_resolved_mask,
        "targetEntityIndex": [target_entity_index],
        "targetEntityResolved": [target_entity_resolved],
        "targetLocation": target_location,
        "targetLocationValid": [target_location_valid],
        "targetLocation2": target_location_2,
        "targetLocation2Valid": [target_location_2_valid],
        "quantity": [quantity],
    }


def build_label_layout_v1_metadata(samples: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    action_type_counts: dict[str, int] = {}
    semantic_masks: dict[str, dict[str, bool]] = {}

    for sample in samples:
        action_type_name = action_type_name_v1(sample, dataset)
        sample_mask = derive_action_type_semantic_mask(sample, dataset)
        sample.setdefault("derivedLabelMetadata", {})["actionTypeNameV1"] = action_type_name
        sample["derivedLabelMetadata"]["semanticMaskV1"] = sample_mask
        action_type_counts[action_type_name] = action_type_counts.get(action_type_name, 0) + 1
        aggregated_mask = semantic_masks.setdefault(
            action_type_name,
            {
                "usesQueue": False,
                "usesUnits": False,
                "usesTargetEntity": False,
                "usesTargetLocation": False,
                "usesTargetLocation2": False,
                "usesQuantity": False,
            },
        )
        for key, value in sample_mask.items():
            aggregated_mask[key] = aggregated_mask[key] or bool(value)

    action_type_names = sorted(action_type_counts)
    action_type_name_to_id = {name: index for index, name in enumerate(action_type_names)}
    action_type_vocabulary = []
    for name in action_type_names:
        action_type_vocabulary.append(
            {
                "id": action_type_name_to_id[name],
                "name": name,
                "count": action_type_counts[name],
                "semanticMask": semantic_masks[name],
            }
        )

    return {
        "version": LABEL_LAYOUT_V1_VERSION,
        "status": "phase_1_and_3_started",
        "scope": LABEL_LAYOUT_V1_ACTION_VOCAB_SCOPE,
        "mainTensorUsesV1Only": False,
        "delayBins": LABEL_LAYOUT_V1_DELAY_BINS,
        "missingInt": LABEL_LAYOUT_V1_MISSING_INT,
        "coreLabelSections": LABEL_LAYOUT_V1_CORE_LABEL_SECTIONS,
        "actionTypeVocabulary": action_type_vocabulary,
        "actionTypeNameToId": action_type_name_to_id,
        "notes": [
            "This is the first V1 implementation step: fine-grained action types and delay bins are derived in chronodivide-bot-sl.",
            "ActionTypeId is replay-local for now because the transformer does not yet build a run-global vocabulary before writing shards.",
            "Legacy coarse label sections are still preserved during the migration period.",
        ],
    }


def augment_dataset_with_label_layout_v1(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    legacy_label_sections = copy.deepcopy(dataset["schema"]["labelSections"])
    dataset["legacyLabelSections"] = legacy_label_sections
    dataset["legacyFlatLabelLength"] = int(dataset["schema"]["flatLabelLength"])

    if not samples:
        dataset["labelLayoutV1"] = {
            "version": LABEL_LAYOUT_V1_VERSION,
            "status": "empty",
            "scope": LABEL_LAYOUT_V1_ACTION_VOCAB_SCOPE,
            "mainTensorUsesV1Only": True,
            "delayBins": LABEL_LAYOUT_V1_DELAY_BINS,
            "missingInt": LABEL_LAYOUT_V1_MISSING_INT,
            "coreLabelSections": LABEL_LAYOUT_V1_CORE_LABEL_SECTIONS,
            "actionTypeVocabulary": [],
            "actionTypeNameToId": {},
        }
        dataset["schema"]["labelSections"] = build_v1_label_sections(legacy_label_sections)
        dataset["schema"]["flatLabelLength"] = compute_flat_length(dataset["schema"]["labelSections"])
        return

    label_layout_v1 = build_label_layout_v1_metadata(samples, dataset)
    label_layout_v1["status"] = "canonical_main_label_layout"
    label_layout_v1["mainTensorUsesV1Only"] = True
    dataset["labelLayoutV1"] = label_layout_v1

    for sample in samples:
        sample["legacyLabelTensors"] = copy.deepcopy(sample["labelTensors"])
        sample["labelTensors"] = build_canonical_label_tensors_v1(sample, dataset)

    dataset["schema"]["labelSections"] = build_v1_label_sections(legacy_label_sections)
    dataset["schema"]["flatLabelLength"] = compute_flat_length(dataset["schema"]["labelSections"])


def register_dataset_action_types_globally(dataset: dict[str, Any], run_state: TransformRunState) -> None:
    label_layout_v1 = dataset.get("labelLayoutV1")
    if not label_layout_v1:
        return

    local_vocabulary = copy.deepcopy(label_layout_v1.get("actionTypeVocabulary", []))
    local_name_to_id = dict(label_layout_v1.get("actionTypeNameToId", {}))
    label_layout_v1["localActionTypeVocabulary"] = local_vocabulary
    label_layout_v1["localActionTypeNameToId"] = local_name_to_id

    updated_vocabulary: list[dict[str, Any]] = []
    for entry in local_vocabulary:
        action_type_name = str(entry["name"])
        if action_type_name not in run_state.action_type_name_to_global_id:
            run_state.action_type_name_to_global_id[action_type_name] = len(run_state.action_type_name_to_global_id)
        global_id = run_state.action_type_name_to_global_id[action_type_name]
        run_state.action_type_counts[action_type_name] = run_state.action_type_counts.get(action_type_name, 0) + int(
            entry.get("count", 0)
        )

        updated_entry = copy.deepcopy(entry)
        updated_entry["localId"] = int(entry["id"])
        updated_entry["id"] = global_id
        updated_entry["globalId"] = global_id
        updated_vocabulary.append(updated_entry)

    for sample in dataset.get("samples", []):
        action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
        if not isinstance(action_type_name, str):
            continue
        global_id = run_state.action_type_name_to_global_id[action_type_name]
        sample["derivedLabelMetadata"]["globalActionTypeIdV1"] = global_id
        sample["labelTensors"]["actionTypeId"] = [global_id]

    label_layout_v1["scope"] = "run_global"
    label_layout_v1["actionTypeVocabulary"] = updated_vocabulary
    label_layout_v1["actionTypeNameToId"] = dict(run_state.action_type_name_to_global_id)
    label_layout_v1.setdefault("notes", []).append(
        "Saved actionTypeId values use run-global ids assigned in first-seen replay order across this transform run."
    )


def build_global_action_type_vocabulary(run_state: TransformRunState) -> list[dict[str, Any]]:
    ordered_names = sorted(run_state.action_type_name_to_global_id.items(), key=lambda item: item[1])
    return [
        {
            "id": action_type_id,
            "name": action_type_name,
            "count": run_state.action_type_counts.get(action_type_name, 0),
        }
        for action_type_name, action_type_id in ordered_names
    ]


def stable_keep_fraction(
    *,
    replay_path: Path,
    player_name: str,
    sample: dict[str, Any],
    sample_index: int,
    filter_seed: int,
) -> float:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    key = "|".join(
        [
            str(filter_seed),
            str(replay_path),
            player_name,
            str(sample.get("tick")),
            str(sample.get("playerId")),
            str(sample_index),
            str(legacy_label_tensors["rawActionId"][0]),
            str(sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64 - 1)


def get_sample_filter_bucket(sample: dict[str, Any], dataset: dict[str, Any]) -> tuple[str, float]:
    filter_config = dataset["filterConfig"]
    action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
    if isinstance(action_type_name, str):
        if action_type_name == "NoAction":
            return "no_action", float(filter_config["noActionKeepProb"])
        if action_type_name == "SelectUnits":
            return "select_units", float(filter_config["selectUnitsKeepProb"])
        if action_type_name.startswith("Order::"):
            parts = action_type_name.split("::")
            order_type_name = parts[1] if len(parts) > 1 else None
            if order_type_name in {"Move", "ForceMove"}:
                return "order_move", float(filter_config["moveOrderKeepProb"])
            if order_type_name == "Gather":
                return "order_gather", float(filter_config["gatherOrderKeepProb"])
            if order_type_name in {"Attack", "ForceAttack", "AttackMove"}:
                return "order_attack", float(filter_config["attackOrderKeepProb"])

    action_family = get_action_family_name(sample, dataset)

    if action_family == "no_action":
        return "no_action", float(filter_config["noActionKeepProb"])

    if action_family == "select_units":
        return "select_units", float(filter_config["selectUnitsKeepProb"])

    if action_family == "order_units":
        order_type_name = get_order_type_name(sample, dataset)
        if order_type_name in {"Move", "ForceMove"}:
            return "order_move", float(filter_config["moveOrderKeepProb"])
        if order_type_name == "Gather":
            return "order_gather", float(filter_config["gatherOrderKeepProb"])
        if order_type_name in {"Attack", "ForceAttack", "AttackMove"}:
            return "order_attack", float(filter_config["attackOrderKeepProb"])

    return "keep_all", 1.0


def build_player_action_counts(samples: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    raw_counts: dict[int, int] = {}
    family_counts: dict[str, int] = {}
    action_type_counts: dict[str, int] = {}

    for sample in samples:
        legacy_label_tensors = get_legacy_label_tensors(sample)
        raw_action_id = int(legacy_label_tensors["rawActionId"][0])
        raw_counts[raw_action_id] = raw_counts.get(raw_action_id, 0) + 1

        family_name = get_action_family_name(sample, dataset)
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
        action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
        if isinstance(action_type_name, str):
            action_type_counts[action_type_name] = action_type_counts.get(action_type_name, 0) + 1

    return {
        "rawActionCounts": [
            {"rawActionId": raw_action_id, "count": count}
            for raw_action_id, count in sorted(raw_counts.items(), key=lambda item: item[0])
        ],
        "actionFamilyCounts": [
            {"actionFamily": family_name, "count": count}
            for family_name, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "actionTypeCountsV1": [
            {"actionTypeName": action_type_name, "count": count}
            for action_type_name, count in sorted(action_type_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


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


def infer_nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)

    first_shape = infer_nested_shape(value[0])
    for item in value[1:]:
        item_shape = infer_nested_shape(item)
        if item_shape != first_shape:
            raise ValueError(f"Inconsistent nested shape: expected {first_shape}, got {item_shape}.")
    return (len(value), *first_shape)


def flatten_nested_values(value: Any) -> list[float | int]:
    if not isinstance(value, list):
        return [value]

    flattened: list[float | int] = []
    for item in value:
        flattened.extend(flatten_nested_values(item))
    return flattened


def flatten_js_nested_numbers(value: Any) -> list[float | int]:
    if not isinstance(value, list):
        return [value]

    flattened: list[float | int] = []
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            for item in reversed(current):
                stack.append(item)
            continue
        flattened.append(current)
    flattened.reverse()
    return flattened


def flatten_section_value(value: Any, shape: list[int]) -> list[float | int]:
    if len(shape) <= 1:
        return flatten_nested_values(value)
    return flatten_js_nested_numbers(value)


def values_match(expected: list[Any], actual: list[Any], tolerance: float = 1e-6) -> bool:
    if len(expected) != len(actual):
        return False

    for left, right in zip(expected, actual, strict=True):
        if isinstance(left, bool) or isinstance(right, bool):
            if bool(left) != bool(right):
                return False
            continue

        if isinstance(left, int) and isinstance(right, int):
            if left != right:
                return False
            continue

        try:
            if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
                return False
        except (TypeError, ValueError):
            if left != right:
                return False
    return True


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


def validate_section_shapes(
    samples: list[dict[str, Any]],
    schema_sections: list[dict[str, Any]],
    tensor_key: str,
    replay_name: str,
    player_name: str,
) -> None:
    for sample_index, sample in enumerate(samples):
        tensor_sections = sample.get(tensor_key, {})
        for section in schema_sections:
            section_name = section["name"]
            if section_name not in tensor_sections:
                raise ValueError(
                    f"{replay_name}/{player_name} sample {sample_index} is missing {tensor_key}.{section_name}."
                )
            observed_shape = infer_nested_shape(tensor_sections[section_name])
            expected_shape = tuple(int(dimension) for dimension in section["shape"])
            if observed_shape != expected_shape:
                raise ValueError(
                    f"{replay_name}/{player_name} sample {sample_index} has shape {observed_shape} for "
                    f"{tensor_key}.{section_name}, expected {expected_shape}."
                )


def validate_flat_matches_sections(samples: list[dict[str, Any]], schema: dict[str, Any], replay_name: str, player_name: str) -> None:
    feature_sections = schema["featureSections"]
    label_sections = schema["labelSections"]

    for sample_index, sample in enumerate(samples):
        flattened_feature: list[Any] = []
        for section in feature_sections:
            flattened_feature.extend(flatten_section_value(sample["featureTensors"][section["name"]], section["shape"]))
        if not values_match(flattened_feature, sample["flatFeatureTensor"]):
            raise ValueError(
                f"{replay_name}/{player_name} sample {sample_index} flat feature tensor does not match structured sections."
            )

        flattened_label: list[Any] = []
        for section in label_sections:
            flattened_label.extend(flatten_section_value(sample["labelTensors"][section["name"]], section["shape"]))
        if not values_match(flattened_label, sample["flatLabelTensor"]):
            raise ValueError(
                f"{replay_name}/{player_name} sample {sample_index} flat label tensor does not match structured sections."
            )


def build_tensors(samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows = [sample["flatFeatureTensor"] for sample in samples]
    label_rows = [sample["flatLabelTensor"] for sample in samples]
    features = torch.tensor(feature_rows, dtype=feature_dtype())
    labels = torch.tensor(label_rows, dtype=label_dtype())
    return features, labels


def build_structured_section_tensors(
    samples: list[dict[str, Any]],
    schema_sections: list[dict[str, Any]],
    tensor_key: str,
) -> dict[str, torch.Tensor]:
    section_tensors: dict[str, torch.Tensor] = {}
    for section in schema_sections:
        section_name = section["name"]
        rows = [sample[tensor_key][section_name] for sample in samples]
        section_tensors[section_name] = torch.tensor(rows, dtype=schema_dtype_to_torch(str(section["dtype"])))
    return section_tensors


def build_sample_context_tensors(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    ticks = torch.tensor([sample["tick"] for sample in samples], dtype=torch.int32)
    player_ids = torch.tensor([sample["playerId"] for sample in samples], dtype=torch.int32)
    return {
        "ticks": ticks,
        "playerIds": player_ids,
    }


def build_section_offsets(schema_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    cursor = 0
    for section in schema_sections:
        length = 1
        for dimension in section["shape"]:
            length *= int(dimension)
        offsets.append(
            {
                "name": section["name"],
                "offset": cursor,
                "length": length,
                "shape": section["shape"],
                "dtype": section["dtype"],
            }
        )
        cursor += length
    return offsets


def rebuild_flat_tensor(sample: dict[str, Any], schema_sections: list[dict[str, Any]], tensor_key: str) -> list[float | int]:
    flattened: list[float | int] = []
    for section in schema_sections:
        flattened.extend(flatten_section_value(sample[tensor_key][section["name"]], section["shape"]))
    return flattened


def rewrite_temporal_context(samples: list[dict[str, Any]]) -> None:
    previous_tick: int | None = None
    previous_raw_action_id = -1
    previous_action_family_id = -1
    previous_queue = -1

    for sample in samples:
        legacy_label_tensors = get_legacy_label_tensors(sample)
        current_tick = int(sample["tick"])
        delay_from_previous = -1 if previous_tick is None else current_tick - previous_tick
        sample["featureTensors"]["lastActionContext"] = [
            delay_from_previous,
            previous_raw_action_id,
            previous_action_family_id,
            previous_queue,
        ]

        previous_tick = current_tick
        previous_raw_action_id = int(legacy_label_tensors["rawActionId"][0])
        previous_action_family_id = int(legacy_label_tensors["actionFamilyId"][0])
        previous_queue = int(legacy_label_tensors["queue"][0])

    for index, sample in enumerate(samples):
        legacy_label_tensors = get_legacy_label_tensors(sample)
        next_tick = int(samples[index + 1]["tick"]) if index + 1 < len(samples) else None
        delay_to_next = -1 if next_tick is None else next_tick - int(sample["tick"])
        legacy_label_tensors["delayToNextAction"] = [delay_to_next]
        sample["labelTensors"]["delayBin"] = [
            LABEL_LAYOUT_V1_MISSING_INT
            if delay_to_next < 0
            else min(delay_to_next, LABEL_LAYOUT_V1_DELAY_BINS - 1)
        ]


def build_filter_config(config: TransformConfig) -> dict[str, Any]:
    return {
        "profile": config.action_filter_profile,
        "filterSeed": config.filter_seed,
        "noActionKeepProb": config.no_action_keep_prob,
        "selectUnitsKeepProb": config.select_units_keep_prob,
        "moveOrderKeepProb": config.move_order_keep_prob,
        "gatherOrderKeepProb": config.gather_order_keep_prob,
        "attackOrderKeepProb": config.attack_order_keep_prob,
        "notes": [
            "This filter is applied in chronodivide-bot-sl after py-chronodivide extracts the action-aligned sample stream.",
            "Temporal fields are rewritten on the final kept stream so lastActionContext and delayToNextAction stay consistent.",
            "The default mas profile downsamples common RA2 action patterns analogously to mini-AlphaStar's replay filtering.",
        ],
    }


def apply_action_filter_profile(
    config: TransformConfig,
    dataset: dict[str, Any],
    replay_path: Path,
    player_name: str,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_count = len(samples)
    if config.action_filter_profile == "none":
        kept_samples = list(samples)
        if kept_samples:
            rewrite_temporal_context(kept_samples)
            for sample in kept_samples:
                sample["flatFeatureTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["featureSections"], "featureTensors")
                sample["flatLabelTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["labelSections"], "labelTensors")
        return kept_samples, {
            "profile": "none",
            "sourceSampleCount": source_count,
            "keptSampleCount": len(kept_samples),
            "droppedSampleCount": source_count - len(kept_samples),
            "bucketCounts": [],
        }

    bucket_counts: dict[str, dict[str, Any]] = {}
    kept_samples: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        bucket_name, keep_prob = get_sample_filter_bucket(sample, dataset)
        bucket_record = bucket_counts.setdefault(
            bucket_name,
            {
                "bucket": bucket_name,
                "keepProb": keep_prob,
                "before": 0,
                "kept": 0,
                "dropped": 0,
            },
        )
        bucket_record["before"] += 1

        keep = keep_prob >= 1.0
        if not keep:
            keep = stable_keep_fraction(
                replay_path=replay_path,
                player_name=player_name,
                sample=sample,
                sample_index=sample_index,
                filter_seed=config.filter_seed,
            ) < keep_prob

        if keep:
            bucket_record["kept"] += 1
            kept_samples.append(sample)
        else:
            bucket_record["dropped"] += 1

    if kept_samples:
        rewrite_temporal_context(kept_samples)
        for sample in kept_samples:
            sample["flatFeatureTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["featureSections"], "featureTensors")
            sample["flatLabelTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["labelSections"], "labelTensors")

    return kept_samples, {
        "profile": config.action_filter_profile,
        "sourceSampleCount": source_count,
        "keptSampleCount": len(kept_samples),
        "droppedSampleCount": source_count - len(kept_samples),
        "bucketCounts": sorted(bucket_counts.values(), key=lambda item: item["bucket"]),
    }


def player_output_stem(replay_path: Path, player_name: str) -> str:
    return f"{replay_path.stem}__{sanitize_filename(player_name)}"


def write_player_shard(
    config: TransformConfig,
    replay_path: Path,
    dataset: dict[str, Any],
    player_name: str,
    samples: list[dict[str, Any]],
    filter_stats: dict[str, Any],
    player_counts: dict[str, Any],
) -> dict[str, Any]:
    output_stem = player_output_stem(replay_path, player_name)
    tensor_path = config.output_dir / f"{output_stem}.pt"
    structured_tensor_path = config.output_dir / f"{output_stem}.sections.pt"
    metadata_path = config.output_dir / f"{output_stem}.meta.json"

    if tensor_path.exists() and structured_tensor_path.exists() and metadata_path.exists() and not config.overwrite:
        return {
            "status": "skipped",
            "replay": str(replay_path),
            "playerName": player_name,
            "tensorPath": str(tensor_path),
            "structuredTensorPath": str(structured_tensor_path),
            "metadataPath": str(metadata_path),
            "reason": "existing shard",
        }

    validate_flat_tensor_lengths(samples, dataset["schema"], replay_path.name, player_name)
    validate_section_shapes(samples, dataset["schema"]["featureSections"], "featureTensors", replay_path.name, player_name)
    validate_section_shapes(samples, dataset["schema"]["labelSections"], "labelTensors", replay_path.name, player_name)
    validate_flat_matches_sections(samples, dataset["schema"], replay_path.name, player_name)
    features, labels = build_tensors(samples)
    feature_section_tensors = build_structured_section_tensors(samples, dataset["schema"]["featureSections"], "featureTensors")
    label_section_tensors = build_structured_section_tensors(samples, dataset["schema"]["labelSections"], "labelTensors")
    sample_context_tensors = build_sample_context_tensors(samples)

    ensure_parent(tensor_path)
    torch.save((features, labels), tensor_path)
    torch.save(
        {
            "featureTensors": feature_section_tensors,
            "labelTensors": label_section_tensors,
            "sampleContext": sample_context_tensors,
        },
        structured_tensor_path,
    )

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
        "legacyLabelSchema": {
            "labelSections": dataset.get("legacyLabelSections", []),
            "flatLabelLength": dataset.get("legacyFlatLabelLength"),
        },
        "featureSectionOffsets": build_section_offsets(dataset["schema"]["featureSections"]),
        "labelSectionOffsets": build_section_offsets(dataset["schema"]["labelSections"]),
        "structuredFeatureShapes": {
            name: list(tensor.shape) for name, tensor in feature_section_tensors.items()
        },
        "structuredLabelShapes": {
            name: list(tensor.shape) for name, tensor in label_section_tensors.items()
        },
        "sampleContextShapes": {
            name: list(tensor.shape) for name, tensor in sample_context_tensors.items()
        },
        "sourceOptions": dataset["options"],
        "sourceCounts": dataset["counts"],
        "playerCounts": player_counts,
        "actionFilter": filter_stats,
        "filterConfig": dataset["filterConfig"],
        "labelLayoutV1": dataset.get("labelLayoutV1"),
        "tensorPath": str(tensor_path),
        "structuredTensorPath": str(structured_tensor_path),
    }
    ensure_parent(metadata_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "status": "saved",
        "replay": str(replay_path),
        "playerName": player_name,
        "sampleCount": len(samples),
        "sourceSampleCount": int(filter_stats["sourceSampleCount"]),
        "droppedSampleCount": int(filter_stats["droppedSampleCount"]),
        "featureShape": list(features.shape),
        "labelShape": list(labels.shape),
        "tensorPath": str(tensor_path),
        "structuredTensorPath": str(structured_tensor_path),
        "metadataPath": str(metadata_path),
    }


def transform_single_replay(config: TransformConfig, replay_path: Path, run_state: TransformRunState) -> list[dict[str, Any]]:
    dataset = run_py_chronodivide_extract(config, replay_path)
    augment_dataset_with_label_layout_v1(dataset)
    register_dataset_action_types_globally(dataset, run_state)
    dataset["filterConfig"] = build_filter_config(config)
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
        filtered_samples, filter_stats = apply_action_filter_profile(config, dataset, replay_path, player_name, player_samples)
        if not filtered_samples:
            shard_results.append(
                {
                    "status": "empty",
                    "replay": str(replay_path),
                    "playerName": player_name,
                    "reason": "all samples were dropped by action filtering",
                    "sourceSampleCount": int(filter_stats["sourceSampleCount"]),
                    "droppedSampleCount": int(filter_stats["droppedSampleCount"]),
                }
            )
            continue

        player_counts = build_player_action_counts(filtered_samples, dataset)
        shard_results.append(
            write_player_shard(
                config,
                replay_path,
                dataset,
                player_name,
                filtered_samples,
                filter_stats,
                player_counts,
            )
        )
    return shard_results


def write_manifest(
    config: TransformConfig,
    replay_paths: list[Path],
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    run_state: TransformRunState,
) -> Path:
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
        "labelLayoutV1GlobalActionVocabulary": build_global_action_type_vocabulary(run_state),
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
    run_state = TransformRunState()

    replay_paths = list_replays(config)
    if not replay_paths:
        raise FileNotFoundError(
            f"No replay files matching {config.replay_glob!r} were found in {config.replay_dir}."
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for replay_path in iter_progress(replay_paths, "Transform replays"):
        try:
            results.extend(transform_single_replay(config, replay_path, run_state))
        except Exception as exc:  # pragma: no cover - failure path exercised by real runs
            error_record = {
                "replay": str(replay_path),
                "errorType": exc.__class__.__name__,
                "error": str(exc),
            }
            errors.append(error_record)
            if config.fail_fast:
                raise

    manifest_path = write_manifest(config, replay_paths, results, errors, run_state)
    print(f"Processed {len(replay_paths)} replay(s).")
    print(f"Saved {sum(1 for result in results if result['status'] == 'saved')} shard(s).")
    print(f"Manifest: {manifest_path}")

    if errors:
        print(f"Encountered {len(errors)} replay error(s). See manifest for details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
