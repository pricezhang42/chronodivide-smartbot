#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transform Chronodivide replays into replay-level (features, labels) torch shards.

This is the SL-transformer-side bridge that mirrors the role of
mini-AlphaStar's `transform_replay_data.py`, while delegating replay parsing,
feature extraction, and action-label decoding to the reusable `py-chronodivide`
package.

Current behavior:
- iterate replay files
- call `py-chronodivide/extract_sl_tensors.mjs`
- group samples by player perspective
- derive static-dict-backed RA2 SL V1 action-type metadata
- apply mAS-style action filtering/downsampling on the action-aligned sample stream
- save replay-player `.pt` shards as `(features, labels)`
- save replay-player `.sections.pt` sidecars with structured feature/label tensors
- save replay-player `.training.pt` sidecars with model-ready derived targets and masks
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
from action_dict import (
    ACTION_INFO_MASK as STATIC_ACTION_INFO_MASK,
    ACTION_TYPE_ID_TO_NAME as STATIC_ACTION_TYPE_ID_TO_NAME,
    ACTION_TYPE_NAME_TO_ID as STATIC_ACTION_TYPE_NAME_TO_ID,
    STATIC_ACTION_DICT_VERSION,
    UNKNOWN_ACTION_TYPE_NAME,
    build_observed_action_type_name,
    canonicalize_action_type_name,
)

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
    return build_observed_action_type_name(
        raw_action_name=raw_action_name,
        raw_action_id=raw_action_id,
        order_type_name=get_order_type_name(sample, dataset),
        target_mode_name=get_target_mode_name(sample, dataset),
        queue_update_type_name=get_queue_update_type_name(sample, dataset),
        item_name=get_shared_name_from_token(dataset, int(legacy_label_tensors["itemNameToken"][0])),
        building_name=get_shared_name_from_token(dataset, int(legacy_label_tensors["buildingNameToken"][0])),
        super_weapon_name=get_super_weapon_type_name(sample, dataset),
    )


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


def build_v1_feature_sections(legacy_feature_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_sections = copy.deepcopy(legacy_feature_sections)
    for section in feature_sections:
        if section["name"] == "lastActionContext":
            section["shape"] = [3]
            break
    else:
        raise KeyError("Feature schema is missing lastActionContext.")
    return feature_sections


def augment_dataset_with_feature_context_v1(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    legacy_feature_sections = copy.deepcopy(dataset["schema"]["featureSections"])
    dataset["legacyFeatureSections"] = legacy_feature_sections
    dataset["legacyFlatFeatureLength"] = int(dataset["schema"]["flatFeatureLength"])

    dataset["featureContextV1"] = {
        "version": FEATURE_CONTEXT_V1_VERSION,
        "mainTensorUsesV1Only": True,
        "lastActionContextShape": [3],
        "lastActionContextFields": [
            "delayFromPreviousAction",
            "lastActionTypeIdV1",
            "lastQueue",
        ],
        "notes": [
            "This mirrors mini-AlphaStar more closely than the legacy four-value context.",
            "Legacy feature context remains available in metadata during the migration period.",
        ],
    }

    for sample in samples:
        sample["legacyFeatureTensors"] = copy.deepcopy(sample["featureTensors"])
        previous_delay = LABEL_LAYOUT_V1_MISSING_INT
        legacy_last_action_context = sample["legacyFeatureTensors"].get("lastActionContext", [])
        if legacy_last_action_context:
            previous_delay = int(legacy_last_action_context[0])
        sample["featureTensors"]["lastActionContext"] = [
            previous_delay,
            LABEL_LAYOUT_V1_MISSING_INT,
            LABEL_LAYOUT_V1_MISSING_INT,
        ]

    dataset["schema"]["featureSections"] = build_v1_feature_sections(legacy_feature_sections)
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])


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
    unseen_action_type_counts: dict[str, int] = {}

    for sample in samples:
        observed_action_type_name = action_type_name_v1(sample, dataset)
        action_type_name = canonicalize_action_type_name(observed_action_type_name)
        action_type_info = STATIC_ACTION_INFO_MASK[STATIC_ACTION_TYPE_NAME_TO_ID[action_type_name]]
        sample_mask = {
            "usesQueue": bool(action_type_info["usesQueue"]),
            "usesUnits": bool(action_type_info["usesUnits"]),
            "usesTargetEntity": bool(action_type_info["usesTargetEntity"]),
            "usesTargetLocation": bool(action_type_info["usesTargetLocation"]),
            "usesTargetLocation2": bool(action_type_info["usesTargetLocation2"]),
            "usesQuantity": bool(action_type_info["usesQuantity"]),
        }
        sample.setdefault("derivedLabelMetadata", {})["observedActionTypeNameV1"] = observed_action_type_name
        sample["derivedLabelMetadata"]["actionTypeNameV1"] = action_type_name
        sample["derivedLabelMetadata"]["semanticMaskV1"] = sample_mask
        sample["derivedLabelMetadata"]["actionTypeWasKnownV1"] = observed_action_type_name == action_type_name
        action_type_counts[action_type_name] = action_type_counts.get(action_type_name, 0) + 1
        if action_type_name == UNKNOWN_ACTION_TYPE_NAME and observed_action_type_name != UNKNOWN_ACTION_TYPE_NAME:
            unseen_action_type_counts[observed_action_type_name] = unseen_action_type_counts.get(observed_action_type_name, 0) + 1

    action_type_vocabulary = []
    for action_type_id, name in STATIC_ACTION_TYPE_ID_TO_NAME.items():
        action_type_info = STATIC_ACTION_INFO_MASK[action_type_id]
        action_type_vocabulary.append(
            {
                "id": action_type_id,
                "name": name,
                "count": action_type_counts.get(name, 0),
                "semanticMask": {
                    "usesQueue": bool(action_type_info["usesQueue"]),
                    "usesUnits": bool(action_type_info["usesUnits"]),
                    "usesTargetEntity": bool(action_type_info["usesTargetEntity"]),
                    "usesTargetLocation": bool(action_type_info["usesTargetLocation"]),
                    "usesTargetLocation2": bool(action_type_info["usesTargetLocation2"]),
                    "usesQuantity": bool(action_type_info["usesQuantity"]),
                },
                "family": action_type_info["family"],
                "observedInDataset": bool(action_type_counts.get(name, 0)),
            }
        )

    return {
        "version": LABEL_LAYOUT_V1_VERSION,
        "status": "static_action_dict",
        "scope": "static_v1",
        "mainTensorUsesV1Only": False,
        "delayBins": LABEL_LAYOUT_V1_DELAY_BINS,
        "missingInt": LABEL_LAYOUT_V1_MISSING_INT,
        "quantityPolicy": LABEL_LAYOUT_V1_QUANTITY_POLICY,
        "supervisionPolicy": LABEL_LAYOUT_V1_SUPERVISION_POLICY,
        "coreLabelSections": LABEL_LAYOUT_V1_CORE_LABEL_SECTIONS,
        "staticActionDictVersion": STATIC_ACTION_DICT_VERSION,
        "actionTypeVocabulary": action_type_vocabulary,
        "actionTypeNameToId": dict(STATIC_ACTION_TYPE_NAME_TO_ID),
        "unseenObservedActionTypes": [
            {"name": name, "count": count}
            for name, count in sorted(unseen_action_type_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "notes": [
            "ActionTypeId is assigned from the static chronodivide-bot-sl action dict, not replay order.",
            "Unknown concrete queue items, buildings, or super-weapon names fall back to explicit <unk> action buckets.",
            "Quantity stays raw integer-valued in canonical V1; bucketing is explicitly deferred.",
            "Training supervision combines semantic masks from action type with replay-time resolution/validity masks.",
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
            "scope": "static_v1",
            "mainTensorUsesV1Only": True,
            "delayBins": LABEL_LAYOUT_V1_DELAY_BINS,
            "missingInt": LABEL_LAYOUT_V1_MISSING_INT,
            "quantityPolicy": LABEL_LAYOUT_V1_QUANTITY_POLICY,
            "supervisionPolicy": LABEL_LAYOUT_V1_SUPERVISION_POLICY,
            "coreLabelSections": LABEL_LAYOUT_V1_CORE_LABEL_SECTIONS,
            "staticActionDictVersion": STATIC_ACTION_DICT_VERSION,
            "actionTypeVocabulary": [
                {
                    "id": action_type_id,
                    "name": action_type_name,
                    "count": 0,
                    "semanticMask": {
                        "usesQueue": bool(action_type_info["usesQueue"]),
                        "usesUnits": bool(action_type_info["usesUnits"]),
                        "usesTargetEntity": bool(action_type_info["usesTargetEntity"]),
                        "usesTargetLocation": bool(action_type_info["usesTargetLocation"]),
                        "usesTargetLocation2": bool(action_type_info["usesTargetLocation2"]),
                        "usesQuantity": bool(action_type_info["usesQuantity"]),
                    },
                    "family": action_type_info["family"],
                    "observedInDataset": False,
                }
                for action_type_id, action_type_name in STATIC_ACTION_TYPE_ID_TO_NAME.items()
                for action_type_info in [STATIC_ACTION_INFO_MASK[action_type_id]]
            ],
            "actionTypeNameToId": dict(STATIC_ACTION_TYPE_NAME_TO_ID),
            "unseenObservedActionTypes": [],
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

    for sample in dataset.get("samples", []):
        action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
        if not isinstance(action_type_name, str):
            continue
        global_id = STATIC_ACTION_TYPE_NAME_TO_ID[action_type_name]
        run_state.action_type_counts[action_type_name] = run_state.action_type_counts.get(action_type_name, 0) + 1
        observed_action_type_name = sample.get("derivedLabelMetadata", {}).get("observedActionTypeNameV1")
        if (
            isinstance(observed_action_type_name, str)
            and observed_action_type_name != action_type_name
            and action_type_name == UNKNOWN_ACTION_TYPE_NAME
        ):
            run_state.unseen_action_type_counts[observed_action_type_name] = (
                run_state.unseen_action_type_counts.get(observed_action_type_name, 0) + 1
            )
        sample["derivedLabelMetadata"]["globalActionTypeIdV1"] = global_id
        sample["labelTensors"]["actionTypeId"] = [global_id]

    label_layout_v1["scope"] = "static_v1"
    label_layout_v1["actionTypeNameToId"] = dict(STATIC_ACTION_TYPE_NAME_TO_ID)
    label_layout_v1.setdefault("notes", []).append(
        "Saved actionTypeId values use the static chronodivide-bot-sl action dict."
    )


def build_global_action_type_vocabulary(run_state: TransformRunState) -> list[dict[str, Any]]:
    return [
        {
            "id": action_type_id,
            "name": action_type_name,
            "count": run_state.action_type_counts.get(action_type_name, 0),
            "family": STATIC_ACTION_INFO_MASK[action_type_id]["family"],
        }
        for action_type_id, action_type_name in STATIC_ACTION_TYPE_ID_TO_NAME.items()
    ]


def get_observation_scalar_index(schema: dict[str, Any], feature_name: str) -> int:
    scalar_feature_names = schema.get("observation", {}).get("scalarFeatureNames", [])
    if feature_name not in scalar_feature_names:
        raise KeyError(f"Observation scalar feature not found in schema: {feature_name}")
    return int(scalar_feature_names.index(feature_name))


def build_training_target_sections(
    *,
    action_vocab_size: int,
    delay_bins: int,
    max_selected_units: int,
    max_entities: int,
    spatial_size: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "targetSections": [
            {"name": "actionTypeOneHot", "shape": [action_vocab_size], "dtype": "int32"},
            {"name": "delayOneHot", "shape": [delay_bins], "dtype": "int32"},
            {"name": "queueOneHot", "shape": [2], "dtype": "int32"},
            {"name": "unitsOneHot", "shape": [max_selected_units, max_entities], "dtype": "int32"},
            {"name": "targetEntityOneHot", "shape": [max_entities], "dtype": "int32"},
            {"name": "targetLocationOneHot", "shape": [spatial_size, spatial_size], "dtype": "int32"},
            {"name": "targetLocation2OneHot", "shape": [spatial_size, spatial_size], "dtype": "int32"},
            {"name": "quantityValue", "shape": [1], "dtype": "int32"},
        ],
        "maskSections": [
            {"name": "actionTypeLossMask", "shape": [1], "dtype": "int32"},
            {"name": "delayLossMask", "shape": [1], "dtype": "int32"},
            {"name": "queueSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "unitsSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetEntitySemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocationSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocation2SemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "quantitySemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "queueLossMask", "shape": [1], "dtype": "int32"},
            {"name": "unitsSequenceMask", "shape": [max_selected_units], "dtype": "int32"},
            {"name": "unitsResolvedMask", "shape": [max_selected_units], "dtype": "int32"},
            {"name": "unitsLossMask", "shape": [max_selected_units], "dtype": "int32"},
            {"name": "targetEntityResolvedMask", "shape": [1], "dtype": "int32"},
            {"name": "targetEntityLossMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocationValidMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocationLossMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocation2ValidMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocation2LossMask", "shape": [1], "dtype": "int32"},
            {"name": "quantityLossMask", "shape": [1], "dtype": "int32"},
        ],
    }


def build_action_type_semantic_lookup(label_layout_v1: dict[str, Any]) -> dict[int, dict[str, bool]]:
    lookup: dict[int, dict[str, bool]] = {}
    for entry in label_layout_v1.get("actionTypeVocabulary", []):
        action_type_id = int(entry["id"])
        semantic_mask = entry.get("semanticMask", {})
        lookup[action_type_id] = {
            "usesQueue": bool(semantic_mask.get("usesQueue", False)),
            "usesUnits": bool(semantic_mask.get("usesUnits", False)),
            "usesTargetEntity": bool(semantic_mask.get("usesTargetEntity", False)),
            "usesTargetLocation": bool(semantic_mask.get("usesTargetLocation", False)),
            "usesTargetLocation2": bool(semantic_mask.get("usesTargetLocation2", False)),
            "usesQuantity": bool(semantic_mask.get("usesQuantity", False)),
        }
    return lookup


def build_semantic_mask_tensor(
    action_type_ids: torch.Tensor,
    semantic_lookup: dict[int, dict[str, bool]],
    semantic_key: str,
) -> torch.Tensor:
    rows = [
        [1 if semantic_lookup.get(int(action_type_id), {}).get(semantic_key, False) else 0]
        for action_type_id in action_type_ids.tolist()
    ]
    return torch.tensor(rows, dtype=torch.int32)


def build_one_hot_matrix(indices: torch.Tensor, size: int) -> torch.Tensor:
    one_hot = torch.zeros((indices.shape[0], size), dtype=torch.int32)
    valid_mask = (indices >= 0) & (indices < size)
    if torch.any(valid_mask):
        rows = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        one_hot[rows, indices[valid_mask].to(torch.long)] = 1
    return one_hot


def build_one_hot_sequence(indices: torch.Tensor, size: int) -> torch.Tensor:
    batch_size, sequence_length = indices.shape
    one_hot = torch.zeros((batch_size, sequence_length, size), dtype=torch.int32)
    valid_mask = (indices >= 0) & (indices < size)
    if torch.any(valid_mask):
        positions = torch.nonzero(valid_mask, as_tuple=False)
        one_hot[
            positions[:, 0],
            positions[:, 1],
            indices[valid_mask].to(torch.long),
        ] = 1
    return one_hot


def tile_to_grid_indices(
    tile_locations: torch.Tensor,
    map_widths: torch.Tensor,
    map_heights: torch.Tensor,
    spatial_size: int,
) -> torch.Tensor:
    x_values = tile_locations[:, 0].to(torch.float32)
    y_values = tile_locations[:, 1].to(torch.float32)
    max_x = torch.clamp(map_widths.to(torch.float32) - 1.0, min=1.0)
    max_y = torch.clamp(map_heights.to(torch.float32) - 1.0, min=1.0)
    normalized_x = torch.clamp(x_values / max_x, min=0.0, max=1.0)
    normalized_y = torch.clamp(y_values / max_y, min=0.0, max=1.0)
    grid_x = torch.clamp(torch.floor(normalized_x * spatial_size).to(torch.int64), min=0, max=spatial_size - 1)
    grid_y = torch.clamp(torch.floor(normalized_y * spatial_size).to(torch.int64), min=0, max=spatial_size - 1)
    return torch.stack((grid_x, grid_y), dim=1)


def build_spatial_one_hot(
    tile_locations: torch.Tensor,
    valid_mask: torch.Tensor,
    map_widths: torch.Tensor,
    map_heights: torch.Tensor,
    spatial_size: int,
) -> torch.Tensor:
    batch_size = tile_locations.shape[0]
    one_hot = torch.zeros((batch_size, spatial_size, spatial_size), dtype=torch.int32)
    effective_valid_mask = (
        (valid_mask.squeeze(1) > 0)
        & (tile_locations[:, 0] >= 0)
        & (tile_locations[:, 1] >= 0)
    )
    if torch.any(effective_valid_mask):
        grid_locations = tile_to_grid_indices(
            tile_locations[effective_valid_mask],
            map_widths[effective_valid_mask],
            map_heights[effective_valid_mask],
            spatial_size,
        )
        rows = torch.nonzero(effective_valid_mask, as_tuple=False).squeeze(1)
        one_hot[rows, grid_locations[:, 1], grid_locations[:, 0]] = 1
    return one_hot


def validate_tensor_section_shapes(
    tensor_sections: dict[str, torch.Tensor],
    schema_sections: list[dict[str, Any]],
    artifact_name: str,
    replay_name: str,
    player_name: str,
    sample_count: int,
) -> None:
    for section in schema_sections:
        section_name = str(section["name"])
        if section_name not in tensor_sections:
            raise ValueError(f"{replay_name}/{player_name} is missing {artifact_name}.{section_name}.")
        expected_shape = (sample_count, *tuple(int(dimension) for dimension in section["shape"]))
        observed_shape = tuple(int(dimension) for dimension in tensor_sections[section_name].shape)
        if observed_shape != expected_shape:
            raise ValueError(
                f"{replay_name}/{player_name} has shape {observed_shape} for {artifact_name}.{section_name}, "
                f"expected {expected_shape}."
            )


def build_training_targets_v1(
    feature_section_tensors: dict[str, torch.Tensor],
    label_section_tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    global_action_vocabulary: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    schema = metadata["schema"]
    label_layout_v1 = metadata.get("labelLayoutV1", {})
    semantic_lookup = build_action_type_semantic_lookup(label_layout_v1)
    action_vocab_size = len(global_action_vocabulary)
    delay_bins = int(label_layout_v1.get("delayBins", LABEL_LAYOUT_V1_DELAY_BINS))
    spatial_size = int(schema["observation"]["spatialSize"])
    max_entities = int(schema["observation"]["maxEntities"])
    max_selected_units = int(label_section_tensors["unitsIndices"].shape[1])
    sample_count = int(label_section_tensors["actionTypeId"].shape[0])
    scalar_tensor = feature_section_tensors["scalar"]
    map_width_index = get_observation_scalar_index(schema, "map_width")
    map_height_index = get_observation_scalar_index(schema, "map_height")
    map_widths = scalar_tensor[:, map_width_index]
    map_heights = scalar_tensor[:, map_height_index]

    action_type_ids = label_section_tensors["actionTypeId"].squeeze(1).to(torch.int64)
    delay_bins_tensor = label_section_tensors["delayBin"].squeeze(1).to(torch.int64)
    queue_tensor = label_section_tensors["queue"].squeeze(1).to(torch.int64)
    units_indices = label_section_tensors["unitsIndices"].to(torch.int64)
    units_sequence_mask = label_section_tensors["unitsMask"].to(torch.int32)
    units_resolved_mask = label_section_tensors["unitsResolvedMask"].to(torch.int32)
    target_entity_indices = label_section_tensors["targetEntityIndex"].squeeze(1).to(torch.int64)
    target_entity_resolved = label_section_tensors["targetEntityResolved"].to(torch.int32)
    target_location = label_section_tensors["targetLocation"].to(torch.int64)
    target_location_valid = label_section_tensors["targetLocationValid"].to(torch.int32)
    target_location_2 = label_section_tensors["targetLocation2"].to(torch.int64)
    target_location_2_valid = label_section_tensors["targetLocation2Valid"].to(torch.int32)
    quantity_value = label_section_tensors["quantity"].to(torch.int32)

    action_type_one_hot = build_one_hot_matrix(action_type_ids, action_vocab_size)
    delay_one_hot = build_one_hot_matrix(delay_bins_tensor, delay_bins)
    queue_one_hot = build_one_hot_matrix(queue_tensor, 2)
    units_one_hot = build_one_hot_sequence(units_indices, max_entities)
    target_entity_one_hot = build_one_hot_matrix(target_entity_indices, max_entities)
    target_location_one_hot = build_spatial_one_hot(
        target_location,
        target_location_valid,
        map_widths,
        map_heights,
        spatial_size,
    )
    target_location_2_one_hot = build_spatial_one_hot(
        target_location_2,
        target_location_2_valid,
        map_widths,
        map_heights,
        spatial_size,
    )

    queue_semantic_mask = build_semantic_mask_tensor(action_type_ids, semantic_lookup, "usesQueue")
    units_semantic_mask = build_semantic_mask_tensor(action_type_ids, semantic_lookup, "usesUnits")
    target_entity_semantic_mask = build_semantic_mask_tensor(action_type_ids, semantic_lookup, "usesTargetEntity")
    target_location_semantic_mask = build_semantic_mask_tensor(action_type_ids, semantic_lookup, "usesTargetLocation")
    target_location_2_semantic_mask = build_semantic_mask_tensor(
        action_type_ids,
        semantic_lookup,
        "usesTargetLocation2",
    )
    quantity_semantic_mask = build_semantic_mask_tensor(action_type_ids, semantic_lookup, "usesQuantity")

    action_type_loss_mask = ((action_type_ids >= 0) & (action_type_ids < action_vocab_size)).to(torch.int32).unsqueeze(1)
    delay_loss_mask = ((delay_bins_tensor >= 0) & (delay_bins_tensor < delay_bins)).to(torch.int32).unsqueeze(1)
    queue_loss_mask = queue_semantic_mask * ((queue_tensor >= 0) & (queue_tensor < 2)).to(torch.int32).unsqueeze(1)
    units_loss_mask = units_semantic_mask * units_sequence_mask * units_resolved_mask
    target_entity_loss_mask = target_entity_semantic_mask * target_entity_resolved
    target_location_loss_mask = target_location_semantic_mask * target_location_valid
    target_location_2_loss_mask = target_location_2_semantic_mask * target_location_2_valid
    quantity_loss_mask = quantity_semantic_mask * (quantity_value >= 0).to(torch.int32)

    training_targets = {
        "actionTypeOneHot": action_type_one_hot,
        "delayOneHot": delay_one_hot,
        "queueOneHot": queue_one_hot,
        "unitsOneHot": units_one_hot,
        "targetEntityOneHot": target_entity_one_hot,
        "targetLocationOneHot": target_location_one_hot,
        "targetLocation2OneHot": target_location_2_one_hot,
        "quantityValue": quantity_value,
    }
    training_masks = {
        "actionTypeLossMask": action_type_loss_mask,
        "delayLossMask": delay_loss_mask,
        "queueSemanticMask": queue_semantic_mask,
        "unitsSemanticMask": units_semantic_mask,
        "targetEntitySemanticMask": target_entity_semantic_mask,
        "targetLocationSemanticMask": target_location_semantic_mask,
        "targetLocation2SemanticMask": target_location_2_semantic_mask,
        "quantitySemanticMask": quantity_semantic_mask,
        "queueLossMask": queue_loss_mask,
        "unitsSequenceMask": units_sequence_mask,
        "unitsResolvedMask": units_resolved_mask,
        "unitsLossMask": units_loss_mask,
        "targetEntityResolvedMask": target_entity_resolved,
        "targetEntityLossMask": target_entity_loss_mask,
        "targetLocationValidMask": target_location_valid,
        "targetLocationLossMask": target_location_loss_mask,
        "targetLocation2ValidMask": target_location_2_valid,
        "targetLocation2LossMask": target_location_2_loss_mask,
        "quantityLossMask": quantity_loss_mask,
    }

    training_schema = {
        "version": TRAINING_TARGETS_V1_VERSION,
        "sourceCanonicalLabelVersion": LABEL_LAYOUT_V1_VERSION,
        "actionVocabularySize": action_vocab_size,
        "delayBins": delay_bins,
        "maxEntities": max_entities,
        "maxSelectedUnits": max_selected_units,
        "spatialSize": spatial_size,
        "quantityPolicy": LABEL_LAYOUT_V1_QUANTITY_POLICY,
        "supervisionPolicy": LABEL_LAYOUT_V1_SUPERVISION_POLICY,
        **build_training_target_sections(
            action_vocab_size=action_vocab_size,
            delay_bins=delay_bins,
            max_selected_units=max_selected_units,
            max_entities=max_entities,
            spatial_size=spatial_size,
        ),
        "notes": [
            "This sidecar expands compact canonical V1 labels into model-ready targets and masks.",
            "Action-type one-hot uses the run-global action vocabulary from the transform manifest.",
            "Spatial one-hot targets reuse the same tile-to-grid mapping as py-chronodivide features.",
            "Quantity stays raw integer-valued in V1 and is supervised directly through quantityValue.",
            "Loss masks are the hard supervision policy: semantic masks gate head applicability and replay-time masks gate target validity.",
            "Canonical labels remain the source of truth; these tensors are derived for training convenience.",
        ],
    }

    replay_name = Path(str(metadata["replay"]["path"])).name
    player_name = str(metadata["playerName"])
    validate_tensor_section_shapes(
        training_targets,
        training_schema["targetSections"],
        "trainingTargets",
        replay_name,
        player_name,
        sample_count,
    )
    validate_tensor_section_shapes(
        training_masks,
        training_schema["maskSections"],
        "trainingMasks",
        replay_name,
        player_name,
        sample_count,
    )
    return training_targets, training_masks, training_schema


def finalize_training_target_sidecars(
    config: TransformConfig,
    results: list[dict[str, Any]],
    global_action_vocabulary: list[dict[str, Any]],
) -> None:
    for result in results:
        if result.get("status") not in {"saved", "skipped"}:
            continue
        structured_tensor_path = result.get("structuredTensorPath")
        metadata_path = result.get("metadataPath")
        if not structured_tensor_path or not metadata_path:
            continue

        structured_path = Path(str(structured_tensor_path))
        metadata_file = Path(str(metadata_path))
        if not structured_path.exists() or not metadata_file.exists():
            continue

        training_tensor_path = Path(str(result["tensorPath"])).with_suffix(".training.pt")
        should_write_training = config.overwrite or not training_tensor_path.exists()

        if should_write_training:
            structured_payload = torch.load(structured_path, map_location="cpu", weights_only=True)
            with metadata_file.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            training_targets, training_masks, training_schema = build_training_targets_v1(
                structured_payload["featureTensors"],
                structured_payload["labelTensors"],
                metadata,
                global_action_vocabulary,
            )
            torch.save(
                {
                    "trainingTargets": training_targets,
                    "trainingMasks": training_masks,
                    "sampleContext": structured_payload.get("sampleContext", {}),
                },
                training_tensor_path,
            )
            metadata["trainingTargetTensorPath"] = str(training_tensor_path)
            metadata["trainingTargetsV1"] = training_schema
            metadata["structuredTrainingTargetShapes"] = {
                name: list(tensor.shape) for name, tensor in training_targets.items()
            }
            metadata["structuredTrainingMaskShapes"] = {
                name: list(tensor.shape) for name, tensor in training_masks.items()
            }
            with metadata_file.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

        result["trainingTargetTensorPath"] = str(training_tensor_path)


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
    previous_action_type_id_v1 = -1
    previous_queue = -1

    for sample in samples:
        legacy_label_tensors = get_legacy_label_tensors(sample)
        current_tick = int(sample["tick"])
        delay_from_previous = -1 if previous_tick is None else current_tick - previous_tick
        sample["featureTensors"]["lastActionContext"] = [
            delay_from_previous,
            previous_action_type_id_v1,
            previous_queue,
        ]

        previous_tick = current_tick
        previous_action_type_id_v1 = int(sample["labelTensors"]["actionTypeId"][0])
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
        "legacyFeatureSchema": {
            "featureSections": dataset.get("legacyFeatureSections", []),
            "flatFeatureLength": dataset.get("legacyFlatFeatureLength"),
        },
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
        "featureContextV1": dataset.get("featureContextV1"),
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
    augment_dataset_with_feature_context_v1(dataset)
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
        "trainingTargetShardCount": sum(1 for result in results if result.get("trainingTargetTensorPath")),
        "emptyReplayCount": sum(1 for result in results if result["status"] == "empty"),
        "errorCount": len(errors),
        "staticActionDictVersion": STATIC_ACTION_DICT_VERSION,
        "labelLayoutV1GlobalActionVocabulary": build_global_action_type_vocabulary(run_state),
        "unseenObservedActionTypes": [
            {"name": name, "count": count}
            for name, count in sorted(run_state.unseen_action_type_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
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

    finalize_training_target_sidecars(config, results, build_global_action_type_vocabulary(run_state))
    manifest_path = write_manifest(config, replay_paths, results, errors, run_state)
    print(f"Processed {len(replay_paths)} replay(s).")
    print(f"Saved {sum(1 for result in results if result['status'] == 'saved')} shard(s).")
    print(f"Training targets: {sum(1 for result in results if result.get('trainingTargetTensorPath'))} shard(s).")
    print(f"Manifest: {manifest_path}")

    if errors:
        print(f"Encountered {len(errors)} replay error(s). See manifest for details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
