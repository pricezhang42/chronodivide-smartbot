from __future__ import annotations

import copy
import re
from typing import Any

from action_dict import (
    ACTION_INFO_MASK as STATIC_ACTION_INFO_MASK,
    ACTION_TYPE_ID_TO_NAME as STATIC_ACTION_TYPE_ID_TO_NAME,
    ACTION_TYPE_NAME_TO_ID as STATIC_ACTION_TYPE_NAME_TO_ID,
    STATIC_ACTION_DICT_VERSION,
    UNKNOWN_ACTION_TYPE_NAME,
    build_observed_action_type_name,
    canonicalize_action_type_name,
)

from transform_lib.common import (
    FEATURE_CONTEXT_V1_VERSION,
    LABEL_LAYOUT_V1_ACTION_VOCAB_SCOPE,
    LABEL_LAYOUT_V1_CORE_LABEL_SECTIONS,
    LABEL_LAYOUT_V1_DELAY_BINS,
    LABEL_LAYOUT_V1_MISSING_INT,
    LABEL_LAYOUT_V1_QUANTITY_POLICY,
    LABEL_LAYOUT_V1_SUPERVISION_POLICY,
    LABEL_LAYOUT_V1_VERSION,
    TransformRunState,
)
from transform_lib.schema_utils import compute_flat_length, get_section_shape


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
        if len(legacy_last_action_context) > 0:
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
        "scope": LABEL_LAYOUT_V1_ACTION_VOCAB_SCOPE,
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
            "scope": LABEL_LAYOUT_V1_ACTION_VOCAB_SCOPE,
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
