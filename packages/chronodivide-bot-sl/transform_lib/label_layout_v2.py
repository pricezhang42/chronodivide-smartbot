from __future__ import annotations

import copy
from typing import Any

from action_dict import (
    BUILDABLE_OBJECT_ID_TO_NAME,
    BUILDABLE_OBJECT_NAME_TO_ID,
    BUILDABLE_OBJECT_NAMES,
    UNKNOWN_BUILDABLE_OBJECT_NAME,
)
from transform_lib.action_labels import (
    canonical_target_entity_index,
    canonical_target_location,
    canonical_target_location_2,
    get_legacy_label_tensors,
    get_order_type_name,
    get_queue_update_type_name,
    get_raw_action_name,
    get_shared_name_from_token,
    get_super_weapon_type_name,
    get_target_mode_name,
    valid_tile,
)
from transform_lib.common import LABEL_LAYOUT_V1_DELAY_BINS, LABEL_LAYOUT_V1_MISSING_INT, LABEL_LAYOUT_V2_VERSION
from transform_lib.schema_utils import compute_flat_length

LABEL_LAYOUT_V2_PREVIEW_VERSION = "v2_preview_command_stream_v1"
LABEL_LAYOUT_V2_ACTION_FAMILIES = [
    "Order",
    "Queue",
    "PlaceBuilding",
    "ActivateSuperWeapon",
    "SellObject",
    "ToggleRepair",
    "ResignGame",
]
LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID = {
    name: index for index, name in enumerate(LABEL_LAYOUT_V2_ACTION_FAMILIES)
}
LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES = ["Add", "Cancel", "AddNext"]
LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPE_TO_ID = {
    name: index for index, name in enumerate(LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES)
}
LABEL_LAYOUT_V2_ITEM_LEVEL_QUEUE_UPDATES = set(LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES)
LABEL_LAYOUT_V2_DISABLED_QUEUE_UPDATES = {"Hold", "Resume"}
LABEL_LAYOUT_V2_CORE_LABEL_SECTIONS = [
    "actionFamilyId",
    "delayBin",
    "orderTypeId",
    "targetModeId",
    "queueFlag",
    "queueUpdateTypeId",
    "buildableObjectToken",
    "superWeaponTypeId",
    "commandedUnitsIndices",
    "commandedUnitsMask",
    "commandedUnitsResolvedMask",
    "targetEntityIndex",
    "targetEntityResolved",
    "targetLocation",
    "targetLocationValid",
    "targetLocation2",
    "targetLocation2Valid",
    "quantity",
]


def get_v2_buildable_object_id(name: str | None) -> int:
    if not isinstance(name, str):
        return int(BUILDABLE_OBJECT_NAME_TO_ID[UNKNOWN_BUILDABLE_OBJECT_NAME])
    return int(BUILDABLE_OBJECT_NAME_TO_ID.get(name, BUILDABLE_OBJECT_NAME_TO_ID[UNKNOWN_BUILDABLE_OBJECT_NAME]))


def _sorted_count_items(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _commanded_units_stats(sample: dict[str, Any]) -> tuple[int, int, int]:
    feature_tensors = sample.get("featureTensors", {})
    return (
        int(feature_tensors.get("currentSelectionCount", [0])[0]),
        int(feature_tensors.get("currentSelectionResolvedCount", [0])[0]),
        int(feature_tensors.get("currentSelectionOverflowCount", [0])[0]),
    )


def _build_v2_preview_candidate(sample: dict[str, Any], dataset: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw_action_name = get_raw_action_name(sample, dataset)
    legacy_label_tensors = get_legacy_label_tensors(sample)

    if raw_action_name == "SelectUnitsAction":
        return None, "selection_folded"

    if raw_action_name == "OrderUnitsAction":
        commanded_units_count, commanded_units_resolved_count, commanded_units_overflow_count = _commanded_units_stats(sample)
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["Order"],
            "actionFamilyName": "Order",
            "rawActionName": raw_action_name,
            "orderTypeName": get_order_type_name(sample, dataset),
            "targetModeName": get_target_mode_name(sample, dataset),
            "queueFlag": int(legacy_label_tensors["queue"][0]),
            "commandedUnitsCount": commanded_units_count,
            "commandedUnitsResolvedCount": commanded_units_resolved_count,
            "commandedUnitsOverflowCount": commanded_units_overflow_count,
        }, None

    if raw_action_name == "UpdateQueueAction":
        queue_update_type_name = get_queue_update_type_name(sample, dataset)
        if queue_update_type_name in LABEL_LAYOUT_V2_DISABLED_QUEUE_UPDATES:
            return None, f"queue_{str(queue_update_type_name).lower()}_disabled"
        if queue_update_type_name not in LABEL_LAYOUT_V2_ITEM_LEVEL_QUEUE_UPDATES:
            return None, "unsupported_queue_update"
        item_name = get_shared_name_from_token(dataset, int(legacy_label_tensors["itemNameToken"][0]))
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["Queue"],
            "actionFamilyName": "Queue",
            "rawActionName": raw_action_name,
            "queueUpdateTypeName": queue_update_type_name,
            "buildableObjectName": item_name,
            "buildableObjectId": get_v2_buildable_object_id(item_name),
            "quantity": int(legacy_label_tensors["quantity"][0]),
        }, None

    if raw_action_name == "PlaceBuildingAction":
        building_name = get_shared_name_from_token(dataset, int(legacy_label_tensors["buildingNameToken"][0]))
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["PlaceBuilding"],
            "actionFamilyName": "PlaceBuilding",
            "rawActionName": raw_action_name,
            "buildableObjectName": building_name,
            "buildableObjectId": get_v2_buildable_object_id(building_name),
        }, None

    if raw_action_name == "ActivateSuperWeaponAction":
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["ActivateSuperWeapon"],
            "actionFamilyName": "ActivateSuperWeapon",
            "rawActionName": raw_action_name,
            "superWeaponTypeName": get_super_weapon_type_name(sample, dataset),
        }, None

    if raw_action_name == "SellObjectAction":
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["SellObject"],
            "actionFamilyName": "SellObject",
            "rawActionName": raw_action_name,
        }, None

    if raw_action_name == "ToggleRepairAction":
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["ToggleRepair"],
            "actionFamilyName": "ToggleRepair",
            "rawActionName": raw_action_name,
        }, None

    if raw_action_name == "ResignGameAction":
        return {
            "tick": int(sample["tick"]),
            "actionFamilyId": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["ResignGame"],
            "actionFamilyName": "ResignGame",
            "rawActionName": raw_action_name,
        }, None

    return None, "unsupported_raw_action"


def build_player_label_layout_v2_preview(
    samples: list[dict[str, Any]],
    dataset: dict[str, Any],
    player_name: str,
) -> dict[str, Any]:
    kept_preview_samples: list[dict[str, Any]] = []
    dropped_reason_counts: dict[str, int] = {}
    action_family_counts: dict[str, int] = {}
    queue_update_type_counts: dict[str, int] = {}
    order_type_counts: dict[str, int] = {}
    source_raw_action_counts: dict[str, int] = {}

    for sample in samples:
        raw_action_name = get_raw_action_name(sample, dataset) or "UnknownRawAction"
        source_raw_action_counts[raw_action_name] = source_raw_action_counts.get(raw_action_name, 0) + 1

        preview_candidate, drop_reason = _build_v2_preview_candidate(sample, dataset)
        derived_metadata = sample.setdefault("derivedLabelMetadata", {})

        if preview_candidate is None:
            reason = drop_reason or "dropped"
            dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
            derived_metadata["labelLayoutV2Preview"] = {
                "keptInCommandStream": False,
                "dropReason": reason,
            }
            continue

        action_family_name = str(preview_candidate["actionFamilyName"])
        action_family_counts[action_family_name] = action_family_counts.get(action_family_name, 0) + 1
        queue_update_type_name = preview_candidate.get("queueUpdateTypeName")
        if isinstance(queue_update_type_name, str):
            queue_update_type_counts[queue_update_type_name] = queue_update_type_counts.get(queue_update_type_name, 0) + 1
        order_type_name = preview_candidate.get("orderTypeName")
        if isinstance(order_type_name, str):
            order_type_counts[order_type_name] = order_type_counts.get(order_type_name, 0) + 1

        preview_payload = dict(preview_candidate)
        preview_payload.update(
            {
                "keptInCommandStream": True,
                "delayToNextCommand": LABEL_LAYOUT_V1_MISSING_INT,
                "delayBin": LABEL_LAYOUT_V1_MISSING_INT,
            }
        )
        derived_metadata["labelLayoutV2Preview"] = preview_payload
        kept_preview_samples.append(sample)

    for index, sample in enumerate(kept_preview_samples):
        preview_payload = sample["derivedLabelMetadata"]["labelLayoutV2Preview"]
        next_tick = int(kept_preview_samples[index + 1]["tick"]) if index + 1 < len(kept_preview_samples) else None
        delay_to_next_command = (
            LABEL_LAYOUT_V1_MISSING_INT if next_tick is None else next_tick - int(sample["tick"])
        )
        preview_payload["delayToNextCommand"] = delay_to_next_command
        preview_payload["delayBin"] = (
            LABEL_LAYOUT_V1_MISSING_INT
            if delay_to_next_command < 0
            else min(delay_to_next_command, LABEL_LAYOUT_V1_DELAY_BINS - 1)
        )

    return {
        "version": LABEL_LAYOUT_V2_PREVIEW_VERSION,
        "playerName": player_name,
        "sourceSampleCount": len(samples),
        "keptCommandSampleCount": len(kept_preview_samples),
        "droppedSampleCount": len(samples) - len(kept_preview_samples),
        "selectionFoldedCount": dropped_reason_counts.get("selection_folded", 0),
        "disabledQueueActionCount": sum(
            dropped_reason_counts.get(reason, 0)
            for reason in ("queue_hold_disabled", "queue_resume_disabled")
        ),
        "sourceRawActionCounts": _sorted_count_items(source_raw_action_counts),
        "droppedReasonCounts": _sorted_count_items(dropped_reason_counts),
        "actionFamilyCounts": _sorted_count_items(action_family_counts),
        "queueUpdateTypeCounts": _sorted_count_items(queue_update_type_counts),
        "orderTypeCounts": _sorted_count_items(order_type_counts),
        "previewExamples": [
            sample["derivedLabelMetadata"]["labelLayoutV2Preview"]
            for sample in kept_preview_samples[:12]
        ],
        "notes": [
            "This is a V2 command-stream preview built from the unfiltered extracted player action stream.",
            "Standalone SelectUnitsAction rows are folded into later commands and do not emit main V2 policy samples.",
            "Queue Hold and Resume are currently excluded from the main V2 action space.",
            "delayToNextCommand is recomputed on the kept V2 preview stream, not copied from the V1 action-aligned stream.",
        ],
    }


def augment_dataset_with_label_layout_v2_preview(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped_samples.setdefault(sample["playerName"], []).append(sample)

    preview_by_player: dict[str, Any] = {}
    aggregate_source_sample_count = 0
    aggregate_kept_command_sample_count = 0
    aggregate_dropped_sample_count = 0
    aggregate_selection_folded_count = 0
    aggregate_disabled_queue_action_count = 0

    for player_name, player_samples in grouped_samples.items():
        preview = build_player_label_layout_v2_preview(player_samples, dataset, player_name)
        preview_by_player[player_name] = preview
        aggregate_source_sample_count += int(preview["sourceSampleCount"])
        aggregate_kept_command_sample_count += int(preview["keptCommandSampleCount"])
        aggregate_dropped_sample_count += int(preview["droppedSampleCount"])
        aggregate_selection_folded_count += int(preview["selectionFoldedCount"])
        aggregate_disabled_queue_action_count += int(preview["disabledQueueActionCount"])

    dataset["labelLayoutV2Preview"] = {
        "version": LABEL_LAYOUT_V2_PREVIEW_VERSION,
        "status": "preview_only",
        "mainTensorUsesV2": False,
        "currentStagePolicy": {
            "foldSelectionIntoCommands": True,
            "keepQueueUpdates": sorted(LABEL_LAYOUT_V2_ITEM_LEVEL_QUEUE_UPDATES),
            "disabledQueueUpdates": sorted(LABEL_LAYOUT_V2_DISABLED_QUEUE_UPDATES),
        },
        "actionFamilies": [
            {"id": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID[name], "name": name}
            for name in LABEL_LAYOUT_V2_ACTION_FAMILIES
        ],
        "sourceSampleCount": aggregate_source_sample_count,
        "keptCommandSampleCount": aggregate_kept_command_sample_count,
        "droppedSampleCount": aggregate_dropped_sample_count,
        "selectionFoldedCount": aggregate_selection_folded_count,
        "disabledQueueActionCount": aggregate_disabled_queue_action_count,
        "byPlayer": preview_by_player,
    }


def build_v2_label_sections(max_commanded_units: int) -> list[dict[str, Any]]:
    return [
        {"name": "actionFamilyId", "shape": [1], "dtype": "int32"},
        {"name": "delayBin", "shape": [1], "dtype": "int32"},
        {"name": "orderTypeId", "shape": [1], "dtype": "int32"},
        {"name": "targetModeId", "shape": [1], "dtype": "int32"},
        {"name": "queueFlag", "shape": [1], "dtype": "int32"},
        {"name": "queueUpdateTypeId", "shape": [1], "dtype": "int32"},
        {"name": "buildableObjectToken", "shape": [1], "dtype": "int32"},
        {"name": "superWeaponTypeId", "shape": [1], "dtype": "int32"},
        {"name": "commandedUnitsIndices", "shape": [max_commanded_units], "dtype": "int32"},
        {"name": "commandedUnitsMask", "shape": [max_commanded_units], "dtype": "int32"},
        {"name": "commandedUnitsResolvedMask", "shape": [max_commanded_units], "dtype": "int32"},
        {"name": "targetEntityIndex", "shape": [1], "dtype": "int32"},
        {"name": "targetEntityResolved", "shape": [1], "dtype": "int32"},
        {"name": "targetLocation", "shape": [2], "dtype": "int32"},
        {"name": "targetLocationValid", "shape": [1], "dtype": "int32"},
        {"name": "targetLocation2", "shape": [2], "dtype": "int32"},
        {"name": "targetLocation2Valid", "shape": [1], "dtype": "int32"},
        {"name": "quantity", "shape": [1], "dtype": "int32"},
    ]


def _inactive_commanded_units(sample: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    feature_tensors = sample["featureTensors"]
    length = len(feature_tensors["currentSelectionIndices"])
    return (
        [LABEL_LAYOUT_V1_MISSING_INT] * length,
        [0] * length,
        [0] * length,
    )


def _target_location_or_missing(tile: list[int]) -> tuple[list[int], int]:
    normalized_tile = [int(value) for value in tile]
    return normalized_tile, int(valid_tile(normalized_tile))


def build_v2_canonical_label_tensors(sample: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any] | None:
    preview_payload = sample.get("derivedLabelMetadata", {}).get("labelLayoutV2Preview", {})
    if not preview_payload.get("keptInCommandStream", False):
        return None

    legacy_label_tensors = get_legacy_label_tensors(sample)
    raw_action_name = get_raw_action_name(sample, dataset)
    feature_tensors = sample["featureTensors"]

    order_type_id = LABEL_LAYOUT_V1_MISSING_INT
    target_mode_id = LABEL_LAYOUT_V1_MISSING_INT
    queue_flag = LABEL_LAYOUT_V1_MISSING_INT
    queue_update_type_id = LABEL_LAYOUT_V1_MISSING_INT
    buildable_object_token = LABEL_LAYOUT_V1_MISSING_INT
    super_weapon_type_id = LABEL_LAYOUT_V1_MISSING_INT
    quantity = LABEL_LAYOUT_V1_MISSING_INT
    commanded_units_indices, commanded_units_mask, commanded_units_resolved_mask = _inactive_commanded_units(sample)
    target_entity_index = LABEL_LAYOUT_V1_MISSING_INT
    target_entity_resolved = 0
    target_location = [LABEL_LAYOUT_V1_MISSING_INT, LABEL_LAYOUT_V1_MISSING_INT]
    target_location_valid = 0
    target_location_2 = [LABEL_LAYOUT_V1_MISSING_INT, LABEL_LAYOUT_V1_MISSING_INT]
    target_location_2_valid = 0

    if raw_action_name == "OrderUnitsAction":
        order_type_id = int(legacy_label_tensors["orderTypeId"][0])
        target_mode_id = int(legacy_label_tensors["targetModeId"][0])
        queue_flag = int(legacy_label_tensors["queue"][0])
        commanded_units_indices = [int(value) for value in feature_tensors["currentSelectionIndices"]]
        commanded_units_mask = [int(value) for value in feature_tensors["currentSelectionMask"]]
        commanded_units_resolved_mask = [int(value) for value in feature_tensors["currentSelectionResolvedMask"]]
        if get_target_mode_name(sample, dataset) == "object":
            target_entity_index = canonical_target_entity_index(sample, dataset)
            target_entity_resolved = int(target_entity_index >= 0)
        if get_target_mode_name(sample, dataset) in {"tile", "ore_tile"}:
            target_location, target_location_valid = _target_location_or_missing(
                canonical_target_location(sample, dataset)
            )

    elif raw_action_name == "UpdateQueueAction":
        queue_update_type_name = get_queue_update_type_name(sample, dataset)
        queue_update_type_id = LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPE_TO_ID[queue_update_type_name]
        buildable_object_name = get_shared_name_from_token(dataset, int(legacy_label_tensors["itemNameToken"][0]))
        buildable_object_token = get_v2_buildable_object_id(buildable_object_name)
        quantity = int(legacy_label_tensors["quantity"][0])

    elif raw_action_name == "PlaceBuildingAction":
        buildable_object_name = get_shared_name_from_token(dataset, int(legacy_label_tensors["buildingNameToken"][0]))
        buildable_object_token = get_v2_buildable_object_id(buildable_object_name)
        target_location, target_location_valid = _target_location_or_missing(
            canonical_target_location(sample, dataset)
        )

    elif raw_action_name == "ActivateSuperWeaponAction":
        super_weapon_type_id = int(legacy_label_tensors["superWeaponTypeId"][0])
        target_location, target_location_valid = _target_location_or_missing(
            canonical_target_location(sample, dataset)
        )
        target_location_2, target_location_2_valid = _target_location_or_missing(
            canonical_target_location_2(sample, dataset)
        )

    elif raw_action_name in {"SellObjectAction", "ToggleRepairAction"}:
        target_entity_index = canonical_target_entity_index(sample, dataset)
        target_entity_resolved = int(target_entity_index >= 0)

    return {
        "actionFamilyId": [int(preview_payload["actionFamilyId"])],
        "delayBin": [LABEL_LAYOUT_V1_MISSING_INT],
        "orderTypeId": [order_type_id],
        "targetModeId": [target_mode_id],
        "queueFlag": [queue_flag],
        "queueUpdateTypeId": [queue_update_type_id],
        "buildableObjectToken": [buildable_object_token],
        "superWeaponTypeId": [super_weapon_type_id],
        "commandedUnitsIndices": commanded_units_indices,
        "commandedUnitsMask": commanded_units_mask,
        "commandedUnitsResolvedMask": commanded_units_resolved_mask,
        "targetEntityIndex": [target_entity_index],
        "targetEntityResolved": [target_entity_resolved],
        "targetLocation": target_location,
        "targetLocationValid": [target_location_valid],
        "targetLocation2": target_location_2,
        "targetLocation2Valid": [target_location_2_valid],
        "quantity": [quantity],
    }


def build_player_v2_canonical_samples(
    samples: list[dict[str, Any]],
    dataset: dict[str, Any],
) -> list[dict[str, Any]]:
    kept_samples: list[dict[str, Any]] = []
    previous_tick: int | None = None
    previous_action_family_id = LABEL_LAYOUT_V1_MISSING_INT
    previous_queue_flag = LABEL_LAYOUT_V1_MISSING_INT

    for source_index, sample in enumerate(samples):
        label_tensors = build_v2_canonical_label_tensors(sample, dataset)
        if label_tensors is None:
            continue

        feature_tensors = copy.deepcopy(sample["featureTensors"])
        current_tick = int(sample["tick"])
        delay_from_previous = LABEL_LAYOUT_V1_MISSING_INT if previous_tick is None else current_tick - previous_tick
        feature_tensors["lastActionContext"] = [
            delay_from_previous,
            previous_action_family_id,
            previous_queue_flag,
        ]

        kept_sample = {
            "tick": current_tick,
            "playerId": int(sample["playerId"]),
            "playerName": sample["playerName"],
            "featureTensors": feature_tensors,
            "labelTensors": label_tensors,
            "sourceSampleIndex": source_index,
        }
        kept_samples.append(kept_sample)
        previous_tick = current_tick
        previous_action_family_id = int(label_tensors["actionFamilyId"][0])
        previous_queue_flag = int(label_tensors["queueFlag"][0])

    for index, sample in enumerate(kept_samples):
        next_tick = int(kept_samples[index + 1]["tick"]) if index + 1 < len(kept_samples) else None
        delay_to_next = LABEL_LAYOUT_V1_MISSING_INT if next_tick is None else next_tick - int(sample["tick"])
        sample["labelTensors"]["delayBin"] = [
            LABEL_LAYOUT_V1_MISSING_INT
            if delay_to_next < 0
            else min(delay_to_next, LABEL_LAYOUT_V1_DELAY_BINS - 1)
        ]

    return kept_samples


def build_label_layout_v2_metadata(dataset: dict[str, Any], player_name: str, sample_count: int) -> dict[str, Any]:
    action_schema = dataset["schema"]["action"]
    preview_metadata = dataset.get("labelLayoutV2Preview", {}).get("byPlayer", {}).get(player_name, {})
    max_commanded_units = int(dataset["options"]["maxSelectedUnits"])

    label_sections = build_v2_label_sections(max_commanded_units)
    order_type_count = max(int(key) for key in action_schema.get("orderTypeNames", {}).keys()) + 1
    super_weapon_type_count = max(int(key) for key in action_schema.get("superWeaponTypeNames", {}).keys()) + 1

    return {
        "version": LABEL_LAYOUT_V2_VERSION,
        "status": "parallel_canonical_sidecar",
        "mainTensorUsesV2": False,
        "currentStagePolicy": {
            "foldSelectionIntoCommands": True,
            "keepQueueUpdates": list(LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES),
            "disabledQueueUpdates": sorted(LABEL_LAYOUT_V2_DISABLED_QUEUE_UPDATES),
        },
        "coreLabelSections": LABEL_LAYOUT_V2_CORE_LABEL_SECTIONS,
        "delayBins": LABEL_LAYOUT_V1_DELAY_BINS,
        "missingInt": LABEL_LAYOUT_V1_MISSING_INT,
        "actionFamilies": [
            {"id": LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID[name], "name": name}
            for name in LABEL_LAYOUT_V2_ACTION_FAMILIES
        ],
        "queueUpdateTypes": [
            {"id": LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPE_TO_ID[name], "name": name}
            for name in LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES
        ],
        "orderTypeCount": order_type_count,
        "targetModeCount": len(action_schema.get("targetModes", [])),
        "buildableObjectVocabularySize": len(BUILDABLE_OBJECT_NAMES),
        "buildableObjectVocabulary": [
            {"id": buildable_object_id, "name": buildable_object_name}
            for buildable_object_id, buildable_object_name in BUILDABLE_OBJECT_ID_TO_NAME.items()
        ],
        "superWeaponTypeCount": super_weapon_type_count,
        "commandSampleCount": sample_count,
        "sourcePreviewCounts": {
            "sourceSampleCount": preview_metadata.get("sourceSampleCount", 0),
            "keptCommandSampleCount": preview_metadata.get("keptCommandSampleCount", sample_count),
            "selectionFoldedCount": preview_metadata.get("selectionFoldedCount", 0),
            "disabledQueueActionCount": preview_metadata.get("disabledQueueActionCount", 0),
        },
        "labelSections": label_sections,
        "flatLabelLength": compute_flat_length(label_sections),
        "featureContext": {
            "version": "v2_parallel_sidecar_v1",
            "lastActionContextShape": [3],
            "lastActionContextFields": [
                "delayFromPreviousCommand",
                "lastActionFamilyIdV2",
                "lastQueueFlag",
            ],
            "notes": [
                "V2 sidecars currently reuse the existing feature sections and rewrite lastActionContext on the kept V2 command stream.",
                "This is a parallel artifact path; the main saved flat tensors remain V1.",
            ],
        },
        "notes": [
            "Standalone SelectUnitsAction rows are folded out of the canonical V2 sidecar.",
            "Queue Hold and Resume are excluded from the canonical V2 sidecar for the current stage.",
            "Queue supervision is item-level only in the current-stage V2 path.",
            "buildableObjectToken uses a static cross-replay vocabulary, not the replay-local sharedNameVocabulary ids.",
        ],
    }
