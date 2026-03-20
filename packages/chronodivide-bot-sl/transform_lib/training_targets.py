from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from transform_lib.common import (
    LABEL_LAYOUT_V1_DELAY_BINS,
    LABEL_LAYOUT_V1_QUANTITY_POLICY,
    LABEL_LAYOUT_V1_SUPERVISION_POLICY,
    LABEL_LAYOUT_V1_VERSION,
    LABEL_LAYOUT_V2_VERSION,
    TRAINING_TARGETS_V1_VERSION,
    TRAINING_TARGETS_V2_VERSION,
    TransformConfig,
)
from transform_lib.label_layout_v2 import (
    LABEL_LAYOUT_V2_ACTION_FAMILIES,
    LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID,
    LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES,
)


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
    location_target_size: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "targetSections": [
            {"name": "actionTypeOneHot", "shape": [action_vocab_size], "dtype": "int32"},
            {"name": "delayOneHot", "shape": [delay_bins], "dtype": "int32"},
            {"name": "queueOneHot", "shape": [2], "dtype": "int32"},
            {"name": "unitsOneHot", "shape": [max_selected_units, max_entities], "dtype": "int32"},
            {"name": "targetEntityOneHot", "shape": [max_entities], "dtype": "int32"},
            {"name": "targetLocationOneHot", "shape": [location_target_size, location_target_size], "dtype": "int32"},
            {"name": "targetLocation2OneHot", "shape": [location_target_size, location_target_size], "dtype": "int32"},
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


def build_training_target_sections_v2(
    *,
    action_family_count: int,
    delay_bins: int,
    order_type_count: int,
    target_mode_count: int,
    queue_update_type_count: int,
    buildable_object_vocab_size: int,
    super_weapon_type_count: int,
    max_selected_units: int,
    max_entities: int,
    location_target_size: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "targetSections": [
            {"name": "actionFamilyOneHot", "shape": [action_family_count], "dtype": "int32"},
            {"name": "delayOneHot", "shape": [delay_bins], "dtype": "int32"},
            {"name": "orderTypeOneHot", "shape": [order_type_count], "dtype": "int32"},
            {"name": "targetModeOneHot", "shape": [target_mode_count], "dtype": "int32"},
            {"name": "queueFlagOneHot", "shape": [2], "dtype": "int32"},
            {"name": "queueUpdateTypeOneHot", "shape": [queue_update_type_count], "dtype": "int32"},
            {"name": "buildableObjectOneHot", "shape": [buildable_object_vocab_size], "dtype": "int32"},
            {"name": "superWeaponTypeOneHot", "shape": [super_weapon_type_count], "dtype": "int32"},
            {"name": "commandedUnitsOneHot", "shape": [max_selected_units, max_entities], "dtype": "int32"},
            {"name": "targetEntityOneHot", "shape": [max_entities], "dtype": "int32"},
            {"name": "targetLocationOneHot", "shape": [location_target_size, location_target_size], "dtype": "int32"},
            {"name": "targetLocation2OneHot", "shape": [location_target_size, location_target_size], "dtype": "int32"},
            {"name": "quantityValue", "shape": [1], "dtype": "int32"},
        ],
        "maskSections": [
            {"name": "actionFamilyLossMask", "shape": [1], "dtype": "int32"},
            {"name": "delayLossMask", "shape": [1], "dtype": "int32"},
            {"name": "orderTypeSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetModeSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "queueFlagSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "queueUpdateTypeSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "buildableObjectSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "superWeaponTypeSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "commandedUnitsSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetEntitySemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocationSemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "targetLocation2SemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "quantitySemanticMask", "shape": [1], "dtype": "int32"},
            {"name": "orderTypeLossMask", "shape": [1], "dtype": "int32"},
            {"name": "targetModeLossMask", "shape": [1], "dtype": "int32"},
            {"name": "queueFlagLossMask", "shape": [1], "dtype": "int32"},
            {"name": "queueUpdateTypeLossMask", "shape": [1], "dtype": "int32"},
            {"name": "buildableObjectLossMask", "shape": [1], "dtype": "int32"},
            {"name": "superWeaponTypeLossMask", "shape": [1], "dtype": "int32"},
            {"name": "commandedUnitsSequenceMask", "shape": [max_selected_units], "dtype": "int32"},
            {"name": "commandedUnitsResolvedMask", "shape": [max_selected_units], "dtype": "int32"},
            {"name": "commandedUnitsLossMask", "shape": [max_selected_units], "dtype": "int32"},
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
    observation_spatial_size = int(schema["observation"]["spatialSize"])
    location_target_size = int(schema["observation"].get("minimapSize", observation_spatial_size))
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
        location_target_size,
    )
    target_location_2_one_hot = build_spatial_one_hot(
        target_location_2,
        target_location_2_valid,
        map_widths,
        map_heights,
        location_target_size,
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
        "spatialSize": location_target_size,
        "observationSpatialSize": observation_spatial_size,
        "locationTargetSize": location_target_size,
        "quantityPolicy": LABEL_LAYOUT_V1_QUANTITY_POLICY,
        "supervisionPolicy": LABEL_LAYOUT_V1_SUPERVISION_POLICY,
        **build_training_target_sections(
            action_vocab_size=action_vocab_size,
            delay_bins=delay_bins,
            max_selected_units=max_selected_units,
            max_entities=max_entities,
            location_target_size=location_target_size,
        ),
        "notes": [
            "This sidecar expands compact canonical V1 labels into model-ready targets and masks.",
            "Action-type one-hot uses the run-global action vocabulary from the transform manifest.",
            "Spatial one-hot targets reuse the same tile-to-grid mapping as py-chronodivide features.",
            "Location targets now use the observation minimap resolution when available, so target planes are native 64x64 on the current dataset path.",
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


def build_training_targets_v2(
    feature_section_tensors: dict[str, torch.Tensor],
    label_section_tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    schema = metadata["schema"]
    label_layout_v2 = metadata.get("labelLayoutV2", {})
    delay_bins = int(label_layout_v2.get("delayBins", LABEL_LAYOUT_V1_DELAY_BINS))
    observation_spatial_size = int(schema["observation"]["spatialSize"])
    location_target_size = int(schema["observation"].get("minimapSize", observation_spatial_size))
    max_entities = int(schema["observation"]["maxEntities"])
    max_commanded_units = int(label_section_tensors["commandedUnitsIndices"].shape[1])
    sample_count = int(label_section_tensors["actionFamilyId"].shape[0])
    scalar_tensor = feature_section_tensors["scalar"]
    map_width_index = get_observation_scalar_index(schema, "map_width")
    map_height_index = get_observation_scalar_index(schema, "map_height")
    map_widths = scalar_tensor[:, map_width_index]
    map_heights = scalar_tensor[:, map_height_index]

    order_type_count = int(label_layout_v2.get("orderTypeCount", 0))
    target_mode_count = int(label_layout_v2.get("targetModeCount", 0))
    queue_update_type_count = len(LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES)
    buildable_object_vocab_size = int(label_layout_v2.get("buildableObjectVocabularySize", 0))
    super_weapon_type_count = int(label_layout_v2.get("superWeaponTypeCount", 0))

    action_family_ids = label_section_tensors["actionFamilyId"].squeeze(1).to(torch.int64)
    delay_bins_tensor = label_section_tensors["delayBin"].squeeze(1).to(torch.int64)
    order_type_ids = label_section_tensors["orderTypeId"].squeeze(1).to(torch.int64)
    target_mode_ids = label_section_tensors["targetModeId"].squeeze(1).to(torch.int64)
    queue_flag_tensor = label_section_tensors["queueFlag"].squeeze(1).to(torch.int64)
    queue_update_type_ids = label_section_tensors["queueUpdateTypeId"].squeeze(1).to(torch.int64)
    buildable_object_tokens = label_section_tensors["buildableObjectToken"].squeeze(1).to(torch.int64)
    super_weapon_type_ids = label_section_tensors["superWeaponTypeId"].squeeze(1).to(torch.int64)
    commanded_units_indices = label_section_tensors["commandedUnitsIndices"].to(torch.int64)
    commanded_units_sequence_mask = label_section_tensors["commandedUnitsMask"].to(torch.int32)
    commanded_units_resolved_mask = label_section_tensors["commandedUnitsResolvedMask"].to(torch.int32)
    target_entity_indices = label_section_tensors["targetEntityIndex"].squeeze(1).to(torch.int64)
    target_entity_resolved = label_section_tensors["targetEntityResolved"].to(torch.int32)
    target_location = label_section_tensors["targetLocation"].to(torch.int64)
    target_location_valid = label_section_tensors["targetLocationValid"].to(torch.int32)
    target_location_2 = label_section_tensors["targetLocation2"].to(torch.int64)
    target_location_2_valid = label_section_tensors["targetLocation2Valid"].to(torch.int32)
    quantity_value = label_section_tensors["quantity"].to(torch.int32)

    order_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["Order"]
    queue_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["Queue"]
    place_building_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["PlaceBuilding"]
    super_weapon_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["ActivateSuperWeapon"]
    sell_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["SellObject"]
    toggle_repair_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["ToggleRepair"]

    target_mode_names = list(schema.get("action", {}).get("targetModes", []))
    target_mode_id_by_name = {name: index for index, name in enumerate(target_mode_names)}
    tile_target_mode_ids = {
        target_mode_id_by_name[name]
        for name in ("tile", "ore_tile")
        if name in target_mode_id_by_name
    }
    object_target_mode_id = target_mode_id_by_name.get("object", -999)

    order_active = (action_family_ids == order_family_id)
    queue_active = (action_family_ids == queue_family_id)
    place_building_active = (action_family_ids == place_building_family_id)
    super_weapon_active = (action_family_ids == super_weapon_family_id)
    sell_or_toggle_active = (action_family_ids == sell_family_id) | (action_family_ids == toggle_repair_family_id)

    target_mode_tile_active = torch.zeros_like(order_active)
    for target_mode_id in tile_target_mode_ids:
        target_mode_tile_active = target_mode_tile_active | (target_mode_ids == target_mode_id)
    target_mode_object_active = target_mode_ids == object_target_mode_id

    order_type_semantic_mask = order_active.to(torch.int32).unsqueeze(1)
    target_mode_semantic_mask = order_active.to(torch.int32).unsqueeze(1)
    queue_flag_semantic_mask = order_active.to(torch.int32).unsqueeze(1)
    queue_update_type_semantic_mask = queue_active.to(torch.int32).unsqueeze(1)
    buildable_object_semantic_mask = (queue_active | place_building_active).to(torch.int32).unsqueeze(1)
    super_weapon_type_semantic_mask = super_weapon_active.to(torch.int32).unsqueeze(1)
    commanded_units_semantic_mask = order_active.to(torch.int32).unsqueeze(1)
    target_entity_semantic_mask = ((order_active & target_mode_object_active) | sell_or_toggle_active).to(torch.int32).unsqueeze(1)
    target_location_semantic_mask = ((order_active & target_mode_tile_active) | place_building_active | super_weapon_active).to(torch.int32).unsqueeze(1)
    target_location_2_semantic_mask = super_weapon_active.to(torch.int32).unsqueeze(1)
    quantity_semantic_mask = queue_active.to(torch.int32).unsqueeze(1)

    action_family_one_hot = build_one_hot_matrix(action_family_ids, len(LABEL_LAYOUT_V2_ACTION_FAMILIES))
    delay_one_hot = build_one_hot_matrix(delay_bins_tensor, delay_bins)
    order_type_one_hot = build_one_hot_matrix(order_type_ids, order_type_count)
    target_mode_one_hot = build_one_hot_matrix(target_mode_ids, target_mode_count)
    queue_flag_one_hot = build_one_hot_matrix(queue_flag_tensor, 2)
    queue_update_type_one_hot = build_one_hot_matrix(queue_update_type_ids, queue_update_type_count)
    buildable_object_one_hot = build_one_hot_matrix(buildable_object_tokens, buildable_object_vocab_size)
    super_weapon_type_one_hot = build_one_hot_matrix(super_weapon_type_ids, super_weapon_type_count)
    commanded_units_one_hot = build_one_hot_sequence(commanded_units_indices, max_entities)
    target_entity_one_hot = build_one_hot_matrix(target_entity_indices, max_entities)
    target_location_one_hot = build_spatial_one_hot(
        target_location,
        target_location_valid,
        map_widths,
        map_heights,
        location_target_size,
    )
    target_location_2_one_hot = build_spatial_one_hot(
        target_location_2,
        target_location_2_valid,
        map_widths,
        map_heights,
        location_target_size,
    )

    action_family_loss_mask = (
        (action_family_ids >= 0) & (action_family_ids < len(LABEL_LAYOUT_V2_ACTION_FAMILIES))
    ).to(torch.int32).unsqueeze(1)
    delay_loss_mask = ((delay_bins_tensor >= 0) & (delay_bins_tensor < delay_bins)).to(torch.int32).unsqueeze(1)
    order_type_loss_mask = order_type_semantic_mask * ((order_type_ids >= 0) & (order_type_ids < order_type_count)).to(torch.int32).unsqueeze(1)
    target_mode_loss_mask = target_mode_semantic_mask * ((target_mode_ids >= 0) & (target_mode_ids < target_mode_count)).to(torch.int32).unsqueeze(1)
    queue_flag_loss_mask = queue_flag_semantic_mask * ((queue_flag_tensor >= 0) & (queue_flag_tensor < 2)).to(torch.int32).unsqueeze(1)
    queue_update_type_loss_mask = queue_update_type_semantic_mask * ((queue_update_type_ids >= 0) & (queue_update_type_ids < queue_update_type_count)).to(torch.int32).unsqueeze(1)
    buildable_object_loss_mask = buildable_object_semantic_mask * (
        (buildable_object_tokens >= 0) & (buildable_object_tokens < buildable_object_vocab_size)
    ).to(torch.int32).unsqueeze(1)
    super_weapon_type_loss_mask = super_weapon_type_semantic_mask * (
        (super_weapon_type_ids >= 0) & (super_weapon_type_ids < super_weapon_type_count)
    ).to(torch.int32).unsqueeze(1)
    commanded_units_loss_mask = commanded_units_semantic_mask * commanded_units_sequence_mask * commanded_units_resolved_mask
    target_entity_loss_mask = target_entity_semantic_mask * target_entity_resolved
    target_location_loss_mask = target_location_semantic_mask * target_location_valid
    target_location_2_loss_mask = target_location_2_semantic_mask * target_location_2_valid
    quantity_loss_mask = quantity_semantic_mask * (quantity_value >= 0).to(torch.int32)

    training_targets = {
        "actionFamilyOneHot": action_family_one_hot,
        "delayOneHot": delay_one_hot,
        "orderTypeOneHot": order_type_one_hot,
        "targetModeOneHot": target_mode_one_hot,
        "queueFlagOneHot": queue_flag_one_hot,
        "queueUpdateTypeOneHot": queue_update_type_one_hot,
        "buildableObjectOneHot": buildable_object_one_hot,
        "superWeaponTypeOneHot": super_weapon_type_one_hot,
        "commandedUnitsOneHot": commanded_units_one_hot,
        "targetEntityOneHot": target_entity_one_hot,
        "targetLocationOneHot": target_location_one_hot,
        "targetLocation2OneHot": target_location_2_one_hot,
        "quantityValue": quantity_value,
    }
    training_masks = {
        "actionFamilyLossMask": action_family_loss_mask,
        "delayLossMask": delay_loss_mask,
        "orderTypeSemanticMask": order_type_semantic_mask,
        "targetModeSemanticMask": target_mode_semantic_mask,
        "queueFlagSemanticMask": queue_flag_semantic_mask,
        "queueUpdateTypeSemanticMask": queue_update_type_semantic_mask,
        "buildableObjectSemanticMask": buildable_object_semantic_mask,
        "superWeaponTypeSemanticMask": super_weapon_type_semantic_mask,
        "commandedUnitsSemanticMask": commanded_units_semantic_mask,
        "targetEntitySemanticMask": target_entity_semantic_mask,
        "targetLocationSemanticMask": target_location_semantic_mask,
        "targetLocation2SemanticMask": target_location_2_semantic_mask,
        "quantitySemanticMask": quantity_semantic_mask,
        "orderTypeLossMask": order_type_loss_mask,
        "targetModeLossMask": target_mode_loss_mask,
        "queueFlagLossMask": queue_flag_loss_mask,
        "queueUpdateTypeLossMask": queue_update_type_loss_mask,
        "buildableObjectLossMask": buildable_object_loss_mask,
        "superWeaponTypeLossMask": super_weapon_type_loss_mask,
        "commandedUnitsSequenceMask": commanded_units_sequence_mask,
        "commandedUnitsResolvedMask": commanded_units_resolved_mask,
        "commandedUnitsLossMask": commanded_units_loss_mask,
        "targetEntityResolvedMask": target_entity_resolved,
        "targetEntityLossMask": target_entity_loss_mask,
        "targetLocationValidMask": target_location_valid,
        "targetLocationLossMask": target_location_loss_mask,
        "targetLocation2ValidMask": target_location_2_valid,
        "targetLocation2LossMask": target_location_2_loss_mask,
        "quantityLossMask": quantity_loss_mask,
    }

    training_schema = {
        "version": TRAINING_TARGETS_V2_VERSION,
        "sourceCanonicalLabelVersion": LABEL_LAYOUT_V2_VERSION,
        "actionFamilyCount": len(LABEL_LAYOUT_V2_ACTION_FAMILIES),
        "delayBins": delay_bins,
        "orderTypeCount": order_type_count,
        "targetModeCount": target_mode_count,
        "queueUpdateTypeCount": queue_update_type_count,
        "buildableObjectVocabularySize": buildable_object_vocab_size,
        "superWeaponTypeCount": super_weapon_type_count,
        "maxEntities": max_entities,
        "maxCommandedUnits": max_commanded_units,
        "spatialSize": location_target_size,
        "observationSpatialSize": observation_spatial_size,
        "locationTargetSize": location_target_size,
        **build_training_target_sections_v2(
            action_family_count=len(LABEL_LAYOUT_V2_ACTION_FAMILIES),
            delay_bins=delay_bins,
            order_type_count=order_type_count,
            target_mode_count=target_mode_count,
            queue_update_type_count=queue_update_type_count,
            buildable_object_vocab_size=buildable_object_vocab_size,
            super_weapon_type_count=super_weapon_type_count,
            max_selected_units=max_commanded_units,
            max_entities=max_entities,
            location_target_size=location_target_size,
        ),
        "notes": [
            "This sidecar expands canonical V2 labels into hierarchical model-ready targets and masks.",
            "Standalone SelectUnitsAction is folded out before these targets are derived.",
            "Queue Hold and Resume are excluded from the current-stage V2 supervised action space.",
            "Buildable object supervision is shared by queue item actions and PlaceBuilding.",
        ],
    }

    replay_name = Path(str(metadata["replay"]["path"])).name
    player_name = str(metadata["playerName"])
    validate_tensor_section_shapes(
        training_targets,
        training_schema["targetSections"],
        "trainingTargetsV2",
        replay_name,
        player_name,
        sample_count,
    )
    validate_tensor_section_shapes(
        training_masks,
        training_schema["maskSections"],
        "trainingMasksV2",
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
        structured_v2_tensor_path = result.get("structuredV2TensorPath")
        training_v2_tensor_path = result.get("trainingTargetTensorPathV2")
        if structured_v2_tensor_path:
            structured_v2_path = Path(str(structured_v2_tensor_path))
            training_v2_path = (
                Path(str(training_v2_tensor_path))
                if training_v2_tensor_path
                else Path(str(result["tensorPath"])).with_suffix(".v2.training.pt")
            )
            should_write_training_v2 = config.overwrite or not training_v2_path.exists()
            if should_write_training_v2 and structured_v2_path.exists():
                structured_v2_payload = torch.load(structured_v2_path, map_location="cpu", weights_only=True)
                with metadata_file.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)

                training_targets_v2, training_masks_v2, training_schema_v2 = build_training_targets_v2(
                    structured_v2_payload["featureTensors"],
                    structured_v2_payload["labelTensors"],
                    metadata,
                )
                torch.save(
                    {
                        "trainingTargets": training_targets_v2,
                        "trainingMasks": training_masks_v2,
                        "sampleContext": structured_v2_payload.get("sampleContext", {}),
                    },
                    training_v2_path,
                )
                metadata["trainingTargetTensorPathV2"] = str(training_v2_path)
                metadata["trainingTargetsV2"] = training_schema_v2
                metadata["structuredTrainingTargetShapesV2"] = {
                    name: list(tensor.shape) for name, tensor in training_targets_v2.items()
                }
                metadata["structuredTrainingMaskShapesV2"] = {
                    name: list(tensor.shape) for name, tensor in training_masks_v2.items()
                }
                with metadata_file.open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

            result["trainingTargetTensorPathV2"] = str(training_v2_path)
