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
    TRAINING_TARGETS_V1_VERSION,
    TransformConfig,
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
