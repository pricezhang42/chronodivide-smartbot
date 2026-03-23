from __future__ import annotations

from typing import Any

import torch


SCALAR_FEATURE_SECTION_NAMES = (
    "scalar",
    "lastActionContext",
    "currentSelectionCount",
    "currentSelectionResolvedCount",
    "currentSelectionOverflowCount",
    "currentSelectionIndices",
    "currentSelectionMask",
    "currentSelectionResolvedMask",
    "currentSelectionSummary",
    "availableActionMask",
    "ownedCompositionBow",
    "enemyMemoryBow",
    "enemyMemoryTechFlags",
    "buildOrderTrace",
    "techState",
    "productionState",
    "superWeaponState",
    "timeEncoding",
    "gameStats",
)

ENTITY_FEATURE_SECTION_NAMES = (
    "entityFeatures",
    "entityMask",
    "entityNameTokens",
)

SPATIAL_FEATURE_SECTION_NAMES = (
    "spatial",
    "minimap",
    "mapStatic",
)


def _stack_tensor_dict(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        return {}
    keys = samples[0].keys()
    return {key: _stack_tensor_list([sample[key] for sample in samples]) for key in keys}


def _stack_tensor_list(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("_stack_tensor_list requires a non-empty tensor list.")
    reference_shape = tuple(tensors[0].shape)
    if all(tuple(tensor.shape) == reference_shape for tensor in tensors):
        return torch.stack(tensors, dim=0)

    max_shape = [max(int(tensor.shape[dim]) for tensor in tensors) for dim in range(tensors[0].ndim)]
    padded_tensors = [_pad_tensor_to_shape(tensor, max_shape) for tensor in tensors]
    return torch.stack(padded_tensors, dim=0)


def _pad_tensor_to_shape(tensor: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
    if list(tensor.shape) == target_shape:
        return tensor
    padded = torch.zeros(
        *target_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    slices = tuple(slice(0, int(size)) for size in tensor.shape)
    padded[slices] = tensor
    return padded


def build_model_inputs(feature_sections: dict[str, torch.Tensor]) -> dict[str, Any]:
    scalar_sections = {
        name: feature_sections[name]
        for name in SCALAR_FEATURE_SECTION_NAMES
        if name in feature_sections
    }
    entity_inputs = {
        "features": feature_sections["entityFeatures"],
        "mask": feature_sections["entityMask"],
    }
    if "entityNameTokens" in feature_sections:
        entity_inputs["name_tokens"] = feature_sections["entityNameTokens"]

    spatial_inputs = {
        "spatial": feature_sections["spatial"],
        "minimap": feature_sections["minimap"],
        "map_static": feature_sections["mapStatic"],
    }

    return {
        "scalar_sections": scalar_sections,
        "entity": entity_inputs,
        "spatial": spatial_inputs,
    }


def collate_model_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_model_samples requires a non-empty sample list.")

    feature_sections = _stack_tensor_dict([sample["feature_sections"] for sample in samples])
    label_sections = _stack_tensor_dict([sample["label_sections"] for sample in samples])
    training_targets = _stack_tensor_dict([sample["training_targets"] for sample in samples])
    training_masks = _stack_tensor_dict([sample["training_masks"] for sample in samples])
    sample_context = _stack_tensor_dict([sample["sample_context"] for sample in samples])
    metadata = [sample["metadata"] for sample in samples]

    return {
        "feature_sections": feature_sections,
        "label_sections": label_sections,
        "training_targets": training_targets,
        "training_masks": training_masks,
        "sample_context": sample_context,
        "metadata": metadata,
        "model_inputs": build_model_inputs(feature_sections),
    }
