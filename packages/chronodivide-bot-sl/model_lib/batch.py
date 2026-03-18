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
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in keys
    }


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
