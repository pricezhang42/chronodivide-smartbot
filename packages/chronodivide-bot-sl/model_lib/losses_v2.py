from __future__ import annotations

from typing import Any

import torch

from model_lib.losses import (
    RA2SLLossOutput,
    _build_predicted_units_targets,
    _flatten_class_targets,
    _flatten_logits_and_targets,
    _flatten_scalar_mask,
    _masked_accuracy,
    _masked_classification_accuracy_from_predictions,
    _masked_classification_loss,
    _masked_mean,
    _masked_quantity_accuracy,
    _masked_quantity_loss,
    _masked_sequence_exact_match,
    _masked_spatial_accuracy,
    _masked_spatial_loss,
    _one_hot_to_index,
    _spatial_indices,
)
from model_lib.units_autoregressive import build_units_autoregressive_targets


def compute_ra2_sl_v2_free_running_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    targets = batch["training_targets"]
    masks = batch["training_masks"]

    action_family_targets = _flatten_class_targets(targets["actionFamilyOneHot"])
    delay_targets = _flatten_class_targets(targets["delayOneHot"])
    order_type_targets = _flatten_class_targets(targets["orderTypeOneHot"])
    target_mode_targets = _flatten_class_targets(targets["targetModeOneHot"])
    queue_flag_targets = _flatten_class_targets(targets["queueFlagOneHot"])
    queue_update_type_targets = _flatten_class_targets(targets["queueUpdateTypeOneHot"])
    buildable_object_targets = _flatten_class_targets(targets["buildableObjectOneHot"])
    super_weapon_type_targets = _flatten_class_targets(targets["superWeaponTypeOneHot"])
    target_entity_targets = _flatten_class_targets(targets["targetEntityOneHot"])
    target_location_targets = _spatial_indices(
        targets["targetLocationOneHot"],
        outputs["targetLocationLogits"].shape[-2:],
    )
    target_location2_targets = _spatial_indices(
        targets["targetLocation2OneHot"],
        outputs["targetLocation2Logits"].shape[-2:],
    )
    quantity_targets = targets["quantityValue"].reshape(-1).to(torch.float32)

    action_family_mask = _flatten_scalar_mask(masks["actionFamilyLossMask"])
    delay_mask = _flatten_scalar_mask(masks["delayLossMask"])
    order_type_mask = _flatten_scalar_mask(masks["orderTypeLossMask"])
    target_mode_mask = _flatten_scalar_mask(masks["targetModeLossMask"])
    queue_flag_mask = _flatten_scalar_mask(masks["queueFlagLossMask"])
    queue_update_type_mask = _flatten_scalar_mask(masks["queueUpdateTypeLossMask"])
    buildable_object_mask = _flatten_scalar_mask(masks["buildableObjectLossMask"])
    super_weapon_type_mask = _flatten_scalar_mask(masks["superWeaponTypeLossMask"])
    target_entity_mask = _flatten_scalar_mask(masks["targetEntityLossMask"])
    target_location_mask = _flatten_scalar_mask(masks["targetLocationLossMask"])
    target_location2_mask = _flatten_scalar_mask(masks["targetLocation2LossMask"])
    quantity_mask = _flatten_scalar_mask(masks["quantityLossMask"])

    action_family_predictions = torch.argmax(
        outputs["actionFamilyLogits"].reshape(-1, outputs["actionFamilyLogits"].shape[-1]),
        dim=-1,
    )
    delay_predictions = torch.argmax(outputs["delayLogits"].reshape(-1, outputs["delayLogits"].shape[-1]), dim=-1)
    order_type_predictions = torch.argmax(
        outputs["orderTypeLogits"].reshape(-1, outputs["orderTypeLogits"].shape[-1]),
        dim=-1,
    )
    target_mode_predictions = torch.argmax(
        outputs["targetModeLogits"].reshape(-1, outputs["targetModeLogits"].shape[-1]),
        dim=-1,
    )
    queue_flag_predictions = torch.argmax(
        outputs["queueFlagLogits"].reshape(-1, outputs["queueFlagLogits"].shape[-1]),
        dim=-1,
    )
    queue_update_type_predictions = torch.argmax(
        outputs["queueUpdateTypeLogits"].reshape(-1, outputs["queueUpdateTypeLogits"].shape[-1]),
        dim=-1,
    )
    buildable_object_predictions = torch.argmax(
        outputs["buildableObjectLogits"].reshape(-1, outputs["buildableObjectLogits"].shape[-1]),
        dim=-1,
    )
    super_weapon_type_predictions = torch.argmax(
        outputs["superWeaponTypeLogits"].reshape(-1, outputs["superWeaponTypeLogits"].shape[-1]),
        dim=-1,
    )
    target_entity_predictions = torch.argmax(
        outputs["targetEntityLogits"].reshape(-1, outputs["targetEntityLogits"].shape[-1]),
        dim=-1,
    )
    target_location_predictions = torch.argmax(
        outputs["targetLocationLogits"].reshape(
            -1,
            outputs["targetLocationLogits"].shape[-2] * outputs["targetLocationLogits"].shape[-1],
        ),
        dim=-1,
    )
    target_location2_predictions = torch.argmax(
        outputs["targetLocation2Logits"].reshape(
            -1,
            outputs["targetLocation2Logits"].shape[-2] * outputs["targetLocation2Logits"].shape[-1],
        ),
        dim=-1,
    )
    quantity_predictions = torch.round(outputs["quantityPred"].reshape(-1))

    units_targets = build_units_autoregressive_targets(
        targets["commandedUnitsOneHot"],
        masks["commandedUnitsLossMask"],
    )
    predicted_units_target_ids, predicted_units_target_mask = _build_predicted_units_targets(
        outputs["commandedUnitsSelectedIds"],
        outputs["commandedUnitsSelectedMask"],
        eof_index=units_targets.eof_index,
    )
    flat_predicted_units_ids = predicted_units_target_ids.reshape(-1, predicted_units_target_ids.shape[-1])
    flat_target_units_ids = units_targets.target_ids.reshape(-1, units_targets.target_ids.shape[-1])
    flat_units_mask = units_targets.target_mask.reshape(-1, units_targets.target_mask.shape[-1]).to(torch.bool)
    commanded_units_token_accuracy = _masked_classification_accuracy_from_predictions(
        flat_predicted_units_ids.reshape(-1),
        flat_target_units_ids.reshape(-1),
        flat_units_mask.reshape(-1),
    )
    commanded_units_sequence_exact_match = _masked_sequence_exact_match(
        predicted_units_target_ids,
        predicted_units_target_mask,
        units_targets.target_ids,
        units_targets.target_mask,
    )

    row_count = action_family_targets.shape[0]
    full_command_match = torch.ones(row_count, dtype=torch.bool, device=action_family_predictions.device)
    full_command_match &= action_family_predictions == action_family_targets
    full_command_match &= (~delay_mask) | (delay_predictions == delay_targets)
    full_command_match &= (~order_type_mask) | (order_type_predictions == order_type_targets)
    full_command_match &= (~target_mode_mask) | (target_mode_predictions == target_mode_targets)
    full_command_match &= (~queue_flag_mask) | (queue_flag_predictions == queue_flag_targets)
    full_command_match &= (~queue_update_type_mask) | (
        queue_update_type_predictions == queue_update_type_targets
    )
    full_command_match &= (~buildable_object_mask) | (
        buildable_object_predictions == buildable_object_targets
    )
    full_command_match &= (~super_weapon_type_mask) | (
        super_weapon_type_predictions == super_weapon_type_targets
    )
    full_command_match &= (~target_entity_mask) | (target_entity_predictions == target_entity_targets)
    full_command_match &= (~target_location_mask) | (target_location_predictions == target_location_targets)
    full_command_match &= (~target_location2_mask) | (
        target_location2_predictions == target_location2_targets
    )
    full_command_match &= (~quantity_mask) | (quantity_predictions == quantity_targets)
    flat_units_sequence_match = (
        (flat_predicted_units_ids == flat_target_units_ids).logical_or(~flat_units_mask).all(dim=1)
        & (
            predicted_units_target_mask.reshape(-1, predicted_units_target_mask.shape[-1]).to(torch.bool)
            == flat_units_mask
        ).all(dim=1)
    )
    full_command_match &= flat_units_sequence_match

    return {
        "actionFamilyAccuracy": _masked_classification_accuracy_from_predictions(
            action_family_predictions,
            action_family_targets,
            action_family_mask,
        ),
        "delayAccuracy": _masked_classification_accuracy_from_predictions(
            delay_predictions,
            delay_targets,
            delay_mask,
        ),
        "orderTypeAccuracy": _masked_classification_accuracy_from_predictions(
            order_type_predictions,
            order_type_targets,
            order_type_mask,
        ),
        "targetModeAccuracy": _masked_classification_accuracy_from_predictions(
            target_mode_predictions,
            target_mode_targets,
            target_mode_mask,
        ),
        "queueFlagAccuracy": _masked_classification_accuracy_from_predictions(
            queue_flag_predictions,
            queue_flag_targets,
            queue_flag_mask,
        ),
        "queueUpdateTypeAccuracy": _masked_classification_accuracy_from_predictions(
            queue_update_type_predictions,
            queue_update_type_targets,
            queue_update_type_mask,
        ),
        "buildableObjectAccuracy": _masked_classification_accuracy_from_predictions(
            buildable_object_predictions,
            buildable_object_targets,
            buildable_object_mask,
        ),
        "superWeaponTypeAccuracy": _masked_classification_accuracy_from_predictions(
            super_weapon_type_predictions,
            super_weapon_type_targets,
            super_weapon_type_mask,
        ),
        "commandedUnitsTokenAccuracy": commanded_units_token_accuracy,
        "commandedUnitsSequenceExactMatch": commanded_units_sequence_exact_match,
        "targetEntityAccuracy": _masked_classification_accuracy_from_predictions(
            target_entity_predictions,
            target_entity_targets,
            target_entity_mask,
        ),
        "targetLocationAccuracy": _masked_classification_accuracy_from_predictions(
            target_location_predictions,
            target_location_targets,
            target_location_mask,
        ),
        "targetLocation2Accuracy": _masked_classification_accuracy_from_predictions(
            target_location2_predictions,
            target_location2_targets,
            target_location2_mask,
        ),
        "quantityAccuracy": _masked_classification_accuracy_from_predictions(
            quantity_predictions,
            quantity_targets,
            quantity_mask,
        ),
        "fullCommandExactMatch": _masked_mean(full_command_match.to(torch.float32), action_family_mask),
    }


def compute_ra2_sl_v2_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    action_family_class_weights: torch.Tensor | None = None,
) -> RA2SLLossOutput:
    targets = batch["training_targets"]
    masks = batch["training_masks"]

    action_family_targets = _one_hot_to_index(targets["actionFamilyOneHot"])
    delay_targets = _one_hot_to_index(targets["delayOneHot"])
    order_type_targets = _one_hot_to_index(targets["orderTypeOneHot"])
    target_mode_targets = _one_hot_to_index(targets["targetModeOneHot"])
    queue_flag_targets = _one_hot_to_index(targets["queueFlagOneHot"])
    queue_update_type_targets = _one_hot_to_index(targets["queueUpdateTypeOneHot"])
    buildable_object_targets = _one_hot_to_index(targets["buildableObjectOneHot"])
    super_weapon_type_targets = _one_hot_to_index(targets["superWeaponTypeOneHot"])
    target_entity_targets = _one_hot_to_index(targets["targetEntityOneHot"])

    action_family_mask = masks["actionFamilyLossMask"].squeeze(-1) > 0
    delay_mask = masks["delayLossMask"].squeeze(-1) > 0
    order_type_mask = masks["orderTypeLossMask"].squeeze(-1) > 0
    target_mode_mask = masks["targetModeLossMask"].squeeze(-1) > 0
    queue_flag_mask = masks["queueFlagLossMask"].squeeze(-1) > 0
    queue_update_type_mask = masks["queueUpdateTypeLossMask"].squeeze(-1) > 0
    buildable_object_mask = masks["buildableObjectLossMask"].squeeze(-1) > 0
    super_weapon_type_mask = masks["superWeaponTypeLossMask"].squeeze(-1) > 0
    target_entity_mask = masks["targetEntityLossMask"].squeeze(-1) > 0
    target_location_mask = masks["targetLocationLossMask"].squeeze(-1) > 0
    target_location2_mask = masks["targetLocation2LossMask"].squeeze(-1) > 0
    quantity_mask = masks["quantityLossMask"].squeeze(-1) > 0

    flat_action_family_logits, flat_action_family_targets, flat_action_family_mask = _flatten_logits_and_targets(
        outputs["actionFamilyLogits"],
        action_family_targets,
        action_family_mask,
    )
    flat_delay_logits, flat_delay_targets, flat_delay_mask = _flatten_logits_and_targets(
        outputs["delayLogits"],
        delay_targets,
        delay_mask,
    )
    flat_order_type_logits, flat_order_type_targets, flat_order_type_mask = _flatten_logits_and_targets(
        outputs["orderTypeLogits"],
        order_type_targets,
        order_type_mask,
    )
    flat_target_mode_logits, flat_target_mode_targets, flat_target_mode_mask = _flatten_logits_and_targets(
        outputs["targetModeLogits"],
        target_mode_targets,
        target_mode_mask,
    )
    flat_queue_flag_logits, flat_queue_flag_targets, flat_queue_flag_mask = _flatten_logits_and_targets(
        outputs["queueFlagLogits"],
        queue_flag_targets,
        queue_flag_mask,
    )
    flat_queue_update_type_logits, flat_queue_update_type_targets, flat_queue_update_type_mask = _flatten_logits_and_targets(
        outputs["queueUpdateTypeLogits"],
        queue_update_type_targets,
        queue_update_type_mask,
    )
    flat_buildable_object_logits, flat_buildable_object_targets, flat_buildable_object_mask = _flatten_logits_and_targets(
        outputs["buildableObjectLogits"],
        buildable_object_targets,
        buildable_object_mask,
    )
    flat_super_weapon_type_logits, flat_super_weapon_type_targets, flat_super_weapon_type_mask = _flatten_logits_and_targets(
        outputs["superWeaponTypeLogits"],
        super_weapon_type_targets,
        super_weapon_type_mask,
    )

    action_family_loss = _masked_classification_loss(
        flat_action_family_logits,
        flat_action_family_targets,
        flat_action_family_mask,
        class_weights=action_family_class_weights,
    )
    delay_loss = _masked_classification_loss(flat_delay_logits, flat_delay_targets, flat_delay_mask)
    order_type_loss = _masked_classification_loss(
        flat_order_type_logits,
        flat_order_type_targets,
        flat_order_type_mask,
    )
    target_mode_loss = _masked_classification_loss(
        flat_target_mode_logits,
        flat_target_mode_targets,
        flat_target_mode_mask,
    )
    queue_flag_loss = _masked_classification_loss(
        flat_queue_flag_logits,
        flat_queue_flag_targets,
        flat_queue_flag_mask,
    )
    queue_update_type_loss = _masked_classification_loss(
        flat_queue_update_type_logits,
        flat_queue_update_type_targets,
        flat_queue_update_type_mask,
    )
    buildable_object_loss = _masked_classification_loss(
        flat_buildable_object_logits,
        flat_buildable_object_targets,
        flat_buildable_object_mask,
    )
    super_weapon_type_loss = _masked_classification_loss(
        flat_super_weapon_type_logits,
        flat_super_weapon_type_targets,
        flat_super_weapon_type_mask,
    )

    commanded_units_targets = build_units_autoregressive_targets(
        targets["commandedUnitsOneHot"],
        masks["commandedUnitsLossMask"],
    )
    commanded_units_logits, commanded_units_target_flat, commanded_units_mask_flat = _flatten_logits_and_targets(
        outputs["commandedUnitsLogits"],
        commanded_units_targets.target_ids,
        commanded_units_targets.target_mask,
    )
    commanded_units_loss = _masked_classification_loss(
        commanded_units_logits,
        commanded_units_target_flat,
        commanded_units_mask_flat,
    )

    flat_target_entity_logits, flat_target_entity_targets, flat_target_entity_mask = _flatten_logits_and_targets(
        outputs["targetEntityLogits"],
        target_entity_targets,
        target_entity_mask,
    )
    target_entity_loss = _masked_classification_loss(
        flat_target_entity_logits,
        flat_target_entity_targets,
        flat_target_entity_mask,
    )
    target_location_loss = _masked_spatial_loss(
        outputs["targetLocationLogits"],
        targets["targetLocationOneHot"],
        target_location_mask,
    )
    target_location2_loss = _masked_spatial_loss(
        outputs["targetLocation2Logits"],
        targets["targetLocation2OneHot"],
        target_location2_mask,
    )
    quantity_loss = _masked_quantity_loss(outputs["quantityPred"], targets["quantityValue"], quantity_mask)

    loss_by_head = {
        "actionFamily": action_family_loss,
        "delay": delay_loss,
        "orderType": order_type_loss,
        "targetMode": target_mode_loss,
        "queueFlag": queue_flag_loss,
        "queueUpdateType": queue_update_type_loss,
        "buildableObject": buildable_object_loss,
        "superWeaponType": super_weapon_type_loss,
        "commandedUnits": commanded_units_loss,
        "targetEntity": target_entity_loss,
        "targetLocation": target_location_loss,
        "targetLocation2": target_location2_loss,
        "quantity": quantity_loss,
    }
    total_loss = sum(loss_by_head.values())

    metrics = {
        "actionFamilyAccuracy": _masked_accuracy(
            flat_action_family_logits,
            flat_action_family_targets,
            flat_action_family_mask,
        ),
        "delayAccuracy": _masked_accuracy(flat_delay_logits, flat_delay_targets, flat_delay_mask),
        "orderTypeAccuracy": _masked_accuracy(
            flat_order_type_logits,
            flat_order_type_targets,
            flat_order_type_mask,
        ),
        "targetModeAccuracy": _masked_accuracy(
            flat_target_mode_logits,
            flat_target_mode_targets,
            flat_target_mode_mask,
        ),
        "queueFlagAccuracy": _masked_accuracy(
            flat_queue_flag_logits,
            flat_queue_flag_targets,
            flat_queue_flag_mask,
        ),
        "queueUpdateTypeAccuracy": _masked_accuracy(
            flat_queue_update_type_logits,
            flat_queue_update_type_targets,
            flat_queue_update_type_mask,
        ),
        "buildableObjectAccuracy": _masked_accuracy(
            flat_buildable_object_logits,
            flat_buildable_object_targets,
            flat_buildable_object_mask,
        ),
        "superWeaponTypeAccuracy": _masked_accuracy(
            flat_super_weapon_type_logits,
            flat_super_weapon_type_targets,
            flat_super_weapon_type_mask,
        ),
        "commandedUnitsAccuracy": _masked_accuracy(
            commanded_units_logits,
            commanded_units_target_flat,
            commanded_units_mask_flat,
        ),
        "targetEntityAccuracy": _masked_accuracy(
            flat_target_entity_logits,
            flat_target_entity_targets,
            flat_target_entity_mask,
        ),
        "targetLocationAccuracy": _masked_spatial_accuracy(
            outputs["targetLocationLogits"],
            targets["targetLocationOneHot"],
            target_location_mask,
        ),
        "targetLocation2Accuracy": _masked_spatial_accuracy(
            outputs["targetLocation2Logits"],
            targets["targetLocation2OneHot"],
            target_location2_mask,
        ),
        "quantityAccuracy": _masked_quantity_accuracy(
            outputs["quantityPred"],
            targets["quantityValue"],
            quantity_mask,
        ),
    }

    return RA2SLLossOutput(
        total_loss=total_loss,
        loss_by_head=loss_by_head,
        metrics=metrics,
    )
