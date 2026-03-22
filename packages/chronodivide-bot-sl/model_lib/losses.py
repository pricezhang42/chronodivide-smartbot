from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from model_lib.units_autoregressive import build_units_autoregressive_targets


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(values.dtype)
    denom = torch.clamp(mask_float.sum(), min=1.0)
    return (values * mask_float).sum() / denom


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Weighted masked mean. Falls back to _masked_mean when sample_weights is None."""
    if sample_weights is None:
        return _masked_mean(values, mask)
    mask_float = mask.to(values.dtype)
    weights = sample_weights.to(values.dtype) * mask_float
    denom = torch.clamp(weights.sum(), min=1.0)
    return (values * weights).sum() / denom


def _one_hot_to_index(one_hot: torch.Tensor) -> torch.Tensor:
    return torch.argmax(one_hot.to(torch.float32), dim=-1)


def _flatten_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_count = int(logits.shape[-1])
    return (
        logits.reshape(-1, class_count),
        targets.reshape(-1),
        mask.reshape(-1),
    )


def _masked_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, targets, reduction="none", weight=class_weights)
    return _masked_weighted_mean(loss, mask, sample_weights)


def _masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).to(torch.float32)
    return _masked_mean(correct, mask)


def _resize_spatial_target_one_hot(target_one_hot: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    source_height = int(target_one_hot.shape[-2])
    source_width = int(target_one_hot.shape[-1])
    target_height, target_width = int(target_hw[0]), int(target_hw[1])
    if (source_height, source_width) == (target_height, target_width):
        return target_one_hot

    flat_source = target_one_hot.reshape(-1, source_height * source_width)
    source_indices = torch.argmax(flat_source.to(torch.float32), dim=-1)
    source_y = source_indices // source_width
    source_x = source_indices % source_width

    scaled_y = torch.clamp((source_y * target_height) // source_height, min=0, max=target_height - 1)
    scaled_x = torch.clamp((source_x * target_width) // source_width, min=0, max=target_width - 1)

    resized = torch.zeros(
        flat_source.shape[0],
        target_height * target_width,
        dtype=target_one_hot.dtype,
        device=target_one_hot.device,
    )
    resized_indices = scaled_y * target_width + scaled_x
    resized.scatter_(1, resized_indices.unsqueeze(-1), 1)
    return resized.reshape(*target_one_hot.shape[:-2], target_height, target_width)


def _masked_spatial_loss(
    logits: torch.Tensor,
    target_one_hot: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    target_one_hot = _resize_spatial_target_one_hot(target_one_hot, logits.shape[-2:])
    flat_logits = logits.reshape(-1, logits.shape[-2] * logits.shape[-1])
    flat_targets = _one_hot_to_index(target_one_hot.reshape(-1, target_one_hot.shape[-2] * target_one_hot.shape[-1]))
    return _masked_classification_loss(flat_logits, flat_targets, mask.reshape(-1), sample_weights=sample_weights)


def _masked_spatial_accuracy(logits: torch.Tensor, target_one_hot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_one_hot = _resize_spatial_target_one_hot(target_one_hot, logits.shape[-2:])
    flat_logits = logits.reshape(-1, logits.shape[-2] * logits.shape[-1])
    flat_targets = _one_hot_to_index(target_one_hot.reshape(-1, target_one_hot.shape[-2] * target_one_hot.shape[-1]))
    return _masked_accuracy(flat_logits, flat_targets, mask.reshape(-1))


def _masked_quantity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_pred = pred.squeeze(-1).reshape(-1)
    flat_target = target.to(torch.float32).squeeze(-1).reshape(-1)
    value_loss = F.smooth_l1_loss(flat_pred, flat_target, reduction="none")
    return _masked_weighted_mean(value_loss, mask.reshape(-1), sample_weights)


def _masked_quantity_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rounded_pred = torch.round(pred.squeeze(-1).reshape(-1))
    flat_target = target.to(torch.float32).squeeze(-1).reshape(-1)
    correct = (rounded_pred == flat_target).to(torch.float32)
    return _masked_mean(correct, mask.reshape(-1))


@dataclass
class RA2SLLossOutput:
    total_loss: torch.Tensor
    loss_by_head: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]


def _flatten_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(-1, *tensor.shape[2:]) if tensor.ndim >= 3 else tensor.reshape(-1, tensor.shape[-1])


def _flatten_scalar_mask(mask: torch.Tensor) -> torch.Tensor:
    flattened = mask.reshape(-1)
    return flattened.to(torch.bool)


def _flatten_class_targets(one_hot: torch.Tensor) -> torch.Tensor:
    return _one_hot_to_index(one_hot).reshape(-1)


def _spatial_indices(one_hot: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    if target_hw is not None:
        one_hot = _resize_spatial_target_one_hot(one_hot, target_hw)
    flat = one_hot.reshape(-1, one_hot.shape[-2] * one_hot.shape[-1])
    return _one_hot_to_index(flat)


def _build_predicted_units_targets(
    selected_ids: torch.Tensor,
    selected_non_eof_mask: torch.Tensor,
    *,
    eof_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if selected_ids.shape != selected_non_eof_mask.shape:
        raise ValueError(
            f"selected_ids and selected_non_eof_mask must match, got {tuple(selected_ids.shape)} vs {tuple(selected_non_eof_mask.shape)}."
        )
    *leading_shape, sequence_length = selected_ids.shape
    flat_selected_ids = selected_ids.reshape(-1, sequence_length)
    flat_non_eof_mask = selected_non_eof_mask.to(torch.bool).reshape(-1, sequence_length)
    flat_target_ids = torch.full_like(flat_selected_ids, fill_value=eof_index)
    flat_target_mask = torch.zeros_like(flat_non_eof_mask)
    counts = flat_non_eof_mask.sum(dim=1, dtype=torch.long)
    for row_index in range(flat_selected_ids.shape[0]):
        count = int(counts[row_index].item())
        if count > 0:
            flat_target_ids[row_index, :count] = flat_selected_ids[row_index, :count]
        if count < sequence_length:
            flat_target_ids[row_index, count] = eof_index
            flat_target_mask[row_index, : count + 1] = True
    return (
        flat_target_ids.reshape(*leading_shape, sequence_length),
        flat_target_mask.reshape(*leading_shape, sequence_length),
    )


def _masked_sequence_exact_match(
    predicted_ids: torch.Tensor,
    predicted_mask: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    flat_predicted_ids = predicted_ids.reshape(-1, predicted_ids.shape[-1])
    flat_predicted_mask = predicted_mask.to(torch.bool).reshape(-1, predicted_mask.shape[-1])
    flat_target_ids = target_ids.reshape(-1, target_ids.shape[-1])
    flat_target_mask = target_mask.to(torch.bool).reshape(-1, target_mask.shape[-1])
    row_active = flat_target_mask.any(dim=1)
    if not bool(row_active.any()):
        return torch.tensor(1.0, device=predicted_ids.device, dtype=torch.float32)
    ids_match = flat_predicted_ids == flat_target_ids
    mask_match = flat_predicted_mask == flat_target_mask
    row_match = ids_match.logical_or(~flat_target_mask).all(dim=1) & mask_match.all(dim=1)
    return row_match[row_active].to(torch.float32).mean()


def _masked_classification_accuracy_from_predictions(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    correct = (predicted_ids == target_ids).to(torch.float32)
    return _masked_mean(correct, mask)


def compute_ra2_sl_free_running_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    targets = batch["training_targets"]
    masks = batch["training_masks"]
    scalar_sections = batch["model_inputs"]["scalar_sections"]

    action_type_targets = _flatten_class_targets(targets["actionTypeOneHot"])
    delay_targets = _flatten_class_targets(targets["delayOneHot"])
    queue_targets = _flatten_class_targets(targets["queueOneHot"])
    target_entity_targets = _flatten_class_targets(targets["targetEntityOneHot"])
    target_location_targets = _spatial_indices(targets["targetLocationOneHot"], outputs["targetLocationLogits"].shape[-2:])
    target_location2_targets = _spatial_indices(targets["targetLocation2OneHot"], outputs["targetLocation2Logits"].shape[-2:])
    quantity_targets = targets["quantityValue"].reshape(-1).to(torch.float32)

    action_type_mask = _flatten_scalar_mask(masks["actionTypeLossMask"])
    delay_mask = _flatten_scalar_mask(masks["delayLossMask"])
    queue_mask = _flatten_scalar_mask(masks["queueLossMask"])
    target_entity_mask = _flatten_scalar_mask(masks["targetEntityLossMask"])
    target_location_mask = _flatten_scalar_mask(masks["targetLocationLossMask"])
    target_location2_mask = _flatten_scalar_mask(masks["targetLocation2LossMask"])
    quantity_mask = _flatten_scalar_mask(masks["quantityLossMask"])

    action_type_predictions = torch.argmax(outputs["actionTypeLogits"].reshape(-1, outputs["actionTypeLogits"].shape[-1]), dim=-1)
    delay_predictions = torch.argmax(outputs["delayLogits"].reshape(-1, outputs["delayLogits"].shape[-1]), dim=-1)
    queue_predictions = torch.argmax(outputs["queueLogits"].reshape(-1, outputs["queueLogits"].shape[-1]), dim=-1)
    target_entity_predictions = torch.argmax(
        outputs["targetEntityLogits"].reshape(-1, outputs["targetEntityLogits"].shape[-1]),
        dim=-1,
    )
    target_location_predictions = torch.argmax(
        outputs["targetLocationLogits"].reshape(-1, outputs["targetLocationLogits"].shape[-2] * outputs["targetLocationLogits"].shape[-1]),
        dim=-1,
    )
    target_location2_predictions = torch.argmax(
        outputs["targetLocation2Logits"].reshape(-1, outputs["targetLocation2Logits"].shape[-2] * outputs["targetLocation2Logits"].shape[-1]),
        dim=-1,
    )
    quantity_predictions = torch.round(outputs["quantityPred"].reshape(-1))

    units_targets = build_units_autoregressive_targets(targets["unitsOneHot"], masks["unitsLossMask"])
    predicted_units_target_ids, predicted_units_target_mask = _build_predicted_units_targets(
        outputs["unitsSelectedIds"],
        outputs["unitsSelectedMask"],
        eof_index=units_targets.eof_index,
    )
    flat_predicted_units_ids = predicted_units_target_ids.reshape(-1, predicted_units_target_ids.shape[-1])
    flat_target_units_ids = units_targets.target_ids.reshape(-1, units_targets.target_ids.shape[-1])
    flat_units_mask = units_targets.target_mask.reshape(-1, units_targets.target_mask.shape[-1]).to(torch.bool)
    units_token_accuracy = _masked_classification_accuracy_from_predictions(
        flat_predicted_units_ids.reshape(-1),
        flat_target_units_ids.reshape(-1),
        flat_units_mask.reshape(-1),
    )
    units_sequence_exact_match = _masked_sequence_exact_match(
        predicted_units_target_ids,
        predicted_units_target_mask,
        units_targets.target_ids,
        units_targets.target_mask,
    )

    available_action_mask = scalar_sections.get("availableActionMask")
    gold_action_suppressed_rate = torch.tensor(0.0, device=action_type_predictions.device, dtype=torch.float32)
    if available_action_mask is not None:
        flat_available_action_mask = available_action_mask.reshape(-1, available_action_mask.shape[-1])
        valid_rows = torch.nonzero(action_type_mask, as_tuple=False).reshape(-1)
        if valid_rows.numel() > 0:
            suppressed = flat_available_action_mask[valid_rows, action_type_targets[valid_rows]] <= 0
            gold_action_suppressed_rate = suppressed.to(torch.float32).mean()

    row_count = action_type_targets.shape[0]
    full_action_match = torch.ones(row_count, dtype=torch.bool, device=action_type_predictions.device)
    full_action_match &= action_type_predictions == action_type_targets
    full_action_match &= (~delay_mask) | (delay_predictions == delay_targets)
    full_action_match &= (~queue_mask) | (queue_predictions == queue_targets)
    full_action_match &= (~target_entity_mask) | (target_entity_predictions == target_entity_targets)
    full_action_match &= (~target_location_mask) | (target_location_predictions == target_location_targets)
    full_action_match &= (~target_location2_mask) | (target_location2_predictions == target_location2_targets)
    full_action_match &= (~quantity_mask) | (quantity_predictions == quantity_targets)
    flat_units_sequence_match = (
        (
            flat_predicted_units_ids == flat_target_units_ids
        ).logical_or(~flat_units_mask).all(dim=1)
        & (predicted_units_target_mask.reshape(-1, predicted_units_target_mask.shape[-1]).to(torch.bool) == flat_units_mask).all(dim=1)
    )
    full_action_match &= flat_units_sequence_match

    metrics = {
        "actionTypeAccuracy": _masked_classification_accuracy_from_predictions(
            action_type_predictions,
            action_type_targets,
            action_type_mask,
        ),
        "delayAccuracy": _masked_classification_accuracy_from_predictions(
            delay_predictions,
            delay_targets,
            delay_mask,
        ),
        "queueAccuracy": _masked_classification_accuracy_from_predictions(
            queue_predictions,
            queue_targets,
            queue_mask,
        ),
        "unitsTokenAccuracy": units_token_accuracy,
        "unitsSequenceExactMatch": units_sequence_exact_match,
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
        "fullActionExactMatch": _masked_mean(full_action_match.to(torch.float32), action_type_mask),
        "goldActionSuppressedRate": gold_action_suppressed_rate,
    }
    return metrics


def compute_ra2_sl_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    action_type_class_weights: torch.Tensor | None = None,
    pseudo_reward_config: "PseudoRewardConfig | None" = None,
    composition_aux_scale: float = 0.0,
) -> RA2SLLossOutput:
    """Compute RA2 supervised learning loss with optional pseudo-reward weighting.

    When pseudo_reward_config is provided and enabled, per-sample importance
    weights are computed based on action type (production actions get higher
    weight, Noop gets lower weight).

    When composition_aux_scale > 0 and the model outputs compositionPred,
    a composition prediction auxiliary loss is added (Hamming-distance inspired).
    """
    targets = batch["training_targets"]
    masks = batch["training_masks"]

    action_type_targets = _one_hot_to_index(targets["actionTypeOneHot"])
    delay_targets = _one_hot_to_index(targets["delayOneHot"])
    queue_targets = _one_hot_to_index(targets["queueOneHot"])
    target_entity_targets = _one_hot_to_index(targets["targetEntityOneHot"])

    action_type_mask = masks["actionTypeLossMask"].squeeze(-1) > 0
    delay_mask = masks["delayLossMask"].squeeze(-1) > 0
    queue_mask = masks["queueLossMask"].squeeze(-1) > 0
    units_autoregressive_targets = build_units_autoregressive_targets(
        targets["unitsOneHot"],
        masks["unitsLossMask"],
    )
    target_entity_mask = masks["targetEntityLossMask"].squeeze(-1) > 0
    target_location_mask = masks["targetLocationLossMask"].squeeze(-1) > 0
    target_location2_mask = masks["targetLocation2LossMask"].squeeze(-1) > 0
    quantity_mask = masks["quantityLossMask"].squeeze(-1) > 0

    # --- Compute per-sample importance weights (pseudo-reward weighting) ---
    sample_weights: torch.Tensor | None = None
    if pseudo_reward_config is not None and pseudo_reward_config.enabled:
        from model_lib.pseudo_reward import compute_sample_importance_weights
        sample_weights = compute_sample_importance_weights(
            batch,
            pseudo_reward_config,
            noop_family_index=0,
        )
        # Ensure sample_weights is on the right device and flattened.
        device = action_type_targets.device
        sample_weights = sample_weights.to(device)

    flat_action_type_logits, flat_action_type_targets, flat_action_type_mask = _flatten_logits_and_targets(
        outputs["actionTypeLogits"],
        action_type_targets,
        action_type_mask,
    )
    flat_delay_logits, flat_delay_targets, flat_delay_mask = _flatten_logits_and_targets(
        outputs["delayLogits"],
        delay_targets,
        delay_mask,
    )
    flat_queue_logits, flat_queue_targets, flat_queue_mask = _flatten_logits_and_targets(
        outputs["queueLogits"],
        queue_targets,
        queue_mask,
    )
    action_type_loss = _masked_classification_loss(
        flat_action_type_logits,
        flat_action_type_targets,
        flat_action_type_mask,
        class_weights=action_type_class_weights,
        sample_weights=sample_weights,
    )
    delay_loss = _masked_classification_loss(
        flat_delay_logits, flat_delay_targets, flat_delay_mask,
        sample_weights=sample_weights,
    )
    queue_loss = _masked_classification_loss(
        flat_queue_logits, flat_queue_targets, flat_queue_mask,
        sample_weights=sample_weights,
    )

    units_logits, units_target_flat, units_mask_flat = _flatten_logits_and_targets(
        outputs["unitsLogits"],
        units_autoregressive_targets.target_ids,
        units_autoregressive_targets.target_mask,
    )
    # Note: units loss has a different sample count (expanded by sequence steps),
    # so we don't apply sample_weights directly here to avoid shape mismatch.
    units_loss = _masked_classification_loss(units_logits, units_target_flat, units_mask_flat)

    flat_target_entity_logits, flat_target_entity_targets, flat_target_entity_mask = _flatten_logits_and_targets(
        outputs["targetEntityLogits"],
        target_entity_targets,
        target_entity_mask,
    )
    target_entity_loss = _masked_classification_loss(
        flat_target_entity_logits,
        flat_target_entity_targets,
        flat_target_entity_mask,
        sample_weights=sample_weights,
    )
    target_location_loss = _masked_spatial_loss(
        outputs["targetLocationLogits"],
        targets["targetLocationOneHot"],
        target_location_mask,
        sample_weights=sample_weights,
    )
    target_location2_loss = _masked_spatial_loss(
        outputs["targetLocation2Logits"],
        targets["targetLocation2OneHot"],
        target_location2_mask,
        sample_weights=sample_weights,
    )
    quantity_loss = _masked_quantity_loss(
        outputs["quantityPred"], targets["quantityValue"], quantity_mask,
        sample_weights=sample_weights,
    )

    loss_by_head = {
        "actionType": action_type_loss,
        "delay": delay_loss,
        "queue": queue_loss,
        "units": units_loss,
        "targetEntity": target_entity_loss,
        "targetLocation": target_location_loss,
        "targetLocation2": target_location2_loss,
        "quantity": quantity_loss,
    }

    # --- Auxiliary losses: composition prediction (Hamming-distance inspired) ---
    if (
        composition_aux_scale > 0
        and "compositionPred" in outputs
        and "ownedCompositionBow" in batch.get("feature_sections", {})
    ):
        from model_lib.pseudo_reward import composition_prediction_loss

        composition_target = batch["feature_sections"]["ownedCompositionBow"]
        # ownedCompositionBow is [batch, (seq_len,) 2, vocab_size] — flatten rows to [N, vocab_size]
        # Sum units + buildings rows to get total composition.
        if composition_target.ndim >= 3:
            composition_target = composition_target.sum(dim=-2)  # [..., vocab_size]
        composition_pred = outputs["compositionPred"]
        # Create a mask: all samples are valid for composition prediction.
        comp_mask = torch.ones(
            composition_pred.reshape(-1, composition_pred.shape[-1]).shape[0],
            dtype=torch.bool,
            device=composition_pred.device,
        )
        comp_loss = composition_prediction_loss(composition_pred, composition_target, comp_mask)
        loss_by_head["compositionAux"] = comp_loss * composition_aux_scale

    total_loss = sum(loss_by_head.values())

    metrics = {
        "actionTypeAccuracy": _masked_accuracy(flat_action_type_logits, flat_action_type_targets, flat_action_type_mask),
        "delayAccuracy": _masked_accuracy(flat_delay_logits, flat_delay_targets, flat_delay_mask),
        "queueAccuracy": _masked_accuracy(flat_queue_logits, flat_queue_targets, flat_queue_mask),
        "unitsAccuracy": _masked_accuracy(units_logits, units_target_flat, units_mask_flat),
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
    if sample_weights is not None:
        metrics["pseudoRewardAvgWeight"] = sample_weights.mean()
    if "compositionAux" in loss_by_head:
        metrics["compositionAuxLoss"] = loss_by_head["compositionAux"].detach()

    return RA2SLLossOutput(
        total_loss=total_loss,
        loss_by_head=loss_by_head,
        metrics=metrics,
    )


# Type import for annotation only.
if False:
    from model_lib.pseudo_reward import PseudoRewardConfig
