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
) -> torch.Tensor:
    loss = F.cross_entropy(logits, targets, reduction="none", weight=class_weights)
    return _masked_mean(loss, mask)


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


def _masked_spatial_loss(logits: torch.Tensor, target_one_hot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_one_hot = _resize_spatial_target_one_hot(target_one_hot, logits.shape[-2:])
    flat_logits = logits.reshape(-1, logits.shape[-2] * logits.shape[-1])
    flat_targets = _one_hot_to_index(target_one_hot.reshape(-1, target_one_hot.shape[-2] * target_one_hot.shape[-1]))
    return _masked_classification_loss(flat_logits, flat_targets, mask.reshape(-1))


def _masked_spatial_accuracy(logits: torch.Tensor, target_one_hot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_one_hot = _resize_spatial_target_one_hot(target_one_hot, logits.shape[-2:])
    flat_logits = logits.reshape(-1, logits.shape[-2] * logits.shape[-1])
    flat_targets = _one_hot_to_index(target_one_hot.reshape(-1, target_one_hot.shape[-2] * target_one_hot.shape[-1]))
    return _masked_accuracy(flat_logits, flat_targets, mask.reshape(-1))


def _masked_quantity_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_pred = pred.squeeze(-1).reshape(-1)
    flat_target = target.to(torch.float32).squeeze(-1).reshape(-1)
    value_loss = F.smooth_l1_loss(flat_pred, flat_target, reduction="none")
    return _masked_mean(value_loss, mask.reshape(-1))


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


def compute_ra2_sl_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    action_type_class_weights: torch.Tensor | None = None,
) -> RA2SLLossOutput:
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
    )
    delay_loss = _masked_classification_loss(flat_delay_logits, flat_delay_targets, flat_delay_mask)
    queue_loss = _masked_classification_loss(flat_queue_logits, flat_queue_targets, flat_queue_mask)

    units_logits, units_target_flat, units_mask_flat = _flatten_logits_and_targets(
        outputs["unitsLogits"],
        units_autoregressive_targets.target_ids,
        units_autoregressive_targets.target_mask,
    )
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
        "actionType": action_type_loss,
        "delay": delay_loss,
        "queue": queue_loss,
        "units": units_loss,
        "targetEntity": target_entity_loss,
        "targetLocation": target_location_loss,
        "targetLocation2": target_location2_loss,
        "quantity": quantity_loss,
    }
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

    return RA2SLLossOutput(
        total_loss=total_loss,
        loss_by_head=loss_by_head,
        metrics=metrics,
    )
