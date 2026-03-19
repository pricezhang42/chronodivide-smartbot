from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_dict import ACTION_INFO_MASK, ACTION_TYPE_ID_TO_NAME
from model_lib.units_autoregressive import (
    UnitsAutoregressiveTargets,
    build_units_autoregressive_targets,
    summarize_selected_units,
)
from transform_lib.common import LABEL_LAYOUT_V1_DELAY_BINS


def _masked_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(torch.bool)
    return logits.masked_fill(~mask, -1e9)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def _one_hot_to_index(one_hot: torch.Tensor) -> torch.Tensor:
    return torch.argmax(one_hot.to(torch.float32), dim=-1)


def _resolve_condition_ids(
    logits: torch.Tensor,
    *,
    teacher_forcing_one_hot: torch.Tensor | None = None,
    teacher_forcing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted_ids = torch.argmax(logits, dim=-1)
    if teacher_forcing_one_hot is not None:
        teacher_ids = _one_hot_to_index(teacher_forcing_one_hot)
        if teacher_forcing_mask is None:
            return teacher_ids
        mask = teacher_forcing_mask.to(torch.bool)
        while mask.ndim > teacher_ids.ndim:
            mask = mask.squeeze(-1)
        while mask.ndim < teacher_ids.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, teacher_ids, predicted_ids)
    return predicted_ids


def _is_teacher_forcing_enabled(mode: str, stage: str) -> bool:
    if mode == "none":
        return False
    if mode == "action_type":
        return stage == "action_type"
    if mode == "action_type_queue":
        return stage in {"action_type", "queue"}
    if mode == "full":
        return stage in {
            "action_type",
            "delay",
            "queue",
            "units",
            "target_entity",
            "target_location",
            "target_location2",
        }
    raise ValueError(f"Unsupported teacher-forcing mode: {mode}")


def _blend_teacher_forced_weights(
    predicted_weights: torch.Tensor,
    *,
    teacher_forcing_one_hot: torch.Tensor | None = None,
    teacher_forcing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if teacher_forcing_one_hot is None:
        return predicted_weights
    teacher_weights = teacher_forcing_one_hot.to(predicted_weights.dtype)
    if teacher_forcing_mask is None:
        return teacher_weights
    mask = teacher_forcing_mask.to(torch.bool)
    while mask.ndim < teacher_weights.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, teacher_weights, predicted_weights)


def _resize_spatial_teacher_one_hot(teacher_one_hot: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    source_height = int(teacher_one_hot.shape[-2])
    source_width = int(teacher_one_hot.shape[-1])
    target_height, target_width = int(target_hw[0]), int(target_hw[1])
    if (source_height, source_width) == (target_height, target_width):
        return teacher_one_hot

    flat_source = teacher_one_hot.reshape(-1, source_height * source_width)
    source_indices = torch.argmax(flat_source.to(torch.float32), dim=-1)
    source_y = source_indices // source_width
    source_x = source_indices % source_width

    scaled_y = torch.clamp((source_y * target_height) // source_height, min=0, max=target_height - 1)
    scaled_x = torch.clamp((source_x * target_width) // source_width, min=0, max=target_width - 1)

    resized = torch.zeros(
        flat_source.shape[0],
        target_height * target_width,
        dtype=teacher_one_hot.dtype,
        device=teacher_one_hot.device,
    )
    resized_indices = scaled_y * target_width + scaled_x
    resized.scatter_(1, resized_indices.unsqueeze(-1), 1)
    return resized.reshape(*teacher_one_hot.shape[:-2], target_height, target_width)


def _pool_target_entity_condition(
    entity_embeddings: torch.Tensor,
    target_entity_logits: torch.Tensor,
    *,
    teacher_forcing_one_hot: torch.Tensor | None = None,
    teacher_forcing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted_weights = torch.softmax(target_entity_logits, dim=-1)
    weights = _blend_teacher_forced_weights(
        predicted_weights,
        teacher_forcing_one_hot=teacher_forcing_one_hot,
        teacher_forcing_mask=teacher_forcing_mask,
    )
    pooled = torch.einsum("be,bed->bd", weights, entity_embeddings)
    if teacher_forcing_mask is not None:
        mask = teacher_forcing_mask.to(pooled.dtype)
        while mask.ndim > pooled.ndim:
            mask = mask.squeeze(-1)
        while mask.ndim < pooled.ndim:
            mask = mask.unsqueeze(-1)
        pooled = pooled * mask
    return pooled


def _pool_spatial_condition(
    spatial_feature_map: torch.Tensor,
    spatial_logits: torch.Tensor,
    *,
    teacher_forcing_one_hot: torch.Tensor | None = None,
    teacher_forcing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted_weights = torch.softmax(spatial_logits.reshape(spatial_logits.shape[0], -1), dim=-1)
    teacher_weights = None
    if teacher_forcing_one_hot is not None:
        resized_teacher = _resize_spatial_teacher_one_hot(teacher_forcing_one_hot, spatial_logits.shape[-2:])
        teacher_weights = resized_teacher.reshape(resized_teacher.shape[0], -1)
    weights = _blend_teacher_forced_weights(
        predicted_weights,
        teacher_forcing_one_hot=teacher_weights,
        teacher_forcing_mask=teacher_forcing_mask,
    )
    flat_feature_map = spatial_feature_map.reshape(
        spatial_feature_map.shape[0],
        spatial_feature_map.shape[1],
        -1,
    )
    pooled = torch.einsum("bn,bcn->bc", weights, flat_feature_map)
    if teacher_forcing_mask is not None:
        mask = teacher_forcing_mask.to(pooled.dtype)
        while mask.ndim > pooled.ndim:
            mask = mask.squeeze(-1)
        while mask.ndim < pooled.ndim:
            mask = mask.unsqueeze(-1)
        pooled = pooled * mask
    return pooled


class ActionTypeHead(nn.Module):
    def __init__(self, input_dim: int, action_vocab_size: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = MLPHead(input_dim, hidden_dim, action_vocab_size, dropout)

    def forward(self, latent: torch.Tensor, available_action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.head(latent)
        if available_action_mask is not None:
            logits = _masked_logits(logits, available_action_mask > 0)
        return logits


class DelayHead(nn.Module):
    def __init__(self, input_dim: int, delay_bins: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = MLPHead(input_dim, hidden_dim, delay_bins, dropout)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)


class QueueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = MLPHead(input_dim, hidden_dim, 2, dropout)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)


class UnitsHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        entity_dim: int,
        max_selected_units: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_selected_units = max_selected_units
        self.num_steps = max_selected_units + 1
        self.step_embeddings = nn.Embedding(self.num_steps, hidden_dim)
        self.base_query_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_init = nn.Linear(latent_dim, hidden_dim)
        self.cell_init = nn.Linear(latent_dim, hidden_dim)
        self.query_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
        self.prev_choice_proj = nn.Linear(entity_dim, hidden_dim)
        self.eof_key = nn.Parameter(torch.zeros(hidden_dim))
        self.eof_value = nn.Parameter(torch.zeros(hidden_dim))
        self.scale = math.sqrt(float(hidden_dim))
        nn.init.normal_(self.eof_key, std=0.02)
        nn.init.normal_(self.eof_value, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        teacher_forcing_targets: UnitsAutoregressiveTargets | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_entities, _ = entity_embeddings.shape
        eof_index = max_entities

        entity_mask_bool = entity_mask.to(torch.bool)
        remaining_entity_mask = entity_mask_bool.clone()

        entity_keys = self.entity_proj(entity_embeddings)
        entity_values = self.prev_choice_proj(entity_embeddings)
        eof_key = self.eof_key.view(1, 1, -1).expand(batch_size, -1, -1)
        eof_value = self.eof_value.view(1, 1, -1).expand(batch_size, -1, -1)
        candidate_keys = torch.cat([entity_keys, eof_key], dim=1)
        candidate_values = torch.cat([entity_values, eof_value], dim=1)

        base_query = self.base_query_proj(latent)
        hidden = torch.tanh(self.hidden_init(latent))
        cell = torch.tanh(self.cell_init(latent))
        previous_choice = torch.zeros_like(hidden)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=latent.device)

        logits_per_step: list[torch.Tensor] = []
        selected_ids_per_step: list[torch.Tensor] = []
        selected_non_eof_mask_per_step: list[torch.Tensor] = []
        batch_index = torch.arange(batch_size, device=latent.device)

        teacher_ids = None
        teacher_mask = None
        if teacher_forcing_targets is not None:
            teacher_ids = teacher_forcing_targets.target_ids
            teacher_mask = teacher_forcing_targets.target_mask
            if teacher_ids.shape != (batch_size, self.num_steps):
                raise ValueError(
                    "Teacher-forced unit ids must match the autoregressive sequence shape: "
                    f"{tuple(teacher_ids.shape)} vs {(batch_size, self.num_steps)}."
                )

        for step_index in range(self.num_steps):
            step_ids = torch.full((batch_size,), step_index, dtype=torch.long, device=latent.device)
            step_input = base_query + self.step_embeddings(step_ids) + previous_choice
            hidden, cell = self.query_cell(step_input, (hidden, cell))

            raw_logits = torch.einsum("bd,bed->be", hidden, candidate_keys) / self.scale
            allow_eof = torch.ones(batch_size, 1, dtype=torch.bool, device=latent.device)
            if step_index == 0:
                allow_eof = ~remaining_entity_mask.any(dim=1, keepdim=True)
            step_candidate_mask = torch.cat([remaining_entity_mask, allow_eof], dim=1)
            if torch.any(finished):
                eof_only_mask = torch.zeros_like(step_candidate_mask)
                eof_only_mask[:, eof_index] = True
                step_candidate_mask = torch.where(finished.unsqueeze(-1), eof_only_mask, step_candidate_mask)
            step_logits = _masked_logits(raw_logits, step_candidate_mask)

            predicted_ids = torch.argmax(step_logits, dim=-1)
            if teacher_ids is not None and teacher_mask is not None:
                chosen_ids = torch.where(teacher_mask[:, step_index], teacher_ids[:, step_index], predicted_ids)
            else:
                chosen_ids = predicted_ids

            active_before_choice = ~finished
            chosen_non_eof_mask = active_before_choice & (chosen_ids != eof_index)
            if torch.any(chosen_non_eof_mask):
                selected_rows = torch.nonzero(chosen_non_eof_mask, as_tuple=False).squeeze(-1)
                remaining_entity_mask[selected_rows, chosen_ids[selected_rows]] = False

            previous_choice = candidate_values[batch_index, chosen_ids]
            finished = finished | (chosen_ids == eof_index)

            logits_per_step.append(step_logits.unsqueeze(1))
            selected_ids_per_step.append(chosen_ids.unsqueeze(1))
            selected_non_eof_mask_per_step.append(chosen_non_eof_mask.unsqueeze(1))

        selected_ids = torch.cat(selected_ids_per_step, dim=1)
        selected_non_eof_mask = torch.cat(selected_non_eof_mask_per_step, dim=1)
        units_summary = summarize_selected_units(
            entity_embeddings,
            selected_ids,
            selected_non_eof_mask,
            eof_index=eof_index,
        )
        return {
            "logits": torch.cat(logits_per_step, dim=1),
            "selectedIds": selected_ids,
            "selectedMask": selected_non_eof_mask,
            "summary": units_summary,
        }


class TargetEntityHead(nn.Module):
    def __init__(self, latent_dim: int, entity_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
        self.scale = math.sqrt(float(hidden_dim))

    def forward(
        self,
        latent: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(latent)
        entity_keys = self.entity_proj(entity_embeddings)
        logits = torch.einsum("bd,bed->be", query, entity_keys) / self.scale
        return _masked_logits(logits, entity_mask > 0)


class SpatialLocationHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        map_channels: int,
        hidden_dim: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_size = int(output_size)
        self.gate_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, map_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, spatial_feature_map: torch.Tensor) -> torch.Tensor:
        if spatial_feature_map.shape[-2:] != (self.output_size, self.output_size):
            spatial_feature_map = F.interpolate(
                spatial_feature_map,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        gate = self.gate_proj(latent).unsqueeze(-1).unsqueeze(-1)
        gated_map = spatial_feature_map + gate
        return self.conv(gated_map).squeeze(1)


class QuantityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = MLPHead(input_dim, hidden_dim, 1, dropout)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)


@dataclass
class RA2SLHeadsConfig:
    action_vocab_size: int = len(ACTION_TYPE_ID_TO_NAME)
    delay_bins: int = LABEL_LAYOUT_V1_DELAY_BINS
    max_selected_units: int = 64
    max_entities: int = 128
    spatial_size: int = 32
    fusion_dim: int = 256
    entity_dim: int = 128
    spatial_map_channels: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1


class RA2SLPredictionHeads(nn.Module):
    def __init__(self, config: RA2SLHeadsConfig) -> None:
        super().__init__()
        self.config = config
        self.action_type_head = ActionTypeHead(
            input_dim=config.fusion_dim,
            action_vocab_size=config.action_vocab_size,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.action_type_condition_embedding = nn.Embedding(config.action_vocab_size, config.fusion_dim)
        self.delay_condition_embedding = nn.Embedding(config.delay_bins, config.fusion_dim)
        self.delay_head = DelayHead(
            input_dim=config.fusion_dim,
            delay_bins=config.delay_bins,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.queue_head = QueueHead(
            input_dim=config.fusion_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.queue_condition_embedding = nn.Embedding(2, config.fusion_dim)
        self.units_condition_proj = nn.Linear(config.entity_dim, config.fusion_dim)
        self.target_entity_condition_proj = nn.Linear(config.entity_dim, config.fusion_dim)
        self.target_location_condition_proj = nn.Linear(config.spatial_map_channels, config.fusion_dim)
        self.target_location2_condition_proj = nn.Linear(config.spatial_map_channels, config.fusion_dim)
        self.units_head = UnitsHead(
            latent_dim=config.fusion_dim,
            entity_dim=config.entity_dim,
            max_selected_units=config.max_selected_units,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.target_entity_head = TargetEntityHead(
            latent_dim=config.fusion_dim,
            entity_dim=config.entity_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.target_location_head = SpatialLocationHead(
            latent_dim=config.fusion_dim,
            map_channels=config.spatial_map_channels,
            hidden_dim=config.hidden_dim,
            output_size=config.spatial_size,
            dropout=config.dropout,
        )
        self.target_location2_head = SpatialLocationHead(
            latent_dim=config.fusion_dim,
            map_channels=config.spatial_map_channels,
            hidden_dim=config.hidden_dim,
            output_size=config.spatial_size,
            dropout=config.dropout,
        )
        self.quantity_head = QuantityHead(
            input_dim=config.fusion_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.condition_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "action_type_uses_queue_mask",
            torch.tensor(
                [bool(ACTION_INFO_MASK[action_type_id]["usesQueue"]) for action_type_id in range(config.action_vocab_size)],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_type_uses_units_mask",
            torch.tensor(
                [bool(ACTION_INFO_MASK[action_type_id]["usesUnits"]) for action_type_id in range(config.action_vocab_size)],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_type_uses_target_entity_mask",
            torch.tensor(
                [
                    bool(ACTION_INFO_MASK[action_type_id]["usesTargetEntity"])
                    for action_type_id in range(config.action_vocab_size)
                ],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_type_uses_target_location_mask",
            torch.tensor(
                [
                    bool(ACTION_INFO_MASK[action_type_id]["usesTargetLocation"])
                    for action_type_id in range(config.action_vocab_size)
                ],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_type_uses_target_location2_mask",
            torch.tensor(
                [
                    bool(ACTION_INFO_MASK[action_type_id]["usesTargetLocation2"])
                    for action_type_id in range(config.action_vocab_size)
                ],
                dtype=torch.bool,
            ),
            persistent=False,
        )

    def forward(
        self,
        *,
        fused_latent: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_mask: torch.Tensor,
        spatial_feature_map: torch.Tensor,
        available_action_mask: torch.Tensor | None = None,
        teacher_forcing_targets: dict[str, torch.Tensor] | None = None,
        teacher_forcing_masks: dict[str, torch.Tensor] | None = None,
        teacher_forcing_mode: str = "none",
    ) -> dict[str, torch.Tensor]:
        autoregressive_latent = fused_latent

        action_type_logits = self.action_type_head(autoregressive_latent, available_action_mask=available_action_mask)
        action_type_teacher = (
            None
            if teacher_forcing_targets is None or not _is_teacher_forcing_enabled(teacher_forcing_mode, "action_type")
            else teacher_forcing_targets.get("actionTypeOneHot")
        )
        action_type_ids = _resolve_condition_ids(
            action_type_logits,
            teacher_forcing_one_hot=action_type_teacher,
            teacher_forcing_mask=None if teacher_forcing_masks is None else teacher_forcing_masks.get("actionTypeLossMask"),
        )
        action_type_condition = self.condition_dropout(self.action_type_condition_embedding(action_type_ids))
        autoregressive_latent = autoregressive_latent + action_type_condition

        delay_logits = self.delay_head(autoregressive_latent)
        delay_teacher = (
            None
            if teacher_forcing_targets is None or not _is_teacher_forcing_enabled(teacher_forcing_mode, "delay")
            else teacher_forcing_targets.get("delayOneHot")
        )
        delay_ids = _resolve_condition_ids(
            delay_logits,
            teacher_forcing_one_hot=delay_teacher,
            teacher_forcing_mask=None if teacher_forcing_masks is None else teacher_forcing_masks.get("delayLossMask"),
        )
        delay_condition = self.condition_dropout(self.delay_condition_embedding(delay_ids))
        autoregressive_latent = autoregressive_latent + delay_condition

        queue_logits = self.queue_head(autoregressive_latent)

        queue_teacher = None
        if teacher_forcing_targets is not None and _is_teacher_forcing_enabled(teacher_forcing_mode, "queue"):
            queue_teacher = teacher_forcing_targets.get("queueOneHot")
        queue_ids = _resolve_condition_ids(
            queue_logits,
            teacher_forcing_one_hot=queue_teacher,
            teacher_forcing_mask=None if teacher_forcing_masks is None else teacher_forcing_masks.get("queueLossMask"),
        )
        queue_condition = self.condition_dropout(self.queue_condition_embedding(queue_ids))
        queue_semantic_mask = self.action_type_uses_queue_mask[action_type_ids].to(queue_condition.dtype).unsqueeze(-1)
        autoregressive_latent = autoregressive_latent + queue_condition * queue_semantic_mask

        units_teacher_targets = None
        units_teacher_enabled = teacher_forcing_targets is not None and _is_teacher_forcing_enabled(
            teacher_forcing_mode, "units"
        )
        if units_teacher_enabled and teacher_forcing_masks is not None:
            raw_units_teacher = teacher_forcing_targets.get("unitsOneHot")
            raw_units_teacher_mask = teacher_forcing_masks.get("unitsLossMask")
            if raw_units_teacher is not None and raw_units_teacher_mask is not None:
                units_teacher_targets = build_units_autoregressive_targets(
                    raw_units_teacher,
                    raw_units_teacher_mask,
                )
        units_outputs = self.units_head(
            autoregressive_latent,
            entity_embeddings,
            entity_mask,
            teacher_forcing_targets=units_teacher_targets,
        )
        units_logits = units_outputs["logits"]
        units_summary = units_outputs["summary"]
        units_condition = self.condition_dropout(self.units_condition_proj(units_summary))
        units_semantic_mask = self.action_type_uses_units_mask[action_type_ids].to(units_condition.dtype).unsqueeze(-1)
        autoregressive_latent = autoregressive_latent + units_condition * units_semantic_mask

        target_entity_logits = self.target_entity_head(autoregressive_latent, entity_embeddings, entity_mask)
        target_entity_teacher_enabled = teacher_forcing_targets is not None and _is_teacher_forcing_enabled(
            teacher_forcing_mode, "target_entity"
        )
        target_entity_teacher = (
            None if not target_entity_teacher_enabled else teacher_forcing_targets.get("targetEntityOneHot")
        )
        target_entity_teacher_mask = (
            None
            if not target_entity_teacher_enabled or teacher_forcing_masks is None
            else teacher_forcing_masks.get("targetEntityLossMask")
        )
        target_entity_summary = _pool_target_entity_condition(
            entity_embeddings,
            target_entity_logits,
            teacher_forcing_one_hot=target_entity_teacher,
            teacher_forcing_mask=target_entity_teacher_mask,
        )
        target_entity_condition = self.condition_dropout(self.target_entity_condition_proj(target_entity_summary))
        target_entity_semantic_mask = self.action_type_uses_target_entity_mask[action_type_ids].to(
            target_entity_condition.dtype
        ).unsqueeze(-1)
        autoregressive_latent = autoregressive_latent + target_entity_condition * target_entity_semantic_mask

        target_location_logits = self.target_location_head(autoregressive_latent, spatial_feature_map)
        target_location_teacher_enabled = teacher_forcing_targets is not None and _is_teacher_forcing_enabled(
            teacher_forcing_mode, "target_location"
        )
        target_location_teacher = (
            None if not target_location_teacher_enabled else teacher_forcing_targets.get("targetLocationOneHot")
        )
        target_location_teacher_mask = (
            None
            if not target_location_teacher_enabled or teacher_forcing_masks is None
            else teacher_forcing_masks.get("targetLocationLossMask")
        )
        target_location_summary = _pool_spatial_condition(
            spatial_feature_map,
            target_location_logits,
            teacher_forcing_one_hot=target_location_teacher,
            teacher_forcing_mask=target_location_teacher_mask,
        )
        target_location_condition = self.condition_dropout(self.target_location_condition_proj(target_location_summary))
        target_location_semantic_mask = self.action_type_uses_target_location_mask[action_type_ids].to(
            target_location_condition.dtype
        ).unsqueeze(-1)
        autoregressive_latent = autoregressive_latent + target_location_condition * target_location_semantic_mask

        target_location2_logits = self.target_location2_head(autoregressive_latent, spatial_feature_map)
        target_location2_teacher_enabled = teacher_forcing_targets is not None and _is_teacher_forcing_enabled(
            teacher_forcing_mode, "target_location2"
        )
        target_location2_teacher = (
            None if not target_location2_teacher_enabled else teacher_forcing_targets.get("targetLocation2OneHot")
        )
        target_location2_teacher_mask = (
            None
            if not target_location2_teacher_enabled or teacher_forcing_masks is None
            else teacher_forcing_masks.get("targetLocation2LossMask")
        )
        target_location2_summary = _pool_spatial_condition(
            spatial_feature_map,
            target_location2_logits,
            teacher_forcing_one_hot=target_location2_teacher,
            teacher_forcing_mask=target_location2_teacher_mask,
        )
        target_location2_condition = self.condition_dropout(
            self.target_location2_condition_proj(target_location2_summary)
        )
        target_location2_semantic_mask = self.action_type_uses_target_location2_mask[action_type_ids].to(
            target_location2_condition.dtype
        ).unsqueeze(-1)
        autoregressive_latent = autoregressive_latent + target_location2_condition * target_location2_semantic_mask

        return {
            "actionTypeLogits": action_type_logits,
            "delayLogits": delay_logits,
            "queueLogits": queue_logits,
            "unitsLogits": units_logits,
            "unitsSelectedIds": units_outputs["selectedIds"],
            "unitsSelectedMask": units_outputs["selectedMask"],
            "targetEntityLogits": target_entity_logits,
            "targetLocationLogits": target_location_logits,
            "targetLocation2Logits": target_location2_logits,
            "quantityPred": self.quantity_head(autoregressive_latent),
        }
