from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UnitsAutoregressiveTargets:
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    non_eof_mask: torch.Tensor
    eof_mask: torch.Tensor
    eof_index: int


def _one_hot_to_index(one_hot: torch.Tensor) -> torch.Tensor:
    return torch.argmax(one_hot.to(torch.float32), dim=-1).to(torch.long)


def build_units_autoregressive_targets(
    units_one_hot: torch.Tensor,
    units_loss_mask: torch.Tensor,
) -> UnitsAutoregressiveTargets:
    if units_one_hot.ndim < 3:
        raise ValueError(
            f"units_one_hot must have shape [..., max_selected_units, max_entities], got {tuple(units_one_hot.shape)}."
        )
    if units_loss_mask.shape != units_one_hot.shape[:-1]:
        raise ValueError(
            "units_loss_mask must match units_one_hot without the entity axis: "
            f"{tuple(units_loss_mask.shape)} vs {tuple(units_one_hot.shape[:-1])}."
        )

    *leading_shape, max_selected_units, max_entities = units_one_hot.shape
    eof_index = int(max_entities)

    unit_ids = _one_hot_to_index(units_one_hot)
    loss_mask = units_loss_mask.to(torch.bool)

    target_ids = torch.full(
        (*leading_shape, max_selected_units + 1),
        fill_value=eof_index,
        dtype=torch.long,
        device=units_one_hot.device,
    )
    target_mask = torch.zeros_like(target_ids, dtype=torch.bool)

    target_ids[..., :max_selected_units] = unit_ids
    target_mask[..., :max_selected_units] = loss_mask

    step_indices = torch.arange(max_selected_units, device=units_one_hot.device, dtype=torch.long)
    expand_shape = (1,) * (loss_mask.ndim - 1) + (max_selected_units,)
    step_indices = step_indices.view(expand_shape).expand_as(loss_mask)
    last_supervised_step = torch.where(loss_mask, step_indices, -torch.ones_like(step_indices)).amax(dim=-1)
    supervise_eof = last_supervised_step >= 0
    eof_positions = (last_supervised_step + 1).clamp(max=max_selected_units)

    flat_target_ids = target_ids.reshape(-1, max_selected_units + 1)
    flat_target_mask = target_mask.reshape(-1, max_selected_units + 1)
    flat_supervise_eof = supervise_eof.reshape(-1)
    flat_eof_positions = eof_positions.reshape(-1)
    if torch.any(flat_supervise_eof):
        selected_rows = torch.nonzero(flat_supervise_eof, as_tuple=False).squeeze(-1)
        flat_target_ids[selected_rows, flat_eof_positions[selected_rows]] = eof_index
        flat_target_mask[selected_rows, flat_eof_positions[selected_rows]] = True

    non_eof_mask = target_mask & (target_ids != eof_index)
    eof_mask = target_mask & (target_ids == eof_index)
    return UnitsAutoregressiveTargets(
        target_ids=target_ids,
        target_mask=target_mask,
        non_eof_mask=non_eof_mask,
        eof_mask=eof_mask,
        eof_index=eof_index,
    )


def summarize_selected_units(
    entity_embeddings: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    eof_index: int,
) -> torch.Tensor:
    if entity_embeddings.ndim != 3:
        raise ValueError(
            f"entity_embeddings must have shape [batch, max_entities, entity_dim], got {tuple(entity_embeddings.shape)}."
        )
    if selected_ids.ndim != 2 or selected_mask.ndim != 2:
        raise ValueError(
            "selected_ids and selected_mask must have shape [batch, sequence_length], "
            f"got {tuple(selected_ids.shape)} and {tuple(selected_mask.shape)}."
        )

    batch_size, max_entities, entity_dim = entity_embeddings.shape
    if selected_ids.shape[0] != batch_size or selected_mask.shape[0] != batch_size:
        raise ValueError("selected_ids and selected_mask batch size must match entity_embeddings.")

    safe_ids = torch.clamp(selected_ids, min=0, max=max_entities - 1)
    gathered = torch.gather(
        entity_embeddings,
        dim=1,
        index=safe_ids.unsqueeze(-1).expand(-1, -1, entity_dim),
    )

    non_eof_mask = selected_mask.to(torch.bool) & (selected_ids != eof_index)
    non_eof_mask_float = non_eof_mask.to(gathered.dtype).unsqueeze(-1)
    denom = torch.clamp(non_eof_mask_float.sum(dim=1), min=1.0)
    pooled = (gathered * non_eof_mask_float).sum(dim=1) / denom
    any_selected = non_eof_mask.any(dim=1, keepdim=True)
    return torch.where(any_selected, pooled, torch.zeros_like(pooled))
