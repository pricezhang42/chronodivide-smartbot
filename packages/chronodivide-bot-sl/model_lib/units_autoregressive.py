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

    flat_unit_ids = unit_ids.reshape(-1, max_selected_units)
    flat_loss_mask = loss_mask.reshape(-1, max_selected_units)
    flat_target_ids = torch.full(
        (flat_unit_ids.shape[0], max_selected_units + 1),
        fill_value=eof_index,
        dtype=torch.long,
        device=units_one_hot.device,
    )
    flat_target_mask = torch.zeros_like(flat_target_ids, dtype=torch.bool)

    supervised_counts = flat_loss_mask.sum(dim=1, dtype=torch.long)
    for row_index in range(flat_unit_ids.shape[0]):
        row_mask = flat_loss_mask[row_index]
        row_count = int(supervised_counts[row_index].item())
        if row_count <= 0:
            continue
        packed_unit_ids = flat_unit_ids[row_index][row_mask]
        flat_target_ids[row_index, :row_count] = packed_unit_ids
        flat_target_mask[row_index, :row_count] = True
        flat_target_ids[row_index, row_count] = eof_index
        flat_target_mask[row_index, row_count] = True

    target_ids = flat_target_ids.reshape(*leading_shape, max_selected_units + 1)
    target_mask = flat_target_mask.reshape(*leading_shape, max_selected_units + 1)

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
