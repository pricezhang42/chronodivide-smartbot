from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from action_dict import ACTION_TYPE_ID_TO_NAME
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
        self.slot_embeddings = nn.Embedding(max_selected_units, hidden_dim)
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
        batch_size = latent.shape[0]
        slot_ids = torch.arange(self.max_selected_units, device=latent.device)
        slot_embed = self.slot_embeddings(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)
        base_query = self.query_proj(latent).unsqueeze(1)
        query = base_query + slot_embed
        entity_keys = self.entity_proj(entity_embeddings)
        logits = torch.einsum("bsd,bed->bse", query, entity_keys) / self.scale
        return _masked_logits(logits, entity_mask.unsqueeze(1) > 0)


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
    def __init__(self, latent_dim: int, map_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
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
            dropout=config.dropout,
        )
        self.target_location2_head = SpatialLocationHead(
            latent_dim=config.fusion_dim,
            map_channels=config.spatial_map_channels,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.quantity_head = QuantityHead(
            input_dim=config.fusion_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        *,
        fused_latent: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_mask: torch.Tensor,
        spatial_feature_map: torch.Tensor,
        available_action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "actionTypeLogits": self.action_type_head(fused_latent, available_action_mask=available_action_mask),
            "delayLogits": self.delay_head(fused_latent),
            "queueLogits": self.queue_head(fused_latent),
            "unitsLogits": self.units_head(fused_latent, entity_embeddings, entity_mask),
            "targetEntityLogits": self.target_entity_head(fused_latent, entity_embeddings, entity_mask),
            "targetLocationLogits": self.target_location_head(fused_latent, spatial_feature_map),
            "targetLocation2Logits": self.target_location2_head(fused_latent, spatial_feature_map),
            "quantityPred": self.quantity_head(fused_latent),
        }
