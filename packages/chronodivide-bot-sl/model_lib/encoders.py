from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(sequence.dtype).unsqueeze(-1)
    summed = torch.sum(sequence * mask_float, dim=1)
    counts = torch.clamp(torch.sum(mask_float, dim=1), min=1.0)
    return summed / counts


class SectionMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        flattened = tensor.to(torch.float32).reshape(tensor.shape[0], -1)
        return self.net(flattened)


class BuildOrderTraceEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, build_order_trace: torch.Tensor) -> torch.Tensor:
        safe_ids = torch.clamp(build_order_trace.to(torch.int64) + 1, min=0)
        valid_mask = build_order_trace >= 0
        embedded = self.embedding(safe_ids)
        pooled = masked_mean(embedded, valid_mask)
        return self.proj(pooled)


@dataclass
class ScalarEncoderConfig:
    build_order_vocab_size: int
    hidden_dim: int = 64
    output_dim: int = 128
    dropout: float = 0.1
    section_order: tuple[str, ...] = (
        "scalar",
        "lastActionContext",
        "currentSelectionCount",
        "currentSelectionResolvedCount",
        "currentSelectionOverflowCount",
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


class ScalarEncoder(nn.Module):
    def __init__(self, config: ScalarEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.generic_sections = tuple(
            section_name
            for section_name in config.section_order
            if section_name != "buildOrderTrace"
        )
        self.section_mlps = nn.ModuleDict(
            {
                section_name: SectionMLP(config.hidden_dim, config.dropout)
                for section_name in self.generic_sections
            }
        )
        self.build_order_encoder = BuildOrderTraceEncoder(
            vocab_size=config.build_order_vocab_size,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.output_proj = nn.Sequential(
            nn.LazyLinear(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
            nn.GELU(),
        )

    def forward(self, scalar_sections: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        section_embeddings: dict[str, torch.Tensor] = {}
        ordered_embeddings: list[torch.Tensor] = []
        for section_name in self.config.section_order:
            if section_name not in scalar_sections:
                continue
            if section_name == "buildOrderTrace":
                section_embedding = self.build_order_encoder(scalar_sections[section_name])
            else:
                section_embedding = self.section_mlps[section_name](scalar_sections[section_name])
            section_embeddings[section_name] = section_embedding
            ordered_embeddings.append(section_embedding)

        if not ordered_embeddings:
            raise ValueError("ScalarEncoder received no known scalar sections.")

        concatenated = torch.cat(ordered_embeddings, dim=-1)
        output = self.output_proj(concatenated)
        return {
            "section_embeddings": section_embeddings,
            "concat_embedding": concatenated,
            "pooled": output,
        }


@dataclass
class EntityEncoderConfig:
    entity_name_vocab_size: int
    feature_dim: int = 74
    name_embedding_dim: int = 32
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class EntityEncoder(nn.Module):
    def __init__(self, config: EntityEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_proj = nn.Sequential(
            nn.Linear(config.feature_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.name_embedding = nn.Embedding(config.entity_name_vocab_size + 1, config.name_embedding_dim, padding_idx=0)
        self.name_proj = nn.Linear(config.name_embedding_dim, config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        *,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        entity_name_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_mask = entity_mask > 0
        safe_name_tokens = torch.where(
            valid_mask,
            torch.clamp(entity_name_tokens.to(torch.int64), min=0) + 1,
            torch.zeros_like(entity_name_tokens.to(torch.int64)),
        )
        feature_embedding = self.feature_proj(entity_features.to(torch.float32))
        name_embedding = self.name_proj(self.name_embedding(safe_name_tokens))
        entity_embedding = feature_embedding + name_embedding
        entity_embedding = entity_embedding.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        encoded = self.transformer(entity_embedding, src_key_padding_mask=~valid_mask)
        encoded = self.output_norm(encoded)
        pooled = masked_mean(encoded, valid_mask)
        return {
            "per_entity": encoded,
            "pooled": pooled,
            "mask": valid_mask,
        }


@dataclass
class SpatialEncoderConfig:
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.1


class SpatialEncoder(nn.Module):
    def __init__(self, config: SpatialEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.LazyConv2d(config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim // 2, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        *,
        spatial: torch.Tensor,
        minimap: torch.Tensor,
        map_static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = minimap.shape[-2:]
        resized_spatial = F.interpolate(
            spatial.to(torch.float32),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        resized_map_static = F.interpolate(
            map_static.to(torch.float32),
            size=target_size,
            mode="nearest",
        )
        stacked = torch.cat(
            [
                resized_spatial,
                minimap.to(torch.float32),
                resized_map_static,
            ],
            dim=1,
        )
        feature_map = self.trunk(stacked)
        pooled = torch.mean(feature_map, dim=(2, 3))
        output = self.output_proj(pooled)
        return {
            "feature_map": feature_map,
            "pooled": output,
        }


def pool_selected_entity_embeddings(
    entity_embeddings: torch.Tensor,
    entity_mask: torch.Tensor,
    selection_indices: torch.Tensor,
    selection_resolved_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_entities, embedding_dim = entity_embeddings.shape
    safe_indices = torch.clamp(selection_indices.to(torch.int64), min=0, max=max_entities - 1)
    gathered = torch.gather(
        entity_embeddings,
        dim=1,
        index=safe_indices.unsqueeze(-1).expand(batch_size, safe_indices.shape[1], embedding_dim),
    )
    selected_valid = (
        (selection_resolved_mask > 0)
        & (selection_indices >= 0)
        & torch.gather(entity_mask > 0, dim=1, index=safe_indices)
    )
    return masked_mean(gathered, selected_valid)
