from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(sequence.dtype).unsqueeze(-1)
    summed = torch.sum(sequence * mask_float, dim=1)
    counts = torch.clamp(torch.sum(mask_float, dim=1), min=1.0)
    return summed / counts


# ---------------------------------------------------------------------------
# Bucketed one-hot helpers
# ---------------------------------------------------------------------------

def sqrt_bucket(value: torch.Tensor, num_bins: int, max_value: float) -> torch.Tensor:
    """Bucket continuous values via sqrt scaling into one-hot vectors.

    Maps [0, max_value] → [0, num_bins-1] via sqrt, then one-hot encodes.
    """
    clamped = torch.clamp(value, min=0.0, max=max_value)
    scaled = torch.sqrt(clamped) / math.sqrt(max_value) * (num_bins - 1)
    indices = torch.clamp(scaled.to(torch.int64), min=0, max=num_bins - 1)
    return F.one_hot(indices, num_classes=num_bins).to(value.dtype)


def linear_bucket(value: torch.Tensor, num_bins: int, max_value: float) -> torch.Tensor:
    """Bucket continuous values linearly into one-hot vectors."""
    clamped = torch.clamp(value, min=0.0, max=max_value)
    scaled = clamped / max_value * (num_bins - 1)
    indices = torch.clamp(scaled.to(torch.int64), min=0, max=num_bins - 1)
    return F.one_hot(indices, num_classes=num_bins).to(value.dtype)


# ---------------------------------------------------------------------------
# Entity feature preprocessor — expands raw continuous columns into buckets
# ---------------------------------------------------------------------------

# Entity feature indices from features.mjs ENTITY_FEATURE_NAMES.
# These are stable across all data versions.
_ENTITY_HIT_POINTS_IDX = 14
_ENTITY_MAX_HIT_POINTS_IDX = 15
_ENTITY_PURCHASE_VALUE_IDX = 19
_ENTITY_PRIMARY_COOLDOWN_IDX = 56
_ENTITY_SECONDARY_COOLDOWN_IDX = 57

# Bucket sizes for each expanded feature.
_HP_BINS = 20           # sqrt bucketed, max 1500
_HP_MAX = 1500.0
_PURCHASE_BINS = 16     # sqrt bucketed, max 3000
_PURCHASE_MAX = 3000.0
_COOLDOWN_BINS = 16     # linear bucketed, max 120 ticks (~4s)
_COOLDOWN_MAX = 120.0

# Columns to replace: (raw_index, num_bins).
# Order matters — processed left-to-right, replacements happen in output order.
_BUCKET_SPECS: list[tuple[int, int]] = [
    (_ENTITY_HIT_POINTS_IDX, _HP_BINS),
    (_ENTITY_MAX_HIT_POINTS_IDX, _HP_BINS),
    (_ENTITY_PURCHASE_VALUE_IDX, _PURCHASE_BINS),
    (_ENTITY_PRIMARY_COOLDOWN_IDX, _COOLDOWN_BINS),
    (_ENTITY_SECONDARY_COOLDOWN_IDX, _COOLDOWN_BINS),
]

# Set of raw indices that get replaced (removed from passthrough).
_BUCKET_RAW_INDICES = frozenset(idx for idx, _ in _BUCKET_SPECS)

# Total expansion: each replaced column goes from 1 dim → num_bins dims.
_BUCKET_EXTRA_DIMS = sum(bins - 1 for _, bins in _BUCKET_SPECS)


def _compute_expanded_feature_dim(raw_dim: int) -> int:
    return raw_dim + _BUCKET_EXTRA_DIMS


class EntityFeaturePreprocessor(nn.Module):
    """Replaces selected continuous entity columns with bucketed one-hot vectors.

    Input:  [batch, max_entities, raw_dim]  (e.g. 74)
    Output: [batch, max_entities, expanded_dim]  (e.g. 74 + 83 = 157)
    """

    def __init__(self, raw_feature_dim: int) -> None:
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.expanded_dim = _compute_expanded_feature_dim(raw_feature_dim)

        # Precompute which raw columns pass through unchanged.
        self._passthrough_indices = [
            i for i in range(raw_feature_dim) if i not in _BUCKET_RAW_INDICES
        ]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        f = features.to(torch.float32)

        # Passthrough columns (all the one-hot flags, ratios, etc.)
        parts: list[torch.Tensor] = [f[..., self._passthrough_indices]]

        # Bucketed expansions.
        for raw_idx, num_bins in _BUCKET_SPECS:
            raw_col = f[..., raw_idx]
            if raw_idx in (_ENTITY_PRIMARY_COOLDOWN_IDX, _ENTITY_SECONDARY_COOLDOWN_IDX):
                parts.append(linear_bucket(raw_col, num_bins, _COOLDOWN_MAX))
            else:
                max_val = _HP_MAX if num_bins == _HP_BINS else _PURCHASE_MAX
                parts.append(sqrt_bucket(raw_col, num_bins, max_val))

        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Action context encoder — replaces generic SectionMLP for lastActionContext
# ---------------------------------------------------------------------------

_DELAY_BINS = 32
_DELAY_MAX = 1800.0  # ~1 min at 30 tps


class ActionContextEncoder(nn.Module):
    """Encodes lastActionContext [delay, actionTypeId, queueFlag] into a fixed-dim vector.

    - delay: sqrt-bucketed one-hot (32 bins)
    - actionTypeId: embedding lookup
    - queueFlag: passed through as-is (0 or 1)
    """

    def __init__(self, action_vocab_size: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_vocab_size + 2, hidden_dim // 2, padding_idx=0)
        input_dim = _DELAY_BINS + hidden_dim // 2 + 1  # delay_onehot + action_emb + queue
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        ctx = context.to(torch.float32).reshape(context.shape[0], -1)
        delay = ctx[..., 0]
        action_id = ctx[..., 1]
        queue_flag = ctx[..., 2:3]

        # Delay: sqrt bucketed one-hot.
        delay_safe = torch.clamp(delay, min=0.0)
        delay_onehot = sqrt_bucket(delay_safe, _DELAY_BINS, _DELAY_MAX)

        # Action type: embedding lookup. Shift by +1 so -1 (missing) maps to 0 (padding).
        action_indices = torch.clamp(action_id.to(torch.int64) + 1, min=0)
        action_emb = self.action_embedding(action_indices)

        combined = torch.cat([delay_onehot, action_emb, queue_flag], dim=-1)
        return self.proj(combined)


# ---------------------------------------------------------------------------
# SectionMLP — generic scalar section encoder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BuildOrderTraceEncoder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ScalarEncoder
# ---------------------------------------------------------------------------

_SPECIAL_SECTIONS = frozenset({"buildOrderTrace", "lastActionContext"})


@dataclass
class ScalarEncoderConfig:
    build_order_vocab_size: int
    action_context_vocab_size: int = 0
    hidden_dim: int = 64
    output_dim: int = 128
    dropout: float = 0.1
    section_order: tuple[str, ...] = (
        "scalar",
        "timeEncoding",
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
            if section_name not in _SPECIAL_SECTIONS
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
        action_vocab = config.action_context_vocab_size if config.action_context_vocab_size > 0 else config.build_order_vocab_size
        self.action_context_encoder = ActionContextEncoder(
            action_vocab_size=action_vocab,
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
            elif section_name == "lastActionContext":
                section_embedding = self.action_context_encoder(scalar_sections[section_name])
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


# ---------------------------------------------------------------------------
# EntityEncoder
# ---------------------------------------------------------------------------

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
        self.preprocessor = EntityFeaturePreprocessor(config.feature_dim)
        expanded_dim = self.preprocessor.expanded_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(expanded_dim, config.model_dim),
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
        preprocessed = self.preprocessor(entity_features)
        feature_embedding = self.feature_proj(preprocessed)
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


# ---------------------------------------------------------------------------
# SpatialEncoder
# ---------------------------------------------------------------------------

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
