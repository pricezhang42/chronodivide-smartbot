from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sqrt_bucket(value: torch.Tensor, num_bins: int, max_value: float) -> torch.Tensor:
    """Sqrt-scaled one-hot bucket: maps [0, max_value] -> num_bins bins via sqrt scaling."""
    clamped = torch.clamp(value.float(), min=0.0, max=max_value)
    normalized = torch.sqrt(clamped / max_value)
    bin_index = torch.clamp((normalized * num_bins).long(), max=num_bins - 1)
    return F.one_hot(bin_index, num_bins).float()


def linear_bucket(value: torch.Tensor, num_bins: int, max_value: float) -> torch.Tensor:
    """Linear one-hot bucket: maps [0, max_value] -> num_bins bins."""
    clamped = torch.clamp(value.float(), min=0.0, max=max_value)
    normalized = clamped / max_value
    bin_index = torch.clamp((normalized * num_bins).long(), max=num_bins - 1)
    return F.one_hot(bin_index, num_bins).float()


# Entity feature column indices and their bucketing configs:
# (column_index, num_bins, max_value, bucket_fn)
_ENTITY_BUCKET_SPECS = [
    (14, 20, 1500.0, sqrt_bucket),   # hit_points
    (15, 20, 1500.0, sqrt_bucket),   # max_hit_points
    (19, 16, 3000.0, sqrt_bucket),   # purchase_value
    (56, 16, 120.0, linear_bucket),  # primary_weapon_cooldown_ticks
    (57, 16, 120.0, linear_bucket),  # secondary_weapon_cooldown_ticks
]
_ENTITY_BUCKET_COLUMNS = {spec[0] for spec in _ENTITY_BUCKET_SPECS}
_ENTITY_BUCKET_EXTRA_DIM = sum(spec[1] - 1 for spec in _ENTITY_BUCKET_SPECS)


class EntityFeaturePreprocessor(nn.Module):
    """Replaces 5 continuous entity columns with bucketed one-hot vectors.

    Expands entity feature dim from 74 -> 74 + extra_dim (each column expands
    from 1 to num_bins, so net gain = sum(num_bins - 1) per column).
    """

    def __init__(self) -> None:
        super().__init__()
        self._bucket_specs = _ENTITY_BUCKET_SPECS
        self._bucket_columns = sorted(_ENTITY_BUCKET_COLUMNS)
        self._extra_dim = _ENTITY_BUCKET_EXTRA_DIM

    @property
    def extra_dim(self) -> int:
        return self._extra_dim

    def forward(self, entity_features: torch.Tensor) -> torch.Tensor:
        """entity_features: [batch, max_entities, feature_dim] -> [..., feature_dim + extra_dim]"""
        parts: list[torch.Tensor] = []
        prev_col = 0
        for col_idx, num_bins, max_val, bucket_fn in self._bucket_specs:
            if prev_col < col_idx:
                parts.append(entity_features[..., prev_col:col_idx])
            parts.append(bucket_fn(entity_features[..., col_idx], num_bins, max_val))
            prev_col = col_idx + 1
        if prev_col < entity_features.shape[-1]:
            parts.append(entity_features[..., prev_col:])
        return torch.cat(parts, dim=-1)


class ActionContextEncoder(nn.Module):
    """Dedicated encoder for lastActionContext [delay, actionTypeId, ...flags].

    - delay -> sqrt bucketed one-hot (32 bins, max 1800 ticks)
    - actionTypeId -> embedding lookup
    - remaining columns -> passthrough
    """

    _DELAY_BINS = 32
    _DELAY_MAX_TICKS = 1800.0

    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        # Use LazyLinear so input_dim is inferred from the first forward pass
        self.proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        """action_context: [batch, 3] or [batch, 4] — delay, actionTypeId, queueFlag(s)."""
        delay = action_context[:, 0].float()
        action_id = torch.clamp(action_context[:, 1].long() + 1, min=0)
        queue_flags = action_context[:, 2:].float()
        delay_onehot = sqrt_bucket(delay, self._DELAY_BINS, self._DELAY_MAX_TICKS)
        action_embed = self.action_embedding(action_id)
        combined = torch.cat([delay_onehot, action_embed, queue_flags], dim=-1)
        return self.proj(combined)


# Sections that use a dedicated encoder instead of generic SectionMLP
_SPECIAL_SECTIONS = {"buildOrderTrace", "lastActionContext"}


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
    """Transformer-based build order encoder with positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        dropout: float,
        max_len: int = 20,
        num_heads: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
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
        batch_size, seq_len = safe_ids.shape
        positions = torch.arange(seq_len, device=safe_ids.device).unsqueeze(0).expand(batch_size, -1)
        embedded = self.token_embedding(safe_ids) + self.position_embedding(positions)
        embedded = embedded.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        encoded = self.transformer(embedded, src_key_padding_mask=~valid_mask)
        encoded = self.output_norm(encoded)
        pooled = masked_mean(encoded, valid_mask)
        return self.proj(pooled)


@dataclass
class ScalarEncoderConfig:
    build_order_vocab_size: int
    action_context_vocab_size: int = 0
    hidden_dim: int = 64
    output_dim: int = 128
    dropout: float = 0.1
    build_order_dim: int = 128
    build_order_num_heads: int = 4
    build_order_num_layers: int = 3
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
        "timeEncoding",
        "gameStats",
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
            hidden_dim=config.build_order_dim,
            dropout=config.dropout,
            num_heads=config.build_order_num_heads,
            num_layers=config.build_order_num_layers,
        )
        # Project build order output to the shared hidden_dim for concatenation
        if config.build_order_dim != config.hidden_dim:
            self.build_order_proj: nn.Module | None = nn.Linear(config.build_order_dim, config.hidden_dim)
        else:
            self.build_order_proj = None
        if config.action_context_vocab_size > 0:
            self.action_context_encoder: nn.Module | None = ActionContextEncoder(
                vocab_size=config.action_context_vocab_size,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        else:
            self.action_context_encoder = None
        self.output_proj = nn.Sequential(
            nn.LazyLinear(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
            nn.GELU(),
        )

    def forward(self, scalar_sections: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ordered_embeddings: list[torch.Tensor] = []
        for section_name in self.config.section_order:
            if section_name not in scalar_sections:
                continue
            if section_name == "buildOrderTrace":
                section_embedding = self.build_order_encoder(scalar_sections[section_name])
                if self.build_order_proj is not None:
                    section_embedding = self.build_order_proj(section_embedding)
            elif section_name == "lastActionContext" and self.action_context_encoder is not None:
                section_embedding = self.action_context_encoder(scalar_sections[section_name])
            else:
                section_embedding = self.section_mlps[section_name](scalar_sections[section_name])
            ordered_embeddings.append(section_embedding)

        if not ordered_embeddings:
            raise ValueError("ScalarEncoder received no known scalar sections.")

        concatenated = torch.cat(ordered_embeddings, dim=-1)
        return {"pooled": self.output_proj(concatenated)}


@dataclass
class EntityEncoderConfig:
    entity_name_vocab_size: int
    name_embedding_dim: int = 32
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class EntityEncoder(nn.Module):
    def __init__(self, config: EntityEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.preprocessor = EntityFeaturePreprocessor()
        # Use LazyLinear so that the input dim is inferred from actual data
        # (entity features grow as augmentation adds intent/threat columns).
        self.feature_proj = nn.Sequential(
            nn.LazyLinear(config.model_dim),
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
        preprocessed = self.preprocessor(entity_features.to(torch.float32))
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


_SCATTER_CHANNELS = 8


def scatter_entity_embeddings(
    entity_embeddings: torch.Tensor,
    entity_features: torch.Tensor,
    entity_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Scatter entity embeddings onto a spatial grid using normalized tile positions.

    Args:
        entity_embeddings: [batch, max_entities, embed_dim]
        entity_features: [batch, max_entities, feature_dim] (raw, pre-preprocessor)
        entity_mask: [batch, max_entities] bool
        grid_h, grid_w: spatial grid dimensions

    Returns:
        [batch, embed_dim, grid_h, grid_w] — scattered embedding map
    """
    batch_size, max_entities, embed_dim = entity_embeddings.shape
    # tile_x_norm is column 12, tile_y_norm is column 13 in entity features
    tile_x_norm = entity_features[:, :, 12].float()  # [batch, max_entities]
    tile_y_norm = entity_features[:, :, 13].float()

    grid_x = torch.clamp((tile_x_norm * grid_w).long(), min=0, max=grid_w - 1)
    grid_y = torch.clamp((tile_y_norm * grid_h).long(), min=0, max=grid_h - 1)
    flat_index = grid_y * grid_w + grid_x  # [batch, max_entities]

    # Mask out invalid entities
    mask_float = entity_mask.float().unsqueeze(-1)  # [batch, max_entities, 1]
    masked_embeddings = entity_embeddings * mask_float

    # Scatter add into flat spatial grid
    output = torch.zeros(batch_size, grid_h * grid_w, embed_dim, device=entity_embeddings.device, dtype=entity_embeddings.dtype)
    flat_index_expanded = flat_index.unsqueeze(-1).expand_as(masked_embeddings)
    output.scatter_add_(1, flat_index_expanded, masked_embeddings)

    # Reshape to [batch, embed_dim, grid_h, grid_w]
    return output.permute(0, 2, 1).reshape(batch_size, embed_dim, grid_h, grid_w)


@dataclass
class SpatialEncoderConfig:
    hidden_dim: int = 64
    output_dim: int = 64
    dropout: float = 0.1
    entity_scatter_dim: int = 0


class SpatialEncoder(nn.Module):
    def __init__(self, config: SpatialEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.entity_scatter_dim > 0:
            self.scatter_proj = nn.Sequential(
                nn.Linear(config.entity_scatter_dim, _SCATTER_CHANNELS),
                nn.GELU(),
            )
        else:
            self.scatter_proj = None
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
        entity_scatter_map: torch.Tensor | None = None,
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
        planes = [
            resized_spatial,
            minimap.to(torch.float32),
            resized_map_static,
        ]
        if entity_scatter_map is not None and self.scatter_proj is not None:
            # entity_scatter_map: [batch, embed_dim, H, W] -> project to _SCATTER_CHANNELS
            b, c, h, w = entity_scatter_map.shape
            resized_scatter = F.interpolate(
                entity_scatter_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ) if (h, w) != target_size else entity_scatter_map
            # Project: permute to [b, H, W, C], project, permute back
            scatter_flat = resized_scatter.permute(0, 2, 3, 1)  # [b, H, W, embed_dim]
            scatter_proj = self.scatter_proj(scatter_flat)  # [b, H, W, _SCATTER_CHANNELS]
            planes.append(scatter_proj.permute(0, 3, 1, 2))  # [b, _SCATTER_CHANNELS, H, W]

        stacked = torch.cat(planes, dim=1)
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
