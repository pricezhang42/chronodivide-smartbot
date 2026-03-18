from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from action_dict import ACTION_TYPE_ID_TO_NAME
from model_lib.encoders import (
    EntityEncoder,
    EntityEncoderConfig,
    ScalarEncoder,
    ScalarEncoderConfig,
    SpatialEncoder,
    SpatialEncoderConfig,
    pool_selected_entity_embeddings,
)
from model_lib.heads import RA2SLHeadsConfig, RA2SLPredictionHeads
from transform_lib.common import LABEL_LAYOUT_V1_DELAY_BINS


@dataclass
class RA2SLCoreConfig:
    entity_name_vocab_size: int
    build_order_vocab_size: int = len(ACTION_TYPE_ID_TO_NAME)
    scalar_hidden_dim: int = 64
    scalar_output_dim: int = 256
    entity_model_dim: int = 128
    entity_num_heads: int = 4
    entity_num_layers: int = 2
    spatial_hidden_dim: int = 128
    spatial_output_dim: int = 128
    fusion_hidden_dim: int = 256
    dropout: float = 0.1


class RA2SLCoreModel(nn.Module):
    def __init__(self, config: RA2SLCoreConfig) -> None:
        super().__init__()
        self.config = config
        self.scalar_encoder = ScalarEncoder(
            ScalarEncoderConfig(
                build_order_vocab_size=config.build_order_vocab_size,
                hidden_dim=config.scalar_hidden_dim,
                output_dim=config.scalar_output_dim,
                dropout=config.dropout,
            )
        )
        self.entity_encoder = EntityEncoder(
            EntityEncoderConfig(
                entity_name_vocab_size=config.entity_name_vocab_size,
                model_dim=config.entity_model_dim,
                num_heads=config.entity_num_heads,
                num_layers=config.entity_num_layers,
                dropout=config.dropout,
            )
        )
        self.spatial_encoder = SpatialEncoder(
            SpatialEncoderConfig(
                hidden_dim=config.spatial_hidden_dim,
                output_dim=config.spatial_output_dim,
                dropout=config.dropout,
            )
        )
        fusion_input_dim = (
            config.scalar_output_dim
            + config.entity_model_dim
            + config.entity_model_dim
            + config.spatial_output_dim
        )
        self.fusion_torso = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.GELU(),
        )

    def forward(self, model_inputs: dict[str, object]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        scalar_sections = model_inputs["scalar_sections"]
        entity_inputs = model_inputs["entity"]
        spatial_inputs = model_inputs["spatial"]

        scalar_outputs = self.scalar_encoder(scalar_sections)
        entity_outputs = self.entity_encoder(
            entity_features=entity_inputs["features"],
            entity_mask=entity_inputs["mask"],
            entity_name_tokens=entity_inputs["name_tokens"],
        )
        selected_entity_summary = pool_selected_entity_embeddings(
            entity_outputs["per_entity"],
            entity_outputs["mask"],
            scalar_sections["currentSelectionIndices"],
            scalar_sections["currentSelectionResolvedMask"],
        )
        spatial_outputs = self.spatial_encoder(
            spatial=spatial_inputs["spatial"],
            minimap=spatial_inputs["minimap"],
            map_static=spatial_inputs["map_static"],
        )

        fused_input = torch.cat(
            [
                scalar_outputs["pooled"],
                entity_outputs["pooled"],
                selected_entity_summary,
                spatial_outputs["pooled"],
            ],
            dim=-1,
        )
        fused_latent = self.fusion_torso(fused_input)
        return {
            "scalar": scalar_outputs["pooled"],
            "entity": entity_outputs["pooled"],
            "selected_entity": selected_entity_summary,
            "spatial": spatial_outputs["pooled"],
            "entity_embeddings": entity_outputs["per_entity"],
            "entity_mask": entity_outputs["mask"],
            "spatial_feature_map": spatial_outputs["feature_map"],
            "fused_latent": fused_latent,
        }


@dataclass
class RA2SLBaselineConfig:
    entity_name_vocab_size: int
    action_vocab_size: int = len(ACTION_TYPE_ID_TO_NAME)
    delay_bins: int = LABEL_LAYOUT_V1_DELAY_BINS
    max_selected_units: int = 64
    max_entities: int = 128
    spatial_size: int = 32
    build_order_vocab_size: int = len(ACTION_TYPE_ID_TO_NAME)
    scalar_hidden_dim: int = 64
    scalar_output_dim: int = 256
    entity_model_dim: int = 128
    entity_num_heads: int = 4
    entity_num_layers: int = 2
    spatial_hidden_dim: int = 128
    spatial_output_dim: int = 128
    fusion_hidden_dim: int = 256
    head_hidden_dim: int = 256
    dropout: float = 0.1


class RA2SLBaselineModel(nn.Module):
    def __init__(self, config: RA2SLBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.core = RA2SLCoreModel(
            RA2SLCoreConfig(
                entity_name_vocab_size=config.entity_name_vocab_size,
                build_order_vocab_size=config.build_order_vocab_size,
                scalar_hidden_dim=config.scalar_hidden_dim,
                scalar_output_dim=config.scalar_output_dim,
                entity_model_dim=config.entity_model_dim,
                entity_num_heads=config.entity_num_heads,
                entity_num_layers=config.entity_num_layers,
                spatial_hidden_dim=config.spatial_hidden_dim,
                spatial_output_dim=config.spatial_output_dim,
                fusion_hidden_dim=config.fusion_hidden_dim,
                dropout=config.dropout,
            )
        )
        self.heads = RA2SLPredictionHeads(
            RA2SLHeadsConfig(
                action_vocab_size=config.action_vocab_size,
                delay_bins=config.delay_bins,
                max_selected_units=config.max_selected_units,
                max_entities=config.max_entities,
                spatial_size=config.spatial_size,
                fusion_dim=config.fusion_hidden_dim,
                entity_dim=config.entity_model_dim,
                spatial_map_channels=config.spatial_hidden_dim,
                hidden_dim=config.head_hidden_dim,
                dropout=config.dropout,
            )
        )

    def forward(self, model_inputs: dict[str, object]) -> dict[str, torch.Tensor]:
        core_outputs = self.core(model_inputs)
        head_outputs = self.heads(
            fused_latent=core_outputs["fused_latent"],
            entity_embeddings=core_outputs["entity_embeddings"],
            entity_mask=core_outputs["entity_mask"],
            spatial_feature_map=core_outputs["spatial_feature_map"],
            available_action_mask=model_inputs["scalar_sections"].get("availableActionMask"),
        )
        return {**core_outputs, **head_outputs}
