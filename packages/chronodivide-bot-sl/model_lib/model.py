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
    scalar_output_dim: int = 128
    entity_model_dim: int = 64
    entity_num_heads: int = 4
    entity_num_layers: int = 2
    spatial_hidden_dim: int = 64
    spatial_output_dim: int = 64
    fusion_hidden_dim: int = 128
    use_lstm_core: bool = False
    lstm_num_layers: int = 1
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
        if config.use_lstm_core:
            self.lstm_core = nn.LSTM(
                input_size=config.fusion_hidden_dim,
                hidden_size=config.fusion_hidden_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.lstm_output_norm = nn.LayerNorm(config.fusion_hidden_dim)
        else:
            self.lstm_core = None
            self.lstm_output_norm = None

    @staticmethod
    def _is_sequence_batch(scalar_sections: dict[str, torch.Tensor]) -> bool:
        first_tensor = next(iter(scalar_sections.values()))
        return first_tensor.ndim >= 3

    @staticmethod
    def _flatten_sequence_tensor(tensor: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
        return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])

    @staticmethod
    def _unflatten_sequence_tensor(tensor: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
        return tensor.reshape(batch_size, sequence_length, *tensor.shape[1:])

    def forward(self, model_inputs: dict[str, object]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        scalar_sections = model_inputs["scalar_sections"]
        entity_inputs = model_inputs["entity"]
        spatial_inputs = model_inputs["spatial"]
        is_sequence_batch = self._is_sequence_batch(scalar_sections)
        if is_sequence_batch:
            batch_size = int(next(iter(scalar_sections.values())).shape[0])
            sequence_length = int(next(iter(scalar_sections.values())).shape[1])
            flat_scalar_sections = {
                name: self._flatten_sequence_tensor(tensor, batch_size, sequence_length)
                for name, tensor in scalar_sections.items()
            }
            flat_entity_inputs = {
                name: self._flatten_sequence_tensor(tensor, batch_size, sequence_length)
                for name, tensor in entity_inputs.items()
            }
            flat_spatial_inputs = {
                name: self._flatten_sequence_tensor(tensor, batch_size, sequence_length)
                for name, tensor in spatial_inputs.items()
            }
        else:
            batch_size = int(next(iter(scalar_sections.values())).shape[0])
            sequence_length = 1
            flat_scalar_sections = scalar_sections
            flat_entity_inputs = entity_inputs
            flat_spatial_inputs = spatial_inputs

        scalar_outputs = self.scalar_encoder(flat_scalar_sections)
        entity_outputs = self.entity_encoder(
            entity_features=flat_entity_inputs["features"],
            entity_mask=flat_entity_inputs["mask"],
            entity_name_tokens=flat_entity_inputs["name_tokens"],
        )
        selected_entity_summary = pool_selected_entity_embeddings(
            entity_outputs["per_entity"],
            entity_outputs["mask"],
            flat_scalar_sections["currentSelectionIndices"],
            flat_scalar_sections["currentSelectionResolvedMask"],
        )
        spatial_outputs = self.spatial_encoder(
            spatial=flat_spatial_inputs["spatial"],
            minimap=flat_spatial_inputs["minimap"],
            map_static=flat_spatial_inputs["map_static"],
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

        if is_sequence_batch:
            fused_sequence = self._unflatten_sequence_tensor(fused_latent, batch_size, sequence_length)
        else:
            fused_sequence = fused_latent.unsqueeze(1)

        if self.lstm_core is not None:
            recurrent_output, _ = self.lstm_core(fused_sequence)
            fused_sequence = self.lstm_output_norm(recurrent_output + fused_sequence)

        def reshape_output(tensor: torch.Tensor) -> torch.Tensor:
            if is_sequence_batch:
                return self._unflatten_sequence_tensor(tensor, batch_size, sequence_length)
            return tensor

        fused_output = fused_sequence if is_sequence_batch else fused_sequence.squeeze(1)
        return {
            "scalar": reshape_output(scalar_outputs["pooled"]),
            "entity": reshape_output(entity_outputs["pooled"]),
            "selected_entity": reshape_output(selected_entity_summary),
            "spatial": reshape_output(spatial_outputs["pooled"]),
            "entity_embeddings": reshape_output(entity_outputs["per_entity"]),
            "entity_mask": reshape_output(entity_outputs["mask"]),
            "spatial_feature_map": reshape_output(spatial_outputs["feature_map"]),
            "fused_latent": fused_output,
        }


@dataclass
class RA2SLBaselineConfig:
    entity_name_vocab_size: int
    action_vocab_size: int = len(ACTION_TYPE_ID_TO_NAME)
    delay_bins: int = LABEL_LAYOUT_V1_DELAY_BINS
    max_selected_units: int = 64
    max_entities: int = 128
    spatial_size: int = 64
    build_order_vocab_size: int = len(ACTION_TYPE_ID_TO_NAME)
    scalar_hidden_dim: int = 64
    scalar_output_dim: int = 128
    entity_model_dim: int = 64
    entity_num_heads: int = 4
    entity_num_layers: int = 2
    spatial_hidden_dim: int = 64
    spatial_output_dim: int = 64
    fusion_hidden_dim: int = 128
    head_hidden_dim: int = 256
    use_lstm_core: bool = False
    lstm_num_layers: int = 1
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
                use_lstm_core=config.use_lstm_core,
                lstm_num_layers=config.lstm_num_layers,
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

    def forward(
        self,
        model_inputs: dict[str, object],
        *,
        teacher_forcing_targets: dict[str, torch.Tensor] | None = None,
        teacher_forcing_masks: dict[str, torch.Tensor] | None = None,
        teacher_forcing_mode: str = "none",
    ) -> dict[str, torch.Tensor]:
        core_outputs = self.core(model_inputs)
        fused_latent = core_outputs["fused_latent"]
        is_sequence_batch = fused_latent.ndim >= 3
        if is_sequence_batch:
            batch_size = int(fused_latent.shape[0])
            sequence_length = int(fused_latent.shape[1])

            def flatten_sequence_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
                if tensor is None:
                    return None
                return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])

            head_outputs = self.heads(
                fused_latent=flatten_sequence_tensor(core_outputs["fused_latent"]),
                entity_embeddings=flatten_sequence_tensor(core_outputs["entity_embeddings"]),
                entity_mask=flatten_sequence_tensor(core_outputs["entity_mask"]),
                spatial_feature_map=flatten_sequence_tensor(core_outputs["spatial_feature_map"]),
                available_action_mask=flatten_sequence_tensor(model_inputs["scalar_sections"].get("availableActionMask")),
                teacher_forcing_targets=(
                    None
                    if teacher_forcing_targets is None
                    else {
                        name: flatten_sequence_tensor(tensor)
                        for name, tensor in teacher_forcing_targets.items()
                    }
                ),
                teacher_forcing_masks=(
                    None
                    if teacher_forcing_masks is None
                    else {
                        name: flatten_sequence_tensor(tensor)
                        for name, tensor in teacher_forcing_masks.items()
                    }
                ),
                teacher_forcing_mode=teacher_forcing_mode,
            )

            reshaped_head_outputs = {
                name: tensor.reshape(batch_size, sequence_length, *tensor.shape[1:])
                for name, tensor in head_outputs.items()
            }
            return {**core_outputs, **reshaped_head_outputs}

        head_outputs = self.heads(
            fused_latent=fused_latent,
            entity_embeddings=core_outputs["entity_embeddings"],
            entity_mask=core_outputs["entity_mask"],
            spatial_feature_map=core_outputs["spatial_feature_map"],
            available_action_mask=model_inputs["scalar_sections"].get("availableActionMask"),
            teacher_forcing_targets=teacher_forcing_targets,
            teacher_forcing_masks=teacher_forcing_masks,
            teacher_forcing_mode=teacher_forcing_mode,
        )
        return {**core_outputs, **head_outputs}
