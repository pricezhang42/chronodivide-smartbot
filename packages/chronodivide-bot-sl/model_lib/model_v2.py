from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model_lib.heads import (
    DelayHead,
    MLPHead,
    QuantityHead,
    QueueHead,
    SpatialLocationHead,
    TargetEntityHead,
    UnitsHead,
    _resolve_condition_ids,
)
from model_lib.model import RA2SLCoreConfig, RA2SLCoreModel
from model_lib.units_autoregressive import build_units_autoregressive_targets
from transform_lib.common import LABEL_LAYOUT_V1_DELAY_BINS


@dataclass
class RA2SLV2DebugConfig:
    entity_name_vocab_size: int
    action_family_count: int
    order_type_count: int
    target_mode_count: int
    queue_update_type_count: int
    buildable_object_vocab_size: int
    super_weapon_type_count: int
    delay_bins: int = LABEL_LAYOUT_V1_DELAY_BINS
    max_commanded_units: int = 64
    max_entities: int = 128
    spatial_size: int = 64
    build_order_vocab_size: int = 512
    action_context_vocab_size: int = 0
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
    build_order_dim: int = 128
    build_order_num_heads: int = 4
    build_order_num_layers: int = 3


class RA2SLV2DebugModel(nn.Module):
    def __init__(self, config: RA2SLV2DebugConfig) -> None:
        super().__init__()
        self.config = config
        self.core = RA2SLCoreModel(
            RA2SLCoreConfig(
                entity_name_vocab_size=config.entity_name_vocab_size,
                build_order_vocab_size=config.build_order_vocab_size,
                action_context_vocab_size=config.action_context_vocab_size,
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
                build_order_dim=config.build_order_dim,
                build_order_num_heads=config.build_order_num_heads,
                build_order_num_layers=config.build_order_num_layers,
            )
        )
        self.action_family_head = MLPHead(
            config.fusion_hidden_dim,
            config.head_hidden_dim,
            config.action_family_count,
            config.dropout,
        )
        self.action_family_condition_embedding = nn.Embedding(config.action_family_count, config.fusion_hidden_dim)
        self.delay_head = DelayHead(
            input_dim=config.fusion_hidden_dim,
            delay_bins=config.delay_bins,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.order_type_head = MLPHead(
            config.fusion_hidden_dim,
            config.head_hidden_dim,
            config.order_type_count,
            config.dropout,
        )
        self.target_mode_head = MLPHead(
            config.fusion_hidden_dim,
            config.head_hidden_dim,
            config.target_mode_count,
            config.dropout,
        )
        self.queue_flag_head = QueueHead(
            input_dim=config.fusion_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.queue_update_type_head = MLPHead(
            config.fusion_hidden_dim,
            config.head_hidden_dim,
            config.queue_update_type_count,
            config.dropout,
        )
        self.buildable_object_head = MLPHead(
            config.fusion_hidden_dim,
            config.head_hidden_dim,
            config.buildable_object_vocab_size,
            config.dropout,
        )
        self.super_weapon_type_head = MLPHead(
            config.fusion_hidden_dim,
            config.head_hidden_dim,
            config.super_weapon_type_count,
            config.dropout,
        )
        self.commanded_units_head = UnitsHead(
            latent_dim=config.fusion_hidden_dim,
            entity_dim=config.entity_model_dim,
            max_selected_units=config.max_commanded_units,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.target_entity_head = TargetEntityHead(
            latent_dim=config.fusion_hidden_dim,
            entity_dim=config.entity_model_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.target_location_head = SpatialLocationHead(
            latent_dim=config.fusion_hidden_dim,
            map_channels=config.spatial_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            output_size=config.spatial_size,
            dropout=config.dropout,
        )
        self.target_location2_head = SpatialLocationHead(
            latent_dim=config.fusion_hidden_dim,
            map_channels=config.spatial_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            output_size=config.spatial_size,
            dropout=config.dropout,
        )
        self.quantity_head = QuantityHead(
            input_dim=config.fusion_hidden_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.condition_dropout = nn.Dropout(config.dropout)

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

            flat_fused_latent = flatten_sequence_tensor(core_outputs["fused_latent"])
            flat_entity_embeddings = flatten_sequence_tensor(core_outputs["entity_embeddings"])
            flat_entity_mask = flatten_sequence_tensor(core_outputs["entity_mask"])
            flat_spatial_feature_map = flatten_sequence_tensor(core_outputs["spatial_feature_map"])
            flat_teacher_targets = (
                None
                if teacher_forcing_targets is None
                else {name: flatten_sequence_tensor(tensor) for name, tensor in teacher_forcing_targets.items()}
            )
            flat_teacher_masks = (
                None
                if teacher_forcing_masks is None
                else {name: flatten_sequence_tensor(tensor) for name, tensor in teacher_forcing_masks.items()}
            )
            head_outputs = self._forward_flat(
                fused_latent=flat_fused_latent,
                entity_embeddings=flat_entity_embeddings,
                entity_mask=flat_entity_mask,
                spatial_feature_map=flat_spatial_feature_map,
                teacher_forcing_targets=flat_teacher_targets,
                teacher_forcing_masks=flat_teacher_masks,
                teacher_forcing_mode=teacher_forcing_mode,
            )
            reshaped_head_outputs = {
                name: tensor.reshape(batch_size, sequence_length, *tensor.shape[1:])
                for name, tensor in head_outputs.items()
            }
            return {**core_outputs, **reshaped_head_outputs}

        head_outputs = self._forward_flat(
            fused_latent=fused_latent,
            entity_embeddings=core_outputs["entity_embeddings"],
            entity_mask=core_outputs["entity_mask"],
            spatial_feature_map=core_outputs["spatial_feature_map"],
            teacher_forcing_targets=teacher_forcing_targets,
            teacher_forcing_masks=teacher_forcing_masks,
            teacher_forcing_mode=teacher_forcing_mode,
        )
        return {**core_outputs, **head_outputs}

    def _forward_flat(
        self,
        *,
        fused_latent: torch.Tensor,
        entity_embeddings: torch.Tensor,
        entity_mask: torch.Tensor,
        spatial_feature_map: torch.Tensor,
        teacher_forcing_targets: dict[str, torch.Tensor] | None,
        teacher_forcing_masks: dict[str, torch.Tensor] | None,
        teacher_forcing_mode: str,
    ) -> dict[str, torch.Tensor]:
        conditioned_latent = fused_latent
        action_family_logits = self.action_family_head(conditioned_latent)
        action_family_teacher = None
        if teacher_forcing_targets is not None and teacher_forcing_mode != "none":
            action_family_teacher = teacher_forcing_targets.get("actionFamilyOneHot")
        action_family_ids = _resolve_condition_ids(
            action_family_logits,
            teacher_forcing_one_hot=action_family_teacher,
            teacher_forcing_mask=None if teacher_forcing_masks is None else teacher_forcing_masks.get("actionFamilyLossMask"),
        )
        conditioned_latent = conditioned_latent + self.condition_dropout(
            self.action_family_condition_embedding(action_family_ids)
        )

        delay_logits = self.delay_head(conditioned_latent)
        order_type_logits = self.order_type_head(conditioned_latent)
        target_mode_logits = self.target_mode_head(conditioned_latent)
        queue_flag_logits = self.queue_flag_head(conditioned_latent)
        queue_update_type_logits = self.queue_update_type_head(conditioned_latent)
        buildable_object_logits = self.buildable_object_head(conditioned_latent)
        super_weapon_type_logits = self.super_weapon_type_head(conditioned_latent)

        commanded_units_teacher = None
        if teacher_forcing_targets is not None and teacher_forcing_mode == "full" and teacher_forcing_masks is not None:
            raw_units_teacher = teacher_forcing_targets.get("commandedUnitsOneHot")
            raw_units_teacher_mask = teacher_forcing_masks.get("commandedUnitsLossMask")
            if raw_units_teacher is not None and raw_units_teacher_mask is not None:
                commanded_units_teacher = build_units_autoregressive_targets(
                    raw_units_teacher,
                    raw_units_teacher_mask,
                )
        commanded_units_outputs = self.commanded_units_head(
            conditioned_latent,
            entity_embeddings,
            entity_mask,
            teacher_forcing_targets=commanded_units_teacher,
        )
        target_entity_logits = self.target_entity_head(conditioned_latent, entity_embeddings, entity_mask)
        target_location_logits = self.target_location_head(conditioned_latent, spatial_feature_map)
        target_location2_logits = self.target_location2_head(conditioned_latent, spatial_feature_map)

        return {
            "actionFamilyLogits": action_family_logits,
            "delayLogits": delay_logits,
            "orderTypeLogits": order_type_logits,
            "targetModeLogits": target_mode_logits,
            "queueFlagLogits": queue_flag_logits,
            "queueUpdateTypeLogits": queue_update_type_logits,
            "buildableObjectLogits": buildable_object_logits,
            "superWeaponTypeLogits": super_weapon_type_logits,
            "commandedUnitsLogits": commanded_units_outputs["logits"],
            "commandedUnitsSelectedIds": commanded_units_outputs["selectedIds"],
            "commandedUnitsSelectedMask": commanded_units_outputs["selectedMask"],
            "targetEntityLogits": target_entity_logits,
            "targetLocationLogits": target_location_logits,
            "targetLocation2Logits": target_location2_logits,
            "quantityPred": self.quantity_head(conditioned_latent),
        }
