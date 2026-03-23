from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from action_dict import (  # noqa: E402
    ACTION_TYPE_NAME_TO_ID,
    BUILDABLE_OBJECT_NAME_TO_ID,
    get_action_type_id,
)
from model_lib.batch import build_model_inputs  # noqa: E402
from model_lib.model import RA2SLBaselineConfig, RA2SLBaselineModel  # noqa: E402
from model_lib.model_v2 import RA2SLV2DebugConfig, RA2SLV2DebugModel  # noqa: E402
from transform_lib.common import LABEL_LAYOUT_V1_MISSING_INT  # noqa: E402
from transform_lib.feature_layout import (  # noqa: E402
    BUILD_ORDER_TRACE_LEN,
    augment_dataset_with_available_action_mask,
    augment_dataset_with_current_selection_summary,
    augment_dataset_with_enemy_memory_bow,
    augment_dataset_with_entity_intent_summary,
    augment_dataset_with_entity_threat,
    augment_dataset_with_owned_composition_bow,
    augment_dataset_with_production_state,
    augment_dataset_with_scalar_core_identity,
    augment_dataset_with_super_weapon_state,
    augment_dataset_with_tech_state,
    augment_dataset_with_time_encoding,
)
from transform_lib.label_layout_v2 import (  # noqa: E402
    LABEL_LAYOUT_V2_ACTION_FAMILIES,
    LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID,
    LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPE_TO_ID,
)
from transform_lib.schema_utils import (  # noqa: E402
    append_schema_section,
    build_structured_section_tensors,
    compute_flat_length,
)


BUILDING_QUEUE_NAMES = {"Structures", "Armory"}
MISSING_INT = LABEL_LAYOUT_V1_MISSING_INT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live checkpoint-backed production advisor service.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    env_value = os.environ.get("SL_ADVISOR_DEVICE")
    if env_value:
        return torch.device(env_value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _checkpoint_uses_v2(model_state: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("action_family_head.") for key in model_state)


def _get_required_shape(model_state: dict[str, torch.Tensor], key: str) -> tuple[int, ...]:
    tensor = model_state.get(key)
    if tensor is None:
        raise KeyError(f"Missing checkpoint tensor: {key}")
    return tuple(int(size) for size in tensor.shape)


def _detect_action_context_vocab_size(model_state: dict[str, torch.Tensor]) -> int:
    """Detect whether the checkpoint has an ActionContextEncoder, and return its vocab size (or 0)."""
    key = "core.scalar_encoder.action_context_encoder.action_embedding.weight"
    if key in model_state:
        return int(model_state[key].shape[0]) - 2  # vocab_size + 2 (padding)
    return 0


def _detect_build_order_vocab_size(model_state: dict[str, torch.Tensor]) -> int:
    """Detect build order vocab size from checkpoint, handling both old and new key names."""
    # New key: token_embedding.weight (from BuildOrderTraceEncoder)
    new_key = "core.scalar_encoder.build_order_encoder.token_embedding.weight"
    if new_key in model_state:
        return int(model_state[new_key].shape[0]) - 1
    # Old key: embedding.weight
    old_key = "core.scalar_encoder.build_order_encoder.embedding.weight"
    if old_key in model_state:
        return int(model_state[old_key].shape[0]) - 1
    raise KeyError("Cannot find build order embedding in checkpoint (tried token_embedding.weight and embedding.weight)")


def _detect_build_order_config(model_state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Detect build order transformer config from checkpoint weights."""
    new_key = "core.scalar_encoder.build_order_encoder.token_embedding.weight"
    if new_key in model_state:
        build_order_dim = int(model_state[new_key].shape[1])
        # Count transformer layers
        num_layers = 0
        while f"core.scalar_encoder.build_order_encoder.transformer.layers.{num_layers}.self_attn.in_proj_weight" in model_state:
            num_layers += 1
        # Detect num_heads from in_proj_weight shape: (3 * dim, dim) → heads = first_dim / (3 * head_dim)
        # With standard MHA: in_proj is (3*embed_dim, embed_dim), num_heads inferred from layer
        attn_key = "core.scalar_encoder.build_order_encoder.transformer.layers.0.self_attn.in_proj_weight"
        # Default to 4 heads; exact value doesn't matter for loading since it's encoded in weights
        num_heads = 4
        if attn_key in model_state:
            in_proj_rows = int(model_state[attn_key].shape[0])
            # in_proj_weight is (3*embed_dim, embed_dim), try common head counts
            for candidate_heads in [8, 4, 2, 1]:
                if build_order_dim % candidate_heads == 0:
                    num_heads = candidate_heads
                    break
        return {"build_order_dim": build_order_dim, "build_order_num_layers": num_layers, "build_order_num_heads": num_heads}
    # Old-style: uses hidden_dim defaults
    return {}


def _load_v1_model(
    model_state: dict[str, torch.Tensor],
    checkpoint_config: dict[str, Any],
    device: torch.device,
) -> RA2SLBaselineModel:
    entity_name_vocab_size = _get_required_shape(model_state, "core.entity_encoder.name_embedding.weight")[0] - 1
    build_order_vocab_size = _detect_build_order_vocab_size(model_state)
    build_order_config = _detect_build_order_config(model_state)
    action_vocab_size = _get_required_shape(model_state, "heads.action_type_head.head.net.3.weight")[0]
    delay_bins = _get_required_shape(model_state, "heads.delay_head.head.net.3.weight")[0]
    max_selected_units = _get_required_shape(model_state, "heads.units_head.step_embeddings.weight")[0] - 1
    spatial_size = int(checkpoint_config.get("spatial_size", checkpoint_config.get("spatialSize", 64)))
    max_entities = int(checkpoint_config.get("max_entities", checkpoint_config.get("maxEntities", 128)))
    action_context_vocab_size = _detect_action_context_vocab_size(model_state)
    model = RA2SLBaselineModel(
        RA2SLBaselineConfig(
            entity_name_vocab_size=entity_name_vocab_size,
            action_vocab_size=action_vocab_size,
            delay_bins=delay_bins,
            max_selected_units=max_selected_units,
            max_entities=max_entities,
            spatial_size=spatial_size,
            build_order_vocab_size=build_order_vocab_size,
            action_context_vocab_size=action_context_vocab_size,
            use_lstm_core=bool(checkpoint_config.get("use_lstm_core", False)),
            lstm_num_layers=int(checkpoint_config.get("lstm_num_layers", 1)),
            **build_order_config,
        )
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def _load_v2_model(
    model_state: dict[str, torch.Tensor],
    checkpoint_config: dict[str, Any],
    device: torch.device,
) -> RA2SLV2DebugModel:
    entity_name_vocab_size = _get_required_shape(model_state, "core.entity_encoder.name_embedding.weight")[0] - 1
    build_order_vocab_size = _detect_build_order_vocab_size(model_state)
    build_order_config = _detect_build_order_config(model_state)
    action_family_count = _get_required_shape(model_state, "action_family_head.net.3.weight")[0]
    order_type_count = _get_required_shape(model_state, "order_type_head.net.3.weight")[0]
    target_mode_count = _get_required_shape(model_state, "target_mode_head.net.3.weight")[0]
    queue_update_type_count = _get_required_shape(model_state, "queue_update_type_head.net.3.weight")[0]
    buildable_object_vocab_size = _get_required_shape(model_state, "buildable_object_head.net.3.weight")[0]
    super_weapon_type_count = _get_required_shape(model_state, "super_weapon_type_head.net.3.weight")[0]
    delay_bins = _get_required_shape(model_state, "delay_head.head.net.3.weight")[0]
    max_commanded_units = _get_required_shape(model_state, "commanded_units_head.step_embeddings.weight")[0] - 1
    spatial_size = int(checkpoint_config.get("spatial_size", checkpoint_config.get("spatialSize", 64)))
    max_entities = int(checkpoint_config.get("max_entities", checkpoint_config.get("maxEntities", 128)))
    action_context_vocab_size = _detect_action_context_vocab_size(model_state)
    model = RA2SLV2DebugModel(
        RA2SLV2DebugConfig(
            entity_name_vocab_size=entity_name_vocab_size,
            action_family_count=action_family_count,
            order_type_count=order_type_count,
            target_mode_count=target_mode_count,
            queue_update_type_count=queue_update_type_count,
            buildable_object_vocab_size=buildable_object_vocab_size,
            super_weapon_type_count=super_weapon_type_count,
            delay_bins=delay_bins,
            max_commanded_units=max_commanded_units,
            max_entities=max_entities,
            spatial_size=spatial_size,
            build_order_vocab_size=build_order_vocab_size,
            action_context_vocab_size=action_context_vocab_size,
            use_lstm_core=bool(checkpoint_config.get("use_lstm_core", False)),
            lstm_num_layers=int(checkpoint_config.get("lstm_num_layers", 1)),
            **build_order_config,
        )
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def load_runtime_model(checkpoint_path: Path, device: torch.device) -> tuple[str, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint).__name__}")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise KeyError(f"Checkpoint is missing model state: {checkpoint_path}")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}

    if _checkpoint_uses_v2(model_state):
        return "v2", _load_v2_model(model_state, checkpoint_config, device)
    return "v1", _load_v1_model(model_state, checkpoint_config, device)


def _build_base_feature_sections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    observation = payload["featureSchemaObservation"]
    feature_tensors = payload["featureTensors"]
    max_entities = int(observation["maxEntities"])
    spatial_size = int(observation["spatialSize"])
    minimap_size = int(observation["minimapSize"])
    selection_width = len(feature_tensors["currentSelectionIndices"])
    map_static_channel_count = len(feature_tensors["mapStatic"])
    return [
        {"name": "scalar", "shape": [len(observation["scalarFeatureNames"])], "dtype": "float32"},
        {"name": "lastActionContext", "shape": [3], "dtype": "int32"},
        {"name": "currentSelectionCount", "shape": [1], "dtype": "int32"},
        {"name": "currentSelectionResolvedCount", "shape": [1], "dtype": "int32"},
        {"name": "currentSelectionOverflowCount", "shape": [1], "dtype": "int32"},
        {"name": "currentSelectionIndices", "shape": [selection_width], "dtype": "int32"},
        {"name": "currentSelectionMask", "shape": [selection_width], "dtype": "int32"},
        {"name": "currentSelectionResolvedMask", "shape": [selection_width], "dtype": "int32"},
        {"name": "entityNameTokens", "shape": [max_entities], "dtype": "int32"},
        {"name": "entityMask", "shape": [max_entities], "dtype": "int32"},
        {
            "name": "entityFeatures",
            "shape": [max_entities, len(observation["entityFeatureNames"])],
            "dtype": "float32",
        },
        {
            "name": "spatial",
            "shape": [len(observation["spatialChannelNames"]), spatial_size, spatial_size],
            "dtype": "float32",
        },
        {
            "name": "minimap",
            "shape": [len(observation["minimapChannelNames"]), minimap_size, minimap_size],
            "dtype": "float32",
        },
        {
            "name": "mapStatic",
            "shape": [map_static_channel_count, spatial_size, spatial_size],
            "dtype": "float32",
        },
    ]


def _last_action_context(payload: dict[str, Any]) -> list[int]:
    runtime_state = payload.get("runtimeState") or {}
    tick = int(payload.get("tick", 0))
    last_action_tick = runtime_state.get("lastActionTick")
    if isinstance(last_action_tick, int):
        delay_from_previous = max(0, tick - last_action_tick)
    else:
        delay_from_previous = MISSING_INT
    last_action_type_name = runtime_state.get("lastActionTypeNameV1")
    last_action_type_id = (
        get_action_type_id(last_action_type_name)
        if isinstance(last_action_type_name, str) and last_action_type_name
        else MISSING_INT
    )
    last_queue_value = runtime_state.get("lastQueueValue")
    if not isinstance(last_queue_value, int):
        last_queue_value = MISSING_INT
    return [delay_from_previous, last_action_type_id, int(last_queue_value)]


def _build_order_trace(runtime_state: dict[str, Any]) -> list[int]:
    trace = [
        get_action_type_id(action_name)
        for action_name in runtime_state.get("buildOrderActionTypeNamesV1", [])
        if isinstance(action_name, str) and action_name
    ][:BUILD_ORDER_TRACE_LEN]
    if len(trace) < BUILD_ORDER_TRACE_LEN:
        trace.extend([MISSING_INT] * (BUILD_ORDER_TRACE_LEN - len(trace)))
    return trace


def build_runtime_feature_sections(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    feature_tensors = deepcopy(payload["featureTensors"])
    feature_tensors["lastActionContext"] = _last_action_context(payload)
    feature_tensors["buildOrderTrace"] = _build_order_trace(payload.get("runtimeState") or {})

    dataset = {
        "schema": {
            "observation": {
                "scalarFeatureNames": list(payload["featureSchemaObservation"]["scalarFeatureNames"]),
                "entityFeatureNames": list(payload["featureSchemaObservation"]["entityFeatureNames"]),
                "spatialChannelNames": list(payload["featureSchemaObservation"]["spatialChannelNames"]),
                "minimapChannelNames": list(payload["featureSchemaObservation"]["minimapChannelNames"]),
            },
            "featureSections": _build_base_feature_sections(payload),
            "labelSections": [],
            "sharedNameVocabulary": deepcopy(payload["sharedNameVocabulary"]),
        },
        "options": {
            "maxSelectedUnits": len(feature_tensors["currentSelectionIndices"]),
        },
        "replay": {
            "players": deepcopy(payload.get("replayPlayers") or []),
        },
        "superWeaponSchema": {
            "rechargeSecondsByType": deepcopy(payload.get("superWeaponRechargeSecondsByType") or {}),
        },
        "samples": [
            {
                "tick": int(payload["tick"]),
                "playerId": 0,
                "playerName": str(payload["playerName"]),
                "featureTensors": feature_tensors,
                "playerProduction": deepcopy(payload.get("playerProduction") or {}),
                "playerSuperWeapons": deepcopy(payload.get("playerSuperWeapons") or []),
            }
        ],
    }
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="buildOrderTrace",
        shape=[BUILD_ORDER_TRACE_LEN],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    augment_dataset_with_current_selection_summary(dataset)
    augment_dataset_with_available_action_mask(dataset)
    augment_dataset_with_owned_composition_bow(dataset)
    augment_dataset_with_scalar_core_identity(dataset)
    augment_dataset_with_time_encoding(dataset)
    augment_dataset_with_tech_state(dataset)
    augment_dataset_with_production_state(dataset)
    augment_dataset_with_super_weapon_state(dataset)
    augment_dataset_with_enemy_memory_bow(dataset)
    augment_dataset_with_entity_intent_summary(dataset)
    augment_dataset_with_entity_threat(dataset)

    return build_structured_section_tensors(
        dataset["samples"],
        dataset["schema"]["featureSections"],
        "featureTensors",
    )


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def align_v1_feature_sections(
    feature_sections: dict[str, torch.Tensor],
    *,
    action_vocab_size: int,
) -> dict[str, torch.Tensor]:
    aligned = dict(feature_sections)
    available_action_mask = aligned.get("availableActionMask")
    if available_action_mask is not None and int(available_action_mask.shape[-1]) != int(action_vocab_size):
        current_size = int(available_action_mask.shape[-1])
        if current_size < int(action_vocab_size):
            padding = torch.ones(
                available_action_mask.shape[0],
                int(action_vocab_size) - current_size,
                dtype=available_action_mask.dtype,
            )
            aligned["availableActionMask"] = torch.cat([available_action_mask, padding], dim=-1)
        else:
            aligned["availableActionMask"] = available_action_mask[..., : int(action_vocab_size)]
    return aligned


def _softmax_scores(raw_scores: dict[str, float]) -> dict[str, int]:
    if not raw_scores:
        return {}
    finite_items = [(name, score) for name, score in raw_scores.items() if math.isfinite(score)]
    if not finite_items:
        return {}
    names = [name for name, _ in finite_items]
    values = torch.tensor([score for _, score in finite_items], dtype=torch.float32)
    probabilities = torch.softmax(values, dim=0)
    scale = 10.0 * float(min(len(names), 5))
    normalized: dict[str, int] = {}
    for name, probability in zip(names, probabilities.tolist(), strict=True):
        score = int(round(probability * scale))
        if score > 0:
            normalized[name] = score
    return normalized


def _queue_option_names(input_payload: dict[str, Any]) -> dict[str, list[str]]:
    names_by_queue: dict[str, list[str]] = {}
    for queue in input_payload.get("queues", []):
        if not isinstance(queue, dict):
            continue
        queue_name = queue.get("queue")
        if not isinstance(queue_name, str):
            continue
        option_names = [
            str(option.get("name"))
            for option in queue.get("availableOptions", [])
            if isinstance(option, dict) and isinstance(option.get("name"), str)
        ]
        names_by_queue[queue_name] = option_names
    return names_by_queue


def _gather_v1_queue_scores(
    action_type_logits: torch.Tensor,
    input_payload: dict[str, Any],
) -> dict[str, dict[str, int]]:
    option_names_by_queue = _queue_option_names(input_payload)
    recommendations: dict[str, dict[str, int]] = {}
    for queue_name, option_names in option_names_by_queue.items():
        raw_scores: dict[str, float] = {}
        for option_name in option_names:
            candidate_names = [
                f"Queue::Add::{option_name}",
                f"Queue::AddNext::{option_name}",
            ]
            if queue_name in BUILDING_QUEUE_NAMES:
                candidate_names.append(f"PlaceBuilding::{option_name}")
            candidate_ids = [
                ACTION_TYPE_NAME_TO_ID[action_name]
                for action_name in candidate_names
                if action_name in ACTION_TYPE_NAME_TO_ID
            ]
            if not candidate_ids:
                continue
            raw_scores[option_name] = float(torch.max(action_type_logits[candidate_ids]).item())
        normalized = _softmax_scores(raw_scores)
        if normalized:
            recommendations[queue_name] = normalized
    return recommendations


def _safe_logit(logits: torch.Tensor, index: int) -> float:
    if index < 0 or index >= int(logits.shape[-1]):
        return float("-inf")
    return float(logits[index].item())


def _gather_v2_queue_scores(outputs: dict[str, torch.Tensor], input_payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    option_names_by_queue = _queue_option_names(input_payload)
    family_logits = outputs["actionFamilyLogits"][0]
    queue_update_type_logits = outputs["queueUpdateTypeLogits"][0]
    buildable_object_logits = outputs["buildableObjectLogits"][0]

    queue_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["Queue"]
    place_building_family_id = LABEL_LAYOUT_V2_ACTION_FAMILY_TO_ID["PlaceBuilding"]
    add_update_id = LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPE_TO_ID["Add"]
    add_next_update_id = LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPE_TO_ID["AddNext"]

    queue_family_score = _safe_logit(family_logits, queue_family_id)
    place_building_family_score = _safe_logit(family_logits, place_building_family_id)
    add_score = _safe_logit(queue_update_type_logits, add_update_id)
    add_next_score = _safe_logit(queue_update_type_logits, add_next_update_id)

    recommendations: dict[str, dict[str, int]] = {}
    for queue_name, option_names in option_names_by_queue.items():
        raw_scores: dict[str, float] = {}
        for option_name in option_names:
            buildable_object_id = BUILDABLE_OBJECT_NAME_TO_ID.get(option_name)
            if buildable_object_id is None:
                continue
            object_score = _safe_logit(buildable_object_logits, buildable_object_id)
            queue_style_score = queue_family_score + max(add_score, add_next_score) + object_score
            if queue_name in BUILDING_QUEUE_NAMES:
                raw_scores[option_name] = max(queue_style_score, place_building_family_score + object_score)
            else:
                raw_scores[option_name] = queue_style_score
        normalized = _softmax_scores(raw_scores)
        if normalized:
            recommendations[queue_name] = normalized
    return recommendations


def build_recommendations(
    *,
    variant: str,
    model: torch.nn.Module,
    device: torch.device,
    input_payload: dict[str, Any],
) -> dict[str, dict[str, int]] | None:
    checkpoint_features = input_payload.get("checkpointFeatures")
    if not isinstance(checkpoint_features, dict):
        return None

    feature_sections = build_runtime_feature_sections(checkpoint_features)
    if variant == "v1":
        action_vocab_size = int(getattr(getattr(model, "config", None), "action_vocab_size", len(ACTION_TYPE_NAME_TO_ID)))
        feature_sections = align_v1_feature_sections(
            feature_sections,
            action_vocab_size=action_vocab_size,
        )
    model_inputs = build_model_inputs(feature_sections)
    model_inputs = move_to_device(model_inputs, device)

    with torch.no_grad():
        outputs = model(
            model_inputs,
            teacher_forcing_targets=None,
            teacher_forcing_masks=None,
            teacher_forcing_mode="none",
        )

    if variant == "v2":
        recommendations = _gather_v2_queue_scores(outputs, input_payload)
    else:
        recommendations = _gather_v1_queue_scores(outputs["actionTypeLogits"][0], input_payload)
    return recommendations or None


def _json_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def _coerce_request_id(request_payload: dict[str, Any]) -> int:
    request_id = request_payload.get("id")
    if not isinstance(request_id, int):
        raise ValueError("Request is missing integer id.")
    return request_id


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    variant, model = load_runtime_model(args.checkpoint_path.resolve(), device)
    _json_response(
        {
            "type": "ready",
            "checkpointPath": str(args.checkpoint_path.resolve()),
            "device": str(device),
            "variant": variant,
        }
    )

    for line in sys.stdin:
        if not line.strip():
            continue
        request_id: int | None = None
        try:
            request_payload = json.loads(line)
            if not isinstance(request_payload, dict):
                raise ValueError("Request payload must be a JSON object.")
            request_id = _coerce_request_id(request_payload)
            input_payload = request_payload.get("input")
            if not isinstance(input_payload, dict):
                raise ValueError("Request is missing input payload.")
            output = build_recommendations(
                variant=variant,
                model=model,
                device=device,
                input_payload=input_payload,
            )
            _json_response({"id": request_id, "output": output})
        except Exception as exc:  # pragma: no cover - service boundary
            payload: dict[str, Any] = {"error": str(exc)}
            if request_id is not None:
                payload["id"] = request_id
            _json_response(payload)


if __name__ == "__main__":
    main()
