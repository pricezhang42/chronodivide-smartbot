from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from action_dict import (
    BUILDABLE_OBJECT_ID_TO_NAME,
    ORDER_TYPE_NAMES,
    PLACE_BUILDING_NAMES,
    SUPER_WEAPON_TYPE_NAMES,
    TARGET_MODE_NAMES,
    UNKNOWN_BUILDABLE_OBJECT_NAME,
)
from live_production_advisor_service import (
    align_v1_feature_sections,
    build_runtime_feature_sections,
    load_runtime_model,
    move_to_device,
    resolve_device,
)
from model_lib.batch import build_model_inputs
from transform_lib.label_layout_v2 import (
    LABEL_LAYOUT_V2_ACTION_FAMILIES,
    LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES,
)

DEPLOYABLE_OBJECT_NAMES = {"AMCV", "SMCV"}
HARVESTER_OBJECT_NAMES = {"HARV", "CMIN"}
NO_TARGET_ORDER_TYPES = {"Deploy", "DeploySelected", "Stop", "Cheer"}
TILE_TARGET_ORDER_TYPES = {"Move", "ForceMove", "AttackMove", "GuardArea", "Scatter", "PlaceBomb"}
OBJECT_TARGET_ORDER_TYPES = {"Attack", "ForceAttack", "Capture", "Occupy", "Dock", "Repair", "EnterTransport"}
ORE_TILE_TARGET_ORDER_TYPES = {"Gather"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live checkpoint-backed full-control policy service.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _json_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def _coerce_request_id(request_payload: dict[str, Any]) -> int:
    request_id = request_payload.get("id")
    if not isinstance(request_id, int):
        raise ValueError("Request is missing integer id.")
    return request_id


def _build_noop_output(*, note: str, debug: dict[str, Any] | None = None) -> dict[str, Any]:
    base_debug = {
        "familyScore": 0.0,
        "topFamilies": [],
        "topSubtypes": [],
        "notes": [note],
    }
    if isinstance(debug, dict):
        base_debug.update(debug)
    return {
        "family": "Noop",
        "score": 0.0,
        "debug": base_debug,
        "order": None,
        "queue": None,
        "placeBuilding": None,
        "superWeapon": None,
        "targetEntityId": None,
    }


def _softmax_topk(logits: torch.Tensor, names: list[str], k: int = 3) -> list[dict[str, float]]:
    values = torch.softmax(logits.to(torch.float32), dim=0)
    count = min(k, int(values.shape[0]), len(names))
    top_values, top_indices = torch.topk(values, k=count)
    return [
        {
            "name": names[int(index.item())],
            "score": float(value.item()),
        }
        for value, index in zip(top_values, top_indices, strict=True)
    ]


def _scalar_feature_value(payload: dict[str, Any], name: str, default: int = 1) -> int:
    schema = payload.get("featureSchemaObservation", {})
    scalar_names = schema.get("scalarFeatureNames", [])
    scalar_values = payload.get("featureTensors", {}).get("scalar", [])
    if not isinstance(scalar_names, list) or not isinstance(scalar_values, list):
        return default
    if name not in scalar_names:
        return default
    index = scalar_names.index(name)
    if index < 0 or index >= len(scalar_values):
        return default
    value = scalar_values[index]
    return int(value) if isinstance(value, (int, float)) else default


def _grid_to_tile(payload: dict[str, Any], grid_x: int, grid_y: int, grid_size: int) -> dict[str, int]:
    map_width = max(1, _scalar_feature_value(payload, "map_width", 1))
    map_height = max(1, _scalar_feature_value(payload, "map_height", 1))
    max_x = max(1, map_width - 1)
    max_y = max(1, map_height - 1)
    tile_x = int(round(((grid_x + 0.5) / grid_size) * max_x))
    tile_y = int(round(((grid_y + 0.5) / grid_size) * max_y))
    tile_x = max(0, min(tile_x, map_width - 1))
    tile_y = max(0, min(tile_y, map_height - 1))
    return {"rx": tile_x, "ry": tile_y}


def _decode_spatial_target(spatial_logits: torch.Tensor, payload: dict[str, Any]) -> tuple[dict[str, int], float]:
    flat_logits = spatial_logits.reshape(-1)
    best_index = int(torch.argmax(flat_logits).item())
    size_y = int(spatial_logits.shape[-2])
    size_x = int(spatial_logits.shape[-1])
    grid_y = best_index // size_x
    grid_x = best_index % size_x
    return _grid_to_tile(payload, grid_x, grid_y, size_x), float(flat_logits[best_index].item())


def _entity_object_ids(payload: dict[str, Any]) -> list[int]:
    raw_ids = payload.get("entityObjectIds", [])
    if not isinstance(raw_ids, list):
        return []
    return [int(value) if isinstance(value, int) else -1 for value in raw_ids]


def _decode_target_entity(target_entity_logits: torch.Tensor, payload: dict[str, Any]) -> tuple[int | None, float]:
    entity_ids = _entity_object_ids(payload)
    if not entity_ids:
        return None, float("-inf")
    valid_indices = [index for index, object_id in enumerate(entity_ids) if object_id >= 0]
    if not valid_indices:
        return None, float("-inf")
    valid_logits = target_entity_logits[valid_indices]
    best_offset = int(torch.argmax(valid_logits).item())
    best_index = valid_indices[best_offset]
    return entity_ids[best_index], float(target_entity_logits[best_index].item())


def _decode_commanded_unit_ids(outputs: dict[str, torch.Tensor], payload: dict[str, Any]) -> list[int]:
    entity_ids = _entity_object_ids(payload)
    selected_ids = outputs["commandedUnitsSelectedIds"][0]
    selected_mask = outputs["commandedUnitsSelectedMask"][0]
    resolved: list[int] = []
    for slot_index in range(int(selected_ids.shape[0])):
        if int(selected_mask[slot_index].item()) <= 0:
            continue
        entity_index = int(selected_ids[slot_index].item())
        if entity_index < 0 or entity_index >= len(entity_ids):
            continue
        object_id = entity_ids[entity_index]
        if object_id < 0 or object_id in resolved:
            continue
        resolved.append(object_id)
    return resolved


def _feature_index(payload: dict[str, Any], name: str) -> int | None:
    names = payload.get("featureSchemaObservation", {}).get("entityFeatureNames", [])
    if not isinstance(names, list) or name not in names:
        return None
    return int(names.index(name))


def _entity_name_from_row(payload: dict[str, Any], entity_index: int) -> str | None:
    tokens = payload.get("featureTensors", {}).get("entityNameTokens", [])
    id_to_name = payload.get("sharedNameVocabulary", {}).get("idToName", [])
    if not isinstance(tokens, list) or not isinstance(id_to_name, list):
        return None
    if entity_index < 0 or entity_index >= len(tokens):
        return None
    token = tokens[entity_index]
    if not isinstance(token, int) or token < 0 or token >= len(id_to_name):
        return None
    name = id_to_name[token]
    return str(name) if isinstance(name, str) else None


def _selected_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    feature_tensors = payload.get("featureTensors", {})
    selection_indices = feature_tensors.get("currentSelectionIndices", [])
    resolved_mask = feature_tensors.get("currentSelectionResolvedMask", [])
    entity_features = feature_tensors.get("entityFeatures", [])
    entity_ids = _entity_object_ids(payload)
    if not isinstance(selection_indices, list) or not isinstance(resolved_mask, list) or not isinstance(entity_features, list):
        return []
    selected: list[dict[str, Any]] = []
    for slot_index, entity_index in enumerate(selection_indices):
        if slot_index >= len(resolved_mask) or int(resolved_mask[slot_index]) <= 0:
            continue
        if not isinstance(entity_index, int) or entity_index < 0 or entity_index >= len(entity_features):
            continue
        features = entity_features[entity_index]
        if not isinstance(features, list):
            continue
        selected.append(
            {
                "entityIndex": entity_index,
                "objectId": entity_ids[entity_index] if entity_index < len(entity_ids) else None,
                "name": _entity_name_from_row(payload, entity_index),
                "features": features,
            }
        )
    return selected


def _feature_value(entity: dict[str, Any], payload: dict[str, Any], feature_name: str) -> float:
    index = _feature_index(payload, feature_name)
    if index is None:
        return 0.0
    features = entity.get("features", [])
    if not isinstance(features, list) or index < 0 or index >= len(features):
        return 0.0
    value = features[index]
    return float(value) if isinstance(value, (int, float)) else 0.0


def _selected_entity_names(payload: dict[str, Any]) -> set[str]:
    return {
        name
        for entity in _selected_entities(payload)
        if isinstance((name := entity.get("name")), str) and name
    }


def _selected_have_mobile_units(payload: dict[str, Any]) -> bool:
    for entity in _selected_entities(payload):
        if _feature_value(entity, payload, "can_move") > 0.5:
            return True
    return False


def _selected_have_buildings(payload: dict[str, Any]) -> bool:
    for entity in _selected_entities(payload):
        if _feature_value(entity, payload, "object_building") > 0.5:
            return True
    return False


def _self_building_count(payload: dict[str, Any]) -> int:
    entity_features = payload.get("featureTensors", {}).get("entityFeatures", [])
    entity_mask = payload.get("featureTensors", {}).get("entityMask", [])
    if not isinstance(entity_features, list) or not isinstance(entity_mask, list):
        return 0
    building_feature_index = _feature_index(payload, "object_building")
    if building_feature_index is None:
        return 0
    count = 0
    for entity_index, features in enumerate(entity_features):
        if entity_index >= len(entity_mask) or int(entity_mask[entity_index]) <= 0:
            continue
        if not isinstance(features, list):
            continue
        if building_feature_index < len(features) and float(features[building_feature_index]) > 0.5:
            count += 1
    return count


def _selected_have_repairable(payload: dict[str, Any]) -> bool:
    for entity in _selected_entities(payload):
        if _feature_value(entity, payload, "has_wrench_repair") > 0.5:
            return True
    return False


def _selected_have_combat_capable(payload: dict[str, Any]) -> bool:
    for entity in _selected_entities(payload):
        if _feature_value(entity, payload, "can_move") <= 0.5:
            continue
        if _feature_value(entity, payload, "ammo") > 0:
            return True
        if _feature_value(entity, payload, "attack_state_prepare_to_fire") > 0.5:
            return True
        if _feature_value(entity, payload, "attack_state_check_range") > 0.5:
            return True
        if _feature_value(entity, payload, "attack_state_firing") > 0.5:
            return True
        if _feature_value(entity, payload, "primary_weapon_cooldown_ticks") > 0:
            return True
        if _feature_value(entity, payload, "secondary_weapon_cooldown_ticks") > 0:
            return True
    return False


def _available_object_names(payload: dict[str, Any]) -> set[str]:
    player_production = payload.get("playerProduction") or {}
    available_objects = player_production.get("availableObjects", [])
    if not isinstance(available_objects, list):
        return set()
    return {
        str(item["name"])
        for item in available_objects
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _queued_object_names(payload: dict[str, Any]) -> set[str]:
    player_production = payload.get("playerProduction") or {}
    queues = player_production.get("queues", [])
    names: set[str] = set()
    if not isinstance(queues, list):
        return names
    for queue in queues:
        if not isinstance(queue, dict):
            continue
        items = queue.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("objectName"), str):
                names.add(str(item["objectName"]))
    return names


def _pending_building_queue_names(payload: dict[str, Any]) -> set[str]:
    runtime_state = payload.get("runtimeState") or {}
    pending = runtime_state.get("pendingBuildingQueueByQueueName") or {}
    if not isinstance(pending, dict):
        return set()
    return {
        str(queue_name)
        for queue_name, object_name in pending.items()
        if isinstance(queue_name, str) and isinstance(object_name, str) and object_name
    }


def _queue_names_with_pending_buildings(payload: dict[str, Any]) -> set[str]:
    player_production = payload.get("playerProduction") or {}
    queues = player_production.get("queues", [])
    queue_names = _pending_building_queue_names(payload)
    if not isinstance(queues, list):
        return queue_names
    for queue in queues:
        if not isinstance(queue, dict):
            continue
        queue_name = queue.get("typeName")
        if not isinstance(queue_name, str):
            continue
        items = queue.get("items", [])
        if not isinstance(items, list):
            continue
        if any(
            isinstance(item, dict)
            and isinstance(item.get("objectName"), str)
            and str(item["objectName"]) in PLACE_BUILDING_NAMES
            for item in items
        ):
            queue_names.add(queue_name)
    return queue_names


def _available_objects_by_queue_type(payload: dict[str, Any]) -> dict[str, set[str]]:
    player_production = payload.get("playerProduction") or {}
    entries = player_production.get("availableObjectsByQueueType", [])
    names_by_queue: dict[str, set[str]] = {}
    if not isinstance(entries, list):
        return names_by_queue
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        queue_name = entry.get("queueTypeName")
        objects = entry.get("objects", [])
        if not isinstance(queue_name, str) or not isinstance(objects, list):
            continue
        object_names = {
            str(item["name"])
            for item in objects
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if object_names:
            names_by_queue[queue_name] = object_names
    return names_by_queue


def _available_add_object_names(payload: dict[str, Any]) -> set[str]:
    available_names_by_queue = _available_objects_by_queue_type(payload)
    if not available_names_by_queue:
        return _available_object_names(payload)
    queued_names = _queued_object_names(payload)
    ready_building_names = _ready_place_building_names(payload)
    blocked_queue_names = _queue_names_with_pending_buildings(payload)
    available_names: set[str] = set()
    for queue_name, queue_object_names in available_names_by_queue.items():
        if queue_name in blocked_queue_names:
            available_names.update(name for name in queue_object_names if name not in PLACE_BUILDING_NAMES)
            continue
        available_names.update(queue_object_names)
    suppressed_ready_building_names = {
        name for name in queued_names | ready_building_names if name in PLACE_BUILDING_NAMES
    }
    return {name for name in available_names if name not in suppressed_ready_building_names}


def _ready_place_building_names(payload: dict[str, Any]) -> set[str]:
    player_production = payload.get("playerProduction") or {}
    queues = player_production.get("queues", [])
    ready_names: set[str] = set()
    if not isinstance(queues, list):
        return ready_names
    for queue in queues:
        if not isinstance(queue, dict):
            continue
        if queue.get("statusName") != "Ready":
            continue
        items = queue.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            object_name = item.get("objectName")
            if isinstance(object_name, str) and object_name in PLACE_BUILDING_NAMES:
                ready_names.add(object_name)
    return ready_names


def _has_ready_superweapon(payload: dict[str, Any]) -> bool:
    super_weapons = payload.get("playerSuperWeapons", [])
    if not isinstance(super_weapons, list):
        return False
    for entry in super_weapons:
        if isinstance(entry, dict) and entry.get("statusName") == "Ready":
            return True
    return False


def _mask_invalid_logits(logits: torch.Tensor, allowed_indices: list[int]) -> torch.Tensor:
    if not allowed_indices:
        return torch.full_like(logits, -1e9)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    valid = [index for index in allowed_indices if 0 <= index < int(logits.shape[0])]
    if not valid:
        return torch.full_like(logits, -1e9)
    mask[valid] = True
    return logits.masked_fill(~mask, -1e9)


def _allowed_family_names(payload: dict[str, Any]) -> list[str]:
    available_add_names = _available_add_object_names(payload)
    queued_names = _queued_object_names(payload)
    ready_building_names = _ready_place_building_names(payload)
    selected_names = _selected_entity_names(payload)
    current_tick = int(payload.get("tick", 0))
    self_building_count = _self_building_count(payload)

    allowed: list[str] = []
    has_order_units = _selected_have_mobile_units(payload) or bool(selected_names & DEPLOYABLE_OBJECT_NAMES)
    if has_order_units:
        allowed.append("Order")
    if available_add_names or queued_names:
        allowed.append("Queue")
    if ready_building_names:
        allowed.append("PlaceBuilding")
    if _has_ready_superweapon(payload):
        allowed.append("ActivateSuperWeapon")
    if (
        _selected_have_buildings(payload)
        and self_building_count >= 2
        and current_tick >= 900
        and not bool(selected_names & DEPLOYABLE_OBJECT_NAMES)
    ):
        allowed.append("SellObject")
    if _selected_have_repairable(payload):
        allowed.append("ToggleRepair")
    return allowed


def _allowed_order_type_names(payload: dict[str, Any]) -> list[str]:
    selected_names = _selected_entity_names(payload)
    allowed: set[str] = set()
    has_mobile = _selected_have_mobile_units(payload)
    if has_mobile:
        allowed.update({"Move", "ForceMove", "Guard", "GuardArea", "Scatter"})
    if has_mobile and _selected_have_combat_capable(payload):
        allowed.update({"Attack", "ForceAttack", "AttackMove"})
    if selected_names & DEPLOYABLE_OBJECT_NAMES:
        allowed.update({"Deploy", "DeploySelected"})
    if selected_names & HARVESTER_OBJECT_NAMES:
        allowed.update({"Gather", "Dock"})
    return [name for name in ORDER_TYPE_NAMES if name in allowed]


def _allowed_target_mode_names(order_type_name: str | None) -> list[str]:
    if not order_type_name:
        return []
    if order_type_name in NO_TARGET_ORDER_TYPES:
        return ["none"]
    if order_type_name in TILE_TARGET_ORDER_TYPES:
        return ["tile"]
    if order_type_name in OBJECT_TARGET_ORDER_TYPES:
        return ["object"]
    if order_type_name in ORE_TILE_TARGET_ORDER_TYPES:
        return ["ore_tile", "tile"]
    return []


def _allowed_queue_update_type_names(payload: dict[str, Any]) -> list[str]:
    allowed: list[str] = []
    if _available_add_object_names(payload):
        allowed.append("Add")
    if _queued_object_names(payload):
        allowed.extend(["Cancel", "AddNext"])
    return [name for name in LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES if name in allowed]


def _allowed_buildable_object_names(payload: dict[str, Any], family_name: str, queue_update_type_name: str | None = None) -> set[str]:
    if family_name == "PlaceBuilding":
        return _ready_place_building_names(payload)
    if family_name != "Queue":
        return set()
    if queue_update_type_name == "Add":
        return _available_add_object_names(payload)
    if queue_update_type_name in {"Cancel", "AddNext"}:
        return _queued_object_names(payload)
    return set()


def _resolve_queue_name(payload: dict[str, Any], object_name: str) -> str | None:
    player_production = payload.get("playerProduction") or {}
    available_by_queue = player_production.get("availableObjectsByQueueType", [])
    if isinstance(available_by_queue, list):
        for entry in available_by_queue:
            if not isinstance(entry, dict):
                continue
            queue_name = entry.get("queueTypeName")
            objects = entry.get("objects", [])
            if not isinstance(queue_name, str) or not isinstance(objects, list):
                continue
            if any(isinstance(item, dict) and item.get("name") == object_name for item in objects):
                return queue_name

    queues = player_production.get("queues", [])
    if isinstance(queues, list):
        for queue in queues:
            if not isinstance(queue, dict):
                continue
            queue_name = queue.get("typeName")
            items = queue.get("items", [])
            if not isinstance(queue_name, str) or not isinstance(items, list):
                continue
            if any(isinstance(item, dict) and item.get("objectName") == object_name for item in items):
                return queue_name
    return None


def build_policy_action(
    *,
    variant: str,
    model: torch.nn.Module,
    device: torch.device,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    feature_payload = input_payload.get("featurePayload")
    if not isinstance(feature_payload, dict):
        return _build_noop_output(note="missing_feature_payload")

    feature_sections = build_runtime_feature_sections(feature_payload)
    if variant == "v1":
        action_vocab_size = int(getattr(getattr(model, "config", None), "action_vocab_size", 0))
        if action_vocab_size > 0:
            feature_sections = align_v1_feature_sections(feature_sections, action_vocab_size=action_vocab_size)

    model_inputs = build_model_inputs(feature_sections)
    model_inputs = move_to_device(model_inputs, device)

    with torch.no_grad():
        outputs = model(
            model_inputs,
            teacher_forcing_targets=None,
            teacher_forcing_masks=None,
            teacher_forcing_mode="none",
        )

    if variant != "v2":
        return _build_noop_output(
            note="full_control_runtime_requires_v2_checkpoint",
            debug={"topFamilies": [{"name": "Noop", "score": 1.0}]},
        )

    allowed_family_names = _allowed_family_names(feature_payload)
    if not allowed_family_names:
        return _build_noop_output(note="no_legal_family_from_live_state")

    family_logits = _mask_invalid_logits(
        outputs["actionFamilyLogits"][0],
        [LABEL_LAYOUT_V2_ACTION_FAMILIES.index(name) for name in allowed_family_names if name in LABEL_LAYOUT_V2_ACTION_FAMILIES],
    )
    family_probabilities = torch.softmax(family_logits.to(torch.float32), dim=0)
    family_index = int(torch.argmax(family_probabilities).item())
    family_name = LABEL_LAYOUT_V2_ACTION_FAMILIES[family_index]
    family_score = float(family_probabilities[family_index].item())

    debug: dict[str, Any] = {
        "familyScore": family_score,
        "topFamilies": _softmax_topk(family_logits, LABEL_LAYOUT_V2_ACTION_FAMILIES),
        "topSubtypes": [],
    }

    output = {
        "family": family_name,
        "score": family_score,
        "debug": debug,
        "order": None,
        "queue": None,
        "placeBuilding": None,
        "superWeapon": None,
        "targetEntityId": None,
    }

    if family_name == "Order":
        allowed_order_type_names = _allowed_order_type_names(feature_payload)
        if not allowed_order_type_names:
            return _build_noop_output(
                note="no_legal_order_type_from_live_state",
                debug=debug,
            )
        order_logits = _mask_invalid_logits(
            outputs["orderTypeLogits"][0],
            [ORDER_TYPE_NAMES.index(name) for name in allowed_order_type_names if name in ORDER_TYPE_NAMES],
        )
        order_index = int(torch.argmax(order_logits).item())
        order_name = ORDER_TYPE_NAMES[order_index] if order_index < len(ORDER_TYPE_NAMES) else None
        allowed_target_mode_names = _allowed_target_mode_names(order_name)
        if not allowed_target_mode_names:
            return _build_noop_output(note="no_legal_target_mode_from_live_state", debug=debug)
        target_mode_logits = _mask_invalid_logits(
            outputs["targetModeLogits"][0],
            [TARGET_MODE_NAMES.index(name) for name in allowed_target_mode_names if name in TARGET_MODE_NAMES],
        )
        target_mode_index = int(torch.argmax(target_mode_logits).item())
        target_entity_id, _ = _decode_target_entity(outputs["targetEntityLogits"][0], feature_payload)
        target_location, _ = _decode_spatial_target(outputs["targetLocationLogits"][0], feature_payload)
        target_location_2, _ = _decode_spatial_target(outputs["targetLocation2Logits"][0], feature_payload)
        output["order"] = {
            "orderType": order_name,
            "targetMode": TARGET_MODE_NAMES[target_mode_index] if target_mode_index < len(TARGET_MODE_NAMES) else None,
            "queueFlag": bool(torch.argmax(outputs["queueFlagLogits"][0]).item() == 1),
            "unitIds": _decode_commanded_unit_ids(outputs, feature_payload),
            "targetEntityId": target_entity_id,
            "targetLocation": target_location,
            "targetLocation2": target_location_2,
        }
        debug["topSubtypes"] = _softmax_topk(order_logits, ORDER_TYPE_NAMES)
        output["targetEntityId"] = target_entity_id
        return output

    if family_name == "Queue":
        allowed_queue_update_names = _allowed_queue_update_type_names(feature_payload)
        if not allowed_queue_update_names:
            return _build_noop_output(note="no_legal_queue_update_from_live_state", debug=debug)
        queue_update_logits = _mask_invalid_logits(
            outputs["queueUpdateTypeLogits"][0],
            [LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES.index(name) for name in allowed_queue_update_names if name in LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES],
        )
        queue_update_index = int(torch.argmax(queue_update_logits).item())
        queue_update_name = (
            LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES[queue_update_index]
            if queue_update_index < len(LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES)
            else None
        )
        allowed_buildable_names = _allowed_buildable_object_names(feature_payload, "Queue", queue_update_name)
        if not allowed_buildable_names:
            return _build_noop_output(note="no_legal_queue_object_from_live_state", debug=debug)
        buildable_object_logits = _mask_invalid_logits(
            outputs["buildableObjectLogits"][0],
            [
                buildable_object_id
                for buildable_object_id, buildable_object_name in BUILDABLE_OBJECT_ID_TO_NAME.items()
                if buildable_object_name in allowed_buildable_names
            ],
        )
        buildable_object_index = int(torch.argmax(buildable_object_logits).item())
        object_name = BUILDABLE_OBJECT_ID_TO_NAME.get(buildable_object_index, UNKNOWN_BUILDABLE_OBJECT_NAME)
        output["queue"] = {
            "queue": _resolve_queue_name(feature_payload, object_name),
            "updateType": queue_update_name,
            "objectName": None if object_name == UNKNOWN_BUILDABLE_OBJECT_NAME else object_name,
            "quantity": max(1, int(round(float(outputs["quantityPred"][0].item())))),
        }
        debug["topSubtypes"] = _softmax_topk(queue_update_logits, LABEL_LAYOUT_V2_QUEUE_UPDATE_TYPES)
        return output

    if family_name == "PlaceBuilding":
        allowed_buildable_names = _allowed_buildable_object_names(feature_payload, "PlaceBuilding")
        if not allowed_buildable_names:
            return _build_noop_output(note="no_ready_building_to_place", debug=debug)
        buildable_object_logits = _mask_invalid_logits(
            outputs["buildableObjectLogits"][0],
            [
                buildable_object_id
                for buildable_object_id, buildable_object_name in BUILDABLE_OBJECT_ID_TO_NAME.items()
                if buildable_object_name in allowed_buildable_names
            ],
        )
        buildable_object_index = int(torch.argmax(buildable_object_logits).item())
        object_name = BUILDABLE_OBJECT_ID_TO_NAME.get(buildable_object_index, UNKNOWN_BUILDABLE_OBJECT_NAME)
        target_location, _ = _decode_spatial_target(outputs["targetLocationLogits"][0], feature_payload)
        output["placeBuilding"] = {
            "buildingName": None if object_name == UNKNOWN_BUILDABLE_OBJECT_NAME else object_name,
            "targetLocation": target_location,
        }
        debug["topSubtypes"] = _softmax_topk(
            buildable_object_logits,
            [BUILDABLE_OBJECT_ID_TO_NAME.get(index, UNKNOWN_BUILDABLE_OBJECT_NAME) for index in range(int(buildable_object_logits.shape[0]))],
        )
        return output

    if family_name == "ActivateSuperWeapon":
        ready_super_weapon_names = {
            str(entry["typeName"])
            for entry in feature_payload.get("playerSuperWeapons", [])
            if isinstance(entry, dict) and entry.get("statusName") == "Ready" and isinstance(entry.get("typeName"), str)
        }
        if not ready_super_weapon_names:
            return _build_noop_output(note="no_ready_superweapon_from_live_state", debug=debug)
        super_weapon_logits = _mask_invalid_logits(
            outputs["superWeaponTypeLogits"][0],
            [
                index
                for index, type_name in enumerate(SUPER_WEAPON_TYPE_NAMES)
                if type_name in ready_super_weapon_names
            ],
        )
        super_weapon_index = int(torch.argmax(super_weapon_logits).item())
        target_location, _ = _decode_spatial_target(outputs["targetLocationLogits"][0], feature_payload)
        target_location_2, _ = _decode_spatial_target(outputs["targetLocation2Logits"][0], feature_payload)
        output["superWeapon"] = {
            "typeName": SUPER_WEAPON_TYPE_NAMES[super_weapon_index]
            if super_weapon_index < len(SUPER_WEAPON_TYPE_NAMES)
            else None,
            "targetLocation": target_location,
            "targetLocation2": target_location_2,
        }
        debug["topSubtypes"] = _softmax_topk(super_weapon_logits, SUPER_WEAPON_TYPE_NAMES)
        return output

    if family_name in {"SellObject", "ToggleRepair"}:
        target_entity_id, _ = _decode_target_entity(outputs["targetEntityLogits"][0], feature_payload)
        output["targetEntityId"] = target_entity_id
        return output

    return output


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
            output = build_policy_action(
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
