"""Static RA2 supervised-learning action dictionary.

This is the SL-package analogue of mini-AlphaStar's action dict: a versioned,
stable action-type vocabulary that `chronodivide-bot-sl` can use across runs.

Design notes:
- keep `py-chronodivide` generic; project-specific action ids live here
- keep a stable id space for checkpoints and tensor shards
- include unknown fallbacks for replay actions that are semantically understood
  but use unseen concrete names, such as queue items or buildings
"""

from __future__ import annotations

from typing import Any


STATIC_ACTION_DICT_VERSION = "ra2_sl_v2"
UNKNOWN_ACTION_TYPE_NAME = "<unk>"
UNKNOWN_QUEUE_ITEM_NAME = "<unk_item>"
UNKNOWN_BUILDING_NAME = "<unk_building>"
UNKNOWN_SUPER_WEAPON_NAME = "<unk_super_weapon>"

TARGET_MODE_NAMES = ["none", "tile", "object", "ore_tile"]
ORDER_TYPE_NAMES = [
    "Move",
    "ForceMove",
    "Attack",
    "ForceAttack",
    "AttackMove",
    "Guard",
    "GuardArea",
    "Capture",
    "Occupy",
    "Deploy",
    "DeploySelected",
    "Stop",
    "Cheer",
    "Dock",
    "Gather",
    "Repair",
    "Scatter",
    "EnterTransport",
    "PlaceBomb",
]
QUEUE_UPDATE_TYPE_NAMES = ["Add", "Cancel", "Hold"]
SUPER_WEAPON_TYPE_NAMES = [
    "MultiMissile",
    "IronCurtain",
    "LightningStorm",
    "ChronoSphere",
    "ChronoWarp",
    "ParaDrop",
    "AmerParaDrop",
]

# Seeded from the currently validated replay corpus and widened with unknown
# fallbacks so later replays can still map into the static dict.
QUEUE_ITEM_NAMES = sorted(
    [
        "ADOG",
        "AMRADR",
        "ATESLA",
        "CLEG",
        "CMIN",
        "DESO",
        "DOG",
        "E1",
        "E2",
        "ENGINEER",
        "FV",
        "GACSPH",
        "GAOREP",
        "GAPILE",
        "GAPILL",
        "GAPOWR",
        "GAREFN",
        "GATECH",
        "GAWEAP",
        "GAWEAT",
        "HARV",
        "HTK",
        "HTNK",
        "JUMPJET",
        "MGTK",
        "MTNK",
        "NAFLAK",
        "NAHAND",
        "NALASR",
        "NANRCT",
        "NAPOWR",
        "NARADR",
        "NAREFN",
        "NATECH",
        "NAWEAP",
        "ORCA",
        "SENGINEER",
        "SPY",
        "SREF",
    ]
)
PLACE_BUILDING_NAMES = sorted(
    [
        "AMRADR",
        "ATESLA",
        "GACSPH",
        "GAOREP",
        "GAPILE",
        "GAPILL",
        "GAPOWR",
        "GAREFN",
        "GATECH",
        "GAWEAP",
        "GAWEAT",
        "NAFLAK",
        "NAHAND",
        "NALASR",
        "NAPOWR",
        "NARADR",
        "NAREFN",
        "NATECH",
        "NAWEAP",
    ]
)

BASE_ACTION_NAMES = [
    UNKNOWN_ACTION_TYPE_NAME,
    "NoAction",
    "DropPlayer",
    "PingLocation",
    "SelectUnits",
    "SellObject",
    "ToggleRepair",
    "ResignGame",
]


def build_action_type_names() -> list[str]:
    names = list(BASE_ACTION_NAMES)

    for order_type_name in ORDER_TYPE_NAMES:
        for target_mode_name in TARGET_MODE_NAMES:
            names.append(f"Order::{order_type_name}::{target_mode_name}")

    for queue_update_type_name in QUEUE_UPDATE_TYPE_NAMES:
        names.append(f"Queue::{queue_update_type_name}::{UNKNOWN_QUEUE_ITEM_NAME}")
        for item_name in QUEUE_ITEM_NAMES:
            names.append(f"Queue::{queue_update_type_name}::{item_name}")

    names.append(f"PlaceBuilding::{UNKNOWN_BUILDING_NAME}")
    for building_name in PLACE_BUILDING_NAMES:
        names.append(f"PlaceBuilding::{building_name}")

    names.append(f"ActivateSuperWeapon::{UNKNOWN_SUPER_WEAPON_NAME}")
    for super_weapon_name in SUPER_WEAPON_TYPE_NAMES:
        names.append(f"ActivateSuperWeapon::{super_weapon_name}")

    return names


ACTION_TYPE_NAMES = build_action_type_names()
ACTION_TYPE_ID_TO_NAME = {index: name for index, name in enumerate(ACTION_TYPE_NAMES)}
ACTION_TYPE_NAME_TO_ID = {name: index for index, name in ACTION_TYPE_ID_TO_NAME.items()}


def _semantic_mask(
    *,
    uses_queue: bool = False,
    uses_units: bool = False,
    uses_target_entity: bool = False,
    uses_target_location: bool = False,
    uses_target_location_2: bool = False,
    uses_quantity: bool = False,
) -> dict[str, bool]:
    return {
        "usesQueue": uses_queue,
        "usesUnits": uses_units,
        "usesTargetEntity": uses_target_entity,
        "usesTargetLocation": uses_target_location,
        "usesTargetLocation2": uses_target_location_2,
        "usesQuantity": uses_quantity,
    }


def build_action_info(name: str) -> dict[str, Any]:
    semantic_mask = _semantic_mask()
    family = "unknown"

    if name == UNKNOWN_ACTION_TYPE_NAME:
        family = "unknown"
    elif name == "NoAction":
        family = "no_action"
    elif name == "DropPlayer":
        family = "drop_player"
    elif name == "PingLocation":
        family = "ping_location"
        semantic_mask = _semantic_mask(uses_target_location=True)
    elif name == "SelectUnits":
        family = "select_units"
        semantic_mask = _semantic_mask(uses_units=True)
    elif name == "SellObject":
        family = "sell_object"
        semantic_mask = _semantic_mask(uses_target_entity=True)
    elif name == "ToggleRepair":
        family = "toggle_repair"
        semantic_mask = _semantic_mask(uses_target_entity=True)
    elif name == "ResignGame":
        family = "resign_game"
    elif name.startswith("Order::"):
        family = "order_units"
        parts = name.split("::")
        target_mode_name = parts[2] if len(parts) >= 3 else "none"
        semantic_mask = _semantic_mask(
            uses_queue=True,
            uses_units=True,
            uses_target_entity=target_mode_name == "object",
            uses_target_location=target_mode_name in {"tile", "ore_tile"},
        )
    elif name.startswith("Queue::"):
        family = "update_queue"
        semantic_mask = _semantic_mask(uses_quantity=True)
    elif name.startswith("PlaceBuilding::"):
        family = "place_building"
        semantic_mask = _semantic_mask(uses_target_location=True)
    elif name.startswith("ActivateSuperWeapon::"):
        family = "activate_super_weapon"
        semantic_mask = _semantic_mask(uses_target_location=True, uses_target_location_2=True)

    return {
        "name": name,
        "family": family,
        **semantic_mask,
    }


ACTION_INFO_MASK = {
    action_type_id: {
        "id": action_type_id,
        **build_action_info(action_type_name),
    }
    for action_type_id, action_type_name in ACTION_TYPE_ID_TO_NAME.items()
}


def normalize_action_type_component(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    normalized = "".join(character if character.isalnum() else "_" for character in value.strip())
    normalized = normalized.strip("_")
    return normalized or fallback


def build_observed_action_type_name(
    *,
    raw_action_name: str | None,
    raw_action_id: int | None = None,
    order_type_name: str | None = None,
    target_mode_name: str | None = None,
    queue_update_type_name: str | None = None,
    item_name: str | None = None,
    building_name: str | None = None,
    super_weapon_name: str | None = None,
) -> str:
    if raw_action_name == "SelectUnitsAction":
        return "SelectUnits"

    if raw_action_name == "OrderUnitsAction":
        normalized_order_type = normalize_action_type_component(order_type_name, "UnknownOrder")
        normalized_target_mode = normalize_action_type_component(target_mode_name, "none")
        return f"Order::{normalized_order_type}::{normalized_target_mode}"

    if raw_action_name == "UpdateQueueAction":
        normalized_queue_update_type = normalize_action_type_component(queue_update_type_name, "UnknownQueueUpdate")
        normalized_item_name = normalize_action_type_component(item_name, "UnknownItem")
        return f"Queue::{normalized_queue_update_type}::{normalized_item_name}"

    if raw_action_name == "PlaceBuildingAction":
        normalized_building_name = normalize_action_type_component(building_name, "UnknownBuilding")
        return f"PlaceBuilding::{normalized_building_name}"

    if raw_action_name == "ActivateSuperWeaponAction":
        normalized_super_weapon_name = normalize_action_type_component(super_weapon_name, "UnknownSuperWeapon")
        return f"ActivateSuperWeapon::{normalized_super_weapon_name}"

    if raw_action_name == "SellObjectAction":
        return "SellObject"

    if raw_action_name == "ToggleRepairAction":
        return "ToggleRepair"

    if raw_action_name == "ResignGameAction":
        return "ResignGame"

    if raw_action_name == "NoAction":
        return "NoAction"

    if raw_action_name == "DropPlayerAction":
        return "DropPlayer"

    if raw_action_name == "PingLocationAction":
        return "PingLocation"

    return normalize_action_type_component(raw_action_name, f"RawAction_{raw_action_id if raw_action_id is not None else -1}")


def canonicalize_action_type_name(action_type_name: str | None) -> str:
    if not action_type_name:
        return UNKNOWN_ACTION_TYPE_NAME

    if action_type_name in ACTION_TYPE_NAME_TO_ID:
        return action_type_name

    if action_type_name.startswith("Queue::"):
        parts = action_type_name.split("::", 2)
        if len(parts) == 3 and parts[1] in QUEUE_UPDATE_TYPE_NAMES:
            fallback_name = f"Queue::{parts[1]}::{UNKNOWN_QUEUE_ITEM_NAME}"
            if fallback_name in ACTION_TYPE_NAME_TO_ID:
                return fallback_name

    if action_type_name.startswith("PlaceBuilding::"):
        fallback_name = f"PlaceBuilding::{UNKNOWN_BUILDING_NAME}"
        if fallback_name in ACTION_TYPE_NAME_TO_ID:
            return fallback_name

    if action_type_name.startswith("ActivateSuperWeapon::"):
        fallback_name = f"ActivateSuperWeapon::{UNKNOWN_SUPER_WEAPON_NAME}"
        if fallback_name in ACTION_TYPE_NAME_TO_ID:
            return fallback_name

    return UNKNOWN_ACTION_TYPE_NAME


def get_action_type_id(action_type_name: str | None) -> int:
    canonical_name = canonicalize_action_type_name(action_type_name)
    return ACTION_TYPE_NAME_TO_ID[canonical_name]
