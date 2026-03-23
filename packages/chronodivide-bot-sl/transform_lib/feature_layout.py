from __future__ import annotations

import math
import re
from typing import Any

from action_dict import ACTION_TYPE_ID_TO_NAME, PLACE_BUILDING_NAMES, QUEUE_ITEM_NAMES
from transform_lib.common import LABEL_LAYOUT_V1_MISSING_INT

from transform_lib.schema_utils import append_schema_section, compute_flat_length


FEATURE_LAYOUT_V1_VERSION = "v1"

FEATURE_LAYOUT_V1_TOP_LEVEL_SECTIONS = [
    "scalarCore",
    "lastActionContext",
    "currentSelection",
    "availableActionMask",
    "ownedCompositionBow",
    "enemyMemoryBow",
    "buildOrderTrace",
    "techState",
    "productionState",
    "superWeaponState",
    "entity",
    "spatial",
    "minimap",
    "mapStatic",
]

FEATURE_LAYOUT_V1_PRIORITY_ZERO = [
    "availableActionMask",
    "ownedCompositionBow",
    "scalarCore.factionIdentity",
]

FEATURE_LAYOUT_V1_PRIORITY_ONE = [
    "buildOrderTrace",
    "techState",
    "productionState",
    "mapStatic.core",
]

FEATURE_LAYOUT_V1_PRIORITY_TWO = [
    "enemyMemoryBow",
    "superWeaponState",
    "mapStatic.extended",
    "entity.intentSummary",
]

HARVESTER_NAMES = {"HARV", "CMIN"}
DEPLOYABLE_NAMES = {"AMCV", "SMCV"}
CONSTRUCTION_YARD_NAMES = {"NACNST", "GACNST", "CACNST", "YACNST"}
SUPERWEAPON_BUILDING_REQUIREMENTS = {
    "MultiMissile": {"NAMISL"},
    "IronCurtain": {"NAIRON"},
    "LightningStorm": {"GAWEAT"},
    "ChronoSphere": {"GACSPH"},
    "ChronoWarp": {"GACSPH"},
}
SUPERWEAPON_STRUCTURE_NAMES = set().union(*SUPERWEAPON_BUILDING_REQUIREMENTS.values())
OWNED_COMPOSITION_UNKNOWN_NAME = "<unk>"
OWNED_COMPOSITION_VOCABULARY = [
    OWNED_COMPOSITION_UNKNOWN_NAME,
    *sorted(
        {
            *QUEUE_ITEM_NAMES,
            *PLACE_BUILDING_NAMES,
            *CONSTRUCTION_YARD_NAMES,
            *SUPERWEAPON_STRUCTURE_NAMES,
        }
    ),
]
OWNED_COMPOSITION_NAME_TO_ID = {
    name: index for index, name in enumerate(OWNED_COMPOSITION_VOCABULARY)
}
OWNED_COMPOSITION_ROW_NAMES = ["units", "buildings"]
BUILD_ORDER_TRACE_LEN = 20
IDENTITY_UNKNOWN_NAME = "<unk>"
SIDE_ID_TO_NAME = {
    0: "GDI",
    1: "Nod",
    2: "ThirdSide",
    3: "Civilian",
    4: "Mutant",
}
SIDE_VOCABULARY = [
    IDENTITY_UNKNOWN_NAME,
    *[SIDE_ID_TO_NAME[index] for index in sorted(SIDE_ID_TO_NAME)],
]
SIDE_NAME_TO_ID = {name: index for index, name in enumerate(SIDE_VOCABULARY)}
COUNTRY_VOCABULARY = [
    IDENTITY_UNKNOWN_NAME,
    "Americans",
    "Alliance",
    "Koreans",
    "French",
    "Germans",
    "British",
    "Africans",
    "Libyans",
    "Arabs",
    "Iraqis",
    "Confederation",
    "Cubans",
    "Russians",
    "YuriCountry",
    "Yuri",
]
COUNTRY_NAME_TO_ID = {name: index for index, name in enumerate(COUNTRY_VOCABULARY)}
POWER_BUILDING_NAMES = {"GAPOWR", "NAPOWR", "NANRCT", "YAPOWR"}
BARRACKS_NAMES = {"GAPILE", "NAHAND", "YABRCK"}
REFINERY_NAMES = {"GAREFN", "NAREFN", "YAREFN"}
FACTORY_NAMES = {"GAWEAP", "NAWEAP", "YAWEAP"}
AIRFIELD_NAMES = {"GAAIRC"}
NAVAL_YARD_NAMES = {"GAYARD", "NAYARD"}
RADAR_NAMES = {"AMRADR", "NARADR"}
SERVICE_DEPOT_NAMES = {"GADEPT", "NADEPT"}
TECH_CENTER_NAMES = {"GATECH", "NATECH"}
ORE_PURIFIER_NAMES = {"GAOREP"}
GAP_GENERATOR_NAMES = {"GAGAP"}
CLONING_VAT_NAMES = {"NACLON"}
PSYCHIC_SENSOR_NAMES = {"NAPSIS", "YAPSIS"}
NUCLEAR_SILO_NAMES = {"NAMISL"}
IRON_CURTAIN_NAMES = {"NAIRON"}
WEATHER_CONTROL_NAMES = {"GAWEAT"}
CHRONOSPHERE_NAMES = {"GACSPH"}
TECH_STATE_FLAG_NAMES = [
    "owned_has_construction_yard",
    "owned_has_power",
    "owned_has_barracks",
    "owned_has_refinery",
    "owned_has_factory",
    "owned_has_airfield",
    "owned_has_naval_yard",
    "owned_has_radar",
    "owned_has_service_depot",
    "owned_has_tech_center",
    "owned_has_ore_purifier",
    "owned_has_gap_generator",
    "owned_has_cloning_vat",
    "owned_has_psychic_sensor",
    "owned_has_nuclear_silo",
    "owned_has_iron_curtain",
    "owned_has_weather_control",
    "owned_has_chronosphere",
    "unlocks_infantry_production",
    "unlocks_vehicle_production",
    "unlocks_air_production",
    "unlocks_naval_production",
    "unlocks_tier2",
    "unlocks_tier3",
    "power_low",
    "power_satisfied",
]
PRODUCTION_QUEUE_TYPE_NAMES = ["Structures", "Armory", "Infantry", "Vehicles", "Aircrafts", "Ships"]
PRODUCTION_QUEUE_STATUS_NAMES = ["Idle", "Active", "OnHold", "Ready"]
PRODUCTION_STATE_GLOBAL_FEATURE_NAMES = [
    "production_max_tech_level",
    "production_build_speed_modifier",
    "production_queue_count",
    "production_available_object_count",
]
PRODUCTION_STATE_QUEUE_FEATURE_SUFFIXES = [
    "factory_count",
    "available_count",
    "has_queue",
    "status_idle",
    "status_active",
    "status_on_hold",
    "status_ready",
    "size",
    "max_size",
    "max_item_quantity",
    "has_items",
    "total_item_quantity",
    "first_item_name_token",
    "first_item_quantity",
    "first_item_progress",
    "first_item_cost",
]
SUPER_WEAPON_STATE_TYPE_NAMES = [
    "MultiMissile",
    "IronCurtain",
    "LightningStorm",
    "ChronoSphere",
    "ChronoWarp",
    "ParaDrop",
    "AmerParaDrop",
]
SUPER_WEAPON_STATUS_NAMES = ["Charging", "Paused", "Ready"]
SUPER_WEAPON_STATE_GLOBAL_FEATURE_NAMES = [
    "super_weapon_count",
    "super_weapon_unknown_type_count",
    "super_weapon_charging_count",
    "super_weapon_paused_count",
    "super_weapon_ready_count",
]
SUPER_WEAPON_STATE_PER_TYPE_FEATURE_SUFFIXES = [
    "count",
    "has",
    "status_charging",
    "status_paused",
    "status_ready",
    "charge_progress_01",
]
ENEMY_MEMORY_TECH_FLAG_NAMES = [
    "seen_enemy_has_construction_yard",
    "seen_enemy_has_power",
    "seen_enemy_has_barracks",
    "seen_enemy_has_refinery",
    "seen_enemy_has_factory",
    "seen_enemy_has_airfield",
    "seen_enemy_has_naval_yard",
    "seen_enemy_has_radar",
    "seen_enemy_has_service_depot",
    "seen_enemy_has_tech_center",
    "seen_enemy_has_ore_purifier",
    "seen_enemy_has_gap_generator",
    "seen_enemy_has_cloning_vat",
    "seen_enemy_has_psychic_sensor",
    "seen_enemy_has_nuclear_silo",
    "seen_enemy_has_iron_curtain",
    "seen_enemy_has_weather_control",
    "seen_enemy_has_chronosphere",
    "seen_enemy_unlocks_infantry_production",
    "seen_enemy_unlocks_vehicle_production",
    "seen_enemy_unlocks_air_production",
    "seen_enemy_unlocks_naval_production",
    "seen_enemy_unlocks_tier2",
    "seen_enemy_unlocks_tier3",
]
ENTITY_INTENT_SUMMARY_FEATURE_NAMES = [
    "intent_idle",
    "intent_move",
    "intent_attack",
    "intent_build",
    "intent_harvest",
    "intent_repair",
    "intent_factory_delivery",
    "intent_rally_point_valid",
    "intent_target_mode_none",
    "intent_target_mode_tile",
    "intent_target_mode_object",
    "intent_target_mode_resource",
    "intent_progress_01",
    "weapon_ready_any",
    "weapon_cooldown_progress_01",
    "intent_rally_distance_norm",
]
ENTITY_INTENT_WEAPON_COOLDOWN_CLAMP_TICKS = 90.0
TIME_ENCODING_DIM = 32
TIME_ENCODING_MAX_TICKS = 54000  # ~30 minutes at 30 tps
GAME_STATS_FEATURE_NAMES = [
    "stats_score",
    "stats_credits_gained",
    "stats_buildings_captured",
    "stats_units_built_aircraft",
    "stats_units_built_building",
    "stats_units_built_infantry",
    "stats_units_built_vehicle",
    "stats_units_killed_aircraft",
    "stats_units_killed_building",
    "stats_units_killed_infantry",
    "stats_units_killed_vehicle",
    "stats_units_lost_aircraft",
    "stats_units_lost_building",
    "stats_units_lost_infantry",
    "stats_units_lost_vehicle",
]
GAME_STATS_DIM = len(GAME_STATS_FEATURE_NAMES)
CURRENT_SELECTION_SUMMARY_FEATURE_NAMES = [
    "selected_infantry_count",
    "selected_vehicle_count",
    "selected_aircraft_count",
    "selected_building_count",
    "selected_can_move",
    "selected_can_attack",
    "selected_can_deploy",
    "selected_can_gather",
    "selected_can_repair",
    "selected_mixed_type",
]


def build_feature_layout_v1_contract() -> dict[str, object]:
    return {
        "version": FEATURE_LAYOUT_V1_VERSION,
        "topLevelSections": list(FEATURE_LAYOUT_V1_TOP_LEVEL_SECTIONS),
        "priority0": list(FEATURE_LAYOUT_V1_PRIORITY_ZERO),
        "priority1": list(FEATURE_LAYOUT_V1_PRIORITY_ONE),
        "priority2": list(FEATURE_LAYOUT_V1_PRIORITY_TWO),
        "notes": [
            "This module freezes the feature-layout V1 section names before they are fully implemented.",
            "The current transformer does not yet emit all of these sections.",
        ],
    }


def _feature_name_index(schema: dict[str, Any], feature_name: str) -> int:
    entity_feature_names = schema["observation"]["entityFeatureNames"]
    return int(entity_feature_names.index(feature_name))


def _optional_feature_name_index(schema: dict[str, Any], feature_name: str) -> int | None:
    entity_feature_names = schema["observation"]["entityFeatureNames"]
    try:
        return int(entity_feature_names.index(feature_name))
    except ValueError:
        return None


def _scalar_name_index(schema: dict[str, Any], feature_name: str) -> int:
    scalar_feature_names = schema["observation"]["scalarFeatureNames"]
    return int(scalar_feature_names.index(feature_name))


def _decode_name_token(schema: dict[str, Any], token: int) -> str | None:
    if token < 0:
        return None
    vocabulary = schema.get("sharedNameVocabulary", {})
    id_to_name = vocabulary.get("idToName", [])
    if 0 <= token < len(id_to_name):
        name = id_to_name[token]
        if isinstance(name, str):
            return name
    return None


def _normalize_owned_object_name(name: str | None) -> str:
    if not name or name in {"<pad>", "<unk>"}:
        return OWNED_COMPOSITION_UNKNOWN_NAME
    return name


def _identity_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "unknown"


def _build_production_state_feature_names() -> list[str]:
    feature_names = list(PRODUCTION_STATE_GLOBAL_FEATURE_NAMES)
    for queue_type_name in PRODUCTION_QUEUE_TYPE_NAMES:
        queue_prefix = _identity_suffix(queue_type_name).lower()
        feature_names.extend(
            f"production_{queue_prefix}_{feature_suffix}"
            for feature_suffix in PRODUCTION_STATE_QUEUE_FEATURE_SUFFIXES
        )
    return feature_names


PRODUCTION_STATE_FEATURE_NAMES = _build_production_state_feature_names()


def _build_super_weapon_state_feature_names() -> list[str]:
    feature_names = list(SUPER_WEAPON_STATE_GLOBAL_FEATURE_NAMES)
    for type_name in SUPER_WEAPON_STATE_TYPE_NAMES:
        type_prefix = _identity_suffix(type_name).lower()
        feature_names.extend(
            f"super_weapon_{type_prefix}_{feature_suffix}"
            for feature_suffix in SUPER_WEAPON_STATE_PER_TYPE_FEATURE_SUFFIXES
        )
    return feature_names


SUPER_WEAPON_STATE_FEATURE_NAMES = _build_super_weapon_state_feature_names()


def _name_token_from_shared_vocabulary(dataset: dict[str, Any], value: str | None) -> int:
    if not value:
        return LABEL_LAYOUT_V1_MISSING_INT
    vocabulary = dataset["schema"].get("sharedNameVocabulary", {})
    name_to_id = vocabulary.get("nameToId", {})
    if not isinstance(name_to_id, dict):
        return LABEL_LAYOUT_V1_MISSING_INT
    if value in name_to_id:
        return int(name_to_id[value])
    return int(name_to_id.get("<unk>", LABEL_LAYOUT_V1_MISSING_INT))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _safe_optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0.0 else None


def _entity_row_value(feature_row: Any, index: int | None, fallback: float = 0.0) -> float:
    if index is None:
        return float(fallback)
    return _safe_float(feature_row[index], fallback)


def _update_entity_feature_section_shape(dataset: dict[str, Any]) -> None:
    entity_feature_count = len(dataset["schema"]["observation"]["entityFeatureNames"])
    for section in dataset["schema"]["featureSections"]:
        if section["name"] == "entityFeatures":
            section["shape"] = [int(section["shape"][0]), entity_feature_count]
            break
    else:
        raise KeyError("Feature schema is missing entityFeatures section.")


def _find_replay_player(dataset: dict[str, Any], player_name: str) -> dict[str, Any] | None:
    for replay_player in dataset.get("replay", {}).get("players", []):
        if replay_player.get("name") == player_name:
            return replay_player
    return None


def _build_enemy_replay_players(dataset: dict[str, Any], player_name: str) -> list[dict[str, Any]]:
    replay_players = list(dataset.get("replay", {}).get("players", []))
    self_player = _find_replay_player(dataset, player_name)
    enemy_players = [player for player in replay_players if player.get("name") != player_name]
    if not self_player:
        return enemy_players

    self_team_id = self_player.get("teamId")
    team_filtered = [
        player
        for player in enemy_players
        if self_team_id is not None and player.get("teamId") is not None and player.get("teamId") != self_team_id
    ]
    if team_filtered:
        return team_filtered
    return enemy_players


def _side_bucket_name(player: dict[str, Any] | None) -> str:
    if not player:
        return IDENTITY_UNKNOWN_NAME
    side_id = player.get("sideId")
    if side_id is None:
        return IDENTITY_UNKNOWN_NAME
    return SIDE_ID_TO_NAME.get(int(side_id), IDENTITY_UNKNOWN_NAME)


def _country_bucket_name(player: dict[str, Any] | None) -> str:
    if not player:
        return IDENTITY_UNKNOWN_NAME
    country_name = player.get("countryName")
    if isinstance(country_name, str) and country_name in COUNTRY_NAME_TO_ID:
        return country_name
    return IDENTITY_UNKNOWN_NAME


def _one_hot(bucket_name: str, vocabulary_size: int, name_to_id: dict[str, int]) -> list[float]:
    values = [0.0] * vocabulary_size
    values[name_to_id.get(bucket_name, 0)] = 1.0
    return values


def _multi_hot(players: list[dict[str, Any]], vocabulary_size: int, bucket_fn: Any, name_to_id: dict[str, int]) -> list[float]:
    values = [0.0] * vocabulary_size
    if not players:
        values[0] = 1.0
        return values
    for player in players:
        values[name_to_id.get(bucket_fn(player), 0)] = 1.0
    return values


def _sum_spatial_channel(sample: dict[str, Any], schema: dict[str, Any], channel_name: str) -> float:
    channel_names = schema["observation"]["spatialChannelNames"]
    channel_index = int(channel_names.index(channel_name))
    plane = sample["featureTensors"]["spatial"][channel_index]
    return float(sum(sum(float(value) for value in row) for row in plane))


def _collect_self_entities(sample: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    schema = dataset["schema"]
    relation_self_index = _feature_name_index(schema, "relation_self")
    object_aircraft_index = _feature_name_index(schema, "object_aircraft")
    object_building_index = _feature_name_index(schema, "object_building")
    object_infantry_index = _feature_name_index(schema, "object_infantry")
    object_vehicle_index = _feature_name_index(schema, "object_vehicle")
    can_move_index = _feature_name_index(schema, "can_move")
    has_wrench_repair_index = _feature_name_index(schema, "has_wrench_repair")
    ammo_index = _feature_name_index(schema, "ammo")
    attack_state_check_range_index = _feature_name_index(schema, "attack_state_check_range")
    attack_state_prepare_to_fire_index = _feature_name_index(schema, "attack_state_prepare_to_fire")
    attack_state_fire_up_index = _feature_name_index(schema, "attack_state_fire_up")
    attack_state_firing_index = _feature_name_index(schema, "attack_state_firing")
    attack_state_just_fired_index = _feature_name_index(schema, "attack_state_just_fired")
    primary_weapon_cooldown_ticks_index = _optional_feature_name_index(schema, "primary_weapon_cooldown_ticks")
    secondary_weapon_cooldown_ticks_index = _optional_feature_name_index(schema, "secondary_weapon_cooldown_ticks")

    entity_mask = sample["featureTensors"]["entityMask"]
    entity_names = sample["featureTensors"]["entityNameTokens"]
    entity_features = sample["featureTensors"]["entityFeatures"]
    entities: list[dict[str, Any]] = []

    for entity_index, active in enumerate(entity_mask):
        if int(active) == 0:
            continue
        feature_row = entity_features[entity_index]
        if float(feature_row[relation_self_index]) <= 0.5:
            continue
        entity_name = _decode_name_token(schema, int(entity_names[entity_index]))
        can_attack = any(
            _entity_row_value(feature_row, index) > 0.5
            for index in [
                attack_state_check_range_index,
                attack_state_prepare_to_fire_index,
                attack_state_fire_up_index,
                attack_state_firing_index,
                attack_state_just_fired_index,
            ]
        ) or any(
            _entity_row_value(feature_row, index) > 0.0
            for index in [ammo_index, primary_weapon_cooldown_ticks_index, secondary_weapon_cooldown_ticks_index]
        )
        entities.append(
            {
                "index": entity_index,
                "name": entity_name,
                "isAircraft": float(feature_row[object_aircraft_index]) > 0.5,
                "isBuilding": float(feature_row[object_building_index]) > 0.5,
                "isInfantry": float(feature_row[object_infantry_index]) > 0.5,
                "isVehicle": float(feature_row[object_vehicle_index]) > 0.5,
                "canMove": float(feature_row[can_move_index]) > 0.5,
                "canAttack": bool(can_attack),
                "canDeploy": bool(entity_name and entity_name in DEPLOYABLE_NAMES),
                "canGather": bool(entity_name and entity_name in HARVESTER_NAMES),
                "canRepair": _entity_row_value(feature_row, has_wrench_repair_index) > 0.5,
            }
        )
    return entities


def _collect_enemy_entities(sample: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    schema = dataset["schema"]
    relation_enemy_index = _feature_name_index(schema, "relation_enemy")
    object_building_index = _feature_name_index(schema, "object_building")

    entity_mask = sample["featureTensors"]["entityMask"]
    entity_names = sample["featureTensors"]["entityNameTokens"]
    entity_features = sample["featureTensors"]["entityFeatures"]
    entities: list[dict[str, Any]] = []

    for entity_index, active in enumerate(entity_mask):
        if int(active) == 0:
            continue
        feature_row = entity_features[entity_index]
        if float(feature_row[relation_enemy_index]) <= 0.5:
            continue
        entities.append(
            {
                "index": entity_index,
                "name": _decode_name_token(schema, int(entity_names[entity_index])),
                "isBuilding": float(feature_row[object_building_index]) > 0.5,
            }
        )
    return entities


def _normalized_cooldown_progress(min_cooldown_ticks: float, has_weapon_signal: bool) -> float:
    if not has_weapon_signal:
        return 0.0
    if min_cooldown_ticks <= 0.0:
        return 1.0
    return max(
        0.0,
        1.0 - min(min_cooldown_ticks, ENTITY_INTENT_WEAPON_COOLDOWN_CLAMP_TICKS) / ENTITY_INTENT_WEAPON_COOLDOWN_CLAMP_TICKS,
    )


def _build_entity_intent_summary_row(feature_row: Any, entity_name: str | None, schema: dict[str, Any]) -> list[float]:
    object_building_index = _feature_name_index(schema, "object_building")
    is_idle_index = _feature_name_index(schema, "is_idle")
    can_move_index = _feature_name_index(schema, "can_move")
    build_status_build_up_index = _feature_name_index(schema, "build_status_build_up")
    build_status_build_down_index = _feature_name_index(schema, "build_status_build_down")
    attack_state_check_range_index = _feature_name_index(schema, "attack_state_check_range")
    attack_state_prepare_to_fire_index = _feature_name_index(schema, "attack_state_prepare_to_fire")
    attack_state_fire_up_index = _feature_name_index(schema, "attack_state_fire_up")
    attack_state_firing_index = _feature_name_index(schema, "attack_state_firing")
    attack_state_just_fired_index = _feature_name_index(schema, "attack_state_just_fired")
    has_wrench_repair_index = _feature_name_index(schema, "has_wrench_repair")
    harvested_ore_index = _feature_name_index(schema, "harvested_ore")
    harvested_gems_index = _feature_name_index(schema, "harvested_gems")
    hit_points_ratio_index = _feature_name_index(schema, "hit_points_ratio")
    tile_x_norm_index = _feature_name_index(schema, "tile_x_norm")
    tile_y_norm_index = _feature_name_index(schema, "tile_y_norm")
    factory_status_delivering_index = _optional_feature_name_index(schema, "factory_status_delivering")
    factory_has_delivery_index = _optional_feature_name_index(schema, "factory_has_delivery")
    rally_point_valid_index = _optional_feature_name_index(schema, "rally_point_valid")
    rally_x_norm_index = _optional_feature_name_index(schema, "rally_x_norm")
    rally_y_norm_index = _optional_feature_name_index(schema, "rally_y_norm")
    primary_weapon_cooldown_ticks_index = _optional_feature_name_index(schema, "primary_weapon_cooldown_ticks")
    secondary_weapon_cooldown_ticks_index = _optional_feature_name_index(schema, "secondary_weapon_cooldown_ticks")
    ammo_index = _feature_name_index(schema, "ammo")

    is_building = _entity_row_value(feature_row, object_building_index) > 0.5
    is_idle = _entity_row_value(feature_row, is_idle_index) > 0.5
    can_move = _entity_row_value(feature_row, can_move_index) > 0.5
    build_active = (
        _entity_row_value(feature_row, build_status_build_up_index) > 0.5
        or _entity_row_value(feature_row, build_status_build_down_index) > 0.5
    )
    attack_active = any(
        _entity_row_value(feature_row, index) > 0.5
        for index in [
            attack_state_check_range_index,
            attack_state_prepare_to_fire_index,
            attack_state_fire_up_index,
            attack_state_firing_index,
            attack_state_just_fired_index,
        ]
    )
    has_repair_signal = _entity_row_value(feature_row, has_wrench_repair_index) > 0.5
    harvest_load = _entity_row_value(feature_row, harvested_ore_index) + _entity_row_value(feature_row, harvested_gems_index)
    is_harvester = bool(entity_name and entity_name in HARVESTER_NAMES)
    harvest_active = is_harvester and harvest_load > 0.0
    factory_delivery_active = (
        _entity_row_value(feature_row, factory_status_delivering_index) > 0.5
        or _entity_row_value(feature_row, factory_has_delivery_index) > 0.5
    )
    rally_point_valid = _entity_row_value(feature_row, rally_point_valid_index) > 0.5
    moving_active = can_move and not is_idle and not attack_active and not build_active and not harvest_active
    repair_active = (
        has_repair_signal
        and not is_idle
        and not attack_active
        and not build_active
        and not harvest_active
        and not factory_delivery_active
    )

    target_mode_resource = harvest_active
    target_mode_object = attack_active or repair_active or factory_delivery_active
    target_mode_tile = moving_active or (is_building and rally_point_valid)
    target_mode_none = not (target_mode_resource or target_mode_object or target_mode_tile)

    primary_cooldown_ticks = max(_entity_row_value(feature_row, primary_weapon_cooldown_ticks_index), 0.0)
    secondary_cooldown_ticks = max(_entity_row_value(feature_row, secondary_weapon_cooldown_ticks_index), 0.0)
    positive_cooldowns = [value for value in [primary_cooldown_ticks, secondary_cooldown_ticks] if value > 0.0]
    min_cooldown_ticks = min(positive_cooldowns) if positive_cooldowns else 0.0
    has_weapon_signal = bool(
        positive_cooldowns
        or attack_active
        or _entity_row_value(feature_row, ammo_index) > 0.0
    )
    weapon_ready_any = 1.0 if has_weapon_signal and min_cooldown_ticks <= 0.0 else 0.0
    weapon_cooldown_progress_01 = _normalized_cooldown_progress(min_cooldown_ticks, has_weapon_signal)

    rally_distance_norm = 0.0
    if rally_point_valid:
        rally_distance_norm = min(
            1.0,
            (
                (_entity_row_value(feature_row, rally_x_norm_index) - _entity_row_value(feature_row, tile_x_norm_index)) ** 2
                + (_entity_row_value(feature_row, rally_y_norm_index) - _entity_row_value(feature_row, tile_y_norm_index)) ** 2
            )
            ** 0.5,
        )

    if build_active:
        intent_progress_01 = min(1.0, max(0.0, _entity_row_value(feature_row, hit_points_ratio_index)))
    elif attack_active:
        intent_progress_01 = weapon_cooldown_progress_01
    elif factory_delivery_active:
        intent_progress_01 = 1.0
    elif harvest_active or moving_active or repair_active:
        intent_progress_01 = 0.5
    else:
        intent_progress_01 = 0.0

    intent_idle = 1.0 if is_idle and not (build_active or attack_active or harvest_active or repair_active or factory_delivery_active) else 0.0

    return [
        intent_idle,
        1.0 if moving_active else 0.0,
        1.0 if attack_active else 0.0,
        1.0 if build_active else 0.0,
        1.0 if harvest_active else 0.0,
        1.0 if repair_active else 0.0,
        1.0 if factory_delivery_active else 0.0,
        1.0 if rally_point_valid else 0.0,
        1.0 if target_mode_none else 0.0,
        1.0 if target_mode_tile else 0.0,
        1.0 if target_mode_object else 0.0,
        1.0 if target_mode_resource else 0.0,
        intent_progress_01,
        weapon_ready_any,
        weapon_cooldown_progress_01,
        rally_distance_norm,
    ]


def build_entity_intent_summary(sample: dict[str, Any], dataset: dict[str, Any]) -> list[list[float]]:
    schema = dataset["schema"]
    entity_mask = sample["featureTensors"]["entityMask"]
    entity_names = sample["featureTensors"]["entityNameTokens"]
    entity_features = sample["featureTensors"]["entityFeatures"]
    intent_rows: list[list[float]] = []

    for entity_index, feature_row in enumerate(entity_features):
        if int(entity_mask[entity_index]) == 0:
            intent_rows.append([0.0] * len(ENTITY_INTENT_SUMMARY_FEATURE_NAMES))
            continue
        entity_name = _decode_name_token(schema, int(entity_names[entity_index]))
        intent_rows.append(_build_entity_intent_summary_row(feature_row, entity_name, schema))

    return intent_rows


def _collect_selected_entities(sample: dict[str, Any], self_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_indices = sample["featureTensors"]["currentSelectionIndices"]
    selected_mask = sample["featureTensors"]["currentSelectionMask"]
    entity_by_index = {entity["index"]: entity for entity in self_entities}
    selected_entities: list[dict[str, Any]] = []
    for selection_index, filled in enumerate(selected_mask):
        if int(filled) == 0:
            continue
        entity_index = int(selected_indices[selection_index])
        entity = entity_by_index.get(entity_index)
        if entity is not None:
            selected_entities.append(entity)
    return selected_entities


def build_current_selection_summary(sample: dict[str, Any], dataset: dict[str, Any]) -> list[int]:
    self_entities = _collect_self_entities(sample, dataset)
    selected_entities = _collect_selected_entities(sample, self_entities)

    infantry_count = sum(1 for entity in selected_entities if entity.get("isInfantry"))
    vehicle_count = sum(1 for entity in selected_entities if entity.get("isVehicle"))
    aircraft_count = sum(1 for entity in selected_entities if entity.get("isAircraft"))
    building_count = sum(1 for entity in selected_entities if entity.get("isBuilding"))
    other_count = max(0, len(selected_entities) - infantry_count - vehicle_count - aircraft_count - building_count)
    mixed_type = sum(
        int(count > 0)
        for count in [infantry_count, vehicle_count, aircraft_count, building_count, other_count]
    ) > 1

    return [
        infantry_count,
        vehicle_count,
        aircraft_count,
        building_count,
        int(any(entity.get("canMove") for entity in selected_entities)),
        int(any(entity.get("canAttack") for entity in selected_entities)),
        int(any(entity.get("canDeploy") for entity in selected_entities)),
        int(any(entity.get("canGather") for entity in selected_entities)),
        int(any(entity.get("canRepair") for entity in selected_entities)),
        int(mixed_type),
    ]


def _available_object_names_by_queue(player_production: dict[str, Any]) -> dict[str, set[str]]:
    available_names_by_queue: dict[str, set[str]] = {}
    for entry in player_production.get("availableObjectsByQueueType", []):
        if not isinstance(entry, dict):
            continue
        queue_type_name = entry.get("queueTypeName")
        if not isinstance(queue_type_name, str):
            continue
        names = {
            str(obj.get("name"))
            for obj in entry.get("objects", [])
            if isinstance(obj, dict) and isinstance(obj.get("name"), str)
        }
        available_names_by_queue[queue_type_name] = names
    return available_names_by_queue


def _queued_object_names(player_production: dict[str, Any]) -> set[str]:
    queued_names: set[str] = set()
    for queue in player_production.get("queues", []):
        if not isinstance(queue, dict):
            continue
        for item in queue.get("items", []):
            if isinstance(item, dict) and isinstance(item.get("objectName"), str):
                queued_names.add(str(item["objectName"]))
    return queued_names


def _is_order_action_available(
    action_type_name: str,
    *,
    has_self_entities: bool,
    has_selection: bool,
    has_selected_mobile: bool,
    has_selected_harvester: bool,
    has_selected_deployable: bool,
    selection_identity_unknown: bool,
) -> int:
    del action_type_name
    del has_selected_mobile
    del has_selected_harvester
    del has_selected_deployable
    del selection_identity_unknown

    if has_selection:
        return 1
    return 1 if has_self_entities else 0


def build_available_action_mask(sample: dict[str, Any], dataset: dict[str, Any]) -> list[int]:
    schema = dataset["schema"]
    scalar = sample["featureTensors"]["scalar"]
    player_production = sample.get("playerProduction") or {}
    player_super_weapons = sample.get("playerSuperWeapons") or []
    current_selection_count = int(sample["featureTensors"]["currentSelectionCount"][0])
    current_selection_resolved_count = int(sample["featureTensors"].get("currentSelectionResolvedCount", [0])[0])
    self_unit_count = float(scalar[_scalar_name_index(schema, "self_unit_count")])
    self_building_count = float(scalar[_scalar_name_index(schema, "self_building_count")])
    entity_mask = sample["featureTensors"]["entityMask"]
    available_names_by_queue = _available_object_names_by_queue(player_production)
    structure_available_names = available_names_by_queue.get("Structures", set())
    available_object_names = set().union(*available_names_by_queue.values()) if available_names_by_queue else set()
    queued_object_names = _queued_object_names(player_production)
    queue_count = int(_safe_float(player_production.get("queueCount"), 0.0))
    available_counts_by_name = {
        str(entry.get("queueTypeName")): _safe_float(entry.get("count"), 0.0)
        for entry in player_production.get("availableCountsByQueueType", [])
        if isinstance(entry, dict) and entry.get("queueTypeName") is not None
    }
    factory_counts_by_name = {
        str(entry.get("queueTypeName")): _safe_float(entry.get("count"), 0.0)
        for entry in player_production.get("factoryCounts", [])
        if isinstance(entry, dict) and entry.get("queueTypeName") is not None
    }
    has_build_sidebar = any(value > 0.0 for value in available_counts_by_name.values()) or any(
        value > 0.0 for value in factory_counts_by_name.values()
    ) or bool(available_object_names)
    known_super_weapon_types = {
        str(entry.get("typeName"))
        for entry in player_super_weapons
        if isinstance(entry, dict) and isinstance(entry.get("typeName"), str)
    }

    self_entities = _collect_self_entities(sample, dataset)
    selected_entities = _collect_selected_entities(sample, self_entities)
    self_entity_names = {entity["name"] for entity in self_entities if entity["name"]}
    selected_names = {entity["name"] for entity in selected_entities if entity["name"]}

    has_self_entities = (self_unit_count + self_building_count) > 0 or current_selection_count > 0 or any(
        int(active) != 0 for active in entity_mask
    )
    has_self_buildings = self_building_count > 0 or any(entity["isBuilding"] for entity in self_entities) or has_build_sidebar
    has_selection = current_selection_count > 0
    selection_identity_unknown = has_selection and (
        current_selection_resolved_count != len(selected_entities) or not selected_entities
    )
    has_selected_mobile = any(entity["canMove"] for entity in selected_entities)
    has_selected_harvester = bool(selected_names & HARVESTER_NAMES)
    has_selected_deployable = bool(selected_names & DEPLOYABLE_NAMES)
    has_construction_yard = any(
        name in CONSTRUCTION_YARD_NAMES or name.endswith("CNST")
        for name in self_entity_names
    )
    has_any_superweapon_structure = bool(self_entity_names & SUPERWEAPON_STRUCTURE_NAMES) or bool(known_super_weapon_types)

    mask = [1] * len(ACTION_TYPE_ID_TO_NAME)
    for action_type_id, action_type_name in ACTION_TYPE_ID_TO_NAME.items():
        enabled = 1

        if action_type_name in {"<unk>", "DropPlayer", "PingLocation"}:
            enabled = 0
        elif action_type_name == "NoAction":
            enabled = 1
        elif action_type_name == "SelectUnits":
            enabled = 1 if has_self_entities else 0
        elif action_type_name == "SellObject":
            enabled = 1 if has_self_buildings else 0
        elif action_type_name == "ToggleRepair":
            enabled = 1 if has_self_entities else 0
        elif action_type_name == "ResignGame":
            enabled = 1
        elif action_type_name.startswith("Order::"):
            enabled = _is_order_action_available(
                action_type_name,
                has_self_entities=has_self_entities,
                has_selection=has_selection,
                has_selected_mobile=has_selected_mobile,
                has_selected_harvester=has_selected_harvester,
                has_selected_deployable=has_selected_deployable,
                selection_identity_unknown=selection_identity_unknown,
            )
        elif action_type_name.startswith("Queue::"):
            _, queue_update_type_name, item_name = action_type_name.split("::", 2)
            if queue_update_type_name == "Add":
                if item_name == "<unk_item>":
                    enabled = 1 if (available_object_names or has_build_sidebar or has_self_buildings) else 0
                else:
                    enabled = 1 if (
                        item_name in available_object_names or (not available_object_names and (has_build_sidebar or has_self_buildings))
                    ) else 0
            else:
                if item_name == "<unk_item>":
                    enabled = 1 if (queued_object_names or queue_count > 0 or has_self_buildings) else 0
                else:
                    enabled = 1 if (
                        item_name in queued_object_names
                        or queue_count > 0
                        or (not queued_object_names and has_self_buildings)
                    ) else 0
        elif action_type_name.startswith("PlaceBuilding::"):
            building_name = action_type_name.split("::", 1)[1]
            if building_name == "<unk_building>":
                enabled = 1 if (structure_available_names or has_construction_yard or has_build_sidebar) else 0
            else:
                enabled = 1 if (
                    building_name in available_object_names
                    or (
                        not available_object_names
                        and (building_name in structure_available_names or has_construction_yard or has_build_sidebar)
                    )
                ) else 0
        elif action_type_name.startswith("ActivateSuperWeapon::"):
            superweapon_name = action_type_name.split("::", 1)[1]
            if known_super_weapon_types:
                if superweapon_name in known_super_weapon_types:
                    enabled = 1
                elif superweapon_name == "<unk_super_weapon>":
                    enabled = 1
                else:
                    enabled = 0
            else:
                required_buildings = SUPERWEAPON_BUILDING_REQUIREMENTS.get(superweapon_name)
                if required_buildings is not None:
                    enabled = 1 if (self_entity_names & required_buildings) else 0
                else:
                    enabled = 1 if has_any_superweapon_structure else 0

        mask[action_type_id] = int(enabled)

    return mask


def build_owned_composition_bow(sample: dict[str, Any], dataset: dict[str, Any]) -> list[list[int]]:
    unit_counts = [0] * len(OWNED_COMPOSITION_VOCABULARY)
    building_counts = [0] * len(OWNED_COMPOSITION_VOCABULARY)

    for entity in _collect_self_entities(sample, dataset):
        name = _normalize_owned_object_name(entity["name"])
        bucket = OWNED_COMPOSITION_NAME_TO_ID.get(name, 0)
        target_counts = building_counts if entity["isBuilding"] else unit_counts
        target_counts[bucket] += 1

    return [unit_counts, building_counts]


def build_scalar_core_identity(sample: dict[str, Any], dataset: dict[str, Any]) -> list[float]:
    player_name = str(sample["playerName"])
    self_player = _find_replay_player(dataset, player_name)
    enemy_players = _build_enemy_replay_players(dataset, player_name)

    return [
        *_one_hot(_side_bucket_name(self_player), len(SIDE_VOCABULARY), SIDE_NAME_TO_ID),
        *_multi_hot(enemy_players, len(SIDE_VOCABULARY), _side_bucket_name, SIDE_NAME_TO_ID),
        *_one_hot(_country_bucket_name(self_player), len(COUNTRY_VOCABULARY), COUNTRY_NAME_TO_ID),
        *_multi_hot(enemy_players, len(COUNTRY_VOCABULARY), _country_bucket_name, COUNTRY_NAME_TO_ID),
    ]


def build_tech_state(sample: dict[str, Any], dataset: dict[str, Any]) -> list[int]:
    schema = dataset["schema"]
    scalar = sample["featureTensors"]["scalar"]
    self_entity_names = {entity["name"] for entity in _collect_self_entities(sample, dataset) if entity["name"]}
    power_low = float(scalar[_scalar_name_index(schema, "power_low")]) > 0.5

    flags = {
        "owned_has_construction_yard": bool(self_entity_names & CONSTRUCTION_YARD_NAMES),
        "owned_has_power": bool(self_entity_names & POWER_BUILDING_NAMES),
        "owned_has_barracks": bool(self_entity_names & BARRACKS_NAMES),
        "owned_has_refinery": bool(self_entity_names & REFINERY_NAMES),
        "owned_has_factory": bool(self_entity_names & FACTORY_NAMES),
        "owned_has_airfield": bool(self_entity_names & AIRFIELD_NAMES),
        "owned_has_naval_yard": bool(self_entity_names & NAVAL_YARD_NAMES),
        "owned_has_radar": bool(self_entity_names & RADAR_NAMES),
        "owned_has_service_depot": bool(self_entity_names & SERVICE_DEPOT_NAMES),
        "owned_has_tech_center": bool(self_entity_names & TECH_CENTER_NAMES),
        "owned_has_ore_purifier": bool(self_entity_names & ORE_PURIFIER_NAMES),
        "owned_has_gap_generator": bool(self_entity_names & GAP_GENERATOR_NAMES),
        "owned_has_cloning_vat": bool(self_entity_names & CLONING_VAT_NAMES),
        "owned_has_psychic_sensor": bool(self_entity_names & PSYCHIC_SENSOR_NAMES),
        "owned_has_nuclear_silo": bool(self_entity_names & NUCLEAR_SILO_NAMES),
        "owned_has_iron_curtain": bool(self_entity_names & IRON_CURTAIN_NAMES),
        "owned_has_weather_control": bool(self_entity_names & WEATHER_CONTROL_NAMES),
        "owned_has_chronosphere": bool(self_entity_names & CHRONOSPHERE_NAMES),
    }
    flags.update(
        {
            "unlocks_infantry_production": flags["owned_has_barracks"],
            "unlocks_vehicle_production": flags["owned_has_factory"],
            "unlocks_air_production": flags["owned_has_airfield"],
            "unlocks_naval_production": flags["owned_has_naval_yard"],
            "unlocks_tier2": flags["owned_has_radar"] or flags["owned_has_service_depot"] or flags["owned_has_tech_center"],
            "unlocks_tier3": flags["owned_has_tech_center"],
            "power_low": power_low,
            "power_satisfied": not power_low,
        }
    )
    return [int(flags[name]) for name in TECH_STATE_FLAG_NAMES]


def build_production_state(sample: dict[str, Any], dataset: dict[str, Any]) -> list[float]:
    player_production = sample.get("playerProduction") or {}
    queues = player_production.get("queues", [])
    queue_by_name = {
        str(queue.get("typeName")): queue
        for queue in queues
        if isinstance(queue, dict) and isinstance(queue.get("typeName"), str)
    }
    factory_count_by_name = {
        str(entry.get("queueTypeName")): _safe_float(entry.get("count"), 0.0)
        for entry in player_production.get("factoryCounts", [])
        if isinstance(entry, dict) and entry.get("queueTypeName") is not None
    }
    available_count_by_name = {
        str(entry.get("queueTypeName")): _safe_float(entry.get("count"), 0.0)
        for entry in player_production.get("availableCountsByQueueType", [])
        if isinstance(entry, dict) and entry.get("queueTypeName") is not None
    }

    values: list[float] = [
        _safe_float(player_production.get("maxTechLevel"), 0.0),
        _safe_float(player_production.get("buildSpeedModifier"), 0.0),
        _safe_float(player_production.get("queueCount"), 0.0),
        _safe_float(player_production.get("availableObjectCount"), 0.0),
    ]

    for queue_type_name in PRODUCTION_QUEUE_TYPE_NAMES:
        queue = queue_by_name.get(queue_type_name)
        queue_items = queue.get("items", []) if isinstance(queue, dict) else []
        first_item = queue_items[0] if queue_items else {}
        first_item_name = first_item.get("objectName") if isinstance(first_item, dict) else None
        first_item_name_token = _name_token_from_shared_vocabulary(dataset, str(first_item_name) if first_item_name else None)
        queue_status_name = queue.get("statusName") if isinstance(queue, dict) else None
        total_item_quantity = 0.0
        for item in queue_items:
            if isinstance(item, dict):
                total_item_quantity += _safe_float(item.get("quantity"), 0.0)

        values.extend(
            [
                factory_count_by_name.get(queue_type_name, 0.0),
                available_count_by_name.get(queue_type_name, 0.0),
                1.0 if queue else 0.0,
                1.0 if queue_status_name == "Idle" else 0.0,
                1.0 if queue_status_name == "Active" else 0.0,
                1.0 if queue_status_name == "OnHold" else 0.0,
                1.0 if queue_status_name == "Ready" else 0.0,
                _safe_float(queue.get("size") if isinstance(queue, dict) else None, 0.0),
                _safe_float(queue.get("maxSize") if isinstance(queue, dict) else None, 0.0),
                _safe_float(queue.get("maxItemQuantity") if isinstance(queue, dict) else None, 0.0),
                1.0 if queue_items else 0.0,
                total_item_quantity,
                float(first_item_name_token),
                _safe_float(first_item.get("quantity") if isinstance(first_item, dict) else None, 0.0),
                _safe_float(first_item.get("progress") if isinstance(first_item, dict) else None, 0.0),
                _safe_float(first_item.get("cost") if isinstance(first_item, dict) else None, 0.0),
            ]
        )

    return values


def _normalized_super_weapon_progress(
    status_name: str | None,
    timer_seconds: float,
    nominal_recharge_seconds: float | None,
) -> float:
    if status_name == "Ready":
        return 1.0
    if nominal_recharge_seconds is None or nominal_recharge_seconds <= 0.0:
        return 0.0
    progress = 1.0 - (max(0.0, timer_seconds) / nominal_recharge_seconds)
    return max(0.0, min(1.0, progress))


def build_super_weapon_state(sample: dict[str, Any], dataset: dict[str, Any]) -> list[float]:
    player_super_weapons = sample.get("playerSuperWeapons") or []
    known_type_names = set(SUPER_WEAPON_STATE_TYPE_NAMES)
    recharge_seconds_by_type = ((dataset.get("superWeaponSchema") or {}).get("rechargeSecondsByType") or {})
    status_counts = {status_name: 0.0 for status_name in SUPER_WEAPON_STATUS_NAMES}
    type_summaries = {
        type_name: {
            "count": 0.0,
            "statusFlags": {status_name: 0.0 for status_name in SUPER_WEAPON_STATUS_NAMES},
            "chargeProgress01": 0.0,
        }
        for type_name in SUPER_WEAPON_STATE_TYPE_NAMES
    }

    total_count = 0.0
    unknown_type_count = 0.0
    for entry in player_super_weapons:
        if not isinstance(entry, dict):
            continue

        type_name = entry.get("typeName")
        status_name = entry.get("statusName")
        timer_seconds = _safe_float(entry.get("timerSeconds"), 0.0)
        total_count += 1.0

        if isinstance(status_name, str) and status_name in status_counts:
            status_counts[status_name] += 1.0

        if not isinstance(type_name, str) or type_name not in known_type_names:
            unknown_type_count += 1.0
            continue

        summary = type_summaries[type_name]
        summary["count"] += 1.0
        if isinstance(status_name, str) and status_name in summary["statusFlags"]:
            summary["statusFlags"][status_name] = 1.0
        nominal_recharge_seconds = _safe_optional_float(recharge_seconds_by_type.get(type_name))
        progress = _normalized_super_weapon_progress(status_name, timer_seconds, nominal_recharge_seconds)
        summary["chargeProgress01"] = max(float(summary["chargeProgress01"]), progress)

    values: list[float] = [
        total_count,
        unknown_type_count,
        status_counts["Charging"],
        status_counts["Paused"],
        status_counts["Ready"],
    ]
    for type_name in SUPER_WEAPON_STATE_TYPE_NAMES:
        summary = type_summaries[type_name]
        values.extend(
            [
                float(summary["count"]),
                1.0 if summary["count"] > 0 else 0.0,
                float(summary["statusFlags"]["Charging"]),
                float(summary["statusFlags"]["Paused"]),
                float(summary["statusFlags"]["Ready"]),
                float(summary["chargeProgress01"]),
            ]
        )

    return values


def build_enemy_memory_tech_flags(seen_enemy_building_names: set[str]) -> list[int]:
    flags = {
        "seen_enemy_has_construction_yard": bool(seen_enemy_building_names & CONSTRUCTION_YARD_NAMES),
        "seen_enemy_has_power": bool(seen_enemy_building_names & POWER_BUILDING_NAMES),
        "seen_enemy_has_barracks": bool(seen_enemy_building_names & BARRACKS_NAMES),
        "seen_enemy_has_refinery": bool(seen_enemy_building_names & REFINERY_NAMES),
        "seen_enemy_has_factory": bool(seen_enemy_building_names & FACTORY_NAMES),
        "seen_enemy_has_airfield": bool(seen_enemy_building_names & AIRFIELD_NAMES),
        "seen_enemy_has_naval_yard": bool(seen_enemy_building_names & NAVAL_YARD_NAMES),
        "seen_enemy_has_radar": bool(seen_enemy_building_names & RADAR_NAMES),
        "seen_enemy_has_service_depot": bool(seen_enemy_building_names & SERVICE_DEPOT_NAMES),
        "seen_enemy_has_tech_center": bool(seen_enemy_building_names & TECH_CENTER_NAMES),
        "seen_enemy_has_ore_purifier": bool(seen_enemy_building_names & ORE_PURIFIER_NAMES),
        "seen_enemy_has_gap_generator": bool(seen_enemy_building_names & GAP_GENERATOR_NAMES),
        "seen_enemy_has_cloning_vat": bool(seen_enemy_building_names & CLONING_VAT_NAMES),
        "seen_enemy_has_psychic_sensor": bool(seen_enemy_building_names & PSYCHIC_SENSOR_NAMES),
        "seen_enemy_has_nuclear_silo": bool(seen_enemy_building_names & NUCLEAR_SILO_NAMES),
        "seen_enemy_has_iron_curtain": bool(seen_enemy_building_names & IRON_CURTAIN_NAMES),
        "seen_enemy_has_weather_control": bool(seen_enemy_building_names & WEATHER_CONTROL_NAMES),
        "seen_enemy_has_chronosphere": bool(seen_enemy_building_names & CHRONOSPHERE_NAMES),
    }
    flags.update(
        {
            "seen_enemy_unlocks_infantry_production": flags["seen_enemy_has_barracks"],
            "seen_enemy_unlocks_vehicle_production": flags["seen_enemy_has_factory"],
            "seen_enemy_unlocks_air_production": flags["seen_enemy_has_airfield"],
            "seen_enemy_unlocks_naval_production": flags["seen_enemy_has_naval_yard"],
            "seen_enemy_unlocks_tier2": flags["seen_enemy_has_radar"]
            or flags["seen_enemy_has_service_depot"]
            or flags["seen_enemy_has_tech_center"],
            "seen_enemy_unlocks_tier3": flags["seen_enemy_has_tech_center"],
        }
    )
    return [int(flags[name]) for name in ENEMY_MEMORY_TECH_FLAG_NAMES]


def _is_build_order_action_type_name(action_type_name: str | None) -> bool:
    if not isinstance(action_type_name, str):
        return False
    if action_type_name.startswith("Queue::Add::"):
        return True
    if action_type_name.startswith("PlaceBuilding::"):
        return True
    if action_type_name.startswith("Order::Deploy::"):
        return True
    if action_type_name.startswith("Order::DeploySelected::"):
        return True
    return False


def _should_collapse_build_order_duplicate(action_type_name: str | None) -> bool:
    if not isinstance(action_type_name, str):
        return False
    return action_type_name.startswith("Order::Deploy::") or action_type_name.startswith("Order::DeploySelected::")


def _build_order_trace_tensor(history: list[int]) -> list[int]:
    trace = list(history[:BUILD_ORDER_TRACE_LEN])
    if len(trace) < BUILD_ORDER_TRACE_LEN:
        trace.extend([LABEL_LAYOUT_V1_MISSING_INT] * (BUILD_ORDER_TRACE_LEN - len(trace)))
    return trace


def augment_dataset_with_available_action_mask(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="availableActionMask",
        shape=[len(ACTION_TYPE_ID_TO_NAME)],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "lastActionContext",
            "availableActionMask",
        }
    )
    feature_layout_v1["availableActionMask"] = {
        "version": "v1_conservative_observation_driven",
        "shape": [len(ACTION_TYPE_ID_TO_NAME)],
        "policy": "observation_driven_conservative",
        "notes": [
            "This first pass disables only clearly impossible actions and leaves ambiguous ones enabled.",
            "The mask is derived in chronodivide-bot-sl from current selection, current player production summaries, current player super-weapon state, and static action-type semantics.",
            "It intentionally avoids hidden enemy state and any omniscient legality solver.",
        ],
    }

    for sample in samples:
        sample["featureTensors"]["availableActionMask"] = build_available_action_mask(sample, dataset)


def augment_dataset_with_current_selection_summary(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="currentSelectionSummary",
        shape=[len(CURRENT_SELECTION_SUMMARY_FEATURE_NAMES)],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "currentSelection.summary",
        }
    )
    current_selection_meta = dict(feature_layout_v1.get("currentSelection", {}))
    current_selection_meta["summary"] = {
        "version": "v0_capability_summary",
        "shape": [len(CURRENT_SELECTION_SUMMARY_FEATURE_NAMES)],
        "featureNames": list(CURRENT_SELECTION_SUMMARY_FEATURE_NAMES),
        "notes": [
            "This section is derived from the currently resolved self selection, not from hidden or omniscient state.",
            "Counts and capability flags can undercount if part of the selection could not be resolved into the current entity tensor.",
            "selected_can_attack uses a conservative heuristic from visible entity weapon and attack-state signals.",
        ],
    }
    feature_layout_v1["currentSelection"] = current_selection_meta

    for sample in samples:
        sample["featureTensors"]["currentSelectionSummary"] = build_current_selection_summary(sample, dataset)


def augment_dataset_with_owned_composition_bow(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="ownedCompositionBow",
        shape=[len(OWNED_COMPOSITION_ROW_NAMES), len(OWNED_COMPOSITION_VOCABULARY)],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "lastActionContext",
            "availableActionMask",
            "ownedCompositionBow",
        }
    )
    feature_layout_v1["ownedCompositionBow"] = {
        "version": "v0_static_vocab",
        "shape": [len(OWNED_COMPOSITION_ROW_NAMES), len(OWNED_COMPOSITION_VOCABULARY)],
        "rowNames": list(OWNED_COMPOSITION_ROW_NAMES),
        "vocabularyScope": "static_ruleset_seeded",
        "vocabulary": {
            "idToName": list(OWNED_COMPOSITION_VOCABULARY),
            "nameToId": dict(OWNED_COMPOSITION_NAME_TO_ID),
            "unknownName": OWNED_COMPOSITION_UNKNOWN_NAME,
        },
        "notes": [
            "This first pass counts current self-owned objects by static name bucket, split into unit and building rows.",
            "Counts are derived from the self entity branch, so they remain observation-safe but can undercount if max-entity truncation drops self objects.",
            "Unknown or unresolved names are folded into the unknown bucket instead of changing tensor width across runs.",
        ],
    }

    for sample in samples:
        sample["featureTensors"]["ownedCompositionBow"] = build_owned_composition_bow(sample, dataset)


def augment_dataset_with_scalar_core_identity(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    scalar_feature_names = list(dataset["schema"]["observation"]["scalarFeatureNames"])
    if any(name.startswith("self_side_") for name in scalar_feature_names):
        return

    identity_feature_names = [
        *[f"self_side_{_identity_suffix(name)}" for name in SIDE_VOCABULARY],
        *[f"enemy_side_{_identity_suffix(name)}" for name in SIDE_VOCABULARY],
        *[f"self_country_{_identity_suffix(name)}" for name in COUNTRY_VOCABULARY],
        *[f"enemy_country_{_identity_suffix(name)}" for name in COUNTRY_VOCABULARY],
    ]
    dataset["schema"]["observation"]["scalarFeatureNames"] = scalar_feature_names + identity_feature_names

    for section in dataset["schema"]["featureSections"]:
        if section["name"] == "scalar":
            section["shape"] = [len(dataset["schema"]["observation"]["scalarFeatureNames"])]
            break
    else:
        raise KeyError("Feature schema is missing scalar section.")
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "scalarCore.factionIdentity",
        }
    )
    feature_layout_v1["scalarCore"] = {
        **feature_layout_v1.get("scalarCore", {}),
        "identity": {
            "version": "v0_country_side",
            "selfSideEncoding": "one_hot",
            "enemySideEncoding": "multi_hot_union",
            "selfCountryEncoding": "one_hot",
            "enemyCountryEncoding": "multi_hot_union",
            "sideVocabulary": {
                "idToName": list(SIDE_VOCABULARY),
                "nameToId": dict(SIDE_NAME_TO_ID),
                "unknownName": IDENTITY_UNKNOWN_NAME,
            },
            "countryVocabulary": {
                "idToName": list(COUNTRY_VOCABULARY),
                "nameToId": dict(COUNTRY_NAME_TO_ID),
                "unknownName": IDENTITY_UNKNOWN_NAME,
            },
            "notes": [
                "Identity features are derived from replay-global player metadata, not inferred from unit composition.",
                "Enemy identity uses a multi-hot union across replay players not on the acting player's team.",
                "When no opposing player metadata is available, the unknown enemy bucket is enabled.",
            ],
        },
    }

    for sample in samples:
        sample["featureTensors"]["scalar"] = [
            *sample["featureTensors"]["scalar"],
            *build_scalar_core_identity(sample, dataset),
        ]


def build_time_encoding(tick: float) -> list[float]:
    """Sinusoidal positional encoding of game tick, like transformer position encoding.

    Returns TIME_ENCODING_DIM floats: pairs of (sin, cos) at exponentially spaced frequencies.
    This lets the model distinguish early/mid/late game without learning nonlinear boundaries.
    """
    encoding = []
    half = TIME_ENCODING_DIM // 2
    for i in range(half):
        frequency = 1.0 / (TIME_ENCODING_MAX_TICKS ** (i / max(half - 1, 1)))
        angle = tick * frequency
        encoding.append(math.sin(angle))
        encoding.append(math.cos(angle))
    return encoding


def augment_dataset_with_time_encoding(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    schema = dataset["schema"]
    append_schema_section(
        schema["featureSections"],
        name="timeEncoding",
        shape=[TIME_ENCODING_DIM],
        dtype="float32",
    )
    schema["flatFeatureLength"] = compute_flat_length(schema["featureSections"])

    tick_index = _scalar_name_index(schema, "tick")
    for sample in samples:
        tick = float(sample["featureTensors"]["scalar"][tick_index])
        sample["featureTensors"]["timeEncoding"] = build_time_encoding(tick)


def augment_dataset_with_game_stats(dataset: dict[str, Any]) -> None:
    """Promotes gameStats from featureTensors into a separate schema section.

    gameStats is only present when replays are extracted with internalGame access.
    At live inference, this section will be absent and the model will skip it.
    """
    samples = dataset.get("samples", [])
    has_any = any("gameStats" in sample.get("featureTensors", {}) for sample in samples)
    if not has_any:
        return

    schema = dataset["schema"]
    append_schema_section(
        schema["featureSections"],
        name="gameStats",
        shape=[GAME_STATS_DIM],
        dtype="float32",
    )
    schema["flatFeatureLength"] = compute_flat_length(schema["featureSections"])

    # Samples without gameStats get zeros (shouldn't happen within a single replay,
    # but defensive in case of mixed data).
    for sample in samples:
        if "gameStats" not in sample["featureTensors"]:
            sample["featureTensors"]["gameStats"] = [0.0] * GAME_STATS_DIM


def augment_dataset_with_build_order_trace(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="buildOrderTrace",
        shape=[BUILD_ORDER_TRACE_LEN],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "buildOrderTrace",
        }
    )
    feature_layout_v1["buildOrderTrace"] = {
        "version": "v0_earliest_actions",
        "shape": [BUILD_ORDER_TRACE_LEN],
        "valueType": "static_action_type_id",
        "missingValue": LABEL_LAYOUT_V1_MISSING_INT,
        "contributingActionRules": [
            "Queue::Add::*",
            "PlaceBuilding::*",
            "Order::Deploy::*",
            "Order::DeploySelected::*",
        ],
        "semantics": "First N contributing self build/production action ids observed before the current action.",
        "truncationPolicy": "keep_earliest_only",
        "notes": [
            "This is the RA2 V1 analogue to mAS beginning_build_order.",
            "The trace is built on the full extracted action stream before SL downsampling, so dropped actions can still appear in later traces.",
            "Only additive production/build actions contribute in V1; cancel/hold actions are excluded.",
            "Consecutive duplicate deploy actions are collapsed to reduce opening spam from repeated deploy commands.",
        ],
    }

    history_by_player: dict[str, list[int]] = {}
    for sample in samples:
        player_name = str(sample["playerName"])
        history = history_by_player.setdefault(player_name, [])
        sample["featureTensors"]["buildOrderTrace"] = _build_order_trace_tensor(history)

        action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
        action_type_id = int(sample["labelTensors"]["actionTypeId"][0])
        if not _is_build_order_action_type_name(action_type_name):
            continue
        if (
            _should_collapse_build_order_duplicate(action_type_name)
            and history
            and history[-1] == action_type_id
        ):
            continue
        if len(history) < BUILD_ORDER_TRACE_LEN:
            history.append(action_type_id)


def augment_dataset_with_tech_state(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="techState",
        shape=[len(TECH_STATE_FLAG_NAMES)],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "techState",
        }
    )
    feature_layout_v1["techState"] = {
        "version": "v0_owned_state",
        "shape": [len(TECH_STATE_FLAG_NAMES)],
        "featureNames": list(TECH_STATE_FLAG_NAMES),
        "valueType": "binary_flag",
        "rulesetAssumptions": {
            "constructionYardNames": sorted(CONSTRUCTION_YARD_NAMES),
            "powerBuildingNames": sorted(POWER_BUILDING_NAMES),
            "barracksNames": sorted(BARRACKS_NAMES),
            "refineryNames": sorted(REFINERY_NAMES),
            "factoryNames": sorted(FACTORY_NAMES),
            "airfieldNames": sorted(AIRFIELD_NAMES),
            "navalYardNames": sorted(NAVAL_YARD_NAMES),
            "radarNames": sorted(RADAR_NAMES),
            "serviceDepotNames": sorted(SERVICE_DEPOT_NAMES),
            "techCenterNames": sorted(TECH_CENTER_NAMES),
        },
        "notes": [
            "This first pass uses current self-owned buildings plus scalar power state only.",
            "It is a compact prerequisite and production-unlock summary, not an omniscient legality solver.",
            "Special-tech flags are ownership flags; readiness and cooldown belong in a later superWeaponState section.",
        ],
    }

    for sample in samples:
        sample["featureTensors"]["techState"] = build_tech_state(sample, dataset)


def augment_dataset_with_production_state(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="productionState",
        shape=[len(PRODUCTION_STATE_FEATURE_NAMES)],
        dtype="float32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "productionState",
        }
    )
    feature_layout_v1["productionState"] = {
        "version": "v0_queue_summary",
        "shape": [len(PRODUCTION_STATE_FEATURE_NAMES)],
        "featureNames": list(PRODUCTION_STATE_FEATURE_NAMES),
        "queueTypeNames": list(PRODUCTION_QUEUE_TYPE_NAMES),
        "queueStatusNames": list(PRODUCTION_QUEUE_STATUS_NAMES),
        "nameTokenMissingValue": LABEL_LAYOUT_V1_MISSING_INT,
        "notes": [
            "This first pass summarizes self production queues from the internal production API only.",
            "Per-queue features include queue/factory counts, coarse status flags, occupancy, and the first queued item summary.",
            "Structure placement readiness is represented indirectly through the Structures queue Ready status.",
            "Super-weapon charge and cooldown state remain separate future work in superWeaponState.",
        ],
    }

    for sample in samples:
        sample["featureTensors"]["productionState"] = build_production_state(sample, dataset)


def augment_dataset_with_super_weapon_state(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="superWeaponState",
        shape=[len(SUPER_WEAPON_STATE_FEATURE_NAMES)],
        dtype="float32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "superWeaponState",
        }
    )
    feature_layout_v1["superWeaponState"] = {
        "version": "v0_generic_charge_summary",
        "shape": [len(SUPER_WEAPON_STATE_FEATURE_NAMES)],
        "featureNames": list(SUPER_WEAPON_STATE_FEATURE_NAMES),
        "typeNames": list(SUPER_WEAPON_STATE_TYPE_NAMES),
        "statusNames": list(SUPER_WEAPON_STATUS_NAMES),
        "rechargeSecondsByType": dict((dataset.get("superWeaponSchema") or {}).get("rechargeSecondsByType") or {}),
        "notes": [
            "This first pass summarizes the acting player's generic super-weapon records only.",
            "Per-type features include count, presence, status flags, and normalized charge progress in [0,1].",
            "The generic API does not expose a separate availability field here, so readiness is represented by the Ready status flag rather than a distinct legality bit.",
            "charge_progress_01 is computed against per-type nominal RechargeTime values exported from rules.ini.",
        ],
    }

    for sample in samples:
        sample["featureTensors"]["superWeaponState"] = build_super_weapon_state(sample, dataset)


def augment_dataset_with_enemy_memory_bow(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="enemyMemoryBow",
        shape=[len(OWNED_COMPOSITION_ROW_NAMES), len(OWNED_COMPOSITION_VOCABULARY)],
        dtype="int32",
    )
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="enemyMemoryTechFlags",
        shape=[len(ENEMY_MEMORY_TECH_FLAG_NAMES)],
        dtype="int32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "enemyMemoryBow",
        }
    )
    feature_layout_v1["enemyMemoryBow"] = {
        "version": "v0_monotonic_max_visible",
        "shape": [len(OWNED_COMPOSITION_ROW_NAMES), len(OWNED_COMPOSITION_VOCABULARY)],
        "rowNames": list(OWNED_COMPOSITION_ROW_NAMES),
        "vocabulary": {
            "idToName": list(OWNED_COMPOSITION_VOCABULARY),
            "nameToId": dict(OWNED_COMPOSITION_NAME_TO_ID),
            "unknownName": OWNED_COMPOSITION_UNKNOWN_NAME,
        },
        "countSemantics": "max_visible_count_so_far",
        "techFlagsShape": [len(ENEMY_MEMORY_TECH_FLAG_NAMES)],
        "techFlagNames": list(ENEMY_MEMORY_TECH_FLAG_NAMES),
        "notes": [
            "Enemy memory is accumulated per player perspective using only currently visible enemy entities from the safe observation tensor.",
            "The count rows store monotonic max-visible counts by static name bucket, so they never decrease across samples for a player.",
            "Enemy tech flags are derived from seen enemy building names only and never backfilled from global hidden state.",
        ],
    }

    memory_by_player: dict[str, dict[str, Any]] = {}
    for sample in samples:
        player_name = str(sample["playerName"])
        player_memory = memory_by_player.setdefault(
            player_name,
            {
                "unitCounts": [0] * len(OWNED_COMPOSITION_VOCABULARY),
                "buildingCounts": [0] * len(OWNED_COMPOSITION_VOCABULARY),
                "seenBuildingNames": set(),
            },
        )

        current_unit_counts = [0] * len(OWNED_COMPOSITION_VOCABULARY)
        current_building_counts = [0] * len(OWNED_COMPOSITION_VOCABULARY)
        for entity in _collect_enemy_entities(sample, dataset):
            name = _normalize_owned_object_name(entity["name"])
            bucket = OWNED_COMPOSITION_NAME_TO_ID.get(name, 0)
            if entity["isBuilding"]:
                current_building_counts[bucket] += 1
                if name != OWNED_COMPOSITION_UNKNOWN_NAME:
                    player_memory["seenBuildingNames"].add(name)
            else:
                current_unit_counts[bucket] += 1

        for bucket, count in enumerate(current_unit_counts):
            if count > player_memory["unitCounts"][bucket]:
                player_memory["unitCounts"][bucket] = count
        for bucket, count in enumerate(current_building_counts):
            if count > player_memory["buildingCounts"][bucket]:
                player_memory["buildingCounts"][bucket] = count

        sample["featureTensors"]["enemyMemoryBow"] = [
            list(player_memory["unitCounts"]),
            list(player_memory["buildingCounts"]),
        ]
        sample["featureTensors"]["enemyMemoryTechFlags"] = build_enemy_memory_tech_flags(
            set(player_memory["seenBuildingNames"])
        )


def augment_dataset_with_entity_intent_summary(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples", [])
    entity_feature_names = dataset["schema"]["observation"]["entityFeatureNames"]
    for feature_name in ENTITY_INTENT_SUMMARY_FEATURE_NAMES:
        if feature_name not in entity_feature_names:
            entity_feature_names.append(feature_name)

    _update_entity_feature_section_shape(dataset)
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "entity.intentSummary",
        }
    )
    feature_layout_v1["entityIntentSummary"] = {
        "version": "v0_heuristic_compact_summary",
        "appendedTo": "entityFeatures",
        "featureNames": list(ENTITY_INTENT_SUMMARY_FEATURE_NAMES),
        "weaponCooldownClampTicks": ENTITY_INTENT_WEAPON_COOLDOWN_CLAMP_TICKS,
        "rawFeatureDependencies": [
            "is_idle",
            "can_move",
            "build_status_build_up",
            "build_status_build_down",
            "attack_state_check_range",
            "attack_state_prepare_to_fire",
            "attack_state_fire_up",
            "attack_state_firing",
            "attack_state_just_fired",
            "has_wrench_repair",
            "harvested_ore",
            "harvested_gems",
            "hit_points_ratio",
            "tile_x_norm",
            "tile_y_norm",
            "factory_status_delivering",
            "factory_has_delivery",
            "rally_point_valid",
            "rally_x_norm",
            "rally_y_norm",
            "primary_weapon_cooldown_ticks",
            "secondary_weapon_cooldown_ticks",
            "ammo",
        ],
        "notes": [
            "This is a compact RA2 analogue to SC2 per-unit order and cooldown summaries, built from observation-safe transient state only.",
            "Chronodivide does not currently expose a clean generic current-order field, so V1 uses heuristic intent categories derived from attack/build/harvest/repair/factory/rally signals.",
            "The summary is appended to the entity feature rows instead of creating a separate top-level branch.",
            "weapon_cooldown_progress_01 uses a fixed 90-tick clamp in V1; this is a generic monotonic readiness proxy, not a weapon-specific perfect normalization.",
        ],
    }

    for sample in samples:
        base_entity_rows = sample["featureTensors"]["entityFeatures"]
        intent_summary_rows = build_entity_intent_summary(sample, dataset)
        augmented_rows: list[list[float]] = []
        for base_row, intent_row in zip(base_entity_rows, intent_summary_rows):
            if hasattr(base_row, "tolist"):
                base_values = list(base_row.tolist())
            else:
                base_values = list(base_row)
            augmented_rows.append([*base_values, *intent_row])
        sample["featureTensors"]["entityFeatures"] = augmented_rows


def augment_dataset_with_map_static(dataset: dict[str, Any]) -> None:
    static_map_by_player = dataset.get("staticMapByPlayer", {})
    if not static_map_by_player:
        raise KeyError("Dataset is missing staticMapByPlayer required for mapStatic.")

    first_static_map = next(iter(static_map_by_player.values()))
    channel_names = list(first_static_map.get("channelNames", []))
    height = int(first_static_map.get("height", 0))
    width = int(first_static_map.get("width", 0))
    append_schema_section(
        dataset["schema"]["featureSections"],
        name="mapStatic",
        shape=[len(channel_names), height, width],
        dtype="float32",
    )
    dataset["schema"]["flatFeatureLength"] = compute_flat_length(dataset["schema"]["featureSections"])

    feature_layout_v1 = dataset.setdefault("featureLayoutV1", build_feature_layout_v1_contract())
    feature_layout_v1["implementedSections"] = sorted(
        {
            *feature_layout_v1.get("implementedSections", []),
            "mapStatic.core",
        }
    )
    feature_layout_v1["mapStatic"] = {
        "version": "v0_compact_static_planes",
        "representationPolicy": "separate_branch",
        "shape": [len(channel_names), height, width],
        "channelNames": channel_names,
        "notes": [
            "mapStatic is replay-constant and player-constant over a shard; the same planes are attached to each sample for that player.",
            "The first pass stores compact static priors for passability, buildability, normalized height, and all start locations.",
            "Buildability is a side-specific reference-building prior with adjacency checks disabled, not a dynamic legality mask.",
        ],
    }

    for sample in dataset.get("samples", []):
        player_name = str(sample["playerName"])
        static_map = static_map_by_player.get(player_name)
        if static_map is None:
            raise KeyError(f"Static map features missing for player: {player_name}")
        sample["featureTensors"]["mapStatic"] = static_map["data"]
