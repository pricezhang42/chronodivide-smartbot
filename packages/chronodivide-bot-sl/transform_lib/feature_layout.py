from __future__ import annotations

from typing import Any

from action_dict import ACTION_TYPE_ID_TO_NAME, PLACE_BUILDING_NAMES, QUEUE_ITEM_NAMES

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


def _sum_spatial_channel(sample: dict[str, Any], schema: dict[str, Any], channel_name: str) -> float:
    channel_names = schema["observation"]["spatialChannelNames"]
    channel_index = int(channel_names.index(channel_name))
    plane = sample["featureTensors"]["spatial"][channel_index]
    return float(sum(sum(float(value) for value in row) for row in plane))


def _collect_self_entities(sample: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    schema = dataset["schema"]
    relation_self_index = _feature_name_index(schema, "relation_self")
    object_building_index = _feature_name_index(schema, "object_building")
    can_move_index = _feature_name_index(schema, "can_move")

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
        entities.append(
            {
                "index": entity_index,
                "name": _decode_name_token(schema, int(entity_names[entity_index])),
                "isBuilding": float(feature_row[object_building_index]) > 0.5,
                "canMove": float(feature_row[can_move_index]) > 0.5,
            }
        )
    return entities


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


def _is_order_action_available(
    action_type_name: str,
    *,
    has_selection: bool,
    has_selected_mobile: bool,
    has_selected_harvester: bool,
    has_selected_deployable: bool,
    visible_resource_total: float,
) -> int:
    if not has_selection:
        return 0

    _, order_type_name, target_mode_name = action_type_name.split("::", 2)

    movement_like_orders = {
        "Move",
        "ForceMove",
        "AttackMove",
        "GuardArea",
        "Dock",
        "Gather",
        "Repair",
        "Scatter",
        "EnterTransport",
        "PlaceBomb",
    }
    if order_type_name in movement_like_orders and not has_selected_mobile:
        return 0

    if order_type_name in {"Deploy", "DeploySelected"} and not has_selected_deployable:
        return 0

    if order_type_name == "Gather":
        if not has_selected_harvester:
            return 0
        if target_mode_name == "ore_tile" and visible_resource_total <= 0:
            return 0

    return 1


def build_available_action_mask(sample: dict[str, Any], dataset: dict[str, Any]) -> list[int]:
    schema = dataset["schema"]
    scalar = sample["featureTensors"]["scalar"]
    current_selection_count = int(sample["featureTensors"]["currentSelectionCount"][0])
    self_unit_count = float(scalar[_scalar_name_index(schema, "self_unit_count")])
    self_building_count = float(scalar[_scalar_name_index(schema, "self_building_count")])

    self_entities = _collect_self_entities(sample, dataset)
    selected_entities = _collect_selected_entities(sample, self_entities)
    self_entity_names = {entity["name"] for entity in self_entities if entity["name"]}
    selected_names = {entity["name"] for entity in selected_entities if entity["name"]}

    has_self_entities = (self_unit_count + self_building_count) > 0
    has_self_buildings = self_building_count > 0 or any(entity["isBuilding"] for entity in self_entities)
    has_selection = current_selection_count > 0
    has_selected_mobile = any(entity["canMove"] for entity in selected_entities)
    has_selected_harvester = bool(selected_names & HARVESTER_NAMES)
    has_selected_deployable = bool(selected_names & DEPLOYABLE_NAMES)
    has_construction_yard = any(
        name in CONSTRUCTION_YARD_NAMES or name.endswith("CNST")
        for name in self_entity_names
    )
    has_any_superweapon_structure = bool(self_entity_names & SUPERWEAPON_STRUCTURE_NAMES)
    visible_resource_total = _sum_spatial_channel(sample, schema, "visible_ore") + _sum_spatial_channel(
        sample,
        schema,
        "visible_gems",
    )

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
                has_selection=has_selection,
                has_selected_mobile=has_selected_mobile,
                has_selected_harvester=has_selected_harvester,
                has_selected_deployable=has_selected_deployable,
                visible_resource_total=visible_resource_total,
            )
        elif action_type_name.startswith("Queue::"):
            enabled = 1 if has_self_buildings else 0
        elif action_type_name.startswith("PlaceBuilding::"):
            enabled = 1 if has_construction_yard else 0
        elif action_type_name.startswith("ActivateSuperWeapon::"):
            superweapon_name = action_type_name.split("::", 1)[1]
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
        "version": "v0_conservative",
        "shape": [len(ACTION_TYPE_ID_TO_NAME)],
        "policy": "observation_driven_conservative",
        "notes": [
            "This first pass disables only clearly impossible actions and leaves ambiguous ones enabled.",
            "The mask is derived in chronodivide-bot-sl from current selection, visible self state, and static action-type semantics.",
            "It intentionally avoids hidden enemy state and any omniscient legality solver.",
        ],
    }

    for sample in samples:
        sample["featureTensors"]["availableActionMask"] = build_available_action_mask(sample, dataset)


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
