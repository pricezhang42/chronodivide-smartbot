from __future__ import annotations


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
