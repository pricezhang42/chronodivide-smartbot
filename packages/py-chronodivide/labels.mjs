function tileToPlain(tile) {
  if (!tile) {
    return null;
  }

  return {
    x: tile.x ?? tile.rx,
    y: tile.y ?? tile.ry,
  };
}

function playerOwnerToName(owner) {
  if (!owner) {
    return null;
  }
  if (typeof owner === "string") {
    return owner;
  }
  return owner.name ?? null;
}

function objectTypeToName(type) {
  const names = {
    0: "None",
    1: "Aircraft",
    2: "Building",
    3: "Infantry",
    4: "Overlay",
    5: "Smudge",
    6: "Terrain",
    7: "Vehicle",
    8: "Animation",
    9: "Projectile",
    10: "VoxelAnim",
    11: "Debris",
  };
  return names[type] ?? `ObjectType_${type}`;
}

export const RAW_ACTION_ID_NAMES = {
  0: "NoAction",
  1: "DropPlayerAction",
  3: "ResignGameAction",
  5: "PlaceBuildingAction",
  6: "SellObjectAction",
  7: "ToggleRepairAction",
  8: "SelectUnitsAction",
  9: "OrderUnitsAction",
  10: "UpdateQueueAction",
  12: "ActivateSuperWeaponAction",
  13: "PingLocationAction",
};

export const ACTION_FAMILIES = [
  "no_action",
  "select_units",
  "order_units",
  "update_queue",
  "place_building",
  "sell_object",
  "toggle_repair",
  "activate_super_weapon",
  "resign_game",
  "drop_player",
  "ping_location",
  "unknown",
];

const ACTION_FAMILY_IDS = Object.fromEntries(ACTION_FAMILIES.map((name, index) => [name, index]));

export const ORDER_TYPE_NAMES = {
  0: "Move",
  1: "ForceMove",
  2: "Attack",
  3: "ForceAttack",
  4: "AttackMove",
  5: "Guard",
  6: "GuardArea",
  7: "Capture",
  8: "Occupy",
  9: "Deploy",
  10: "DeploySelected",
  11: "Stop",
  12: "Cheer",
  13: "Dock",
  14: "Gather",
  15: "Repair",
  16: "Scatter",
  17: "EnterTransport",
  18: "PlaceBomb",
};

export const QUEUE_TYPE_NAMES = {
  0: "Structures",
  1: "Armory",
  2: "Infantry",
  3: "Vehicles",
  4: "Aircrafts",
  5: "Ships",
};

export const QUEUE_UPDATE_TYPE_NAMES = {
  0: "Add",
  1: "Cancel",
  2: "Hold",
};

export const SUPER_WEAPON_TYPE_NAMES = {
  0: "MultiMissile",
  1: "IronCurtain",
  2: "LightningStorm",
  3: "ChronoSphere",
  4: "ChronoWarp",
  5: "ParaDrop",
  6: "AmerParaDrop",
};

export const TARGET_MODE_NAMES = ["none", "tile", "object", "ore_tile"];
const TARGET_MODE_IDS = Object.fromEntries(TARGET_MODE_NAMES.map((name, index) => [name, index]));

const UI_ONLY_RAW_ACTION_IDS = new Set([1, 13]);

export function isUiOnlyRawActionId(rawActionId) {
  return UI_ONLY_RAW_ACTION_IDS.has(rawActionId);
}

export function isNoActionRawActionId(rawActionId) {
  return rawActionId === 0;
}

export function getRawActionName(rawActionId) {
  return RAW_ACTION_ID_NAMES[rawActionId] ?? `RawAction_${rawActionId}`;
}

function getActionFamilyName(rawActionId, className) {
  if (className === "SelectUnitsAction") {
    return "select_units";
  }
  if (className === "OrderUnitsAction") {
    return "order_units";
  }
  if (className === "UpdateQueueAction") {
    return "update_queue";
  }
  if (className === "PlaceBuildingAction") {
    return "place_building";
  }
  if (className === "SellObjectAction") {
    return "sell_object";
  }
  if (className === "ToggleRepairAction") {
    return "toggle_repair";
  }
  if (className === "ActivateSuperWeaponAction") {
    return "activate_super_weapon";
  }
  if (className === "ResignGameAction") {
    return "resign_game";
  }
  if (className === "DropPlayerAction") {
    return "drop_player";
  }
  if (className === "PingLocationAction") {
    return "ping_location";
  }
  if (className === "NoAction" || rawActionId === 0) {
    return "no_action";
  }
  return "unknown";
}

function getTargetModeName(action) {
  if (!action.target) {
    return "none";
  }
  if (action.target.obj) {
    return "object";
  }
  if (action.target.tile && action.target.isOre) {
    return "ore_tile";
  }
  if (action.target.tile) {
    return "tile";
  }
  return "none";
}

function getNamedEnumValue(map, value, prefix) {
  if (value === null || value === undefined) {
    return null;
  }
  return map[value] ?? `${prefix}_${value}`;
}

function resolveOrderTarget(action) {
  const targetMode = getTargetModeName(action);
  const targetObject = action.target?.obj;

  return {
    targetMode,
    targetModeId: TARGET_MODE_IDS[targetMode],
    targetTile: tileToPlain(action.target?.tile),
    targetObjectId: targetObject?.id ?? null,
    targetObjectName: targetObject?.name ?? null,
    targetObjectOwner: playerOwnerToName(targetObject?.owner),
    targetObjectType: targetObject?.type ?? null,
    targetObjectTypeName: targetObject?.type !== undefined ? objectTypeToName(targetObject.type) : null,
    targetIsOre: Boolean(action.target?.isOre),
  };
}

function resolveGameObject(gameApi, objectId) {
  if (objectId === null || objectId === undefined) {
    return {
      objectId: null,
      objectName: null,
      objectOwner: null,
      objectType: null,
      objectTypeName: null,
      objectTile: null,
    };
  }

  const obj = gameApi.getUnitData(objectId) ?? gameApi.getGameObjectData(objectId);
  if (!obj) {
    return {
      objectId,
      objectName: null,
      objectOwner: null,
      objectType: null,
      objectTypeName: null,
      objectTile: null,
    };
  }

  return {
    objectId,
    objectName: obj.name ?? null,
    objectOwner: playerOwnerToName(obj.owner),
    objectType: obj.type ?? null,
    objectTypeName: obj.type !== undefined ? objectTypeToName(obj.type) : null,
    objectTile: tileToPlain(obj.tile),
  };
}

export function shouldKeepRawActionId(
  rawActionId,
  { includeNoAction = false, includeUiActions = false } = {},
) {
  if (!includeNoAction && isNoActionRawActionId(rawActionId)) {
    return false;
  }
  if (!includeUiActions && isUiOnlyRawActionId(rawActionId)) {
    return false;
  }
  return true;
}

export function buildActionTimelines(
  replay,
  { playerNames = null, includeNoAction = false, includeUiActions = false } = {},
) {
  const allowedPlayers = playerNames ? new Set(playerNames) : null;
  const timelines = new Map();

  for (const event of replay.events) {
    if (event.constructor.name !== "TurnActionsReplayEvent") {
      continue;
    }

    for (const [playerId, actions] of event.payload) {
      const playerName = replay.gameOpts.humanPlayers[playerId]?.name ?? `player_${playerId}`;
      if (allowedPlayers && !allowedPlayers.has(playerName)) {
        continue;
      }

      for (const payload of actions) {
        if (!shouldKeepRawActionId(payload.id, { includeNoAction, includeUiActions })) {
          continue;
        }

        const timeline = timelines.get(playerName) ?? [];
        timeline.push({
          tick: event.tickNo,
          rawActionId: payload.id,
          rawActionName: getRawActionName(payload.id),
        });
        timelines.set(playerName, timeline);
      }
    }
  }

  for (const timeline of timelines.values()) {
    for (let index = 0; index < timeline.length; index += 1) {
      const previous = timeline[index - 1];
      const current = timeline[index];
      const next = timeline[index + 1];
      current.timelineIndex = index;
      current.delayFromPreviousAction = previous ? current.tick - previous.tick : null;
      current.delayToNextAction = next ? next.tick - current.tick : null;
    }
  }

  return timelines;
}

export function updateSelectionFromAction(action, selectionBefore = []) {
  if (action.constructor.name === "SelectUnitsAction") {
    return Array.isArray(action._unitIds) ? action._unitIds.slice() : [];
  }
  return selectionBefore.slice();
}

export function getActionLabelSchema() {
  return {
    rawActionIdNames: { ...RAW_ACTION_ID_NAMES },
    actionFamilies: ACTION_FAMILIES.slice(),
    actionFamilyIds: { ...ACTION_FAMILY_IDS },
    orderTypeNames: { ...ORDER_TYPE_NAMES },
    queueTypeNames: { ...QUEUE_TYPE_NAMES },
    queueUpdateTypeNames: { ...QUEUE_UPDATE_TYPE_NAMES },
    superWeaponTypeNames: { ...SUPER_WEAPON_TYPE_NAMES },
    targetModes: TARGET_MODE_NAMES.slice(),
    targetModeIds: { ...TARGET_MODE_IDS },
    excludedByDefault: {
      noActionRawIds: [0],
      uiOnlyRawIds: [1, 13],
    },
    notes: [
      "Selection for order actions is inferred from the replay's preceding SelectUnitsAction stream.",
      "PingLocationAction and DropPlayerAction are excluded by default because they are not useful RL/SL action targets.",
      "delayToNextAction is computed on the kept action stream after filtering no-action/UI-only actions.",
    ],
  };
}

export function decodeActionLabel(
  action,
  {
    rawActionId,
    gameApi,
    selectionBefore = [],
    selectionAfter = [],
    timelineEntry = null,
  } = {},
) {
  const rawActionName = getRawActionName(rawActionId);
  const actionFamily = getActionFamilyName(rawActionId, action.constructor.name);

  const label = {
    rawActionId,
    rawActionName,
    actionClassName: action.constructor.name,
    actionFamily,
    actionFamilyId: ACTION_FAMILY_IDS[actionFamily] ?? ACTION_FAMILY_IDS.unknown,
    debugPrint: typeof action.print === "function" ? action.print() : "",
    timelineIndex: timelineEntry?.timelineIndex ?? null,
    delayFromPreviousAction: timelineEntry?.delayFromPreviousAction ?? null,
    delayToNextAction: timelineEntry?.delayToNextAction ?? null,
    selectionBeforeActionIds: selectionBefore.slice(),
    selectionAfterActionIds: selectionAfter.slice(),
    actionSelectedUnitIds: [],
    queue: null,
    orderTypeId: null,
    orderTypeName: null,
    targetMode: "none",
    targetModeId: TARGET_MODE_IDS.none,
    targetTile: null,
    targetObjectId: null,
    targetObjectName: null,
    targetObjectOwner: null,
    targetObjectType: null,
    targetObjectTypeName: null,
    targetIsOre: false,
    queueTypeId: null,
    queueTypeName: null,
    queueUpdateTypeId: null,
    queueUpdateTypeName: null,
    quantity: null,
    itemName: null,
    itemType: null,
    itemTypeName: null,
    itemCost: null,
    itemFactoryType: null,
    itemUiName: null,
    buildingName: null,
    buildingType: null,
    buildingTypeName: null,
    buildingCost: null,
    buildingTile: null,
    objectId: null,
    objectName: null,
    objectOwner: null,
    objectType: null,
    objectTypeName: null,
    objectTile: null,
    superWeaponTypeId: null,
    superWeaponTypeName: null,
    superWeaponTile: null,
    superWeaponTile2: null,
    pingTile: null,
  };

  switch (action.constructor.name) {
    case "SelectUnitsAction":
      label.actionSelectedUnitIds = selectionAfter.slice();
      break;
    case "OrderUnitsAction":
      label.actionSelectedUnitIds = selectionBefore.slice();
      label.queue = Boolean(action.queue);
      label.orderTypeId = action.orderType ?? null;
      label.orderTypeName = getNamedEnumValue(ORDER_TYPE_NAMES, action.orderType, "OrderType");
      Object.assign(label, resolveOrderTarget(action));
      break;
    case "UpdateQueueAction":
      label.queueTypeId = action.queueType ?? null;
      label.queueTypeName = getNamedEnumValue(QUEUE_TYPE_NAMES, action.queueType, "QueueType");
      label.queueUpdateTypeId = action.updateType ?? null;
      label.queueUpdateTypeName = getNamedEnumValue(
        QUEUE_UPDATE_TYPE_NAMES,
        action.updateType,
        "QueueUpdateType",
      );
      label.quantity = action.quantity ?? null;
      label.itemName = action.item?.name ?? null;
      label.itemType = action.item?.type ?? null;
      label.itemTypeName = action.item?.type !== undefined ? objectTypeToName(action.item.type) : null;
      label.itemCost = action.item?.cost ?? null;
      label.itemFactoryType = action.item?.factory ?? null;
      label.itemUiName = action.item?.uiName ?? null;
      break;
    case "PlaceBuildingAction":
      label.buildingName = action.buildingRules?.name ?? null;
      label.buildingType = action.buildingRules?.type ?? null;
      label.buildingTypeName =
        action.buildingRules?.type !== undefined ? objectTypeToName(action.buildingRules.type) : null;
      label.buildingCost = action.buildingRules?.cost ?? null;
      label.buildingTile = tileToPlain(action.tile);
      break;
    case "SellObjectAction":
      Object.assign(label, resolveGameObject(gameApi, action.objectId ?? null));
      break;
    case "ToggleRepairAction":
      Object.assign(label, resolveGameObject(gameApi, action.buildingId ?? null));
      break;
    case "ActivateSuperWeaponAction":
      label.superWeaponTypeId = action.superWeaponType ?? null;
      label.superWeaponTypeName = getNamedEnumValue(
        SUPER_WEAPON_TYPE_NAMES,
        action.superWeaponType,
        "SuperWeaponType",
      );
      label.superWeaponTile = tileToPlain(action.tile);
      label.superWeaponTile2 = tileToPlain(action.tile2);
      break;
    case "PingLocationAction":
      label.pingTile = tileToPlain(action.tile);
      break;
    default:
      break;
  }

  return label;
}

