function vectorToPlain(vector) {
  if (!vector) {
    return undefined;
  }

  const result = {};
  if ("x" in vector) {
    result.x = vector.x;
  }
  if ("y" in vector) {
    result.y = vector.y;
  }
  if ("z" in vector) {
    result.z = vector.z;
  }
  return result;
}

function tileToPlain(tile) {
  if (!tile) {
    return undefined;
  }

  return {
    x: tile.rx ?? tile.x,
    y: tile.ry ?? tile.y,
  };
}

function powerToPlain(power) {
  if (!power) {
    return undefined;
  }

  return {
    total: power.total,
    drain: power.drain,
    isLowPower: power.isLowPower,
  };
}

const QUEUE_STATUS_NAMES = {
  0: "Idle",
  1: "Active",
  2: "OnHold",
  3: "Ready",
};

const QUEUE_TYPE_NAMES = {
  0: "Structures",
  1: "Armory",
  2: "Infantry",
  3: "Vehicles",
  4: "Aircrafts",
  5: "Ships",
};

const SUPER_WEAPON_TYPE_NAMES = {
  0: "MultiMissile",
  1: "IronCurtain",
  2: "LightningStorm",
  3: "ChronoSphere",
  4: "ChronoWarp",
  5: "ParaDrop",
  6: "AmerParaDrop",
};

const SUPER_WEAPON_STATUS_NAMES = {
  0: "Charging",
  1: "Paused",
  2: "Ready",
};

const TAG_REPEAT_TYPE_NAMES = {
  0: "OnceAny",
  1: "OnceAll",
  2: "Repeat",
};

function safeNumber(value, fallback = undefined) {
  return Number.isFinite(value) ? value : fallback;
}

function toPlainRecursive(value, seen = new WeakSet()) {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map((item) => toPlainRecursive(item, seen));
  }

  if (value instanceof Map) {
    return Array.from(value.entries()).map(([key, itemValue]) => ({
      key: toPlainRecursive(key, seen),
      value: toPlainRecursive(itemValue, seen),
    }));
  }

  if (value instanceof Set) {
    return Array.from(value.values()).map((item) => toPlainRecursive(item, seen));
  }

  if (typeof value === "object") {
    if (seen.has(value)) {
      return "[Circular]";
    }

    seen.add(value);
    const plain = {};
    for (const [key, itemValue] of Object.entries(value)) {
      const normalized = toPlainRecursive(itemValue, seen);
      if (normalized !== undefined) {
        plain[key] = normalized;
      }
    }
    seen.delete(value);
    return plain;
  }

  return String(value);
}

function mapCounterToPlain(counter, keyName = "key") {
  if (!counter) {
    return [];
  }

  if (counter instanceof Map) {
    return Array.from(counter.entries())
      .map(([key, count]) => ({
        [keyName]: key,
        count: safeNumber(count, 0),
      }))
      .sort((left, right) => {
        const leftKey = String(left[keyName]);
        const rightKey = String(right[keyName]);
        return leftKey.localeCompare(rightKey);
      });
  }

  if (typeof counter === "object") {
    return Object.entries(counter)
      .map(([key, count]) => ({
        [keyName]: key,
        count: safeNumber(count, 0),
      }))
      .sort((left, right) => String(left[keyName]).localeCompare(String(right[keyName])));
  }

  return [];
}

function factoryToPlain(factory) {
  if (!factory) {
    return undefined;
  }

  return {
    status: factory.status,
    deliveringUnit: safeNumber(factory.deliveringUnit),
  };
}

function queueItemToPlain(item) {
  if (!item) {
    return undefined;
  }

  return {
    objectName: item.rules?.name,
    objectType: item.rules?.type,
    cost: safeNumber(item.rules?.cost),
    quantity: safeNumber(item.quantity, 0),
    progress: safeNumber(item.progress),
    creditsEach: safeNumber(item.creditsEach),
    creditsSpent: safeNumber(item.creditsSpent),
    creditsSpentLeftover: safeNumber(item.creditsSpentLeftover),
  };
}

function queueToPlain(queue) {
  if (!queue) {
    return undefined;
  }

  return {
    type: queue.type,
    typeName: QUEUE_TYPE_NAMES[queue.type] ?? `QueueType_${queue.type}`,
    status: queue.status,
    statusName: QUEUE_STATUS_NAMES[queue.status] ?? `QueueStatus_${queue.status}`,
    size: safeNumber(queue.size, 0),
    maxSize: safeNumber(queue.maxSize ?? queue._maxSize),
    maxItemQuantity: safeNumber(queue.maxItemQuantity),
    items: Array.isArray(queue.items) ? queue.items.map(queueItemToPlain).filter(Boolean) : [],
  };
}

function productionToPlain(production) {
  if (!production) {
    return undefined;
  }

  const queues = production.queues instanceof Map ? Array.from(production.queues.values()) : [];
  const queueSummaries = queues.map(queueToPlain).filter(Boolean);
  const factoryCounts =
    production.factoryCounts instanceof Map
      ? Array.from(production.factoryCounts.entries()).map(([queueType, count]) => ({
          queueType,
          queueTypeName: QUEUE_TYPE_NAMES[queueType] ?? `QueueType_${queueType}`,
          count: safeNumber(count, 0),
        }))
      : [];

  return {
    maxTechLevel: safeNumber(production.maxTechLevel),
    buildSpeedModifier: safeNumber(production.buildSpeedModifier),
    queueCount: queueSummaries.length,
    queues: queueSummaries,
    factoryCounts,
    availableObjectCount: Array.isArray(production.allAvailableObjects) ? production.allAvailableObjects.length : 0,
    availableObjects:
      Array.isArray(production.allAvailableObjects)
        ? production.allAvailableObjects.map((rules) => ({
            name: rules?.name,
            type: rules?.type,
            cost: safeNumber(rules?.cost),
            techLevel: safeNumber(rules?.techLevel),
          }))
        : [],
  };
}

function playerStatsToPlain(internalPlayer) {
  if (!internalPlayer) {
    return undefined;
  }

  return {
    creditsGained: safeNumber(internalPlayer.creditsGained, 0),
    buildingsCaptured: safeNumber(internalPlayer.buildingsCaptured, 0),
    cratesPickedUp: safeNumber(internalPlayer.cratesPickedUp, 0),
    unitsBuiltByType: mapCounterToPlain(internalPlayer.unitsBuiltByType, "objectType"),
    unitsKilledByType: mapCounterToPlain(internalPlayer.unitsKilledByType, "objectType"),
    unitsLostByType: mapCounterToPlain(internalPlayer.unitsLostByType, "objectType"),
    limitedUnitsBuiltByName: mapCounterToPlain(internalPlayer.limitedUnitsBuiltByName, "objectName"),
  };
}

function weaponToPlain(weapon) {
  if (!weapon) {
    return undefined;
  }

  return {
    type: weapon.type,
    minRange: weapon.minRange,
    maxRange: weapon.maxRange,
    speed: safeNumber(weapon.speed),
    cooldownTicks: safeNumber(weapon.cooldownTicks),
    weaponName: weapon.rules?.name,
    projectileName: weapon.projectileRules?.name,
    warheadName: weapon.warheadRules?.name,
  };
}

function superWeaponToPlain(superWeapon) {
  if (!superWeapon) {
    return undefined;
  }

  return {
    playerName: superWeapon.playerName,
    type: superWeapon.type,
    typeName: SUPER_WEAPON_TYPE_NAMES[superWeapon.type] ?? `SuperWeaponType_${superWeapon.type}`,
    status: superWeapon.status,
    statusName: SUPER_WEAPON_STATUS_NAMES[superWeapon.status] ?? `SuperWeaponStatus_${superWeapon.status}`,
    timerSeconds: safeNumber(superWeapon.timerSeconds),
  };
}

function resourceTileToPlain(resource) {
  if (!resource) {
    return undefined;
  }

  return {
    tile: tileToPlain(resource.tile),
    gems: safeNumber(resource.gems, 0),
    ore: safeNumber(resource.ore, 0),
    spawnsOre: Boolean(resource.spawnsOre),
  };
}

function tileResourceOnTileToPlain(resource) {
  if (!resource) {
    return undefined;
  }

  return {
    gems: safeNumber(resource.gems, 0),
    ore: safeNumber(resource.ore, 0),
    spawnsOre: Boolean(resource.spawnsOre),
  };
}

function mapToPlain(gameApi) {
  return {
    ...gameApi.map.getRealMapSize(),
    theaterType: gameApi.map.getTheaterType(),
    startingLocations: gameApi.map.getStartingLocations().map(vectorToPlain),
  };
}

function tagToPlain(tag) {
  if (!tag) {
    return undefined;
  }

  return {
    id: tag.id,
    name: tag.name,
    triggerId: tag.triggerId,
    repeatType: tag.repeatType,
    repeatTypeName: TAG_REPEAT_TYPE_NAMES[tag.repeatType] ?? `TagRepeatType_${tag.repeatType}`,
  };
}

function iniSectionToPlain(section) {
  if (!section) {
    return undefined;
  }

  const entries = Object.fromEntries(section.entries.entries());
  const children = section.getOrderedSections().map(iniSectionToPlain).filter(Boolean);

  return {
    name: section.name,
    entries,
    sections: children,
  };
}

function iniFileToPlain(ini) {
  if (!ini) {
    return undefined;
  }

  const sections = ini.getOrderedSections().map(iniSectionToPlain).filter(Boolean);
  return {
    sectionCount: sections.length,
    sections,
  };
}

function differenceIds(left, right) {
  const rightSet = new Set(right);
  return left.filter((id) => !rightSet.has(id));
}

function splitVisibleNonEnemyHostiles(gameApi, playerName, hostileIds, enemyIds) {
  const enemyIdSet = new Set(enemyIds);
  const playerNames = new Set(gameApi.getPlayers());
  const neutralUnitIds = [];
  const otherHostileUnitIds = [];

  for (const id of hostileIds) {
    if (enemyIdSet.has(id)) {
      continue;
    }

    const unit = gameApi.getUnitData(id) ?? gameApi.getGameObjectData(id);
    const owner = unit?.owner;

    if (!owner || !playerNames.has(owner)) {
      neutralUnitIds.push(id);
      continue;
    }

    if (owner === playerName || gameApi.areAlliedPlayers(playerName, owner)) {
      continue;
    }

    otherHostileUnitIds.push(id);
  }

  return {
    neutralUnitIds: neutralUnitIds.sort((left, right) => left - right),
    otherHostileUnitIds: otherHostileUnitIds.sort((left, right) => left - right),
  };
}

function uniqueSortedIds(ids) {
  return [...new Set(ids)].sort((left, right) => left - right);
}

function resolveObjectById(gameApi, id) {
  try {
    const unit = gameApi.getUnitData(id);
    if (unit) {
      return unit;
    }
  } catch {}

  try {
    return gameApi.getGameObjectData(id);
  } catch {
    return undefined;
  }
}

function collectUnitsByIds(gameApi, ids, unitLimit = null) {
  const units = uniqueSortedIds(ids)
    .map((id) => resolveObjectById(gameApi, id))
    .filter(Boolean)
    .map(unitToPlain)
    .sort((left, right) => left.id - right.id);

  return unitLimit === null ? units : units.slice(0, unitLimit);
}

function unitToPlain(unit) {
  const plain = {
    id: unit.id,
    type: unit.type,
    name: unit.name,
    owner: unit.owner,
    tile: tileToPlain(unit.tile),
    worldPosition: vectorToPlain(unit.worldPosition),
    tileElevation: unit.tileElevation,
    sight: unit.sight,
    veteranLevel: unit.veteranLevel,
    guardMode: unit.guardMode,
    purchaseValue: unit.purchaseValue,
    deathWeapon: weaponToPlain(unit.deathWeapon),
    foundation: unit.foundation
      ? {
          width: unit.foundation.width,
          height: unit.foundation.height,
        }
      : undefined,
    hitPoints: unit.hitPoints,
    maxHitPoints: unit.maxHitPoints,
    direction: unit.direction,
    onBridge: unit.onBridge,
    zone: unit.zone,
    buildStatus: unit.buildStatus,
    attackState: unit.attackState,
    factory: factoryToPlain(unit.factory),
    rallyPoint: tileToPlain(unit.rallyPoint),
    isPoweredOn: unit.isPoweredOn,
    hasWrenchRepair: unit.hasWrenchRepair,
    garrisonUnitCount: unit.garrisonUnitCount,
    garrisonUnitsMax: unit.garrisonUnitsMax,
    turretFacing: unit.turretFacing,
    turretNo: unit.turretNo,
    isIdle: unit.isIdle,
    canMove: unit.canMove,
    velocity: vectorToPlain(unit.velocity),
    stance: unit.stance,
    harvestedOre: unit.harvestedOre,
    harvestedGems: unit.harvestedGems,
    passengerSlotCount: unit.passengerSlotCount,
    passengerSlotMax: unit.passengerSlotMax,
    ammo: unit.ammo,
    isWarpedOut: unit.isWarpedOut,
    mindControlledBy: unit.mindControlledBy,
    tntTimer: unit.tntTimer,
    primaryWeapon: weaponToPlain(unit.primaryWeapon),
    secondaryWeapon: weaponToPlain(unit.secondaryWeapon),
  };

  Object.keys(plain).forEach((key) => {
    if (plain[key] === undefined) {
      delete plain[key];
    }
  });

  return plain;
}

function collectVisibleTiles(gameApi, playerName) {
  const size = gameApi.map.getRealMapSize();
  const tiles = gameApi.map.getTilesInRect({ x: 0, y: 0, width: size.width, height: size.height });
  const visibleTiles = [];

  for (const tile of tiles) {
    if (gameApi.map.isVisibleTile(tile, playerName)) {
      visibleTiles.push(tileToPlain(tile));
    }
  }

  return visibleTiles;
}

export function collectStaticMapDump(
  gameApi,
  {
    includeTerrainObjects = true,
    includeNeutralObjects = true,
  } = {},
) {
  const summary = mapToPlain(gameApi);
  const tiles = gameApi.map
    .getTilesInRect({ x: 0, y: 0, width: summary.width, height: summary.height })
    .slice()
    .sort((left, right) => (left.ry - right.ry) || (left.rx - right.rx));
  const terrainObjectIds = gameApi.getAllTerrainObjects().slice().sort((left, right) => left - right);
  const neutralObjectIds = gameApi.getNeutralUnits().slice().sort((left, right) => left - right);
  const terrainObjectIdSet = new Set(terrainObjectIds);
  const neutralObjectIdSet = new Set(neutralObjectIds);
  let bridgeTileCount = 0;
  let highBridgeTileCount = 0;
  let taggedTileCount = 0;
  let resourceTileCount = 0;

  const tileDump = tiles.map((tile) => {
    const hasBridge = gameApi.map.hasBridgeOnTile(tile);
    const hasHighBridge = gameApi.map.hasHighBridgeOnTile(tile);
    const resource = gameApi.map.getTileResourceData(tile);
    const objectIds = gameApi.map.getObjectsOnTile(tile);
    const terrainIds = objectIds.filter((id) => terrainObjectIdSet.has(id)).sort((left, right) => left - right);
    const neutralIds = objectIds.filter((id) => neutralObjectIdSet.has(id)).sort((left, right) => left - right);

    if (hasBridge) {
      bridgeTileCount += 1;
    }
    if (hasHighBridge) {
      highBridgeTileCount += 1;
    }
    if (tile.tag) {
      taggedTileCount += 1;
    }
    if (resource) {
      resourceTileCount += 1;
    }

    return {
      x: tile.rx,
      y: tile.ry,
      dx: tile.dx,
      dy: tile.dy,
      z: tile.z,
      tileNum: tile.tileNum,
      subTile: tile.subTile,
      terrainType: tile.terrainType,
      landType: tile.landType,
      onBridgeLandType: tile.onBridgeLandType,
      rampType: tile.rampType,
      id: tile.id,
      occluded: Boolean(tile.occluded),
      hasBridge,
      hasHighBridge,
      tag: tagToPlain(tile.tag),
      resource: tileResourceOnTileToPlain(resource),
      terrainObjectIds: terrainIds.length ? terrainIds : undefined,
      neutralObjectIds: neutralIds.length ? neutralIds : undefined,
    };
  });

  return {
    ...summary,
    tileCount: tileDump.length,
    bridgeTileCount,
    highBridgeTileCount,
    taggedTileCount,
    resourceTileCount,
    terrainObjectCount: terrainObjectIds.length,
    neutralObjectCount: neutralObjectIds.length,
    terrainObjects: includeTerrainObjects ? collectUnitsByIds(gameApi, terrainObjectIds, null) : undefined,
    neutralObjects: includeNeutralObjects ? collectUnitsByIds(gameApi, neutralObjectIds, null) : undefined,
    tiles: tileDump,
  };
}

export function collectVisibilitySnapshot(gameApi, playerName, { includeVisibleTiles = false } = {}) {
  const visibleTiles = collectVisibleTiles(gameApi, playerName);

  const selfUnitIds = gameApi.getVisibleUnits(playerName, "self").slice().sort((a, b) => a - b);
  const alliedUnitIds = gameApi.getVisibleUnits(playerName, "allied").slice().sort((a, b) => a - b);
  const enemyUnitIds = gameApi.getVisibleUnits(playerName, "enemy").slice().sort((a, b) => a - b);
  const hostileUnitIds = gameApi.getVisibleUnits(playerName, "hostile").slice().sort((a, b) => a - b);
  const { neutralUnitIds, otherHostileUnitIds } = splitVisibleNonEnemyHostiles(
    gameApi,
    playerName,
    hostileUnitIds,
    enemyUnitIds,
  );

  return {
    visibleTileCount: visibleTiles.length,
    visibleTiles: includeVisibleTiles ? visibleTiles : undefined,
    selfUnitIds,
    alliedUnitIds,
    enemyUnitIds,
    hostileUnitIds,
    neutralUnitIds,
    otherHostileUnitIds,
  };
}

export function collectPlayerObservationSnapshot(
  gameApi,
  {
    playerName,
    unitLimit = null,
    internalGame = null,
    includeVisibleTiles = false,
    includeVisibleResourceTiles = false,
    includeSuperWeapons = false,
    includeProduction = false,
    includePlayerStats = false,
  } = {},
) {
  if (!playerName) {
    throw new Error("collectPlayerObservationSnapshot requires a playerName.");
  }

  const internalPlayer = internalGame?.getPlayerByName ? internalGame.getPlayerByName(playerName) : null;
  const player = gameApi.getPlayerData(playerName);
  const visibility = collectVisibilitySnapshot(gameApi, playerName, {
    includeVisibleTiles,
  });
  const alliedOtherIds = differenceIds(visibility.alliedUnitIds, visibility.selfUnitIds);
  const visibleResourceTiles =
    includeVisibleResourceTiles && Array.isArray(visibility.visibleTiles)
      ? visibility.visibleTiles
          .map((tile) => gameApi.map.getTile(tile.x, tile.y))
          .filter(Boolean)
          .map((tile) => gameApi.map.getTileResourceData(tile))
          .filter(Boolean)
          .map(resourceTileToPlain)
      : undefined;
  const superWeapons = includeSuperWeapons
    ? gameApi
        .getAllSuperWeaponData()
        .filter((superWeapon) => superWeapon.playerName === playerName)
        .map(superWeaponToPlain)
        .filter(Boolean)
    : undefined;

  return {
    tick: gameApi.getCurrentTick(),
    gameTime: gameApi.getCurrentTime(),
    tickRate: gameApi.getTickRate(),
    baseTickRate: gameApi.getBaseTickRate(),
    map: mapToPlain(gameApi),
    player: {
      name: player.name,
      country: player.country?.name,
      isObserver: player.isObserver,
      isAi: player.isAi,
      isCombatant: player.isCombatant,
      isDefeated: gameApi.isPlayerDefeated(playerName),
      resigned: internalPlayer?.resigned,
      dropped: internalPlayer?.dropped,
      score: internalPlayer?.score,
      credits: player.credits,
      power: powerToPlain(player.power),
      radarDisabled: player.radarDisabled,
      startLocation: vectorToPlain(player.startLocation),
      production: includeProduction ? productionToPlain(internalPlayer?.production) : undefined,
      stats: includePlayerStats ? playerStatsToPlain(internalPlayer) : undefined,
    },
    observation: visibility,
    visibleResourceTiles,
    superWeapons,
    units: {
      self: collectUnitsByIds(gameApi, visibility.selfUnitIds, unitLimit),
      allied: collectUnitsByIds(gameApi, alliedOtherIds, unitLimit),
      enemy: collectUnitsByIds(gameApi, visibility.enemyUnitIds, unitLimit),
      neutral: collectUnitsByIds(gameApi, visibility.neutralUnitIds, unitLimit),
      otherHostile: collectUnitsByIds(gameApi, visibility.otherHostileUnitIds, unitLimit),
    },
  };
}

export function collectGameSnapshot(
  gameApi,
  {
    playerName,
    unitLimit = null,
    internalGame = null,
    includeTerrainObjects = false,
    includeNeutralUnits = false,
    includeTileResources = false,
    includeSuperWeapons = false,
    includePlayerProduction = false,
    includePlayerStats = false,
    includeVisibleTiles = false,
  } = {},
) {
  const players = gameApi
    .getPlayers()
    .slice()
    .sort()
    .map((name) => {
      const player = gameApi.getPlayerData(name);
      const internalPlayer = internalGame?.getPlayerByName ? internalGame.getPlayerByName(name) : null;
      return {
        name: player.name,
        country: player.country?.name,
        isObserver: player.isObserver,
        isAi: player.isAi,
        isCombatant: player.isCombatant,
        isDefeated: gameApi.isPlayerDefeated(name),
        resigned: internalPlayer?.resigned,
        dropped: internalPlayer?.dropped,
        score: internalPlayer?.score,
        isNeutral: internalPlayer?.isNeutral,
        credits: player.credits,
        power: powerToPlain(player.power),
        radarDisabled: player.radarDisabled,
        startLocation: vectorToPlain(player.startLocation),
        production: includePlayerProduction ? productionToPlain(internalPlayer?.production) : undefined,
        stats: includePlayerStats ? playerStatsToPlain(internalPlayer) : undefined,
      };
    });

  const allUnits = gameApi
    .getAllUnits()
    .map((id) => gameApi.getUnitData(id) ?? gameApi.getGameObjectData(id))
    .filter(Boolean)
    .map(unitToPlain)
    .sort((left, right) => left.id - right.id);

  const units = unitLimit === null ? allUnits : allUnits.slice(0, unitLimit);
  const neutralUnits = includeNeutralUnits
    ? collectUnitsByIds(gameApi, gameApi.getNeutralUnits().slice().sort((left, right) => left - right), unitLimit)
    : undefined;
  const terrainObjects = includeTerrainObjects
    ? collectUnitsByIds(gameApi, gameApi.getAllTerrainObjects().slice().sort((left, right) => left - right), unitLimit)
    : undefined;
  const tileResources = includeTileResources
    ? gameApi.map.getAllTilesResourceData().map(resourceTileToPlain).filter(Boolean)
    : undefined;
  const superWeapons = includeSuperWeapons
    ? gameApi.getAllSuperWeaponData().map(superWeaponToPlain).filter(Boolean)
    : undefined;
  const snapshot = {
    tick: gameApi.getCurrentTick(),
    gameTime: gameApi.getCurrentTime(),
    tickRate: gameApi.getTickRate(),
    baseTickRate: gameApi.getBaseTickRate(),
    map: mapToPlain(gameApi),
    players,
    totalUnitCount: allUnits.length,
    units,
    neutralUnits,
    terrainObjects,
    tileResources,
    superWeapons,
  };

  if (playerName) {
    snapshot.observation = collectVisibilitySnapshot(gameApi, playerName, {
      includeVisibleTiles,
    });
    snapshot.playerName = playerName;
  }

  return snapshot;
}

export function collectStaticGameData(gameApi, { includeMapDump = false } = {}) {
  return {
    capturedTick: gameApi.getCurrentTick(),
    map: mapToPlain(gameApi),
    generalRules: toPlainRecursive(gameApi.getGeneralRules()),
    rulesIni: iniFileToPlain(gameApi.getRulesIni()),
    artIni: iniFileToPlain(gameApi.getArtIni()),
    aiIni: iniFileToPlain(gameApi.getAiIni()),
    mapDump: includeMapDump ? collectStaticMapDump(gameApi) : undefined,
  };
}

export function collectPlayerStatsAtCurrentTick(gameApi, internalGame) {
  if (!internalGame?.getNonNeutralPlayers) {
    return [];
  }

  return internalGame
    .getNonNeutralPlayers()
    .filter((player) => !player.isObserver)
    .map((internalPlayer) => {
      const player = gameApi.getPlayerData(internalPlayer.name);
      return {
        name: internalPlayer.name,
        country: player?.country?.name,
        ai: Boolean(internalPlayer.isAi),
        defeated: Boolean(internalPlayer.defeated),
        credits: safeNumber(internalPlayer._credits, safeNumber(player?.credits, 0)),
        startLocation: vectorToPlain(player?.startLocation ?? internalPlayer.startLocation),
        score: safeNumber(internalPlayer.score),
        resigned: Boolean(internalPlayer.resigned),
        dropped: Boolean(internalPlayer.dropped),
        stats: playerStatsToPlain(internalPlayer),
      };
    });
}
