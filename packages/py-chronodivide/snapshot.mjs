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

function factoryToPlain(factory) {
  if (!factory) {
    return undefined;
  }

  return {
    status: factory.status,
    progress: factory.progress,
    queueType: factory.queueType,
    objectName: factory.objectName,
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
  };
}

function differenceIds(left, right) {
  const rightSet = new Set(right);
  return left.filter((id) => !rightSet.has(id));
}

function uniqueSortedIds(ids) {
  return [...new Set(ids)].sort((left, right) => left - right);
}

function collectUnitsByIds(gameApi, ids, unitLimit = null) {
  const units = uniqueSortedIds(ids)
    .map((id) => gameApi.getUnitData(id) ?? gameApi.getGameObjectData(id))
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

export function collectVisibilitySnapshot(gameApi, playerName) {
  const size = gameApi.map.getRealMapSize();
  const tiles = gameApi.map.getTilesInRect({ x: 0, y: 0, width: size.width, height: size.height });

  let visibleTileCount = 0;
  for (const tile of tiles) {
    if (gameApi.map.isVisibleTile(tile, playerName)) {
      visibleTileCount += 1;
    }
  }

  return {
    visibleTileCount,
    selfUnitIds: gameApi.getVisibleUnits(playerName, "self").slice().sort((a, b) => a - b),
    alliedUnitIds: gameApi.getVisibleUnits(playerName, "allied").slice().sort((a, b) => a - b),
    enemyUnitIds: gameApi.getVisibleUnits(playerName, "enemy").slice().sort((a, b) => a - b),
    hostileUnitIds: gameApi.getVisibleUnits(playerName, "hostile").slice().sort((a, b) => a - b),
  };
}

export function collectPlayerObservationSnapshot(gameApi, { playerName, unitLimit = null } = {}) {
  if (!playerName) {
    throw new Error("collectPlayerObservationSnapshot requires a playerName.");
  }

  const player = gameApi.getPlayerData(playerName);
  const visibility = collectVisibilitySnapshot(gameApi, playerName);
  const alliedOtherIds = differenceIds(visibility.alliedUnitIds, visibility.selfUnitIds);
  const neutralOrOtherHostileIds = differenceIds(visibility.hostileUnitIds, visibility.enemyUnitIds);

  return {
    tick: gameApi.getCurrentTick(),
    gameTime: gameApi.getCurrentTime(),
    tickRate: gameApi.getTickRate(),
    map: {
      ...gameApi.map.getRealMapSize(),
    },
    player: {
      name: player.name,
      country: player.country?.name,
      isObserver: player.isObserver,
      isAi: player.isAi,
      isCombatant: player.isCombatant,
      credits: player.credits,
      power: powerToPlain(player.power),
      radarDisabled: player.radarDisabled,
      startLocation: vectorToPlain(player.startLocation),
    },
    observation: visibility,
    units: {
      self: collectUnitsByIds(gameApi, visibility.selfUnitIds, unitLimit),
      allied: collectUnitsByIds(gameApi, alliedOtherIds, unitLimit),
      enemy: collectUnitsByIds(gameApi, visibility.enemyUnitIds, unitLimit),
      neutralOrHostileOther: collectUnitsByIds(gameApi, neutralOrOtherHostileIds, unitLimit),
    },
  };
}

export function collectGameSnapshot(gameApi, { playerName, unitLimit = null } = {}) {
  const players = gameApi
    .getPlayers()
    .slice()
    .sort()
    .map((name) => {
      const player = gameApi.getPlayerData(name);
      return {
        name: player.name,
        country: player.country?.name,
        isObserver: player.isObserver,
        isAi: player.isAi,
        isCombatant: player.isCombatant,
        credits: player.credits,
        power: powerToPlain(player.power),
        radarDisabled: player.radarDisabled,
        startLocation: vectorToPlain(player.startLocation),
      };
    });

  const allUnits = gameApi
    .getAllUnits()
    .map((id) => gameApi.getUnitData(id) ?? gameApi.getGameObjectData(id))
    .filter(Boolean)
    .map(unitToPlain)
    .sort((left, right) => left.id - right.id);

  const units = unitLimit === null ? allUnits : allUnits.slice(0, unitLimit);
  const snapshot = {
    tick: gameApi.getCurrentTick(),
    gameTime: gameApi.getCurrentTime(),
    tickRate: gameApi.getTickRate(),
    map: {
      ...gameApi.map.getRealMapSize(),
    },
    players,
    totalUnitCount: allUnits.length,
    units,
  };

  if (playerName) {
    snapshot.observation = collectVisibilitySnapshot(gameApi, playerName);
    snapshot.playerName = playerName;
  }

  return snapshot;
}
