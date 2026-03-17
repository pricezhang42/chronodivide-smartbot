import { collectPlayerObservationSnapshot } from "./snapshot.mjs";

const OBJECT_TYPE = {
  Aircraft: 1,
  Building: 2,
  Infantry: 3,
  Vehicle: 7,
};

const RELATIONS = ["self", "allied", "enemy", "neutralOrHostileOther"];

const SCALAR_FEATURE_NAMES = [
  "tick",
  "game_time",
  "tick_rate",
  "map_width",
  "map_height",
  "start_x",
  "start_y",
  "credits",
  "power_total",
  "power_drain",
  "power_margin",
  "power_low",
  "radar_disabled",
  "visible_tile_count",
  "visible_tile_fraction",
  "self_unit_count",
  "self_building_count",
  "self_mobile_count",
  "self_infantry_count",
  "self_vehicle_count",
  "self_aircraft_count",
  "allied_unit_count",
  "enemy_unit_count",
  "enemy_building_count",
  "enemy_mobile_count",
  "enemy_infantry_count",
  "enemy_vehicle_count",
  "enemy_aircraft_count",
  "neutral_or_other_hostile_count",
  "self_purchase_value_sum",
  "enemy_visible_purchase_value_sum",
  "self_hit_points_sum",
  "enemy_visible_hit_points_sum",
];

const ENTITY_FEATURE_NAMES_BASE = [
  "relation_self",
  "relation_allied",
  "relation_enemy",
  "relation_neutral_or_other_hostile",
  "object_aircraft",
  "object_building",
  "object_infantry",
  "object_vehicle",
  "object_other",
  "tile_x",
  "tile_y",
  "tile_x_norm",
  "tile_y_norm",
  "hit_points",
  "max_hit_points",
  "hit_points_ratio",
  "sight",
  "veteran_level",
  "purchase_value",
  "direction_sin",
  "direction_cos",
  "turret_direction_sin",
  "turret_direction_cos",
  "velocity_x",
  "velocity_z",
  "foundation_width",
  "foundation_height",
  "is_idle",
  "can_move",
  "guard_mode",
  "on_bridge",
  "build_status_build_up",
  "build_status_ready",
  "build_status_build_down",
  "attack_state_idle",
  "attack_state_check_range",
  "attack_state_prepare_to_fire",
  "attack_state_fire_up",
  "attack_state_firing",
  "attack_state_just_fired",
  "is_powered_on",
  "has_wrench_repair",
  "garrison_fill_ratio",
  "passenger_fill_ratio",
  "harvested_ore",
  "harvested_gems",
  "ammo",
  "is_warped_out",
  "tnt_timer",
];

const SPATIAL_CHANNEL_NAMES = [
  "visible_tiles",
  "visible_ore",
  "visible_gems",
  "visible_ore_spawners",
  "self_presence",
  "allied_presence",
  "enemy_presence",
  "neutral_or_other_hostile_presence",
  "self_hit_points",
  "allied_hit_points",
  "enemy_hit_points",
  "neutral_or_other_hostile_hit_points",
  "self_mobile_presence",
  "self_building_presence",
  "enemy_mobile_presence",
  "enemy_building_presence",
];

function boolToNumber(value) {
  return value ? 1 : 0;
}

function safeNumber(value) {
  return Number.isFinite(value) ? value : 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function ratio(numerator, denominator) {
  if (!denominator) {
    return 0;
  }
  return numerator / denominator;
}

function angleToSinCos(angle) {
  if (!Number.isFinite(angle)) {
    return [0, 0];
  }

  const radians = (angle * Math.PI) / 180;
  return [Math.sin(radians), Math.cos(radians)];
}

function getObjectTypeFlags(type) {
  return {
    isAircraft: type === OBJECT_TYPE.Aircraft,
    isBuilding: type === OBJECT_TYPE.Building,
    isInfantry: type === OBJECT_TYPE.Infantry,
    isVehicle: type === OBJECT_TYPE.Vehicle,
    isOther:
      type !== OBJECT_TYPE.Aircraft &&
      type !== OBJECT_TYPE.Building &&
      type !== OBJECT_TYPE.Infantry &&
      type !== OBJECT_TYPE.Vehicle,
  };
}

function isMobileType(type) {
  return type === OBJECT_TYPE.Aircraft || type === OBJECT_TYPE.Infantry || type === OBJECT_TYPE.Vehicle;
}

function relationOneHot(relation) {
  return RELATIONS.map((candidate) => boolToNumber(candidate === relation));
}

function oneHotFromIndex(index, size) {
  return Array.from({ length: size }, (_, candidate) => boolToNumber(candidate === index));
}

function optionalOneHotFromIndex(index, size) {
  if (!Number.isInteger(index) || index < 0 || index >= size) {
    return Array.from({ length: size }, () => 0);
  }
  return oneHotFromIndex(index, size);
}

function createEmptyPlane(size) {
  return Array.from({ length: size }, () => Array(size).fill(0));
}

function createSpatialPlanes(size) {
  return SPATIAL_CHANNEL_NAMES.map(() => createEmptyPlane(size));
}

function tileToGrid(tile, map, spatialSize) {
  const maxX = Math.max(1, map.width - 1);
  const maxY = Math.max(1, map.height - 1);
  const normalizedX = clamp(safeNumber(tile?.x) / maxX, 0, 1);
  const normalizedY = clamp(safeNumber(tile?.y) / maxY, 0, 1);
  const gridX = clamp(Math.floor(normalizedX * spatialSize), 0, spatialSize - 1);
  const gridY = clamp(Math.floor(normalizedY * spatialSize), 0, spatialSize - 1);

  return {
    gridX,
    gridY,
    normalizedX,
    normalizedY,
  };
}

function incrementPlane(plane, x, y, value = 1) {
  plane[y][x] += value;
}

function aggregateRelation(units) {
  const result = {
    unitCount: units.length,
    buildingCount: 0,
    mobileCount: 0,
    infantryCount: 0,
    vehicleCount: 0,
    aircraftCount: 0,
    hitPointsSum: 0,
    purchaseValueSum: 0,
    countsByName: {},
  };

  for (const unit of units) {
    const flags = getObjectTypeFlags(unit.type);
    if (flags.isBuilding) {
      result.buildingCount += 1;
    }
    if (isMobileType(unit.type)) {
      result.mobileCount += 1;
    }
    if (flags.isInfantry) {
      result.infantryCount += 1;
    }
    if (flags.isVehicle) {
      result.vehicleCount += 1;
    }
    if (flags.isAircraft) {
      result.aircraftCount += 1;
    }

    result.hitPointsSum += safeNumber(unit.hitPoints);
    result.purchaseValueSum += safeNumber(unit.purchaseValue);
    result.countsByName[unit.name] = (result.countsByName[unit.name] ?? 0) + 1;
  }

  return result;
}

function sortedCountMap(countsByName) {
  return Object.fromEntries(
    Object.entries(countsByName).sort((left, right) => left[0].localeCompare(right[0])),
  );
}

function buildScalarFeatures(snapshot, aggregates) {
  const { map, player, observation } = snapshot;
  const visibleTileFraction = ratio(observation.visibleTileCount, map.width * map.height);
  const powerTotal = safeNumber(player.power?.total);
  const powerDrain = safeNumber(player.power?.drain);

  return [
    safeNumber(snapshot.tick),
    safeNumber(snapshot.gameTime),
    safeNumber(snapshot.tickRate),
    safeNumber(map.width),
    safeNumber(map.height),
    safeNumber(player.startLocation?.x),
    safeNumber(player.startLocation?.y),
    safeNumber(player.credits),
    powerTotal,
    powerDrain,
    powerTotal - powerDrain,
    boolToNumber(player.power?.isLowPower),
    boolToNumber(player.radarDisabled),
    safeNumber(observation.visibleTileCount),
    visibleTileFraction,
    aggregates.self.unitCount,
    aggregates.self.buildingCount,
    aggregates.self.mobileCount,
    aggregates.self.infantryCount,
    aggregates.self.vehicleCount,
    aggregates.self.aircraftCount,
    aggregates.allied.unitCount,
    aggregates.enemy.unitCount,
    aggregates.enemy.buildingCount,
    aggregates.enemy.mobileCount,
    aggregates.enemy.infantryCount,
    aggregates.enemy.vehicleCount,
    aggregates.enemy.aircraftCount,
    aggregates.neutralOrHostileOther.unitCount,
    aggregates.self.purchaseValueSum,
    aggregates.enemy.purchaseValueSum,
    aggregates.self.hitPointsSum,
    aggregates.enemy.hitPointsSum,
  ];
}

function encodeUnit(unit, relation, map) {
  const relationFlags = relationOneHot(relation);
  const objectFlags = getObjectTypeFlags(unit.type);
  const typeFlags = [
    boolToNumber(objectFlags.isAircraft),
    boolToNumber(objectFlags.isBuilding),
    boolToNumber(objectFlags.isInfantry),
    boolToNumber(objectFlags.isVehicle),
    boolToNumber(objectFlags.isOther),
  ];
  const tile = tileToGrid(unit.tile ?? { x: 0, y: 0 }, map, 1024);
  const [directionSin, directionCos] = angleToSinCos(unit.direction);
  const [turretSin, turretCos] = angleToSinCos(unit.turretFacing);
  const buildStatus = optionalOneHotFromIndex(unit.buildStatus, 3);
  const attackState = optionalOneHotFromIndex(unit.attackState, 6);

  return {
    features: [
      ...relationFlags,
      ...typeFlags,
      safeNumber(unit.tile?.x),
      safeNumber(unit.tile?.y),
      tile.normalizedX,
      tile.normalizedY,
      safeNumber(unit.hitPoints),
      safeNumber(unit.maxHitPoints),
      ratio(safeNumber(unit.hitPoints), safeNumber(unit.maxHitPoints)),
      safeNumber(unit.sight),
      safeNumber(unit.veteranLevel),
      safeNumber(unit.purchaseValue),
      directionSin,
      directionCos,
      turretSin,
      turretCos,
      safeNumber(unit.velocity?.x),
      safeNumber(unit.velocity?.z),
      safeNumber(unit.foundation?.width),
      safeNumber(unit.foundation?.height),
      boolToNumber(unit.isIdle),
      boolToNumber(unit.canMove),
      boolToNumber(unit.guardMode),
      boolToNumber(unit.onBridge),
      ...buildStatus,
      ...attackState,
      boolToNumber(unit.isPoweredOn),
      boolToNumber(unit.hasWrenchRepair),
      ratio(safeNumber(unit.garrisonUnitCount), safeNumber(unit.garrisonUnitsMax)),
      ratio(safeNumber(unit.passengerSlotCount), safeNumber(unit.passengerSlotMax)),
      safeNumber(unit.harvestedOre),
      safeNumber(unit.harvestedGems),
      safeNumber(unit.ammo),
      boolToNumber(unit.isWarpedOut),
      safeNumber(unit.tntTimer),
    ],
    meta: {
      id: unit.id,
      name: unit.name,
      owner: unit.owner,
      relation,
      objectType: unit.type,
      tile: unit.tile,
    },
  };
}

function encodeEntities(snapshot, maxEntities) {
  const orderedUnits = RELATIONS.flatMap((relation) =>
    (snapshot.units[relation] ?? [])
      .slice()
      .sort((left, right) => left.id - right.id)
      .map((unit) => ({ relation, unit })),
  );

  const active = orderedUnits.slice(0, maxEntities).map(({ relation, unit }) => encodeUnit(unit, relation, snapshot.map));
  const zeroVector = ENTITY_FEATURE_NAMES_BASE.map(() => 0);
  const entityFeatures = active.map((item) => item.features);
  const entityMask = active.map(() => 1);

  while (entityFeatures.length < maxEntities) {
    entityFeatures.push(zeroVector.slice());
    entityMask.push(0);
  }

  return {
    entityCount: active.length,
    entityMask,
    entityFeatures,
    entityMeta: active.map((item) => item.meta),
  };
}

function buildSpatialFeatures(gameApi, snapshot, spatialSize) {
  const planes = createSpatialPlanes(spatialSize);
  const map = gameApi.map;
  const size = map.getRealMapSize();
  const tiles = map.getTilesInRect({ x: 0, y: 0, width: size.width, height: size.height });

  for (const tile of tiles) {
    if (!map.isVisibleTile(tile, snapshot.player.name)) {
      continue;
    }

    const { gridX, gridY } = tileToGrid({ x: tile.rx ?? tile.x, y: tile.ry ?? tile.y }, snapshot.map, spatialSize);
    incrementPlane(planes[0], gridX, gridY, 1);

    const resource = map.getTileResourceData(tile);
    if (resource) {
      incrementPlane(planes[1], gridX, gridY, safeNumber(resource.ore));
      incrementPlane(planes[2], gridX, gridY, safeNumber(resource.gems));
      incrementPlane(planes[3], gridX, gridY, boolToNumber(resource.spawnsOre));
    }
  }

  for (const relation of RELATIONS) {
    for (const unit of snapshot.units[relation] ?? []) {
      const { gridX, gridY } = tileToGrid(unit.tile ?? { x: 0, y: 0 }, snapshot.map, spatialSize);
      const hp = safeNumber(unit.hitPoints);

      if (relation === "self") {
        incrementPlane(planes[4], gridX, gridY, 1);
        incrementPlane(planes[8], gridX, gridY, hp);
        incrementPlane(planes[12], gridX, gridY, boolToNumber(isMobileType(unit.type)));
        incrementPlane(planes[13], gridX, gridY, boolToNumber(unit.type === OBJECT_TYPE.Building));
      } else if (relation === "allied") {
        incrementPlane(planes[5], gridX, gridY, 1);
        incrementPlane(planes[9], gridX, gridY, hp);
      } else if (relation === "enemy") {
        incrementPlane(planes[6], gridX, gridY, 1);
        incrementPlane(planes[10], gridX, gridY, hp);
        incrementPlane(planes[14], gridX, gridY, boolToNumber(isMobileType(unit.type)));
        incrementPlane(planes[15], gridX, gridY, boolToNumber(unit.type === OBJECT_TYPE.Building));
      } else {
        incrementPlane(planes[7], gridX, gridY, 1);
        incrementPlane(planes[11], gridX, gridY, hp);
      }
    }
  }

  return {
    channelNames: SPATIAL_CHANNEL_NAMES.slice(),
    width: spatialSize,
    height: spatialSize,
    data: planes,
  };
}

function aggregateSnapshot(snapshot) {
  return {
    self: aggregateRelation(snapshot.units.self ?? []),
    allied: aggregateRelation(snapshot.units.allied ?? []),
    enemy: aggregateRelation(snapshot.units.enemy ?? []),
    neutralOrHostileOther: aggregateRelation(snapshot.units.neutralOrHostileOther ?? []),
  };
}

export function getObservationFeatureSchema({ maxEntities = 128, spatialSize = 32 } = {}) {
  return {
    scalarFeatureNames: SCALAR_FEATURE_NAMES.slice(),
    entityFeatureNames: ENTITY_FEATURE_NAMES_BASE.slice(),
    spatialChannelNames: SPATIAL_CHANNEL_NAMES.slice(),
    maxEntities,
    spatialSize,
    notes: [
      "All features are extracted from player-safe observation APIs.",
      "Do not replace this with getAllUnits() or enemy getPlayerData() calls for SL training data.",
      "Object-name tokens are generated per extraction run and emitted separately from the numeric entity features.",
    ],
  };
}

export function extractObservationFeatureSample(
  gameApi,
  { playerName, maxEntities = 128, spatialSize = 32 } = {},
) {
  const snapshot = collectPlayerObservationSnapshot(gameApi, {
    playerName,
    unitLimit: null,
  });
  const aggregates = aggregateSnapshot(snapshot);
  const entities = encodeEntities(snapshot, maxEntities);

  return {
    tick: snapshot.tick,
    gameTime: snapshot.gameTime,
    player: snapshot.player,
    observation: snapshot.observation,
    scalarFeatureNames: SCALAR_FEATURE_NAMES.slice(),
    scalarFeatures: buildScalarFeatures(snapshot, aggregates),
    entityFeatureNames: ENTITY_FEATURE_NAMES_BASE.slice(),
    entityCount: entities.entityCount,
    entityMask: entities.entityMask,
    entityFeatures: entities.entityFeatures,
    entityMeta: entities.entityMeta,
    spatial: buildSpatialFeatures(gameApi, snapshot, spatialSize),
    countsByName: {
      self: sortedCountMap(aggregates.self.countsByName),
      allied: sortedCountMap(aggregates.allied.countsByName),
      enemy: sortedCountMap(aggregates.enemy.countsByName),
      neutralOrHostileOther: sortedCountMap(aggregates.neutralOrHostileOther.countsByName),
    },
  };
}
