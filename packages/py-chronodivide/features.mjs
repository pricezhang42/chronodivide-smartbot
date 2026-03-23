import { collectPlayerObservationSnapshot } from "./snapshot.mjs";

const OBJECT_TYPE = {
  Aircraft: 1,
  Building: 2,
  Infantry: 3,
  Vehicle: 7,
};

const SPEED_TYPE = {
  Foot: 0,
  Track: 1,
};

const RELATIONS = ["self", "allied", "enemy", "neutral", "otherHostile"];

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
  "neutral_unit_count",
  "other_hostile_unit_count",
  "self_purchase_value_sum",
  "enemy_visible_purchase_value_sum",
  "self_hit_points_sum",
  "enemy_visible_hit_points_sum",
];

// Cumulative game statistics — only available during replay extraction (not at live inference).
// These are extracted as a separate feature group so the model can handle their absence.
const GAME_STATS_FEATURE_NAMES = [
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
];

const ENTITY_FEATURE_NAMES_BASE = [
  "relation_self",
  "relation_allied",
  "relation_enemy",
  "relation_neutral",
  "relation_other_hostile",
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
  "factory_status_idle",
  "factory_status_delivering",
  "factory_has_delivery",
  "rally_point_valid",
  "rally_x_norm",
  "rally_y_norm",
  "primary_weapon_cooldown_ticks",
  "secondary_weapon_cooldown_ticks",
];

const SPATIAL_CHANNEL_NAMES = [
  "visible_tiles",
  "visible_ore",
  "visible_gems",
  "visible_ore_spawners",
  "self_presence",
  "allied_presence",
  "enemy_presence",
  "neutral_presence",
  "other_hostile_presence",
  "self_hit_points",
  "allied_hit_points",
  "enemy_hit_points",
  "neutral_hit_points",
  "other_hostile_hit_points",
  "self_mobile_presence",
  "self_building_presence",
  "enemy_mobile_presence",
  "enemy_building_presence",
];

const MINIMAP_CHANNEL_NAMES = [
  "visible_tiles",
  "hidden_tiles",
  "visible_ore",
  "visible_gems",
  "visible_ore_spawners",
  "self_presence",
  "allied_presence",
  "enemy_presence",
  "neutral_presence",
  "other_hostile_presence",
  "self_building_presence",
  "self_mobile_presence",
  "enemy_building_presence",
  "enemy_mobile_presence",
  "self_hit_points",
  "allied_hit_points",
  "enemy_hit_points",
  "self_start_location",
];

const STATIC_MAP_CHANNEL_NAMES = [
  "foot_passable",
  "track_passable",
  "buildable_reference",
  "terrain_height_norm",
  "start_locations",
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

function createPlanes(channelNames, size) {
  return channelNames.map(() => createEmptyPlane(size));
}

function createSpatialPlanes(size) {
  return createPlanes(SPATIAL_CHANNEL_NAMES, size);
}

function createMinimapPlanes(size) {
  return createPlanes(MINIMAP_CHANNEL_NAMES, size);
}

function createStaticMapPlanes(size) {
  return createPlanes(STATIC_MAP_CHANNEL_NAMES, size);
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

function markTileOnPlane(plane, tile, map, size, value = 1) {
  const { gridX, gridY } = tileToGrid(tile, map, size);
  incrementPlane(plane, gridX, gridY, value);
}

export function getStaticMapFeatureSchema({ spatialSize = 32 } = {}) {
  return {
    staticMapChannelNames: STATIC_MAP_CHANNEL_NAMES.slice(),
    spatialSize,
    notes: [
      "Static map features are replay-constant, observation-safe priors derived at replay start.",
      "Buildability uses a caller-supplied reference building and ignores adjacency checks in the first pass.",
      "Passability is summarized as a fraction of tiles in each grid cell that are passable for the chosen speed type.",
    ],
  };
}

export function extractStaticMapFeatureSample(
  gameApi,
  { playerName, spatialSize = 32, buildabilityReferenceName = null } = {},
) {
  if (!playerName) {
    throw new Error("extractStaticMapFeatureSample requires a playerName.");
  }

  const planes = createStaticMapPlanes(spatialSize);
  const countPlane = createEmptyPlane(spatialSize);
  const map = gameApi.map;
  const size = map.getRealMapSize();
  const tiles = map.getTilesInRect({ x: 0, y: 0, width: size.width, height: size.height });
  let minZ = Infinity;
  let maxZ = -Infinity;

  for (const tile of tiles) {
    const z = safeNumber(tile?.z);
    if (z < minZ) {
      minZ = z;
    }
    if (z > maxZ) {
      maxZ = z;
    }
  }
  const zRange = Math.max(1, maxZ - minZ);

  for (const tile of tiles) {
    const tilePoint = { x: tile.rx ?? tile.x, y: tile.ry ?? tile.y };
    const { gridX, gridY } = tileToGrid(tilePoint, { width: size.width, height: size.height }, spatialSize);
    countPlane[gridY][gridX] += 1;

    const onBridge = map.hasBridgeOnTile(tile);
    const footPassable = map.isPassableTile(tile, SPEED_TYPE.Foot, false, false) || (onBridge && map.isPassableTile(tile, SPEED_TYPE.Foot, true, false));
    const trackPassable = map.isPassableTile(tile, SPEED_TYPE.Track, false, false) || (onBridge && map.isPassableTile(tile, SPEED_TYPE.Track, true, false));
    const buildableReference =
      buildabilityReferenceName !== null
        ? gameApi.canPlaceBuilding(playerName, buildabilityReferenceName, tile, { ignoreAdjacent: true })
        : false;
    const heightNorm = clamp((safeNumber(tile?.z) - minZ) / zRange, 0, 1);

    incrementPlane(planes[0], gridX, gridY, boolToNumber(footPassable));
    incrementPlane(planes[1], gridX, gridY, boolToNumber(trackPassable));
    incrementPlane(planes[2], gridX, gridY, boolToNumber(buildableReference));
    incrementPlane(planes[3], gridX, gridY, heightNorm);
  }

  for (let y = 0; y < spatialSize; y += 1) {
    for (let x = 0; x < spatialSize; x += 1) {
      const count = countPlane[y][x];
      if (!count) {
        continue;
      }
      planes[0][y][x] /= count;
      planes[1][y][x] /= count;
      planes[2][y][x] /= count;
      planes[3][y][x] /= count;
    }
  }

  for (const location of map.getStartingLocations()) {
    markTileOnPlane(planes[4], location, { width: size.width, height: size.height }, spatialSize, 1);
  }

  return {
    channelNames: STATIC_MAP_CHANNEL_NAMES.slice(),
    width: spatialSize,
    height: spatialSize,
    buildabilityReferenceName,
    data: planes,
  };
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

function counterByTypeCount(counterArray, objectType) {
  if (!Array.isArray(counterArray)) return 0;
  const entry = counterArray.find((entry) => entry.objectType === objectType);
  return safeNumber(entry?.count, 0);
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
    aggregates.neutral.unitCount,
    aggregates.otherHostile.unitCount,
    aggregates.self.purchaseValueSum,
    aggregates.enemy.purchaseValueSum,
    aggregates.self.hitPointsSum,
    aggregates.enemy.hitPointsSum,
  ];
}

function buildGameStatsFeatures(snapshot) {
  const stats = snapshot.player?.stats;
  if (!stats) {
    return null;
  }
  return [
    safeNumber(snapshot.player.score, 0),
    safeNumber(stats.creditsGained, 0),
    safeNumber(stats.buildingsCaptured, 0),
    counterByTypeCount(stats.unitsBuiltByType, OBJECT_TYPE.Aircraft),
    counterByTypeCount(stats.unitsBuiltByType, OBJECT_TYPE.Building),
    counterByTypeCount(stats.unitsBuiltByType, OBJECT_TYPE.Infantry),
    counterByTypeCount(stats.unitsBuiltByType, OBJECT_TYPE.Vehicle),
    counterByTypeCount(stats.unitsKilledByType, OBJECT_TYPE.Aircraft),
    counterByTypeCount(stats.unitsKilledByType, OBJECT_TYPE.Building),
    counterByTypeCount(stats.unitsKilledByType, OBJECT_TYPE.Infantry),
    counterByTypeCount(stats.unitsKilledByType, OBJECT_TYPE.Vehicle),
    counterByTypeCount(stats.unitsLostByType, OBJECT_TYPE.Aircraft),
    counterByTypeCount(stats.unitsLostByType, OBJECT_TYPE.Building),
    counterByTypeCount(stats.unitsLostByType, OBJECT_TYPE.Infantry),
    counterByTypeCount(stats.unitsLostByType, OBJECT_TYPE.Vehicle),
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
  const factoryStatus = optionalOneHotFromIndex(unit.factory?.status, 2);
  const rallyPoint = unit.rallyPoint ? tileToGrid(unit.rallyPoint, map, 1024) : null;
  const primaryWeaponCooldownTicks = safeNumber(unit.primaryWeapon?.cooldownTicks);
  const secondaryWeaponCooldownTicks = safeNumber(unit.secondaryWeapon?.cooldownTicks);

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
      ...factoryStatus,
      boolToNumber(Number.isFinite(unit.factory?.deliveringUnit)),
      boolToNumber(Boolean(rallyPoint)),
      rallyPoint?.normalizedX ?? 0,
      rallyPoint?.normalizedY ?? 0,
      primaryWeaponCooldownTicks,
      secondaryWeaponCooldownTicks,
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
        incrementPlane(planes[9], gridX, gridY, hp);
        incrementPlane(planes[14], gridX, gridY, boolToNumber(isMobileType(unit.type)));
        incrementPlane(planes[15], gridX, gridY, boolToNumber(unit.type === OBJECT_TYPE.Building));
      } else if (relation === "allied") {
        incrementPlane(planes[5], gridX, gridY, 1);
        incrementPlane(planes[10], gridX, gridY, hp);
      } else if (relation === "enemy") {
        incrementPlane(planes[6], gridX, gridY, 1);
        incrementPlane(planes[11], gridX, gridY, hp);
        incrementPlane(planes[16], gridX, gridY, boolToNumber(isMobileType(unit.type)));
        incrementPlane(planes[17], gridX, gridY, boolToNumber(unit.type === OBJECT_TYPE.Building));
      } else if (relation === "neutral") {
        incrementPlane(planes[7], gridX, gridY, 1);
        incrementPlane(planes[12], gridX, gridY, hp);
      } else {
        incrementPlane(planes[8], gridX, gridY, 1);
        incrementPlane(planes[13], gridX, gridY, hp);
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

function buildMinimapFeatures(gameApi, snapshot, minimapSize) {
  const planes = createMinimapPlanes(minimapSize);
  const map = gameApi.map;
  const size = map.getRealMapSize();
  const tiles = map.getTilesInRect({ x: 0, y: 0, width: size.width, height: size.height });

  for (const tile of tiles) {
    const tilePoint = { x: tile.rx ?? tile.x, y: tile.ry ?? tile.y };
    if (map.isVisibleTile(tile, snapshot.player.name)) {
      markTileOnPlane(planes[0], tilePoint, snapshot.map, minimapSize, 1);

      const resource = map.getTileResourceData(tile);
      if (resource) {
        markTileOnPlane(planes[2], tilePoint, snapshot.map, minimapSize, safeNumber(resource.ore));
        markTileOnPlane(planes[3], tilePoint, snapshot.map, minimapSize, safeNumber(resource.gems));
        markTileOnPlane(planes[4], tilePoint, snapshot.map, minimapSize, boolToNumber(resource.spawnsOre));
      }
    } else {
      markTileOnPlane(planes[1], tilePoint, snapshot.map, minimapSize, 1);
    }
  }

  for (const relation of RELATIONS) {
    for (const unit of snapshot.units[relation] ?? []) {
      const tilePoint = unit.tile ?? { x: 0, y: 0 };
      const hp = safeNumber(unit.hitPoints);

      if (relation === "self") {
        markTileOnPlane(planes[5], tilePoint, snapshot.map, minimapSize, 1);
        markTileOnPlane(planes[10], tilePoint, snapshot.map, minimapSize, boolToNumber(unit.type === OBJECT_TYPE.Building));
        markTileOnPlane(planes[11], tilePoint, snapshot.map, minimapSize, boolToNumber(isMobileType(unit.type)));
        markTileOnPlane(planes[14], tilePoint, snapshot.map, minimapSize, hp);
      } else if (relation === "allied") {
        markTileOnPlane(planes[6], tilePoint, snapshot.map, minimapSize, 1);
        markTileOnPlane(planes[15], tilePoint, snapshot.map, minimapSize, hp);
      } else if (relation === "enemy") {
        markTileOnPlane(planes[7], tilePoint, snapshot.map, minimapSize, 1);
        markTileOnPlane(planes[12], tilePoint, snapshot.map, minimapSize, boolToNumber(unit.type === OBJECT_TYPE.Building));
        markTileOnPlane(planes[13], tilePoint, snapshot.map, minimapSize, boolToNumber(isMobileType(unit.type)));
        markTileOnPlane(planes[16], tilePoint, snapshot.map, minimapSize, hp);
      } else if (relation === "neutral") {
        markTileOnPlane(planes[8], tilePoint, snapshot.map, minimapSize, 1);
      } else {
        markTileOnPlane(planes[9], tilePoint, snapshot.map, minimapSize, 1);
      }
    }
  }

  if (snapshot.player.startLocation) {
    markTileOnPlane(planes[17], snapshot.player.startLocation, snapshot.map, minimapSize, 1);
  }

  return {
    channelNames: MINIMAP_CHANNEL_NAMES.slice(),
    width: minimapSize,
    height: minimapSize,
    data: planes,
  };
}

function aggregateSnapshot(snapshot) {
  return {
    self: aggregateRelation(snapshot.units.self ?? []),
    allied: aggregateRelation(snapshot.units.allied ?? []),
    enemy: aggregateRelation(snapshot.units.enemy ?? []),
    neutral: aggregateRelation(snapshot.units.neutral ?? []),
    otherHostile: aggregateRelation(snapshot.units.otherHostile ?? []),
  };
}

export function getObservationFeatureSchema({ maxEntities = 128, spatialSize = 32, minimapSize = 64 } = {}) {
  return {
    scalarFeatureNames: SCALAR_FEATURE_NAMES.slice(),
    entityFeatureNames: ENTITY_FEATURE_NAMES_BASE.slice(),
    spatialChannelNames: SPATIAL_CHANNEL_NAMES.slice(),
    minimapChannelNames: MINIMAP_CHANNEL_NAMES.slice(),
    maxEntities,
    spatialSize,
    minimapSize,
    notes: [
      "All features are extracted from player-safe observation APIs.",
      "Do not replace this with getAllUnits() or enemy getPlayerData() calls for SL training data.",
      "Object-name tokens are generated per extraction run and emitted separately from the numeric entity features.",
      "The minimap is a compact top-down tensor, not a rendered UI image.",
    ],
  };
}

export function extractObservationFeatureSample(
  gameApi,
  { playerName, maxEntities = 128, spatialSize = 32, minimapSize = 64, internalGame = null } = {},
) {
  const snapshot = collectPlayerObservationSnapshot(gameApi, {
    playerName,
    unitLimit: null,
    internalGame,
    includePlayerStats: internalGame != null,
  });
  const aggregates = aggregateSnapshot(snapshot);
  const entities = encodeEntities(snapshot, maxEntities);

  const gameStats = buildGameStatsFeatures(snapshot);
  return {
    tick: snapshot.tick,
    gameTime: snapshot.gameTime,
    player: snapshot.player,
    observation: snapshot.observation,
    scalarFeatureNames: SCALAR_FEATURE_NAMES.slice(),
    scalarFeatures: buildScalarFeatures(snapshot, aggregates),
    gameStatsFeatureNames: gameStats != null ? GAME_STATS_FEATURE_NAMES.slice() : undefined,
    gameStatsFeatures: gameStats ?? undefined,
    entityFeatureNames: ENTITY_FEATURE_NAMES_BASE.slice(),
    entityCount: entities.entityCount,
    entityMask: entities.entityMask,
    entityFeatures: entities.entityFeatures,
    entityMeta: entities.entityMeta,
    spatial: buildSpatialFeatures(gameApi, snapshot, spatialSize),
    minimap: buildMinimapFeatures(gameApi, snapshot, minimapSize),
    countsByName: {
      self: sortedCountMap(aggregates.self.countsByName),
      allied: sortedCountMap(aggregates.allied.countsByName),
      enemy: sortedCountMap(aggregates.enemy.countsByName),
      neutral: sortedCountMap(aggregates.neutral.countsByName),
      otherHostile: sortedCountMap(aggregates.otherHostile.countsByName),
    },
  };
}
