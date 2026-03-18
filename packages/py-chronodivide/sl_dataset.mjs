import { extractObservationFeatureSample, extractStaticMapFeatureSample, getObservationFeatureSchema, getStaticMapFeatureSchema } from "./features.mjs";
import { buildStaticTechTree } from "./availability.mjs";
import {
  buildActionTimelines,
  decodeActionLabel,
  getActionLabelSchema,
  shouldKeepRawActionId,
  updateSelectionFromAction,
} from "./labels.mjs";
import { buildReplayMetadata, createReplayResimContext } from "./resim_core.mjs";
import { productionToPlain, superWeaponToPlain } from "./snapshot.mjs";

const DEFAULT_LAST_ACTION_RAW_ID = -1;
const DEFAULT_LAST_ACTION_FAMILY_ID = -1;
const DEFAULT_LAST_ACTION_QUEUE = -1;
const DEFAULT_MISSING_INT = -1;
const STATIC_MAP_BUILDING_BY_SIDE_ID = {
  0: "GAPOWR",
  1: "NAPOWR",
  2: "YAPOWR",
};
const SUPER_WEAPON_RULE_SECTION_BY_TYPE_NAME = {
  MultiMissile: "MultiSpecial",
  IronCurtain: "IronCurtainSpecial",
  LightningStorm: "LightningStormSpecial",
  ChronoSphere: "ChronoSphereSpecial",
  ChronoWarp: "ChronoWarpSpecial",
  ParaDrop: "ParaDropSpecial",
  AmerParaDrop: "AmericanParaDropSpecial",
};

function numericOrDefault(value, fallback = 0) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function boolToInt(value, fallback = DEFAULT_MISSING_INT) {
  if (value === null || value === undefined) {
    return fallback;
  }
  return value ? 1 : 0;
}

function getStaticMapReferenceBuildingName(replayMetadata, playerName) {
  const replayPlayer = replayMetadata.players.find((player) => player.name === playerName);
  const sideId = replayPlayer?.sideId;
  if (sideId in STATIC_MAP_BUILDING_BY_SIDE_ID) {
    return STATIC_MAP_BUILDING_BY_SIDE_ID[sideId];
  }
  return "GAPOWR";
}

function buildSuperWeaponSchema(gameApi) {
  const rulesIni = gameApi?.getRulesIni?.();
  const typeNames = Object.keys(SUPER_WEAPON_RULE_SECTION_BY_TYPE_NAME);
  const rechargeRulesMinutesByType = {};
  const rechargeSecondsByType = {};
  const rulesSectionByType = {};

  for (const typeName of typeNames) {
    const sectionName = SUPER_WEAPON_RULE_SECTION_BY_TYPE_NAME[typeName];
    rulesSectionByType[typeName] = sectionName;
    let rechargeSeconds = null;

    if (rulesIni?.getSection) {
      const section = rulesIni.getSection(sectionName);
      const rawRechargeTime = section?.entries?.get?.("RechargeTime");
      const numericRechargeTime = rawRechargeTime === undefined ? Number.NaN : Number(rawRechargeTime);
      if (Number.isFinite(numericRechargeTime)) {
        rechargeSeconds = numericRechargeTime * 60.0;
        rechargeRulesMinutesByType[typeName] = numericRechargeTime;
      }
    }

    if (!(typeName in rechargeRulesMinutesByType)) {
      rechargeRulesMinutesByType[typeName] = null;
    }
    rechargeSecondsByType[typeName] = rechargeSeconds;
  }

  return {
    typeNames,
    rulesSectionByType,
    rechargeRulesMinutesByType,
    rechargeSecondsByType,
  };
}

function tileToTensor(tile) {
  if (!tile) {
    return [DEFAULT_MISSING_INT, DEFAULT_MISSING_INT];
  }
  return [numericOrDefault(tile.x, DEFAULT_MISSING_INT), numericOrDefault(tile.y, DEFAULT_MISSING_INT)];
}

function flattenNestedNumbers(value) {
  if (!Array.isArray(value)) {
    return [numericOrDefault(value, 0)];
  }
  const flattened = [];
  const stack = [value];

  while (stack.length) {
    const current = stack.pop();
    if (Array.isArray(current)) {
      for (let index = current.length - 1; index >= 0; index -= 1) {
        stack.push(current[index]);
      }
      continue;
    }
    flattened.push(numericOrDefault(current, 0));
  }

  return flattened;
}

function buildActionCounts(samples) {
  const rawCounts = new Map();
  const familyCounts = new Map();

  for (const sample of samples) {
    rawCounts.set(sample.label.rawActionId, (rawCounts.get(sample.label.rawActionId) ?? 0) + 1);
    familyCounts.set(sample.label.actionFamily, (familyCounts.get(sample.label.actionFamily) ?? 0) + 1);
  }

  return {
    rawActionCounts: Array.from(rawCounts, ([rawActionId, count]) => ({ rawActionId, count })).sort(
      (left, right) => left.rawActionId - right.rawActionId,
    ),
    actionFamilyCounts: Array.from(familyCounts, ([actionFamily, count]) => ({ actionFamily, count })).sort(
      (left, right) => right.count - left.count,
    ),
  };
}

function buildNameVocabulary(rawSamples) {
  const names = new Set();

  for (const sample of rawSamples) {
    for (const entity of sample.feature.entityMeta) {
      if (entity?.name) {
        names.add(entity.name);
      }
    }

    const candidates = [
      sample.label.targetObjectName,
      sample.label.itemName,
      sample.label.buildingName,
      sample.label.objectName,
    ];

    for (const candidate of candidates) {
      if (candidate) {
        names.add(candidate);
      }
    }

    for (const queue of sample.playerProduction?.queues ?? []) {
      for (const item of queue?.items ?? []) {
        if (item?.objectName) {
          names.add(item.objectName);
        }
      }
    }

    for (const availableObject of sample.playerProduction?.availableObjects ?? []) {
      if (availableObject?.name) {
        names.add(availableObject.name);
      }
    }
  }

  const idToName = ["<pad>", "<unk>", ...Array.from(names).sort((left, right) => left.localeCompare(right))];
  return {
    idToName,
    nameToId: Object.fromEntries(idToName.map((name, index) => [name, index])),
  };
}

function tokenForName(value, vocabulary, { allowMissing = true } = {}) {
  if (!value) {
    return allowMissing ? DEFAULT_MISSING_INT : vocabulary.nameToId["<pad>"];
  }
  return vocabulary.nameToId[value] ?? vocabulary.nameToId["<unk>"];
}

function buildEntityIndexById(entityMeta) {
  return new Map(entityMeta.map((entity, index) => [entity.id, index]));
}

function buildSelectionTensor(entityMeta, selectedUnitIds, maxSelectedUnits) {
  const ids = Array.isArray(selectedUnitIds) ? selectedUnitIds.slice() : [];
  const indices = Array(maxSelectedUnits).fill(DEFAULT_MISSING_INT);
  const mask = Array(maxSelectedUnits).fill(0);
  const resolvedMask = Array(maxSelectedUnits).fill(0);
  const entityIndexById = buildEntityIndexById(entityMeta);
  let resolvedCount = 0;

  for (let index = 0; index < Math.min(ids.length, maxSelectedUnits); index += 1) {
    const unitId = ids[index];
    const entityIndex = entityIndexById.has(unitId) ? entityIndexById.get(unitId) : DEFAULT_MISSING_INT;
    indices[index] = entityIndex;
    mask[index] = 1;
    if (entityIndex !== DEFAULT_MISSING_INT) {
      resolvedMask[index] = 1;
      resolvedCount += 1;
    }
  }

  return {
    indices,
    mask,
    resolvedMask,
    count: ids.length,
    resolvedCount,
    overflowCount: Math.max(0, ids.length - maxSelectedUnits),
  };
}

function resolveEntityIndex(entityMeta, objectId) {
  if (objectId === null || objectId === undefined) {
    return DEFAULT_MISSING_INT;
  }
  const entityIndexById = buildEntityIndexById(entityMeta);
  return entityIndexById.has(objectId) ? entityIndexById.get(objectId) : DEFAULT_MISSING_INT;
}

function buildFeatureTensors(rawSample, nameVocabulary, options) {
  const currentSelection = buildSelectionTensor(
    rawSample.feature.entityMeta,
    rawSample.selectionBeforeActionIds,
    options.maxSelectedUnits,
  );
  const entityNameTokens = rawSample.feature.entityMeta.map((entity) =>
    tokenForName(entity?.name, nameVocabulary, { allowMissing: false }),
  );

  while (entityNameTokens.length < options.maxEntities) {
    entityNameTokens.push(nameVocabulary.nameToId["<pad>"]);
  }

  return {
    scalar: rawSample.feature.scalarFeatures.slice(),
    lastActionContext: [
      numericOrDefault(rawSample.label.delayFromPreviousAction, DEFAULT_MISSING_INT),
      numericOrDefault(rawSample.lastActionContext.lastRawActionId, DEFAULT_LAST_ACTION_RAW_ID),
      numericOrDefault(rawSample.lastActionContext.lastActionFamilyId, DEFAULT_LAST_ACTION_FAMILY_ID),
      numericOrDefault(rawSample.lastActionContext.lastQueue, DEFAULT_LAST_ACTION_QUEUE),
    ],
    currentSelectionCount: [currentSelection.count],
    currentSelectionResolvedCount: [currentSelection.resolvedCount],
    currentSelectionOverflowCount: [currentSelection.overflowCount],
    currentSelectionIndices: currentSelection.indices,
    currentSelectionMask: currentSelection.mask,
    currentSelectionResolvedMask: currentSelection.resolvedMask,
    entityNameTokens,
    entityMask: rawSample.feature.entityMask.slice(),
    entityFeatures: rawSample.feature.entityFeatures.map((row) => row.slice()),
    spatial: rawSample.feature.spatial.data.map((plane) => plane.map((row) => row.slice())),
    minimap: rawSample.feature.minimap.data.map((plane) => plane.map((row) => row.slice())),
  };
}

function buildLabelTensors(rawSample, nameVocabulary, options) {
  const actionSelection = buildSelectionTensor(
    rawSample.feature.entityMeta,
    rawSample.label.actionSelectedUnitIds,
    options.maxSelectedUnits,
  );

  return {
    rawActionId: [numericOrDefault(rawSample.label.rawActionId, DEFAULT_MISSING_INT)],
    actionFamilyId: [numericOrDefault(rawSample.label.actionFamilyId, DEFAULT_MISSING_INT)],
    delayToNextAction: [numericOrDefault(rawSample.label.delayToNextAction, DEFAULT_MISSING_INT)],
    queue: [boolToInt(rawSample.label.queue)],
    orderTypeId: [numericOrDefault(rawSample.label.orderTypeId, DEFAULT_MISSING_INT)],
    targetModeId: [numericOrDefault(rawSample.label.targetModeId, DEFAULT_MISSING_INT)],
    targetEntityIndex: [resolveEntityIndex(rawSample.feature.entityMeta, rawSample.label.targetObjectId)],
    targetNameToken: [tokenForName(rawSample.label.targetObjectName, nameVocabulary)],
    targetObjectType: [numericOrDefault(rawSample.label.targetObjectType, DEFAULT_MISSING_INT)],
    targetTile: tileToTensor(rawSample.label.targetTile),
    targetIsOre: [rawSample.label.targetIsOre ? 1 : 0],
    actionSelectedUnitCount: [actionSelection.count],
    actionSelectedUnitResolvedCount: [actionSelection.resolvedCount],
    actionSelectedUnitOverflowCount: [actionSelection.overflowCount],
    actionSelectedUnitIndices: actionSelection.indices,
    actionSelectedUnitMask: actionSelection.mask,
    actionSelectedUnitResolvedMask: actionSelection.resolvedMask,
    queueTypeId: [numericOrDefault(rawSample.label.queueTypeId, DEFAULT_MISSING_INT)],
    queueUpdateTypeId: [numericOrDefault(rawSample.label.queueUpdateTypeId, DEFAULT_MISSING_INT)],
    quantity: [numericOrDefault(rawSample.label.quantity, DEFAULT_MISSING_INT)],
    itemNameToken: [tokenForName(rawSample.label.itemName, nameVocabulary)],
    itemType: [numericOrDefault(rawSample.label.itemType, DEFAULT_MISSING_INT)],
    itemCost: [numericOrDefault(rawSample.label.itemCost, DEFAULT_MISSING_INT)],
    buildingNameToken: [tokenForName(rawSample.label.buildingName, nameVocabulary)],
    buildingType: [numericOrDefault(rawSample.label.buildingType, DEFAULT_MISSING_INT)],
    buildingCost: [numericOrDefault(rawSample.label.buildingCost, DEFAULT_MISSING_INT)],
    buildingTile: tileToTensor(rawSample.label.buildingTile),
    objectEntityIndex: [resolveEntityIndex(rawSample.feature.entityMeta, rawSample.label.objectId)],
    objectNameToken: [tokenForName(rawSample.label.objectName, nameVocabulary)],
    objectType: [numericOrDefault(rawSample.label.objectType, DEFAULT_MISSING_INT)],
    superWeaponTypeId: [numericOrDefault(rawSample.label.superWeaponTypeId, DEFAULT_MISSING_INT)],
    superWeaponTile: tileToTensor(rawSample.label.superWeaponTile),
    superWeaponTile2: tileToTensor(rawSample.label.superWeaponTile2),
    pingTile: tileToTensor(rawSample.label.pingTile),
  };
}

export function flattenFeatureTensors(featureTensors) {
  return [
    ...featureTensors.scalar,
    ...featureTensors.lastActionContext,
    ...featureTensors.currentSelectionCount,
    ...featureTensors.currentSelectionResolvedCount,
    ...featureTensors.currentSelectionOverflowCount,
    ...featureTensors.currentSelectionIndices,
    ...featureTensors.currentSelectionMask,
    ...featureTensors.currentSelectionResolvedMask,
    ...featureTensors.entityNameTokens,
    ...featureTensors.entityMask,
    ...flattenNestedNumbers(featureTensors.entityFeatures),
    ...flattenNestedNumbers(featureTensors.spatial),
    ...flattenNestedNumbers(featureTensors.minimap),
  ];
}

export function flattenLabelTensors(labelTensors) {
  return [
    ...labelTensors.rawActionId,
    ...labelTensors.actionFamilyId,
    ...labelTensors.delayToNextAction,
    ...labelTensors.queue,
    ...labelTensors.orderTypeId,
    ...labelTensors.targetModeId,
    ...labelTensors.targetEntityIndex,
    ...labelTensors.targetNameToken,
    ...labelTensors.targetObjectType,
    ...labelTensors.targetTile,
    ...labelTensors.targetIsOre,
    ...labelTensors.actionSelectedUnitCount,
    ...labelTensors.actionSelectedUnitResolvedCount,
    ...labelTensors.actionSelectedUnitOverflowCount,
    ...labelTensors.actionSelectedUnitIndices,
    ...labelTensors.actionSelectedUnitMask,
    ...labelTensors.actionSelectedUnitResolvedMask,
    ...labelTensors.queueTypeId,
    ...labelTensors.queueUpdateTypeId,
    ...labelTensors.quantity,
    ...labelTensors.itemNameToken,
    ...labelTensors.itemType,
    ...labelTensors.itemCost,
    ...labelTensors.buildingNameToken,
    ...labelTensors.buildingType,
    ...labelTensors.buildingCost,
    ...labelTensors.buildingTile,
    ...labelTensors.objectEntityIndex,
    ...labelTensors.objectNameToken,
    ...labelTensors.objectType,
    ...labelTensors.superWeaponTypeId,
    ...labelTensors.superWeaponTile,
    ...labelTensors.superWeaponTile2,
    ...labelTensors.pingTile,
  ];
}

function buildTensorSchema(options, nameVocabulary) {
  const observationSchema = getObservationFeatureSchema({
    maxEntities: options.maxEntities,
    spatialSize: options.spatialSize,
    minimapSize: options.minimapSize,
  });
  const actionSchema = getActionLabelSchema();

  const featureSections = [
    { name: "scalar", shape: [observationSchema.scalarFeatureNames.length], dtype: "float32" },
    { name: "lastActionContext", shape: [4], dtype: "int32" },
    { name: "currentSelectionCount", shape: [1], dtype: "int32" },
    { name: "currentSelectionResolvedCount", shape: [1], dtype: "int32" },
    { name: "currentSelectionOverflowCount", shape: [1], dtype: "int32" },
    { name: "currentSelectionIndices", shape: [options.maxSelectedUnits], dtype: "int32" },
    { name: "currentSelectionMask", shape: [options.maxSelectedUnits], dtype: "int32" },
    { name: "currentSelectionResolvedMask", shape: [options.maxSelectedUnits], dtype: "int32" },
    { name: "entityNameTokens", shape: [options.maxEntities], dtype: "int32" },
    { name: "entityMask", shape: [options.maxEntities], dtype: "int32" },
    {
      name: "entityFeatures",
      shape: [options.maxEntities, observationSchema.entityFeatureNames.length],
      dtype: "float32",
    },
    {
      name: "spatial",
      shape: [observationSchema.spatialChannelNames.length, options.spatialSize, options.spatialSize],
      dtype: "float32",
    },
    {
      name: "minimap",
      shape: [observationSchema.minimapChannelNames.length, options.minimapSize, options.minimapSize],
      dtype: "float32",
    },
  ];

  const labelSections = [
    { name: "rawActionId", shape: [1], dtype: "int32" },
    { name: "actionFamilyId", shape: [1], dtype: "int32" },
    { name: "delayToNextAction", shape: [1], dtype: "int32" },
    { name: "queue", shape: [1], dtype: "int32" },
    { name: "orderTypeId", shape: [1], dtype: "int32" },
    { name: "targetModeId", shape: [1], dtype: "int32" },
    { name: "targetEntityIndex", shape: [1], dtype: "int32" },
    { name: "targetNameToken", shape: [1], dtype: "int32" },
    { name: "targetObjectType", shape: [1], dtype: "int32" },
    { name: "targetTile", shape: [2], dtype: "int32" },
    { name: "targetIsOre", shape: [1], dtype: "int32" },
    { name: "actionSelectedUnitCount", shape: [1], dtype: "int32" },
    { name: "actionSelectedUnitResolvedCount", shape: [1], dtype: "int32" },
    { name: "actionSelectedUnitOverflowCount", shape: [1], dtype: "int32" },
    { name: "actionSelectedUnitIndices", shape: [options.maxSelectedUnits], dtype: "int32" },
    { name: "actionSelectedUnitMask", shape: [options.maxSelectedUnits], dtype: "int32" },
    { name: "actionSelectedUnitResolvedMask", shape: [options.maxSelectedUnits], dtype: "int32" },
    { name: "queueTypeId", shape: [1], dtype: "int32" },
    { name: "queueUpdateTypeId", shape: [1], dtype: "int32" },
    { name: "quantity", shape: [1], dtype: "int32" },
    { name: "itemNameToken", shape: [1], dtype: "int32" },
    { name: "itemType", shape: [1], dtype: "int32" },
    { name: "itemCost", shape: [1], dtype: "int32" },
    { name: "buildingNameToken", shape: [1], dtype: "int32" },
    { name: "buildingType", shape: [1], dtype: "int32" },
    { name: "buildingCost", shape: [1], dtype: "int32" },
    { name: "buildingTile", shape: [2], dtype: "int32" },
    { name: "objectEntityIndex", shape: [1], dtype: "int32" },
    { name: "objectNameToken", shape: [1], dtype: "int32" },
    { name: "objectType", shape: [1], dtype: "int32" },
    { name: "superWeaponTypeId", shape: [1], dtype: "int32" },
    { name: "superWeaponTile", shape: [2], dtype: "int32" },
    { name: "superWeaponTile2", shape: [2], dtype: "int32" },
    { name: "pingTile", shape: [2], dtype: "int32" },
  ];

  return {
    observation: observationSchema,
    action: actionSchema,
    featureSections,
    labelSections,
    flatFeatureLength: featureSections.reduce(
      (length, section) => length + section.shape.reduce((product, dimension) => product * dimension, 1),
      0,
    ),
    flatLabelLength: labelSections.reduce(
      (length, section) => length + section.shape.reduce((product, dimension) => product * dimension, 1),
      0,
    ),
    sharedNameVocabulary: nameVocabulary,
    notes: [
      "Samples are action-centric: one observation tensor is aligned to one kept replay action.",
      "Current-selection tensors are inferred from the replay command stream before the current action is processed.",
      "Entity-relative selections and targets use indices into the current visible entity tensor and fall back to -1 when unresolved.",
      "Name vocabularies are built per extraction run for now; later dataset-wide passes should stabilize them.",
      "Per-sample playerProduction is carried through as a generic raw production summary for downstream feature builders.",
      "Per-sample playerSuperWeapons is carried through as a generic raw support-power summary for downstream feature builders.",
      "superWeaponSchema carries nominal recharge seconds from rules.ini so downstream feature builders can normalize super-weapon timers by type.",
      "JSON output contains numeric arrays that are tensor-ready but not framework-native .pt/.npz files.",
    ],
  };
}

function buildTensorSample(rawSample, nameVocabulary, options, includeFlat, includeDebug) {
  const featureTensors = buildFeatureTensors(rawSample, nameVocabulary, options);
  const labelTensors = buildLabelTensors(rawSample, nameVocabulary, options);
  const result = {
    tick: rawSample.tick,
    playerId: rawSample.playerId,
    playerName: rawSample.playerName,
    playerProduction: rawSample.playerProduction ?? null,
    playerSuperWeapons: rawSample.playerSuperWeapons ?? [],
    featureTensors,
    labelTensors,
  };

  if (includeFlat) {
    result.flatFeatureTensor = flattenFeatureTensors(featureTensors);
    result.flatLabelTensor = flattenLabelTensors(labelTensors);
  }

  if (includeDebug) {
    result.debug = {
      rawLabel: rawSample.label,
      selectionBeforeActionIds: rawSample.selectionBeforeActionIds.slice(),
      selectionAfterActionIds: rawSample.selectionAfterActionIds.slice(),
      entityMeta: rawSample.feature.entityMeta.map((entity) => ({ ...entity })),
      countsByName: {
        self: { ...rawSample.feature.countsByName.self },
        allied: { ...rawSample.feature.countsByName.allied },
        enemy: { ...rawSample.feature.countsByName.enemy },
        neutral: { ...rawSample.feature.countsByName.neutral },
        otherHostile: { ...rawSample.feature.countsByName.otherHostile },
      },
    };
  }

  return result;
}

function inferSampledPlayers(context, playerArg) {
  const replayPlayerNames = context.replay.gameOpts.humanPlayers.map((player) => player.name);
  if (playerArg === "all") {
    return replayPlayerNames;
  }
  return [playerArg ?? replayPlayerNames[0]].filter(Boolean);
}

export async function extractReplaySupervisedDataset({
  dataDir,
  replayPath,
  player = null,
  includeNoAction = false,
  includeUiActions = false,
  maxActions = null,
  maxTick = null,
  maxEntities = 128,
  maxSelectedUnits = 64,
  spatialSize = 32,
  minimapSize = 64,
  includeFlat = false,
  includeDebug = false,
} = {}) {
  if (!dataDir) {
    throw new Error("extractReplaySupervisedDataset requires a dataDir.");
  }
  if (!replayPath) {
    throw new Error("extractReplaySupervisedDataset requires a replayPath.");
  }

  const context = await createReplayResimContext({
    dataDir,
    replayPath,
  });
  const replayMetadata = buildReplayMetadata(context);
  const sampledPlayers = inferSampledPlayers(context, player);
  if (!sampledPlayers.length) {
    throw new Error("Could not infer a player for supervised dataset extraction.");
  }

  const sampledPlayerSet = new Set(sampledPlayers);
  const replayPlayerNames = context.replay.gameOpts.humanPlayers.map((playerInfo) => playerInfo.name);
  const timelines = buildActionTimelines(context.replay, {
    playerNames: sampledPlayers,
    includeNoAction,
    includeUiActions,
  });
  const timelineCursors = new Map(sampledPlayers.map((name) => [name, 0]));
  const currentSelections = new Map(replayPlayerNames.map((name) => [name, []]));
  const lastActionContexts = new Map(
    replayPlayerNames.map((name) => [
      name,
      {
        lastRawActionId: DEFAULT_LAST_ACTION_RAW_ID,
        lastActionFamilyId: DEFAULT_LAST_ACTION_FAMILY_ID,
        lastQueue: DEFAULT_LAST_ACTION_QUEUE,
      },
    ]),
  );
  const rawSamples = [];
  const resolvedMaxTick = maxTick === null ? context.replay.endTick : Math.min(maxTick, context.replay.endTick);

  while (context.gameApi.getCurrentTick() < resolvedMaxTick) {
    const currentTick = context.gameApi.getCurrentTick();
    const tickEvents = context.replayEventsByTick.get(currentTick) ?? [];
    const stagedActions = [];
    const featureCache = new Map();
    const productionCache = new Map();
    const superWeaponCache = new Map();

    for (const event of tickEvents) {
      if (event.constructor.name !== "TurnActionsReplayEvent") {
        continue;
      }

      for (const [playerId, actions] of event.payload) {
        const actionPlayer = context.internalGame.getPlayer(playerId);
        const playerName =
          actionPlayer?.name ?? context.replay.gameOpts.humanPlayers[playerId]?.name ?? `player_${playerId}`;

        for (const payload of actions) {
          const action = context.actionFactory.create(payload.id);
          action.player = actionPlayer;
          action.unserialize(payload.params);

          const selectionBeforeActionIds = currentSelections.get(playerName) ?? [];
          const selectionAfterActionIds = updateSelectionFromAction(action, selectionBeforeActionIds);
          const keepAction =
            sampledPlayerSet.has(playerName) &&
            shouldKeepRawActionId(payload.id, {
              includeNoAction,
              includeUiActions,
            });

          if (keepAction && (maxActions === null || rawSamples.length < maxActions)) {
            if (!featureCache.has(playerName)) {
              featureCache.set(
                playerName,
                extractObservationFeatureSample(context.gameApi, {
                  playerName,
                  maxEntities,
                  spatialSize,
                  minimapSize,
                }),
              );
            }
            if (!productionCache.has(playerName)) {
              const internalPlayer = context.internalGame.getPlayerByName
                ? context.internalGame.getPlayerByName(playerName)
                : null;
              productionCache.set(playerName, productionToPlain(internalPlayer?.production));
            }
            if (!superWeaponCache.has(playerName)) {
              const playerSuperWeapons = context.gameApi
                .getAllSuperWeaponData()
                .filter((superWeapon) => superWeapon.playerName === playerName)
                .map(superWeaponToPlain)
                .filter(Boolean);
              superWeaponCache.set(playerName, playerSuperWeapons);
            }

            const timeline = timelines.get(playerName) ?? [];
            const cursor = timelineCursors.get(playerName) ?? 0;
            const timelineEntry = timeline[cursor] ?? null;
            if (timelineEntry) {
              timelineCursors.set(playerName, cursor + 1);
            }

            const label = decodeActionLabel(action, {
              rawActionId: payload.id,
              gameApi: context.gameApi,
              selectionBefore: selectionBeforeActionIds,
              selectionAfter: selectionAfterActionIds,
              timelineEntry,
            });

            rawSamples.push({
              tick: currentTick,
              playerId,
              playerName,
              feature: featureCache.get(playerName),
              playerProduction: productionCache.get(playerName) ?? null,
              playerSuperWeapons: superWeaponCache.get(playerName) ?? [],
              selectionBeforeActionIds: selectionBeforeActionIds.slice(),
              selectionAfterActionIds: selectionAfterActionIds.slice(),
              label,
              lastActionContext: { ...(lastActionContexts.get(playerName) ?? {}) },
            });

            lastActionContexts.set(playerName, {
              lastRawActionId: label.rawActionId,
              lastActionFamilyId: label.actionFamilyId,
              lastQueue: boolToInt(label.queue),
            });
          }

          currentSelections.set(playerName, selectionAfterActionIds);
          stagedActions.push(action);
        }
      }
    }

    for (const action of stagedActions) {
      action.process();
    }

    context.internalGame.update();

    if (maxActions !== null && rawSamples.length >= maxActions) {
      break;
    }
  }

  const sharedNameVocabulary = buildNameVocabulary(rawSamples);
  const options = {
    maxEntities,
    maxSelectedUnits,
    spatialSize,
    minimapSize,
  };
  const superWeaponSchema = buildSuperWeaponSchema(context.gameApi);
  const staticTechTree = buildStaticTechTree(context.gameApi);
  const staticMapByPlayer = Object.fromEntries(
    sampledPlayers.map((playerName) => [
      playerName,
      extractStaticMapFeatureSample(context.gameApi, {
        playerName,
        spatialSize,
        buildabilityReferenceName: getStaticMapReferenceBuildingName(replayMetadata, playerName),
      }),
    ]),
  );
  const samples = rawSamples.map((rawSample) =>
    buildTensorSample(rawSample, sharedNameVocabulary, options, includeFlat, includeDebug),
  );

  return {
    replay: replayMetadata,
    sampledPlayers,
    options: {
      includeNoAction,
      includeUiActions,
      maxActions,
      maxTick: resolvedMaxTick,
      maxEntities,
      maxSelectedUnits,
      spatialSize,
      minimapSize,
      includeFlat,
      includeDebug,
    },
    schema: buildTensorSchema(options, sharedNameVocabulary),
    superWeaponSchema,
    staticTechTree,
    staticMapByPlayer,
    staticMapSchema: getStaticMapFeatureSchema({ spatialSize }),
    counts: buildActionCounts(rawSamples),
    samples,
  };
}
