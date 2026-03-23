import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { BotContext, GameApi, ObjectType, PlayerApi, PlayerData, Vector2 } from "@chronodivide/game-api";

import { computeAdjacentRect } from "../common/tileUtils.js";

import type {
    LivePolicyFeaturePayload,
    LivePolicyReplayPlayer,
    LivePolicyRuntimeState,
} from "./livePolicyTypes.js";

const DEFAULT_MAX_ENTITIES = 128;
const DEFAULT_SPATIAL_SIZE = 32;
const DEFAULT_MINIMAP_SIZE = 64;
const DEFAULT_MAX_SELECTED_UNITS = 64;
const DEFAULT_MISSING_INT = -1;
const STATIC_MAP_BUILDING_BY_SIDE_ID: Record<number, string> = {
    0: "GAPOWR",
    1: "NAPOWR",
    2: "YAPOWR",
};

type LiveFeatureModules = {
    extractObservationFeatureSample: (
        gameApi: GameApi,
        options: {
            playerName: string;
            maxEntities: number;
            spatialSize: number;
            minimapSize: number;
        },
    ) => any;
    extractStaticMapFeatureSample: (
        gameApi: GameApi,
        options: {
            playerName: string;
            spatialSize: number;
            buildabilityReferenceName: string;
        },
    ) => any;
    productionToPlain: (production: unknown) => Record<string, unknown> | undefined;
    superWeaponToPlain: (superWeapon: unknown) => Record<string, unknown> | undefined;
};

type LiveFeatureBuilderContext = Pick<BotContext, "game" | "player"> & {
    game: GameApi;
    player: PlayerApi;
};

let featureModulesPromise: Promise<LiveFeatureModules> | null = null;

function numericOrDefault(value: unknown, fallback = 0): number {
    return Number.isFinite(value) ? Number(value) : fallback;
}

function getFeatureModules(): Promise<LiveFeatureModules> {
    if (featureModulesPromise !== null) {
        return featureModulesPromise;
    }

    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const featuresModuleUrl = pathToFileURL(
        path.resolve(currentDir, "../../../../../py-chronodivide/features.mjs"),
    ).href;
    const snapshotModuleUrl = pathToFileURL(
        path.resolve(currentDir, "../../../../../py-chronodivide/snapshot.mjs"),
    ).href;

    featureModulesPromise = Promise.all([import(featuresModuleUrl), import(snapshotModuleUrl)]).then(
        ([featuresModule, snapshotModule]) => ({
            extractObservationFeatureSample: featuresModule.extractObservationFeatureSample,
            extractStaticMapFeatureSample: featuresModule.extractStaticMapFeatureSample,
            productionToPlain: snapshotModule.productionToPlain,
            superWeaponToPlain: snapshotModule.superWeaponToPlain,
        }),
    );
    return featureModulesPromise;
}

function getStaticMapReferenceBuildingName(playerData: PlayerData): string {
    const sideId = numericOrDefault((playerData.country as any)?.side ?? (playerData.country as any)?.rules?.side, -1);
    if (sideId in STATIC_MAP_BUILDING_BY_SIDE_ID) {
        return STATIC_MAP_BUILDING_BY_SIDE_ID[sideId];
    }
    return "GAPOWR";
}

function buildReplayPlayers(gameApi: GameApi): LivePolicyReplayPlayer[] {
    return gameApi
        .getPlayers()
        .map((playerName) => {
            const player = gameApi.getPlayerData(playerName);
            return {
                name: player.name,
                countryName: player.country?.name ?? null,
                sideId: numericOrDefault((player.country as any)?.side ?? (player.country as any)?.rules?.side, -1),
            };
        })
        .sort((left, right) => left.name.localeCompare(right.name));
}

function buildNameVocabulary(
    featureSample: any,
    playerProduction: Record<string, unknown> | null,
): { idToName: string[]; nameToId: Record<string, number> } {
    const names = new Set<string>();
    const entityMeta = Array.isArray(featureSample?.entityMeta)
        ? (featureSample.entityMeta as Array<{ name?: unknown }>)
        : [];
    entityMeta.forEach((entity) => {
        if (typeof entity?.name === "string" && entity.name) {
            names.add(entity.name);
        }
    });

    const queues = Array.isArray((playerProduction as any)?.queues) ? ((playerProduction as any)?.queues as any[]) : [];
    queues.forEach((queue) => {
        const items = Array.isArray(queue?.items) ? queue.items : [];
        items.forEach((item: { objectName?: unknown }) => {
            if (typeof item?.objectName === "string" && item.objectName) {
                names.add(item.objectName);
            }
        });
    });

    const availableObjects = Array.isArray((playerProduction as any)?.availableObjects)
        ? (((playerProduction as any)?.availableObjects as any[]) || [])
        : [];
    availableObjects.forEach((availableObject: { name?: unknown }) => {
        if (typeof availableObject?.name === "string" && availableObject.name) {
            names.add(availableObject.name);
        }
    });

    const idToName = ["<pad>", "<unk>", ...Array.from(names).sort((left, right) => left.localeCompare(right))];
    return {
        idToName,
        nameToId: Object.fromEntries(idToName.map((name, index) => [name, index])),
    };
}

function tokenForName(
    value: string | null | undefined,
    vocabulary: { nameToId: Record<string, number> },
    allowMissing: boolean,
): number {
    if (!value) {
        return allowMissing ? DEFAULT_MISSING_INT : vocabulary.nameToId["<pad>"];
    }
    return vocabulary.nameToId[value] ?? vocabulary.nameToId["<unk>"];
}

function buildEntityNameTokens(
    featureSample: any,
    vocabulary: { nameToId: Record<string, number> },
): number[] {
    const entityMeta = Array.isArray(featureSample?.entityMeta)
        ? (featureSample.entityMeta as Array<{ name?: unknown }>)
        : [];
    const tokens = entityMeta.map((entity) =>
        tokenForName(typeof entity?.name === "string" ? entity.name : undefined, vocabulary, false),
    );
    while (tokens.length < DEFAULT_MAX_ENTITIES) {
        tokens.push(vocabulary.nameToId["<pad>"]);
    }
    return tokens;
}

function buildEntityObjectIds(featureSample: any): number[] {
    const entityMeta = Array.isArray(featureSample?.entityMeta)
        ? (featureSample.entityMeta as Array<{ id?: unknown }>)
        : [];
    const ids = entityMeta.map((entity) => (Number.isInteger(entity?.id) ? Number(entity.id) : DEFAULT_MISSING_INT));
    while (ids.length < DEFAULT_MAX_ENTITIES) {
        ids.push(DEFAULT_MISSING_INT);
    }
    return ids;
}

function buildSelectionTensor(entityObjectIds: number[], selectedObjectIds: number[]) {
    const indices = Array(DEFAULT_MAX_SELECTED_UNITS).fill(DEFAULT_MISSING_INT);
    const mask = Array(DEFAULT_MAX_SELECTED_UNITS).fill(0);
    const resolvedMask = Array(DEFAULT_MAX_SELECTED_UNITS).fill(0);
    const objectIdToEntityIndex = new Map<number, number>();
    entityObjectIds.forEach((objectId, entityIndex) => {
        if (objectId >= 0 && !objectIdToEntityIndex.has(objectId)) {
            objectIdToEntityIndex.set(objectId, entityIndex);
        }
    });

    let resolvedCount = 0;
    for (let index = 0; index < Math.min(DEFAULT_MAX_SELECTED_UNITS, selectedObjectIds.length); index += 1) {
        const objectId = selectedObjectIds[index];
        mask[index] = 1;
        const entityIndex = objectIdToEntityIndex.get(objectId);
        if (entityIndex !== undefined) {
            indices[index] = entityIndex;
            resolvedMask[index] = 1;
            resolvedCount += 1;
        }
    }

    return {
        count: selectedObjectIds.length,
        resolvedCount,
        overflowCount: Math.max(0, selectedObjectIds.length - DEFAULT_MAX_SELECTED_UNITS),
        indices,
        mask,
        resolvedMask,
    };
}

function resolveVirtualSelectionObjectIds(
    context: LiveFeatureBuilderContext,
    runtimeState: LivePolicyRuntimeState,
): number[] {
    const explicitSelection = Array.isArray(runtimeState.currentSelectedObjectIds)
        ? runtimeState.currentSelectedObjectIds
              .filter((value) => Number.isInteger(value))
              .map((value) => Number(value))
              .slice(0, DEFAULT_MAX_SELECTED_UNITS)
        : [];
    if (explicitSelection.length > 0) {
        return explicitSelection;
    }
    return context.player
        .getVisibleUnits("self")
        .filter((value) => Number.isInteger(value))
        .map((value) => Number(value))
        .slice(0, DEFAULT_MAX_SELECTED_UNITS);
}

function buildPlayerSuperWeapons(
    gameApi: GameApi,
    playerName: string,
    superWeaponToPlain: LiveFeatureModules["superWeaponToPlain"],
) {
    return gameApi
        .getAllSuperWeaponData()
        .filter((superWeapon) => superWeapon.playerName === playerName)
        .map((superWeapon: unknown) => superWeaponToPlain(superWeapon))
        .filter(Boolean) as Array<Record<string, unknown>>;
}

function buildSuperWeaponRechargeSecondsByType(
    playerSuperWeapons: Array<Record<string, unknown>>,
): Record<string, number | null> {
    const values: Record<string, number | null> = {};
    playerSuperWeapons.forEach((entry) => {
        const typeName = typeof entry.typeName === "string" ? entry.typeName : null;
        if (!typeName) {
            return;
        }
        if (!(typeName in values)) {
            values[typeName] = null;
        }
    });
    return values;
}

const BUILDABILITY_ADJACENT_DISTANCE = 3;

function buildLiveBuildabilityMask(
    context: LiveFeatureBuilderContext,
    buildingName: string,
    spatialSize: number,
): number[][] {
    const mask: number[][] = Array.from({ length: spatialSize }, () => Array(spatialSize).fill(0));
    const mapSize = context.game.map.getRealMapSize();
    if (!mapSize || mapSize.width <= 0 || mapSize.height <= 0) {
        return mask;
    }

    const maxTileX = Math.max(1, mapSize.width - 1);
    const maxTileY = Math.max(1, mapSize.height - 1);
    const playerName = context.player.name;

    // Get all own buildings to find candidate placement areas.
    const buildingIds = context.game.getVisibleUnits(playerName, "self", (r) => r.type === ObjectType.Building);
    const checkedTiles = new Set<string>();

    for (const buildingId of buildingIds) {
        const building = context.game.getUnitData(buildingId);
        if (!building?.rules?.baseNormal || !building.foundation || !building.tile) {
            continue;
        }
        const buildingBase = new Vector2(building.tile.rx, building.tile.ry);
        const buildingSize = { width: building.foundation.width, height: building.foundation.height };
        const rect = computeAdjacentRect(buildingBase, buildingSize, BUILDABILITY_ADJACENT_DISTANCE);
        const adjacentTiles = context.game.mapApi.getTilesInRect(rect);

        for (const tile of adjacentTiles) {
            const tileKey = `${tile.rx},${tile.ry}`;
            if (checkedTiles.has(tileKey)) {
                continue;
            }
            checkedTiles.add(tileKey);

            if (context.game.canPlaceBuilding(playerName, buildingName, tile)) {
                const gridX = Math.min(spatialSize - 1, Math.max(0, Math.floor((tile.rx / maxTileX) * spatialSize)));
                const gridY = Math.min(spatialSize - 1, Math.max(0, Math.floor((tile.ry / maxTileY) * spatialSize)));
                mask[gridY][gridX] = 1;
            }
        }
    }

    return mask;
}

export async function buildLivePolicyFeaturePayload(
    context: LiveFeatureBuilderContext,
    runtimeState: LivePolicyRuntimeState,
): Promise<LivePolicyFeaturePayload> {
    const modules = await getFeatureModules();
    const playerData = context.game.getPlayerData(context.player.name);
    const observationSample = modules.extractObservationFeatureSample(context.game, {
        playerName: context.player.name,
        maxEntities: DEFAULT_MAX_ENTITIES,
        spatialSize: DEFAULT_SPATIAL_SIZE,
        minimapSize: DEFAULT_MINIMAP_SIZE,
    });
    const staticMapSample = modules.extractStaticMapFeatureSample(context.game, {
        playerName: context.player.name,
        spatialSize: DEFAULT_SPATIAL_SIZE,
        buildabilityReferenceName: getStaticMapReferenceBuildingName(playerData),
    });
    const referenceBuildingName = getStaticMapReferenceBuildingName(playerData);
    const liveBuildabilityMask = buildLiveBuildabilityMask(context, referenceBuildingName, DEFAULT_SPATIAL_SIZE);

    const playerProduction = context.player.production
        ? modules.productionToPlain(context.player.production) ?? null
        : null;
    const normalizedPlayerProduction = playerProduction ?? {
        maxTechLevel: 0,
        buildSpeedModifier: 0,
        queueCount: 0,
        queues: [],
        factoryCounts: [],
        availableCountsByQueueType: [],
        availableObjectsByQueueType: [],
        availableObjectCount: 0,
        availableObjects: [],
        catalogObjectCount: 0,
        catalogObjects: [],
    };

    const entityObjectIds = buildEntityObjectIds(observationSample);
    const sharedNameVocabulary = buildNameVocabulary(observationSample, normalizedPlayerProduction);
    const selectedObjectIds = resolveVirtualSelectionObjectIds(context, runtimeState);
    const currentSelection = buildSelectionTensor(entityObjectIds, selectedObjectIds);
    const playerSuperWeapons = buildPlayerSuperWeapons(context.game, context.player.name, modules.superWeaponToPlain);
    const delayFromPreviousAction =
        runtimeState.lastActionTick === null
            ? DEFAULT_MISSING_INT
            : Math.max(0, context.game.getCurrentTick() - runtimeState.lastActionTick);

    return {
        playerName: context.player.name,
        tick: context.game.getCurrentTick(),
        featureSchemaObservation: {
            scalarFeatureNames: Array.isArray(observationSample.scalarFeatureNames)
                ? observationSample.scalarFeatureNames.slice()
                : [],
            entityFeatureNames: Array.isArray(observationSample.entityFeatureNames)
                ? observationSample.entityFeatureNames.slice()
                : [],
            spatialChannelNames: Array.isArray(observationSample.spatial?.channelNames)
                ? observationSample.spatial.channelNames.slice()
                : [],
            minimapChannelNames: Array.isArray(observationSample.minimap?.channelNames)
                ? observationSample.minimap.channelNames.slice()
                : [],
            maxEntities: DEFAULT_MAX_ENTITIES,
            spatialSize: DEFAULT_SPATIAL_SIZE,
            minimapSize: DEFAULT_MINIMAP_SIZE,
        },
        featureTensors: {
            scalar: Array.isArray(observationSample.scalarFeatures) ? observationSample.scalarFeatures.slice() : [],
            lastActionContext: [delayFromPreviousAction, DEFAULT_MISSING_INT, runtimeState.lastQueueValue],
            currentSelectionCount: [currentSelection.count],
            currentSelectionResolvedCount: [currentSelection.resolvedCount],
            currentSelectionOverflowCount: [currentSelection.overflowCount],
            currentSelectionIndices: currentSelection.indices,
            currentSelectionMask: currentSelection.mask,
            currentSelectionResolvedMask: currentSelection.resolvedMask,
            entityNameTokens: buildEntityNameTokens(observationSample, sharedNameVocabulary),
            entityMask: Array.isArray(observationSample.entityMask) ? observationSample.entityMask.slice() : [],
            entityFeatures: Array.isArray(observationSample.entityFeatures)
                ? observationSample.entityFeatures.map((row: number[]) => row.slice())
                : [],
            spatial: Array.isArray(observationSample.spatial?.data)
                ? observationSample.spatial.data.map((plane: number[][]) => plane.map((row: number[]) => row.slice()))
                : [],
            minimap: Array.isArray(observationSample.minimap?.data)
                ? observationSample.minimap.data.map((plane: number[][]) => plane.map((row: number[]) => row.slice()))
                : [],
            mapStatic: Array.isArray(staticMapSample?.data)
                ? staticMapSample.data.map((plane: number[][]) => plane.map((row: number[]) => row.slice()))
                : [],
        },
        entityObjectIds,
        sharedNameVocabulary,
        playerProduction: normalizedPlayerProduction,
        playerSuperWeapons,
        replayPlayers: buildReplayPlayers(context.game),
        superWeaponRechargeSecondsByType: buildSuperWeaponRechargeSecondsByType(playerSuperWeapons),
        liveBuildabilityMask,
        runtimeState: {
            ...runtimeState,
            buildOrderActionTypeNamesV1: runtimeState.buildOrderActionTypeNamesV1.slice(),
            currentSelectedObjectIds: selectedObjectIds.slice(),
            recentActionFamilyNamesV2: (runtimeState.recentActionFamilyNamesV2 ?? []).slice(),
            recentOrderTypeNamesV2: (runtimeState.recentOrderTypeNamesV2 ?? []).slice(),
            pendingBuildingQueueByQueueName: {
                ...(runtimeState.pendingBuildingQueueByQueueName ?? {}),
            },
            lastOrderByUnitId: runtimeState.lastOrderByUnitId
                ? { ...runtimeState.lastOrderByUnitId }
                : undefined,
        },
    };
}
