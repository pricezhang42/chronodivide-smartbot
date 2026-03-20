import { GameApi, QueueStatus, QueueType } from "@chronodivide/game-api";

import { countBy } from "../common/utils.js";
import { SupabotContext } from "../common/context.js";
import { buildLivePolicyFeaturePayload } from "../modelcontrol/liveFeaturePayloadBuilder.js";
import { UnitRequest } from "../mission/missionController.js";
import { GlobalThreat } from "../threat/threat.js";
import { QUEUE_TYPE_TO_ADVISOR_QUEUE } from "./productionAdvisorTypes.js";
import type { CheckpointAdvisorRuntimeState, ProductionAdvisorInput } from "./productionAdvisorTypes.js";

const QUEUES = [
    QueueType.Structures,
    QueueType.Armory,
    QueueType.Infantry,
    QueueType.Vehicles,
    QueueType.Aircrafts,
    QueueType.Ships,
];

function getOwnedCounts(context: SupabotContext, playerName: string, relationship: string): Record<string, number> {
    const ids = context.game.getVisibleUnits(playerName, relationship as any);
    return countBy(
        ids
            .map((unitId) => context.game.getGameObjectData(unitId))
            .filter((unit): unit is NonNullable<typeof unit> => !!unit),
        (unit) => unit.name,
    );
}

function getHarvesterCount(context: SupabotContext): number {
    return context.game.getVisibleUnits(context.player.name, "self", (rules) => !!rules.harvester).length;
}

function toThreatSummary(threat: GlobalThreat | null) {
    if (!threat) {
        return null;
    }
    return {
        certainty: threat.certainty,
        enemyLandThreat: threat.totalOffensiveLandThreat,
        enemyAirThreat: threat.totalOffensiveAirThreat,
        enemyAntiAirThreat: threat.totalOffensiveAntiAirThreat,
        enemyDefensiveThreat: threat.totalDefensiveThreat,
        ownDefensivePower: threat.totalDefensivePower,
        ownAntiGroundPower: threat.totalAvailableAntiGroundFirepower,
        ownAntiAirPower: threat.totalAvailableAntiAirFirepower,
        ownAirPower: threat.totalAvailableAirPower,
    };
}

function toQueueSnapshot(context: SupabotContext, queueType: QueueType) {
    const queueData = context.player.production.getQueueData(queueType);
    const availableOptions = context.player.production.getAvailableObjects(queueType).map((rules) => ({
        name: rules.name,
        cost: Number.isFinite(rules.cost) ? rules.cost : 0,
    }));
    return {
        queue: QUEUE_TYPE_TO_ADVISOR_QUEUE[queueType],
        status: queueData.status ?? QueueStatus.Idle,
        activeItems: queueData.items.map((item) => item.rules.name),
        availableOptions,
    };
}

function toRequestedUnitTypes(requestedUnitTypes: Map<string, UnitRequest>): Record<string, number> {
    return Object.fromEntries([...requestedUnitTypes.entries()].map(([name, request]) => [name, request.priority]));
}

function isSamePosition(a: { x: number; y: number }, b: { x: number; y: number }): boolean {
    return a.x === b.x && a.y === b.y;
}

function roundDistance(distance: number | null): number | null {
    return distance === null ? null : Math.round(distance * 10) / 10;
}

function getDistanceBetweenPoints(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function getNearestDistance(origin: { x: number; y: number }, points: Array<{ x: number; y: number }>): number | null {
    return points.length > 0 ? roundDistance(Math.min(...points.map((point) => getDistanceBetweenPoints(origin, point)))) : null;
}

function classifyMapSize(tileArea: number, nearestDistance: number | null): string {
    if (tileArea <= 10000 || (nearestDistance !== null && nearestDistance <= 56)) {
        return "small";
    }
    if (tileArea <= 18000 || (nearestDistance !== null && nearestDistance <= 88)) {
        return "medium";
    }
    return "large";
}

function classifyRushDistance(nearestEnemyStartDistance: number | null, nearestStartingLocationDistance: number | null): string {
    const distance = nearestEnemyStartDistance ?? nearestStartingLocationDistance;
    if (distance === null) {
        return "unknown";
    }
    if (distance <= 56) {
        return "close";
    }
    if (distance <= 88) {
        return "medium";
    }
    return "far";
}

function classifyRushRisk(sizeClass: string, rushDistanceClass: string): string {
    if (sizeClass === "small" || rushDistanceClass === "close") {
        return "high";
    }
    if (sizeClass === "medium" || rushDistanceClass === "medium" || rushDistanceClass === "unknown") {
        return "medium";
    }
    return "low";
}

function toMapSummary(context: SupabotContext) {
    const playerData = context.game.getPlayerData(context.player.name);
    const { width, height } = context.game.mapApi.getRealMapSize();
    const tileArea = width * height;
    const startingLocations = context.game.mapApi
        .getStartingLocations()
        .filter((location) => !isSamePosition(location, playerData.startLocation));
    const enemyStartLocations = context.game
        .getPlayers()
        .filter((playerName) => playerName !== playerData.name && !context.game.areAlliedPlayers(playerData.name, playerName))
        .map((playerName) => context.game.getPlayerData(playerName).startLocation)
        .filter((location) => !isSamePosition(location, playerData.startLocation));
    const nearestStartingLocationDistance = getNearestDistance(playerData.startLocation, startingLocations);
    const nearestEnemyStartDistance = getNearestDistance(playerData.startLocation, enemyStartLocations);
    const sizeClass = classifyMapSize(tileArea, nearestEnemyStartDistance ?? nearestStartingLocationDistance);
    const rushDistanceClass = classifyRushDistance(nearestEnemyStartDistance, nearestStartingLocationDistance);
    const rushRisk = classifyRushRisk(sizeClass, rushDistanceClass);
    return {
        width,
        height,
        tileArea,
        startingLocationCount: context.game.mapApi.getStartingLocations().length,
        nearestStartingLocationDistance,
        nearestEnemyStartDistance,
        sizeClass,
        rushDistanceClass,
        rushRisk,
        summary: `${sizeClass} map (${width}x${height}) with ${rushDistanceClass} rush distance and ${rushRisk} early rush risk.`,
    };
}

export async function buildProductionAdvisorInput(
    context: SupabotContext,
    threat: GlobalThreat | null,
    requestedUnitTypes: Map<string, UnitRequest>,
    runtimeState: CheckpointAdvisorRuntimeState,
    includeCheckpointFeatures: boolean,
): Promise<ProductionAdvisorInput> {
    const playerData = context.game.getPlayerData(context.player.name);
    const input: ProductionAdvisorInput = {
        tick: context.game.getCurrentTick(),
        attackMode: context.matchAwareness.shouldAttack(),
        credits: playerData.credits,
        harvesters: getHarvesterCount(context),
        ownCounts: getOwnedCounts(context, context.player.name, "self"),
        enemyCounts: getOwnedCounts(context, context.player.name, "enemy"),
        requestedUnitTypes: toRequestedUnitTypes(requestedUnitTypes),
        queues: QUEUES.map((queueType) => toQueueSnapshot(context, queueType)),
        threat: toThreatSummary(threat),
        map: toMapSummary(context),
    };

    if (includeCheckpointFeatures) {
        input.checkpointFeatures = await buildLivePolicyFeaturePayload(context, runtimeState);
    }

    return input;
}
