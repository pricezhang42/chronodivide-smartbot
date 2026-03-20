import {
    ActionsApi,
    GameApi,
    PlayerData,
    ProductionApi,
    QueueStatus,
    QueueType,
    TechnoRules,
} from "@chronodivide/game-api";
import { GlobalThreat } from "../threat/threat.js";
import { TechnoRulesWithPriority } from "./buildingRules.js";
import { SupabotContext } from "../common/context.js";
import { UnitRequest } from "../mission/missionController.js";
import type { ProductionAdvisor } from "./productionAdvisor.js";
import { NullProductionAdvisor } from "./nullProductionAdvisor.js";
import { buildProductionAdvisorInput } from "./productionAdvisorSerializer.js";
import type { CheckpointAdvisorRuntimeState, ProductionAdvisorOutput } from "./productionAdvisorTypes.js";

export const QUEUES = [
    QueueType.Structures,
    QueueType.Armory,
    QueueType.Infantry,
    QueueType.Vehicles,
    QueueType.Aircrafts,
    QueueType.Ships,
];

function isBuildingQueue(queueType: QueueType): boolean {
    return queueType === QueueType.Structures || queueType === QueueType.Armory;
}

export const queueTypeToName = (queue: QueueType) => {
    switch (queue) {
        case QueueType.Structures:
            return "Structures";
        case QueueType.Armory:
            return "Armory";
        case QueueType.Infantry:
            return "Infantry";
        case QueueType.Vehicles:
            return "Vehicles";
        case QueueType.Aircrafts:
            return "Aircrafts";
        case QueueType.Ships:
            return "Ships";
        default:
            return "Unknown";
    }
};

type QueueState = {
    queue: QueueType;
    /** sorted in ascending order (last item is the topItem) */
    items: TechnoRulesWithPriority[];
    topItem: TechnoRulesWithPriority | undefined;
};

const REPAIR_CHECK_INTERVAL = 30;
const PRODUCTION_ADVISOR_UPDATE_INTERVAL = 15;
const BUILD_ORDER_TRACE_LIMIT = 20;
const MISSING_INT = -1;

export class QueueController {
    private queueStates: QueueState[] = [];
    private lastRepairCheckAt = 0;
    private latestProductionAdvisorRecommendations: ProductionAdvisorOutput = null;
    private pendingProductionAdvisorRequest: Promise<void> | null = null;
    private lastProductionAdvisorRequestAt = Number.NEGATIVE_INFINITY;
    private runtimeState: CheckpointAdvisorRuntimeState = {
        lastActionTick: null,
        lastActionTypeNameV1: null,
        lastQueueValue: MISSING_INT,
        buildOrderActionTypeNamesV1: [],
    };

    constructor(private productionAdvisor: ProductionAdvisor = new NullProductionAdvisor()) {}

    public onAiUpdate(
        context: SupabotContext,
        threatCache: GlobalThreat | null,
        unitTypeRequests: Map<string, UnitRequest>,
        logger: (message: string) => void,
    ) {
        const { game, player } = context;
        const { production: productionApi, actions: actionsApi } = player;
        const playerData = game.getPlayerData(player.name);
        this.maybeRequestProductionAdvisorRecommendations(context, threatCache, unitTypeRequests, logger);
        const blendedUnitTypeRequests = this.blendAdvisorRecommendations(unitTypeRequests);
        this.queueStates = QUEUES.map((queueType) => {
            const options = productionApi.getAvailableObjects(queueType);
            const items = QueueController.getPrioritiesForBuildingOptions(options, blendedUnitTypeRequests);
            const topItem = items.length > 0 ? items[items.length - 1] : undefined;
            return {
                queue: queueType,
                items,
                // only if the top item has a  priority above zero
                topItem: topItem && topItem.priority > 0 ? topItem : undefined,
            };
        });
        const totalWeightAcrossQueues = this.queueStates
            .map((decision) => decision.topItem?.priority!)
            .reduce((pV, cV) => pV + cV, 0);
        const totalCostAcrossQueues = this.queueStates
            .map((decision) => decision.topItem?.unit.cost!)
            .reduce((pV, cV) => pV + cV, 0);

        this.queueStates.forEach((decision) => {
            this.updateBuildQueue(
                game,
                productionApi,
                actionsApi,
                playerData,
                threatCache,
                blendedUnitTypeRequests,
                decision.queue,
                decision.topItem,
                totalWeightAcrossQueues,
                totalCostAcrossQueues,
                logger,
            );
        });

        // Repair is simple - just repair everything that's damaged.
        if (playerData.credits > 0 && game.getCurrentTick() > this.lastRepairCheckAt + REPAIR_CHECK_INTERVAL) {
            game.getVisibleUnits(playerData.name, "self", (r) => r.repairable).forEach((unitId) => {
                const unit = game.getUnitData(unitId);
                if (!unit || !unit.hitPoints || !unit.maxHitPoints || unit.hasWrenchRepair) {
                    return;
                }
                if (unit.hitPoints < unit.maxHitPoints) {
                    actionsApi.toggleRepairWrench(unitId);
                }
            });
            this.lastRepairCheckAt = game.getCurrentTick();
        }
    }

    public hasPendingProductionAdvisorRequest(): boolean {
        return this.pendingProductionAdvisorRequest !== null;
    }

    public async waitForPendingProductionAdvisorRequest(): Promise<void> {
        await this.pendingProductionAdvisorRequest;
    }

    public async dispose(): Promise<void> {
        await this.productionAdvisor.dispose?.();
    }

    private maybeRequestProductionAdvisorRecommendations(
        context: SupabotContext,
        threatCache: GlobalThreat | null,
        unitTypeRequests: Map<string, UnitRequest>,
        logger: (message: string) => void,
    ): void {
        if (!this.productionAdvisor.enabled || this.pendingProductionAdvisorRequest !== null) {
            return;
        }
        const currentTick = context.game.getCurrentTick();
        if (currentTick < this.lastProductionAdvisorRequestAt + PRODUCTION_ADVISOR_UPDATE_INTERVAL) {
            return;
        }

        this.lastProductionAdvisorRequestAt = currentTick;
        const requestedUnitTypes = new Map(
            [...unitTypeRequests.entries()].map(([name, request]) => [name, { ...request }]),
        );
        const includeCheckpointFeatures = this.productionAdvisor.requiresLiveFeaturePayload;
        this.pendingProductionAdvisorRequest = buildProductionAdvisorInput(
            context,
            threatCache,
            requestedUnitTypes,
            this.runtimeState,
            includeCheckpointFeatures,
        )
            .then((input) => this.productionAdvisor.getRecommendations(input))
            .then((recommendations) => {
                this.latestProductionAdvisorRecommendations = recommendations;
            })
            .catch((error: unknown) => {
                logger(`Production advisor request failed: ${String(error)}`);
            })
            .finally(() => {
                this.pendingProductionAdvisorRequest = null;
            });
    }

    private blendAdvisorRecommendations(unitTypeRequests: Map<string, UnitRequest>): Map<string, UnitRequest> {
        const blended = new Map<string, UnitRequest>(
            [...unitTypeRequests.entries()].map(([name, request]) => [name, { ...request }]),
        );
        if (!this.latestProductionAdvisorRecommendations) {
            return blended;
        }

        Object.entries(this.latestProductionAdvisorRecommendations).forEach(([queueName, recommendations]) => {
            if (!recommendations) {
                return;
            }
            const isBuildingQueueName = queueName === "Structures" || queueName === "Armory";
            Object.entries(recommendations).forEach(([unitName, score]) => {
                if (!(score > 0)) {
                    return;
                }
                const existingRequest = blended.get(unitName);
                if (isBuildingQueueName && (!existingRequest || existingRequest.specificLocation === null)) {
                    return;
                }
                const nextPriority = Math.max(0, (existingRequest?.priority ?? 0) + score);
                if (nextPriority <= 0) {
                    return;
                }
                blended.set(unitName, {
                    priority: nextPriority,
                    specificLocation: existingRequest?.specificLocation ?? null,
                });
            });
        });

        return blended;
    }

    private recordProductionAction(tick: number, actionTypeNameV1: string): void {
        this.runtimeState.lastActionTick = tick;
        this.runtimeState.lastActionTypeNameV1 = actionTypeNameV1;
        this.runtimeState.lastQueueValue = MISSING_INT;
        if (
            actionTypeNameV1.startsWith("Queue::Add::") ||
            actionTypeNameV1.startsWith("PlaceBuilding::") ||
            actionTypeNameV1.startsWith("Order::Deploy::") ||
            actionTypeNameV1.startsWith("Order::DeploySelected::")
        ) {
            if (this.runtimeState.buildOrderActionTypeNamesV1.length < BUILD_ORDER_TRACE_LIMIT) {
                this.runtimeState.buildOrderActionTypeNamesV1.push(actionTypeNameV1);
            }
        }
    }

    private updateBuildQueue(
        game: GameApi,
        productionApi: ProductionApi,
        actionsApi: ActionsApi,
        playerData: PlayerData,
        threatCache: GlobalThreat | null,
        unitTypeRequests: Map<string, UnitRequest>,
        queueType: QueueType,
        decision: TechnoRulesWithPriority | undefined,
        totalWeightAcrossQueues: number,
        totalCostAcrossQueues: number,
        logger: (message: string) => void,
    ): void {
        const myCredits = playerData.credits;

        const queueData = productionApi.getQueueData(queueType);
        if (queueData.status == QueueStatus.Idle) {
            // Start building the decided item.
            if (decision !== undefined) {
                logger(`Decision (${queueTypeToName(queueType)}): ${decision.unit.name}`);
                actionsApi.queueForProduction(queueType, decision.unit.name, decision.unit.type, 1);
                this.recordProductionAction(game.getCurrentTick(), `Queue::Add::${decision.unit.name}`);
            }
        } else if (queueData.status == QueueStatus.Ready && queueData.items.length > 0) {
            if (isBuildingQueue(queueType)) {
                const readyUnit = queueData.items[0].rules;
                const currentRequest = unitTypeRequests.get(readyUnit.name);
                if (!currentRequest) {
                    // No one is requesting this anymore, cancel
                    logger(`Cancelling ready ${readyUnit.name} because no one is requesting anymore`);
                    actionsApi.unqueueFromProduction(queueType, readyUnit.name, readyUnit.type, 1);
                    this.recordProductionAction(game.getCurrentTick(), `Queue::Cancel::${readyUnit.name}`);
                    return;
                }
                if (!currentRequest.specificLocation) {
                    // No one is requesting this anymore, cancel
                    logger(`Cancelling ready ${readyUnit.name} because location is unspecified`);
                    actionsApi.unqueueFromProduction(queueType, readyUnit.name, readyUnit.type, 1);
                    this.recordProductionAction(game.getCurrentTick(), `Queue::Cancel::${readyUnit.name}`);
                    return;
                }
                actionsApi.placeBuilding(
                    readyUnit.name,
                    currentRequest.specificLocation.x,
                    currentRequest.specificLocation.y,
                );
                this.recordProductionAction(game.getCurrentTick(), `PlaceBuilding::${readyUnit.name}`);
            }
        } else if (queueData.status == QueueStatus.Active && queueData.items.length > 0 && decision != null) {
            // Consider cancelling if something else is significantly higher priority than what is currently being produced.

            const currentProduction = queueData.items[0].rules;
            if (decision.unit != currentProduction) {
                // Changing our mind.
                const currentRequest = unitTypeRequests.get(currentProduction.name);
                const currentItemPriority = currentRequest ? currentRequest.priority : 0;
                const newItemPriority = decision.priority;
                if (newItemPriority > currentItemPriority * 2) {
                    logger(
                        `Dequeueing queue ${queueTypeToName(queueData.type)} unit ${currentProduction.name} because ${
                            decision.unit.name
                        } has 2x higher priority.`,
                    );
                    actionsApi.unqueueFromProduction(queueData.type, currentProduction.name, currentProduction.type, 1);
                    this.recordProductionAction(game.getCurrentTick(), `Queue::Cancel::${currentProduction.name}`);
                }
            } else {
                // Not changing our mind, but maybe other queues are more important for now.
                if (totalCostAcrossQueues > myCredits && decision.priority < totalWeightAcrossQueues * 0.25) {
                    logger(
                        `Pausing queue ${queueTypeToName(queueData.type)} because weight is low (${
                            decision.priority
                        }/${totalWeightAcrossQueues})`,
                    );
                    actionsApi.pauseProduction(queueData.type);
                    this.recordProductionAction(game.getCurrentTick(), `Queue::Hold::${queueTypeToName(queueData.type)}`);
                }
            }
        } else if (queueData.status == QueueStatus.OnHold) {
            // Consider resuming queue if priority is high relative to other queues.
            if (myCredits >= totalCostAcrossQueues) {
                logger(`Resuming queue ${queueTypeToName(queueData.type)} because credits are high`);
                actionsApi.resumeProduction(queueData.type);
                this.recordProductionAction(game.getCurrentTick(), `Queue::Resume::${queueTypeToName(queueData.type)}`);
            } else if (decision && decision.priority >= totalWeightAcrossQueues * 0.25) {
                logger(
                    `Resuming queue ${queueTypeToName(queueData.type)} because weight is high (${
                        decision.priority
                    }/${totalWeightAcrossQueues})`,
                );
                actionsApi.resumeProduction(queueData.type);
                this.recordProductionAction(game.getCurrentTick(), `Queue::Resume::${queueTypeToName(queueData.type)}`);
            }
        }
    }

    private static getPrioritiesForBuildingOptions(
        options: TechnoRules[],
        unitTypeRequests: Map<string, UnitRequest>,
    ): TechnoRulesWithPriority[] {
        let priorityQueue: TechnoRulesWithPriority[] = [];
        options.forEach((option) => {
            const priority = unitTypeRequests.get(option.name)?.priority ?? 0;
            if (priority > 0) {
                priorityQueue.push({ unit: option, priority });
            }
        });

        priorityQueue = priorityQueue.sort((a, b) => a.priority - b.priority);
        return priorityQueue;
    }

    public getGlobalDebugText(gameApi: GameApi, productionApi: ProductionApi) {
        const productionState = QUEUES.reduce((prev, queueType) => {
            if (productionApi.getQueueData(queueType).size === 0) {
                return prev;
            }
            const paused = productionApi.getQueueData(queueType).status === QueueStatus.OnHold;
            return (
                prev +
                " [" +
                queueTypeToName(queueType) +
                (paused ? " PAUSED" : "") +
                ": " +
                productionApi
                    .getQueueData(queueType)
                    .items.map((item) => item.rules.name + (item.quantity > 1 ? "x" + item.quantity : "")) +
                "]"
            );
        }, "");

        const queueStates = this.queueStates
            .filter((queueState) => queueState.items.length > 0)
            .map((queueState) => {
                const queueString = queueState.items
                    .map((item) => item.unit.name + "(" + Math.round(item.priority * 10) / 10 + ")")
                    .join(", ");
                return `${queueTypeToName(queueState.queue)} Prios: ${queueString}\n`;
            })
            .join("");

        return `Production: ${productionState}\n${queueStates}`;
    }
}
