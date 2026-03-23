import {
    ApiEvent,
    Bot,
    GameApi,
    ObjectType,
    OrderType,
    QueueType,
    SuperWeaponType,
    TechnoRules,
    UnitData,
} from "@chronodivide/game-api";

import { formatTimeDuration } from "./logic/common/utils.js";
import { CheckpointPolicyClient } from "./logic/modelcontrol/checkpointPolicyClient.js";
import { buildLivePolicyFeaturePayload } from "./logic/modelcontrol/liveFeaturePayloadBuilder.js";
import type {
    AdvisorQueueName,
    LivePolicyFamily,
    LivePolicyOutput,
    LivePolicyRuntimeState,
} from "./logic/modelcontrol/livePolicyTypes.js";
import { QUEUE_TYPE_TO_ADVISOR_QUEUE, createInitialLivePolicyRuntimeState } from "./logic/modelcontrol/livePolicyTypes.js";
import { BuildOrderCurriculum } from "./logic/modelcontrol/buildOrderCurriculum.js";

const NATURAL_TICK_RATE = 15;
const POLICY_UPDATE_INTERVAL_TICKS = 15;
const BUILD_ORDER_TRACE_LIMIT = 20;
const MISSING_INT = -1;
const ORDER_UNIT_COOLDOWN_TICKS = 30;
const DUPLICATE_QUEUE_ACTION_COOLDOWN_TICKS = 45;
const NO_TARGET_ORDER_TYPE_NAMES = new Set(["Deploy", "DeploySelected", "Stop", "Cheer"]);
const TILE_TARGET_ORDER_TYPE_NAMES = new Set(["Move", "ForceMove", "AttackMove", "GuardArea", "Scatter", "PlaceBomb"]);
const OBJECT_TARGET_ORDER_TYPE_NAMES = new Set([
    "Attack",
    "ForceAttack",
    "Capture",
    "Occupy",
    "Dock",
    "Repair",
    "EnterTransport",
]);
const ORE_TILE_TARGET_ORDER_TYPE_NAMES = new Set(["Gather"]);

const ADVISOR_QUEUE_NAME_TO_QUEUE_TYPE: Record<AdvisorQueueName, QueueType> = {
    Structures: QueueType.Structures,
    Armory: QueueType.Armory,
    Infantry: QueueType.Infantry,
    Vehicles: QueueType.Vehicles,
    Aircrafts: QueueType.Aircrafts,
    Ships: QueueType.Ships,
};

const BUILDING_QUEUE_TYPES = new Set<QueueType>([QueueType.Structures, QueueType.Armory]);

const FAMILY_COOLDOWN_TICKS: Partial<Record<LivePolicyFamily, number>> = {
    Queue: 45,
    PlaceBuilding: 60,
    ActivateSuperWeapon: 180,
    SellObject: 30,
    ToggleRepair: 30,
};

type PendingPolicyAction = {
    requestedAtTick: number;
    output: LivePolicyOutput;
};

export type LiveControlBotStats = {
    predictedFamilyCounts: Record<string, number>;
    executedFamilyCounts: Record<string, number>;
    noopReasonCounts: Record<string, number>;
    predictedOrderTypeCounts: Record<string, number>;
    executedOrderTypeCounts: Record<string, number>;
    predictedQueueUpdateCounts: Record<string, number>;
    executedQueueUpdateCounts: Record<string, number>;
    predictedQueueObjectCounts: Record<string, number>;
    executedQueueObjectCounts: Record<string, number>;
    predictedPlaceBuildingCounts: Record<string, number>;
    executedPlaceBuildingCounts: Record<string, number>;
    lastFailedActionReason: string | null;
    lastPredictedFamily: LivePolicyFamily | null;
    lastPredictedScore: number | null;
};

function orderTypeFromName(value: string | null): OrderType | null {
    if (!value) {
        return null;
    }
    const candidate = (OrderType as unknown as Record<string, unknown>)[value];
    return typeof candidate === "number" ? (candidate as OrderType) : null;
}

function superWeaponTypeFromName(value: string | null): SuperWeaponType | null {
    if (!value) {
        return null;
    }
    const candidate = (SuperWeaponType as unknown as Record<string, unknown>)[value];
    return typeof candidate === "number" ? (candidate as SuperWeaponType) : null;
}

export class CheckpointControlBot extends Bot {
    private pendingPolicyRequest: Promise<void> | null = null;
    private queuedPolicyAction: PendingPolicyAction | null = null;
    private lastPolicyRequestTick = Number.NEGATIVE_INFINITY;
    private runtimeState: LivePolicyRuntimeState = createInitialLivePolicyRuntimeState();
    private readonly policyClient: CheckpointPolicyClient;
    private readonly predictedFamilyCounts: Record<string, number> = {};
    private readonly executedFamilyCounts: Record<string, number> = {};
    private readonly noopReasonCounts: Record<string, number> = {};
    private readonly predictedOrderTypeCounts: Record<string, number> = {};
    private readonly executedOrderTypeCounts: Record<string, number> = {};
    private readonly predictedQueueUpdateCounts: Record<string, number> = {};
    private readonly executedQueueUpdateCounts: Record<string, number> = {};
    private readonly predictedQueueObjectCounts: Record<string, number> = {};
    private readonly executedQueueObjectCounts: Record<string, number> = {};
    private readonly predictedPlaceBuildingCounts: Record<string, number> = {};
    private readonly executedPlaceBuildingCounts: Record<string, number> = {};
    private readonly lastExecutedFamilyTick: Partial<Record<LivePolicyFamily, number>> = {};
    private readonly lastUnitCommandTickById = new Map<number, number>();
    private readonly lastUnitOrderById = new Map<number, { orderType: string; tick: number }>();
    private lastPredictedFamily: LivePolicyFamily | null = null;
    private lastPredictedScore: number | null = null;
    private lastQueueActionSignature: string | null = null;
    private lastQueueActionTick = Number.NEGATIVE_INFINITY;
    private curriculum: BuildOrderCurriculum | null = null;

    constructor(
        name: string,
        country: string,
        private readonly tryAllyWith: string[] = [],
        private readonly enableLogging = true,
        checkpointPath: string = process.env.SL_CHECKPOINT_PATH ?? "",
        pythonExecutable: string = process.env.SL_PYTHON_EXECUTABLE ?? "python",
    ) {
        super(name, country);
        this.policyClient = new CheckpointPolicyClient(checkpointPath, pythonExecutable);
    }

    override onGameStart(game: GameApi) {
        this.tryAllyWith
            .filter((playerName) => playerName !== this.name)
            .forEach((playerName) => this.player.actions.toggleAlliance(playerName, true));

        const playerData = game.getPlayerData(this.name);
        const sideId =
            (playerData.country as any)?.side ?? (playerData.country as any)?.rules?.side ?? 1;
        this.curriculum = new BuildOrderCurriculum(Number(sideId));
        this.logBotStatus(`CheckpointControlBot started (side=${sideId}).`);
    }

    override onGameTick(_game: GameApi) {
        const currentTick = this.game.getCurrentTick();
        this.maybeExecuteQueuedPolicyAction();
        this.maybeRequestPolicyAction(currentTick);
    }

    override onGameEvent(_ev: ApiEvent) {}

    public hasPendingPolicyRequest(): boolean {
        return this.pendingPolicyRequest !== null;
    }

    public hasPendingProductionAdvisorRequest(): boolean {
        return this.hasPendingPolicyRequest();
    }

    public async waitForPendingPolicyRequest(): Promise<void> {
        await this.pendingPolicyRequest;
    }

    public async waitForPendingProductionAdvisorRequest(): Promise<void> {
        await this.waitForPendingPolicyRequest();
    }

    public async dispose(): Promise<void> {
        await this.policyClient.dispose();
    }

    public getLiveControlStats(): LiveControlBotStats {
        return {
            predictedFamilyCounts: { ...this.predictedFamilyCounts },
            executedFamilyCounts: { ...this.executedFamilyCounts },
            noopReasonCounts: { ...this.noopReasonCounts },
            predictedOrderTypeCounts: { ...this.predictedOrderTypeCounts },
            executedOrderTypeCounts: { ...this.executedOrderTypeCounts },
            predictedQueueUpdateCounts: { ...this.predictedQueueUpdateCounts },
            executedQueueUpdateCounts: { ...this.executedQueueUpdateCounts },
            predictedQueueObjectCounts: { ...this.predictedQueueObjectCounts },
            executedQueueObjectCounts: { ...this.executedQueueObjectCounts },
            predictedPlaceBuildingCounts: { ...this.predictedPlaceBuildingCounts },
            executedPlaceBuildingCounts: { ...this.executedPlaceBuildingCounts },
            lastFailedActionReason: this.runtimeState.lastFailedActionReason ?? null,
            lastPredictedFamily: this.lastPredictedFamily,
            lastPredictedScore: this.lastPredictedScore,
        };
    }

    private maybeRequestPolicyAction(currentTick: number): void {
        if (this.pendingPolicyRequest !== null) {
            return;
        }
        if (currentTick < this.lastPolicyRequestTick + POLICY_UPDATE_INTERVAL_TICKS) {
            return;
        }

        // Curriculum phase: execute scripted build order before handing off to ML.
        if (this.curriculum && !this.curriculum.isComplete(currentTick)) {
            this.lastPolicyRequestTick = currentTick;
            const playerData = this.game.getPlayerData(this.name);
            const action = this.curriculum.tryExecuteNextStep(this.game, this.player, playerData);
            if (action.executed) {
                this.recordCurriculumAction(action.family, action.description);
            }
            return;
        }

        this.lastPolicyRequestTick = currentTick;
        this.syncPerUnitOrderState(currentTick);
        this.pendingPolicyRequest = buildLivePolicyFeaturePayload(this.context, this.runtimeState)
            .then((featurePayload) => this.policyClient.getAction({ featurePayload }))
            .then((output) => {
                this.queuedPolicyAction = {
                    requestedAtTick: currentTick,
                    output,
                };
            })
            .catch((error: unknown) => {
                this.runtimeState.lastFailedActionReason = String(error);
                this.logBotStatus(`Policy request failed: ${String(error)}`);
            })
            .finally(() => {
                this.pendingPolicyRequest = null;
            });
    }

    private maybeExecuteQueuedPolicyAction(): void {
        if (!this.queuedPolicyAction) {
            return;
        }
        const { output } = this.queuedPolicyAction;
        this.queuedPolicyAction = null;
        this.executePolicyAction(output);
    }

    private executePolicyAction(output: LivePolicyOutput): void {
        this.incrementCount(this.predictedFamilyCounts, output.family);
        this.recordPredictedSubtypes(output);
        this.lastPredictedFamily = output.family;
        this.lastPredictedScore = output.score;

        const familyCooldown = this.getFamilyCooldownTicks(output.family);
        const currentTick = this.game.getCurrentTick();
        const lastExecutedTick = this.lastExecutedFamilyTick[output.family];
        if (
            familyCooldown > 0 &&
            lastExecutedTick !== undefined &&
            currentTick < lastExecutedTick + familyCooldown
        ) {
            this.recordNoop(`family_cooldown:${output.family}`);
            return;
        }

        switch (output.family) {
            case "Order":
                this.executeOrder(output);
                break;
            case "Queue":
                this.executeQueue(output);
                break;
            case "PlaceBuilding":
                this.executePlaceBuilding(output);
                break;
            case "ActivateSuperWeapon":
                this.executeSuperWeapon(output);
                break;
            case "SellObject":
                this.executeSellObject(output);
                break;
            case "ToggleRepair":
                this.executeToggleRepair(output);
                break;
            case "ResignGame":
                this.recordPolicyAction(output, "ResignGame", []);
                this.player.actions.quitGame();
                break;
            case "Noop":
            default:
                this.recordNoop(`family=${output.family}`);
                break;
        }
    }

    private executeOrder(output: LivePolicyOutput): void {
        const order = output.order;
        if (!order || !order.orderType || order.unitIds.length === 0) {
            this.recordNoop("invalid_order_payload");
            return;
        }
        const orderTypeName = order.orderType;
        if (!this.isSupportedLiveOrderTargetMode(orderTypeName, order.targetMode)) {
            this.recordNoop(`unsupported_order_target_mode:${orderTypeName}:${order.targetMode ?? "null"}`);
            return;
        }

        const orderType = orderTypeFromName(orderTypeName);
        if (orderType === null) {
            this.recordNoop(`unsupported_order_type:${orderTypeName}`);
            return;
        }

        const ownedUnits = order.unitIds
            .map((unitId) => this.game.getUnitData(unitId))
            .filter((unit): unit is UnitData => !!unit && unit.owner === this.name);
        const controlledUnits = ownedUnits.filter((unit) => this.isUnitCapableOfLiveOrder(unit, orderTypeName));
        const currentTick = this.game.getCurrentTick();
        const cooledUnits = controlledUnits.filter((unit) => {
            const lastTick = this.lastUnitCommandTickById.get(unit.id);
            return lastTick === undefined || currentTick >= lastTick + ORDER_UNIT_COOLDOWN_TICKS;
        });
        const ownedUnitIds = cooledUnits.map((unit) => unit.id);
        if (ownedUnitIds.length === 0) {
            const hadControlledUnits = controlledUnits.length > 0;
            this.recordNoop(hadControlledUnits ? `order_unit_cooldown:${orderTypeName}` : `order_units_unresolved:${orderTypeName}`);
            return;
        }

        try {
            if (order.targetMode === "object") {
                if (order.targetEntityId === null) {
                    this.recordNoop("order_missing_target_entity");
                    return;
                }
                const targetUnit = this.game.getUnitData(order.targetEntityId);
                if (!targetUnit) {
                    this.recordNoop(`order_target_entity_unresolved:${orderTypeName}`);
                    return;
                }
                this.player.actions.orderUnits(ownedUnitIds, orderType, order.targetEntityId);
            } else if (order.targetMode === "tile" || order.targetMode === "ore_tile") {
                if (!order.targetLocation) {
                    this.recordNoop("order_missing_target_location");
                    return;
                }
                if (!this.isTileWithinMap(order.targetLocation.rx, order.targetLocation.ry)) {
                    this.recordNoop(`order_target_tile_out_of_bounds:${orderTypeName}`);
                    return;
                }
                this.player.actions.orderUnits(
                    ownedUnitIds,
                    orderType,
                    order.targetLocation.rx,
                    order.targetLocation.ry,
                );
            } else {
                this.player.actions.orderUnits(ownedUnitIds, orderType);
            }
        } catch (error) {
            this.recordNoop(`order_dispatch_failed:${orderTypeName}:${String(error)}`);
            return;
        }

        this.recordPolicyAction(output, this.buildLegacyActionName(output), ownedUnitIds);
        if (order.queueFlag) {
            this.logBotStatus("Order queueFlag predicted but live API has no queue flag parameter.");
        }
    }

    private executeQueue(output: LivePolicyOutput): void {
        const queue = output.queue;
        if (!queue || !queue.updateType || !queue.objectName) {
            this.recordNoop("invalid_queue_payload");
            return;
        }

        const queueType = this.resolveQueueType(queue.queue, queue.objectName);
        if (queueType === null) {
            this.recordNoop(`queue_type_unresolved:${queue.objectName}`);
            return;
        }

        const rules = this.findTechnoRules(queueType, queue.objectName);
        if (!rules) {
            this.recordNoop(`queue_rules_unresolved:${queue.objectName}`);
            return;
        }

        const quantity = Math.max(1, queue.quantity ?? 1);
        const queueActionSignature = `${queue.updateType}:${queueType}:${rules.name}:${quantity}`;
        const currentTick = this.game.getCurrentTick();
        if (
            this.lastQueueActionSignature === queueActionSignature &&
            currentTick < this.lastQueueActionTick + DUPLICATE_QUEUE_ACTION_COOLDOWN_TICKS
        ) {
            this.recordNoop(`queue_repeat_cooldown:${queue.updateType}:${queue.objectName}`);
            return;
        }
        if (queue.updateType === "Add" || queue.updateType === "AddNext") {
            this.player.actions.queueForProduction(queueType, rules.name, rules.type, quantity);
            this.recordPendingBuildingQueue(queueType, queue.objectName);
        } else if (queue.updateType === "Cancel") {
            this.player.actions.unqueueFromProduction(queueType, rules.name, rules.type, quantity);
            this.clearPendingBuildingQueue(queueType, queue.objectName);
        } else {
            this.recordNoop(`unsupported_queue_update:${queue.updateType}`);
            return;
        }

        this.lastQueueActionSignature = queueActionSignature;
        this.lastQueueActionTick = currentTick;
        this.recordPolicyAction(output, this.buildLegacyActionName(output), []);
    }

    private executePlaceBuilding(output: LivePolicyOutput): void {
        const placeBuilding = output.placeBuilding;
        if (!placeBuilding?.buildingName || !placeBuilding.targetLocation) {
            this.recordNoop("invalid_place_building_payload");
            return;
        }

        const tile = {
            rx: placeBuilding.targetLocation.rx,
            ry: placeBuilding.targetLocation.ry,
        } as any;
        if (!this.player.canPlaceBuilding(placeBuilding.buildingName, tile)) {
            this.recordNoop(`illegal_building_location:${placeBuilding.buildingName}`);
            return;
        }

        this.player.actions.placeBuilding(
            placeBuilding.buildingName,
            placeBuilding.targetLocation.rx,
            placeBuilding.targetLocation.ry,
        );
        this.clearPendingBuildingQueueByObjectName(placeBuilding.buildingName);
        this.recordPolicyAction(output, this.buildLegacyActionName(output), []);
    }

    private executeSuperWeapon(output: LivePolicyOutput): void {
        const superWeapon = output.superWeapon;
        if (!superWeapon?.typeName || !superWeapon.targetLocation) {
            this.recordNoop("invalid_superweapon_payload");
            return;
        }

        const superWeaponType = superWeaponTypeFromName(superWeapon.typeName);
        if (superWeaponType === null) {
            this.recordNoop(`unsupported_superweapon:${superWeapon.typeName}`);
            return;
        }

        this.player.actions.activateSuperWeapon(
            superWeaponType,
            {
                rx: superWeapon.targetLocation.rx,
                ry: superWeapon.targetLocation.ry,
            },
            superWeapon.targetLocation2
                ? {
                      rx: superWeapon.targetLocation2.rx,
                      ry: superWeapon.targetLocation2.ry,
                  }
                : undefined,
        );
        this.recordPolicyAction(output, this.buildLegacyActionName(output), []);
    }

    private executeSellObject(output: LivePolicyOutput): void {
        const targetEntityId = output.targetEntityId;
        if (targetEntityId === null) {
            this.recordNoop("sell_missing_target");
            return;
        }
        this.player.actions.sellObject(targetEntityId);
        this.recordPolicyAction(output, "SellObject", []);
    }

    private executeToggleRepair(output: LivePolicyOutput): void {
        const targetEntityId = output.targetEntityId;
        if (targetEntityId === null) {
            this.recordNoop("repair_missing_target");
            return;
        }
        this.player.actions.toggleRepairWrench(targetEntityId);
        this.recordPolicyAction(output, "ToggleRepair", []);
    }

    private resolveQueueType(preferredQueueName: AdvisorQueueName | null, objectName: string): QueueType | null {
        if (preferredQueueName) {
            return ADVISOR_QUEUE_NAME_TO_QUEUE_TYPE[preferredQueueName];
        }
        const queueTypes = Object.values(ADVISOR_QUEUE_NAME_TO_QUEUE_TYPE);
        for (const queueType of queueTypes) {
            const rules = this.findTechnoRules(queueType, objectName);
            if (rules) {
                return queueType;
            }
        }
        const resolvedRules = this.findGlobalTechnoRules(objectName);
        if (resolvedRules) {
            return this.player.production.getQueueTypeForObject(resolvedRules);
        }
        return null;
    }

    private findTechnoRules(queueType: QueueType, objectName: string): TechnoRules | null {
        const availableRules = this.player.production.getAvailableObjects(queueType);
        const availableMatch = availableRules.find((rules) => rules.name === objectName);
        if (availableMatch) {
            return availableMatch;
        }

        const queueData = this.player.production.getQueueData(queueType);
        const queuedMatch = queueData.items.find((item) => item.rules.name === objectName)?.rules;
        if (queuedMatch) {
            return queuedMatch;
        }

        const resolvedRules = this.findGlobalTechnoRules(objectName);
        if (resolvedRules && this.player.production.getQueueTypeForObject(resolvedRules) === queueType) {
            return resolvedRules;
        }

        return null;
    }

    private findGlobalTechnoRules(objectName: string): TechnoRules | null {
        return (
            this.game.rules.buildingRules.get(objectName) ??
            this.game.rules.infantryRules.get(objectName) ??
            this.game.rules.vehicleRules.get(objectName) ??
            this.game.rules.aircraftRules.get(objectName) ??
            null
        );
    }

    private buildLegacyActionName(output: LivePolicyOutput): string | null {
        switch (output.family) {
            case "Order":
                if (!output.order?.orderType || !output.order.targetMode) {
                    return null;
                }
                return `Order::${output.order.orderType}::${output.order.targetMode}`;
            case "Queue":
                if (!output.queue?.updateType || !output.queue.objectName) {
                    return null;
                }
                return `Queue::${output.queue.updateType}::${output.queue.objectName}`;
            case "PlaceBuilding":
                if (!output.placeBuilding?.buildingName) {
                    return null;
                }
                return `PlaceBuilding::${output.placeBuilding.buildingName}`;
            case "ActivateSuperWeapon":
                if (!output.superWeapon?.typeName) {
                    return null;
                }
                return `ActivateSuperWeapon::${output.superWeapon.typeName}`;
            case "SellObject":
                return "SellObject";
            case "ToggleRepair":
                return "ToggleRepair";
            case "ResignGame":
                return "ResignGame";
            default:
                return null;
        }
    }

    private recordPolicyAction(
        output: LivePolicyOutput,
        legacyActionName: string | null,
        selectedUnitIds: number[],
    ): void {
        const tick = this.game.getCurrentTick();
        const family = output.family;
        this.runtimeState.lastActionTick = tick;
        this.runtimeState.lastActionTypeNameV1 = legacyActionName;
        this.runtimeState.lastQueueValue = MISSING_INT;
        this.runtimeState.currentSelectedObjectIds = selectedUnitIds.slice(0, 64);
        this.runtimeState.lastFailedActionReason = null;

        const recentFamilies = [...(this.runtimeState.recentActionFamilyNamesV2 ?? []), family];
        this.runtimeState.recentActionFamilyNamesV2 = recentFamilies.slice(-16);
        this.incrementCount(this.executedFamilyCounts, family);
        this.recordExecutedSubtypes(output);
        this.lastExecutedFamilyTick[family] = tick;
        if (family === "Order" && legacyActionName?.startsWith("Order::")) {
            const parts = legacyActionName.split("::");
            const orderType = parts[1] ?? "Unknown";
            const recentOrders = [...(this.runtimeState.recentOrderTypeNamesV2 ?? []), orderType];
            this.runtimeState.recentOrderTypeNamesV2 = recentOrders.slice(-16);
            selectedUnitIds.forEach((unitId) => {
                this.lastUnitCommandTickById.set(unitId, tick);
                this.lastUnitOrderById.set(unitId, { orderType, tick });
            });
        }

        if (
            legacyActionName &&
            (legacyActionName.startsWith("Queue::Add::") ||
                legacyActionName.startsWith("PlaceBuilding::") ||
                legacyActionName.startsWith("Order::Deploy::") ||
                legacyActionName.startsWith("Order::DeploySelected::"))
        ) {
            const trace = this.runtimeState.buildOrderActionTypeNamesV1.slice(-BUILD_ORDER_TRACE_LIMIT + 1);
            trace.push(legacyActionName);
            this.runtimeState.buildOrderActionTypeNamesV1 = trace;
        }
    }

    private recordNoop(reason: string): void {
        this.runtimeState.lastFailedActionReason = reason;
        this.incrementCount(this.noopReasonCounts, reason);
        this.logBotStatus(`Noop: ${reason}`);
    }

    private syncPerUnitOrderState(currentTick: number): void {
        const aliveIds = new Set(this.player.getVisibleUnits("self"));
        // Clean dead units from tracking maps.
        for (const unitId of this.lastUnitOrderById.keys()) {
            if (!aliveIds.has(unitId)) {
                this.lastUnitOrderById.delete(unitId);
                this.lastUnitCommandTickById.delete(unitId);
            }
        }
        // Build snapshot for runtimeState.
        const snapshot: Record<number, { orderType: string; ticksSinceOrder: number }> = {};
        for (const [unitId, entry] of this.lastUnitOrderById) {
            snapshot[unitId] = {
                orderType: entry.orderType,
                ticksSinceOrder: Math.max(0, currentTick - entry.tick),
            };
        }
        this.runtimeState.lastOrderByUnitId = snapshot;
    }

    private recordCurriculumAction(family: string, description: string): void {
        const tick = this.game.getCurrentTick();
        this.runtimeState.lastActionTick = tick;
        this.runtimeState.lastActionTypeNameV1 = description;
        this.runtimeState.lastFailedActionReason = null;

        const recentFamilies = [...(this.runtimeState.recentActionFamilyNamesV2 ?? []), family];
        this.runtimeState.recentActionFamilyNamesV2 = recentFamilies.slice(-16);
        this.incrementCount(this.executedFamilyCounts, family);

        if (
            description.startsWith("curriculum:queue:") ||
            description.startsWith("curriculum:place:") ||
            description.startsWith("curriculum:deploy")
        ) {
            const trace = this.runtimeState.buildOrderActionTypeNamesV1.slice(-BUILD_ORDER_TRACE_LIMIT + 1);
            trace.push(description);
            this.runtimeState.buildOrderActionTypeNamesV1 = trace;
        }
        this.logBotStatus(`Curriculum: ${description}`);
    }

    private recordPendingBuildingQueue(queueType: QueueType, objectName: string): void {
        if (!BUILDING_QUEUE_TYPES.has(queueType)) {
            return;
        }
        const queueName = QUEUE_TYPE_TO_ADVISOR_QUEUE[queueType];
        if (!queueName) {
            return;
        }
        const pending = { ...(this.runtimeState.pendingBuildingQueueByQueueName ?? {}) };
        pending[queueName] = objectName;
        this.runtimeState.pendingBuildingQueueByQueueName = pending;
    }

    private clearPendingBuildingQueue(queueType: QueueType, objectName: string | null): void {
        if (!BUILDING_QUEUE_TYPES.has(queueType)) {
            return;
        }
        const queueName = QUEUE_TYPE_TO_ADVISOR_QUEUE[queueType];
        if (!queueName) {
            return;
        }
        const pending = { ...(this.runtimeState.pendingBuildingQueueByQueueName ?? {}) };
        const currentObjectName = pending[queueName];
        if (!objectName || !currentObjectName || currentObjectName === objectName) {
            delete pending[queueName];
            this.runtimeState.pendingBuildingQueueByQueueName = pending;
        }
    }

    private clearPendingBuildingQueueByObjectName(objectName: string): void {
        const pendingEntries = Object.entries(this.runtimeState.pendingBuildingQueueByQueueName ?? {});
        if (pendingEntries.length === 0) {
            return;
        }
        const pending = { ...(this.runtimeState.pendingBuildingQueueByQueueName ?? {}) };
        for (const [queueName, pendingObjectName] of pendingEntries) {
            if (pendingObjectName === objectName) {
                delete pending[queueName];
            }
        }
        this.runtimeState.pendingBuildingQueueByQueueName = pending;
    }

    private getFamilyCooldownTicks(family: LivePolicyFamily): number {
        return FAMILY_COOLDOWN_TICKS[family] ?? 0;
    }

    private recordPredictedSubtypes(output: LivePolicyOutput): void {
        if (output.family === "Order" && output.order?.orderType) {
            this.incrementCount(this.predictedOrderTypeCounts, output.order.orderType);
            return;
        }
        if (output.family === "Queue" && output.queue) {
            if (output.queue.updateType) {
                this.incrementCount(this.predictedQueueUpdateCounts, output.queue.updateType);
            }
            if (output.queue.objectName) {
                this.incrementCount(this.predictedQueueObjectCounts, output.queue.objectName);
            }
            return;
        }
        if (output.family === "PlaceBuilding" && output.placeBuilding?.buildingName) {
            this.incrementCount(this.predictedPlaceBuildingCounts, output.placeBuilding.buildingName);
        }
    }

    private recordExecutedSubtypes(output: LivePolicyOutput): void {
        if (output.family === "Order" && output.order?.orderType) {
            this.incrementCount(this.executedOrderTypeCounts, output.order.orderType);
            return;
        }
        if (output.family === "Queue" && output.queue) {
            if (output.queue.updateType) {
                this.incrementCount(this.executedQueueUpdateCounts, output.queue.updateType);
            }
            if (output.queue.objectName) {
                this.incrementCount(this.executedQueueObjectCounts, output.queue.objectName);
            }
            return;
        }
        if (output.family === "PlaceBuilding" && output.placeBuilding?.buildingName) {
            this.incrementCount(this.executedPlaceBuildingCounts, output.placeBuilding.buildingName);
        }
    }

    private isSupportedLiveOrderTargetMode(orderTypeName: string, targetMode: string | null): boolean {
        if (NO_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            return targetMode === "none";
        }
        if (TILE_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            return targetMode === "tile";
        }
        if (ORE_TILE_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            return targetMode === "ore_tile" || targetMode === "tile";
        }
        if (OBJECT_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            return targetMode === "object";
        }
        return false;
    }

    private isUnitCapableOfLiveOrder(unit: UnitData, orderTypeName: string): boolean {
        if (NO_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            if (orderTypeName === "Deploy" || orderTypeName === "DeploySelected") {
                return Boolean(unit.rules.deploysInto || unit.rules.constructionYard);
            }
            return unit.canMove === true || Boolean(unit.rules.deploysInto || unit.rules.constructionYard);
        }
        if (TILE_TARGET_ORDER_TYPE_NAMES.has(orderTypeName) || ORE_TILE_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            return unit.canMove === true;
        }
        if (OBJECT_TARGET_ORDER_TYPE_NAMES.has(orderTypeName)) {
            return unit.canMove === true;
        }
        return false;
    }

    private isTileWithinMap(rx: number, ry: number): boolean {
        return !!this.game.map.getTile(rx, ry);
    }

    private incrementCount(target: Record<string, number>, key: string): void {
        target[key] = (target[key] ?? 0) + 1;
    }

    private getHumanTimestamp(game: GameApi) {
        return formatTimeDuration(game.getCurrentTick() / NATURAL_TICK_RATE);
    }

    private logBotStatus(message: string) {
        if (!this.enableLogging) {
            return;
        }
        const timestamp = this.getHumanTimestamp(this.game);
        this.logger.info(`[CheckpointControlBot] ${timestamp}: ${message}`);
    }
}
