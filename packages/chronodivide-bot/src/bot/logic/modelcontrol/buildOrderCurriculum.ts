import {
    GameApi,
    OrderType,
    PlayerApi,
    PlayerData,
    QueueStatus,
    QueueType,
    TechnoRules,
    Vector2,
} from "@chronodivide/game-api";

import { getDefaultPlacementLocation } from "../building/buildingRules.js";

type CurriculumStepType = "deploy_mcv" | "queue" | "place";

type CurriculumStep = {
    type: CurriculumStepType;
    objectName: string;
    queueType?: QueueType;
};

export type CurriculumAction = {
    executed: boolean;
    family: string;
    description: string;
};

const SOVIET_BUILD_ORDER: CurriculumStep[] = [
    { type: "deploy_mcv", objectName: "SMCV" },
    { type: "queue", objectName: "NAPOWR", queueType: QueueType.Structures },
    { type: "place", objectName: "NAPOWR" },
    { type: "queue", objectName: "NAREFN", queueType: QueueType.Structures },
    { type: "place", objectName: "NAREFN" },
    { type: "queue", objectName: "NAHAND", queueType: QueueType.Structures },
    { type: "place", objectName: "NAHAND" },
    { type: "queue", objectName: "NAWEAP", queueType: QueueType.Structures },
    { type: "place", objectName: "NAWEAP" },
];

const ALLIED_BUILD_ORDER: CurriculumStep[] = [
    { type: "deploy_mcv", objectName: "AMCV" },
    { type: "queue", objectName: "GAPOWR", queueType: QueueType.Structures },
    { type: "place", objectName: "GAPOWR" },
    { type: "queue", objectName: "GAREFN", queueType: QueueType.Structures },
    { type: "place", objectName: "GAREFN" },
    { type: "queue", objectName: "GAPILE", queueType: QueueType.Structures },
    { type: "place", objectName: "GAPILE" },
    { type: "queue", objectName: "GAWEAP", queueType: QueueType.Structures },
    { type: "place", objectName: "GAWEAP" },
];

const MAX_CURRICULUM_TICK = 5000;

function getBuildOrderForSide(sideId: number): CurriculumStep[] {
    if (sideId === 0) return ALLIED_BUILD_ORDER;
    return SOVIET_BUILD_ORDER;
}

export class BuildOrderCurriculum {
    private stepIndex = 0;
    private readonly steps: CurriculumStep[];
    private done = false;

    constructor(sideId: number) {
        this.steps = getBuildOrderForSide(sideId);
    }

    isComplete(currentTick: number): boolean {
        return this.done || this.stepIndex >= this.steps.length || currentTick > MAX_CURRICULUM_TICK;
    }

    tryExecuteNextStep(
        game: GameApi,
        player: PlayerApi,
        playerData: PlayerData,
    ): CurriculumAction {
        if (this.isComplete(game.getCurrentTick())) {
            return { executed: false, family: "Noop", description: "curriculum_complete" };
        }

        const step = this.steps[this.stepIndex];
        let result: CurriculumAction;
        try {
            result = this.executeStep(step, game, player, playerData);
        } catch (error: unknown) {
            process.stderr.write(`[CURRICULUM] tick=${game.getCurrentTick()} step=${this.stepIndex} ERROR: ${String(error)}\n`);
            return { executed: false, family: "Noop", description: `curriculum_error:${String(error)}` };
        }
        if (result.executed) {
            process.stderr.write(`[CURRICULUM] tick=${game.getCurrentTick()} step=${this.stepIndex} type=${step.type} obj=${step.objectName} desc=${result.description}\n`);
            this.stepIndex += 1;
        }
        return result;
    }

    private executeStep(
        step: CurriculumStep,
        game: GameApi,
        player: PlayerApi,
        playerData: PlayerData,
    ): CurriculumAction {
        switch (step.type) {
            case "deploy_mcv":
                return this.executeDeployMcv(step, game, player);
            case "queue":
                return this.executeQueueStep(step, player);
            case "place":
                return this.executePlaceStep(step, game, player, playerData);
        }
    }

    private executeDeployMcv(
        step: CurriculumStep,
        game: GameApi,
        player: PlayerApi,
    ): CurriculumAction {
        // Check if construction yard already exists (MCV already deployed).
        const conYards = game.getVisibleUnits(
            player.name,
            "self",
            (r: TechnoRules) => r.constructionYard === true,
        );
        if (conYards.length > 0) {
            return { executed: true, family: "Noop", description: `curriculum_skip:conyard_exists` };
        }

        // Find MCV unit.
        const baseUnits = game.getGeneralRules().baseUnit ?? [];
        const mcvIds = game.getVisibleUnits(
            player.name,
            "self",
            (r: TechnoRules) => !!r.deploysInto && baseUnits.includes(r.name),
        );
        if (mcvIds.length === 0) {
            this.done = true;
            return { executed: false, family: "Noop", description: "curriculum_fail:no_mcv" };
        }

        player.actions.orderUnits([mcvIds[0]], OrderType.DeploySelected);
        return { executed: true, family: "Order", description: `curriculum:deploy_mcv` };
    }

    private executeQueueStep(
        step: CurriculumStep,
        player: PlayerApi,
    ): CurriculumAction {
        const queueType = step.queueType!;
        let queueData;
        try {
            queueData = player.production.getQueueData(queueType);
        } catch {
            return { executed: false, family: "Noop", description: `curriculum_wait:no_production:${step.objectName}` };
        }

        // Wait if queue is busy (building something or has a ready item to place).
        if (queueData.status === QueueStatus.Active || queueData.status === QueueStatus.Ready) {
            return { executed: false, family: "Noop", description: `curriculum_wait:queue_busy:${step.objectName}` };
        }

        // Check the object is available.
        const available = player.production.getAvailableObjects(queueType);
        const rules = available.find((r) => r.name === step.objectName);
        if (!rules) {
            // Not yet available — wait (tech requirement not met yet).
            return { executed: false, family: "Noop", description: `curriculum_wait:not_available:${step.objectName}` };
        }

        player.actions.queueForProduction(queueType, rules.name, rules.type, 1);
        return { executed: true, family: "Queue", description: `curriculum:queue:${step.objectName}` };
    }

    private executePlaceStep(
        step: CurriculumStep,
        game: GameApi,
        player: PlayerApi,
        playerData: PlayerData,
    ): CurriculumAction {
        const queueType = QueueType.Structures;
        let queueData;
        try {
            queueData = player.production.getQueueData(queueType);
        } catch {
            return { executed: false, family: "Noop", description: `curriculum_wait:no_production:${step.objectName}` };
        }

        // Check if the building already exists on the map — it may have been auto-placed
        // by the game engine (e.g. refineries). The queue status may not reflect this.
        const existingBuildings = game.getVisibleUnits(
            player.name, "self", (r: TechnoRules) => r.name === step.objectName,
        );
        if (existingBuildings.length > 0) {
            return { executed: true, family: "Noop", description: `curriculum_skip:already_placed:${step.objectName}` };
        }

        // Wait if the building isn't ready yet.
        if (queueData.status !== QueueStatus.Ready) {
            return { executed: false, family: "Noop", description: `curriculum_wait:not_ready:${step.objectName}` };
        }

        // Find the ready item.
        const readyItem = queueData.items.find((item) => item.rules.name === step.objectName);
        if (!readyItem) {
            // Ready item is a different building — skip this step.
            return { executed: true, family: "Noop", description: `curriculum_skip:wrong_ready_item:${step.objectName}` };
        }

        // Find a construction yard as the ideal placement reference.
        const conYards = game.getVisibleUnits(
            player.name,
            "self",
            (r: TechnoRules) => r.constructionYard === true,
        );
        let idealPoint: Vector2;
        if (conYards.length > 0) {
            const cyData = game.getUnitData(conYards[0]);
            idealPoint = new Vector2(cyData!.tile.rx, cyData!.tile.ry);
        } else {
            // Fallback: use map center.
            const mapSize = game.map.getRealMapSize();
            idealPoint = new Vector2(Math.floor(mapSize.width / 2), Math.floor(mapSize.height / 2));
        }

        const location = getDefaultPlacementLocation(game, playerData, idealPoint, readyItem.rules);
        if (!location) {
            // No valid placement found — skip.
            return { executed: true, family: "Noop", description: `curriculum_fail:no_placement:${step.objectName}` };
        }

        player.actions.placeBuilding(step.objectName, location.rx, location.ry);
        return { executed: true, family: "PlaceBuilding", description: `curriculum:place:${step.objectName}` };
    }
}
