import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import {
    FactoryType,
    GameApi,
    ObjectType,
    TechnoRules,
    UnitData,
    cdapi,
} from "@chronodivide/game-api";
import { SupalosaBot } from "../../chronodivide-bot/dist/bot/bot.js";
import { CheckpointControlBot } from "../../chronodivide-bot/dist/bot/checkpointControlBot.js";
import { Countries } from "../../chronodivide-bot/dist/bot/logic/common/utils.js";
import { createProductionAdvisor } from "../../chronodivide-bot/dist/bot/logic/building/productionAdvisorFactory.js";
import { NullProductionAdvisor } from "../../chronodivide-bot/dist/bot/logic/building/nullProductionAdvisor.js";
import { DummyBot } from "./dummyBot/dummyBot.js";

const DEFAULT_MAP_NAME = "2_pinch_point_le.map";
const DEFAULT_MATCH_COUNT = 3;
const DEFAULT_MAX_TICKS = 12000;
const DEFAULT_SAMPLE_INTERVAL_TICKS = 15;
const DEFAULT_OUTPUT_DIR = path.resolve(process.cwd(), "arena-eval-results");

type BotMode = "baseline" | "advisor" | "control" | "dummy";

type LiveControlBotStats = {
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
    lastPredictedFamily: string | null;
    lastPredictedScore: number | null;
};

type ParsedArgs = {
    outputDir: string;
    replayDir: string;
    mapName: string;
    matchCount: number;
    maxTicks: number;
    sampleIntervalTicks: number;
    mixDir: string;
    checkpointPath: string | null;
    candidateMode: BotMode;
    candidateCountry: Countries;
    opponentMode: BotMode;
    opponentCountry: Countries;
};

type PlayerSnapshot = {
    credits: number;
    economyValue: number;
    developmentValue: number;
    militaryValue: number;
    harvesterCount: number;
    refineryCount: number;
    powerBuildingCount: number;
    productionBuildingCount: number;
    techBuildingCount: number;
    combatUnitCount: number;
    defenseBuildingCount: number;
};

type PlayerTracker = {
    playerName: string;
    countedOwnedIds: Set<number>;
    usedMoney: number;
    peaks: {
        economyValue: number;
        developmentValue: number;
        militaryValue: number;
        credits: number;
    };
    latest: PlayerSnapshot | null;
};

type MatchPlayerSummary = PlayerSnapshot & {
    defeated: boolean;
    usedMoney: number;
    peakEconomyValue: number;
    peakDevelopmentValue: number;
    peakMilitaryValue: number;
    peakCredits: number;
};

type MatchSummary = {
    matchIndex: number;
    finished: boolean;
    winner: "candidate" | "opponent" | "draw" | "unknown";
    currentTick: number;
    durationSeconds: number;
    replayPath: string;
    checkpointPath: string | null;
    candidate: MatchPlayerSummary;
    opponent: MatchPlayerSummary;
    candidateControlStats?: LiveControlBotStats | null;
    opponentControlStats?: LiveControlBotStats | null;
};

const COUNTRY_BY_NAME: Record<string, Countries> = {
    USA: Countries.USA,
    AMERICA: Countries.USA,
    FRANCE: Countries.FRANCE,
    GERMANY: Countries.GERMANY,
    GREAT_BRITAIN: Countries.GREAT_BRITAIN,
    KOREA: Countries.KOREA,
    IRAQ: Countries.IRAQ,
    CUBA: Countries.CUBA,
    LIBYA: Countries.LIBYA,
    RUSSIA: Countries.RUSSIA,
};

function ensureDir(dir: string): void {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

function parseInteger(value: string | undefined, fallback: number): number {
    if (!value) {
        return fallback;
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function parseCountry(rawValue: string | undefined, fallback: Countries): Countries {
    const value = rawValue?.trim().toUpperCase();
    if (!value) {
        return fallback;
    }
    return COUNTRY_BY_NAME[value] ?? fallback;
}

function parseBotMode(rawValue: string | undefined, fallback: BotMode): BotMode {
    const value = rawValue?.trim().toLowerCase();
    if (value === "baseline" || value === "advisor" || value === "control" || value === "dummy") {
        return value;
    }
    return fallback;
}

function parseArgs(argv: string[]): ParsedArgs {
    const args = new Map<string, string>();
    for (let index = 0; index < argv.length; index += 1) {
        const token = argv[index];
        if (!token.startsWith("--")) {
            continue;
        }
        const key = token.slice(2);
        const next = argv[index + 1];
        if (!next || next.startsWith("--")) {
            args.set(key, "true");
            continue;
        }
        args.set(key, next);
        index += 1;
    }

    const outputDir = path.resolve(args.get("output-dir") ?? DEFAULT_OUTPUT_DIR);
    return {
        outputDir,
        replayDir: path.resolve(args.get("replay-dir") ?? path.join(outputDir, "replays")),
        mapName: args.get("map-name") ?? DEFAULT_MAP_NAME,
        matchCount: parseInteger(args.get("matches"), DEFAULT_MATCH_COUNT),
        maxTicks: parseInteger(args.get("max-ticks"), DEFAULT_MAX_TICKS),
        sampleIntervalTicks: parseInteger(args.get("sample-interval-ticks"), DEFAULT_SAMPLE_INTERVAL_TICKS),
        mixDir: path.resolve(args.get("mix-dir") ?? process.env.MIX_DIR ?? "./../../../ra2-headless-mix"),
        checkpointPath: args.get("checkpoint-path") ?? process.env.SL_CHECKPOINT_PATH ?? null,
        candidateMode: parseBotMode(args.get("candidate-mode"), "advisor"),
        candidateCountry: parseCountry(args.get("candidate-country"), Countries.IRAQ),
        opponentMode: parseBotMode(args.get("opponent-mode"), "baseline"),
        opponentCountry: parseCountry(args.get("opponent-country"), Countries.IRAQ),
    };
}

function buildBot(
    mode: BotMode,
    name: string,
    country: Countries,
    allies: string[],
    checkpointPath: string | null,
): SupalosaBot | CheckpointControlBot | DummyBot {
    if (mode === "control") {
        return new CheckpointControlBot(
            name,
            country,
            allies,
            true,
            checkpointPath ?? process.env.SL_CHECKPOINT_PATH ?? "",
            process.env.SL_PYTHON_EXECUTABLE ?? "python",
        );
    }
    if (mode === "dummy") {
        return new DummyBot(name, country);
    }
    const advisor = mode === "advisor" ? createProductionAdvisor() : new NullProductionAdvisor();
    return new SupalosaBot(name, country, allies, true, undefined, advisor);
}

async function waitForAdvisorRequests(bots: Array<SupalosaBot | CheckpointControlBot | DummyBot>): Promise<void> {
    const pendingBots = bots.filter((bot) => {
        const candidate = bot as unknown as {
            hasPendingProductionAdvisorRequest?: () => boolean;
        };
        return typeof candidate.hasPendingProductionAdvisorRequest === "function"
            ? candidate.hasPendingProductionAdvisorRequest()
            : false;
    });
    if (pendingBots.length === 0) {
        return;
    }
    await Promise.all(
        pendingBots.map((bot) => {
            const candidate = bot as unknown as {
                waitForPendingProductionAdvisorRequest?: () => Promise<void>;
            };
            return typeof candidate.waitForPendingProductionAdvisorRequest === "function"
                ? candidate.waitForPendingProductionAdvisorRequest()
                : Promise.resolve();
        }),
    );
}

function isEconomyAsset(unit: UnitData): boolean {
    const rules = unit.rules;
    return (
        rules.harvester ||
        rules.refinery ||
        rules.produceCashAmount > 0 ||
        rules.produceCashStartup > 0
    );
}

function isMilitaryAsset(unit: UnitData): boolean {
    const rules = unit.rules;
    if (rules.harvester) {
        return false;
    }
    if (rules.isBaseDefense || rules.isSelectableCombatant) {
        return true;
    }
    return Boolean(
        unit.primaryWeapon ||
            unit.secondaryWeapon ||
            rules.primary ||
            rules.secondary ||
            rules.elitePrimary ||
            rules.eliteSecondary,
    );
}

function isProductionBuilding(rules: TechnoRules): boolean {
    return rules.factory !== FactoryType.None || rules.weaponsFactory || rules.helipad;
}

function isTechBuilding(rules: TechnoRules): boolean {
    return Boolean(
        rules.radar ||
            rules.superWeapon ||
            rules.techLevel > 1 ||
            (rules.prerequisite && rules.prerequisite.length > 0 && !rules.refinery && !rules.isBaseDefense),
    );
}

function getUnitValue(unit: UnitData): number {
    return Number.isFinite(unit.purchaseValue) && unit.purchaseValue > 0 ? unit.purchaseValue : unit.rules.cost;
}

function getOwnedUnits(gameApi: GameApi, playerName: string): UnitData[] {
    const units: UnitData[] = [];
    gameApi.getAllUnits().forEach((unitId) => {
        const unit = gameApi.getUnitData(unitId);
        if (!unit || unit.owner !== playerName) {
            return;
        }
        units.push(unit);
    });
    return units;
}

function samplePlayer(gameApi: GameApi, playerName: string): PlayerSnapshot {
    const player = gameApi.getPlayerData(playerName);
    const units = getOwnedUnits(gameApi, playerName);
    const snapshot: PlayerSnapshot = {
        credits: player.credits,
        economyValue: 0,
        developmentValue: 0,
        militaryValue: 0,
        harvesterCount: 0,
        refineryCount: 0,
        powerBuildingCount: 0,
        productionBuildingCount: 0,
        techBuildingCount: 0,
        combatUnitCount: 0,
        defenseBuildingCount: 0,
    };

    units.forEach((unit) => {
        const rules = unit.rules;
        const value = getUnitValue(unit);
        if (isEconomyAsset(unit)) {
            snapshot.economyValue += value;
            if (rules.harvester) {
                snapshot.harvesterCount += 1;
            }
            if (rules.refinery) {
                snapshot.refineryCount += 1;
            }
            return;
        }
        if (isMilitaryAsset(unit)) {
            snapshot.militaryValue += value;
            if (unit.type === ObjectType.Building && rules.isBaseDefense) {
                snapshot.defenseBuildingCount += 1;
            } else if (rules.isSelectableCombatant) {
                snapshot.combatUnitCount += 1;
            }
            return;
        }

        snapshot.developmentValue += value;
        if (unit.type === ObjectType.Building) {
            if (rules.power > 0) {
                snapshot.powerBuildingCount += 1;
            }
            if (isProductionBuilding(rules)) {
                snapshot.productionBuildingCount += 1;
            }
            if (isTechBuilding(rules)) {
                snapshot.techBuildingCount += 1;
            }
        } else if (rules.deploysInto || rules.constructionYard) {
            snapshot.techBuildingCount += 1;
        }
    });

    return snapshot;
}

function createTracker(gameApi: GameApi, playerName: string): PlayerTracker {
    const baselineIds = new Set<number>();
    getOwnedUnits(gameApi, playerName).forEach((unit) => baselineIds.add(unit.id));
    const initial = samplePlayer(gameApi, playerName);
    return {
        playerName,
        countedOwnedIds: baselineIds,
        usedMoney: 0,
        peaks: {
            economyValue: initial.economyValue,
            developmentValue: initial.developmentValue,
            militaryValue: initial.militaryValue,
            credits: initial.credits,
        },
        latest: initial,
    };
}

function updateFirstSeenOwners(gameApi: GameApi, firstSeenOwnerById: Map<number, string | null>): void {
    gameApi.getAllUnits().forEach((unitId) => {
        if (firstSeenOwnerById.has(unitId)) {
            return;
        }
        const unit = gameApi.getUnitData(unitId);
        firstSeenOwnerById.set(unitId, unit?.owner ?? null);
    });
}

function updateTracker(gameApi: GameApi, tracker: PlayerTracker, firstSeenOwnerById: Map<number, string | null>): void {
    const ownedUnits = getOwnedUnits(gameApi, tracker.playerName);
    ownedUnits.forEach((unit) => {
        if (tracker.countedOwnedIds.has(unit.id)) {
            return;
        }
        tracker.countedOwnedIds.add(unit.id);
        if (firstSeenOwnerById.get(unit.id) === tracker.playerName) {
            tracker.usedMoney += getUnitValue(unit);
        }
    });

    const snapshot = samplePlayer(gameApi, tracker.playerName);
    tracker.latest = snapshot;
    tracker.peaks.economyValue = Math.max(tracker.peaks.economyValue, snapshot.economyValue);
    tracker.peaks.developmentValue = Math.max(tracker.peaks.developmentValue, snapshot.developmentValue);
    tracker.peaks.militaryValue = Math.max(tracker.peaks.militaryValue, snapshot.militaryValue);
    tracker.peaks.credits = Math.max(tracker.peaks.credits, snapshot.credits);
}

function determineWinner(candidateDefeated: boolean, opponentDefeated: boolean): "candidate" | "opponent" | "draw" | "unknown" {
    if (candidateDefeated && opponentDefeated) {
        return "draw";
    }
    if (!candidateDefeated && opponentDefeated) {
        return "candidate";
    }
    if (candidateDefeated && !opponentDefeated) {
        return "opponent";
    }
    return "unknown";
}

function summarizeTrackedPlayer(tracker: PlayerTracker, defeated: boolean): MatchPlayerSummary {
    const latest = tracker.latest ?? {
        credits: 0,
        economyValue: 0,
        developmentValue: 0,
        militaryValue: 0,
        harvesterCount: 0,
        refineryCount: 0,
        powerBuildingCount: 0,
        productionBuildingCount: 0,
        techBuildingCount: 0,
        combatUnitCount: 0,
        defenseBuildingCount: 0,
    };
    return {
        ...latest,
        defeated,
        usedMoney: tracker.usedMoney,
        peakEconomyValue: tracker.peaks.economyValue,
        peakDevelopmentValue: tracker.peaks.developmentValue,
        peakMilitaryValue: tracker.peaks.militaryValue,
        peakCredits: tracker.peaks.credits,
    };
}

function getLiveControlStats(bot: SupalosaBot | CheckpointControlBot | DummyBot): LiveControlBotStats | null {
    const candidate = bot as unknown as {
        getLiveControlStats?: () => LiveControlBotStats;
    };
    return typeof candidate.getLiveControlStats === "function" ? candidate.getLiveControlStats() : null;
}

function sumRecordCounts(records: Array<Record<string, number> | undefined | null>): Record<string, number> {
    const totals: Record<string, number> = {};
    records.forEach((record) => {
        if (!record) {
            return;
        }
        Object.entries(record).forEach(([key, value]) => {
            totals[key] = (totals[key] ?? 0) + value;
        });
    });
    return Object.fromEntries(Object.entries(totals).sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])));
}

async function runMatch(args: ParsedArgs, matchIndex: number): Promise<MatchSummary> {
    const timestamp = String(Date.now()).slice(-6);
    const candidateName = `Candidate_${matchIndex}_${timestamp}`;
    const opponentName = `Supalosa_${matchIndex}_${timestamp}`;
    const candidateBot = buildBot(args.candidateMode, candidateName, args.candidateCountry, [candidateName], args.checkpointPath);
    const opponentBot = buildBot(args.opponentMode, opponentName, args.opponentCountry, [opponentName], args.checkpointPath);

    let game: Awaited<ReturnType<typeof cdapi.createGame>> | null = null;
    try {
        game = await cdapi.createGame({
            buildOffAlly: false,
            cratesAppear: false,
            credits: 10000,
            gameMode: cdapi.getAvailableGameModes(args.mapName)[0],
            gameSpeed: 6,
            mapName: args.mapName,
            mcvRepacks: true,
            shortGame: true,
            superWeapons: false,
            unitCount: 0,
            online: false,
            agents: [candidateBot, opponentBot],
        });

        const firstSeenOwnerById = new Map<number, string | null>();
        updateFirstSeenOwners(game.gameApi, firstSeenOwnerById);
        const candidateTracker = createTracker(game.gameApi, candidateName);
        const opponentTracker = createTracker(game.gameApi, opponentName);

        let loopCount = 0;
        while (!game.isFinished() && game.getCurrentTick() < args.maxTicks) {
            await game.update();
            await waitForAdvisorRequests([candidateBot, opponentBot]);
            updateFirstSeenOwners(game.gameApi, firstSeenOwnerById);
            if (game.getCurrentTick() % args.sampleIntervalTicks === 0) {
                updateTracker(game.gameApi, candidateTracker, firstSeenOwnerById);
                updateTracker(game.gameApi, opponentTracker, firstSeenOwnerById);
            }
            loopCount += 1;
            if (loopCount % 200 === 0) {
                console.log(
                    `[arena-eval] match ${matchIndex}/${args.matchCount} tick=${game.getCurrentTick()} candidateCredits=${candidateTracker.latest?.credits ?? 0} opponentCredits=${opponentTracker.latest?.credits ?? 0}`,
                );
            }
        }

        updateFirstSeenOwners(game.gameApi, firstSeenOwnerById);
        updateTracker(game.gameApi, candidateTracker, firstSeenOwnerById);
        updateTracker(game.gameApi, opponentTracker, firstSeenOwnerById);

        const stats = game.getPlayerStats();
        const candidateStats = stats.find((player) => player.name === candidateName);
        const opponentStats = stats.find((player) => player.name === opponentName);
        const replayPath = game.saveReplay(args.replayDir);
        const currentTick = game.getCurrentTick();
        const tickRate = game.getTickRate();
        const finished = game.isFinished();

        const candidateDefeated = candidateStats?.defeated ?? false;
        const opponentDefeated = opponentStats?.defeated ?? false;
        return {
            matchIndex,
            finished,
            winner: determineWinner(candidateDefeated, opponentDefeated),
            currentTick,
            durationSeconds: tickRate > 0 ? currentTick / tickRate : 0,
            replayPath,
            checkpointPath: args.checkpointPath,
            candidate: summarizeTrackedPlayer(candidateTracker, candidateDefeated),
            opponent: summarizeTrackedPlayer(opponentTracker, opponentDefeated),
            candidateControlStats: getLiveControlStats(candidateBot),
            opponentControlStats: getLiveControlStats(opponentBot),
        };
    } finally {
        game?.dispose();
        if ("dispose" in candidateBot && typeof candidateBot.dispose === "function") await candidateBot.dispose();
        if ("dispose" in opponentBot && typeof opponentBot.dispose === "function") await opponentBot.dispose();
    }
}

function mean(values: number[]): number {
    if (values.length === 0) {
        return 0;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeMatches(matches: MatchSummary[]) {
    const wins = matches.filter((match) => match.winner === "candidate").length;
    const losses = matches.filter((match) => match.winner === "opponent").length;
    const draws = matches.filter((match) => match.winner === "draw" || match.winner === "unknown").length;

    const candidateMetrics = {
        averageUsedMoney: mean(matches.map((match) => match.candidate.usedMoney)),
        averageFinalCredits: mean(matches.map((match) => match.candidate.credits)),
        averageEconomyValueFinal: mean(matches.map((match) => match.candidate.economyValue)),
        averageDevelopmentValueFinal: mean(matches.map((match) => match.candidate.developmentValue)),
        averageMilitaryValueFinal: mean(matches.map((match) => match.candidate.militaryValue)),
        averageEconomyValuePeak: mean(matches.map((match) => match.candidate.peakEconomyValue)),
        averageDevelopmentValuePeak: mean(matches.map((match) => match.candidate.peakDevelopmentValue)),
        averageMilitaryValuePeak: mean(matches.map((match) => match.candidate.peakMilitaryValue)),
        averageHarvestersFinal: mean(matches.map((match) => match.candidate.harvesterCount)),
        averageRefineriesFinal: mean(matches.map((match) => match.candidate.refineryCount)),
        averagePowerBuildingsFinal: mean(matches.map((match) => match.candidate.powerBuildingCount)),
        averageProductionBuildingsFinal: mean(matches.map((match) => match.candidate.productionBuildingCount)),
        averageTechBuildingsFinal: mean(matches.map((match) => match.candidate.techBuildingCount)),
        averageCombatUnitsFinal: mean(matches.map((match) => match.candidate.combatUnitCount)),
        averageDefenseBuildingsFinal: mean(matches.map((match) => match.candidate.defenseBuildingCount)),
    };

    const opponentMetrics = {
        averageUsedMoney: mean(matches.map((match) => match.opponent.usedMoney)),
        averageFinalCredits: mean(matches.map((match) => match.opponent.credits)),
        averageEconomyValueFinal: mean(matches.map((match) => match.opponent.economyValue)),
        averageDevelopmentValueFinal: mean(matches.map((match) => match.opponent.developmentValue)),
        averageMilitaryValueFinal: mean(matches.map((match) => match.opponent.militaryValue)),
        averageEconomyValuePeak: mean(matches.map((match) => match.opponent.peakEconomyValue)),
        averageDevelopmentValuePeak: mean(matches.map((match) => match.opponent.peakDevelopmentValue)),
        averageMilitaryValuePeak: mean(matches.map((match) => match.opponent.peakMilitaryValue)),
        averageHarvestersFinal: mean(matches.map((match) => match.opponent.harvesterCount)),
        averageRefineriesFinal: mean(matches.map((match) => match.opponent.refineryCount)),
        averagePowerBuildingsFinal: mean(matches.map((match) => match.opponent.powerBuildingCount)),
        averageProductionBuildingsFinal: mean(matches.map((match) => match.opponent.productionBuildingCount)),
        averageTechBuildingsFinal: mean(matches.map((match) => match.opponent.techBuildingCount)),
        averageCombatUnitsFinal: mean(matches.map((match) => match.opponent.combatUnitCount)),
        averageDefenseBuildingsFinal: mean(matches.map((match) => match.opponent.defenseBuildingCount)),
    };

    return {
        wins,
        losses,
        draws,
        winRate: matches.length > 0 ? wins / matches.length : 0,
        averageDurationSeconds: mean(matches.map((match) => match.durationSeconds)),
        averageCurrentTick: mean(matches.map((match) => match.currentTick)),
        candidate: candidateMetrics,
        opponent: opponentMetrics,
        candidateControl: {
            predictedFamilyCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.predictedFamilyCounts)),
            executedFamilyCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.executedFamilyCounts)),
            noopReasonCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.noopReasonCounts)),
            predictedOrderTypeCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.predictedOrderTypeCounts)),
            executedOrderTypeCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.executedOrderTypeCounts)),
            predictedQueueUpdateCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.predictedQueueUpdateCounts)),
            executedQueueUpdateCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.executedQueueUpdateCounts)),
            predictedQueueObjectCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.predictedQueueObjectCounts)),
            executedQueueObjectCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.executedQueueObjectCounts)),
            predictedPlaceBuildingCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.predictedPlaceBuildingCounts)),
            executedPlaceBuildingCounts: sumRecordCounts(matches.map((match) => match.candidateControlStats?.executedPlaceBuildingCounts)),
        },
        opponentControl: {
            predictedFamilyCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.predictedFamilyCounts)),
            executedFamilyCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.executedFamilyCounts)),
            noopReasonCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.noopReasonCounts)),
            predictedOrderTypeCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.predictedOrderTypeCounts)),
            executedOrderTypeCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.executedOrderTypeCounts)),
            predictedQueueUpdateCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.predictedQueueUpdateCounts)),
            executedQueueUpdateCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.executedQueueUpdateCounts)),
            predictedQueueObjectCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.predictedQueueObjectCounts)),
            executedQueueObjectCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.executedQueueObjectCounts)),
            predictedPlaceBuildingCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.predictedPlaceBuildingCounts)),
            executedPlaceBuildingCounts: sumRecordCounts(matches.map((match) => match.opponentControlStats?.executedPlaceBuildingCounts)),
        },
    };
}

async function main(): Promise<void> {
    const args = parseArgs(process.argv.slice(2));
    ensureDir(args.outputDir);
    ensureDir(args.replayDir);
    await cdapi.init(args.mixDir);

    const matches: MatchSummary[] = [];
    for (let matchIndex = 1; matchIndex <= args.matchCount; matchIndex += 1) {
        const summary = await runMatch(args, matchIndex);
        matches.push(summary);
        console.log(
            `[arena-eval] match ${matchIndex}/${args.matchCount}: winner=${summary.winner} ticks=${summary.currentTick} replay=${summary.replayPath}`,
        );
    }

    const report = {
        generatedAt: new Date().toISOString(),
        mapName: args.mapName,
        matchCount: args.matchCount,
        maxTicks: args.maxTicks,
        sampleIntervalTicks: args.sampleIntervalTicks,
        checkpointPath: args.checkpointPath,
        candidate: {
            mode: args.candidateMode,
            country: String(args.candidateCountry),
        },
        opponent: {
            mode: args.opponentMode,
            country: String(args.opponentCountry),
        },
        aggregate: summarizeMatches(matches),
        matches,
    };

    const outputPath = path.join(args.outputDir, "summary.json");
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2), "utf8");
    console.log(`[arena-eval] summary written to ${outputPath}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
