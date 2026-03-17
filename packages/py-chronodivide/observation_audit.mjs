import { parseArgs, parseCsvIntegerArg, parseIntegerArg, requireArg } from "./cli_args.mjs";
import { createReplayResimContext, stepReplayTick } from "./resim_core.mjs";
import { collectPlayerObservationSnapshot } from "./snapshot.mjs";

function usage() {
  return [
    "Usage:",
    "  node observation_audit.mjs --replay <path> --data-dir <path>",
    "                             [--player <name>] [--max-tick <n>]",
    "                             [--sample-every <n>] [--sample-ticks 1,50,100]",
    "                             [--unit-limit <n>]",
    "",
    "Checks whether player-visible APIs behave as expected and whether global APIs",
    "still expose hidden enemy state that would leak into SL features.",
  ].join("\n");
}

function collectAuditAtCurrentTick(context, playerName, unitLimit) {
  const gameApi = context.gameApi;
  const allUnits = gameApi.getAllUnits();
  const players = gameApi.getPlayers();
  const enemyPlayerNames = players.filter(
    (name) => name !== playerName && !gameApi.areAlliedPlayers(playerName, name),
  );

  const visibleEnemyIds = gameApi.getVisibleUnits(playerName, "enemy").slice().sort((left, right) => left - right);
  const visibleEnemyIdSet = new Set(visibleEnemyIds);
  const hiddenEnemyIds = allUnits
    .map((id) => gameApi.getUnitData(id) ?? gameApi.getGameObjectData(id))
    .filter(Boolean)
    .filter((unit) => enemyPlayerNames.includes(unit.owner))
    .map((unit) => unit.id)
    .filter((id) => !visibleEnemyIdSet.has(id))
    .sort((left, right) => left - right);

  const hiddenEnemySamples = hiddenEnemyIds
    .slice(0, unitLimit ?? 5)
    .map((id) => gameApi.getUnitData(id) ?? gameApi.getGameObjectData(id))
    .filter(Boolean)
    .map((unit) => ({
      id: unit.id,
      owner: unit.owner,
      name: unit.name,
      tile: unit.tile ? { x: unit.tile.rx, y: unit.tile.ry } : undefined,
      hitPoints: unit.hitPoints,
      maxHitPoints: unit.maxHitPoints,
    }));

  const enemyPlayerData = enemyPlayerNames.map((name) => {
    const player = gameApi.getPlayerData(name);
    return {
      name: player.name,
      credits: player.credits,
      power: player.power,
      radarDisabled: player.radarDisabled,
    };
  });

  return {
    tick: gameApi.getCurrentTick(),
    playerName,
    enemyPlayerNames,
    visibleEnemyCount: visibleEnemyIds.length,
    hiddenEnemyCount: hiddenEnemyIds.length,
    hiddenEnemyDataAccessible: hiddenEnemyIds.length > 0 && hiddenEnemySamples.length > 0,
    enemyPlayerDataAccessible: enemyPlayerData.length > 0,
    enemyPlayerData,
    hiddenEnemySamples,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const replayPath = requireArg(args, "replay");
  const dataDir = requireArg(args, "data-dir");
  const maxTick = parseIntegerArg(args["max-tick"], 300, "max-tick");
  const sampleEvery = parseIntegerArg(args["sample-every"], 50, "sample-every");
  const unitLimit = parseIntegerArg(args["unit-limit"], 5, "unit-limit");
  const sampleTicks = parseCsvIntegerArg(args["sample-ticks"]);

  const context = await createReplayResimContext({
    dataDir,
    replayPath,
  });

  const playerName = args.player
    ? String(args.player)
    : context.replay.gameOpts.humanPlayers[0]?.name;

  if (!playerName) {
    throw new Error("Could not infer a player name for the observation audit.");
  }

  const targetTicks = new Set(sampleTicks);
  const auditSamples = [];
  const observationSamples = [];

  while (context.gameApi.getCurrentTick() < Math.min(maxTick, context.replay.endTick)) {
    const progressed = stepReplayTick(context);
    if (!progressed) {
      break;
    }

    const tick = context.gameApi.getCurrentTick();
    const shouldSample =
      targetTicks.has(tick) ||
      (sampleEvery > 0 && tick % sampleEvery === 0) ||
      tick === maxTick;

    if (shouldSample) {
      auditSamples.push(collectAuditAtCurrentTick(context, playerName, unitLimit));
      observationSamples.push(
        collectPlayerObservationSnapshot(context.gameApi, {
          playerName,
          unitLimit,
        }),
      );
    }
  }

  const result = {
    replayPath: context.replayPath,
    mapName: context.replay.gameOpts.mapName,
    endTick: context.replay.endTick,
    playerName,
    summary: {
      hiddenEnemyDataAccessibleAtAnySample: auditSamples.some((sample) => sample.hiddenEnemyDataAccessible),
      enemyPlayerDataAccessibleAtAnySample: auditSamples.some((sample) => sample.enemyPlayerDataAccessible),
      sampleCount: auditSamples.length,
    },
    auditSamples,
    observationSamples,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
