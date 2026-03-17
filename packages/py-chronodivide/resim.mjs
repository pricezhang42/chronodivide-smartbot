import fs from "node:fs";
import path from "node:path";

import { parseArgs, parseCsvIntegerArg, parseIntegerArg, requireArg } from "./cli_args.mjs";
import {
  buildReplayResultBase,
  collectReplaySamples,
  createReplayResimContext,
  resimulateReplay,
} from "./resim_core.mjs";
import { collectPlayerStatsAtCurrentTick, collectStaticGameData } from "./snapshot.mjs";

function usage() {
  return [
    "Usage:",
    "  node resim.mjs --replay <path> --data-dir <path> [--player <name>] [--max-tick <n>]",
    "                 [--sample-every <n>] [--sample-ticks 1,50,100] [--unit-limit <n>]",
    "                 [--sample-mode global|observation] [--include-visible-tiles true|false]",
    "                 [--include-visible-resource-tiles true|false] [--include-super-weapons true|false]",
    "                 [--include-terrain-objects true|false] [--include-neutral-units true|false]",
    "                 [--include-tile-resources true|false] [--include-player-production true|false]",
    "                 [--include-player-stats true|false] [--include-static-data true|false]",
    "                 [--include-static-map true|false]",
    "                 [--compact true|false] [--stream-output true|false]",
    "                 [--output <path>]",
    "",
    "Example:",
    "  node resim.mjs \\",
    "    --replay ..\\chronodivide-bot-sl\\ladder_replays_top50\\00758dde-b725-4442-ae8f-a657069251a0.rpl \\",
    "    --data-dir d:\\workspace\\ra2-headless-mix \\",
    "    --max-tick 300 --sample-ticks 1,50,100,200,300",
  ].join("\n");
}

function stringifyJson(value, compact = false) {
  return compact ? JSON.stringify(value) : JSON.stringify(value, null, 2);
}

function writeSync(fd, text) {
  fs.writeSync(fd, text, undefined, "utf8");
}

function writeStreamedReplayRecording({
  outputPath,
  resultBase,
  context,
  sampleOptions,
  includePlayerStats,
}) {
  const fd = fs.openSync(outputPath, "w");
  let sampleCount = 0;

  try {
    writeSync(fd, '{"replay":');
    writeSync(fd, JSON.stringify(resultBase.replay));
    writeSync(fd, ',"dataDir":');
    writeSync(fd, JSON.stringify(resultBase.dataDir));
    writeSync(fd, ',"sampledPlayer":');
    writeSync(fd, JSON.stringify(resultBase.sampledPlayer));
    writeSync(fd, ',"sampleMode":');
    writeSync(fd, JSON.stringify(resultBase.sampleMode));
    if (resultBase.staticData !== undefined) {
      writeSync(fd, ',"staticData":');
      writeSync(fd, JSON.stringify(resultBase.staticData));
    }
    writeSync(fd, ',"samples":[');

    const { sampleCount: writtenSamples } = collectReplaySamples(context, {
      ...sampleOptions,
      collectSamples: false,
      onSample(sample) {
        if (sampleCount > 0) {
          writeSync(fd, ",");
        }
        writeSync(fd, JSON.stringify(sample));
        sampleCount += 1;
      },
    });

    writeSync(fd, "]");

    const playerStatsAtStop = includePlayerStats
      ? collectPlayerStatsAtCurrentTick(context.gameApi, context.internalGame)
      : undefined;
    const stoppedTick = context.gameApi.getCurrentTick();
    const playbackReachedEnd = stoppedTick >= context.replay.endTick;

    writeSync(fd, ',"stoppedTick":');
    writeSync(fd, JSON.stringify(stoppedTick));
    writeSync(fd, ',"playbackReachedEnd":');
    writeSync(fd, JSON.stringify(playbackReachedEnd));
    if (playerStatsAtStop !== undefined) {
      writeSync(fd, ',"playerStatsAtStop":');
      writeSync(fd, JSON.stringify(playerStatsAtStop));
    }
    writeSync(fd, "}");

    return {
      sampleCount: writtenSamples,
      stoppedTick,
      playbackReachedEnd,
      playerStatsAtStop,
    };
  } finally {
    fs.closeSync(fd);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const replayPath = requireArg(args, "replay");
  const dataDir = requireArg(args, "data-dir");
  const maxTick = parseIntegerArg(args["max-tick"], null, "max-tick");
  const sampleEvery = parseIntegerArg(args["sample-every"], 50, "sample-every");
  const unitLimit = parseIntegerArg(args["unit-limit"], null, "unit-limit");
  const sampleTicks = parseCsvIntegerArg(args["sample-ticks"]);
  const sampleMode = String(args["sample-mode"] ?? "global");
  const includeVisibleTiles = Boolean(args["include-visible-tiles"] ?? false);
  const includeVisibleResourceTiles = Boolean(args["include-visible-resource-tiles"] ?? false);
  const includeSuperWeapons = Boolean(args["include-super-weapons"] ?? false);
  const includeTerrainObjects = Boolean(args["include-terrain-objects"] ?? false);
  const includeNeutralUnits = Boolean(args["include-neutral-units"] ?? false);
  const includeTileResources = Boolean(args["include-tile-resources"] ?? false);
  const includePlayerProduction = Boolean(args["include-player-production"] ?? false);
  const includePlayerStats = Boolean(args["include-player-stats"] ?? false);
  const includeStaticData = Boolean(args["include-static-data"] ?? false);
  const includeStaticMap = Boolean(args["include-static-map"] ?? false);
  const compact = Boolean(args.compact ?? false);
  const streamOutput = Boolean(args["stream-output"] ?? false);
  if (!["global", "observation"].includes(sampleMode)) {
    throw new Error(`Expected --sample-mode to be "global" or "observation", got "${sampleMode}".`);
  }
  if (streamOutput && !args.output) {
    throw new Error("--stream-output requires --output so the CLI can write one compact JSON file safely.");
  }

  if (streamOutput) {
    const outputPath = path.resolve(String(args.output));
    const context = await createReplayResimContext({
      dataDir,
      replayPath,
    });
    const resolvedPlayerName = args.player ? String(args.player) : null;
    const staticData = includeStaticData
      ? collectStaticGameData(context.gameApi, { includeMapDump: includeStaticMap })
      : undefined;
    const resultBase = buildReplayResultBase(context, {
      playerName: resolvedPlayerName,
      sampleMode,
      staticData,
    });

    writeStreamedReplayRecording({
      outputPath,
      resultBase,
      context,
      sampleOptions: {
        playerName: resultBase.sampledPlayer,
        maxTick,
        sampleEvery,
        sampleTicks,
        unitLimit,
        sampleMode,
        includeVisibleTiles,
        includeVisibleResourceTiles,
        includeSuperWeapons,
        includeTerrainObjects,
        includeNeutralUnits,
        includeTileResources,
        includePlayerProduction,
        includePlayerStats,
      },
      includePlayerStats,
    });

    console.log(`Wrote ${outputPath}`);
    return;
  }

  const result = await resimulateReplay({
    replayPath,
    dataDir,
    playerName: args.player ? String(args.player) : null,
    maxTick,
    sampleEvery,
    sampleTicks,
    unitLimit,
    sampleMode,
    includeVisibleTiles,
    includeVisibleResourceTiles,
    includeSuperWeapons,
    includeTerrainObjects,
    includeNeutralUnits,
    includeTileResources,
    includePlayerProduction,
    includePlayerStats,
    includeStaticData,
    includeStaticMap,
  });

  const json = stringifyJson(result, compact);
  if (args.output) {
    const outputPath = path.resolve(String(args.output));
    fs.writeFileSync(outputPath, json, "utf8");
    console.log(`Wrote ${outputPath}`);
  } else {
    console.log(json);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
