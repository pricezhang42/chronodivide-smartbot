import fs from "node:fs";
import path from "node:path";

import { parseArgs, parseCsvIntegerArg, parseIntegerArg, requireArg } from "./cli_args.mjs";
import { extractObservationFeatureSample, getObservationFeatureSchema } from "./features.mjs";
import { createReplayResimContext, stepReplayTick } from "./resim_core.mjs";

function usage() {
  return [
    "Usage:",
    "  node extract_features.mjs --replay <path> --data-dir <path>",
    "                           [--player <name>] [--max-tick <n>]",
    "                           [--sample-every <n>] [--sample-ticks 1,50,100]",
    "                           [--max-entities <n>] [--spatial-size <n>]",
    "                           [--output <path>]",
    "",
    "Extracts SL-safe observation features from a replay.",
    "This script intentionally avoids omniscient APIs such as getAllUnits().",
  ].join("\n");
}

function buildObjectNameVocabulary(samples, maxEntities) {
  const names = new Set();
  for (const sample of samples) {
    for (const entity of sample.entityMeta) {
      if (entity?.name) {
        names.add(entity.name);
      }
    }
  }

  const ordered = ["<pad>", "<unk>", ...Array.from(names).sort((left, right) => left.localeCompare(right))];
  const idsByName = Object.fromEntries(ordered.map((name, index) => [name, index]));

  for (const sample of samples) {
    const tokens = sample.entityMeta.map((entity) => idsByName[entity.name] ?? idsByName["<unk>"]);
    while (tokens.length < maxEntities) {
      tokens.push(idsByName["<pad>"]);
    }
    sample.entityNameTokens = tokens;
  }

  return {
    idToName: ordered,
    nameToId: idsByName,
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
  const maxEntities = parseIntegerArg(args["max-entities"], 128, "max-entities");
  const spatialSize = parseIntegerArg(args["spatial-size"], 32, "spatial-size");
  const sampleTicks = parseCsvIntegerArg(args["sample-ticks"]);
  const outputPath = args.output ? String(args.output) : null;

  const context = await createReplayResimContext({
    dataDir,
    replayPath,
  });

  const playerName = args.player
    ? String(args.player)
    : context.replay.gameOpts.humanPlayers[0]?.name;

  if (!playerName) {
    throw new Error("Could not infer a player name for feature extraction.");
  }

  const targetTicks = new Set(sampleTicks);
  const resolvedMaxTick = Math.min(maxTick, context.replay.endTick);
  const samples = [];

  while (context.gameApi.getCurrentTick() < resolvedMaxTick) {
    const progressed = stepReplayTick(context);
    if (!progressed) {
      break;
    }

    const tick = context.gameApi.getCurrentTick();
    const shouldSample =
      targetTicks.has(tick) ||
      (sampleEvery > 0 && tick % sampleEvery === 0) ||
      tick === resolvedMaxTick;

    if (shouldSample) {
      samples.push(
        extractObservationFeatureSample(context.gameApi, {
          playerName,
          maxEntities,
          spatialSize,
        }),
      );
    }
  }

  const objectNameVocabulary = buildObjectNameVocabulary(samples, maxEntities);
  const result = {
    replay: {
      path: path.resolve(replayPath),
      gameId: context.replay.gameId,
      mapName: context.replay.gameOpts.mapName,
      endTick: context.replay.endTick,
      players: context.replay.gameOpts.humanPlayers.map((player) => ({
        name: player.name,
        countryId: player.countryId,
        colorId: player.colorId,
        startPos: player.startPos,
        teamId: player.teamId,
      })),
    },
    sampledPlayer: playerName,
    schema: {
      ...getObservationFeatureSchema({
        maxEntities,
        spatialSize,
      }),
      entityNameTokenFeature: "entity_name_token",
      objectNameVocabulary,
    },
    samples,
  };

  const serialized = JSON.stringify(result, null, 2);
  if (outputPath) {
    fs.writeFileSync(path.resolve(outputPath), serialized);
  } else {
    console.log(serialized);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});

