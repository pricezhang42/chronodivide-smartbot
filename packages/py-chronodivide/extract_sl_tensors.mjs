import fs from "node:fs";
import { once } from "node:events";
import path from "node:path";

import { parseArgs, parseIntegerArg, requireArg } from "./cli_args.mjs";
import { extractReplaySupervisedDataset } from "./sl_dataset.mjs";

function usage() {
  return [
    "Usage:",
    "  node extract_sl_tensors.mjs --replay <path> --data-dir <path>",
    "                              [--player <name|all>]",
    "                              [--include-no-action true|false]",
    "                              [--include-ui-actions true|false]",
    "                              [--max-actions <n>]",
    "                              [--max-tick <n>]",
    "                              [--max-entities <n>]",
    "                              [--max-selected-units <n>]",
    "                              [--spatial-size <n>]",
    "                              [--minimap-size <n>]",
    "                              [--include-flat true|false]",
    "                              [--include-debug true|false]",
    "                              [--output <path>]",
    "",
    "Extracts an action-aligned supervised-learning tensor dataset from a replay.",
    "The output is JSON with fixed-shape numeric tensor blocks plus schema metadata.",
  ].join("\n");
}

async function writeChunk(stream, chunk) {
  if (!stream.write(chunk)) {
    await once(stream, "drain");
  }
}

async function endStream(stream) {
  stream.end();
  await once(stream, "finish");
}

async function writeDatasetJson(outputPath, result) {
  const stream = fs.createWriteStream(path.resolve(outputPath), { encoding: "utf-8" });
  stream.on("error", (error) => {
    throw error;
  });

  try {
    await writeChunk(stream, "{");
    const topLevelEntries = [
      ["replay", result.replay],
      ["sampledPlayers", result.sampledPlayers],
      ["options", result.options],
      ["schema", result.schema],
      ["counts", result.counts],
    ];

    for (let index = 0; index < topLevelEntries.length; index += 1) {
      const [key, value] = topLevelEntries[index];
      if (index > 0) {
        await writeChunk(stream, ",");
      }
      await writeChunk(stream, JSON.stringify(key));
      await writeChunk(stream, ":");
      await writeChunk(stream, JSON.stringify(value));
    }

    await writeChunk(stream, ',\"samples\":[');
    for (let index = 0; index < result.samples.length; index += 1) {
      if (index > 0) {
        await writeChunk(stream, ",");
      }
      await writeChunk(stream, JSON.stringify(result.samples[index]));
    }
    await writeChunk(stream, "]}");
    await endStream(stream);
  } catch (error) {
    stream.destroy();
    throw error;
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
  const player = args.player ? String(args.player) : null;
  const includeNoAction = Boolean(args["include-no-action"] ?? false);
  const includeUiActions = Boolean(args["include-ui-actions"] ?? false);
  const maxActions = args["max-actions"] === undefined ? null : parseIntegerArg(args["max-actions"], 0, "max-actions");
  const maxTick = args["max-tick"] === undefined ? null : parseIntegerArg(args["max-tick"], 0, "max-tick");
  const maxEntities = parseIntegerArg(args["max-entities"], 128, "max-entities");
  const maxSelectedUnits = parseIntegerArg(args["max-selected-units"], 64, "max-selected-units");
  const spatialSize = parseIntegerArg(args["spatial-size"], 32, "spatial-size");
  const minimapSize = parseIntegerArg(args["minimap-size"], 64, "minimap-size");
  const includeFlat = Boolean(args["include-flat"] ?? false);
  const includeDebug = Boolean(args["include-debug"] ?? false);
  const outputPath = args.output ? String(args.output) : null;

  const result = await extractReplaySupervisedDataset({
    dataDir,
    replayPath,
    player,
    includeNoAction,
    includeUiActions,
    maxActions,
    maxTick,
    maxEntities,
    maxSelectedUnits,
    spatialSize,
    minimapSize,
    includeFlat,
    includeDebug,
  });

  if (outputPath) {
    await writeDatasetJson(outputPath, result);
  } else {
    console.log(JSON.stringify(result));
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
