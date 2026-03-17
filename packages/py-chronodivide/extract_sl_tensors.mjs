import fs from "node:fs";
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
