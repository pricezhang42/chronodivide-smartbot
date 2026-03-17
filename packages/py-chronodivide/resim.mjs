import fs from "node:fs";
import path from "node:path";

import { parseArgs, parseCsvIntegerArg, parseIntegerArg, requireArg } from "./cli_args.mjs";
import { resimulateReplay } from "./resim_core.mjs";

function usage() {
  return [
    "Usage:",
    "  node resim.mjs --replay <path> --data-dir <path> [--player <name>] [--max-tick <n>]",
    "                 [--sample-every <n>] [--sample-ticks 1,50,100] [--unit-limit <n>]",
    "                 [--sample-mode global|observation] [--output <path>]",
    "",
    "Example:",
    "  node resim.mjs \\",
    "    --replay ..\\chronodivide-bot-sl\\ladder_replays_top50\\00758dde-b725-4442-ae8f-a657069251a0.rpl \\",
    "    --data-dir d:\\workspace\\ra2-headless-mix \\",
    "    --max-tick 300 --sample-ticks 1,50,100,200,300",
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
  const maxTick = parseIntegerArg(args["max-tick"], null, "max-tick");
  const sampleEvery = parseIntegerArg(args["sample-every"], 50, "sample-every");
  const unitLimit = parseIntegerArg(args["unit-limit"], null, "unit-limit");
  const sampleTicks = parseCsvIntegerArg(args["sample-ticks"]);
  const sampleMode = String(args["sample-mode"] ?? "global");
  if (!["global", "observation"].includes(sampleMode)) {
    throw new Error(`Expected --sample-mode to be "global" or "observation", got "${sampleMode}".`);
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
  });

  const json = JSON.stringify(result, null, 2);
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
