import fs from "node:fs";
import path from "node:path";

import { cdapi as publicCdapi } from "@chronodivide/game-api";

import { parseArgs, parseIntegerArg } from "./cli_args.mjs";
import { resimulateReplay } from "./resim_core.mjs";
import { collectGameSnapshot } from "./snapshot.mjs";
import { SupalosaBot } from "../chronodivide-bot/dist/bot/bot.js";
import { Countries } from "../chronodivide-bot/dist/bot/logic/common/utils.js";

function usage() {
  return [
    "Usage:",
    "  node roundtrip_check.mjs --data-dir <path> [--map <map>] [--max-tick <n>]",
    "                           [--sample-every <n>] [--output <path>] [--keep-replay]",
    "",
    "This script creates a local replay, re-simulates it through the replay bridge,",
    "and checks that sampled snapshots match exactly.",
  ].join("\n");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const dataDir = path.resolve(String(args["data-dir"] ?? "d:/workspace/ra2-headless-mix"));
  const mapName = String(args.map ?? "simple-1v1-no-preview.map");
  const maxTick = parseIntegerArg(args["max-tick"], 150, "max-tick");
  const sampleEvery = parseIntegerArg(args["sample-every"], 50, "sample-every");

  await publicCdapi.init(dataDir);

  const playerA = "ReplayCheckA";
  const playerB = "ReplayCheckB";

  const game = await publicCdapi.createGame({
    buildOffAlly: false,
    cratesAppear: false,
    credits: 10000,
    gameMode: publicCdapi.getAvailableGameModes(mapName)[0],
    gameSpeed: 6,
    mapName,
    mcvRepacks: true,
    shortGame: true,
    superWeapons: false,
    unitCount: 0,
    online: false,
    agents: [
      new SupalosaBot(playerA, Countries.FRANCE, [playerB], false),
      new SupalosaBot(playerB, Countries.IRAQ, [playerA], false),
    ],
  });

  const originalSamples = [];
  for (let i = 0; i < maxTick && !game.isFinished(); i++) {
    await game.update();
    if (game.getCurrentTick() % sampleEvery === 0 || game.getCurrentTick() === maxTick) {
      originalSamples.push(
        collectGameSnapshot(game.gameApi, {
          playerName: playerA,
        }),
      );
    }
  }

  const replayDir = path.resolve("packages/py-chronodivide");
  const replayPath = game.saveReplay(replayDir);
  const endTick = game.getCurrentTick();
  game.dispose();

  const resimResult = await resimulateReplay({
    replayPath,
    dataDir,
    playerName: playerA,
    maxTick: endTick,
    sampleEvery,
  });

  const matches = JSON.stringify(originalSamples) === JSON.stringify(resimResult.samples);
  const summary = {
    replayPath,
    endTick,
    sampleEvery,
    originalSampleTicks: originalSamples.map((sample) => sample.tick),
    resimSampleTicks: resimResult.samples.map((sample) => sample.tick),
    matches,
  };

  if (!args["keep-replay"]) {
    fs.unlinkSync(replayPath);
  }

  const json = JSON.stringify(summary, null, 2);
  if (args.output) {
    fs.writeFileSync(path.resolve(String(args.output)), json, "utf8");
  }

  console.log(json);
  if (!matches) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
