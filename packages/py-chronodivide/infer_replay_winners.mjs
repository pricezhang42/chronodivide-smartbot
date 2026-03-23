/**
 * Batch replay winner inference.
 *
 * Simulates each replay to endTick and checks defeat state to determine the winner.
 * Outputs one JSONL record per replay to stdout.
 *
 * Usage:
 *   node infer_replay_winners.mjs --data-dir <path> --input <file-of-paths>
 */

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

import { parseArgs, requireArg } from "./cli_args.mjs";
import { createReplayResimContext, stepReplayTick } from "./resim_core.mjs";
import { collectPlayerStatsAtCurrentTick } from "./snapshot.mjs";
import { loadGameApiBridge } from "./bridge.mjs";

const initializedDataDirsByBridge = new WeakMap();

async function initBridgeForDataDir(bridge, dataDir) {
  const resolvedDataDir = path.resolve(dataDir);
  let initializedDataDirs = initializedDataDirsByBridge.get(bridge);
  if (!initializedDataDirs) {
    initializedDataDirs = new Set();
    initializedDataDirsByBridge.set(bridge, initializedDataDirs);
  }
  if (!initializedDataDirs.has(resolvedDataDir)) {
    await bridge.cdapi.init(resolvedDataDir);
    initializedDataDirs.add(resolvedDataDir);
  }
  return resolvedDataDir;
}

function determineWinner(playerStats) {
  if (playerStats.length !== 2) {
    return { winner: null, reason: `unexpected_player_count_${playerStats.length}` };
  }

  const [p0, p1] = playerStats;

  // Check defeat state
  if (p0.defeated && !p1.defeated) {
    return { winner: p1.name, loser: p0.name, reason: "defeat" };
  }
  if (!p0.defeated && p1.defeated) {
    return { winner: p0.name, loser: p1.name, reason: "defeat" };
  }
  if (p0.defeated && p1.defeated) {
    return { winner: null, reason: "both_defeated" };
  }

  // Check resignation
  if (p0.resigned && !p1.resigned) {
    return { winner: p1.name, loser: p0.name, reason: "resignation" };
  }
  if (!p0.resigned && p1.resigned) {
    return { winner: p0.name, loser: p1.name, reason: "resignation" };
  }

  // Check disconnection
  if (p0.dropped && !p1.dropped) {
    return { winner: p1.name, loser: p0.name, reason: "disconnect" };
  }
  if (!p0.dropped && p1.dropped) {
    return { winner: p0.name, loser: p1.name, reason: "disconnect" };
  }

  // Fall back to score comparison
  if (p0.score > p1.score) {
    return { winner: p0.name, loser: p1.name, reason: "score" };
  }
  if (p1.score > p0.score) {
    return { winner: p1.name, loser: p0.name, reason: "score" };
  }

  return { winner: null, reason: "unknown" };
}

async function inferWinner(bridge, dataDir, replayPath) {
  const context = await createReplayResimContext({
    dataDir,
    replayPath,
    bridge,
  });

  // Fast-forward to endTick
  while (stepReplayTick(context)) {
    // just simulate
  }

  const playerStats = collectPlayerStatsAtCurrentTick(context.gameApi, context.internalGame);
  const result = determineWinner(playerStats);

  return {
    path: path.resolve(replayPath),
    gameId: context.replay.gameId,
    endTick: context.replay.endTick,
    finalTick: context.gameApi.getCurrentTick(),
    players: playerStats.map((p) => ({
      name: p.name,
      country: p.country,
      defeated: p.defeated,
      resigned: p.resigned,
      dropped: p.dropped,
      score: p.score,
    })),
    ...result,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dataDir = requireArg(args, "data-dir");
  const inputPath = args.input ? String(args.input) : null;

  const bridge = await loadGameApiBridge();

  // Read replay paths
  let replayPaths;
  if (inputPath) {
    const content = fs.readFileSync(path.resolve(inputPath), "utf8");
    replayPaths = content
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
  } else {
    const rl = readline.createInterface({ input: process.stdin });
    replayPaths = [];
    for await (const line of rl) {
      const trimmed = line.trim();
      if (trimmed) replayPaths.push(trimmed);
    }
  }

  let processed = 0;
  let errors = 0;

  for (const replayPath of replayPaths) {
    try {
      const result = await inferWinner(bridge, dataDir, replayPath);
      process.stdout.write(JSON.stringify(result) + "\n");
      processed++;
    } catch (err) {
      const errorRecord = {
        path: path.resolve(replayPath),
        error: err instanceof Error ? err.message : String(err),
      };
      process.stdout.write(JSON.stringify(errorRecord) + "\n");
      errors++;
    }
  }

  process.stderr.write(`Processed ${processed} replays, ${errors} errors.\n`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
