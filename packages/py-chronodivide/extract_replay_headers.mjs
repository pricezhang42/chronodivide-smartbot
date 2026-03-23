/**
 * Batch replay header extractor.
 *
 * Reads a newline-delimited list of replay paths from stdin (or --input file),
 * parses each replay's header + initializes the game engine to resolve country
 * names and side IDs, then outputs one JSON line per replay to stdout.
 *
 * This avoids full simulation — it only initializes the game to tick 0.
 *
 * Usage:
 *   node extract_replay_headers.mjs --data-dir <path> --input <path>
 *   echo "path1.rpl\npath2.rpl" | node extract_replay_headers.mjs --data-dir <path>
 */

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

import { parseArgs, requireArg } from "./cli_args.mjs";
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

function getRulesOverride(bridge, replay) {
  const gameModes = bridge.Engine.getMpModes();
  return bridge.Engine.getIni(gameModes.getById(replay.gameOpts.gameMode).rulesOverride);
}

function getMixinRules(bridge, replay) {
  return bridge.MixinRules.getTypes(replay.gameOpts)
    .map((type) => bridge.Engine.mixinRulesFileNames.get(type))
    .filter(Boolean)
    .map((iniName) => bridge.Engine.getIni(iniName));
}

async function extractHeader(bridge, replayPath) {
  const resolvedPath = path.resolve(replayPath);
  const replayText = fs.readFileSync(resolvedPath, "utf8");
  const replay = new bridge.Replay();
  replay.unserialize(replayText, {
    name: path.basename(resolvedPath),
    timestamp: Date.now(),
  });

  // Initialize game engine to resolve country names and side IDs
  const mapFile = new bridge.MapFile(await bridge.Engine.vfs.openFileWithRfs(replay.gameOpts.mapName));
  await bridge.Engine.loadTheater(mapFile.theaterType);

  const theaterSettings = bridge.Engine.getTheaterSettings(bridge.Engine.getActiveEngine(), mapFile.theaterType);
  const theaterIni = bridge.Engine.getTheaterIni(bridge.Engine.getActiveEngine(), mapFile.theaterType);
  const tileSets = new bridge.TileSets(theaterIni);
  tileSets.loadTileData(bridge.Engine.getTileData(), theaterSettings.extension);

  const internalGame = bridge.GameFactory.create(
    mapFile,
    tileSets,
    bridge.Engine.getRules(),
    bridge.Engine.getArt(),
    bridge.Engine.getAi(),
    getRulesOverride(bridge, replay),
    getMixinRules(bridge, replay),
    replay.gameId,
    replay.gameTimestamp,
    replay.gameOpts,
    bridge.Engine.getMpModes(),
    false,
    { version: "0.0.0" },
    bridge.AppLogger.get("ini"),
    new bridge.BoxedVar(false),
    new bridge.BoxedVar(0),
    bridge.AppLogger.get("action"),
  );

  internalGame.init(undefined);
  internalGame.start();

  const gameApi = new bridge.GameApi(internalGame, false);

  const players = replay.gameOpts.humanPlayers.map((player) => ({
    name: player.name,
    countryId: player.countryId,
    countryName: gameApi.getPlayerData(player.name)?.country?.name ?? null,
    sideId: gameApi.getPlayerData(player.name)?.country?.side ?? null,
    colorId: player.colorId,
    startPos: player.startPos,
    teamId: player.teamId,
  }));

  return {
    path: resolvedPath,
    gameId: replay.gameId,
    mapName: replay.gameOpts.mapName,
    gameMode: replay.gameOpts.gameMode,
    endTick: replay.endTick,
    playerCount: players.length,
    players,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dataDir = requireArg(args, "data-dir");
  const inputPath = args.input ? String(args.input) : null;

  const bridge = await loadGameApiBridge();
  await initBridgeForDataDir(bridge, dataDir);

  // Read replay paths from input file or stdin
  let replayPaths;
  if (inputPath) {
    const content = fs.readFileSync(path.resolve(inputPath), "utf8");
    replayPaths = content.split("\n").map((line) => line.trim()).filter(Boolean);
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
      const header = await extractHeader(bridge, replayPath);
      // Output one JSON line per replay (JSONL format)
      process.stdout.write(JSON.stringify(header) + "\n");
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
