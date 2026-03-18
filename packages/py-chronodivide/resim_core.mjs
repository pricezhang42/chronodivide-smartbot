import fs from "node:fs";
import path from "node:path";

import { loadGameApiBridge } from "./bridge.mjs";
import {
  collectGameSnapshot,
  collectPlayerObservationSnapshot,
  collectPlayerStatsAtCurrentTick,
  collectStaticGameData,
} from "./snapshot.mjs";

const initializedDataDirsByBridge = new WeakMap();

function buildReplayEventIndex(replay) {
  const replayEventsByTick = new Map();
  for (const event of replay.events) {
    if (!replayEventsByTick.has(event.tickNo)) {
      replayEventsByTick.set(event.tickNo, []);
    }
    replayEventsByTick.get(event.tickNo).push(event);
  }
  return replayEventsByTick;
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

async function createInternalGame(bridge, replay) {
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

  const actionFactory = new bridge.ActionFactory();
  new bridge.ActionFactoryReg().register(actionFactory, internalGame, undefined);

  return {
    internalGame,
    gameApi: new bridge.GameApi(internalGame, false),
    actionFactory,
  };
}

export async function createReplayResimContext({ dataDir, replayPath, bridge = null }) {
  const loadedBridge = bridge ?? (await loadGameApiBridge());
  const resolvedDataDir = await initBridgeForDataDir(loadedBridge, dataDir);
  const resolvedReplayPath = path.resolve(replayPath);

  const replayText = fs.readFileSync(resolvedReplayPath, "utf8");
  const replay = new loadedBridge.Replay();
  replay.unserialize(replayText, {
    name: path.basename(resolvedReplayPath),
    timestamp: Date.now(),
  });

  const { internalGame, gameApi, actionFactory } = await createInternalGame(loadedBridge, replay);
  const replayEventsByTick = buildReplayEventIndex(replay);

  return {
    bridge: loadedBridge,
    dataDir: resolvedDataDir,
    replayPath: resolvedReplayPath,
    replay,
    internalGame,
    gameApi,
    actionFactory,
    replayEventsByTick,
  };
}

export function applyReplayEventsAtCurrentTick(context) {
  const tick = context.gameApi.getCurrentTick();
  const events = context.replayEventsByTick.get(tick) ?? [];

  for (const event of events) {
    if (event.constructor.name !== "TurnActionsReplayEvent") {
      continue;
    }

    for (const [playerId, actions] of event.payload) {
      for (const payload of actions) {
        const action = context.actionFactory.create(payload.id);
        action.player = context.internalGame.getPlayer(playerId);
        action.unserialize(payload.params);
        action.process();
      }
    }
  }
}

export function stepReplayTick(context) {
  if (context.gameApi.getCurrentTick() >= context.replay.endTick) {
    return false;
  }

  applyReplayEventsAtCurrentTick(context);
  context.internalGame.update();
  return true;
}

export function resolveReplayPlayerName(context, playerName = null) {
  return playerName ?? context.replay.gameOpts.humanPlayers[0]?.name ?? null;
}

export function buildReplayMetadata(context) {
  return {
    path: context.replayPath,
    gameId: context.replay.gameId,
    mapName: context.replay.gameOpts.mapName,
    endTick: context.replay.endTick,
    players: context.replay.gameOpts.humanPlayers.map((player) => ({
      name: player.name,
      countryId: player.countryId,
      countryName: context.gameApi.getPlayerData(player.name)?.country?.name ?? null,
      sideId: context.gameApi.getPlayerData(player.name)?.country?.side ?? null,
      colorId: player.colorId,
      startPos: player.startPos,
      teamId: player.teamId,
    })),
  };
}

export function buildReplayResultBase(
  context,
  {
    playerName = null,
    sampleMode = "global",
    staticData = undefined,
  } = {},
) {
  return {
    replay: buildReplayMetadata(context),
    dataDir: context.dataDir,
    sampledPlayer: resolveReplayPlayerName(context, playerName),
    sampleMode,
    staticData,
  };
}

export function collectReplaySamples(
  context,
  {
    playerName = null,
    maxTick = null,
    sampleEvery = 50,
    sampleTicks = [],
    unitLimit = null,
    sampleMode = "global",
    includeVisibleTiles = false,
    includeVisibleResourceTiles = false,
    includeSuperWeapons = false,
    includeTerrainObjects = false,
    includeNeutralUnits = false,
    includeTileResources = false,
    includePlayerProduction = false,
    includePlayerStats = false,
    collectSamples = true,
    onSample = null,
  } = {},
) {
  const resolvedMaxTick = Math.min(maxTick ?? context.replay.endTick, context.replay.endTick);
  const sampleTickSet = new Set(sampleTicks);
  const samples = collectSamples ? [] : undefined;
  let sampleCount = 0;

  while (context.gameApi.getCurrentTick() < resolvedMaxTick) {
    const progressed = stepReplayTick(context);
    if (!progressed) {
      break;
    }

    const currentTick = context.gameApi.getCurrentTick();
    const shouldSample =
      sampleTickSet.has(currentTick) ||
      (sampleEvery > 0 && currentTick % sampleEvery === 0) ||
      currentTick === resolvedMaxTick;

    if (shouldSample) {
      const sample =
        sampleMode === "observation"
          ? collectPlayerObservationSnapshot(context.gameApi, {
              playerName,
              unitLimit,
              internalGame: context.internalGame,
              includeVisibleTiles,
              includeVisibleResourceTiles,
              includeSuperWeapons,
              includeProduction: includePlayerProduction,
              includePlayerStats,
            })
          : collectGameSnapshot(context.gameApi, {
              playerName,
              unitLimit,
              internalGame: context.internalGame,
              includeVisibleTiles,
              includeSuperWeapons,
              includeTerrainObjects,
              includeNeutralUnits,
              includeTileResources,
              includePlayerProduction,
              includePlayerStats,
            });

      if (samples) {
        samples.push(sample);
      }
      if (typeof onSample === "function") {
        onSample(sample, {
          tick: currentTick,
          sampleIndex: sampleCount,
        });
      }
      sampleCount += 1;
    }
  }

  return {
    samples: samples ?? [],
    sampleCount,
  };
}

export async function resimulateReplay({
  dataDir,
  replayPath,
  playerName = null,
  maxTick = null,
  sampleEvery = 50,
  sampleTicks = [],
  unitLimit = null,
  sampleMode = "global",
  includeVisibleTiles = false,
  includeVisibleResourceTiles = false,
  includeSuperWeapons = false,
  includeTerrainObjects = false,
  includeNeutralUnits = false,
  includeTileResources = false,
  includePlayerProduction = false,
  includePlayerStats = false,
  includeStaticData = false,
  includeStaticMap = false,
} = {}) {
  const context = await createReplayResimContext({
    dataDir,
    replayPath,
  });

  const resolvedPlayerName = resolveReplayPlayerName(context, playerName);
  const staticData = includeStaticData
    ? collectStaticGameData(context.gameApi, { includeMapDump: includeStaticMap })
    : undefined;
  const { samples } = collectReplaySamples(context, {
    playerName: resolvedPlayerName,
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
  });

  const playerStatsAtStop = includePlayerStats
    ? collectPlayerStatsAtCurrentTick(context.gameApi, context.internalGame)
    : undefined;
  const stoppedTick = context.gameApi.getCurrentTick();

  return {
    ...buildReplayResultBase(context, {
      playerName: resolvedPlayerName,
      sampleMode,
      staticData,
    }),
    stoppedTick,
    playbackReachedEnd: stoppedTick >= context.replay.endTick,
    playerStatsAtStop,
    samples,
  };
}
