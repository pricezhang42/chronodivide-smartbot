import fs from "node:fs";
import path from "node:path";

import { parseArgs, parseIntegerArg, requireArg } from "./cli_args.mjs";
import {
  buildActionTimelines,
  decodeActionLabel,
  shouldKeepRawActionId,
  updateSelectionFromAction,
} from "./labels.mjs";
import { createReplayResimContext } from "./resim_core.mjs";

function usage() {
  return [
    "Usage:",
    "  node extract_action_labels.mjs --replay <path> --data-dir <path>",
    "                                [--player <name|all>]",
    "                                [--include-no-action true|false]",
    "                                [--include-ui-actions true|false]",
    "                                [--max-actions <n>]",
    "                                [--max-tick <n>]",
    "                                [--output <path>]",
    "",
    "Extracts lightweight action-label records from a replay without building observation tensors.",
  ].join("\n");
}

function buildReplaySummary(context) {
  return {
    path: context.replayPath,
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
  };
}

function inferSampledPlayers(context, playerArg) {
  const replayPlayerNames = context.replay.gameOpts.humanPlayers.map((player) => player.name);
  if (playerArg === "all") {
    return replayPlayerNames;
  }
  return [playerArg ?? replayPlayerNames[0]].filter(Boolean);
}

function buildCounts(actions) {
  const rawActionCounts = new Map();
  const playerCounts = new Map();

  for (const action of actions) {
    rawActionCounts.set(action.rawActionId, (rawActionCounts.get(action.rawActionId) ?? 0) + 1);
    playerCounts.set(action.playerName, (playerCounts.get(action.playerName) ?? 0) + 1);
  }

  return {
    rawActionCounts: Array.from(rawActionCounts, ([rawActionId, count]) => ({ rawActionId, count })).sort(
      (left, right) => left.rawActionId - right.rawActionId,
    ),
    playerCounts: Array.from(playerCounts, ([playerName, count]) => ({ playerName, count })).sort(
      (left, right) => right.count - left.count,
    ),
  };
}

async function extractReplayActionLabels({
  dataDir,
  replayPath,
  player = null,
  includeNoAction = false,
  includeUiActions = false,
  maxActions = null,
  maxTick = null,
} = {}) {
  if (!dataDir) {
    throw new Error("extractReplayActionLabels requires a dataDir.");
  }
  if (!replayPath) {
    throw new Error("extractReplayActionLabels requires a replayPath.");
  }

  const context = await createReplayResimContext({
    dataDir,
    replayPath,
  });
  const sampledPlayers = inferSampledPlayers(context, player);
  if (!sampledPlayers.length) {
    throw new Error("Could not infer a player for action-label extraction.");
  }

  const sampledPlayerSet = new Set(sampledPlayers);
  const replayPlayerNames = context.replay.gameOpts.humanPlayers.map((playerInfo) => playerInfo.name);
  const timelines = buildActionTimelines(context.replay, {
    playerNames: sampledPlayers,
    includeNoAction,
    includeUiActions,
  });
  const timelineCursors = new Map(sampledPlayers.map((name) => [name, 0]));
  const currentSelections = new Map(replayPlayerNames.map((name) => [name, []]));
  const actions = [];
  const resolvedMaxTick = maxTick === null ? context.replay.endTick : Math.min(maxTick, context.replay.endTick);

  while (context.gameApi.getCurrentTick() < resolvedMaxTick) {
    const currentTick = context.gameApi.getCurrentTick();
    const tickEvents = context.replayEventsByTick.get(currentTick) ?? [];
    const stagedActions = [];

    for (const event of tickEvents) {
      if (event.constructor.name !== "TurnActionsReplayEvent") {
        continue;
      }

      for (const [playerId, payloads] of event.payload) {
        const actionPlayer = context.internalGame.getPlayer(playerId);
        const playerName =
          actionPlayer?.name ?? context.replay.gameOpts.humanPlayers[playerId]?.name ?? `player_${playerId}`;

        for (const payload of payloads) {
          const action = context.actionFactory.create(payload.id);
          action.player = actionPlayer;
          action.unserialize(payload.params);

          const selectionBeforeActionIds = currentSelections.get(playerName) ?? [];
          const selectionAfterActionIds = updateSelectionFromAction(action, selectionBeforeActionIds);
          const keepAction =
            sampledPlayerSet.has(playerName) &&
            shouldKeepRawActionId(payload.id, {
              includeNoAction,
              includeUiActions,
            });

          if (keepAction && (maxActions === null || actions.length < maxActions)) {
            const timeline = timelines.get(playerName) ?? [];
            const cursor = timelineCursors.get(playerName) ?? 0;
            const timelineEntry = timeline[cursor] ?? null;
            if (timelineEntry) {
              timelineCursors.set(playerName, cursor + 1);
            }

            const label = decodeActionLabel(action, {
              rawActionId: payload.id,
              gameApi: context.gameApi,
              selectionBefore: selectionBeforeActionIds,
              selectionAfter: selectionAfterActionIds,
              timelineEntry,
            });

            actions.push({
              tick: currentTick,
              playerId,
              playerName,
              rawActionId: label.rawActionId,
              rawActionName: label.rawActionName,
              delayFromPreviousAction: label.delayFromPreviousAction,
              delayToNextAction: label.delayToNextAction,
              queue: label.queue,
              orderTypeId: label.orderTypeId,
              orderTypeName: label.orderTypeName,
              targetMode: label.targetMode,
              targetModeId: label.targetModeId,
              queueUpdateTypeId: label.queueUpdateTypeId,
              queueUpdateTypeName: label.queueUpdateTypeName,
              itemName: label.itemName,
              buildingName: label.buildingName,
              superWeaponTypeId: label.superWeaponTypeId,
              superWeaponTypeName: label.superWeaponTypeName,
            });
          }

          currentSelections.set(playerName, selectionAfterActionIds);
          stagedActions.push(action);
        }
      }
    }

    for (const action of stagedActions) {
      action.process();
    }

    context.internalGame.update();

    if (maxActions !== null && actions.length >= maxActions) {
      break;
    }
  }

  return {
    replay: buildReplaySummary(context),
    sampledPlayers,
    options: {
      includeNoAction,
      includeUiActions,
      maxActions,
      maxTick: resolvedMaxTick,
    },
    counts: buildCounts(actions),
    actions,
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
  const player = args.player ? String(args.player) : null;
  const includeNoAction = Boolean(args["include-no-action"] ?? false);
  const includeUiActions = Boolean(args["include-ui-actions"] ?? false);
  const maxActions = args["max-actions"] === undefined ? null : parseIntegerArg(args["max-actions"], 0, "max-actions");
  const maxTick = args["max-tick"] === undefined ? null : parseIntegerArg(args["max-tick"], 0, "max-tick");
  const outputPath = args.output ? String(args.output) : null;

  const result = await extractReplayActionLabels({
    dataDir,
    replayPath,
    player,
    includeNoAction,
    includeUiActions,
    maxActions,
    maxTick,
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
