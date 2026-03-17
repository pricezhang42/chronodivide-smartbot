import fs from "node:fs";
import path from "node:path";

import { parseArgs, parseIntegerArg, requireArg } from "./cli_args.mjs";
import {
  buildActionTimelines,
  decodeActionLabel,
  getActionLabelSchema,
  shouldKeepRawActionId,
  updateSelectionFromAction,
} from "./labels.mjs";
import { createReplayResimContext } from "./resim_core.mjs";

function usage() {
  return [
    "Usage:",
    "  node extract_labels.mjs --replay <path> --data-dir <path>",
    "                         [--player <name|all>]",
    "                         [--include-no-action true|false]",
    "                         [--include-ui-actions true|false]",
    "                         [--max-actions <n>]",
    "                         [--output <path>]",
    "",
    "Extracts structured action labels from a replay.",
    "This walks the replay tick-by-tick so selection state and object references stay aligned.",
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

function buildActionCounts(samples) {
  const rawCounts = new Map();
  const familyCounts = new Map();

  for (const sample of samples) {
    rawCounts.set(sample.label.rawActionId, (rawCounts.get(sample.label.rawActionId) ?? 0) + 1);
    familyCounts.set(sample.label.actionFamily, (familyCounts.get(sample.label.actionFamily) ?? 0) + 1);
  }

  return {
    rawActionCounts: Array.from(rawCounts, ([rawActionId, count]) => ({ rawActionId, count })).sort(
      (left, right) => left.rawActionId - right.rawActionId,
    ),
    actionFamilyCounts: Array.from(familyCounts, ([actionFamily, count]) => ({ actionFamily, count })).sort(
      (left, right) => right.count - left.count,
    ),
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
  const playerArg = args.player ? String(args.player) : null;
  const includeNoAction = Boolean(args["include-no-action"] ?? false);
  const includeUiActions = Boolean(args["include-ui-actions"] ?? false);
  const maxActions = args["max-actions"] === undefined ? null : parseIntegerArg(args["max-actions"], 0, "max-actions");
  const outputPath = args.output ? String(args.output) : null;

  const context = await createReplayResimContext({
    dataDir,
    replayPath,
  });

  const replayPlayerNames = context.replay.gameOpts.humanPlayers.map((player) => player.name);
  const sampledPlayers =
    playerArg === "all"
      ? replayPlayerNames
      : [playerArg ?? replayPlayerNames[0]].filter(Boolean);

  if (!sampledPlayers.length) {
    throw new Error("Could not infer a player for label extraction.");
  }

  const sampledPlayerSet = new Set(sampledPlayers);
  const timelines = buildActionTimelines(context.replay, {
    playerNames: sampledPlayers,
    includeNoAction,
    includeUiActions,
  });
  const timelineCursors = new Map(sampledPlayers.map((name) => [name, 0]));
  const currentSelections = new Map(replayPlayerNames.map((name) => [name, []]));
  const samples = [];

  while (context.gameApi.getCurrentTick() < context.replay.endTick) {
    const currentTick = context.gameApi.getCurrentTick();
    const tickEvents = context.replayEventsByTick.get(currentTick) ?? [];
    const stagedActions = [];

    for (const event of tickEvents) {
      if (event.constructor.name !== "TurnActionsReplayEvent") {
        continue;
      }

      for (const [playerId, actions] of event.payload) {
        const player = context.internalGame.getPlayer(playerId);
        const playerName = player?.name ?? context.replay.gameOpts.humanPlayers[playerId]?.name ?? `player_${playerId}`;

        for (const payload of actions) {
          const action = context.actionFactory.create(payload.id);
          action.player = player;
          action.unserialize(payload.params);

          const selectionBefore = currentSelections.get(playerName) ?? [];
          const selectionAfter = updateSelectionFromAction(action, selectionBefore);
          const keepAction =
            sampledPlayerSet.has(playerName) &&
            shouldKeepRawActionId(payload.id, {
              includeNoAction,
              includeUiActions,
            });

          if (keepAction && (maxActions === null || samples.length < maxActions)) {
            const timeline = timelines.get(playerName) ?? [];
            const cursor = timelineCursors.get(playerName) ?? 0;
            const timelineEntry = timeline[cursor] ?? null;
            if (timelineEntry) {
              timelineCursors.set(playerName, cursor + 1);
            }

            samples.push({
              tick: currentTick,
              playerId,
              playerName,
              label: decodeActionLabel(action, {
                rawActionId: payload.id,
                gameApi: context.gameApi,
                selectionBefore,
                selectionAfter,
                timelineEntry,
              }),
            });
          }

          currentSelections.set(playerName, selectionAfter);
          stagedActions.push(action);
        }
      }
    }

    for (const action of stagedActions) {
      action.process();
    }

    context.internalGame.update();

    if (maxActions !== null && samples.length >= maxActions) {
      break;
    }
  }

  const result = {
    replay: buildReplaySummary(context),
    sampledPlayers,
    options: {
      includeNoAction,
      includeUiActions,
      maxActions,
    },
    schema: getActionLabelSchema(),
    counts: buildActionCounts(samples),
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

