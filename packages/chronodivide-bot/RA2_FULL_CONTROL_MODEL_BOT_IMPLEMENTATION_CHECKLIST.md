# RA2 Full-Control Model Bot Implementation Checklist

This checklist turns [RA2_FULL_CONTROL_MODEL_BOT_PLAN.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/RA2_FULL_CONTROL_MODEL_BOT_PLAN.md) into concrete implementation work.

## Status Legend

- `[x]` done
- `[~]` partially done
- `[ ]` not done

## Current Validated State

- `[x]` A standalone live bot exists in [checkpointControlBot.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/checkpointControlBot.ts).
- `[x]` A full-command Python policy service exists in [live_policy_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_policy_service.py).
- `[x]` A generalized live feature builder exists in [liveFeaturePayloadBuilder.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/liveFeaturePayloadBuilder.ts).
- `[x]` The arena evaluator can launch the new bot in `control` mode via [evaluateAgainstSupalosaBot.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-driver/src/evaluateAgainstSupalosaBot.ts).
- `[x]` The live game API exposes the direct execution primitives we need through [ActionsApi](D:/workspace/supalosa-chronodivide-bot/node_modules/@chronodivide/game-api/dist/index.d.ts).
- `[x]` A real V2 checkpoint loads on CUDA through the live policy service.
- `[x]` A short live smoke match survives and executes real model actions.
- `[~]` A longer live smoke match is stable through the opening, but the bot still collapses into repetitive queue behavior after initial deployment.

## Phase 1: Freeze Runtime Contract

- `[x]` Freeze that the new bot must not delegate policy decisions to:
  - [SupalosaBot](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/bot.ts)
  - `MissionController`
  - `QueueController`
  - handwritten strategy modules
- `[x]` Freeze V2 as the runtime policy contract.
- `[x]` Freeze the current live families:
  - `Order`
  - `Queue`
  - `PlaceBuilding`
  - `ActivateSuperWeapon`
  - `SellObject`
  - `ToggleRepair`
  - `ResignGame`
  - runtime-only `Noop`
- `[x]` Freeze the current queue policy:
  - keep `Add`, `Cancel`, `AddNext`
  - exclude `Hold`, `Resume`
- `[x]` Freeze the rule that live execution uses direct unit ids, not UI-style selection replay.

## Phase 2: New Bot Skeleton

- `[x]` Create `CheckpointControlBot`.
- `[x]` Make it extend raw `Bot`, not `SupalosaBot`.
- `[~]` Add runtime state for:
  - recent executed commands
  - recent predicted families
  - recent chosen unit ids
  - last failed action reason
  - note: family and per-unit cooldown state now exists, but richer queue and target histories are still missing
- `[x]` Add clean startup and shutdown handling for the model service.
- `[~]` Add debug and logging for:
  - current family prediction
  - last executed command
  - last rejected command
  - no-op rate
  - note: counters are now present for families, order types, queue updates, queue objects, and place-building names, but richer per-target diagnostics are still missing

Exit criteria:

- `[x]` The new bot launches in a real game.
- `[x]` The new bot can tick without any dependency on `SupalosaBot`.

## Phase 3: Generalized Runtime Feature Builder

- `[x]` Extract the old building-only checkpoint payload logic into a generalized module.
- `[x]` Create [liveFeaturePayloadBuilder.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/liveFeaturePayloadBuilder.ts).
- `[x]` Preserve snapshot and feature extraction integration with:
  - [py-chronodivide/features.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/features.mjs)
  - [py-chronodivide/snapshot.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/snapshot.mjs)
- `[~]` Extend the payload with runtime command history beyond the old production-advisor path.
- `[~]` Add V2-oriented live history fields:
  - recent family names
  - recent order types
  - virtual selection state
  - note: richer queue and target histories are still missing
- `[x]` Keep the payload compact enough for per-step inference.

Exit criteria:

- `[x]` One live tick can build a full-control model input payload successfully.

## Phase 4: Full-Command Python Policy Service

- `[x]` Create [live_policy_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_policy_service.py).
- `[x]` Reuse robust checkpoint loading logic from [live_production_advisor_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_production_advisor_service.py).
- `[x]` Load V2 checkpoints as the primary path.
- `[~]` Keep V1 support only as a debug path that returns `Noop` for full control.
- `[x]` Run the model in free-running mode for live decisions.
- `[x]` Decode model outputs into a full-command JSON contract.
- `[x]` Return top-k family and subtype scores for debugging.
- `[x]` Add runtime `Noop` behavior when payloads are invalid or decoding is unsafe.

Exit criteria:

- `[x]` The service can accept a live payload and return one executable command payload.

## Phase 5: Runtime Output Contract

- `[x]` Define shared TypeScript and Python payload types for:
  - command family
  - order payload
  - queue payload
  - building placement payload
  - superweapon payload
  - target entity payload
  - no-op and debug metadata
- `[x]` Make the contract explicit about optional versus required fields.
- `[x]` Include confidence scores.
- `[x]` Include decoded ids and names where useful for debugging.

Exit criteria:

- `[x]` The bot and service use one stable full-command schema.

## Phase 6: Non-Order Family Execution

- `[x]` Implement `Queue` execution using:
  - `queueForProduction(...)`
  - `unqueueFromProduction(...)`
- `[x]` Implement `PlaceBuilding` execution using:
  - `placeBuilding(...)`
  - `canPlaceBuilding(...)`
- `[x]` Implement `ActivateSuperWeapon` execution using:
  - `activateSuperWeapon(...)`
- `[x]` Implement `SellObject` execution using:
  - `sellObject(...)`
- `[x]` Implement `ToggleRepair` execution using:
  - `toggleRepairWrench(...)`
- `[x]` Implement `ResignGame` execution using:
  - `quitGame()`
- `[x]` Add family-specific cooldowns to prevent spamming.
- `[~]` Log rejected macro commands and reasons through no-op counters and last-failure tracking.

Exit criteria:

- `[ ]` The bot can complete a full game using only model-driven macro families.

## Phase 7: Order Family Execution

- `[x]` Decode:
  - `orderType`
  - `targetMode`
  - `queueFlag`
  - `commandedUnits`
  - `targetEntity`
  - `targetLocation`
- `[x]` Map decoded order outputs to live `OrderType`.
- `[x]` Build an order executor on top of `ActionsApi.orderUnits(...)`.
- `[ ]` Reuse or adapt [actionBatcher.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/mission/actionBatcher.ts) for dedupe and batching.
- `[x]` Ignore dead, invalid, or non-owned unit ids.
- `[~]` Reject illegal order and unit combinations with basic guards.
- `[x]` Add per-unit cooldowns so units are not re-commanded every inference step.
- `[~]` Support the initial order subset:
  - `Move`
  - `Attack`
  - `AttackMove`
  - `Deploy`
  - `Dock`
  - `Gather`
  - `Repair`
  - note: the executor is generic, but these behaviors have not all been validated in live matches yet
- `[ ]` Add later-order backlog for rarer order types after the base executor is stable.

Exit criteria:

- `[~]` The bot issues model-driven movement and opening orders in live games.
- `[x]` No mission-based unit-control fallback remains.

## Phase 8: Safety And Legality Layer

- `[x]` Add a generic legality check layer before execution.
- `[x]` Convert illegal or degenerate decoded actions into `Noop`.
- `[~]` Add family-specific legality checks:
  - queue object resolvable
  - building placement legal
  - superweapon type valid
  - target entity valid
  - target tile present when required
  - units owned and commandable
- `[~]` Add throttling for repeated failed actions.
- `[x]` Add runtime family and subtype masking in the Python policy service for:
  - legal family selection
  - order type selection
  - target mode selection
  - queue update selection
  - buildable object selection
  - ready superweapon selection
- `[x]` Track rejection rates by family and reason.

Exit criteria:

- `[~]` Short live games no longer thrash on obviously illegal actions.

## Phase 9: Evaluation Integration

- `[x]` Update the arena evaluator so candidate mode can launch the new full-control bot directly.
- `[x]` Keep evaluation against `SupalosaBot`.
- `[x]` Track:
  - win rate
  - used money
  - economy
  - development
  - military
  - per-family action counts
  - no-op and rejection counts
- `[x]` Save replays for each eval match.
- `[x]` Preserve checkpoint path and run metadata in summaries.

Exit criteria:

- `[~]` A trained checkpoint can be evaluated as a real full-control bot after training, but longer and more representative runs are still needed.

## Phase 10: Checkpoint Selection For Live Control

- `[ ]` Add checkpoint selection criteria beyond teacher-forced loss.
- `[ ]` Prefer free-running and family-aware metrics for live-control candidates.
- `[ ]` Add at least these checkpoint tracks:
  - best by free-running family accuracy
  - best by order-family metrics
  - best by queue recall
  - best by arena win rate once the full bot is live
- `[ ]` Decide whether V2 debug should be promoted to the main live-control training path.

## Phase 11: Validation And Spot Checks

- `[x]` Run a smoke match where the new bot survives and executes real model actions.
- `[~]` Run a longer match where the new bot stays stable through the opening without crashing.
- `[~]` Inspect live command behavior for:
  - repeated command spam
  - queue and build stalls
  - inactive armies
- `[~]` Compare live family frequencies to replay priors at a coarse level.
- `[ ]` Add qualitative spot checks for:
  - opening build order
  - first combat engagement
  - first expansion or tech choice
  - first superweapon use

## Recommended Immediate Next Step

- `[ ]` Improve live quality rather than adding more scaffolding:
  - strengthen queue and building handoff logic so the bot stops repeating `Queue::Add::NAPOWR`
  - record richer target and placement diagnostics
  - run longer `control` evaluations against `SupalosaBot`
  - compare multiple V2 checkpoints in live play
