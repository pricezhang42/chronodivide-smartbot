# RA2 Full-Control Model Bot Plan

This note describes how to build a Red Alert 2 bot that is driven by our supervised-learning model and does not rely on `SupalosaBot` for policy decisions.

Implementation tracking lives in [RA2_FULL_CONTROL_MODEL_BOT_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/RA2_FULL_CONTROL_MODEL_BOT_IMPLEMENTATION_CHECKLIST.md).

## Goal

Build a new bot that:

- extends raw [Bot](D:/workspace/supalosa-chronodivide-bot/node_modules/@chronodivide/game-api/dist/index.d.ts)
- loads our checkpoint through a Python inference service
- uses model outputs for:
  - `Order`
  - `Queue`
  - `PlaceBuilding`
  - `ActivateSuperWeapon`
  - `SellObject`
  - `ToggleRepair`
  - `ResignGame`
- does not delegate policy decisions to:
  - [SupalosaBot](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/bot.ts)
  - `MissionController`
  - `QueueController`
  - handwritten strategy modules

## Important Runtime Insight

The live API already supports direct execution through:

- `orderUnits(unitIds, orderType, ...)`
- `queueForProduction(...)`
- `unqueueFromProduction(...)`
- `placeBuilding(...)`
- `sellObject(...)`
- `toggleRepairWrench(...)`
- `activateSuperWeapon(...)`

So the live bot does not need to replay the human UI selection workflow. V2 `commandedUnits` is the correct runtime control interface.

## Current Implementation Snapshot

The first full-control path now exists:

- [CheckpointControlBot](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/checkpointControlBot.ts) owns the live decision loop
- [liveFeaturePayloadBuilder.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/liveFeaturePayloadBuilder.ts) builds generalized live model inputs
- [live_policy_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_policy_service.py) loads V2 checkpoints and returns executable actions
- [evaluateAgainstSupalosaBot.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-driver/src/evaluateAgainstSupalosaBot.ts) can launch the new bot in `control` mode

Validated so far:

- the full-control bot launches in real matches
- the policy service loads a real V2 checkpoint on CUDA
- short live smoke matches execute real `Order` and `Queue` actions
- a longer `1200`-tick smoke stayed stable without crashing

Current main gap:

- the opening is still too passive after initial deployment
- runtime legality masks and cooldowns now exist, but the bot still over-predicts `Queue::Add::NAPOWR`
- richer live diagnostics are still needed around queue-to-placement handoff
- longer full-game evaluation is still pending

## What We Reuse

- feature extraction and snapshot integration from:
  - [liveFeaturePayloadBuilder.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/liveFeaturePayloadBuilder.ts)
  - [py-chronodivide/features.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/features.mjs)
  - [py-chronodivide/snapshot.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/snapshot.mjs)
- checkpoint loading and runtime model helpers from:
  - [live_production_advisor_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_production_advisor_service.py)
  - [live_policy_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_policy_service.py)
  - [model_v2.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_lib/model_v2.py)
- generic execution helpers when they remain useful, such as:
  - [actionBatcher.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/mission/actionBatcher.ts)

## What We Do Not Reuse As Policy

- [SupalosaBot](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/bot.ts) as the controlling bot
- `MissionController`
- `QueueController`
- any handwritten strategy fallback

Allowed runtime safety behavior:

- reject illegal actions
- clip invalid targets
- convert failed actions to `Noop`
- apply small execution cooldowns

Not allowed:

- "if model fails, ask `SupalosaBot` what to do"

## Runtime Architecture

### 1. Bot

Use [CheckpointControlBot](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/checkpointControlBot.ts) as the standalone live bot.

Responsibilities:

- maintain lightweight runtime state
- build live model inputs
- call the Python policy service
- execute model outputs through `ActionsApi`
- record debug and evaluation counters

### 2. Policy Service

Use [live_policy_service.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/live_policy_service.py).

Responsibilities:

- load a checkpoint once
- rebuild V2 model inputs
- run the model in free-running mode
- decode hierarchical outputs into executable payloads
- return confidence and top-k debug info

### 3. Feature Builder

Use [liveFeaturePayloadBuilder.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/liveFeaturePayloadBuilder.ts) as the generalized live payload builder.

It should provide:

- feature tensors
- player production state
- superweapon state
- entity object ids
- shared name vocabulary
- runtime action history
- recent family and order history
- virtual selection state

### 4. Output Contract

The service should return a decoded action payload, not just scores. The current contract is defined through:

- [livePolicyTypes.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/livePolicyTypes.ts)
- [livePolicyValidation.ts](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot/src/bot/logic/modelcontrol/livePolicyValidation.ts)

Main families:

- `Order`
- `Queue`
- `PlaceBuilding`
- `ActivateSuperWeapon`
- `SellObject`
- `ToggleRepair`
- `ResignGame`
- runtime-only `Noop`

## Control Loop

Recommended live cadence:

- run inference every `6` to `15` ticks
- commit only one or a few actions per decision step
- add family and unit cooldowns to prevent thrashing

Runtime state should track:

- recent executed commands
- recent predicted families
- recent chosen unit ids
- recent queue interactions
- per-unit last-command tick
- per-queue last-command tick
- last failed action reason

## Action-Space Contract

Use V2, not V1.

References:

- [RA2_SL_LABEL_LAYOUT_V2.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V2.md)
- [RA2_SL_LABEL_LAYOUT_V2_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V2_IMPLEMENTATION_CHECKLIST.md)

Key rules:

- `SelectUnitsAction` remains folded out of runtime policy
- `Hold` and `Resume` remain disabled from the main policy
- `commandedUnits` is the runtime control unit set

Initial runtime support:

- `Order`
  - `Move`
  - `Attack`
  - `AttackMove`
  - `Deploy`
  - `Dock`
  - `Gather`
  - `Repair`
- `Queue`
  - `Add`
  - `Cancel`
  - `AddNext`
- `PlaceBuilding`
- `ActivateSuperWeapon`
- `SellObject`
- `ToggleRepair`

## Recommended Next Work

The scaffolding is far enough along that the next work should improve live quality, not add another parallel runtime:

1. add family and unit cooldowns
2. improve queue-to-placement handoff so ready buildings become `PlaceBuilding` instead of more `Queue`
3. record richer live diagnostics, especially target and placement counts
4. run longer evaluation and checkpoint comparisons in `control` mode
