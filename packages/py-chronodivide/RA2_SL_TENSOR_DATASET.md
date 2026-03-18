# RA2 SL Tensor Dataset

This note describes the first action-aligned supervised-learning tensor dataset builder in `py-chronodivide`.

## Goal

Provide a reusable replay-to-tensor API that is closer in role to mini-AlphaStar's replay transformer:

- align one player-correct observation to one kept action
- expose fixed-shape numeric tensor sections
- keep schema and vocab metadata next to the extracted samples
- stay usable from other projects without baking in one training stack

The implementation lives in [sl_dataset.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/sl_dataset.mjs).

Replay metadata now also carries per-player replay-global identity fields alongside the action-aligned samples:

- `countryId`
- `countryName`
- `sideId`

These are intended for downstream feature builders such as SL scalar identity features.

The dataset payload can also carry replay-constant static map tensors keyed by player:

- `staticMapSchema`
- `staticMapByPlayer`

These are intended for downstream feature builders such as `mapStatic`.

It can also carry replay-constant super-weapon recharge metadata:

- `superWeaponSchema`

This is intended for downstream feature builders that need to normalize replay `timerSeconds` by per-type nominal `RechargeTime` from the ruleset.

It can also carry replay-constant static rules-driven tech metadata:

- `staticTechTree`

This is intended for downstream projects that want a reusable prerequisite / build-tree view from `RulesApi` without rebuilding it ad hoc.

## Extraction Model

The dataset is action-centric.

For each kept replay action, the extractor records:

- the observation tensor before the action is processed
- the inferred current selection before the action
- the previous kept action context
- the acting player's generic production snapshot before the action
- the decoded label tensor for the current action

This is intentionally close to the `transform_replay_data.py` idea in mini-AlphaStar, where the current observation is paired with the current action label and previous-action context.

## Current API

Reusable library entry point:

- `extractReplaySupervisedDataset(...)`

Reusable tensor helpers:

- `flattenFeatureTensors(...)`
- `flattenLabelTensors(...)`

CLI wrapper:

- [extract_sl_tensors.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/extract_sl_tensors.mjs)

## Feature Tensor Sections

Current feature tensors contain:

- `scalar`
- `lastActionContext`
- `currentSelectionCount`
- `currentSelectionResolvedCount`
- `currentSelectionOverflowCount`
- `currentSelectionIndices`
- `currentSelectionMask`
- `currentSelectionResolvedMask`
- `entityNameTokens`
- `entityMask`
- `entityFeatures`
- `spatial`
- `minimap`

The observation part comes from the existing safe feature extractor.

The raw entity feature block now also carries a small set of generic transient fields that downstream projects can use to build their own intent summaries:

- `factory_status_idle`
- `factory_status_delivering`
- `factory_has_delivery`
- `rally_point_valid`
- `rally_x_norm`
- `rally_y_norm`
- `primary_weapon_cooldown_ticks`
- `secondary_weapon_cooldown_ticks`

The added SL-specific context is:

- previous kept action delay / raw action id / family id / queue flag
- current inferred selection before the action

In addition to the tensor sections, each action-aligned sample now also carries:

- `playerProduction`
- `playerSuperWeapons`

These stay as generic raw summaries for downstream projects to turn into their own feature layout.

`playerProduction` now includes both:

- `availableCountsByQueueType`
- `availableObjectsByQueueType`
- `availableObjects`
- `catalogObjects`

The queue-type summaries and `availableObjects` are intended to represent the current legally available production set at that timestep. They are canonically regrouped with `getQueueTypeForObject(...)` when the engine exposes it, instead of trusting the raw queue-specific query shape directly.

`catalogObjects` is the broader replay-time production catalog exposed by the engine rules/runtime. It is useful for debugging and schema inspection, but downstream action-availability logic should prefer `availableObjects` and `availableObjectsByQueueType`.

## Label Tensor Sections

Current label tensors contain:

- `rawActionId`
- `actionFamilyId`
- `delayToNextAction`
- `queue`
- `orderTypeId`
- `targetModeId`
- `targetEntityIndex`
- `targetNameToken`
- `targetObjectType`
- `targetTile`
- `targetIsOre`
- `actionSelectedUnitCount`
- `actionSelectedUnitResolvedCount`
- `actionSelectedUnitOverflowCount`
- `actionSelectedUnitIndices`
- `actionSelectedUnitMask`
- `actionSelectedUnitResolvedMask`
- `queueTypeId`
- `queueUpdateTypeId`
- `quantity`
- `itemNameToken`
- `itemType`
- `itemCost`
- `buildingNameToken`
- `buildingType`
- `buildingCost`
- `buildingTile`
- `objectEntityIndex`
- `objectNameToken`
- `objectType`
- `superWeaponTypeId`
- `superWeaponTile`
- `superWeaponTile2`
- `pingTile`

These are dense fixed-shape numeric sections, not one-hot expansions yet.

## Entity-Relative Indices

Selections and object targets are encoded relative to the current visible entity tensor whenever possible.

That means:

- selected units become indices into the current `entityFeatures` rows
- target objects become indices into the current entity tensor when visible

If an object cannot be resolved into the current visible entity tensor, the index falls back to `-1`.

To make that less brittle, the dataset also stores:

- masks
- resolved-count fields
- name tokens for relevant objects

## Vocabularies

The current dataset builder creates one shared name vocabulary per extraction run.

That vocabulary is used for:

- visible entity names
- target object names
- queue item names
- building names
- sell / repair object names

This is good enough for per-replay extraction and debugging, but a later dataset-wide pass should stabilize the vocabulary across many replays.

## Output Format

The CLI currently writes JSON.

Important detail:

- the arrays are tensor-shaped numeric data
- but they are not yet framework-native `.pt` or `.npz` files

That choice is deliberate for this phase because the current workspace does not have `torch` or `numpy` installed, and `py-chronodivide` is meant to stay reusable rather than tie itself to one framework immediately.

## Example

```powershell
node .\packages\py-chronodivide\extract_sl_tensors.mjs `
  --replay .\packages\chronodivide-bot-sl\ladder_replays_top50\00758dde-b725-4442-ae8f-a657069251a0.rpl `
  --data-dir d:\workspace\ra2-headless-mix `
  --player mecharse `
  --max-actions 128 `
  --max-entities 128 `
  --max-selected-units 64 `
  --spatial-size 32 `
  --minimap-size 64 `
  --output .\packages\py-chronodivide\sl_tensors_00758dde.json
```

## Current Caveats

- This is a per-extraction-run JSON tensor dataset, not the final multi-replay shard writer.
- Flattened tensors are optional and mainly meant as a bridge to later binary writers.
- Name vocabularies are not stable across separate extraction runs yet.
- Some selections or targets may not resolve into the current entity tensor if they fall outside the visible or padded entity set.
