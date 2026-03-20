# RA2 SL Label Layout V2

This note defines a simpler supervised-learning label layout for Red Alert 2 replay imitation in `chronodivide-bot-sl`.

Implementation tracking lives in [RA2_SL_LABEL_LAYOUT_V2_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V2_IMPLEMENTATION_CHECKLIST.md).

V2 keeps the useful parts of the V1 work, but changes the label surface in two important ways:

- it replaces the giant flat `action_type` problem with a hierarchical command label
- it uses the cleaner command-level treatment of selection: `SelectUnitsAction` is no longer a standalone main action target
- for the current stage, it disables `Hold` / `Resume` queue actions in the main learnable action space

The goal is to make rare actions easier to learn, reduce noisy label fragmentation, and move the dataset closer to the actual command semantics we want the policy to model.

## Goal

Use a label layout that:

- is easier to learn than a single fine-grained `actionTypeId`
- keeps arguments separate from command family and subtype
- treats queue actions according to the game API semantics
- removes standalone selection spam from the main policy target
- stays compact on disk
- remains easy to expand into model-ready targets later

## Main Differences From V1

V2 changes the label philosophy in four ways:

1. `actionFamilyId` becomes the primary top-level action head.
2. Queue actions are decomposed, but the current-stage policy keeps only item-level queue actions in the main action space.
3. `PlaceBuilding` and queue-item actions share a common `buildableObjectToken`.
4. `SelectUnitsAction` is removed from the main action stream and folded into a `commandedUnits` argument on the later command.

## Core Design Principles

1. The top-level head should predict a small command family, not hundreds of flat classes.
2. Family-specific subtype heads should activate only when semantically relevant.
3. Selected units should be modeled as the acted-on unit set for a command, not as a separate standalone action family.
4. Queue-level and item-level queue actions should not be forced into one label shape.
5. Canonical storage should stay compact and index-based.
6. Rich one-hot or spatial targets should remain derived training representations.
7. If queue-level actions are too noisy or ambiguous, they should be preserved for analysis but excluded from the main policy target until a stable normalization rule exists.

## Command-Level Selection Design

V2 adopts the cleaner command-level treatment of selection.

Hard policy:

- `SelectUnitsAction` is not a main supervised action in the V2 policy stream.
- The replay parser still tracks selection state exactly.
- When a later command is emitted, the current selection before that command becomes `commandedUnits`.
- Standalone selection-only actions are dropped from the main policy dataset.

This means the policy learns:

- `what command family was chosen`
- `which units the command was applied to`
- `what arguments were used`

instead of learning:

- `select units`
- then separately `issue command`

### Selection Folding Rules

Recommended folding behavior:

1. Replay walking still processes all raw actions in order.
2. `SelectUnitsAction` updates player selection state but does not emit a main policy sample.
3. `OrderUnitsAction`, `SellObjectAction`, and `ToggleRepairAction` emit a command sample with `commandedUnits = selectionBeforeAction`.
4. Queue, building, and super-weapon actions emit samples as usual, but do not use `commandedUnits`.
5. Delays are computed on the kept command stream, not the raw selection stream.

### Why This Is Better

- It removes the largest and noisiest action class from the main head.
- It matches the actual control decision better.
- It reduces exposure to repeated selection spam.
- It moves RA2 closer to the command-level modeling style we actually want.

## Recommended Main Training Heads

V2 recommends these main supervised heads:

- `actionFamilyId`
- `delayBin`
- `orderTypeId`
- `targetModeId`
- `queueFlag`
- `queueUpdateTypeId`
- `buildableObjectToken`
- `superWeaponTypeId`
- `commandedUnits`
- `targetEntity`
- `targetLocation`
- `targetLocation2`
- `quantity`

## Top-Level Action Families

Recommended main action families:

- `Order`
- `Queue`
- `PlaceBuilding`
- `ActivateSuperWeapon`
- `SellObject`
- `ToggleRepair`
- `ResignGame`

Not kept as main policy families:

- `SelectUnits`
- `Queue::Hold`
- `Queue::Resume`
- `PingLocation`
- `DropPlayer`
- `NoAction`

Those may still be preserved in auxiliary audit/debug metadata, but they should not drive the main policy objective.

## Family-Specific Meaning

### `Order`

Used for `OrderUnitsAction`.

Active heads:

- `actionFamilyId = Order`
- `orderTypeId`
- `targetModeId`
- `queueFlag`
- `commandedUnits`
- `targetEntity` if `targetModeId == object`
- `targetLocation` if `targetModeId in {tile, ore_tile}`

Inactive heads:

- `queueUpdateTypeId`
- `queueTypeId`
- `buildableObjectToken`
- `superWeaponTypeId`
- `targetLocation2`
- `quantity`

### `Queue`

Used for `UpdateQueueAction`.

Current-stage V2 policy:

- keep only item-level queue updates in the main policy dataset
- item-level queue updates are `Add`, `Cancel`, and `AddNext`
- `Hold` and `Resume` stay available in replay metadata and queue-state features, but do not become main supervised policy actions

Active heads for kept queue updates:

- `actionFamilyId = Queue`
- `queueUpdateTypeId in {Add, Cancel, AddNext}`
- `buildableObjectToken`
- `quantity`

Inactive heads for `Queue`:

- `commandedUnits`
- `targetEntity`
- `targetLocation`
- `targetLocation2`
- `queueFlag`
- `orderTypeId`
- `targetModeId`
- `superWeaponTypeId`

Deferred queue-level actions:

- `Hold`
- `Resume`

Recommended handling for those deferred actions:

1. Keep them in raw replay metadata for audit and UI-behavior analysis.
2. Preserve them in queue-state feature engineering if useful.
3. Exclude them from the main learnable action space until queue-intent normalization is implemented.

### `PlaceBuilding`

Used for `PlaceBuildingAction`.

Active heads:

- `actionFamilyId = PlaceBuilding`
- `buildableObjectToken`
- `targetLocation`

Inactive heads:

- `commandedUnits`
- `targetEntity`
- `targetLocation2`
- `queueUpdateTypeId`
- `queueTypeId`
- `quantity`
- `orderTypeId`
- `targetModeId`
- `superWeaponTypeId`

### `ActivateSuperWeapon`

Used for `ActivateSuperWeaponAction`.

Active heads:

- `actionFamilyId = ActivateSuperWeapon`
- `superWeaponTypeId`
- `targetLocation`
- `targetLocation2` when present

Inactive heads:

- `commandedUnits`
- `targetEntity`
- `queueFlag`
- `queueUpdateTypeId`
- `queueTypeId`
- `buildableObjectToken`
- `quantity`
- `orderTypeId`
- `targetModeId`

### `SellObject`

Used for `SellObjectAction`.

Active heads:

- `actionFamilyId = SellObject`
- `targetEntity`

Inactive heads:

- `commandedUnits`
- `targetLocation`
- `targetLocation2`
- `queueFlag`
- `queueUpdateTypeId`
- `queueTypeId`
- `buildableObjectToken`
- `quantity`
- `orderTypeId`
- `targetModeId`
- `superWeaponTypeId`

### `ToggleRepair`

Used for `ToggleRepairAction`.

Active heads:

- `actionFamilyId = ToggleRepair`
- `targetEntity`

Inactive heads:

- same inactive set as `SellObject`

### `ResignGame`

Used for `ResignGameAction`.

Active heads:

- `actionFamilyId = ResignGame`

All argument heads stay inactive.

## Canonical Stored Label Layout

Shapes below are per-sample.

Assumptions:

- `MAX_COMMANDED_UNITS` is configurable, recommended default `64`
- entity indices refer to the current padded entity tensor
- missing or inactive scalar values use `-1`

### Core Scalars

- `actionFamilyId`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: small family id

- `delayBin`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: clipped or bucketed delay to next kept command

- `orderTypeId`
  - shape: `[1]`
  - dtype: `int32`
  - active only for `Order`

- `targetModeId`
  - shape: `[1]`
  - dtype: `int32`
  - active only for `Order`

- `queueFlag`
  - shape: `[1]`
  - dtype: `int32`
  - values: `0` or `1`, `-1` when inactive
  - active only for `Order`

- `queueUpdateTypeId`
  - shape: `[1]`
  - dtype: `int32`
  - active only for `Queue`

- `buildableObjectToken`
  - shape: `[1]`
  - dtype: `int32`
  - active for `Queue Add/Cancel/AddNext` and `PlaceBuilding`
  - shared token space for buildable object names

- `superWeaponTypeId`
  - shape: `[1]`
  - dtype: `int32`
  - active only for `ActivateSuperWeapon`

- `quantity`
  - shape: `[1]`
  - dtype: `int32`
  - active only for `Queue Add/Cancel/AddNext`

### Commanded Units

- `commandedUnitsIndices`
  - shape: `[MAX_COMMANDED_UNITS]`
  - dtype: `int32`
  - meaning: padded entity indices of the units the command acts on

- `commandedUnitsMask`
  - shape: `[MAX_COMMANDED_UNITS]`
  - dtype: `int32`
  - meaning: `1` for filled slots, `0` for padding

- `commandedUnitsResolvedMask`
  - shape: `[MAX_COMMANDED_UNITS]`
  - dtype: `int32`
  - meaning: `1` when the selected unit resolved into the current entity tensor

Hard policy:

- this is active only for `Order`
- `SellObject` and `ToggleRepair` do not use `commandedUnits`
- no explicit `EOF` token is stored canonically
- any autoregressive `EOF` target remains derived later from the mask

### Target Heads

- `targetEntityIndex`
  - shape: `[1]`
  - dtype: `int32`

- `targetEntityResolved`
  - shape: `[1]`
  - dtype: `int32`

- `targetLocation`
  - shape: `[2]`
  - dtype: `int32`

- `targetLocationValid`
  - shape: `[1]`
  - dtype: `int32`

- `targetLocation2`
  - shape: `[2]`
  - dtype: `int32`

- `targetLocation2Valid`
  - shape: `[1]`
  - dtype: `int32`

## Shared Name Spaces

V2 should simplify naming instead of expanding separate token spaces.

Recommended shared vocabularies:

- `buildableObjectToken`
  - shared by queue item actions and building placement
- `queueTypeId`
  - six queue types from the game API
  - preserved for replay analysis and future queue-intent normalization
- `superWeaponTypeId`
  - super-weapon type enum

This means V2 no longer needs separate main prediction heads for:

- `itemNameToken`
- `buildingNameToken`

They are replaced by one `buildableObjectToken`.

## Family-Specific Masking Rules

Hard masking policy:

- inactive heads must store canonical missing values
- inactive heads must have zero loss mask
- masks should be driven by `actionFamilyId` and family-specific subtype values

Important queue rule:

- `Hold` and `Resume` are not main supervised actions in the current-stage V2 policy stream
- `Add`, `Cancel`, and `AddNext` must never activate `queueTypeId`

## Derived Training Targets

Recommended derived training heads:

- `action_family`
- `delay`
- `order_type`
- `target_mode`
- `queue_flag`
- `queue_update_type`
- `buildable_object`
- `super_weapon_type`
- `commanded_units`
- `target_entity`
- `target_location`
- `target_location_2`
- `quantity`

Optional auxiliary target:

- `flatActionTypeIdAux`

This auxiliary target can preserve compatibility with older analysis tools, but it should not be the primary policy head.

## Example Relabeling

### Example 1

Raw replay sequence:

1. `SelectUnitsAction`
2. `OrderUnitsAction(order=AttackMove, target=tile)`

V2 kept sample:

- `actionFamilyId = Order`
- `orderTypeId = AttackMove`
- `targetModeId = tile`
- `commandedUnits = selectionBeforeOrder`
- `targetLocation = tile`

No standalone `SelectUnits` sample is emitted.

### Example 2

Raw replay action:

- `UpdateQueueAction(updateType=Hold, queueType=Infantry, item=null)`

V2 current-stage handling:

- no main policy sample is emitted
- replay/audit metadata may still record:
  - `queueUpdateTypeId = Hold`
  - `queueTypeId = Infantry`

### Example 3

Raw replay action:

- `UpdateQueueAction(updateType=Add, item=HTNK, quantity=1)`

V2 kept sample:

- `actionFamilyId = Queue`
- `queueUpdateTypeId = Add`
- `buildableObjectToken = HTNK`
- `quantity = 1`

### Example 4

Raw replay action:

- `PlaceBuildingAction(building=NAWEAP, tile=[x, y])`

V2 kept sample:

- `actionFamilyId = PlaceBuilding`
- `buildableObjectToken = NAWEAP`
- `targetLocation = [x, y]`

## Things V2 Intentionally Does Not Solve

V2 is simpler, but it is still not the final end-state.

It does not yet require:

- a full command-level merge of queue/building UI context beyond the current action
- free-form macro planning labels
- richer target decomposition for super-weapons
- multi-step command grouping beyond selection folding

## Migration Notes

V2 is not checkpoint-compatible with V1.

Required migration work:

1. add a new label-layout version
2. regenerate tensors
3. update training-target derivation
4. update model heads and losses
5. exclude `Hold` / `Resume` from the main learnable action space
6. keep V1 loading only as a legacy/debug path

## Recommendation

Build V2 in two stages:

1. implement the hierarchical heads, with only item-level queue actions in the main policy stream
2. implement the command-level selection folding and remove `SelectUnitsAction` from the main policy stream

This is the simplest path that materially reduces label complexity while still keeping the dataset grounded in the replay semantics exposed by the game API.
