# RA2 SL Label Layout V2 Implementation Checklist

This checklist turns [RA2_SL_LABEL_LAYOUT_V2.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V2.md) into concrete implementation work.

The goal is to move from the current fine-grained V1 action-type space toward a hierarchical command label with:

- command-level selection folding
- item-level queue supervision only in the main action space
- `Hold` / `Resume` excluded from the main learnable policy target for the current stage
- smaller, cleaner top-level classification problems

## Scope

This checklist is about the V2 label contract and the downstream migration work it will require.

It includes:

- replay-to-label conversion
- canonical stored label sections
- derived training targets
- action-space changes needed by the model and trainer

It does not include:

- feature-schema redesign
- replay filtering policy
- evaluation protocol changes unrelated to label layout

## Status Legend

- `[x]` already done
- `[~]` partially done
- `[ ]` not done yet

## Current Starting Point

- `[x]` V1 canonical label layout is implemented and trainable.
- `[x]` Replay decoding already exposes the raw structured fields needed to build V2.
- `[x]` V2 design note exists.
- `[x]` V2 design now explicitly disables `Hold` / `Resume` in the main learnable action space.
- `[x]` A V2 command-stream preview builder exists in the transformer metadata path.
- `[x]` A parallel V2 canonical label builder exists.
- `[x]` Parallel V2 training-target derivation exists.
- `[x]` A first debug V2 dataset/model/trainer path exists in parallel with V1.

## Phase 1: Freeze The V2 Contract

- `[ ]` Freeze the current-stage V2 top-level action families:
  - `Order`
  - `Queue`
  - `PlaceBuilding`
  - `ActivateSuperWeapon`
  - `SellObject`
  - `ToggleRepair`
  - `ResignGame`

- `[ ]` Freeze the current-stage V2 queue policy:
  - keep `Add`, `Cancel`, `AddNext`
  - exclude `Hold`, `Resume` from the main policy stream
  - preserve `Hold`, `Resume` in raw replay metadata only

- `[ ]` Freeze the command-level selection policy:
  - `SelectUnitsAction` does not emit a main policy sample
  - later command samples use `commandedUnits = selectionBeforeAction`

- `[ ]` Freeze the canonical V2 section names.
  Recommended current-stage set:
  - `actionFamilyId`
  - `delayBin`
  - `orderTypeId`
  - `targetModeId`
  - `queueFlag`
  - `queueUpdateTypeId`
  - `buildableObjectToken`
  - `superWeaponTypeId`
  - `commandedUnitsIndices`
  - `commandedUnitsMask`
  - `commandedUnitsResolvedMask`
  - `targetEntityIndex`
  - `targetEntityResolved`
  - `targetLocation`
  - `targetLocationValid`
  - `targetLocation2`
  - `targetLocation2Valid`
  - `quantity`

## Phase 2: Keep `py-chronodivide` Generic

- `[ ]` Keep raw replay decoding generic in `py-chronodivide`.
- `[ ]` Do not push V2-specific hierarchical ids down into `py-chronodivide`.
- `[ ]` Confirm the raw extracted fields remain sufficient for V2:
  - selection before action
  - raw action family
  - order type
  - target mode
  - queue update type
  - queue type
  - item name
  - building name
  - super-weapon type
  - quantity

## Phase 3: Implement V2 Replay Filtering And Folding

- `[x]` Add a V2 command-stream preview builder in `chronodivide-bot-sl`.
- `[x]` Drop standalone `SelectUnitsAction` from the previewed main policy stream.
- `[x]` Fold current selection into later preview command samples as `commandedUnits` metadata.
- `[x]` Exclude `UpdateQueueAction(Hold)` from the previewed main policy stream.
- `[x]` Exclude `UpdateQueueAction(Resume)` from the previewed main policy stream.
- `[x]` Preserve excluded queue actions in audit/debug metadata.
- `[x]` Recompute `delayToNextCommand` / `delayBin` on the kept V2 preview stream rather than the raw V1 action stream.
- `[x]` Replace the preview-only path with real V2 canonical stored labels in a parallel sidecar path.

## Phase 4: Build The Hierarchical V2 Vocabularies

- `[x]` Build stable ids for:
  - `actionFamilyId`
  - `orderTypeId`
  - `targetModeId`
  - `queueUpdateTypeId`
  - `buildableObjectToken`
  - `superWeaponTypeId`

- `[x]` Freeze the current-stage `queueUpdateTypeId` policy:
  - `Add`
  - `Cancel`
  - `AddNext`

- `[x]` Confirm `queueTypeId` is not a main supervised head in current-stage V2.
- `[x]` Keep `queueTypeId` only in auxiliary metadata if helpful for audit.

## Phase 5: Replace V1 Canonical Sections With V2 Sections

- `[x]` Add a new V2 label-layout version marker.
- `[x]` Build V2 canonical sections in the Python transformer as a parallel sidecar path.
- `[ ]` Stop using V1 `actionTypeId` as the main target when V2 mode is selected.
- `[x]` Replace V1 `units*` fields with V2 `commandedUnits*` fields in the parallel V2 sidecar.
- `[ ]` Keep V1 loading available as a legacy/debug path during migration.

## Phase 6: Derived Training Targets

- `[x]` Add V2 training-target derivation for:
  - `actionFamily`
  - `delay`
  - `orderType`
  - `targetMode`
  - `queueFlag`
  - `queueUpdateType`
  - `buildableObject`
  - `superWeaponType`
  - `commandedUnits`
  - `targetEntity`
  - `targetLocation`
  - `targetLocation2`
  - `quantity`

- `[x]` Add semantic loss masks for the V2 heads.
- `[x]` Confirm `quantity` is active only for `Queue Add/Cancel/AddNext`.
- `[x]` Confirm excluded `Hold` / `Resume` never appear as supervised policy rows.

## Phase 7: Model And Trainer Migration

- `[x]` Add V2 model-head support for hierarchical labels.
- `[x]` Replace the V1 flat `actionType` head with a small `actionFamily` head in a debug V2 model path.
- `[x]` Add family-specific heads:
  - `orderType`
  - `targetMode`
  - `queueUpdateType`
  - `buildableObject`
  - `superWeaponType`

- `[x]` Update losses and metrics for the V2 head set.
- `[x]` Update free-running evaluation to report V2 hierarchical metrics.
- `[x]` Keep the V1 trainer path untouched while the V2 path is still experimental.

## Phase 8: Validation

- `[x]` Run a replay-backed smoke transform for V2.
- `[x]` Audit that `SelectUnitsAction` rows are absent from the main V2 policy stream.
- `[x]` Audit that `Hold` / `Resume` rows are absent from the main V2 policy stream.
- `[x]` Audit that excluded actions remain visible in debug metadata if desired.
- `[x]` Compare V1 vs V2 action-family frequencies on the same replay slice.
- `[x]` Train a tiny V2 debug model and confirm:
  - stable loss
  - correct masking
  - commanded-units supervision works
  - queue actions are cleaner than V1
- `[x]` Run a small real Arab Pinch Point winner-only comparison slice through both V1 and the V2 debug path.
- `[x]` Confirm the first cross-replay V2 trainer can batch shards with different replay-local `buildableObject` widths.
- `[x]` Stabilize cross-replay V2 vocabularies, especially `buildableObjectToken`, so V2 no longer relies on replay-local widths.
- `[x]` Add a small V2 comparison report that tracks:
  - V1 `actionType` free-running accuracy
  - V2 `actionFamily` free-running accuracy
  - V2 per-family confusion, especially `Order` vs `Queue` vs `PlaceBuilding`
- `[x]` Add a first V2 action-family weighting pass to the debug trainer.
- `[x]` Add a family-balanced sampling option to the V2 debug trainer.
- `[~]` On the small winner-only slice, family-balanced sampling partially breaks the pure `Order` collapse:
  - V2 now emits some real `Queue` predictions in free-running eval
  - but `PlaceBuilding`, `ActivateSuperWeapon`, and `SellObject` still collapse into `Order`

## Recommended Immediate Next Step

- `[ ]` Run the stable-vocabulary V2 debug path on a larger Arab Pinch Point winner-only corpus so the family head is not judged only on a 5-replay slice.
- `[ ]` Decide whether family-balanced sampling should become the default V2 debug setting.
- `[ ]` Add per-family V2 checkpoint selection beyond `val_loss`, especially:
  - best free-running `actionFamilyAccuracy`
  - best free-running `Queue` recall
- `[ ]` Add stronger curriculum or balancing for the still-missing V2 families:
  - `PlaceBuilding`
  - `ActivateSuperWeapon`
  - `SellObject`
- `[ ]` Decide whether to promote V2 beyond the debug path after the larger comparison run.
