# RA2 SL Label Layout V1 Implementation Checklist

This checklist turns [RA2_SL_LABEL_LAYOUT_V1.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V1.md) into concrete implementation steps.

The goal is to move from the current coarse RA2 label surface to a mini-AlphaStar-style label layout with:

- fine-grained `action_type`
- compact canonical label storage
- derived training heads
- action-type-driven masking

## Scope

This checklist is about the label side only.

It does not include:

- feature schema changes
- winner-only replay policy
- binary transport changes
- training-loop loss implementation

## Status Legend

- `[x]` already done
- `[~]` partially done
- `[ ]` not done yet

## Current Starting Point

- `[x]` Replay decoding is already available in `py-chronodivide`.
- `[x]` Structured action fields already exist for:
  - `rawActionId`
  - `actionFamilyId`
  - `orderTypeId`
  - `targetModeId`
  - `queue`
  - `actionSelectedUnitIds`
  - `targetObjectId`
  - `targetTile`
  - `queueUpdateTypeId`
  - `itemName`
  - `buildingName`
  - `superWeaponTypeId`
  - `superWeaponTile`
  - `superWeaponTile2`
- `[x]` The Python transformer already saves structured label sections and flat label tensors.
- `[x]` mAS-style action downsampling already exists, but it still keys off the current coarse action surface.
- `[x]` The main saved label tensor now uses the compact V1 canonical label layout.
- `[~]` Legacy coarse label fields are still used internally for filtering and feature-context rewrite during the migration period.

## Phase 1: Freeze The V1 Contract

- `[x]` Freeze the canonical V1 label section names:
  - `actionTypeId`
  - `delayBin`
  - `queue`
  - `unitsIndices`
  - `unitsMask`
  - `unitsResolvedMask`
  - `targetEntityIndex`
  - `targetEntityResolved`
  - `targetLocation`
  - `targetLocationValid`
  - `targetLocation2`
  - `targetLocation2Valid`
  - `quantity`

- `[x]` Freeze the initial default constants:
  - `DELAY_BINS = 128`
  - `MAX_SELECTED_UNITS = 64`
  - unresolved integer sentinel `-1`

- `[ ]` Decide whether `quantity` stays raw integer-valued in V1 or gets clipped into a fixed bucket count.
  Recommended first pass: keep it as integer-valued canonical storage and bucket later only if needed.

## Phase 2: Keep `py-chronodivide` Generic

- `[x]` Keep generic replay decoding in `py-chronodivide`.
- `[ ]` Confirm that no fine-grained `action_type` id mapping is pushed down into `py-chronodivide`.
  `py-chronodivide` should keep emitting generic decoded fields, not project-specific action vocab ids.

- `[ ]` Confirm the following fields are reliably present in the extracted raw samples from `py-chronodivide`:
  - `rawActionId`
  - `delayToNextAction`
  - `queue`
  - `actionSelectedUnitIds`
  - `selectionBeforeActionIds`
  - `orderTypeId`
  - `targetModeId`
  - `targetObjectId`
  - `targetTile`
  - `targetIsOre`
  - `queueUpdateTypeId`
  - `quantity`
  - `itemName`
  - `buildingName`
  - `objectId`
  - `superWeaponTypeId`
  - `superWeaponTile`
  - `superWeaponTile2`

- `[ ]` If any of the above are missing or unstable on edge cases, patch `py-chronodivide` first and keep the fix generic.

## Phase 3: Build The Fine-Grained `action_type` Vocabulary

- `[x]` Add a builder in `chronodivide-bot-sl` that converts one decoded raw sample into one action-type key string.

- `[x]` Implement the V1 mapping rules:
  - `SelectUnitsAction` -> `SelectUnits`
  - `OrderUnitsAction` -> `Order::<orderType>::<targetMode>`
  - `UpdateQueueAction` -> `Queue::<queueUpdateType>::<itemName>`
  - `PlaceBuildingAction` -> `PlaceBuilding::<buildingName>`
  - `ActivateSuperWeaponAction` -> `ActivateSuperWeapon::<superWeaponType>`
  - `SellObjectAction` -> `SellObject`
  - `ToggleRepairAction` -> `ToggleRepair`
  - `ResignGameAction` -> `ResignGame`

- `[x]` Decide and freeze the exact spelling policy for action-type keys.
  Recommendation:
  - ASCII only
  - stable `::` separators
  - no spaces
  - use the decoded semantic names, not raw numeric ids, in the key string

- `[x]` Build a deterministic action vocabulary manifest:
  - `actionTypeId -> actionTypeName`
  - `actionTypeName -> actionTypeId`
  - action counts across the processed replay set

- `[ ]` Decide the reserved ids policy.
  Recommendation:
  - `0` for first real action type if there is no unknown bucket
  - add an explicit `<unk>` id only if the transformer must support partial vocab reuse across shards

- `[x]` Make `actionTypeId` stable across a transform run.
  Current implementation assigns run-global ids in first-seen replay order and writes the full global vocabulary to the run manifest.

## Phase 4: Replace The Current Canonical Label Sections

- `[x]` In the Python transformer, stop treating these as the main final label sections:
  - `rawActionId`
  - `actionFamilyId`
  - `orderTypeId`
  - `targetModeId`
  - `queueTypeId`
  - `queueUpdateTypeId`
  - `itemNameToken`
  - `buildingNameToken`
  - `superWeaponTypeId`
  - `targetNameToken`

- `[x]` Replace the main label schema with the V1 canonical sections:
  - `actionTypeId`
  - `delayBin`
  - `queue`
  - `unitsIndices`
  - `unitsMask`
  - `unitsResolvedMask`
  - `targetEntityIndex`
  - `targetEntityResolved`
  - `targetLocation`
  - `targetLocationValid`
  - `targetLocation2`
  - `targetLocation2Valid`
  - `quantity`

- `[x]` Keep the current richer decoded fields as debug or audit metadata, not as the primary training label vector.

- `[x]` Preserve backward compatibility for a short transition period if needed:
  - old label sections only in metadata or debug output
  - new V1 label sections in the main flat label tensor

## Phase 5: Implement Canonical Section Builders

- `[x]` Implement `actionTypeId` construction from the new vocabulary builder.

- `[x]` Implement `delayBin` conversion from `delayToNextAction`.
  Recommendation:
  - clip to `DELAY_BINS - 1`
  - keep `-1` only for missing values such as terminal no-next-action cases

- `[x]` Implement `queue` as:
  - `0` or `1` when semantically used
  - `-1` when unused

- `[x]` Implement `unitsIndices` from:
  - `actionSelectedUnitIds` for `SelectUnitsAction`
  - `selectionBeforeActionIds` for `OrderUnitsAction`

- `[x]` Implement `unitsMask`:
  - `1` for filled positions
  - `0` for padding

- `[x]` Implement `unitsResolvedMask` based on whether each unit id resolves into the current entity tensor.

- `[x]` Implement `targetEntityIndex` from resolved target object ids.

- `[x]` Implement `targetEntityResolved`:
  - `1` when the target object resolves into the current entity tensor
  - `0` otherwise

- `[x]` Implement `targetLocation`:
  - use `[x, y]` tile coordinates
  - use `[-1, -1]` when absent

- `[x]` Implement `targetLocationValid`.

- `[x]` Implement `targetLocation2` and `targetLocation2Valid`.

- `[x]` Implement `quantity`:
  - raw integer from queue updates
  - `-1` when unused

## Phase 6: Build Action-Type Metadata For Masking

- `[x]` Add an action-type metadata table keyed by `actionTypeId`.

- `[x]` Each action type should expose at least these semantic mask flags:
  - `usesQueue`
  - `usesUnits`
  - `usesTargetEntity`
  - `usesTargetLocation`
  - `usesTargetLocation2`
  - `usesQuantity`

- `[x]` Build the metadata table at the same time the action vocabulary is built.

- `[x]` Save that metadata into the shard manifest so later training code can mask losses without re-deriving the action semantics.

## Phase 7: Separate Semantic Masks From Resolution Masks

- `[x]` Keep semantic masks derived from `actionTypeId`.

- `[x]` Keep resolution masks derived from replay-time alignment results:
  - `unitsResolvedMask`
  - `targetEntityResolved`
  - `targetLocationValid`
  - `targetLocation2Valid`

- `[ ]` Document and enforce the intended training behavior:
  - semantic mask says whether the head matters for this action
  - resolution mask says whether the label could actually be supervised from the current tensor alignment

- `[ ]` Decide the exact supervision policy for unresolved targets.
  Recommendation:
  - do not backpropagate `target_entity` when unresolved
  - still allow `target_location` supervision when a location exists

## Phase 8: Update Flat Label Ordering And Schema Metadata

- `[x]` Define the flat V1 label order explicitly and keep it stable.

- `[x]` Update the transformer's schema object to describe:
  - V1 label section names
  - shapes
  - dtypes
  - flat offsets

- `[x]` Update `.sections.pt` writing so the structured sidecar uses the V1 section names.

- `[x]` Update `.meta.json` writing so metadata includes:
  - action vocabulary
  - action-type mask metadata
  - label section offsets
  - delay bin count
  - selected-unit cap

- `[x]` Save the full run-global action vocabulary in `manifest.json`.

## Phase 9: Update The Action Filter To Use The New Surface

- `[x]` Decide whether the current mAS-style downsampling should still operate on coarse categories or move to the new fine-grained action surface.
  Current implementation keeps coarse behavior conceptually, but now derives the buckets from `actionTypeNameV1` when available.

- `[x]` Recommended V1 behavior:
  - keep the filter policy conceptually coarse
  - compute the coarse filter bucket from `actionTypeName`
  - continue downsampling:
    - `SelectUnits`
    - move-like orders
    - gather orders
    - attack-like orders

- `[x]` Stop depending on `actionFamilyId` as the main future-facing label field.
  The filter now prefers the V1 action names; `actionFamilyId` remains only as legacy support during migration.

## Phase 10: Add Derived Training-Head Helpers

- `[ ]` Add helper code that expands canonical labels into model-ready targets.

- `[ ]` Add helper expansion for:
  - `delayBin -> delay one-hot`
  - `queue -> queue one-hot`
  - `unitsIndices -> selected-units target`
  - `targetEntityIndex -> entity one-hot`
  - `targetLocation -> spatial one-hot`
  - `targetLocation2 -> second spatial one-hot`

- `[ ]` Keep this expansion separate from the canonical stored layout.
  That lets future model variants consume compact indices directly if they want to use embedding or cross-entropy targets.

## Phase 11: Validation

- `[x]` Add a smoke test on one replay and one player.
  Validate:
  - the new V1 label sections exist
  - flat label length matches the schema
  - `actionTypeId` is never missing for kept actions

- `[x]` Add a two-replay validation across different maps.
  Validate:
  - the action vocabulary is stable
  - no map-specific label section breaks occur
  - `targetLocation2` only appears on the expected actions

- `[ ]` Add a label-distribution audit.
  Validate:
  - top `actionTypeName` counts
  - fraction of samples using each optional head
  - fraction of unresolved `targetEntity`
  - fraction of unresolved `units`

- `[ ]` Add targeted sanity checks for representative actions:
  - `SelectUnits`
  - `Order::Move::tile`
  - `Order::Attack::object`
  - `Order::Gather::ore_tile`
  - `Queue::Add::<item>`
  - `PlaceBuilding::<building>`
  - `ActivateSuperWeapon::<type>`
  - `SellObject`
  - `ToggleRepair`

## Phase 12: Documentation

- `[ ]` Update the transformer docs to point to the new V1 label layout.

- `[ ]` Update any schema notes that still describe `rawActionId` and `actionFamilyId` as the main training labels.

- `[ ]` Add one short note to the transformer README or module docstring that explains:
  - canonical storage is compact
  - training heads are derived later
  - audit metadata remains available separately

## File Touchpoints

Expected main code touchpoints:

- `D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\transform_replay_data.py`
  - main label schema replacement
  - vocabulary building
  - flat label rebuilding
  - manifest metadata
  - filter integration

- `D:\workspace\supalosa-chronodivide-bot\packages\py-chronodivide\sl_dataset.mjs`
  - only if a generic decoded field is missing or unstable

- `D:\workspace\supalosa-chronodivide-bot\packages\py-chronodivide\labels.mjs`
  - only if generic replay action decoding needs improvement

## Practical Implementation Order

Recommended order:

1. Freeze V1 names and defaults.
2. Implement the fine-grained `action_type` vocabulary builder.
3. Implement the new canonical label section builders.
4. Replace the main label schema in the Python transformer.
5. Add action-type mask metadata.
6. Update filtering to use the new surface.
7. Run smoke tests and distribution audits.
8. Update docs and remove transitional old-label dependencies.

## Definition Of Done

V1 is done when:

- the main saved label tensor uses the new canonical V1 sections
- `actionTypeId` replaces coarse `actionFamilyId` as the main action head
- semantic masks are driven by `actionTypeId`
- unresolved entity and unit alignments are represented explicitly
- the transformer can produce stable shards on at least two real replays from different maps
- the old coarse label fields remain only as optional metadata or debug output
