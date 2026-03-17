# RA2 SL Label Layout V1

This note defines the proposed supervised-learning label layout for Red Alert 2 replay imitation in `chronodivide-bot-sl`.

It is meant to be the RA2 analogue of mini-AlphaStar's action-label split:

- `action_type`
- `delay`
- `queue`
- `units`
- `target_unit`
- `target_location`

The key difference is that RA2 benefits from a slightly different action surface and one extra spatial head.

## Goal

Use a label layout that:

- matches the spirit of mini-AlphaStar
- keeps the main action head fine-grained
- keeps arguments separate from the action head
- avoids redundant supervision
- stays compact on disk
- can be expanded into model-ready one-hot or spatial targets later

## Design Principles

1. `action_type` should mean the ability or command family actually chosen.
2. Selected units and targets should stay outside `action_type`.
3. Labels should be stored canonically in compact index or coordinate form.
4. One-hot and spatial target planes should be derived later for model training.
5. Masking should be driven mostly by `action_type`, with extra resolution masks when the entity target cannot be aligned to the current entity tensor.

## Recommended Training Heads

V1 recommends these supervised heads:

- `action_type`
- `delay`
- `queue`
- `units`
- `target_entity`
- `target_location`
- `target_location_2`
- `quantity`

### Why these heads

- `action_type`
  The main fine-grained RA2 ability head.
- `delay`
  RTS control is partly about timing, not just command choice.
- `queue`
  Some order actions differ meaningfully between queued and immediate execution.
- `units`
  RA2 control is very selection-heavy, so the acted-on unit set matters.
- `target_entity`
  Needed for attack-object, repair-object, sell-object, capture-object, and similar actions.
- `target_location`
  Needed for move, attack-move, gather, building placement, and many super-weapon actions.
- `target_location_2`
  Needed because some RA2 actions expose a second tile target.
- `quantity`
  Needed for queue update actions such as add, cancel, and hold.

## Heads We Should Not Keep As Main Prediction Heads

These are useful for replay auditing and debugging, but should not usually remain separate model heads if `action_type` is fine-grained:

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
- `itemCost`
- `buildingCost`
- `objectType`

Reason:

- most of these are either already encoded in the proposed fine-grained `action_type`
- or they are metadata useful for analysis rather than core imitation targets

## Fine-Grained `action_type` Vocabulary

The RA2 equivalent of mini-AlphaStar's raw ability space should be built from the already decoded action fields in `py-chronodivide`.

### Proposed construction rules

- `SelectUnitsAction`
  - `SelectUnits`
- `OrderUnitsAction`
  - `Order::<orderType>::<targetMode>`
- `UpdateQueueAction`
  - `Queue::<queueUpdateType>::<itemName>`
- `PlaceBuildingAction`
  - `PlaceBuilding::<buildingName>`
- `ActivateSuperWeaponAction`
  - `ActivateSuperWeapon::<superWeaponType>`
- `SellObjectAction`
  - `SellObject`
- `ToggleRepairAction`
  - `ToggleRepair`
- `ResignGameAction`
  - `ResignGame`

### Examples

- `Order::Move::tile`
- `Order::Attack::object`
- `Order::AttackMove::tile`
- `Order::Gather::ore_tile`
- `Order::Deploy::none`
- `Order::Repair::object`
- `Queue::Add::GrizzlyTank`
- `Queue::Cancel::GrizzlyTank`
- `PlaceBuilding::PowerPlant`
- `ActivateSuperWeapon::LightningStorm`

### Important rule

`queue`, selected units, and exact targets should not be folded into `action_type`.

That keeps the label structure close to mini-AlphaStar:

- action head decides the ability
- argument heads decide how that ability is applied

## Canonical Stored Label Layout

This is the recommended compact on-disk layout for each action sample.

Shapes below are per-sample, not batched.

Assumptions:

- `MAX_SELECTED_UNITS` is configurable, recommended default `64`
- entity indices refer to the current padded entity tensor
- missing or unresolved values use `-1`

### Core sections

- `actionTypeId`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: fine-grained RA2 action id

- `delayBin`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: clipped or bucketed delay-to-next-kept-action
  - recommendation: start with `128` bins for parity with mini-AlphaStar-style delay handling

- `queue`
  - shape: `[1]`
  - dtype: `int32`
  - values: `0` or `1`, `-1` when unused

- `unitsIndices`
  - shape: `[MAX_SELECTED_UNITS]`
  - dtype: `int32`
  - meaning: padded entity indices for the acted-on unit set

- `unitsMask`
  - shape: `[MAX_SELECTED_UNITS]`
  - dtype: `int32`
  - meaning: `1` for filled unit slots, `0` for padding

- `unitsResolvedMask`
  - shape: `[MAX_SELECTED_UNITS]`
  - dtype: `int32`
  - meaning: `1` when the selected unit resolved into the current entity tensor, `0` otherwise

- `targetEntityIndex`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: target object index in the current entity tensor

- `targetEntityResolved`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: `1` if the entity target resolved, `0` otherwise

- `targetLocation`
  - shape: `[2]`
  - dtype: `int32`
  - meaning: primary tile target as `[x, y]`

- `targetLocationValid`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: `1` if `targetLocation` is present, `0` otherwise

- `targetLocation2`
  - shape: `[2]`
  - dtype: `int32`
  - meaning: secondary tile target as `[x, y]`

- `targetLocation2Valid`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: `1` if `targetLocation2` is present, `0` otherwise

- `quantity`
  - shape: `[1]`
  - dtype: `int32`
  - meaning: queue update quantity, `-1` when unused
  - hard V1 policy: store replay quantity as raw integer; do not bucket in canonical V1

## Derived Training Targets

The canonical stored labels above are compact. For training, they can be expanded into model-ready targets.

### mAS-like derived heads

- `action_type`
  - derived from `actionTypeId`
  - categorical or one-hot over the fine-grained action vocabulary

- `delay`
  - derived from `delayBin`
  - categorical or one-hot over `DELAY_BINS`

- `queue`
  - derived from `queue`
  - binary categorical over `{0, 1}`

- `units`
  - derived from `unitsIndices`, `unitsMask`, `unitsResolvedMask`
  - either:
    - padded entity-index sequence with masks
    - or mAS-style one-hot `[MAX_SELECTED_UNITS, MAX_ENTITIES]`

- `target_entity`
  - derived from `targetEntityIndex`, `targetEntityResolved`
  - either:
    - scalar entity index
    - or one-hot over entity slots

- `target_location`
  - derived from `targetLocation`, `targetLocationValid`
  - either:
    - coordinate pair
    - or one-hot spatial plane at the chosen training resolution

- `target_location_2`
  - derived from `targetLocation2`, `targetLocation2Valid`
  - same encoding options as `target_location`

- `quantity`
  - derived from `quantity`
  - hard V1 policy: remains integer-valued as `quantityValue`
  - any bucketing is deferred to a later label version or model-specific derived head

## Action-Type-Driven Masking

Like mini-AlphaStar, most label heads should be masked according to the chosen `action_type`.

### Always active

- `action_type`
- `delay`

### Active for `SelectUnits`

- `units`

Inactive:

- `queue`
- `target_entity`
- `target_location`
- `target_location_2`
- `quantity`

### Active for `Order::<orderType>::<targetMode>`

- `units`
- `queue` only if the specific order type supports queueing
- `target_entity` when `targetMode == object`
- `target_location` when `targetMode == tile`
- `target_location` when `targetMode == ore_tile`

Inactive:

- `target_location_2`
- `quantity`

### Active for `Queue::<queueUpdateType>::<itemName>`

- `quantity`

Inactive:

- `queue`
- `units`
- `target_entity`
- `target_location`
- `target_location_2`

### Active for `PlaceBuilding::<buildingName>`

- `target_location`

Inactive:

- `queue`
- `units`
- `target_entity`
- `target_location_2`
- `quantity`

### Active for `ActivateSuperWeapon::<superWeaponType>`

- `target_location`
- `target_location_2` only for actions that actually expose a second tile

Inactive:

- `queue`
- `units`
- `target_entity`
- `quantity`

### Active for `SellObject`

- `target_entity`

Inactive:

- `queue`
- `units`
- `target_location`
- `target_location_2`
- `quantity`

### Active for `ToggleRepair`

- `target_entity`

Inactive:

- `queue`
- `units`
- `target_location`
- `target_location_2`
- `quantity`

### Active for `ResignGame`

All argument heads inactive:

- `queue`
- `units`
- `target_entity`
- `target_location`
- `target_location_2`
- `quantity`

## Resolution Masks vs Semantic Masks

There are two different masking concepts and we should keep them separate.

### Semantic mask

Derived from `action_type`.

Examples:

- `Order::Attack::object` semantically uses `target_entity`
- `PlaceBuilding::PowerPlant` semantically uses `target_location`

### Resolution mask

Derived from whether the replay-time target could be aligned to the current feature tensor.

Examples:

- a target object exists in the replay state but was truncated out of the entity tensor
- a selected unit exists but was not included in the current padded entity set

This is why V1 stores:

- `unitsResolvedMask`
- `targetEntityResolved`
- `targetLocationValid`
- `targetLocation2Valid`

Training should usually require both:

- semantic head is active
- target resolved or valid

## Hard V1 Supervision Policy

These are fixed V1 rules, not open design questions.

### Quantity

- canonical `quantity` stores the replay quantity as raw integer
- canonical `quantity` stores `-1` when unused
- derived training target keeps raw integer `quantityValue`
- V1 does not bucket quantity
- if quantity bucketing is later needed, that should happen in a new label version or a model-specific derived target layer

### Unresolved target and unit supervision

- `queue`
  - supervise only when the action type semantically uses queue and queue is in `{0, 1}`
- `units`
  - supervise only positions where:
    - the action type semantically uses units
    - `unitsMask == 1`
    - `unitsResolvedMask == 1`
- `target_entity`
  - supervise only when:
    - the action type semantically uses `target_entity`
    - `targetEntityResolved == 1`
- `target_location`
  - supervise when:
    - the action type semantically uses `target_location`
    - `targetLocationValid == 1`
  - entity-resolution failure does not suppress valid location supervision
- `target_location_2`
  - supervise when:
    - the action type semantically uses `target_location_2`
    - `targetLocation2Valid == 1`
- `quantity`
  - supervise only when:
    - the action type semantically uses `quantity`
    - `quantity >= 0`

This is the hard V1 rule for training masks:

- semantic masks decide whether a head conceptually applies to the action type
- replay-time resolution / validity masks decide whether that head can actually receive supervision on this sample

## Current RA2-to-Head Mapping

This is how the existing decoded replay actions map into the V1 heads.

### `SelectUnitsAction`

- `action_type`: `SelectUnits`
- `units`: selected unit list

### `OrderUnitsAction`

- `action_type`: `Order::<orderType>::<targetMode>`
- `queue`: replay queue flag when supported
- `units`: current selected unit set
- `target_entity`: object target when present
- `target_location`: tile target when present

### `UpdateQueueAction`

- `action_type`: `Queue::<queueUpdateType>::<itemName>`
- `quantity`: replay quantity

### `PlaceBuildingAction`

- `action_type`: `PlaceBuilding::<buildingName>`
- `target_location`: building placement tile

### `ActivateSuperWeaponAction`

- `action_type`: `ActivateSuperWeapon::<superWeaponType>`
- `target_location`: primary tile
- `target_location_2`: secondary tile when present

### `SellObjectAction`

- `action_type`: `SellObject`
- `target_entity`: sold object

### `ToggleRepairAction`

- `action_type`: `ToggleRepair`
- `target_entity`: repaired object

### `ResignGameAction`

- `action_type`: `ResignGame`
- no argument heads

## What Stays As Audit Metadata

The following fields are still useful to save somewhere for debugging, replay audits, and vocabulary building:

- `tick`
- `playerId`
- `playerName`
- `rawActionId`
- `rawActionName`
- `actionFamilyId`
- `orderTypeId`
- `orderTypeName`
- `targetModeId`
- `targetModeName`
- `queueTypeId`
- `queueTypeName`
- `queueUpdateTypeId`
- `queueUpdateTypeName`
- `itemName`
- `buildingName`
- `superWeaponTypeId`
- `superWeaponTypeName`

But they should not all be treated as first-class model heads in V1.

## Suggested V1 Defaults

- `DELAY_BINS = 128`
- `MAX_SELECTED_UNITS = 64`
- store canonical labels as compact indices and coordinates
- derive one-hot and spatial heads during training or final tensorization
- exclude UI-only replay actions by default:
  - `NoAction`
  - `DropPlayerAction`
  - `PingLocationAction`

## Non-Goals For V1

V1 intentionally does not include:

- camera or click labels
- chat or taunt labels
- enemy-private information
- redundant heads that can be reconstructed from `action_type`

## Practical Summary

If we want RA2 labels to feel like mini-AlphaStar labels, the right mapping is:

- `action_type`: fine-grained RA2 ability id
- `delay`: next-action timing
- `queue`: queued vs immediate
- `units`: acted-on unit set
- `target_entity`: targeted object
- `target_location`: primary spatial target
- `target_location_2`: secondary spatial target when needed
- `quantity`: queue delta size when needed

That gives us a label layout that is:

- close in spirit to mAS
- natural for RA2 control
- compact to store
- flexible enough for later model variants
