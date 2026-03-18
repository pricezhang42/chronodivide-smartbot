# RA2 SL Feature Layout V1

This note proposes a supervised-learning feature layout for Red Alert 2 that is:

- close in spirit to mini-AlphaStar
- compatible with the current `py-chronodivide` observation-safe extraction path
- informed by the upstream SC2 observation interfaces in `pysc2` and `s2client-proto`
- practical to implement incrementally in `chronodivide-bot-sl`

This is the feature-side counterpart to [RA2_SL_LABEL_LAYOUT_V1.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V1.md).

## Goal

The goal is not to copy SC2 feature names literally.

The goal is to give RA2 SL the same kinds of useful feature groups that mAS has:

- rich scalar and global context
- entity context
- spatial context
- previous-action context
- short action-history and build-history context

At the same time, the upstream SC2 interfaces are a useful design reference:

- `pysc2` exposes `available_actions` from observation-time abilities in [features.py](D:/workspace/pysc2/pysc2/lib/features.py)
- SC2 raw observations expose per-unit orders, rally targets, effects, and static map grids in [raw.proto](D:/workspace/s2client-proto/s2clientprotocol/raw.proto)
- SC2 UI observations expose production queues in [ui.proto](D:/workspace/s2client-proto/s2clientprotocol/ui.proto)

RA2 should copy the useful ideas, not the exact field names.

## Design Principles

1. Keep `py-chronodivide` generic and observation-safe.
2. Keep canonical feature storage compact and structured.
3. Add derived compact summaries when they reduce learning burden.
4. Prefer explicit strategic and global summaries over forcing the model to infer everything from raw visible entities.
5. Encode only information that is available from replay re-simulation without leaking hidden enemy state.
6. Treat `availableActionMask` like `pysc2` `available_actions`: it may be approximate, but it must be observation-driven and never omniscient.
7. Keep UI-heavy or low-value signals such as camera clicks and observer controls out of V1.

## V1 Top-Level Feature Sections

V1 recommends these main feature sections:

- `scalarCore`
- `lastActionContext`
- `currentSelection`
- `availableActionMask`
- `ownedCompositionBow`
- `enemyMemoryBow`
- `buildOrderTrace`
- `techState`
- `productionState`
- `superWeaponState`
- `entity`
- `spatial`
- `minimap`
- `mapStatic`

This looks larger than the current RA2 extractor, but many of these are small scalar or low-width structured sections.

## Section Definitions

### 1. `scalarCore`

This is the current scalar branch plus a few important additions.

Keep:

- time / tick / tick rate
- map width / height
- start position
- credits
- power totals / drain / margin / low-power
- radar disabled
- visible tile count / fraction
- coarse visible self / allied / enemy / neutral / other-hostile counts
- visible HP / purchase-value sums

Add:

- self side one-hot
- enemy side multi-hot union over opposing replay players
- self country one-hot
- enemy country multi-hot union over opposing replay players if replay-global metadata allows it
- number of owned production buildings by category
- number of owned tech buildings by category
- number of currently buildable placement options
- number of currently trainable production options

Why:

- this becomes the RA2 equivalent of the richer mAS scalar branch
- self / enemy side is the closest RA2 analogue to mAS `home_race` / `away_race`
- country identity adds sub-faction detail that matters for build options and bonuses

### 2. `lastActionContext`

Keep the current V1 feature-side temporal context:

- `delayFromPreviousAction`
- `lastActionTypeIdV1`
- `lastQueue`

This is already the right mAS-style analogue.

### 3. `currentSelection`

Keep the current selection tensors:

- `currentSelectionCount`
- `currentSelectionResolvedCount`
- `currentSelectionOverflowCount`
- `currentSelectionIndices`
- `currentSelectionMask`
- `currentSelectionResolvedMask`

Add compact derived selection-summary scalars:

- selected infantry count
- selected vehicle count
- selected aircraft count
- selected building count
- selected can-move flag
- selected can-attack flag
- selected can-deploy flag
- selected can-gather flag
- selected can-repair flag
- selected mixed-type flag

Why:

- raw selection tensors are useful
- compact capability summaries help action availability and head conditioning

### 4. `availableActionMask`

This is the highest-priority new branch.

Add a binary mask over the static SL action dict:

- shape: `[ACTION_TYPE_COUNT]`
- meaning: whether each fine-grained SL action type is currently available

V1 policy:

- this mask may be approximate, but it must never use hidden enemy state
- impossible actions should be masked out confidently
- uncertain actions may remain enabled rather than leaking information
- derive it the way `pysc2` derives `available_actions`: from current capabilities, not from an omniscient planner

Suggested decomposition:

- selection-driven order availability
- sidebar / queue action availability
- placement action availability
- super-weapon availability

Why:

- this is the clearest RA2 analogue to mAS `available_actions`

### 5. `ownedCompositionBow`

Add a self-only bag-of-words count vector over owned unit and building names.

Suggested split:

- `ownedUnitCountBow`
- `ownedBuildingCountBow`

Shape:

- fixed vocabulary over RA2 object and building names

V1 policy:

- use current owned state only
- do not limit this to visible units; self-owned state is known
- use a static ruleset-seeded vocabulary with an explicit unknown bucket
- in the current transformer implementation, store the split as a single
  `ownedCompositionBow` section with two rows: `units` and `buildings`

Why:

- this is the strongest cheap global summary missing from current RA2 features

### 6. `enemyMemoryBow`

Add a known-enemy-memory summary.

Suggested split:

- `seenEnemyBuildingCountBow`
- `seenEnemyUnitCountBow`
- `seenEnemyTechFlags`

V1 policy:

- derived only from observation history up to the current tick
- never from global enemy state

Why:

- this is the RA2 equivalent of enemy-upgrade and enemy-tech strategic context

### 7. `buildOrderTrace`

Add a fixed-length early-game build / production history.

Suggested shape:

- `[BUILD_ORDER_TRACE_LEN]`

Recommended contents:

- first `N` self production/build action types after replay start
- use the same static action dict ids where reasonable
- pad with `-1`

Why:

- this is the RA2 analogue to mAS `beginning_build_order`

### 8. `techState`

Add a compact self-tech and prerequisite block.

Suggested contents:

- owned prerequisite-building flags
- owned tech-building flags
- unlocked production-category flags
- unlocked special-tech flags
- power-satisfied-for-tech flags

Why:

- this is the RA2 analogue to mAS upgrade and tech-state features

### 9. `productionState`

Add a structured summary of current production queues.

Suggested contents:

- active production queue count by queue type
- current item under production by queue type
- current progress by queue type
- hold / paused flags
- queue occupancy / queued-count summaries
- construction-yard placement mode summary
- current rally summary for factories if available

Why:

- RA2 has a much more explicit queue-management loop than SC2 raw actions
- upstream SC2 also exposes production information through `ObservationUI.production`
- the model should know what is already being built or trained

### 10. `superWeaponState`

Add a dedicated global support-power branch.

Suggested contents:

- owned super-weapon presence flags
- ready / not-ready flags
- cooldown / charge progress
- active support-power availability flags

Why:

- this is strategically important and currently absent from features

### 11. `entity`

Keep the current visible-entity tensor:

- relation one-hot
- object-type one-hot
- position
- HP and HP ratio
- sight
- veteran level
- purchase value
- facing / turret facing
- velocity
- foundation size
- idle / move / guard / bridge flags
- build-state one-hot
- attack-state one-hot
- power / repair / warp / TNT flags
- garrison / passenger ratios
- ore / gems
- ammo

Recommended additions:

- weapon-ready / cooldown summary if exposed cheaply
- owner-side / country token if available
- capturable / occupied-tech-building flags for buildings
- current mission or order type
- current order target mode
- current order progress
- rally-intent summary for production buildings

Why:

- SC2 raw units expose current orders, order progress, rally targets, and weapon cooldown
- RA2 should carry a compact analogue so the model sees current intent, not just current state

### 12. `spatial`

Keep the current spatial channels:

- visible tiles
- visible resources
- self / allied / enemy / neutral / other-hostile presence
- HP planes
- self/enemy building/mobile presence

Recommended additions:

- capturable-tech structure locations
- bridge / bridge-zone mask
- buildable tiles
- blocked / impassable tiles
- terrain-height summary
- ore-region prior
- self base-core region prior
- start-location priors

### 13. `minimap`

Keep the current minimap branch.

Recommended additions:

- known enemy structure memory
- known enemy expansion / refinery memory
- friendly production-building locations
- static capturable-tech locations

This branch should stay "map summary" oriented rather than duplicating the entity tensor.

### 14. `mapStatic`

Add a static-map branch sourced from the static map dump.

Suggested contents:

- map size
- tile passability summary
- buildability summary
- terrain-height summary
- playable-area summary
- bridge connectivity summary
- capturable structure locations
- ore / gem field locations
- start-location set

This can be represented either as:

- extra scalar/static metadata
- extra spatial channels
- or both

The closest SC2 analogue is `StartRaw` in [raw.proto](D:/workspace/s2client-proto/s2clientprotocol/raw.proto).

## High-Priority V1 Additions

If we want an incremental first pass, implement in this order.

### Priority 0

- `availableActionMask`
- `ownedCompositionBow`
- faction / country identity in `scalarCore`

### Priority 1

- `buildOrderTrace`
- `techState`
- `productionState`
- the cheapest subset of `mapStatic`: pathability, buildability, terrain-height, and start-location priors

### Priority 2

- `enemyMemoryBow`
- `superWeaponState`
- richer `mapStatic` channels
- entity intent / order summaries

## Suggested Canonical Storage Policy

As with labels, canonical storage should stay structured.

Recommended storage rules:

- small categorical/scalar branches:
  - compact integer or float arrays
- bag-of-words branches:
  - fixed-width count vectors
- build-order trace:
  - fixed-length integer action-id sequence
- entity branch:
  - padded `[MAX_ENTITIES, ENTITY_FEATURE_DIM]`
- spatial / minimap / mapStatic branches:
  - fixed `[C, H, W]`

Do not collapse everything conceptually into one giant flat vector in the design.
Flattening is an output-format concern, not the semantic schema.

## What V1 Should Not Do

V1 should not:

- leak hidden enemy state
- depend on omniscient `getAllUnits()`-style queries
- encode replay-specific dynamic action ids
- encode UI clicks or camera movements
- try to reproduce SC2 feature names literally when the RA2 equivalent is different
- add score panels or control-group state just because the SC2 proto exposes them

## Practical Summary

The current RA2 extractor already has the right skeleton:

- scalar
- previous-action context
- selection context
- entity tensor
- spatial tensor
- minimap tensor

The main job for feature-layout V1 is to add the rich strategic and global context that mAS has in spirit, and that the upstream SC2 interfaces reinforce:

- action availability
- composition bag-of-words
- build-order history
- tech state
- production state
- enemy memory
- static map priors
- compact entity-intent summaries

Those are the most important missing feature groups for RA2 supervised learning.
