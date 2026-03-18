# RA2 SL Feature Layout V1 Implementation Checklist

This checklist turns [RA2_SL_FEATURE_LAYOUT_V1.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_FEATURE_LAYOUT_V1.md) into concrete implementation steps.

The goal is to move from the current RA2 feature surface to a mini-AlphaStar-style feature layout with:

- richer scalar and strategic context
- explicit action availability
- compact composition and memory summaries
- stronger production and tech-state features
- better static-map priors
- compact per-entity intent summaries

## Scope

This checklist is about the feature side only.

It does not include:

- label-schema changes
- training-loop loss implementation
- winner-only replay policy
- binary transport redesign

## Status Legend

- `[x]` already done
- `[~]` partially done
- `[ ]` not done yet

## Current Starting Point

- `[x]` Replay parsing and deterministic playback are already implemented in `py-chronodivide`.
- `[x]` Observation-safe player snapshots already exist in `py-chronodivide/snapshot.mjs`.
- `[x]` The current RA2 feature surface already has these major branches:
  - scalar
  - previous-action context
  - selection context
  - entity tensor
  - spatial tensor
  - minimap tensor
- `[x]` The feature-side previous-action context has already been migrated to the V1 shape in `transform_replay_data.py`:
  - `delayFromPreviousAction`
  - `lastActionTypeIdV1`
  - `lastQueue`
- `[x]` Static action-dict ids already exist in [action_dict.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/action_dict.py), which is a prerequisite for `availableActionMask`.
- `[x]` `py-chronodivide` already records static map data and map dumps.
- `[~]` Static map data exists in the replay-recording path, but it is not yet promoted into the SL feature layout as dedicated feature sections or channels.
- `[~]` The entity branch already carries useful transient object state:
  - build / attack state
  - ammo
  - ore load
  - rally point
  - guard mode
  - power / repair / TNT flags
  but it still lacks a compact explicit intent or order summary.

## Phase 1: Freeze The V1 Feature Contract

- `[ ]` Freeze the V1 top-level feature section names:
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

- `[ ]` Freeze the first-pass constants:
  - `MAX_ENTITIES`
  - `MAX_SELECTION`
  - `BUILD_ORDER_TRACE_LEN`
  - map plane size(s)
  - vocabulary sizing rules for composition and memory BOW sections

- `[ ]` Freeze the feature-side missing-value policy.
  Recommendation:
  - integer categorical fields use `-1` when not applicable
  - masks use `0/1`
  - scalar quantities default to `0` only when semantically safe
  - unavailable or unknown values that must stay distinct from zero use explicit validity masks

## Phase 2: Keep `py-chronodivide` Generic

- `[ ]` Keep generic replay-state extraction in `py-chronodivide`.

- `[ ]` Confirm that `py-chronodivide` continues to expose raw reusable inputs, not SL-specific flattened feature ids.

- `[ ]` Keep these responsibilities in `py-chronodivide`:
  - player-safe observation snapshots
  - generic entity features
  - generic spatial / minimap extraction
  - static map dump extraction
  - generic production / object / player-state accessors where available

- `[ ]` Keep these responsibilities in `chronodivide-bot-sl`:
  - action-dict-sized `availableActionMask`
  - V1 feature section assembly
  - build-order trace construction
  - enemy-memory accumulation policy
  - canonical storage and flattening rules

## Phase 3: Freeze The Observation-Safe Policy

- `[ ]` Write down and enforce the feature-side anti-leak policy:
  - no hidden enemy unit state
  - no global enemy player-state queries
  - no omniscient legality solver for `availableActionMask`
  - memory features may only depend on information seen so far

- `[ ]` Make `availableActionMask` observation-driven, similar in spirit to `pysc2.available_actions`.
  Hard policy:
  - impossible actions should be masked out confidently
  - uncertain actions may remain enabled
  - mask generation must not depend on hidden enemy state

- `[ ]` Write the memory policy for enemy features.
  Hard policy:
  - `enemyMemoryBow` is monotonic over observed information unless an explicit forgetting rule is introduced later
  - no backfilling from omniscient replay state

## Phase 4: Formalize The Current Feature Surface

- `[ ]` Rename the current top-level scalar branch to `scalarCore` in the transformer metadata.

- `[ ]` Document the exact current feature sections and shapes in shard metadata:
  - scalar
  - selection
  - entity
  - spatial
  - minimap
  - previous-action context

- `[ ]` Add schema metadata for every section:
  - section name
  - dtype
  - shape
  - flatten offsets

- `[ ]` Preserve compatibility with the existing flat feature tensor while the structured V1 sections grow.

## Phase 5: Priority 0 - `availableActionMask`

- `[x]` Add a builder that emits `availableActionMask` with width equal to the static SL action dict size.

- `[x]` Decompose mask generation into explicit sources:
  - selection-driven order availability
  - queue/sidebar action availability
  - building placement availability
  - super-weapon availability

- `[x]` Implement a first conservative version:
  - confidently disable clearly impossible actions
  - leave ambiguous actions enabled

- `[ ]` Save mask-generation metadata for audit:
  - action type
  - enabled / disabled
  - source or reason

- `[ ]` Add a mask audit script or manifest summary:
  - per-action enable frequency
  - per-action disable frequency
  - top action types that are almost always disabled or always enabled

- `[x]` Validate on representative replays that actually observed action types are almost always enabled at the action step.

## Phase 6: Priority 0 - `ownedCompositionBow`

- `[x]` Freeze the composition vocabulary source.
  Recommendation:
  - use a static object-name vocabulary aligned to the current RA2 ruleset
  - keep unknown fallback buckets

- `[x]` Add:
  - `ownedUnitCountBow`
  - `ownedBuildingCountBow`
  Implementation note:
  - the current transformer stores these as two rows inside one `ownedCompositionBow` section

- `[x]` Base these on self-known current state, not visibility-limited state.

- `[~]` Save vocabulary metadata in the run manifest:
  - id -> object name
  - object name -> id
  - unknown bucket policy
  Current state:
  - vocabulary metadata is already saved in shard-level `featureLayoutV1` metadata
  - promote it into the run manifest when the feature-layout manifest block is added

- `[ ]` Add a composition audit:
  - top occupied slots
  - unknown-bucket rate
  - sparsity summary

## Phase 7: Priority 0 - Faction / Country Identity

- `[ ]` Confirm which replay-global side / country / faction fields are reliably available.

- `[ ]` Add player identity features to `scalarCore`:
  - self faction or country one-hot
  - enemy faction or country one-hot if replay-global metadata safely provides it

- `[ ]` Document the exact identity policy.
  Recommendation:
  - use replay metadata, not inferred unit composition

## Phase 8: Priority 1 - `buildOrderTrace`

- `[ ]` Freeze `BUILD_ORDER_TRACE_LEN`.

- `[ ]` Define which actions contribute to the trace.
  Recommendation:
  - self production/build actions only
  - static action dict ids where meaningful
  - exclude UI-only or purely selection-only actions

- `[ ]` Add a fixed-length early-game `buildOrderTrace` section.

- `[ ]` Define padding and truncation policy:
  - pad with `-1`
  - keep earliest events only for V1

- `[ ]` Validate against several replay openings:
  - power-first
  - refinery-first
  - barracks-first
  - war-factory / air opening variants if present

## Phase 9: Priority 1 - `techState`

- `[ ]` Freeze the V1 tech-state vocabulary.
  Suggested groups:
  - prerequisite-building flags
  - tech-building flags
  - unlocked production-category flags
  - special-tech or branch flags
  - power-satisfied flags

- `[ ]` Implement `techState` from self-known owned state only.

- `[ ]` Add shard metadata documenting:
  - feature names
  - bit positions
  - ruleset assumptions

- `[ ]` Add validation for prerequisite consistency.
  Example checks:
  - if a high-tier production action is available, prerequisite flags should usually agree
  - if a prerequisite building is absent, dependent availability should not be confidently enabled unless ambiguity policy says otherwise

## Phase 10: Priority 1 - `productionState`

- `[ ]` Confirm which generic production fields are already available in `py-chronodivide`.

- `[ ]` If needed, patch `py-chronodivide` generically to expose:
  - active queues
  - current production item
  - progress
  - hold / paused state
  - construction-yard placement mode
  - rally summary if available

- `[ ]` Add `productionState` to the canonical feature layout.

- `[ ]` Freeze a stable representation for queue slots and categories.
  Recommendation:
  - summarize by queue or building category first
  - avoid overfitting V1 to an unstable UI layout

- `[ ]` Validate production-state alignment on real queue actions:
  - `Queue::Add::*`
  - `Queue::Hold::*`
  - `Queue::Cancel::*`
  - building placement follow-through after construction-yard production

## Phase 11: Priority 2 - `enemyMemoryBow`

- `[ ]` Freeze the memory vocabulary:
  - seen enemy buildings
  - seen enemy units
  - seen enemy tech flags

- `[ ]` Add a replay-time memory accumulator keyed by player perspective.

- `[ ]` Ensure memory updates only from observation-safe visibility.

- `[ ]` Add validation for monotonicity:
  - seen-enemy-building counts or flags should not disappear unless V1 intentionally allows forgetting

- `[ ]` Add a manifest summary:
  - most common seen enemy structures
  - most common seen enemy unit types
  - sparsity

## Phase 12: Priority 2 - `superWeaponState`

- `[ ]` Confirm which super-weapon readiness / cooldown fields are exposed generically today.

- `[ ]` If needed, patch `py-chronodivide` generically first.

- `[ ]` Add:
  - presence flags
  - ready / not-ready flags
  - cooldown or charge progress
  - availability flags if distinct from readiness

- `[ ]` Validate against replays with observed super-weapon actions.

## Phase 13: Priority 2 - `mapStatic`

- `[ ]` Promote the existing static map dump into feature-ready sections.

- `[ ]` Freeze the first map-static channels:
  - pathability
  - buildability
  - terrain-height summary
  - start locations

- `[ ]` Add second-wave channels after the first pass works:
  - bridge topology
  - capturable-tech structure locations
  - ore / gem priors
  - choke / connectivity hints if practical

- `[ ]` Decide representation policy:
  - separate `mapStatic` branch
  - merged extra spatial channels
  - or both

- `[ ]` Validate on at least two different maps with clearly different geometry.

## Phase 14: Entity Intent / Order Summaries

- `[ ]` Add compact intent features to the entity branch.
  Suggested first pass:
  - current mission or order type
  - current order target mode
  - current order progress
  - weapon cooldown / ready summary
  - rally-intent summary for production buildings

- `[ ]` Keep this generic in `py-chronodivide` if it depends on generic object state, but keep SL-specific encoding decisions in `chronodivide-bot-sl`.

- `[ ]` Validate that these summaries help explain observed next actions in:
  - movement chains
  - combat focus-fire
  - harvest cycles
  - production and rally behavior

## Phase 15: Upgrade The Current Selection Summary

- `[ ]` Add derived selection-summary scalars:
  - selected infantry / vehicle / aircraft / building counts
  - selected can-move flag
  - selected can-attack flag
  - selected can-deploy flag
  - selected can-gather flag
  - selected can-repair flag
  - selected mixed-type flag

- `[ ]` Validate that these summaries agree with `currentSelectionIndices` and object capabilities.

## Phase 16: Canonical Storage And Metadata

- `[ ]` Save the full structured feature schema into shard metadata.

- `[ ]` Save flatten offsets so flat tensors can be reconstructed into named sections later.

- `[ ]` Keep canonical storage structured and treat flattening as an output-format concern.

- `[ ]` Preserve backward-compatible flat `(features, labels)` shard output during the migration period.

## Phase 17: Feature Audits And Sanity Checks

- `[ ]` Add a dedicated feature audit script analogous to `audit_training_targets.py`.

- `[ ]` Audit section presence, shapes, dtypes, and flatten consistency.

- `[ ]` Audit sparsity / density for:
  - `availableActionMask`
  - `ownedCompositionBow`
  - `enemyMemoryBow`
  - `buildOrderTrace`
  - `techState`
  - `productionState`

- `[ ]` Audit value ranges for:
  - scalar counts
  - ratios
  - map channels
  - entity-intent fields

- `[ ]` Audit leak safety:
  - no hidden enemy-state dependence in feature sections that claim to be observation-safe

## Phase 18: Replay-Level Validation

- `[ ]` Validate the final V1 feature layout on at least:
  - two maps
  - multiple factions
  - early-game openings
  - queue-heavy replays
  - combat-heavy replays

- `[ ]` Run representative action-step spot checks:
  - action availability vs chosen action
  - build-order trace vs opening sequence
  - techState vs unlocked actions
  - productionState vs queue updates
  - enemyMemoryBow vs visible scouting history
  - mapStatic vs map geometry

## Recommended Implementation Order

If we want the shortest path to a materially better feature layout, the best order is:

1. `availableActionMask`
2. `ownedCompositionBow`
3. faction / country identity
4. `buildOrderTrace`
5. `techState`
6. `productionState`
7. `mapStatic`
8. `enemyMemoryBow`
9. `superWeaponState`
10. entity intent summaries

## Practical Summary

The current RA2 pipeline already has the skeleton of a strong SL feature layout.

The most important missing pieces are the same ones highlighted by the mAS trace and reinforced by the upstream SC2 interfaces:

- observation-driven action availability
- compact composition summaries
- build-order history
- tech and production state
- static map priors
- compact intent summaries

Those are the pieces this checklist is meant to drive to completion.
