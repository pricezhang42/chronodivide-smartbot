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

- `[x]` Make `availableActionMask` observation-driven, similar in spirit to `pysc2.available_actions`.
  Hard policy:
  - impossible actions should be masked out confidently
  - uncertain actions may remain enabled
  - mask generation must not depend on hidden enemy state
  Current state:
  - the mask is now built from plausible self-controlled entity presence, current player production summaries, current queue state, direct per-sample super-weapon state, and static action semantics
  - chosen actions on the latest smoke validation are no longer disabled by the mask

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
  Current state:
  - `audit_feature_tensors.py` checks chosen-action compatibility against `availableActionMask`
  - after fixing deploy / queue-add / place-building gating, the smoke validation run at `available_action_fix_smoke2_20260318` reported `chosenActionDisabledCount = 0` for both saved shards
  - the broader replay-level validation run at `feature_layout_v1_validation_5replays_fix_20260318` also reported `chosenActionMaskCompatible = true`
  - the mask remains intentionally conservative: when selection or queue identity is ambiguous, uncertain actions stay enabled instead of producing false negatives

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

- `[x]` Confirm which replay-global side / country / faction fields are reliably available.

- `[x]` Add player identity features to `scalarCore`:
  - self side one-hot
  - enemy side multi-hot union over opposing replay players
  - self country one-hot
  - enemy country multi-hot union over opposing replay players if replay-global metadata safely provides it

- `[x]` Document the exact identity policy.
  Recommendation:
  - use replay metadata, not inferred unit composition
  Current V1 policy:
  - use replay-global `countryName` and `sideId` metadata
  - self identity is one-hot
  - enemy identity is a multi-hot union across replay players not on the acting player's team
  - unknown buckets are used when metadata is missing

## Phase 8: Priority 1 - `buildOrderTrace`

- `[x]` Freeze `BUILD_ORDER_TRACE_LEN`.

- `[x]` Define which actions contribute to the trace.
  Recommendation:
  - self production/build actions only
  - static action dict ids where meaningful
  - exclude UI-only or purely selection-only actions
  Current V1 policy:
  - `Queue::Add::*`
  - `PlaceBuilding::*`
  - `Order::Deploy::*`
  - `Order::DeploySelected::*`

- `[x]` Add a fixed-length early-game `buildOrderTrace` section.

- `[x]` Define padding and truncation policy:
  - pad with `-1`
  - keep earliest events only for V1

- `[~]` Validate against several replay openings:
  - power-first
  - refinery-first
  - barracks-first
  - war-factory / air opening variants if present
  Current state:
  - smoke validation completed on two real replays with different openings and countries
  - broader opening-pattern validation is still worth doing on a larger replay slice

## Phase 9: Priority 1 - `techState`

- `[x]` Freeze the V1 tech-state vocabulary.
  Suggested groups:
  - prerequisite-building flags
  - tech-building flags
  - unlocked production-category flags
  - special-tech or branch flags
  - power-satisfied flags

- `[x]` Implement `techState` from self-known owned state only.

- `[x]` Add shard metadata documenting:
  - feature names
  - bit positions
  - ruleset assumptions

- `[~]` Add validation for prerequisite consistency.
  Example checks:
  - if a high-tier production action is available, prerequisite flags should usually agree
  - if a prerequisite building is absent, dependent availability should not be confidently enabled unless ambiguity policy says otherwise
  Current state:
  - smoke validation covers section presence and real replay values
  - dedicated prerequisite-consistency audit is still worth adding

## Phase 10: Priority 1 - `productionState`

- `[x]` Confirm which generic production fields are already available in `py-chronodivide`.
  Current state:
  - generic production summaries already exist in `snapshot.mjs`
  - V1 now uses `maxTechLevel`, `buildSpeedModifier`, `queueCount`, queue summaries, factory counts, and available-count summaries by queue type

- `[x]` If needed, patch `py-chronodivide` generically to expose:
  - active queues
  - current production item
  - progress
  - hold / paused state
  - construction-yard placement mode
  - rally summary if available
  Current state:
  - `sl_dataset.mjs` now carries per-sample `playerProduction`
  - `snapshot.mjs` now exports `availableCountsByQueueType` alongside queue summaries
  - rally summary is still future work

- `[x]` Add `productionState` to the canonical feature layout.

- `[x]` Freeze a stable representation for queue slots and categories.
  Recommendation:
  - summarize by queue or building category first
  - avoid overfitting V1 to an unstable UI layout
  Current V1 policy:
  - `productionState` is a compact `100`-wide vector
  - one global block plus six queue-type summaries:
    - `Structures`
    - `Armory`
    - `Infantry`
    - `Vehicles`
    - `Aircrafts`
    - `Ships`
  - each queue summary stores factory count, available count, queue status flags, occupancy, and first-item summary

- `[~]` Validate production-state alignment on real queue actions:
  - `Queue::Add::*`
  - `Queue::Hold::*`
  - `Queue::Cancel::*`
  - building placement follow-through after construction-yard production
  Current state:
  - smoke validation completed on two real replays from different maps
  - saved `productionState` tensors show `Structures` queue transitions through `Idle -> Active -> Ready`
  - infantry queue activity appears in real samples once barracks production starts
  - dedicated hold/cancel focused validation is still worth adding on a larger slice

## Phase 11: Priority 2 - `enemyMemoryBow`

- `[x]` Freeze the memory vocabulary:
  - seen enemy buildings
  - seen enemy units
  - seen enemy tech flags
  Current V1 policy:
  - reuse the static composition vocabulary for seen enemy unit/building counts
  - add a companion `enemyMemoryTechFlags` section for seen enemy tech and unlock summaries

- `[x]` Add a replay-time memory accumulator keyed by player perspective.

- `[x]` Ensure memory updates only from observation-safe visibility.
  Current V1 policy:
  - update memory only from currently visible enemy entities in the safe observation tensor
  - store monotonic `max_visible_count_so_far` counts by name bucket
  - derive seen-tech flags only from enemy building names that have been observed

- `[x]` Add validation for monotonicity:
  - seen-enemy-building counts or flags should not disappear unless V1 intentionally allows forgetting
  Current state:
  - smoke validation on the early two-replay slice confirmed monotonic rows
  - a deeper one-replay slice reached enemy contact and still showed monotonic unit/building memory rows
  - first nonzero memory examples included `DOG`, `SENGINEER`, `NAHAND`, `NAPOWR`, `GAPILE`, and `GAPILL`

- `[ ]` Add a manifest summary:
  - most common seen enemy structures
  - most common seen enemy unit types
  - sparsity

## Phase 12: Priority 2 - `superWeaponState`

- `[x]` Confirm which super-weapon readiness / cooldown fields are exposed generically today.
  Current state:
  - `py-chronodivide` now carries per-sample `playerSuperWeapons` summaries
  - generic fields available today are:
    - `type`
    - `typeName`
    - `status`
    - `statusName`
    - `timerSeconds`

- `[x]` If needed, patch `py-chronodivide` generically first.
  Current state:
  - `superWeaponToPlain(...)` is now exported generically from `snapshot.mjs`
  - `sl_dataset.mjs` now carries `playerSuperWeapons` per action-aligned sample for downstream feature builders

- `[x]` Add:
  - presence flags
  - ready / not-ready flags
  - cooldown or charge progress
  - availability flags if distinct from readiness
  Current V1 policy:
  - `superWeaponState` is a compact `47`-wide vector
  - one global block plus per-type summaries for:
    - `MultiMissile`
    - `IronCurtain`
    - `LightningStorm`
    - `ChronoSphere`
    - `ChronoWarp`
    - `ParaDrop`
    - `AmerParaDrop`
  - each type summary stores:
    - `count`
    - `has`
    - `status_charging`
    - `status_paused`
    - `status_ready`
    - `charge_progress_01`
  - no distinct availability field is exposed generically today, so V1 uses `status_ready` as the first-pass availability/readiness signal
  - `charge_progress_01` is normalized against per-type nominal `RechargeTime` values from `rules.ini`
  - rule `RechargeTime` values are interpreted as minutes and converted to seconds before comparing against replay `timerSeconds`

- `[x]` Validate against replays with observed super-weapon actions.
  Current state:
  - full replay validation on `00758dde-b725-4442-ae8f-a657069251a0.rpl` produced nonzero `superWeaponState` tensors for both players
  - first live vectors showed `ParaDrop` charging with normalized progress near `0.003-0.025`
  - later samples in the same replay reached `Ready` for `ParaDrop`, and `bikerush` also reached nonzero `AmerParaDrop`

## Phase 13: Priority 2 - `mapStatic`

- `[x]` Promote the existing static map dump into feature-ready sections.
  Current state:
  - `py-chronodivide` now emits replay-constant compact static-map tensors per sampled player
  - `chronodivide-bot-sl` attaches them as a separate `mapStatic` feature section

- `[x]` Freeze the first map-static channels:
  - pathability
  - buildability
  - terrain-height summary
  - start locations
  Current V1 channels:
  - `foot_passable`
  - `track_passable`
  - `buildable_reference`
  - `terrain_height_norm`
  - `start_locations`

- `[ ]` Add second-wave channels after the first pass works:
  - bridge topology
  - capturable-tech structure locations
  - ore / gem priors
  - choke / connectivity hints if practical

- `[x]` Decide representation policy:
  - separate `mapStatic` branch
  - merged extra spatial channels
  - or both
  Current V1 policy:
  - separate `mapStatic` branch only

- `[x]` Validate on at least two different maps with clearly different geometry.
  Current state:
  - smoke validation completed on `2_pinch_point_le.map` and `4_caverns_v410.map`
  - `mapStatic` is replay-constant within a shard
  - channel sums differed across maps as expected
  - `buildable_reference` also differed across GDI and Nod players on the same map

## Phase 14: Entity Intent / Order Summaries

- `[x]` Add compact intent features to the entity branch.
  Suggested first pass:
  - current mission or order type
  - current order target mode
  - current order progress
  - weapon cooldown / ready summary
  - rally-intent summary for production buildings
  Current V1 policy:
  - `py-chronodivide` now appends generic raw transient entity fields:
    - `factory_status_idle`
    - `factory_status_delivering`
    - `factory_has_delivery`
    - `rally_point_valid`
    - `rally_x_norm`
    - `rally_y_norm`
    - `primary_weapon_cooldown_ticks`
    - `secondary_weapon_cooldown_ticks`
  - `chronodivide-bot-sl` now derives compact summary fields from those raw signals:
    - `intent_idle`
    - `intent_move`
    - `intent_attack`
    - `intent_build`
    - `intent_harvest`
    - `intent_repair`
    - `intent_factory_delivery`
    - `intent_rally_point_valid`
    - `intent_target_mode_none`
    - `intent_target_mode_tile`
    - `intent_target_mode_object`
    - `intent_target_mode_resource`
    - `intent_progress_01`
    - `weapon_ready_any`
    - `weapon_cooldown_progress_01`
    - `intent_rally_distance_norm`

- `[x]` Keep this generic in `py-chronodivide` if it depends on generic object state, but keep SL-specific encoding decisions in `chronodivide-bot-sl`.
  Current V1 split:
  - raw transient entity fields are exported generically in `py-chronodivide`
  - heuristic intent encoding stays in `chronodivide-bot-sl`
  - no hidden or omniscient enemy state is used to derive these summaries

- `[~]` Validate that these summaries help explain observed next actions in:
  - movement chains
  - combat focus-fire
  - harvest cycles
  - production and rally behavior
  Current state:
  - replay smoke validation completed on `00758dde-b725-4442-ae8f-a657069251a0.rpl`
  - saved `entityFeatures` width increased and the new summary channels were present in the shard schema
  - real nonzero summary examples appeared for:
    - `intent_attack`
    - `intent_build`
    - `intent_factory_delivery`
    - `intent_target_mode_object`
    - `intent_progress_01`
    - `weapon_ready_any`
    - `weapon_cooldown_progress_01`
  - several channels stayed zero on this opening-focused slice, including:
    - `intent_move`
    - `intent_harvest`
    - `intent_repair`
    - `intent_rally_point_valid`
  - that looks plausible for this replay slice, but broader replay-motif coverage is still needed
  - broader usefulness validation across replay motifs still remains

## Phase 15: Upgrade The Current Selection Summary

- `[x]` Add derived selection-summary scalars:
  - selected infantry / vehicle / aircraft / building counts
  - selected can-move flag
  - selected can-attack flag
  - selected can-deploy flag
  - selected can-gather flag
  - selected can-repair flag
  - selected mixed-type flag
  Current V1 policy:
  - the transformer now emits a dedicated `currentSelectionSummary` section
  - category counts are derived from the currently resolved self selection in the visible entity tensor
  - `selected_can_attack` uses a conservative visible-signal heuristic from attack-state, ammo, and weapon-cooldown fields
  - `selected_can_deploy` and `selected_can_gather` use known object-name sets
  - `selected_can_repair` uses the visible `has_wrench_repair` signal

- `[x]` Validate that these summaries agree with `currentSelectionIndices` and object capabilities.
  Current state:
  - `audit_feature_tensors.py` now rebuilds `currentSelectionSummary` directly from the saved `currentSelectionIndices`, `currentSelectionMask`, `currentSelectionResolvedMask`, `entityMask`, `entityNameTokens`, and `entityFeatures`
  - the smoke validation run at `current_selection_summary_smoke_20260318` reported `Issues: 0`
  - early real examples include resolved `AMCV` selections producing `selected_vehicle_count = 1`, `selected_can_move = 1`, and `selected_can_deploy = 1`

## Phase 16: Canonical Storage And Metadata

- `[ ]` Save the full structured feature schema into shard metadata.

- `[ ]` Save flatten offsets so flat tensors can be reconstructed into named sections later.

- `[ ]` Keep canonical storage structured and treat flattening as an output-format concern.

- `[ ]` Preserve backward-compatible flat `(features, labels)` shard output during the migration period.

## Phase 17: Feature Audits And Sanity Checks

- `[x]` Add a dedicated feature audit script analogous to `audit_training_targets.py`.
  Current state:
  - `audit_feature_tensors.py` now audits saved `.pt` and `.sections.pt` shards from a transform run

- `[x]` Audit section presence, shapes, dtypes, and flatten consistency.
  Current state:
  - the audit checks section presence, tensor shapes, section dtypes, and flat-vs-structured feature consistency against the saved flat shard

- `[x]` Audit sparsity / density for:
  - `availableActionMask`
  - `ownedCompositionBow`
  - `enemyMemoryBow`
  - `buildOrderTrace`
  - `techState`
  - `productionState`
  Current state:
  - the audit now writes section-density summaries and top-activation summaries for these sections into `feature_tensor_audit.json`

- `[x]` Audit value ranges for:
  - scalar counts
  - ratios
  - map channels
  - entity-intent fields
  Current state:
  - the audit now checks representative range invariants for:
    - scalar fractions
    - map-static constancy
    - non-negative spatial/minimap/mapStatic channels
    - production progress
    - super-weapon charge progress
    - entity-intent normalized fields

- `[~]` Audit leak safety:
  - no hidden enemy-state dependence in feature sections that claim to be observation-safe
  Current state:
  - the audit now checks a few observation-safe invariants:
    - chosen action enabled by `availableActionMask`
    - `enemyMemoryBow` monotonicity
    - `enemyMemoryTechFlags` monotonicity
    - `mapStatic` constancy within a shard
  - broader hidden-state leak proof is still partly a code-path review task, not something the saved tensors can prove by themselves

## Phase 18: Replay-Level Validation

- `[x]` Validate the final V1 feature layout on at least:
  - two maps
  - multiple factions
  - early-game openings
  - queue-heavy replays
  - combat-heavy replays
  Current state:
  - `validate_feature_layout_v1.py` now runs:
    - `transform_replay_data.py`
    - `audit_feature_tensors.py`
    - `audit_training_targets.py`
    and writes a single replay-level summary artifact
  - the wider validation run at `feature_layout_v1_validation_5replays_fix_20260318` completed with:
    - `5` replays
    - `10` saved shards
    - `4` distinct maps
    - `2` distinct sides
    - `4` distinct countries
    - `432` total samples
    - observed queue, combat, build, move, select, and super-weapon actions
    - `0` transform errors
    - `0` feature-audit issues
    - `0` training-target-audit issues

- `[~]` Run representative action-step spot checks:
  - action availability vs chosen action
  - build-order trace vs opening sequence
  - techState vs unlocked actions
  - productionState vs queue updates
  - enemyMemoryBow vs visible scouting history
  - mapStatic vs map geometry
  Current state:
  - action availability vs chosen action is now covered at replay-slice scale by `validate_feature_layout_v1.py` through `audit_feature_tensors.py`
  - earlier feature-specific smoke validations already covered:
    - build-order trace on opening sequences
    - techState early unlock progression
    - productionState queue transitions
    - enemy-memory monotonicity after first contact
    - mapStatic differences across maps
  - a broader manual spot-check pass across specific action steps is still worth doing, but this is no longer completely unvalidated

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
