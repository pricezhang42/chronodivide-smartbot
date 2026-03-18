# RA2 Feature Layout Gap Analysis

This note compares the current RA2 supervised-learning feature layout against:

- the mini-AlphaStar SL trace in [SL_BATCH_TRACE.MD](D:/workspace/mini-AlphaStar/doc/SL_BATCH_TRACE.MD)
- the mini-AlphaStar scalar/entity/map definitions in [hyper_parameters.py](D:/workspace/mini-AlphaStar/alphastarmini/lib/hyper_parameters.py) and [feature.py](D:/workspace/mini-AlphaStar/alphastarmini/core/sl/feature.py)
- the upstream SC2 observation surface in [features.py](D:/workspace/pysc2/pysc2/lib/features.py), [actions.py](D:/workspace/pysc2/pysc2/lib/actions.py), [sc2api.proto](D:/workspace/s2client-proto/s2clientprotocol/sc2api.proto), [raw.proto](D:/workspace/s2client-proto/s2clientprotocol/raw.proto), [spatial.proto](D:/workspace/s2client-proto/s2clientprotocol/spatial.proto), and [ui.proto](D:/workspace/s2client-proto/s2clientprotocol/ui.proto)

It answers a narrower question than the full `py-chronodivide` replay-recording work:

- what feature groups does mini-AlphaStar feed into SL?
- which of those groups already have a usable RA2 analogue?
- which important groups are still missing or much thinner in the current RA2 pipeline?
- which extra gaps show up once we compare against the upstream SC2 interfaces that `pysc2` sits on top of?

## Baseline: What mAS Feeds Into SL

From [SL_BATCH_TRACE.MD](D:/workspace/mini-AlphaStar/doc/SL_BATCH_TRACE.MD) and the underlying code in [hyper_parameters.py](D:/workspace/mini-AlphaStar/alphastarmini/lib/hyper_parameters.py), the mAS feature row is reconstructed into three major branches:

- `entity_state`
- `statistical_state`
- `map_state`

The important scalar/statistical groups include:

- `agent_statistics`
- `home_race`
- `away_race`
- `upgrades`
- `enemy_upgrades`
- `time`
- `available_actions`
- `unit_counts_bow`
- `mmr`
- `units_buildings`
- `effects`
- `upgrade`
- `beginning_build_order`
- `last_delay`
- `last_action_type`
- `last_repeat_queued`

So even though mAS is often described as "scalar + entity + spatial," its scalar branch is quite rich and includes:

- action availability
- own and enemy tech state
- build-order history
- previous-action context
- structured global summaries of army and economy state

## Useful Upstream SC2 Context

Checking `pysc2` and `s2client-proto` adds an important nuance: mAS only uses part of the full SC2 observation surface.

Upstream SC2 observations expose:

- `Observation.player_common`, `abilities`, `score`, `raw_data`, `feature_layer_data`, and `ui_data` in [sc2api.proto](D:/workspace/s2client-proto/s2clientprotocol/sc2api.proto)
- `ObservationRaw.player.upgrade_ids`, `units`, `map_state`, `effects`, `radar`, and per-unit `orders` / `rally_targets` in [raw.proto](D:/workspace/s2client-proto/s2clientprotocol/raw.proto)
- feature-layer `pathable`, `buildable`, `visibility_map`, `effects`, and minimap layers in [spatial.proto](D:/workspace/s2client-proto/s2clientprotocol/spatial.proto)
- UI panels for control groups, multi-selection, cargo, and production queues in [ui.proto](D:/workspace/s2client-proto/s2clientprotocol/ui.proto)
- static `StartRaw` map grids for pathing, placement, terrain height, playable area, and start locations in [raw.proto](D:/workspace/s2client-proto/s2clientprotocol/raw.proto)

`pysc2` also makes one design choice that is directly useful for RA2: its `available_actions` output is derived from observation-time `abilities` plus function metadata, not from an omniscient perfect legality solver. See [features.py](D:/workspace/pysc2/pysc2/lib/features.py) and [actions.py](D:/workspace/pysc2/pysc2/lib/actions.py).

Two takeaways matter for RA2:

- some RA2 gaps are direct parity gaps with what mAS already uses
- some others are not strict mAS parity gaps, but they are still good design targets because they mirror the richer SC2 interface underneath mAS

## Current RA2 Feature Layout

The current RA2 replay extractor emits these main feature sections, via [features.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/features.mjs) and [sl_dataset.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/sl_dataset.mjs):

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

That means RA2 already has the same broad architectural branches as mAS:

- scalar/global context
- selected-unit context
- visible-entity tensor
- spatial tensor
- minimap tensor

This is a strong starting point.

## What Already Has A Good RA2 Analogue

These mAS ideas already have a reasonable equivalent in the current RA2 pipeline.

### 1. Previous-action context

mAS has:

- `last_delay`
- `last_action_type`
- `last_repeat_queued`

RA2 now has the V1 equivalent:

- `lastActionContext = [delayFromPreviousAction, lastActionTypeIdV1, lastQueue]`

### 2. Multi-unit context

mAS uses selected-unit information heavily on both the feature and label side.

RA2 already carries:

- current selection count
- current selection indices
- current selection mask
- current selection resolved mask

So the "what am I currently controlling?" branch is already present.

### 3. Entity branch

mAS uses a large entity tensor.

RA2 already has:

- padded visible-entity tensor
- entity mask
- relation channels
- object-type channels
- HP, orientation, velocity, build-state, attack-state, cargo, ammo, and ore state

This is not as large or as semantically rich as SC2's entity tensor, but it is structurally the right branch.

### 4. Spatial branch

mAS uses `map_state`.

RA2 already has:

- semantic spatial planes
- minimap planes

These are observation-safe and usable for SL.

## One Important Gap We Were Underselling

The current RA2 entity branch is still thinner than the SC2 raw/entity surface in one important way: current order and intent state.

`pysc2` `FeatureUnit` and the underlying raw proto include per-unit fields such as:

- `order_length`
- `order_id_0..3`
- `order_progress_0..1`
- `weapon_cooldown`
- `assigned_harvesters`
- `ideal_harvesters`
- `add_on_tag` / `addon_unit_type`
- `rally_targets`
- `engaged_target_tag`

See [features.py](D:/workspace/pysc2/pysc2/lib/features.py) and [raw.proto](D:/workspace/s2client-proto/s2clientprotocol/raw.proto).

RA2 currently records useful transient object state such as attack/build state, guard mode, ammo, ore load, and rally point, but it still lacks a compact per-entity summary of:

- current order or mission type
- current order target kind
- current order progress
- factory or yard queue intent
- attack cooldown or reload readiness

That matters because next-action prediction depends not just on "what exists" but also on "what these units are already trying to do."

## Important Gaps Relative To mAS

The gaps below are the important ones, not just differences.

### 1. No `available_actions` equivalent

This is one of the biggest gaps.

mAS explicitly feeds `available_actions` as part of the scalar state. That gives the model a direct mask-like summary of which action types are currently legal.

RA2 currently has no feature block equivalent to:

- "which of the static RA2 SL action types are available now?"

Why this matters:

- it helps constrain the action head
- it reduces impossible-action confusion
- it is especially important in RA2 because build, production, placement, and super-weapon availability depend on game state, tech, power, and selection

Current status:

- missing

### 2. No `beginning_build_order` equivalent

mAS feeds a fixed-length beginning-build-order tensor.

RA2 currently has no explicit build-order history feature.

Why this matters:

- opening decisions strongly determine legal follow-up tech and production
- RA2 has sharp branching openings where knowing the early build queue is often more informative than raw current-state counts alone

Current status:

- missing

### 3. No full `unit_counts_bow` equivalent

mAS includes a structured unit-type count vector.

RA2 currently has only coarse visible or self counts such as:

- self unit / building / infantry / vehicle / aircraft counts
- enemy visible counts at the same coarse level

What is missing is a full bag-of-words style count vector by object name or type, especially for:

- owned units
- owned buildings
- known enemy buildings and tech

Why this matters:

- composition counts are one of the strongest cheap global summaries
- a few coarse categories are not enough for RA2 tech and production modeling

Current status:

- partially covered, but much too coarse

### 4. No faction or side identity feature block

mAS has:

- `home_race`
- `away_race`

RA2 currently does not explicitly encode player side, faction, or country in the SL feature vector.

Why this matters:

- faction strongly changes build trees, unit set, and available buildings
- without this, the model has to infer too much from current visible units and buildings

Current status:

- missing

### 5. No upgrades or tech-state equivalent

mAS has:

- `upgrades`
- `enemy_upgrades`
- upgrade-related scalar groups

RA2 currently has no explicit tech-state or prerequisite-state feature block.

Examples that need an analogue:

- whether a tech prerequisite is present
- whether a given build branch is unlocked
- whether a production tab should exist at all
- whether super-weapons or late tech are online

Current status:

- missing

### 6. No production-state or queue-state feature block

This is not a one-to-one mAS scalar name, but it is a critical RA2 equivalent gap.

RA2 currently does not expose as features:

- factory production queues
- current item under construction
- build progress
- held or paused queue state
- building placement mode
- sidebar production availability

Why this matters:

- queue and production actions are a large fraction of RA2 control
- the model should know what is already building before choosing the next production action

Current status:

- missing

### 7. No explicit super-weapon readiness, cooldown, or support-power state

Current labels can represent super-weapon actions, but the current feature layout does not expose a dedicated readiness or cooldown block for them.

Why this matters:

- otherwise the model has to infer support-power readiness indirectly
- for RA2, this is a highly actionable control state

Current status:

- missing

### 8. Spatial planes are missing static map-property channels

mAS map state is richer than just current visible occupancy.

The upstream SC2 interface makes these kinds of static grids first-class through `StartRaw.pathing_grid`, `placement_grid`, `terrain_height`, and `start_locations`.

RA2 already has semantic spatial/minimap planes, but still lacks many static or semi-static map channels that are now available from the static map dump:

- buildability
- passability
- bridge zones and bridge topology
- terrain-height summaries
- ore-field priors and ore-region masks
- capturable-tech structure locations
- choke and connectivity hints
- start-location priors

Why this matters:

- building placement and movement are tightly constrained by map geometry
- these are useful even when currently unseen

Current status:

- missing

### 9. No enemy memory or known-enemy-state summary

mAS has explicit enemy-upgrade state.

RA2 currently only uses current observation-safe visible enemy state, not "known enemy state so far."

The missing analogue is not omniscient enemy info. It is:

- enemy structures seen so far
- enemy tech branches inferred from seen buildings
- enemy super-weapons seen so far
- known enemy composition counts from observation history

Why this matters:

- pure current-visibility features lose too much strategic context
- RTS policy needs memory, and some of that can be encoded as features instead of leaving it all to recurrence

Current status:

- missing

### 10. No explicit action-availability constraints from selection state

RA2 has current selection tensors, but not a compact derived summary like:

- selected object types
- selected count by type
- selected capability summary
- selected can-deploy / can-attack / can-repair / can-gather flags

Why this matters:

- many RA2 actions are selection-sensitive
- these derived summaries make action availability much easier to learn

Current status:

- partially covered by raw selection tensors, but still missing as compact derived scalars

### 11. No compact per-entity order or intent summary

This is the clearest upstream-feature gap that is not yet captured in the current RA2 layout.

SC2 raw units expose current orders, order progress, rally targets, and weapon cooldown directly. `pysc2` also folds some of that into `FeatureUnit`.

RA2 currently records:

- idle, move, guard, and attack-state style fields
- build-state style fields
- rally point
- cargo, ore, and ammo state

But it does not yet expose a compact per-entity intent block such as:

- current mission or order type
- current order target mode
- current order progress
- current attack cooldown or reload summary
- current production target for factories and construction yards

Why this matters:

- it is a strong predictor of the next command
- it helps action availability
- it reduces the need for the model to infer current intent only from motion or combat state

Current status:

- partially covered, but still much thinner than the SC2 raw/entity surface

## High-Priority Missing Items

If we prioritize by likely training value, the top missing feature groups are:

1. `available_actions` over the static RA2 action dict
2. owned unit and building count bag-of-words
3. faction or side identity
4. tech and prerequisite state
5. production queue and build-progress state
6. build-order history
7. static map-property channels
8. known-enemy memory summaries
9. compact entity-intent summaries

## What Is Probably Not Worth Chasing For Parity

Some upstream SC2 fields are real, but they are not first-order RA2 SL priorities:

- observer camera movement
- UI click history
- control groups
- score panels

Those can stay out of V1. They are either user-interface heavy, not clearly used by mAS SL, or not aligned with the current goal of action-policy supervision.

Some mAS feature groups also do not have a direct RA2 equivalent or are lower priority:

- `mmr`
  Useful for analysis, but not critical as a policy input.
- exact SC2-style `effects`
  RA2 should instead expose the RA2-specific transient state that actually matters.
- exact race-home/away semantics
  RA2 should encode side, country, and ruleset-relevant faction identity instead.

## Practical Summary

RA2 is not missing the overall SL feature architecture. It already has the major branches:

- scalar
- current selection
- entity
- spatial
- minimap
- previous-action context

What it is still missing, compared to mAS in spirit and compared to the upstream SC2 interfaces that mAS depends on, is the rich structured global context:

- action availability
- tech availability
- composition bag-of-words
- build-order history
- production state
- known enemy memory
- static map priors
- compact entity-intent summaries

Those are the most important feature-layout gaps to close next.
