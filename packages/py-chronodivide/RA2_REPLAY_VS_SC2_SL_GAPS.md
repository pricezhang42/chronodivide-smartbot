# RA2 Replay Gaps vs SC2/mAS Supervised Learning

## Scope

This note compares the Chronodivide Red Alert 2 replay files in:

- `packages/chronodivide-bot-sl/ladder_replays_top50`

against the information used by mini-AlphaStar (mAS) when transforming SC2 replays into supervised-learning `(features, labels)` tensors.

The goal is to answer:

1. What is missing in RA2 replays compared to SC2?
2. What important information used by mAS tensorization is missing from the current RA2 replay files?

## What The RA2 Replays Appear To Store

Sample replay:

- `packages/chronodivide-bot-sl/ladder_replays_top50/00758dde-b725-4442-ae8f-a657069251a0.rpl`

Observed structure:

- line 1: replay format/version header
- line 2: engine/version info
- line 3: match metadata
- remaining lines: `tick=player|encoded_command`
- final line: `END ...`

Example:

```text
RA2TSREPL_v6
ENGINE 0.80 1228950714
00758dde-b725-4442-ae8f-a657069251a0 ...
12=0|AgENAAIIBACODAAACQIACgAABAABAAAA
24=0|AgEGAAEJAgAKAAAEAAEAAAA=
...
END 35251
```

This looks like a command log plus metadata, not a rich per-timestep observation dump.

Supporting clue from Chronodivide docs:

- replay reliability is described in terms of deterministic simulation with identical initial state plus identical player inputs/commands

That strongly suggests the replay format is centered on action history, not observation snapshots.

## What mAS Uses From SC2 Replays

In mini-AlphaStar, replay preprocessing turns each replay step into:

- `feature`: flattened game state
- `label`: flattened structured action target

The SL pipeline builds these from SC2 observations in:

- `alphastarmini/core/sl/transform_replay_data.py`
- `alphastarmini/core/sl/feature.py`
- `alphastarmini/core/sl/label.py`

Important mAS feature groups include:

- scalar features:
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
- entity features:
  - per-entity tensor for units/buildings
- spatial features:
  - minimap/map channels

Important mAS label groups include:

- `action_type`
- `delay`
- `queue`
- `selected_units`
- `target_unit`
- `target_location`

## 1. What Is Missing In RA2 Replays Compared To SC2?

### Missing per-step observation state

The RA2 `.rpl` files do not appear to store full per-timestep game observations. By contrast, mAS depends on replay observations as the source for supervised state features.

### Missing structured entity snapshots

The RA2 replays do not appear to include an explicit unit/building table for each step, such as:

- unit type
- owner
- position
- health
- orders
- build progress
- visibility

This is one of the biggest gaps relative to SC2-style replay processing.

### Missing spatial map/minimap state

The RA2 replay files do not appear to contain per-step spatial planes comparable to SC2 minimap/map tensors.

### Missing explicit scalar game-state features

The replay text does not directly expose state fields analogous to:

- resources / production stats
- tech / upgrade state
- available actions
- unit counts
- active effects
- build-order context

Some of this may be reconstructable by re-simulating the match, but it is not present as explicit replay-side data.

### Missing observation perspective / fog-aware state

SC2 replay processing can be done from a player observation perspective. The RA2 replay sample looks like global command history, not player-observation history.

### Missing decoded action structure

The RA2 replays store encoded command payloads, but not directly readable command semantics. So the replay does not immediately provide action heads such as action type, selected units, target unit, or target location.

## 2. Important Information Used By mAS Tensorization That Is Missing In RA2 Replays

### Missing feature-side information

The following mAS-style feature information is not explicitly present in the RA2 replay files:

- full scalar state vector
- full entity tensor
- full spatial tensor
- available-action mask
- unit-count summaries
- tech / upgrade snapshots
- effects / transient world-state indicators
- build-order state
- prior-action context as explicit fields

More specifically, these mAS fields are missing as direct replay data:

- `available_actions`
- `unit_counts_bow`
- `upgrades`
- `enemy_upgrades`
- `effects`
- `beginning_build_order`
- `last_action_type`
- `last_repeat_queued`

### Partially recoverable feature information

Some feature information may be reconstructable, but is not directly stored:

- `time`: recoverable from replay ticks
- `last_delay`: likely recoverable from action timing
- pieces of economy / production / army state: only if the game can be re-simulated and inspected step by step

### Missing label-side information

mAS labels are structured into multiple heads. In the RA2 replay text, only action timing and acting player are immediately visible.

Without decoding the command payloads, the following label information is effectively missing:

- `action_type`
- `queue`
- `selected_units`
- `target_unit`
- `target_location`

`delay` is the main label that appears directly recoverable from the raw replay text, because ticks are explicit.

## Practical Conclusion

The current RA2 replay dataset appears suitable for:

- replay metadata
- action timing
- command history

But it does not appear sufficient by itself for mAS-style supervised learning.

To build SC2-like `(features, labels)` tensors for RA2, you likely need both:

1. A replay-command decoder that converts each encoded payload into a structured action.
2. A simulation or instrumentation path that replays the game and emits per-step game state / player observation tensors.

Without the second piece, the biggest missing component is the feature side, not just the labels.

## Short Answer

### 1. What is missing in RA2 replays compared to SC2?

- Rich per-step observations
- Structured unit/building state
- Spatial map/minimap tensors
- Explicit scalar game-state features
- Perspective/fog-aware observation state
- Readable structured actions

### 2. What important info transformed into `(features, labels)` tensors is missing?

Missing from features:

- scalar state fields such as `available_actions`, `unit_counts_bow`, `upgrades`, `effects`, `beginning_build_order`
- entity tensors
- spatial tensors
- explicit prior-action context fields

Missing from labels unless commands are decoded:

- `action_type`
- `queue`
- `selected_units`
- `target_unit`
- `target_location`

Likely recoverable:

- `delay`
- basic time progression

