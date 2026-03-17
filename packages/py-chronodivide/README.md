# Replay Re-Simulation Probe

This folder contains a first replay re-simulation toolkit for Chronodivide.

## Goal

Check whether `.rpl` files can be:

- re-simulated deterministically
- restored into full game state
- queried for player observations that are useful for SL

This phase intentionally ignores:

- camera movement
- user clicks as UI events
- minimap image rendering as raw UI pixels

The focus is world state and observation state that can feed training data.

## What This Probe Shows

### Current conclusion

Yes, replay re-simulation is possible on this machine, with two important caveats:

1. The public `@chronodivide/game-api` API exposes replay saving, but not replay loading/playback.
2. Replay playback therefore has to go through internal engine classes from the package bundle.

This toolkit handles that by generating a small runtime bridge next to the installed `@chronodivide/game-api` bundle and exporting the internal replay and engine classes needed for playback.

### What was validated

- A locally generated replay was re-simulated and matched sampled snapshots exactly.
- A real ladder replay from `packages/chronodivide-bot-sl/ladder_replays_top50` was stepped successfully through tick 300 on `2_pinch_point_le.map`.
- During re-simulation we were able to query:
  - player resources / power / radar state
  - all units / buildings on the map
  - player-visible self / allied / enemy / hostile units
  - tile visibility count from fog/shroud state
- The replay snapshot path now also records richer replay-state data when requested:
  - unit sight / veterancy / guard mode / purchase value
  - map theater and starting locations
  - player defeated / resigned / dropped / score state
  - player economy / combat counters such as credits gained, units built / lost / killed, buildings captured, and crates picked up
  - production queues and available objects
  - super-weapon status
  - terrain objects, neutral units, and tile-resource records
  - static engine rules data from `General`, `rules.ini`, `art.ini`, and `ai.ini`
- A replay feature extractor now turns player-correct observations into SL-safe feature records:
  - `34` scalar features
  - `50` per-entity numeric features plus object-name tokens
  - `18` coarse spatial channels
  - `18` minimap-style channels
- A supervised dataset builder now aligns one observation tensor to one kept replay action and emits fixed-shape feature/label tensor sections in JSON.

## Files

- `bridge.mjs`
  Runtime loader that exposes internal Chronodivide engine classes without permanently editing package source by hand.
- `snapshot.mjs`
  Helpers to turn `GameApi` state into JSON snapshots.
- `resim_core.mjs`
  Core replay parsing, deterministic playback, and sampling logic.
- `resim.mjs`
  CLI for stepping a replay and exporting sampled state/observation snapshots.
- `roundtrip_check.mjs`
  CLI that generates a fresh replay, re-simulates it, and checks sampled snapshot equality.
- `observation_audit.mjs`
  CLI that checks whether observation-safe APIs behave properly and whether global APIs leak hidden enemy state.
- `features.mjs`
  Safe observation-to-feature encoder for scalar, entity, spatial, and minimap SL inputs.
- `extract_features.mjs`
  CLI that samples a replay and writes SL-safe feature records.
- `RA2_SL_FEATURE_SCHEMA.md`
  Notes on the current RA2 supervised-learning feature schema and caveats.
- `sl_dataset.mjs`
  Action-aligned supervised-learning tensor builder that combines replay observations, inferred selection state, and decoded labels.
- `extract_sl_tensors.mjs`
  CLI that writes one replay's supervised-learning tensor dataset as JSON.
- `RA2_SL_TENSOR_DATASET.md`
  Notes on the current action-aligned tensor dataset schema and caveats.
- `REPLAY_RECORDING_CHECKLIST.md`
  Coverage checklist for replay reconstruction and recording support.
- `labels.mjs`
  Replay action-to-label encoder for structured SL supervision targets.
- `extract_labels.mjs`
  CLI that walks a replay tick-by-tick and writes action labels with inferred selection state.
- `RA2_SL_LABEL_SCHEMA.md`
  Notes on the current RA2 supervised-learning label schema and dataset-level action audit.

## Requirements

You need a usable Chronodivide data directory.

On this machine, the probe was validated with:

- `d:\workspace\ra2-headless-mix`

That directory contains the ladder map `2_pinch_point_le.map`, so real replay playback works there.

## Example: Re-Simulate A Ladder Replay

```powershell
node .\packages\py-chronodivide\resim.mjs `
  --replay .\packages\chronodivide-bot-sl\ladder_replays_top50\00758dde-b725-4442-ae8f-a657069251a0.rpl `
  --data-dir d:\workspace\ra2-headless-mix `
  --max-tick 300 `
  --sample-ticks 1,50,100,200,300 `
  --sample-mode observation `
  --output .\packages\py-chronodivide\sample_00758dde.json
```

## Example: Re-Simulate With Rich Recording

```powershell
node .\packages\py-chronodivide\resim.mjs `
  --replay .\packages\chronodivide-bot-sl\ladder_replays_top50\00758dde-b725-4442-ae8f-a657069251a0.rpl `
  --data-dir d:\workspace\ra2-headless-mix `
  --max-tick 600 `
  --sample-ticks 600 `
  --sample-mode global `
  --include-super-weapons true `
  --include-terrain-objects true `
  --include-neutral-units true `
  --include-tile-resources true `
  --include-player-production true `
  --include-player-stats true `
  --include-static-data true `
  --include-static-map true `
  --output .\packages\py-chronodivide\sample_00758dde_rich.json
```

These richer fields are opt-in because they can make snapshot files much larger.

When enabled, the replay JSON root also includes:

- `playerStatsAtStop`
- `staticData`
- `stoppedTick`
- `playbackReachedEnd`

When `--include-static-map` is enabled, `staticData.mapDump` is captured at replay-start tick `0`, before playback can mutate map-state objects.

## Example: Observation Fidelity Audit

```powershell
node .\packages\py-chronodivide\observation_audit.mjs `
  --replay .\packages\chronodivide-bot-sl\ladder_replays_top50\00758dde-b725-4442-ae8f-a657069251a0.rpl `
  --data-dir d:\workspace\ra2-headless-mix `
  --max-tick 300 `
  --sample-ticks 1,50,100,200,300
```

## Example: Extract SL Features

```powershell
node .\packages\py-chronodivide\extract_features.mjs `
  --replay .\packages\chronodivide-bot-sl\ladder_replays_top50\00758dde-b725-4442-ae8f-a657069251a0.rpl `
  --data-dir d:\workspace\ra2-headless-mix `
  --max-tick 1731 `
  --sample-ticks 1,1731 `
  --max-entities 128 `
  --spatial-size 32 `
  --minimap-size 64 `
  --output .\packages\py-chronodivide\features_00758dde.json
```

This writes feature samples containing:

- scalar features
- padded entity feature matrices
- entity masks
- object-name token IDs
- coarse spatial planes
- explicit minimap planes

Visible hostile-but-not-enemy objects are split into `neutral` and `otherHostile` buckets using visible ownership, instead of being folded into one combined feature bucket.

## Example: Extract SL Labels

```powershell
node .\packages\py-chronodivide\extract_labels.mjs `
  --replay .\packages\chronodivide-bot-sl\ladder_replays_top50\00758dde-b725-4442-ae8f-a657069251a0.rpl `
  --data-dir d:\workspace\ra2-headless-mix `
  --player mecharse `
  --max-actions 128 `
  --output .\packages\py-chronodivide\labels_00758dde.json
```

This writes structured labels containing:

- raw action id and dense action family
- delay from previous / to next kept action
- selection before / after action
- selected unit ids for selection and order actions
- order targets, queue updates, building placement, sell / repair, and super-weapon arguments

## Example: Extract Action-Aligned SL Tensors

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

This writes action-aligned tensor samples containing:

- observation tensors
- inferred current-selection tensors
- previous-action context tensors
- decoded action-label tensors
- schema metadata and shared vocabularies

## Example: Roundtrip Validation

```powershell
node .\packages\py-chronodivide\roundtrip_check.mjs `
  --data-dir d:\workspace\ra2-headless-mix `
  --max-tick 150 `
  --sample-every 50
```

If the replay roundtrip is correct, the script exits successfully and prints `matches: true`.

## What State Can Be Restored

From the current probe, the accessible reconstructed state includes:

- per-player metadata
- credits
- power totals / drain / low-power flag
- defeated / resigned / dropped / score state
- player economy / combat counters such as credits gained and built / killed / lost object counts
- radar disabled state
- production queues, queue items, and available production objects
- map dimensions
- map theater type and player starting locations
- static `GeneralRules`, `rules.ini`, `art.ini`, and `ai.ini` snapshots
- full static map dump with:
  - per-tile geometry and land types
  - bridge flags
  - map tags
  - per-tile resource summaries
  - terrain-object and neutral-object placement
- unit / building IDs
- owner
- type / name
- tile position
- world position
- HP
- sight / veterancy / guard mode / purchase value
- build / movement / stance-related fields exposed by `GameApi`
- weapon speed / cooldown and death-weapon presence
- factory delivery state
- super-weapon timers / status
- neutral units and terrain objects
- tile-resource records
- fog/shroud-derived visibility for a chosen player
- visible tile lists and visible resource tiles for a chosen player

Optional object fields stay sparse in JSON: if the engine field is `undefined` at that tick, the key is omitted instead of being written as `null`.

## Important Observation Caveat

Chronodivide's `GameApi` is still capable of exposing omniscient state during replay re-simulation if you use the wrong methods.

Safe for player-view extraction:

- `getVisibleUnits(playerName, ...)`
- `map.isVisibleTile(tile, playerName)`
- `getPlayerData(playerName)` for the acting player

Unsafe for SL features if you want player-correct observations:

- `getAllUnits()`
- `getUnitData(id)` or `getGameObjectData(id)` for ids obtained from omniscient scans
- `getPlayerData(enemyName)` if you do not want enemy resources/power leakage

This is why `resim.mjs` now supports `--sample-mode observation` in addition to the older global debug snapshot mode.

Heavy recording flags on `resim.mjs`:

- `--include-visible-tiles`
- `--include-visible-resource-tiles`
- `--include-super-weapons`
- `--include-terrain-objects`
- `--include-neutral-units`
- `--include-tile-resources`
- `--include-player-production`
- `--include-player-stats`
- `--include-static-data`
- `--include-static-map`

## What Is Still Missing

This now produces the feature side, the label side, and an action-aligned JSON tensor dataset for one replay, but it does not yet write framework-native `.pt` or `.npz` shards.

For replay recording coverage specifically, see:

- `REPLAY_RECORDING_CHECKLIST.md`

What this phase does provide is the hard part:

- replay parsing
- deterministic playback
- state reconstruction
- observation sampling
- SL-safe feature extraction
- structured action-label extraction
- action-aligned SL tensor extraction

Next steps would be:

1. add a multi-replay shard writer on top of the reusable `py-chronodivide` tensor API
2. align a dataset-wide object vocabulary and normalization pass
3. serialize replay samples plus decoded actions into framework-native binary tensors when the target stack is chosen
