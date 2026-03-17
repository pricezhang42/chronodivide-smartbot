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
- minimap image rendering

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
- A replay feature extractor now turns player-correct observations into SL-safe feature records:
  - `33` scalar features
  - `49` per-entity numeric features plus object-name tokens
  - `16` coarse spatial channels

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
  Safe observation-to-feature encoder for scalar, entity, and spatial SL inputs.
- `extract_features.mjs`
  CLI that samples a replay and writes SL-safe feature records.
- `RA2_SL_FEATURE_SCHEMA.md`
  Notes on the current RA2 supervised-learning feature schema and caveats.

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
  --output .\packages\py-chronodivide\features_00758dde.json
```

This writes feature samples containing:

- scalar features
- padded entity feature matrices
- entity masks
- object-name token IDs
- coarse spatial planes

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
- radar disabled state
- map dimensions
- unit / building IDs
- owner
- type / name
- tile position
- world position
- HP
- build / movement / stance-related fields exposed by `GameApi`
- fog/shroud-derived visibility for a chosen player

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

## What Is Still Missing

This now produces the feature side of the SL pipeline, but it does not yet produce the final `(features, labels)` tensor dataset.

What this phase does provide is the hard part:

- replay parsing
- deterministic playback
- state reconstruction
- observation sampling
- SL-safe feature extraction

Next steps would be:

1. define the RA2 action-label schema
2. align a dataset-wide object vocabulary and normalization pass
3. serialize replay samples plus decoded actions into final tensors
