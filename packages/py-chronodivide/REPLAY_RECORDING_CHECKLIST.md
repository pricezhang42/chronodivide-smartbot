# Replay Recording Checklist

This checklist tracks how close `py-chronodivide` is to a `pysc2`-style replay-recording layer for Chronodivide.

Legend:

- `[x]` implemented and validated on real replays
- `[~]` partially implemented or available only through lower-level hooks
- `[ ]` not exported yet

## Replay Core

- `[x]` Parse `.rpl` files into replay objects
- `[x]` Deterministically re-simulate recorded turns
- `[x]` Rebuild global game state snapshots at sampled ticks
- `[x]` Rebuild player-view observation snapshots at sampled ticks
- `[x]` Keep raw replay event access inside the replay context
- `[~]` Normalize non-turn replay events into public JSON exports

## Dynamic Game State

- `[x]` Current tick / game time / tick rate / base tick rate
- `[x]` Map size, theater type, and starting locations
- `[x]` Per-player credits, power, radar, combatant / observer state
- `[x]` Per-player defeated / resigned / dropped / score state
- `[x]` Per-player production queues and available objects
- `[x]` Per-player economy / combat counters:
  - credits gained
  - buildings captured
  - crates picked up
  - units built / killed / lost by type
  - limited units built by name
- `[x]` Full unit / building snapshots
- `[x]` Neutral unit snapshots
- `[x]` Terrain object snapshots
- `[x]` Tile-resource snapshots
- `[x]` Super-weapon snapshots

## Unit / Object Fields

- `[x]` IDs, names, owners, object type, tile/world position, elevation
- `[x]` HP / max HP
- `[x]` foundation size
- `[x]` sight
- `[x]` veteran level
- `[x]` guard mode
- `[x]` purchase value
- `[x]` movement / stance / bridge / zone data exposed by `GameApi`
- `[x]` build / factory / rally-point state
- `[x]` ammo / transport / garrison / repair-related state
- `[x]` weapon summaries
- `[~]` Every engine-exposed transient field for every object category

## Observation Recording

- `[x]` Visible self / allied / enemy unit sets
- `[x]` Visible neutral and other-hostile unit sets
- `[x]` Visible tile count
- `[x]` Optional visible tile coordinate list
- `[x]` Optional visible resource tile list
- `[x]` Observation-safe scalar / entity / spatial / minimap feature encoding
- `[x]` Explicit separation between safe player-view queries and leaky omniscient queries

## Static Data

- `[x]` `GeneralRules` snapshot
- `[x]` Structured `rules.ini` snapshot
- `[x]` Structured `art.ini` snapshot
- `[x]` Structured `ai.ini` snapshot
- `[~]` Exact original file-text preservation in exported replay samples

## Replay Action Data

- `[x]` Decode core action stream from replay turns
- `[x]` Decode select / order / queue / place / sell / repair / superweapon actions
- `[x]` Recover action timing and inferred selection state
- `[~]` Export every replay event category as a normalized record, not just action-centric ones

## Current Known Gaps

- `[ ]` Public normalized export for non-turn replay events such as chat / other replay-side metadata events
- `[ ]` Full static map dump beyond the current map summary and sampled terrain-object records
- `[ ]` Framework-native tensor shard writers such as `.pt` / `.npz`
- `[ ]` Dataset-transformer package in `chronodivide-bot-sl` that consumes the reusable `py-chronodivide` APIs

## Practical Conclusion

For replay reconstruction, `py-chronodivide` now covers the core data needed for:

- deterministic playback
- rich state recording
- player-correct observation extraction
- SL feature encoding
- SL label decoding

What is still missing is mostly packaging and completeness around auxiliary replay/static exports, not the main re-simulation path.
