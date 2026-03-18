# AI Agent Handoff

This note is a compact handoff for another AI agent taking over work on the RA2 supervised-learning data pipeline.

## Project Goal

Build a Red Alert 2 replay-to-supervised-learning pipeline analogous in role to mini-AlphaStar:

- `py-chronodivide` should play the role of `pysc2`
- `chronodivide-bot-sl` should play the role of `alphastarmini/core/sl/transform_replay_data.py`

That means:

- `py-chronodivide` owns replay parsing, deterministic playback, safe observation extraction, generic feature primitives, and generic action decoding
- `chronodivide-bot-sl` owns SL-specific label layout, feature layout, action filtering/downsampling, flattening, tensor writing, and audits

## Package Boundaries

Keep this split strict:

- `D:\workspace\supalosa-chronodivide-bot\packages\py-chronodivide`
  - generic replay playback
  - generic game-state and observation recording
  - generic feature extraction primitives
  - generic replay action decoding
  - no project-specific SL action dict
- `D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl`
  - static SL action dict
  - SL label layout and feature layout
  - mAS-style action filtering/downsampling
  - replay transformer
  - training-target derivation
  - audits and manifests

## What Was Established

### mini-AlphaStar parity work

We first traced how mAS supervised learning works:

- replay preprocessing
- action-centric observation/label pairing
- teacher-forced autoregressive training
- labels for `action_type`, `delay`, `queue`, `units`, `target_unit`, `target_location`

Useful reference docs in mini-AlphaStar:

- [SL_BATCH_TRACE.MD](D:/workspace/mini-AlphaStar/doc/SL_BATCH_TRACE.MD)
- [transform_replay_data.py](D:/workspace/mini-AlphaStar/alphastarmini/core/sl/transform_replay_data.py)
- [action_dict.py](D:/workspace/mini-AlphaStar/alphastarmini/third/action_dict.py)

### Replay reconstruction and recording

In `py-chronodivide`, replay playback was validated as deterministic and usable for SL:

- replay parsing works
- deterministic playback was validated with roundtrip equality
- safe player observation extraction exists
- hidden enemy data leaks through global engine APIs, so safe observation paths must be used consistently

Important files:

- [bridge.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/bridge.mjs)
- [resim_core.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/resim_core.mjs)
- [snapshot.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/snapshot.mjs)
- [features.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/features.mjs)
- [labels.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/labels.mjs)

Reference notes:

- [REPLAY_RECORDING_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/REPLAY_RECORDING_CHECKLIST.md)
- [RA2_REPLAY_VS_SC2_SL_GAPS.md](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/RA2_REPLAY_VS_SC2_SL_GAPS.md)

### RA2 labels

We designed and implemented a stable V1 RA2 SL label layout:

- static action dict in SL package only
- fine-grained `action_type` space analogous to mAS raw ability space
- canonical V1 label sections
- separate derived model-ready training targets and masks

Important files:

- [action_dict.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/action_dict.py)
- [RA2_SL_LABEL_LAYOUT_V1.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V1.md)
- [RA2_SL_LABEL_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md)
- [audit_action_dict.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/audit_action_dict.py)
- [audit_training_targets.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/audit_training_targets.py)

Current state:

- static action dict version is `ra2_sl_v4`
- raw replay action ids observed: 11
- fine-grained SL action types: 510
- corpus action-dict audit reached 0 fallback rate on the currently loadable replay/map slice

Reference audit:

- [action_dict_audit_100_v4_20260317.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/action_dict_audit_100_v4_20260317.json)

### RA2 features

We designed and partially implemented a V1 feature layout that goes beyond the original minimal feature set and closes major gaps vs mAS.

Core docs:

- [RA2_FEATURE_LAYOUT_GAP_ANALYSIS.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_FEATURE_LAYOUT_GAP_ANALYSIS.md)
- [RA2_SL_FEATURE_LAYOUT_V1.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_FEATURE_LAYOUT_V1.md)
- [RA2_SL_FEATURE_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_FEATURE_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md)

Implemented feature-layout items so far:

- `availableActionMask`
- `ownedCompositionBow`
- scalar identity in `scalarCore` for side/country
- `buildOrderTrace`
- `techState`
- `productionState`
- `enemyMemoryBow`
- `mapStatic`
- `superWeaponState`
- entity intent summaries

### Transformer and training targets

The SL transformer now exists in Python and writes:

- replay-player `.pt` shards with flat `(features, labels)`
- `.sections.pt` sidecars with structured sections
- `.training.pt` sidecars with model-ready targets and masks

Entry point:

- [transform_replay_data.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_replay_data.py)

Refactored helpers:

- [common.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/common.py)
- [action_labels.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/action_labels.py)
- [schema_utils.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/schema_utils.py)
- [training_targets.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/training_targets.py)
- [filtering.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/filtering.py)
- [feature_layout.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/feature_layout.py)

The transformer is already aligned to the mAS broad flow:

- replay iteration
- action-centric observation/label pairing
- previous-action context
- delay supervision
- mAS-style action filtering/downsampling
- replay-player tensor shards

Reference parity note:

- [TRANSFORM_REPLAY_ALIGNMENT_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/TRANSFORM_REPLAY_ALIGNMENT_CHECKLIST.md)

## Performance Work

The biggest performance changes already landed:

- extractor cache in the Python transformer
- streamed JSON path to remove giant-string failure
- binary handoff replacing the old huge JSON dataset payload

Important files:

- [extract_sl_tensors.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/extract_sl_tensors.mjs)
- [transform_replay_data.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_replay_data.py)

Important current invariant:

- extractor cache version should remain in sync with the binary handoff contract

Observed timings on `00758dde-b725-4442-ae8f-a657069251a0.rpl`:

- old JSON path: roughly `306s` cold, `178s` warm
- binary handoff path: roughly `125.5s` cold, `22.3s` warm

Reference output:

- [binary_handoff_benchmark_cold_20260318](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/generated_tensors/binary_handoff_benchmark_cold_20260318)

## Most Important Rules For Future Work

### Anti-leak rule

Never use omniscient game state as an SL input feature.

Allowed:

- player-safe observation snapshots
- visible enemy entities
- visible tiles
- self player state
- replay-global metadata like side/country

Not allowed as input features:

- `getAllUnits()` style hidden enemy access
- enemy player state from omniscient APIs
- exact alive enemy state behind fog

Observation-safe memory features are allowed if they are based only on what was previously visible.

### Package-boundary rule

If a field is generic and reusable across projects, expose it in `py-chronodivide`.

If a field is an SL-specific encoding choice, keep it in `chronodivide-bot-sl`.

Examples:

- generic raw transient entity fields belong in `py-chronodivide`
- heuristic intent-summary encoding belongs in `chronodivide-bot-sl`
- static SL action dict belongs in `chronodivide-bot-sl`

### Stability rule

Do not make schema widths depend on replay-local discovery.

Use:

- static action dict
- static vocabularies where possible
- unknown buckets when a stable fixed width is more important than perfect specificity

## Current Validated Outputs

Useful generated outputs already present:

- label/feature smoke outputs:
  - [static_action_dict_v4_smoke_20260317](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/generated_tensors/static_action_dict_v4_smoke_20260317)
  - [training_audit_5replays_v4_fix_20260317](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/generated_tensors/training_audit_5replays_v4_fix_20260317)
  - [entity_intent_smoke_20260318](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/generated_tensors/entity_intent_smoke_20260318)
- replay recordings:
  - [generated_recordings](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/generated_recordings)

The latest entity-intent smoke run confirmed:

- saved shard schema includes the new entity-intent channels
- `entityFeatures` count is now 74 on that run
- flat feature width was `108283`
- real nonzero summary channels appeared for:
  - `intent_attack`
  - `intent_build`
  - `intent_factory_delivery`
  - `intent_target_mode_object`
  - `intent_progress_01`
  - `weapon_ready_any`
  - `weapon_cooldown_progress_01`

Reference files:

- [00758dde-b725-4442-ae8f-a657069251a0__bikerush.meta.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/generated_tensors/entity_intent_smoke_20260318/00758dde-b725-4442-ae8f-a657069251a0__bikerush.meta.json)
- [00758dde-b725-4442-ae8f-a657069251a0__bikerush.sections.pt](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/generated_tensors/entity_intent_smoke_20260318/00758dde-b725-4442-ae8f-a657069251a0__bikerush.sections.pt)

## Known Gaps / Cautions

- Some replay files still cannot be processed because required maps are missing from the data pack.
- Current missing-map set observed during audits:
  - `offensedefense.map`
  - `mp01t4.map`
  - `2_poltergeist_precaptured.map`
  - `lostlake.map`
- Some feature sections are implemented but only lightly validated on a small replay slice.
- `enemyMemoryBow` is memory of seen enemies, not an exact alive-enemy estimate.
- Entity intent summaries are heuristic because Chronodivide does not expose a clean generic current-order field.

## How To Run The Project

Use the conda environment:

```powershell
conda activate chronodivide-ml
```

### Run the Python transformer

```powershell
python D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\transform_replay_data.py `
  --data-dir D:\workspace\ra2-headless-mix `
  --replay-dir D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\ladder_replays_top50 `
  --output-dir D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\generated_tensors\my_run `
  --player all `
  --max-replays 1 `
  --max-actions 64
```

Notes:

- this uses the binary handoff by default
- extractor cache lives under `<output_dir>\\_extract_cache`
- warm reruns are much faster than cold reruns

### Run action-dict audit

```powershell
python D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\audit_action_dict.py
```

### Run training-target audit

```powershell
python D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\audit_training_targets.py `
  --run-dir D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\generated_tensors\my_run
```

## Recommended Next Steps

Highest-value next steps:

1. Finish Phase 15 in [RA2_SL_FEATURE_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_FEATURE_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md):
   - improve the current selection summary
2. Add feature audit scripts for the new V1 sections:
   - sparsity
   - first nonzero timing
   - occupancy / unknown-bucket rates
3. Broaden validation across more replay motifs:
   - harvest cycles
   - rally behavior
   - repair
   - movement-heavy slices
4. Add manifest summaries for `enemyMemoryBow` and composition features
5. Eventually build training code in RA2 that consumes the saved `.pt` shards similarly to mAS `sl_train_by_tensor.py`

## Working Style Guidance For The Next Agent

- Read the checklist and design doc before changing any V1 layout.
- Prefer extending `py-chronodivide` only when the added field is genuinely generic.
- Keep tensor widths stable.
- Always validate on at least one real replay after a schema change.
- After schema changes, inspect:
  - `.meta.json`
  - `.sections.pt`
  - `.training.pt`
- If you change action dict, label layout, or feature layout, update the matching markdown spec and checklist in the same turn.
- If a replay transform becomes slow again, first check:
  - whether extractor cache is being reused
  - whether binary handoff is still active
  - whether a new pass is rebuilding large Python objects unnecessarily

## Best Entry Points For A New Agent

Start here:

- [transform_replay_data.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_replay_data.py)
- [feature_layout.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/transform_lib/feature_layout.py)
- [action_dict.py](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/action_dict.py)
- [RA2_SL_FEATURE_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_FEATURE_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md)
- [RA2_SL_LABEL_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_LABEL_LAYOUT_V1_IMPLEMENTATION_CHECKLIST.md)

If more replay-core work is needed, continue in:

- [resim_core.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/resim_core.mjs)
- [snapshot.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/snapshot.mjs)
- [sl_dataset.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/sl_dataset.mjs)

