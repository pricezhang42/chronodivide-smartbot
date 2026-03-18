# RA2 SL Model Build Plan

This note defines the recommended model-build path for RA2 supervised learning on top of the current `chronodivide-bot-sl` tensor pipeline.

It is written after re-checking the current RA2 tensor layout and the relevant mini-AlphaStar (`mAS`) code paths:

- `alphastarmini/core/arch/arch_model.py`
- `alphastarmini/core/arch/scalar_encoder.py`
- `alphastarmini/core/arch/entity_encoder.py`
- `alphastarmini/core/arch/spatial_encoder.py`
- `alphastarmini/core/arch/action_type_head.py`
- `alphastarmini/core/arch/delay_head.py`
- `alphastarmini/core/arch/queue_head.py`
- `alphastarmini/core/arch/selected_units_head.py`
- `alphastarmini/core/arch/target_unit_head.py`
- `alphastarmini/core/arch/location_head.py`
- `alphastarmini/core/sl/sl_loss_multi_gpu.py`
- `alphastarmini/core/sl/sl_train_by_tensor.py`

## Goal

Build a first trainable RA2 SL model that:

- consumes the current V1 feature tensors
- predicts the current V1 action heads
- uses the saved V1 masks and derived training targets
- is simple enough to train and debug on a single map (`Pinch Point LE`) and a single winner faction/country slice first

The first target is not “full AlphaStar for RA2”. The first target is:

- stable training
- correct masking
- overfit on a small replay subset
- reasonable validation on the current replay tensor pipeline

## Recommended Principle

Borrow the `structure` of mAS, not the full SC2-specific implementation.

Good things to reuse conceptually:

- separate scalar / entity / spatial encoders
- fused torso
- multi-head action prediction
- masked per-head loss
- teacher-forced supervised decoding
- separate free-running evaluation pass
- action-type logit masking from observed availability
- optional autoregressive argument heads

Things not to copy blindly:

- SC2-specific feature assumptions
- PySC2 `available_actions` indexing
- SC2 entity semantics
- full recurrent AlphaStar complexity before the data path is proven

## Current RA2 Inputs

The current RA2 feature pipeline already provides:

- base observation sections from `py-chronodivide`
- `scalarCore`
- `lastActionContext`
- `currentSelection`
- `currentSelectionSummary`
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

The current canonical labels already provide:

- `actionTypeId`
- `delayBin`
- `queue`
- `unitsIndices`
- `unitsMask`
- `unitsResolvedMask`
- `targetEntityIndex`
- `targetEntityResolved`
- `targetLocation`
- `targetLocationValid`
- `targetLocation2`
- `targetLocation2Valid`
- `quantity`

The current derived training targets already provide:

- one-hot action, delay, queue, units, target entity, target locations
- semantic masks
- loss masks

## Model Strategy

### Stage 1: Strong Baseline

Build a first non-recurrent RA2 SL model with:

- scalar encoder
- entity encoder
- spatial encoder
- simple fusion torso
- multi-head outputs

Rationale:

- current replay samples are action-aligned single-step tensors
- `lastActionContext` and `buildOrderTrace` already inject short temporal context
- this is much easier to debug than a full sequence model

This should be the default first implementation.

Important training-policy note:

- Stage 1 remains single-step and non-recurrent in the model core
- but the training loop should already be designed so it can later support:
  - teacher-forced decoding
  - free-running metric passes
  - sequence-window batching

### Stage 2: AlphaStar-Like Refinement

Only after Stage 1 is training clean:

- add optional sequence batching
- add a recurrent core or transformer core
- upgrade `units` to a true autoregressive selected-units head with derived EOF
- optionally make `target` heads more autoregressive

## Proposed Architecture V1

### 1. Scalar Encoder

Borrow the `idea` from mAS `ScalarEncoder`, not the exact code.

Inputs should include:

- scalar observation branch
- `lastActionContext`
- `currentSelection`
- `currentSelectionSummary`
- `availableActionMask`
- `ownedCompositionBow`
- `enemyMemoryBow`
- `buildOrderTrace`
- `techState`
- `productionState`
- `superWeaponState`

Recommended implementation:

- MLP per scalar-like block
- embedding table for `buildOrderTrace` action ids
- mean-pool or small transformer over `buildOrderTrace`
- concatenate all scalar embeddings
- project to one shared scalar embedding

### 2. Entity Encoder

Borrow from the mAS `EntityEncoder` pattern.

Recommended implementation:

- linear projection per entity row
- apply `entityMask`
- 1-2 transformer layers or masked self-attention blocks
- pooled entity summary for the global torso
- keep per-entity embeddings for:
  - selected-units head
  - target-entity head

For V1, a small transformer encoder is reasonable and much cleaner than a raw MLP over flattened entities.

### 3. Spatial Encoder

Borrow from the mAS `SpatialEncoder` pattern.

Inputs:

- `spatial`
- `minimap`
- `mapStatic`

Recommended implementation:

- stack or separately encode then fuse
- compact CNN trunk
- output:
  - global spatial embedding
  - shared spatial feature map for location heads

For V1, a light CNN is enough.

### 4. Fusion Torso

Concatenate:

- scalar embedding
- pooled entity embedding
- pooled spatial embedding

Then:

- 2-3 layer MLP torso
- one shared latent for head conditioning

Optional:

- small gating from `availableActionMask`
- residual projection

### 5. Prediction Heads

The V1 model should predict:

- `actionType`
- `delay`
- `queue`
- `units`
- `targetEntity`
- `targetLocation`
- `targetLocation2`
- `quantity`

Recommended head design:

- `actionTypeHead`
  - linear classifier over static RA2 action dict
  - apply `availableActionMask` as a logit mask during training and inference
  - keep the mask conservative and observation-driven, like mAS `available_actions`
- `delayHead`
  - classifier over 128 bins
- `queueHead`
  - binary classifier
- `unitsHead`
  - first V1 can be parallel per-slot classification over entity slots using masks
  - do not start with autoregressive EOF unless needed
- `targetEntityHead`
  - classifier over entity slots
- `targetLocationHead`
  - conv/deconv or 1x1 projection over spatial feature map to `32x32`
- `targetLocation2Head`
  - same shape as targetLocation
- `quantityHead`
  - V1 should predict raw integer value as regression or small-class classifier
  - current baseline choice: masked regression on the raw integer target
  - later, move to small-support classification if the observed quantity support remains compact enough

## Head Conditioning Strategy

The mAS batch trace confirms that one of the most useful pieces to borrow later is not just “multiple heads”, but the fact that later heads can depend on earlier argument choices.

Recommended RA2 policy:

- V1 baseline:
  - heads read from the shared fused latent
  - entity and spatial heads also read from shared entity/spatial features
  - do not require full autoregressive conditioning yet
- V1.5 / later:
  - add teacher-forced head conditioning
  - action type conditions queue / units / target heads
  - queue and selected units can condition later heads
  - selected-units head can become autoregressive with derived EOF

Current implementation state:

- partial teacher-forced conditioning is now implemented
- during training, later heads can read the gold `actionType` and `queue`
- during evaluation/inference, the same heads fall back to the model's own predicted arguments
- this improved the small held-out Arab Pinch Point slice materially, but it is still not equivalent to a full `mimic_forward`-style autoregressive decoder

So the model path should be built so that an autoregressive embedding can be inserted later without a large rewrite.

## Loss Strategy

Borrow the `masked per-head loss` idea from `sl_loss_multi_gpu.py`.

V1 loss:

- cross-entropy for:
  - `actionType`
  - `delay`
  - `queue`
  - `units`
  - `targetEntity`
  - `targetLocation`
  - `targetLocation2`
- quantity:
  - first inspect label distribution
  - if small support, classify
  - otherwise regress with mask

Use the saved training masks directly:

- `actionTypeLossMask`
- `delayLossMask`
- `queueLossMask`
- `unitsLossMask`
- `targetEntityLossMask`
- `targetLocationLossMask`
- `targetLocation2LossMask`
- `quantityLossMask`

This should be implemented in one RA2-native loss file, not by trying to route RA2 tensors through SC2 `Label.label2action`.

Chosen V1 policy for `actionType`:

- use capped `sqrt_inverse_frequency` class weights derived from the training slice
- normalize weights so the mean seen-class weight stays near `1.0`
- clamp the weights to a conservative range to avoid exploding rare-action gradients
- keep an escape hatch to disable weighting for ablations

Current practical takeaway:

- this helps the objective somewhat on the tiny Arab Pinch Point slice
- but it does not, by itself, solve the held-out combat / queue-heavy / superweapon errors
- so it should be treated as a useful baseline improvement, not the final answer

## Training Protocol

The mAS batch trace is especially useful here. The most important training-side ideas we should mirror are:

- teacher-forced supervised decoding for the gradient-bearing pass
- a separate free-running pass for metrics
- masked per-head losses driven by semantic and resolution masks

Recommended RA2 training protocol:

### V1 baseline

- one forward pass
- no autoregressive teacher forcing yet
- compute masked per-head losses directly from model outputs and saved targets

### V1.5 / later

- add a `mimic_forward`-style path for teacher-forced decoding
- use GT earlier arguments when training later heads
- run a second unguided pass for reported metrics

This distinction matters because mAS does not report purely teacher-forced accuracy. It trains with teacher forcing and evaluates with a freerunning pass. That is a good pattern for RA2 as well.

## Batch Strategy

The mAS trace also highlights sequence-window batching:

- replay -> flat rows
- rows -> overlapping `[B, S, ...]` windows
- flatten inside loss or sequence core

Recommended RA2 policy:

- V1 baseline:
  - start with sample-wise batches from the current action-aligned rows
- later:
  - add replay-window datasets
  - support `[B, S, ...]` sequence batches
  - optionally add recurrent or transformer sequence cores

For the first RA2 model, this is a deliberate simplification, not an oversight.

## Data Pipeline Plan

### 1. Dataset Layer

Create a section-aware dataset loader in `chronodivide-bot-sl`.

Recommended new files:

- `model_lib/dataset.py`
- `model_lib/batch.py`

Responsibilities:

- load `.sections.pt` and `.training.pt`
- keep structured sections instead of relying only on flat tensors
- expose:
  - model inputs
  - training targets
  - loss masks
  - shard metadata

### 2. Filtering

For the first model build, support easy filtering by:

- map name
- winner side/country
- replay file list

This matters because the current immediate target is `Pinch Point LE` with Soviet/Arab winners.

### 3. Collation

Use a custom collate function that preserves:

- entity masks
- per-slot units masks
- spatial maps
- metadata needed for debugging

## Training Script Plan

Create a separate RA2 trainer rather than forcing mAS trainer code to fit.

Recommended files:

- `model_lib/model.py`
- `model_lib/encoders.py`
- `model_lib/heads.py`
- `model_lib/losses.py`
- `train_sl_model.py`

Trainer responsibilities:

- shard discovery
- train/val split
- logging
- checkpointing
- optional mixed precision
- gradient clipping
- per-head metrics

Borrow from `sl_train_by_tensor.py`:

- overall training loop shape
- checkpoint/save cadence
- batch logging structure

Do not borrow:

- SC2-specific feature unpacking
- SC2-specific loss masking logic

## Recommended Build Order

### Phase A: Data And Contracts

1. Create a section-aware dataset loader.
2. Create a simple train/val shard splitter.
3. Add a tiny inspection script that prints:
   - input shapes
   - nonzero target counts
   - loss-mask coverage per head

### Phase B: Baseline Model

4. Implement scalar encoder.
5. Implement spatial encoder.
6. Implement a simple masked entity encoder.
7. Implement the fused torso.
8. Implement V1 heads.
9. Implement masked V1 losses.

### Phase C: Training And Overfit

10. Build `train_sl_model.py`.
11. Overfit one tiny shard or 1-2 replay slices.
12. Verify all heads can move loss downward.
13. Verify chosen actions are not being trained against impossible masks.

### Phase D: Broader Validation

14. Train on the `Pinch Point LE` + Soviet winner slice.
15. Measure:
   - action-type accuracy
   - per-head masked accuracy
   - top-k action accuracy
   - location-head accuracy
   - quantity accuracy
16. Spot-check replay predictions manually.

### Phase E: AlphaStar-Like Upgrades

17. Add a sequence core.
18. Upgrade selected-units to an autoregressive head with derived EOF.
19. Consider autoregressive dependency order:
   - action type
   - queue
   - units
   - target entity / target location

## What To Borrow From mAS First

Borrow first:

- scalar / entity / spatial encoder decomposition
- separate head decomposition
- masked supervised loss structure
- action-type availability masking
- train loop and checkpoint structure

Borrow later:

- teacher-forced multi-head decoding
- free-running metrics pass
- selected-units autoregressive logic
- recurrent core
- more AlphaStar-like conditioning between heads

Do not borrow directly:

- SC2-specific preprocessing code
- PySC2 action ids
- SC2 label objects

## Recommended Immediate Next Coding Step

Phase A is complete. The next coding step after the current encoder/torso work is:

1. add `model_lib/heads.py`
2. add `model_lib/losses.py`
3. run one-batch backward smoke tests
4. keep the implementation compatible with later:
   - availability-masked action logits
   - teacher-forced head conditioning
   - free-running metric passes
   - sequence-window batching

This is the safest path to a working RA2 SL model without overcommitting to full AlphaStar complexity too early.
