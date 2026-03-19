# RA2 SL Model Build Implementation Checklist

This checklist tracks the implementation work for the first trainable RA2 supervised-learning model in `chronodivide-bot-sl`.

Related design note:

- [RA2_SL_MODEL_BUILD_PLAN.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_MODEL_BUILD_PLAN.md)

## Status Summary

- `[x]` Model-build plan written
- `[x]` Section-aware model dataset implemented
- `[x]` Baseline RA2 SL model implemented
- `[x]` Masked SL losses implemented
- `[x]` Trainer implemented
- `[x]` Tiny-slice overfit check passed
- `[x]` Pinch Point Soviet-slice training run passed
- `[x]` Sequence-window batching implemented
- `[x]` Optional LSTM core implemented

## Phase 0: Contracts And Scope

- `[x]` Re-check current RA2 tensor pipeline and confirm the model should consume structured sections, not only flat tensors.
- `[x]` Re-check relevant mAS architecture and SL loss files.
- `[x]` Write the initial model-build design note.
- `[ ]` Freeze the initial V1 training scope:
  - single-step supervised model
  - no recurrent core in the first version
  - the original first version started without AlphaStar-style autoregressive selected-units decoding
  - the current implementation now includes a derived EOF-autoregressive `units` head
- `[ ]` Freeze the initial training slice:
  - map = `2_pinch_point_le.map`
  - winner side/country filtering policy
  - replay inclusion/exclusion policy

## Phase 1: Dataset And Batch Layer

- `[x]` Create `model_lib/` package under `chronodivide-bot-sl`.
- `[x]` Add `model_lib/dataset.py`.
- `[x]` Add `model_lib/batch.py`.
- `[x]` Add `model_lib/__init__.py`.
- `[x]` Implement shard discovery over generated tensor outputs.
- `[x]` Load `.sections.pt` sidecars directly for model inputs.
- `[x]` Load `.training.pt` sidecars directly for targets and masks.
- `[x]` Load `.meta.json` for schema and run metadata.
- `[x]` Expose a sample object with:
  - feature sections
  - label sections
  - training targets
  - training masks
  - replay/player metadata
- `[x]` Implement a collate function that preserves:
  - scalar sections
  - entity tensors and masks
  - spatial / minimap / mapStatic tensors
  - per-slot selected-unit masks
  - per-head loss masks
- `[x]` Verify dataloader batch shapes on a real saved shard.
- `[x]` Add a small inspection script that prints:
  - feature section names and shapes
  - target section names and shapes
  - mask coverage per head

## Phase 2: Input Schema Freeze

- `[ ]` Freeze the exact model input mapping from saved sections.
- `[ ]` Decide which sections go into the scalar encoder:
  - `scalar`
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
- `[ ]` Freeze the entity input mapping:
  - entity tensor
  - entity mask
- `[ ]` Freeze the spatial input mapping:
  - `spatial`
  - `minimap`
  - `mapStatic`
- `[ ]` Freeze the output target mapping:
  - `actionType`
  - `delay`
  - `queue`
  - `units`
  - `targetEntity`
  - `targetLocation`
  - `targetLocation2`
  - `quantity`
- `[ ]` Freeze the V1 training-policy distinction:
  - baseline training is non-autoregressive
  - full multi-head teacher-forced decoding is implemented
  - later free-running evaluation is a planned upgrade

## Phase 3: Baseline Encoders

### Scalar Encoder

- `[x]` Add `model_lib/encoders.py`.
- `[x]` Implement a scalar-block encoder for dense 1D feature sections.
- `[x]` Implement `buildOrderTrace` embedding:
  - embedding table over static `actionTypeId`
  - pooling or small transformer
- `[x]` Concatenate scalar-section embeddings and project to one scalar latent.
- `[x]` Verify scalar encoder output shape and NaN safety.

### Entity Encoder

- `[x]` Implement a masked entity encoder.
- `[x]` Start with:
  - linear projection per entity row
  - 1-2 transformer/self-attention layers or a simpler masked MLP baseline
- `[x]` Return:
  - per-entity embeddings
  - pooled entity summary
- `[x]` Verify entity masks are respected.

### Spatial Encoder

- `[x]` Implement a CNN spatial encoder for:
  - `spatial`
  - `minimap`
  - `mapStatic`
- `[x]` Return:
  - pooled spatial summary
  - shared spatial feature map for location heads
- `[x]` Verify shape consistency with saved spatial sizes.

## Phase 4: Fusion Torso

- `[x]` Add `model_lib/model.py`.
- `[x]` Fuse:
  - scalar latent
  - pooled entity latent
  - pooled spatial latent
- `[x]` Implement a simple shared torso MLP.
- `[x]` Keep the fusion model non-recurrent for V1.
- `[x]` Verify end-to-end forward pass on one batch.

## Phase 5: Output Heads

- `[x]` Add `model_lib/heads.py`.
- `[x]` Implement `actionTypeHead`.
- `[x]` Implement `delayHead`.
- `[x]` Implement `queueHead`.
- `[x]` Implement `unitsHead`.
- `[x]` Implement `targetEntityHead`.
- `[x]` Implement `targetLocationHead`.
- `[x]` Implement `targetLocation2Head`.
- `[x]` Implement `quantityHead`.

### Head Policy

- `[x]` `actionTypeHead` uses the static RA2 action dict size.
- `[x]` `actionTypeHead` supports availability masking from `availableActionMask`.
- `[x]` `unitsHead` now uses derived EOF-autoregressive decoding while leaving canonical replay tensors unchanged.
- `[x]` `targetLocation` heads produce `64 x 64` logits aligned with native saved training targets.
- `[x]` Decide whether V1 heads are:
  - pure shared-latent heads
  - or partially conditioned on predicted/teacher-forced earlier arguments
- `[x]` Decide and document `quantityHead` type:
  - raw regression
  - or small-support classification

## Phase 6: Losses And Metrics

- `[x]` Add `model_lib/losses.py`.
- `[x]` Implement masked cross-entropy for:
  - `actionType`
  - `delay`
  - `queue`
  - `units`
  - `targetEntity`
  - `targetLocation`
  - `targetLocation2`
- `[x]` Implement masked loss for `quantity`.
- `[x]` Use saved loss masks directly from `.training.pt`.
- `[x]` Apply `availableActionMask` as an action-type logit mask in loss/inference code.
- `[x]` Decide and document action-type loss weighting policy:
  - uniform
  - or mAS-style category-aware weighting
- `[x]` Add per-head metrics:
  - action accuracy
  - delay accuracy
  - queue accuracy
  - units masked accuracy
  - target entity accuracy
  - target location accuracy
  - quantity accuracy
- `[x]` Verify zero-loss-mask rows do not contribute to loss.
- `[ ]` Add a separate free-running metrics path after the training-loss pass.

## Phase 7: Training Script

- `[x]` Add `train_sl_model.py`.
- `[x]` Implement config/CLI for:
  - shard root
  - replay filtering
  - batch size
  - learning rate
  - epochs
  - train/val split
  - checkpoint dir
- `[x]` Add optimizer and scheduler setup.
- `[x]` Add checkpoint save/load.
- `[x]` Add train/val loops.
- `[x]` Keep the training loop structured so it can later support:
  - teacher-forced training forward pass
  - free-running metrics forward pass
  - sequence-window batches
- `[x]` Add logging for:
  - total loss
  - per-head loss
  - per-head accuracy
  - examples/sec or samples/sec
- `[x]` Validate a real training smoke run on the current Arab `Pinch Point LE` shard slice and save checkpoints plus run metadata.

## Phase 8: Tiny-Slice Debugging

- `[x]` Run one-batch forward pass successfully.
- `[x]` Run one-batch backward pass successfully.
- `[x]` Overfit on one tiny shard or a very small replay subset.
- `[x]` Confirm training loss decreases sharply.
- `[x]` Confirm `actionType` head can memorize a tiny subset.
- `[x]` Confirm location heads learn non-trivial targets on a tiny subset.
- `[x]` Confirm masked heads do not produce unstable NaNs.
- `[x]` Confirm action-type masking does not suppress chosen gold actions.
- `[ ]` Confirm free-running metrics can be computed without teacher forcing once enabled.

## Phase 9: Pinch Point Soviet Slice

- `[x]` Build a reproducible training manifest for `Pinch Point LE`.
- `[x]` Decide exact Soviet-country filter:
  - all Soviet winners
  - or `Arabs` only
- `[x]` Train on the chosen slice.
- `[x]` Record:
  - replay count
  - shard count
  - sample count
  - map count
  - country distribution
- `[x]` Run validation on held-out shards.
- `[x]` Save one model checkpoint and one training report.

## Phase 10: Spot Checks

- `[x]` Add a small prediction-inspection script.
- `[x]` Print top-k `actionType` predictions for real samples.
- `[x]` Compare predicted vs gold `queue`, `targetEntity`, and `targetLocation`.
- `[x]` Spot-check:
  - build-order samples
  - combat samples
  - queue-heavy samples
  - superweapon samples

## Phase 11: Optional AlphaStar-Like Upgrades

- `[x]` Add sequence batching if the baseline is stable.
- `[x]` Add an optional recurrent or transformer core.
- `[x]` Add replay-window dataset items with `[B, S, ...]` batch support.
- `[x]` Add partial teacher-forced head conditioning for later heads:
  - gold `actionType` may condition later heads during training
  - gold `queue` may condition later heads during training
  - evaluation/inference still runs free from the model's own predictions
- `[x]` Add a full multi-head teacher-forced training path:
  - gold `actionType`, `delay`, `queue`, `units`, `targetEntity`, `targetLocation`, and `targetLocation2` can condition later heads when valid
  - unresolved targets do not inject teacher-forced garbage because conditioning is masked by valid supervision
- `[ ]` Add a separate free-running evaluation forward pass.
- `[x]` Add a derived EOF-based autoregressive `units` head.
- `[ ]` Add stronger autoregressive coupling between heads.
- `[ ]` Re-evaluate whether the extra complexity improves validation.

## Important Non-Goals For V1

- `[ ]` Do not try to port SC2 preprocessing code directly.
- `[ ]` Do not try to reuse PySC2 action ids or SC2 label objects.
- `[ ]` Do not start with full AlphaStar recurrence before the baseline is proven.

## Recommended Immediate Next Step

- `[ ]` Choose one of the next two high-value directions:
  - expand the Arab `Pinch Point LE` tensor corpus beyond the current 2-shard manifest
  - add a separate free-running evaluation pass and compare it directly with the teacher-forced training pass
