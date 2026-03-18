# RA2 SL Model Build Implementation Checklist

This checklist tracks the implementation work for the first trainable RA2 supervised-learning model in `chronodivide-bot-sl`.

Related design note:

- [RA2_SL_MODEL_BUILD_PLAN.md](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/RA2_SL_MODEL_BUILD_PLAN.md)

## Status Summary

- `[x]` Model-build plan written
- `[x]` Section-aware model dataset implemented
- `[ ]` Baseline RA2 SL model implemented
- `[ ]` Masked SL losses implemented
- `[ ]` Trainer implemented
- `[ ]` Tiny-slice overfit check passed
- `[ ]` Pinch Point Soviet-slice training run passed

## Phase 0: Contracts And Scope

- `[x]` Re-check current RA2 tensor pipeline and confirm the model should consume structured sections, not only flat tensors.
- `[x]` Re-check relevant mAS architecture and SL loss files.
- `[x]` Write the initial model-build design note.
- `[ ]` Freeze the initial V1 training scope:
  - single-step supervised model
  - no recurrent core in the first version
  - no AlphaStar-style autoregressive selected-units head in the first version
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

## Phase 3: Baseline Encoders

### Scalar Encoder

- `[ ]` Add `model_lib/encoders.py`.
- `[ ]` Implement a scalar-block encoder for dense 1D feature sections.
- `[ ]` Implement `buildOrderTrace` embedding:
  - embedding table over static `actionTypeId`
  - pooling or small transformer
- `[ ]` Concatenate scalar-section embeddings and project to one scalar latent.
- `[ ]` Verify scalar encoder output shape and NaN safety.

### Entity Encoder

- `[ ]` Implement a masked entity encoder.
- `[ ]` Start with:
  - linear projection per entity row
  - 1-2 transformer/self-attention layers or a simpler masked MLP baseline
- `[ ]` Return:
  - per-entity embeddings
  - pooled entity summary
- `[ ]` Verify entity masks are respected.

### Spatial Encoder

- `[ ]` Implement a CNN spatial encoder for:
  - `spatial`
  - `minimap`
  - `mapStatic`
- `[ ]` Return:
  - pooled spatial summary
  - shared spatial feature map for location heads
- `[ ]` Verify shape consistency with saved spatial sizes.

## Phase 4: Fusion Torso

- `[ ]` Add `model_lib/model.py`.
- `[ ]` Fuse:
  - scalar latent
  - pooled entity latent
  - pooled spatial latent
- `[ ]` Implement a simple shared torso MLP.
- `[ ]` Keep the fusion model non-recurrent for V1.
- `[ ]` Verify end-to-end forward pass on one batch.

## Phase 5: Output Heads

- `[ ]` Add `model_lib/heads.py`.
- `[ ]` Implement `actionTypeHead`.
- `[ ]` Implement `delayHead`.
- `[ ]` Implement `queueHead`.
- `[ ]` Implement `unitsHead`.
- `[ ]` Implement `targetEntityHead`.
- `[ ]` Implement `targetLocationHead`.
- `[ ]` Implement `targetLocation2Head`.
- `[ ]` Implement `quantityHead`.

### Head Policy

- `[ ]` `actionTypeHead` uses the static RA2 action dict size.
- `[ ]` `actionTypeHead` supports availability masking from `availableActionMask`.
- `[ ]` `unitsHead` V1 is slot-wise masked classification, not autoregressive EOF decoding.
- `[ ]` `targetLocation` heads produce `32 x 32` logits aligned with saved training targets.
- `[ ]` Decide and document `quantityHead` type:
  - raw regression
  - or small-support classification

## Phase 6: Losses And Metrics

- `[ ]` Add `model_lib/losses.py`.
- `[ ]` Implement masked cross-entropy for:
  - `actionType`
  - `delay`
  - `queue`
  - `units`
  - `targetEntity`
  - `targetLocation`
  - `targetLocation2`
- `[ ]` Implement masked loss for `quantity`.
- `[ ]` Use saved loss masks directly from `.training.pt`.
- `[ ]` Add per-head metrics:
  - action accuracy
  - delay accuracy
  - queue accuracy
  - units masked accuracy
  - target entity accuracy
  - target location accuracy
  - quantity accuracy
- `[ ]` Verify zero-loss-mask rows do not contribute to loss.

## Phase 7: Training Script

- `[ ]` Add `train_sl_model.py`.
- `[ ]` Implement config/CLI for:
  - shard root
  - replay filtering
  - batch size
  - learning rate
  - epochs
  - train/val split
  - checkpoint dir
- `[ ]` Add optimizer and scheduler setup.
- `[ ]` Add checkpoint save/load.
- `[ ]` Add train/val loops.
- `[ ]` Add logging for:
  - total loss
  - per-head loss
  - per-head accuracy
  - examples/sec or samples/sec

## Phase 8: Tiny-Slice Debugging

- `[ ]` Run one-batch forward pass successfully.
- `[ ]` Run one-batch backward pass successfully.
- `[ ]` Overfit on one tiny shard or a very small replay subset.
- `[ ]` Confirm training loss decreases sharply.
- `[ ]` Confirm `actionType` head can memorize a tiny subset.
- `[ ]` Confirm location heads learn non-trivial targets on a tiny subset.
- `[ ]` Confirm masked heads do not produce unstable NaNs.

## Phase 9: Pinch Point Soviet Slice

- `[ ]` Build a reproducible training manifest for `Pinch Point LE`.
- `[ ]` Decide exact Soviet-country filter:
  - all Soviet winners
  - or `Arabs` only
- `[ ]` Train on the chosen slice.
- `[ ]` Record:
  - replay count
  - shard count
  - sample count
  - map count
  - country distribution
- `[ ]` Run validation on held-out shards.
- `[ ]` Save one model checkpoint and one training report.

## Phase 10: Spot Checks

- `[ ]` Add a small prediction-inspection script.
- `[ ]` Print top-k `actionType` predictions for real samples.
- `[ ]` Compare predicted vs gold `queue`, `targetEntity`, and `targetLocation`.
- `[ ]` Spot-check:
  - build-order samples
  - combat samples
  - queue-heavy samples
  - superweapon samples

## Phase 11: Optional AlphaStar-Like Upgrades

- `[ ]` Add sequence batching if the baseline is stable.
- `[ ]` Add an optional recurrent or transformer core.
- `[ ]` Add a derived EOF-based autoregressive `units` head.
- `[ ]` Add stronger autoregressive coupling between heads.
- `[ ]` Re-evaluate whether the extra complexity improves validation.

## Important Non-Goals For V1

- `[ ]` Do not try to port SC2 preprocessing code directly.
- `[ ]` Do not try to reuse PySC2 action ids or SC2 label objects.
- `[ ]` Do not start with full AlphaStar recurrence before the baseline is proven.

## Recommended Immediate Next Step

- `[ ]` Implement Phase 1:
  - `model_lib/dataset.py`
  - `model_lib/batch.py`
  - tensor inspection script
