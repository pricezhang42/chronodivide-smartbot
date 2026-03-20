# V1 vs V2 Debug Comparison on Arab Pinch Point Winner-Only Slice

Date: `2026-03-20`

Slice:

- source replays: `5`
- map: `2_pinch_point_le.map`
- country: `Arabs`
- winner-only manifest: [pinch_point_arabs_winner5_v2compare_20260319.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/manifests/pinch_point_arabs_winner5_v2compare_20260319.json)

Runs:

- V1 baseline: [v1_vs_v2_compare_winner5_v1_20260319](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v1_20260319)
- V2 debug: [v1_vs_v2_compare_winner5_v2_20260319](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_20260319)
- V2 debug with stable shared buildable-object vocabulary: [v1_vs_v2_compare_winner5_v2_stable_vocab_20260319](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_stable_vocab_20260319)
- V2 debug with stable vocabulary plus action-family weighting: [v1_vs_v2_compare_winner5_v2_weighted_20260319](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_weighted_20260319)

## V1

Key metrics from [training_summary.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v1_20260319/training_summary.json):

- best `val_loss`: `24.3899` at epoch `1`
- best `val_free_action`: `0.2198`
- best `val_free_full`: `0.0172` at epoch `0`
- final `goldActionSuppressedRate`: `0.0656`

Notes:

- V1 still includes standalone `SelectUnits` in the main action space.
- On this slice, the free-running action-type metric plateaued almost immediately.

## V2 Debug

Key metrics from [training_summary.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_20260319/training_summary.json):

- best `val_loss`: `26.1242`
- final `val_free_family`: `0.7555`
- final `val_free_full`: `0.0000`

Notes:

- V2 folds out `SelectUnitsAction` and drops `Hold` / `Resume`.
- V2 trains on a larger kept command stream than V1 on the same replay slice.
- Cross-replay batching initially failed because `buildableObjectOneHot` width is replay-local; the debug path now zero-pads variable-width sections at collate time.

## Action Family Prior Check

Observed V2 action-family counts across the five winner shards:

- `Order`: `791`
- `Queue`: `186`
- `PlaceBuilding`: `47`
- `ActivateSuperWeapon`: `6`
- `SellObject`: `3`

Total V2 command rows: `1033`

So the dominant-family prior is:

- `Order`: `791 / 1033 = 0.7657`

This is very close to the observed V2 free-running family accuracy (`0.7555`), which suggests the current V2 debug model is mostly learning the dominant top-level family prior on this small slice.

## Takeaways

1. The V2 end-to-end path is real and trainable across multiple replay shards.
2. Folding out standalone selection and disabling `Hold` / `Resume` did not break training.
3. The next real blocker is not V2 plumbing anymore.
4. The next blocker is shared cross-replay vocabularies, especially `buildableObjectToken`.
5. After that, the next comparison should inspect per-family confusion, not just overall family accuracy.

## Stable Vocabulary + Weighted V2 Follow-Up

Key artifacts:

- weighting report: [action_family_weighting.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_weighted_20260319/action_family_weighting.json)
- weighted summary: [training_summary.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_weighted_20260319/training_summary.json)
- weighted family eval: [family_eval_report.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_weighted_20260319/family_eval_report.json)

Observed V2 action-family weights on the train split:

- `Order`: `0.2155`
- `Queue`: `0.4796`
- `PlaceBuilding`: `0.8884`
- `ActivateSuperWeapon`: `1.7083`
- `SellObject`: `1.7083`

Weighted V2 result:

- best `val_loss`: `25.3514`
- final `val_free_family`: `0.7555`
- final `val_free_full`: `0.0000`

Free-running family confusion on the validation rows stayed fully collapsed:

- targets:
  - `Order`: `406`
  - `Queue`: `108`
  - `PlaceBuilding`: `24`
  - `ActivateSuperWeapon`: `3`
  - `SellObject`: `1`
- predictions:
  - `Order`: `542`

So the stable shared vocabulary fix succeeded, and the action-family weighting machinery is working, but on this tiny slice the weighted V2 debug model still predicts `Order` for every validation sample.

Updated takeaway:

1. The remaining blocker is no longer V2 artifact plumbing.
2. Simple action-family reweighting alone is not enough to break the dominant `Order` prior on this slice.
3. The next best experiment should be one of:
   - a larger Arab Pinch Point comparison corpus
   - stronger family balancing, such as weighted sampling or family-balanced minibatches
   - a V2 trainer change that selects checkpoints by free-running per-family behavior rather than only `val_loss`

## Stable Vocabulary + Weighted Sampling Follow-Up

Key artifacts:

- balanced run summary: [training_summary.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_balanced_20260319/training_summary.json)
- balanced split config: [data_split.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_balanced_20260319/data_split.json)
- balanced family eval: [family_eval_report.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/v1_vs_v2_compare_winner5_v2_balanced_20260319/family_eval_report.json)

Balanced V2 result:

- best `val_loss`: `24.9671`
- final `val_free_family`: `0.7665`
- best observed free-running family accuracy during the run: `0.7739`

Free-running family confusion with balanced sampling:

- targets:
  - `Order`: `406`
  - `Queue`: `108`
  - `PlaceBuilding`: `24`
  - `ActivateSuperWeapon`: `3`
  - `SellObject`: `1`
- predictions:
  - `Order`: `515`
  - `Queue`: `27`

Important change from the plain weighted run:

- the model no longer predicts `Order` for every validation sample
- it now recovers `16 / 108` `Queue` rows as `Queue`
- it still misses all `PlaceBuilding`, `ActivateSuperWeapon`, and `SellObject` rows on this slice

So the first useful V2 family-balancing intervention is not plain loss weighting; it is loss weighting plus family-balanced sampling. That is still only a partial fix, but it is the first V2 run on this slice that breaks the pure-majority-family collapse.
