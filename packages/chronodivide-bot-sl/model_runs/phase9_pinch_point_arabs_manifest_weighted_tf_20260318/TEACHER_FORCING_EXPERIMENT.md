# Teacher-Forced Head Conditioning Experiment

Reference runs:

- weighted baseline without teacher-forced conditioning:
  - [phase9_pinch_point_arabs_manifest_weighted_20260318](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_weighted_20260318)
- weighted run with teacher-forced `actionType` + `queue` conditioning:
  - [phase9_pinch_point_arabs_manifest_weighted_tf_20260318](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_weighted_tf_20260318)

Conditioning policy:

- training mode: `action_type_queue`
- later heads see:
  - gold `actionType` during training
  - gold `queue` during training when the action semantically uses queue
- evaluation and inspection still run from the model's own predictions

Best held-out result comparison on the current 2-shard Arab Pinch Point manifest:

- weighted baseline best val loss: `21.7430`
- weighted + teacher-forced conditioning best val loss: `19.8661`

What improved:

- held-out objective improved noticeably on this tiny slice
- early validation stabilized faster
- the training path now better matches the mAS lesson that later heads should not have to learn through noisy earlier predictions from the start

What did not clearly improve:

- qualitative spot checks are still weak on:
  - build-order action-type disambiguation
  - combat action classification
  - queue-heavy control actions
  - superweapon recognition

Conclusion:

- partial teacher-forced conditioning is a worthwhile RA2 upgrade
- it helps the learning problem more than class weighting alone
- but it still does not replace:
  - larger replay coverage
  - a true `mimic_forward`-style path
  - a separate free-running evaluation pass
