# Action-Type Weighting Experiment

Baseline run:

- [phase9_pinch_point_arabs_manifest_20260318](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_20260318)

Weighted run:

- [phase9_pinch_point_arabs_manifest_weighted_20260318](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_weighted_20260318)

Weighting policy:

- `sqrt_inverse_frequency`
- mean-normalized over seen classes
- clamped to `[0.25, 4.0]`

Training-slice weighting summary:

- [action_type_weighting.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_weighted_20260318/action_type_weighting.json)

Observed comparison on the current Arab Pinch Point manifest:

- Baseline best val loss: `22.2828`
- Weighted best val loss: `21.7430`
- So the weighted objective helped modestly on this tiny slice.

What improved:

- The held-out objective improved a little.
- The model put less extreme probability mass on a few dominant actions.
- Queue quantity rounding on the inspected `Queue::Add::NAPOWR` sample improved from predicting `2` to predicting `1`.

What did not clearly improve:

- Build-order action-type confusion remained.
- Combat samples were still misclassified as queue actions.
- Queue-heavy samples were still misclassified.
- The superweapon sample was still not recognized correctly.

Conclusion:

- Keep action-type weighting in the trainer as a useful baseline option.
- Do not expect it to solve the main held-out policy errors by itself.
- The next likely higher-value improvements are:
  - more replay data for the slice
  - teacher-forced head conditioning
  - a freerunning evaluation path
