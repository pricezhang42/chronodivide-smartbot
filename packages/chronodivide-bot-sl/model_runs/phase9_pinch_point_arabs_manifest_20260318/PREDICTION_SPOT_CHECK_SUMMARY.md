# Prediction Spot Check Summary

Checkpoint:

- [best.pt](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_20260318/checkpoints/best.pt)

Raw report:

- [prediction_spot_checks.json](D:/workspace/supalosa-chronodivide-bot/packages/chronodivide-bot-sl/model_runs/phase9_pinch_point_arabs_manifest_20260318/prediction_spot_checks.json)

Summary:

- Build-order samples are partially learned but still confused.
- The opening deploy sample predicted `Order::DeploySelected::none` instead of `Order::Deploy::object`, while still matching the correct target entity and queue flag.
- A `Queue::Add::NAPOWR` sample was still misclassified as `Order::DeploySelected::none`, although `PlaceBuilding::NAPOWR` appeared in the top-5.
- Combat samples are weak.
- Both inspected `Order::AttackMove::tile` samples were predicted as `Queue::Add::DOG`, and the predicted target locations were clearly off the gold grid cells.
- Queue-heavy samples are weak.
- `Queue::Hold::<unk_item>` and `Queue::Cancel::NALASR` were both predicted as `SelectUnits`.
- Superweapon behavior is not learned yet.
- The inspected `ActivateSuperWeapon::ParaDrop` sample was predicted as `Order::Attack::object`, and the target location was far from gold.

What looks encouraging:

- Gold actions were not suppressed by `availableActionMask` in the inspected cases.
- Queue prediction was correct on the inspected order/combat examples where queue supervision applied.
- The build-order deploy example matched the correct target entity.

Main conclusion:

- The current model can train and overfit tiny slices, but the first held-out Arab Pinch Point slice model is still dominated by a few common action priors and is not yet robust on combat, queue-control, or superweapon actions.
