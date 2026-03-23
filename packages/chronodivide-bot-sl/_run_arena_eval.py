"""Run arena evaluation: model vs supalosa bot and model vs dummy."""
import json
import sys
from pathlib import Path

from post_train_arena_eval import run_post_train_arena_eval

DRIVER_DIR = Path("../chronodivide-bot-driver").resolve()
OUTPUT_DIR = Path("model_runs").resolve()
CHECKPOINT_DIR = Path("model_runs/checkpoints").resolve()
MIX_DIR = Path("d:/workspace/ra2-headless-mix").resolve()
PREFERRED_CHECKPOINTS = ["best_val_free_action.pt", "best_val_loss.pt", "best.pt", "latest.pt"]


def run_eval(label, candidate_mode, opponent_mode, output_subdir):
    eval_output = OUTPUT_DIR / output_subdir
    eval_output.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Candidate: {candidate_mode} (IRAQ)")
    print(f"  Opponent:  {opponent_mode} (IRAQ)")
    print(f"{'='*60}\n")

    result = run_post_train_arena_eval(
        enabled=True,
        driver_dir=DRIVER_DIR,
        output_dir=eval_output,
        checkpoint_dir=CHECKPOINT_DIR,
        preferred_checkpoint_names=PREFERRED_CHECKPOINTS,
        match_count=3,
        map_name="2_pinch_point_le.map",
        max_ticks=18000,
        sample_interval_ticks=15,
        candidate_mode=candidate_mode,
        candidate_country="IRAQ",
        opponent_mode=opponent_mode,
        opponent_country="IRAQ",
        mix_dir=MIX_DIR,
    )

    # Print summary
    summary_path = eval_output / "arena_eval" / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        agg = summary.get("aggregate", {})
        print(f"\n--- {label} Results ---")
        print(f"Wins: {agg.get('wins', '?')}, Losses: {agg.get('losses', '?')}, Draws: {agg.get('draws', '?')}")
        print(f"Win Rate: {agg.get('winRate', '?')}")
        print(f"Avg Duration: {agg.get('averageDurationSeconds', '?'):.1f}s")

        for role in ["candidate", "opponent"]:
            p = agg.get(role, {})
            print(f"\n  {role.upper()}:")
            print(f"    Used Money: {p.get('averageUsedMoney', '?'):.0f}")
            print(f"    Military Peak: {p.get('averageMilitaryValuePeak', '?'):.0f}")
            print(f"    Combat Units Final: {p.get('averageCombatUnitsFinal', '?'):.1f}")
            print(f"    Harvesters Final: {p.get('averageHarvestersFinal', '?'):.1f}")
    else:
        print(f"No summary found at {summary_path}")

    return result


if __name__ == "__main__":
    # 1. Model (full control) vs Supalosa bot (baseline)
    run_eval(
        "Model (control) vs Supalosa Bot (baseline)",
        candidate_mode="control",
        opponent_mode="baseline",
        output_subdir="eval_control_vs_baseline",
    )

    # 2. Model (full control) vs Supalosa bot (advisor)
    run_eval(
        "Model (control) vs Supalosa Bot (advisor)",
        candidate_mode="control",
        opponent_mode="advisor",
        output_subdir="eval_control_vs_advisor",
    )

    print("\n\nAll evaluations complete!")
