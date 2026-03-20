from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DRIVER_DIR = PACKAGE_ROOT.parent / "chronodivide-bot-driver"


def resolve_checkpoint_path(checkpoint_dir: Path, preferred_names: list[str]) -> Path:
    for name in preferred_names:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any preferred checkpoint in "
        f"{checkpoint_dir}. Tried: {', '.join(preferred_names)}"
    )


def run_post_train_arena_eval(
    *,
    enabled: bool,
    driver_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    preferred_checkpoint_names: list[str],
    match_count: int,
    map_name: str,
    max_ticks: int,
    sample_interval_ticks: int,
    candidate_mode: str,
    candidate_country: str,
    opponent_mode: str,
    opponent_country: str,
    mix_dir: Path | None,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, preferred_checkpoint_names)
    driver_dir = driver_dir.resolve()
    dist_script = driver_dir / "dist" / "evaluateAgainstSupalosaBot.js"
    if not dist_script.exists():
        raise FileNotFoundError(
            f"Expected arena evaluator at {dist_script}, but it does not exist. "
            "Build chronodivide-bot-driver first."
        )

    eval_output_dir = output_dir / "arena_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "node",
        str(dist_script),
        "--output-dir",
        str(eval_output_dir),
        "--replay-dir",
        str(eval_output_dir / "replays"),
        "--map-name",
        map_name,
        "--matches",
        str(match_count),
        "--max-ticks",
        str(max_ticks),
        "--sample-interval-ticks",
        str(sample_interval_ticks),
        "--checkpoint-path",
        str(checkpoint_path),
        "--candidate-mode",
        candidate_mode,
        "--candidate-country",
        candidate_country,
        "--opponent-mode",
        opponent_mode,
        "--opponent-country",
        opponent_country,
    ]
    if mix_dir is not None:
        command.extend(["--mix-dir", str(mix_dir.resolve())])

    env = os.environ.copy()
    env["SL_CHECKPOINT_PATH"] = str(checkpoint_path)
    env["SL_TRAIN_RUN_DIR"] = str(output_dir)
    env.setdefault("SL_PYTHON_EXECUTABLE", sys.executable)

    subprocess.run(
        command,
        check=True,
        cwd=driver_dir,
        env=env,
    )

    summary_path = eval_output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Arena evaluation finished, but no summary was written to {summary_path}.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "summaryPath": str(summary_path),
        "checkpointPath": str(checkpoint_path),
        "aggregate": summary.get("aggregate"),
        "candidate": summary.get("candidate"),
        "opponent": summary.get("opponent"),
    }
