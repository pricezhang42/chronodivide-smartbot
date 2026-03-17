#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audit replay action coverage against the static RA2 SL action dict.

This is a lightweight companion to `transform_replay_data.py`: it scans replay
actions through a label-only extractor and reports whether the static SL action
dictionary covers the observed action-type surface cleanly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from action_dict import (
    STATIC_ACTION_DICT_VERSION,
    UNKNOWN_ACTION_TYPE_NAME,
    build_observed_action_type_name,
    canonicalize_action_type_name,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_REPLAY_DIR = PACKAGE_ROOT / "ladder_replays_top50"
DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "action_dict_audit.json"
DEFAULT_JS_SCRIPT = PROJECT_ROOT / "packages" / "py-chronodivide" / "extract_action_labels.mjs"
DEFAULT_DATA_DIR = Path("d:/workspace/ra2-headless-mix")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditConfig:
    replay_dir: Path
    output_path: Path
    data_dir: Path
    py_chronodivide_script: Path
    replay_glob: str
    replay_start: int
    max_replays: int | None
    player: str
    include_no_action: bool
    include_ui_actions: bool
    max_actions: int | None
    max_tick: int | None
    overwrite: bool
    fail_fast: bool


def parse_args(argv: list[str]) -> AuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--py-chronodivide-script", type=Path, default=DEFAULT_JS_SCRIPT)
    parser.add_argument("--replay-glob", default="*.rpl")
    parser.add_argument("--replay-start", type=int, default=0)
    parser.add_argument("--max-replays", type=int, default=None)
    parser.add_argument("--player", default="all")
    parser.add_argument("--include-no-action", action="store_true")
    parser.add_argument("--include-ui-actions", action="store_true")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-tick", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    return AuditConfig(
        replay_dir=args.replay_dir.resolve(),
        output_path=args.output.resolve(),
        data_dir=args.data_dir.resolve(),
        py_chronodivide_script=args.py_chronodivide_script.resolve(),
        replay_glob=args.replay_glob,
        replay_start=max(0, args.replay_start),
        max_replays=args.max_replays,
        player=str(args.player),
        include_no_action=bool(args.include_no_action),
        include_ui_actions=bool(args.include_ui_actions),
        max_actions=args.max_actions,
        max_tick=args.max_tick,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
    )


def validate_config(config: AuditConfig) -> None:
    if not config.replay_dir.exists():
        raise FileNotFoundError(f"Replay directory does not exist: {config.replay_dir}")
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Chronodivide data directory does not exist: {config.data_dir}")
    if not config.py_chronodivide_script.exists():
        raise FileNotFoundError(f"Action-label extractor does not exist: {config.py_chronodivide_script}")


def list_replays(config: AuditConfig) -> list[Path]:
    replay_paths = sorted(config.replay_dir.glob(config.replay_glob))
    replay_paths = replay_paths[config.replay_start :]
    if config.max_replays is not None:
        replay_paths = replay_paths[: config.max_replays]
    return replay_paths


def iter_progress(items: list[Path], description: str) -> Any:
    if tqdm is None:
        return items
    return tqdm(items, desc=description)


def build_node_command(config: AuditConfig, replay_path: Path, output_path: Path) -> list[str]:
    command = [
        "node",
        str(config.py_chronodivide_script),
        "--replay",
        str(replay_path),
        "--data-dir",
        str(config.data_dir),
        "--player",
        str(config.player),
        "--output",
        str(output_path),
    ]
    if config.include_no_action:
        command.extend(["--include-no-action", "true"])
    if config.include_ui_actions:
        command.extend(["--include-ui-actions", "true"])
    if config.max_actions is not None:
        command.extend(["--max-actions", str(config.max_actions)])
    if config.max_tick is not None:
        command.extend(["--max-tick", str(config.max_tick)])
    return command


def run_action_label_extract(config: AuditConfig, replay_path: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="chronodivide_action_labels_", suffix=".json", delete=False) as handle:
        temp_output_path = Path(handle.name)
    command = build_node_command(config, replay_path, temp_output_path)
    try:
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(PACKAGE_ROOT),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            stdout_text = (completed.stdout or "").strip()
            details = stderr_text or stdout_text or f"Action-label extractor exited with status {completed.returncode}."
            raise RuntimeError(details)
        with temp_output_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        temp_output_path.unlink(missing_ok=True)


def classify_fallback_reason(observed_name: str, canonical_name: str) -> str:
    if canonical_name == UNKNOWN_ACTION_TYPE_NAME:
        return "unknown_action_type"
    if observed_name.startswith("Queue::") and canonical_name.startswith("Queue::"):
        return "unknown_queue_item"
    if observed_name.startswith("PlaceBuilding::") and canonical_name.startswith("PlaceBuilding::"):
        return "unknown_building"
    if observed_name.startswith("ActivateSuperWeapon::") and canonical_name.startswith("ActivateSuperWeapon::"):
        return "unknown_super_weapon"
    return "canonicalization_fallback"


def top_counts(counter: Counter[str], *, limit: int = 50, key_name: str = "name") -> list[dict[str, Any]]:
    return [
        {key_name: name, "count": count}
        for name, count in counter.most_common(limit)
    ]


def audit_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    observed_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()

    for action in payload.get("actions", []):
        observed_name = build_observed_action_type_name(
            raw_action_name=action.get("rawActionName"),
            raw_action_id=action.get("rawActionId"),
            order_type_name=action.get("orderTypeName"),
            target_mode_name=action.get("targetMode"),
            queue_update_type_name=action.get("queueUpdateTypeName"),
            item_name=action.get("itemName"),
            building_name=action.get("buildingName"),
            super_weapon_name=action.get("superWeaponTypeName"),
        )
        canonical_name = canonicalize_action_type_name(observed_name)
        observed_counts[observed_name] += 1
        canonical_counts[canonical_name] += 1
        if canonical_name != observed_name:
            fallback_counts[observed_name] += 1
            fallback_reason_counts[classify_fallback_reason(observed_name, canonical_name)] += 1

    total_action_count = sum(observed_counts.values())
    fallback_action_count = sum(fallback_counts.values())
    exact_action_count = total_action_count - fallback_action_count

    return {
        "replay": payload.get("replay", {}),
        "sampledPlayers": payload.get("sampledPlayers", []),
        "totalActionCount": total_action_count,
        "exactActionCount": exact_action_count,
        "fallbackActionCount": fallback_action_count,
        "fallbackRate": 0.0 if total_action_count == 0 else fallback_action_count / float(total_action_count),
        "observedActionTypes": top_counts(observed_counts),
        "canonicalActionTypes": top_counts(canonical_counts),
        "fallbackObservedActionTypes": top_counts(fallback_counts),
        "fallbackReasonCounts": top_counts(fallback_reason_counts, key_name="reason"),
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    observed_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    total_action_count = 0
    fallback_action_count = 0

    for report in reports:
        total_action_count += int(report["totalActionCount"])
        fallback_action_count += int(report["fallbackActionCount"])
        for entry in report["observedActionTypes"]:
            observed_counts[str(entry["name"])] += int(entry["count"])
        for entry in report["canonicalActionTypes"]:
            canonical_counts[str(entry["name"])] += int(entry["count"])
        for entry in report["fallbackObservedActionTypes"]:
            fallback_counts[str(entry["name"])] += int(entry["count"])
        for entry in report["fallbackReasonCounts"]:
            fallback_reason_counts[str(entry["reason"])] += int(entry["count"])

    return {
        "totalActionCount": total_action_count,
        "fallbackActionCount": fallback_action_count,
        "exactActionCount": total_action_count - fallback_action_count,
        "fallbackRate": 0.0 if total_action_count == 0 else fallback_action_count / float(total_action_count),
        "observedActionTypes": top_counts(observed_counts, limit=100),
        "canonicalActionTypes": top_counts(canonical_counts, limit=100),
        "fallbackObservedActionTypes": top_counts(fallback_counts, limit=100),
        "fallbackReasonCounts": top_counts(fallback_reason_counts, limit=20, key_name="reason"),
    }


def write_report(config: AuditConfig, replay_paths: list[Path], reports: list[dict[str, Any]], errors: list[dict[str, Any]]) -> Path:
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output already exists: {config.output_path}")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "createdAt": utc_now_iso(),
        "config": {
            **asdict(config),
            "replay_dir": str(config.replay_dir),
            "output_path": str(config.output_path),
            "data_dir": str(config.data_dir),
            "py_chronodivide_script": str(config.py_chronodivide_script),
        },
        "staticActionDictVersion": STATIC_ACTION_DICT_VERSION,
        "replayCount": len(replay_paths),
        "processedReplayCount": len(reports),
        "errorCount": len(errors),
        "summary": aggregate_reports(reports),
        "reports": reports,
        "errors": errors,
    }
    with config.output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return config.output_path


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    validate_config(config)
    replay_paths = list_replays(config)
    if not replay_paths:
        raise FileNotFoundError(f"No replay files matching {config.replay_glob!r} were found in {config.replay_dir}.")

    reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for replay_path in iter_progress(replay_paths, "Audit action dict"):
        try:
            payload = run_action_label_extract(config, replay_path)
            reports.append(audit_replay_payload(payload))
        except Exception as exc:  # pragma: no cover - real-run failure path
            error_record = {
                "replay": str(replay_path),
                "errorType": exc.__class__.__name__,
                "error": str(exc),
            }
            errors.append(error_record)
            if config.fail_fast:
                raise

    report_path = write_report(config, replay_paths, reports, errors)
    print(f"Audited {len(replay_paths)} replay(s).")
    print(f"Report: {report_path}")
    if errors:
        print(f"Encountered {len(errors)} replay error(s). See report for details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
