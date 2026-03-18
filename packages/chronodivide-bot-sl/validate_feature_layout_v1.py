#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run replay-level validation for the current RA2 SL feature layout V1.

This script ties together:

- `transform_replay_data.py`
- `audit_feature_tensors.py`
- `audit_training_targets.py`

It is intended to make Phase 18 validation repeatable instead of relying on
one-off shell runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORM_SCRIPT = SCRIPT_DIR / "transform_replay_data.py"
FEATURE_AUDIT_SCRIPT = SCRIPT_DIR / "audit_feature_tensors.py"
TRAINING_AUDIT_SCRIPT = SCRIPT_DIR / "audit_training_targets.py"
DEFAULT_REPLAY_DIR = SCRIPT_DIR / "ladder_replays_top50"
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "generated_tensors"
DEFAULT_DATA_DIR = Path(r"D:\workspace\ra2-headless-mix")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-replays", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--player", default="all")
    parser.add_argument("--refresh-extract-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args(argv)


def ensure_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (DEFAULT_OUTPUT_BASE / f"feature_layout_v1_validation_{timestamp_slug()}").resolve()


def run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def issue_counts_by_type(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(issue.get("type")) for issue in issues)
    return dict(sorted(counts.items()))


def summarize_observed_action_families(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {
        "queueActionsObserved": 0,
        "combatActionsObserved": 0,
        "buildActionsObserved": 0,
        "moveActionsObserved": 0,
        "selectActionsObserved": 0,
        "superWeaponActionsObserved": 0,
    }
    for entry in manifest.get("labelLayoutV1GlobalActionVocabulary", []):
        name = str(entry.get("name"))
        count = int(entry.get("count", 0))
        if count <= 0:
            continue
        if name == "SelectUnits":
            counts["selectActionsObserved"] += count
        if name.startswith("Queue::"):
            counts["queueActionsObserved"] += count
        if name.startswith("PlaceBuilding::") or name.startswith("Order::Deploy::") or name.startswith("Order::DeploySelected::"):
            counts["buildActionsObserved"] += count
        if name.startswith("Order::Move::") or name.startswith("Order::ForceMove::"):
            counts["moveActionsObserved"] += count
        if name.startswith("Order::Attack::") or name.startswith("Order::ForceAttack::") or name.startswith("Order::AttackMove::"):
            counts["combatActionsObserved"] += count
        if name.startswith("ActivateSuperWeapon::"):
            counts["superWeaponActionsObserved"] += count
    return counts


def summarize_saved_shards(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    saved_results = [result for result in manifest.get("results", []) if result.get("status") == "saved"]
    shards: list[dict[str, Any]] = []
    maps = set()
    sides = set()
    countries = set()
    total_samples = 0

    for result in saved_results:
        metadata_path = Path(str(result["metadataPath"]))
        metadata = load_json(metadata_path)
        replay = metadata.get("replay", {})
        player_name = str(metadata.get("playerName"))
        replay_players = replay.get("players", [])
        player_meta = next((player for player in replay_players if str(player.get("name")) == player_name), {})
        map_name = str(replay.get("mapName"))
        country_name = str(player_meta.get("countryName", "<unknown>"))
        side_id = player_meta.get("sideId")
        side_name = {0: "GDI", 1: "Nod", 2: "ThirdSide", 3: "Civilian", 4: "Mutant"}.get(side_id, "<unknown>")
        sample_count = int(metadata.get("sampleCount", 0))

        shards.append(
            {
                "tensor": Path(str(result["tensorPath"])).name,
                "replay": Path(str(result["replay"])).name,
                "mapName": map_name,
                "playerName": player_name,
                "countryName": country_name,
                "sideName": side_name,
                "sampleCount": sample_count,
            }
        )
        maps.add(map_name)
        sides.add(side_name)
        countries.add(country_name)
        total_samples += sample_count

    coverage = {
        "savedShardCount": len(saved_results),
        "distinctMapCount": len(maps),
        "distinctMaps": sorted(maps),
        "distinctSideCount": len(sides),
        "distinctSides": sorted(sides),
        "distinctCountryCount": len(countries),
        "distinctCountries": sorted(countries),
        "totalSamples": total_samples,
    }
    return shards, coverage


def build_validation_summary(
    *,
    manifest: dict[str, Any],
    feature_audit: dict[str, Any],
    training_audit: dict[str, Any],
) -> dict[str, Any]:
    shards, coverage = summarize_saved_shards(manifest)
    observed_action_families = summarize_observed_action_families(manifest)
    feature_issue_counts = issue_counts_by_type(feature_audit.get("issues", []))
    training_issue_counts = issue_counts_by_type(training_audit.get("issues", []))

    chosen_action_disabled_total = sum(
        int(shard.get("availableActionMask", {}).get("chosenActionDisabledCount", 0))
        for shard in feature_audit.get("shards", [])
    )

    validation_checks = {
        "transformErrorsZero": int(manifest.get("errorCount", 0)) == 0,
        "featureAuditIssuesZero": int(feature_audit.get("auditSummary", {}).get("issueCount", 0)) == 0,
        "trainingAuditIssuesZero": int(training_audit.get("auditSummary", {}).get("issueCount", 0)) == 0,
        "atLeastTwoMaps": coverage["distinctMapCount"] >= 2,
        "multipleSides": coverage["distinctSideCount"] >= 2,
        "multipleCountries": coverage["distinctCountryCount"] >= 2,
        "queueActionsObserved": observed_action_families["queueActionsObserved"] > 0,
        "combatActionsObserved": observed_action_families["combatActionsObserved"] > 0,
        "buildActionsObserved": observed_action_families["buildActionsObserved"] > 0,
        "chosenActionMaskCompatible": chosen_action_disabled_total == 0,
    }

    return {
        "createdAt": utc_now_iso(),
        "transformManifestPath": str(manifest.get("_manifestPath", "")),
        "featureAuditPath": str(feature_audit.get("_reportPath", "")),
        "trainingAuditPath": str(training_audit.get("_reportPath", "")),
        "runSummary": {
            "replayCount": int(manifest.get("replayCount", 0)),
            "savedShardCount": int(manifest.get("savedShardCount", 0)),
            "trainingTargetShardCount": int(manifest.get("trainingTargetShardCount", 0)),
            "errorCount": int(manifest.get("errorCount", 0)),
        },
        "coverage": coverage,
        "observedActionFamilies": observed_action_families,
        "validationChecks": validation_checks,
        "featureAudit": {
            "issueCount": int(feature_audit.get("auditSummary", {}).get("issueCount", 0)),
            "issueCountsByType": feature_issue_counts,
        },
        "trainingAudit": {
            "issueCount": int(training_audit.get("auditSummary", {}).get("issueCount", 0)),
            "issueCountsByType": training_issue_counts,
        },
        "savedShards": shards,
        "notes": [
            "This summary is intended for replay-level V1 validation coverage, not as a training benchmark.",
            "queue/combat/build action coverage is inferred from observed static action-dict counts in the manifest.",
            "Leak safety still requires code-path review in addition to saved-tensor audits.",
        ],
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = ensure_output_dir(args)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transform_command = [
        sys.executable,
        str(TRANSFORM_SCRIPT),
        "--replay-dir",
        str(args.replay_dir.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-dir",
        str(output_dir),
        "--player",
        str(args.player),
        "--max-replays",
        str(int(args.max_replays)),
        "--max-actions",
        str(int(args.max_actions)),
    ]
    if args.refresh_extract_cache:
        transform_command.append("--refresh-extract-cache")

    run_command(transform_command, cwd=SCRIPT_DIR)

    feature_audit_command = [
        sys.executable,
        str(FEATURE_AUDIT_SCRIPT),
        "--input",
        str(output_dir),
        "--top-k",
        str(max(1, int(args.top_k))),
    ]
    run_command(feature_audit_command, cwd=SCRIPT_DIR)

    training_audit_command = [
        sys.executable,
        str(TRAINING_AUDIT_SCRIPT),
        "--input",
        str(output_dir),
        "--top-k",
        str(max(1, int(args.top_k))),
    ]
    run_command(training_audit_command, cwd=SCRIPT_DIR)

    manifest_path = output_dir / "manifest.json"
    feature_audit_path = output_dir / "feature_tensor_audit.json"
    training_audit_path = output_dir / "training_target_audit.json"

    manifest = load_json(manifest_path)
    feature_audit = load_json(feature_audit_path)
    training_audit = load_json(training_audit_path)
    manifest["_manifestPath"] = str(manifest_path)
    feature_audit["_reportPath"] = str(feature_audit_path)
    training_audit["_reportPath"] = str(training_audit_path)

    summary = build_validation_summary(
        manifest=manifest,
        feature_audit=feature_audit,
        training_audit=training_audit,
    )

    summary_path = output_dir / "feature_layout_v1_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.strict:
        failed_checks = [name for name, passed in summary["validationChecks"].items() if not passed]
        if failed_checks:
            print(f"Validation summary written to {summary_path}", file=sys.stderr)
            print(f"Failed checks: {', '.join(failed_checks)}", file=sys.stderr)
            return 1

    print(f"Validation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
