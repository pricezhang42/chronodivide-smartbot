#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audit derived training targets for label distribution and mask sanity.

This companion script loads the replay-player `.training.pt` sidecars generated
by `transform_replay_data.py` and checks that the model-ready targets and loss
masks remain self-consistent. It also summarizes the observed label surface for
the current tensor run so training code can sanity-check class balance and
target coverage before starting a larger experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Tensor output directory or manifest.json path produced by transform_replay_data.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the JSON audit report. Defaults beside the manifest.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many top-count entries to keep for action/distribution summaries.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any invariant violations are found.",
    )
    return parser.parse_args(argv)


def resolve_manifest_path(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.is_dir():
        manifest_path = input_path / "manifest.json"
    else:
        manifest_path = input_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    return manifest_path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def top_counts(counter: Counter[Any], *, key_name: str, limit: int) -> list[dict[str, Any]]:
    return [
        {key_name: key, "count": count}
        for key, count in counter.most_common(limit)
    ]


def count_invalid_binary_values(tensor: torch.Tensor) -> int:
    return int(torch.sum((tensor != 0) & (tensor != 1)).item())


def as_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.reshape(-1).tolist()]


def summarize_numeric_counter(counter: Counter[int], *, key_name: str, limit: int) -> list[dict[str, Any]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        {key_name: key, "count": count}
        for key, count in items[:limit]
    ]


def ensure_shape(
    *,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    issues: list[dict[str, Any]],
    shard_name: str,
    section_name: str,
    group_name: str,
) -> None:
    observed_shape = tuple(int(value) for value in tensor.shape)
    if observed_shape != expected_shape:
        issues.append(
            {
                "type": "shape_mismatch",
                "shard": shard_name,
                "group": group_name,
                "section": section_name,
                "expectedShape": list(expected_shape),
                "observedShape": list(observed_shape),
            }
        )


def append_binary_issue(
    *,
    tensor: torch.Tensor,
    issues: list[dict[str, Any]],
    shard_name: str,
    section_name: str,
    group_name: str,
) -> None:
    invalid_count = count_invalid_binary_values(tensor)
    if invalid_count:
        issues.append(
            {
                "type": "non_binary_values",
                "shard": shard_name,
                "group": group_name,
                "section": section_name,
                "invalidCount": invalid_count,
            }
        )


def compare_tensors_equal(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    issues: list[dict[str, Any]],
    shard_name: str,
    issue_type: str,
    details: dict[str, Any],
) -> None:
    if actual.shape != expected.shape:
        issues.append(
            {
                "type": issue_type,
                "shard": shard_name,
                "message": "shape mismatch during tensor comparison",
                "actualShape": list(actual.shape),
                "expectedShape": list(expected.shape),
                **details,
            }
        )
        return
    mismatch_count = int(torch.sum(actual != expected).item())
    if mismatch_count:
        issues.append(
            {
                "type": issue_type,
                "shard": shard_name,
                "mismatchCount": mismatch_count,
                **details,
            }
        )


def audit_shard(
    *,
    result: dict[str, Any],
    global_action_vocabulary: dict[int, dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Counter[Any]]]:
    shard_name = Path(str(result["tensorPath"])).name
    metadata_path = Path(str(result["metadataPath"]))
    training_target_path = Path(str(result["trainingTargetTensorPath"]))

    metadata = load_json(metadata_path)
    training_payload = torch.load(training_target_path, map_location="cpu", weights_only=True)
    training_targets = training_payload["trainingTargets"]
    training_masks = training_payload["trainingMasks"]
    training_schema = metadata["trainingTargetsV1"]
    label_layout_v1 = metadata["labelLayoutV1"]

    sample_count = int(metadata["sampleCount"])
    issues: list[dict[str, Any]] = []

    for section in training_schema["targetSections"]:
        section_name = str(section["name"])
        expected_shape = (sample_count, *tuple(int(value) for value in section["shape"]))
        ensure_shape(
            tensor=training_targets[section_name],
            expected_shape=expected_shape,
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="trainingTargets",
        )
        if section_name.endswith("OneHot"):
            append_binary_issue(
                tensor=training_targets[section_name],
                issues=issues,
                shard_name=shard_name,
                section_name=section_name,
                group_name="trainingTargets",
            )

    for section in training_schema["maskSections"]:
        section_name = str(section["name"])
        expected_shape = (sample_count, *tuple(int(value) for value in section["shape"]))
        ensure_shape(
            tensor=training_masks[section_name],
            expected_shape=expected_shape,
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="trainingMasks",
        )
        append_binary_issue(
            tensor=training_masks[section_name],
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="trainingMasks",
        )

    action_type_loss_mask = training_masks["actionTypeLossMask"].to(torch.int32)
    delay_loss_mask = training_masks["delayLossMask"].to(torch.int32)
    queue_semantic_mask = training_masks["queueSemanticMask"].to(torch.int32)
    units_semantic_mask = training_masks["unitsSemanticMask"].to(torch.int32)
    target_entity_semantic_mask = training_masks["targetEntitySemanticMask"].to(torch.int32)
    target_location_semantic_mask = training_masks["targetLocationSemanticMask"].to(torch.int32)
    target_location_2_semantic_mask = training_masks["targetLocation2SemanticMask"].to(torch.int32)
    quantity_semantic_mask = training_masks["quantitySemanticMask"].to(torch.int32)
    queue_loss_mask = training_masks["queueLossMask"].to(torch.int32)
    units_sequence_mask = training_masks["unitsSequenceMask"].to(torch.int32)
    units_resolved_mask = training_masks["unitsResolvedMask"].to(torch.int32)
    units_loss_mask = training_masks["unitsLossMask"].to(torch.int32)
    target_entity_resolved_mask = training_masks["targetEntityResolvedMask"].to(torch.int32)
    target_entity_loss_mask = training_masks["targetEntityLossMask"].to(torch.int32)
    target_location_valid_mask = training_masks["targetLocationValidMask"].to(torch.int32)
    target_location_loss_mask = training_masks["targetLocationLossMask"].to(torch.int32)
    target_location_2_valid_mask = training_masks["targetLocation2ValidMask"].to(torch.int32)
    target_location_2_loss_mask = training_masks["targetLocation2LossMask"].to(torch.int32)
    quantity_loss_mask = training_masks["quantityLossMask"].to(torch.int32)

    action_type_row_sums = training_targets["actionTypeOneHot"].sum(dim=1, keepdim=True).to(torch.int32)
    delay_row_sums = training_targets["delayOneHot"].sum(dim=1, keepdim=True).to(torch.int32)
    queue_row_sums = training_targets["queueOneHot"].sum(dim=1, keepdim=True).to(torch.int32)
    units_row_sums = training_targets["unitsOneHot"].sum(dim=2).to(torch.int32)
    target_entity_row_sums = training_targets["targetEntityOneHot"].sum(dim=1, keepdim=True).to(torch.int32)
    target_location_row_sums = training_targets["targetLocationOneHot"].sum(dim=(1, 2)).to(torch.int32).unsqueeze(1)
    target_location_2_row_sums = (
        training_targets["targetLocation2OneHot"].sum(dim=(1, 2)).to(torch.int32).unsqueeze(1)
    )

    compare_tensors_equal(
        actual=action_type_row_sums,
        expected=action_type_loss_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="action_type_one_hot_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=delay_row_sums,
        expected=delay_loss_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="delay_one_hot_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=queue_row_sums,
        expected=queue_loss_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="queue_one_hot_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=units_row_sums,
        expected=units_loss_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="units_one_hot_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_entity_row_sums,
        expected=target_entity_resolved_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="target_entity_one_hot_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_location_row_sums,
        expected=target_location_valid_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="target_location_one_hot_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_location_2_row_sums,
        expected=target_location_2_valid_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="target_location2_one_hot_mismatch",
        details={},
    )

    compare_tensors_equal(
        actual=queue_loss_mask,
        expected=queue_semantic_mask * queue_row_sums,
        issues=issues,
        shard_name=shard_name,
        issue_type="queue_loss_mask_formula_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=units_loss_mask,
        expected=units_semantic_mask * units_sequence_mask * units_resolved_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="units_loss_mask_formula_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_entity_loss_mask,
        expected=target_entity_semantic_mask * target_entity_resolved_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="target_entity_loss_mask_formula_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_location_loss_mask,
        expected=target_location_semantic_mask * target_location_valid_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="target_location_loss_mask_formula_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_location_2_loss_mask,
        expected=target_location_2_semantic_mask * target_location_2_valid_mask,
        issues=issues,
        shard_name=shard_name,
        issue_type="target_location2_loss_mask_formula_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=quantity_loss_mask,
        expected=quantity_semantic_mask * (training_targets["quantityValue"] >= 0).to(torch.int32),
        issues=issues,
        shard_name=shard_name,
        issue_type="quantity_loss_mask_formula_mismatch",
        details={},
    )

    semantic_lookup = {
        int(entry["id"]): {
            "name": str(entry["name"]),
            "usesQueue": bool(entry["semanticMask"]["usesQueue"]),
            "usesUnits": bool(entry["semanticMask"]["usesUnits"]),
            "usesTargetEntity": bool(entry["semanticMask"]["usesTargetEntity"]),
            "usesTargetLocation": bool(entry["semanticMask"]["usesTargetLocation"]),
            "usesTargetLocation2": bool(entry["semanticMask"]["usesTargetLocation2"]),
            "usesQuantity": bool(entry["semanticMask"]["usesQuantity"]),
        }
        for entry in label_layout_v1["actionTypeVocabulary"]
    }

    valid_action_rows = action_type_loss_mask.squeeze(1) > 0
    action_type_ids = torch.argmax(training_targets["actionTypeOneHot"], dim=1)
    semantic_expected = {
        "usesQueue": [],
        "usesUnits": [],
        "usesTargetEntity": [],
        "usesTargetLocation": [],
        "usesTargetLocation2": [],
        "usesQuantity": [],
    }
    for row_index in range(sample_count):
        if not bool(valid_action_rows[row_index]):
            for values in semantic_expected.values():
                values.append(0)
            continue
        action_type_id = int(action_type_ids[row_index].item())
        lookup_entry = semantic_lookup.get(action_type_id, {})
        for semantic_key in semantic_expected:
            semantic_expected[semantic_key].append(1 if lookup_entry.get(semantic_key, False) else 0)

    compare_tensors_equal(
        actual=queue_semantic_mask,
        expected=torch.tensor(semantic_expected["usesQueue"], dtype=torch.int32).unsqueeze(1),
        issues=issues,
        shard_name=shard_name,
        issue_type="queue_semantic_mask_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=units_semantic_mask,
        expected=torch.tensor(semantic_expected["usesUnits"], dtype=torch.int32).unsqueeze(1),
        issues=issues,
        shard_name=shard_name,
        issue_type="units_semantic_mask_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_entity_semantic_mask,
        expected=torch.tensor(semantic_expected["usesTargetEntity"], dtype=torch.int32).unsqueeze(1),
        issues=issues,
        shard_name=shard_name,
        issue_type="target_entity_semantic_mask_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_location_semantic_mask,
        expected=torch.tensor(semantic_expected["usesTargetLocation"], dtype=torch.int32).unsqueeze(1),
        issues=issues,
        shard_name=shard_name,
        issue_type="target_location_semantic_mask_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=target_location_2_semantic_mask,
        expected=torch.tensor(semantic_expected["usesTargetLocation2"], dtype=torch.int32).unsqueeze(1),
        issues=issues,
        shard_name=shard_name,
        issue_type="target_location2_semantic_mask_mismatch",
        details={},
    )
    compare_tensors_equal(
        actual=quantity_semantic_mask,
        expected=torch.tensor(semantic_expected["usesQuantity"], dtype=torch.int32).unsqueeze(1),
        issues=issues,
        shard_name=shard_name,
        issue_type="quantity_semantic_mask_mismatch",
        details={},
    )

    action_counter: Counter[str] = Counter()
    action_family_counter: Counter[str] = Counter()
    delay_counter: Counter[int] = Counter()
    queue_counter: Counter[int] = Counter()
    quantity_counter: Counter[int] = Counter()
    selected_units_counter: Counter[int] = Counter()
    selected_units_loss_counter: Counter[int] = Counter()
    target_entity_by_action: Counter[str] = Counter()
    target_location_by_action: Counter[str] = Counter()
    target_location_2_by_action: Counter[str] = Counter()
    quantity_by_action: Counter[str] = Counter()
    action_head_counts: dict[str, Counter[str]] = defaultdict(Counter)

    delay_ids = torch.argmax(training_targets["delayOneHot"], dim=1)
    queue_ids = torch.argmax(training_targets["queueOneHot"], dim=1)

    for row_index in range(sample_count):
        if not bool(valid_action_rows[row_index]):
            continue
        action_type_id = int(action_type_ids[row_index].item())
        action_entry = global_action_vocabulary.get(
            action_type_id,
            {"name": f"<missing_{action_type_id}>", "family": "unknown"},
        )
        action_name = str(action_entry["name"])
        action_counter[action_name] += 1
        action_family_counter[str(action_entry.get("family", "unknown"))] += 1

        if bool(delay_loss_mask[row_index, 0]):
            delay_counter[int(delay_ids[row_index].item())] += 1
        if bool(queue_loss_mask[row_index, 0]):
            queue_counter[int(queue_ids[row_index].item())] += 1
        if bool(quantity_loss_mask[row_index, 0]):
            quantity_counter[int(training_targets["quantityValue"][row_index, 0].item())] += 1

        selected_count = int(units_sequence_mask[row_index].sum().item())
        selected_loss_count = int(units_loss_mask[row_index].sum().item())
        if bool(units_semantic_mask[row_index, 0]):
            selected_units_counter[selected_count] += 1
            selected_units_loss_counter[selected_loss_count] += 1

        if bool(target_entity_loss_mask[row_index, 0]):
            target_entity_by_action[action_name] += 1
        if bool(target_location_loss_mask[row_index, 0]):
            target_location_by_action[action_name] += 1
        if bool(target_location_2_loss_mask[row_index, 0]):
            target_location_2_by_action[action_name] += 1
        if bool(quantity_loss_mask[row_index, 0]):
            quantity_by_action[action_name] += 1

        action_head_counts[action_name]["sampleCount"] += 1
        action_head_counts[action_name]["queueSemanticCount"] += int(queue_semantic_mask[row_index, 0].item())
        action_head_counts[action_name]["queueLossCount"] += int(queue_loss_mask[row_index, 0].item())
        action_head_counts[action_name]["unitsSemanticCount"] += int(units_semantic_mask[row_index, 0].item())
        action_head_counts[action_name]["unitsLossPositions"] += selected_loss_count
        action_head_counts[action_name]["targetEntitySemanticCount"] += int(
            target_entity_semantic_mask[row_index, 0].item()
        )
        action_head_counts[action_name]["targetEntityLossCount"] += int(target_entity_loss_mask[row_index, 0].item())
        action_head_counts[action_name]["targetLocationSemanticCount"] += int(
            target_location_semantic_mask[row_index, 0].item()
        )
        action_head_counts[action_name]["targetLocationLossCount"] += int(target_location_loss_mask[row_index, 0].item())
        action_head_counts[action_name]["targetLocation2SemanticCount"] += int(
            target_location_2_semantic_mask[row_index, 0].item()
        )
        action_head_counts[action_name]["targetLocation2LossCount"] += int(
            target_location_2_loss_mask[row_index, 0].item()
        )
        action_head_counts[action_name]["quantitySemanticCount"] += int(quantity_semantic_mask[row_index, 0].item())
        action_head_counts[action_name]["quantityLossCount"] += int(quantity_loss_mask[row_index, 0].item())

    semantic_mask_summary = {
        "queueSemanticCount": int(queue_semantic_mask.sum().item()),
        "unitsSemanticCount": int(units_semantic_mask.sum().item()),
        "targetEntitySemanticCount": int(target_entity_semantic_mask.sum().item()),
        "targetLocationSemanticCount": int(target_location_semantic_mask.sum().item()),
        "targetLocation2SemanticCount": int(target_location_2_semantic_mask.sum().item()),
        "quantitySemanticCount": int(quantity_semantic_mask.sum().item()),
    }
    loss_mask_summary = {
        "actionTypeLossCount": int(action_type_loss_mask.sum().item()),
        "delayLossCount": int(delay_loss_mask.sum().item()),
        "queueLossCount": int(queue_loss_mask.sum().item()),
        "unitsLossPositionCount": int(units_loss_mask.sum().item()),
        "targetEntityLossCount": int(target_entity_loss_mask.sum().item()),
        "targetLocationLossCount": int(target_location_loss_mask.sum().item()),
        "targetLocation2LossCount": int(target_location_2_loss_mask.sum().item()),
        "quantityLossCount": int(quantity_loss_mask.sum().item()),
    }

    per_action_summary = []
    for action_name, counts in sorted(
        action_head_counts.items(),
        key=lambda item: (-item[1]["sampleCount"], item[0]),
    )[:top_k]:
        per_action_summary.append(
            {
                "actionType": action_name,
                **{key: int(value) for key, value in counts.items()},
            }
        )

    report = {
        "shard": shard_name,
        "replay": metadata["replay"],
        "playerName": metadata["playerName"],
        "sampleCount": sample_count,
        "issueCount": len(issues),
        "issues": issues,
        "semanticMaskSummary": semantic_mask_summary,
        "lossMaskSummary": loss_mask_summary,
        "actionTypeCounts": top_counts(action_counter, key_name="actionType", limit=top_k),
        "actionFamilyCounts": top_counts(action_family_counter, key_name="actionFamily", limit=top_k),
        "delayBinCounts": summarize_numeric_counter(delay_counter, key_name="delayBin", limit=top_k),
        "queueValueCounts": summarize_numeric_counter(queue_counter, key_name="queueValue", limit=top_k),
        "quantityValueCounts": summarize_numeric_counter(quantity_counter, key_name="quantity", limit=top_k),
        "selectedUnitsCount": summarize_numeric_counter(selected_units_counter, key_name="selectedUnits", limit=top_k),
        "selectedUnitsLossPositions": summarize_numeric_counter(
            selected_units_loss_counter,
            key_name="selectedUnitsWithLoss",
            limit=top_k,
        ),
        "targetEntityByAction": top_counts(target_entity_by_action, key_name="actionType", limit=top_k),
        "targetLocationByAction": top_counts(target_location_by_action, key_name="actionType", limit=top_k),
        "targetLocation2ByAction": top_counts(target_location_2_by_action, key_name="actionType", limit=top_k),
        "quantityByAction": top_counts(quantity_by_action, key_name="actionType", limit=top_k),
        "perActionHeadSummary": per_action_summary,
    }
    counters = {
        "action": action_counter,
        "family": action_family_counter,
        "delay": delay_counter,
        "queue": queue_counter,
        "quantity": quantity_counter,
    }
    return report, counters


def audit_manifest(manifest: dict[str, Any], *, manifest_path: Path, top_k: int) -> dict[str, Any]:
    global_action_vocabulary = {
        int(entry["id"]): entry
        for entry in manifest.get("labelLayoutV1GlobalActionVocabulary", [])
    }
    shard_reports = []
    total_issues = 0
    total_samples = 0
    aggregate_action_counter: Counter[str] = Counter()
    aggregate_family_counter: Counter[str] = Counter()
    aggregate_delay_counter: Counter[int] = Counter()
    aggregate_queue_counter: Counter[int] = Counter()
    aggregate_quantity_counter: Counter[int] = Counter()

    for result in manifest.get("results", []):
        if not result.get("trainingTargetTensorPath") or not result.get("metadataPath"):
            continue
        report, counters = audit_shard(result=result, global_action_vocabulary=global_action_vocabulary, top_k=top_k)
        shard_reports.append(report)
        total_issues += int(report["issueCount"])
        total_samples += int(report["sampleCount"])
        aggregate_action_counter.update(counters["action"])
        aggregate_family_counter.update(counters["family"])
        aggregate_delay_counter.update(counters["delay"])
        aggregate_queue_counter.update(counters["queue"])
        aggregate_quantity_counter.update(counters["quantity"])

    return {
        "createdAt": utc_now_iso(),
        "manifestPath": str(manifest_path),
        "inputConfig": manifest.get("config", {}),
        "staticActionDictVersion": manifest.get("staticActionDictVersion"),
        "replayCount": int(manifest.get("replayCount", 0)),
        "savedShardCount": int(manifest.get("savedShardCount", 0)),
        "trainingTargetShardCount": int(manifest.get("trainingTargetShardCount", 0)),
        "auditedShardCount": len(shard_reports),
        "totalSampleCount": total_samples,
        "issueCount": total_issues,
        "aggregateActionTypeCounts": top_counts(aggregate_action_counter, key_name="actionType", limit=top_k),
        "aggregateActionFamilyCounts": top_counts(aggregate_family_counter, key_name="actionFamily", limit=top_k),
        "aggregateDelayBinCounts": summarize_numeric_counter(aggregate_delay_counter, key_name="delayBin", limit=top_k),
        "aggregateQueueValueCounts": summarize_numeric_counter(
            aggregate_queue_counter,
            key_name="queueValue",
            limit=top_k,
        ),
        "aggregateQuantityValueCounts": summarize_numeric_counter(
            aggregate_quantity_counter,
            key_name="quantity",
            limit=top_k,
        ),
        "shards": shard_reports,
    }


def default_output_path(manifest_path: Path) -> Path:
    return manifest_path.with_name("training_target_audit.json")


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest_path = resolve_manifest_path(args.input)
    manifest = load_json(manifest_path)
    report = audit_manifest(manifest, manifest_path=manifest_path, top_k=max(1, int(args.top_k)))
    output_path = args.output.resolve() if args.output else default_output_path(manifest_path)
    write_report(report, output_path)
    print(f"Audited {report['auditedShardCount']} training-target shard(s).")
    print(f"Samples: {report['totalSampleCount']}")
    print(f"Issues: {report['issueCount']}")
    print(f"Report: {output_path}")
    if args.strict and report["issueCount"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
