#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audit structured feature tensors for schema, ranges, sparsity, and invariants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from transform_lib.common import LABEL_LAYOUT_V1_MISSING_INT, schema_dtype_to_torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def resolve_manifest_path(input_path: Path) -> Path:
    input_path = input_path.resolve()
    manifest_path = input_path / "manifest.json" if input_path.is_dir() else input_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    return manifest_path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def top_counter_entries(counter: Counter[Any], *, key_name: str, limit: int) -> list[dict[str, Any]]:
    return [{key_name: key, "count": count} for key, count in counter.most_common(limit)]


def count_non_binary_values(tensor: torch.Tensor) -> int:
    return int(torch.sum((tensor != 0) & (tensor != 1)).item())


def issue(
    issues: list[dict[str, Any]],
    *,
    issue_type: str,
    shard: str,
    **payload: Any,
) -> None:
    issues.append({"type": issue_type, "shard": shard, **payload})


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
        issue(
            issues,
            issue_type="shape_mismatch",
            shard=shard_name,
            group=group_name,
            section=section_name,
            expectedShape=list(expected_shape),
            observedShape=list(observed_shape),
        )


def ensure_dtype(
    *,
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    issues: list[dict[str, Any]],
    shard_name: str,
    section_name: str,
    group_name: str,
) -> None:
    if tensor.dtype != expected_dtype:
        issue(
            issues,
            issue_type="dtype_mismatch",
            shard=shard_name,
            group=group_name,
            section=section_name,
            expectedDType=str(expected_dtype),
            observedDType=str(tensor.dtype),
        )


def append_non_finite_issue(
    *,
    tensor: torch.Tensor,
    issues: list[dict[str, Any]],
    shard_name: str,
    section_name: str,
    group_name: str,
) -> None:
    if torch.is_floating_point(tensor):
        non_finite = int(torch.sum(~torch.isfinite(tensor)).item())
        if non_finite:
            issue(
                issues,
                issue_type="non_finite_values",
                shard=shard_name,
                group=group_name,
                section=section_name,
                invalidCount=non_finite,
            )


def append_non_binary_issue(
    *,
    tensor: torch.Tensor,
    issues: list[dict[str, Any]],
    shard_name: str,
    section_name: str,
    group_name: str,
) -> None:
    invalid_count = count_non_binary_values(tensor)
    if invalid_count:
        issue(
            issues,
            issue_type="non_binary_values",
            shard=shard_name,
            group=group_name,
            section=section_name,
            invalidCount=invalid_count,
        )


def compare_float_tensors(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[bool, int]:
    if actual.shape != expected.shape:
        return False, -1
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    mismatch_count = int(torch.sum(~close).item())
    return mismatch_count == 0, mismatch_count


def summarize_density(tensor: torch.Tensor) -> dict[str, Any]:
    total = int(tensor.numel())
    nonzero = int(torch.count_nonzero(tensor).item())
    return {
        "total": total,
        "nonzero": nonzero,
        "density": (float(nonzero) / float(total)) if total else 0.0,
    }


def top_indexed_values(
    totals: torch.Tensor,
    *,
    id_to_name: list[str] | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    flat = totals.reshape(-1)
    for index, value in enumerate(flat.tolist()):
        numeric_value = float(value)
        if numeric_value == 0.0:
            continue
        entry: dict[str, Any] = {"index": index, "value": numeric_value}
        if id_to_name is not None and 0 <= index < len(id_to_name):
            entry["name"] = id_to_name[index]
        entries.append(entry)
    entries.sort(key=lambda item: (-item["value"], item["index"]))
    return entries[:limit]


def find_feature_indices(feature_names: list[str], names: list[str]) -> dict[str, int]:
    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    return {name: index_by_name[name] for name in names if name in index_by_name}


def audit_feature_shard(
    *,
    result: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Counter[Any]]]:
    shard_name = Path(str(result["tensorPath"])).name
    metadata = load_json(Path(str(result["metadataPath"])))
    flat_features, flat_labels = torch.load(Path(str(result["tensorPath"])), map_location="cpu", weights_only=True)
    structured_payload = torch.load(Path(str(result["structuredTensorPath"])), map_location="cpu", weights_only=True)

    feature_tensors = structured_payload["featureTensors"]
    label_tensors = structured_payload["labelTensors"]

    sample_count = int(metadata["sampleCount"])
    feature_sections = metadata["schema"]["featureSections"]
    label_sections = metadata["schema"]["labelSections"]
    feature_shape = tuple(int(value) for value in metadata["featureShape"])
    label_shape = tuple(int(value) for value in metadata["labelShape"])

    issues: list[dict[str, Any]] = []
    aggregates: dict[str, Counter[Any]] = defaultdict(Counter)

    if tuple(int(value) for value in flat_features.shape) != feature_shape:
        issue(
            issues,
            issue_type="flat_feature_shape_mismatch",
            shard=shard_name,
            expectedShape=list(feature_shape),
            observedShape=list(flat_features.shape),
        )
    if tuple(int(value) for value in flat_labels.shape) != label_shape:
        issue(
            issues,
            issue_type="flat_label_shape_mismatch",
            shard=shard_name,
            expectedShape=list(label_shape),
            observedShape=list(flat_labels.shape),
        )

    flattened_feature_sections: list[torch.Tensor] = []
    for section in feature_sections:
        section_name = str(section["name"])
        if section_name not in feature_tensors:
            issue(issues, issue_type="missing_feature_section", shard=shard_name, section=section_name)
            continue
        tensor = feature_tensors[section_name]
        expected_shape = (sample_count, *tuple(int(value) for value in section["shape"]))
        ensure_shape(
            tensor=tensor,
            expected_shape=expected_shape,
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="featureTensors",
        )
        ensure_dtype(
            tensor=tensor,
            expected_dtype=schema_dtype_to_torch(str(section["dtype"])),
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="featureTensors",
        )
        append_non_finite_issue(
            tensor=tensor,
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="featureTensors",
        )
        flattened = tensor.reshape(sample_count, -1)
        flattened_feature_sections.append(flattened.to(dtype=flat_features.dtype))

    for section in label_sections:
        section_name = str(section["name"])
        if section_name not in label_tensors:
            issue(issues, issue_type="missing_label_section", shard=shard_name, section=section_name)
            continue
        ensure_shape(
            tensor=label_tensors[section_name],
            expected_shape=(sample_count, *tuple(int(value) for value in section["shape"])),
            issues=issues,
            shard_name=shard_name,
            section_name=section_name,
            group_name="labelTensors",
        )

    if flattened_feature_sections:
        rebuilt_sections: list[torch.Tensor] = []
        for section, flattened in zip(feature_sections, flattened_feature_sections, strict=True):
            offset_entry = next(entry for entry in metadata["featureSectionOffsets"] if entry["name"] == section["name"])
            start = int(offset_entry["offset"])
            end = start + int(offset_entry["length"])
            flat_slice = flat_features[:, start:end]
            best = flattened
            best_mismatch = int(torch.sum(~torch.isclose(flat_slice, flattened, atol=1e-6, rtol=1e-6)).item())
            if len(section["shape"]) > 1:
                flipped = torch.flip(flattened, dims=[1])
                flipped_mismatch = int(torch.sum(~torch.isclose(flat_slice, flipped, atol=1e-6, rtol=1e-6)).item())
                if flipped_mismatch < best_mismatch:
                    best = flipped
                    best_mismatch = flipped_mismatch
            if best_mismatch:
                issue(
                    issues,
                    issue_type="feature_section_flatten_mismatch",
                    shard=shard_name,
                    section=section["name"],
                    mismatchCount=best_mismatch,
                )
            rebuilt_sections.append(best)

        rebuilt_flat = torch.cat(rebuilt_sections, dim=1)
        matches, mismatch_count = compare_float_tensors(actual=flat_features, expected=rebuilt_flat)
        if not matches:
            issue(issues, issue_type="feature_flatten_mismatch", shard=shard_name, mismatchCount=mismatch_count)

    section_density = {section_name: summarize_density(tensor) for section_name, tensor in feature_tensors.items()}

    available_action_mask = feature_tensors.get("availableActionMask")
    action_type_ids = label_tensors.get("actionTypeId")
    available_action_summary: dict[str, Any] = {}
    if available_action_mask is not None and action_type_ids is not None:
        append_non_binary_issue(
            tensor=available_action_mask,
            issues=issues,
            shard_name=shard_name,
            section_name="availableActionMask",
            group_name="featureTensors",
        )
        chosen_ids = action_type_ids[:, 0].to(torch.int64)
        out_of_range = int(torch.sum((chosen_ids < 0) | (chosen_ids >= available_action_mask.shape[1])).item())
        if out_of_range:
            issue(issues, issue_type="action_type_id_out_of_mask_range", shard=shard_name, invalidCount=out_of_range)
        safe_ids = chosen_ids.clamp(min=0, max=max(0, available_action_mask.shape[1] - 1))
        chosen_enabled = available_action_mask[torch.arange(sample_count), safe_ids]
        disabled_chosen = int(torch.sum(chosen_enabled == 0).item())
        if disabled_chosen:
            issue(issues, issue_type="chosen_action_disabled_by_mask", shard=shard_name, mismatchCount=disabled_chosen)
        enabled_counts = available_action_mask.to(torch.int64).sum(dim=0)
        aggregates["available_enabled_total"].update({int(idx): int(value) for idx, value in enumerate(enabled_counts.tolist()) if int(value)})
        aggregates["available_disabled_total"].update(
            {int(idx): int(sample_count - value) for idx, value in enumerate(enabled_counts.tolist()) if int(sample_count - value)}
        )
        available_action_summary = {
            "density": section_density["availableActionMask"],
            "chosenActionDisabledCount": disabled_chosen,
        }

    owned_composition = feature_tensors.get("ownedCompositionBow")
    owned_composition_summary: dict[str, Any] = {}
    if owned_composition is not None:
        if int(torch.sum(owned_composition < 0).item()):
            issue(issues, issue_type="negative_owned_composition_count", shard=shard_name)
        composition_meta = metadata["featureLayoutV1"].get("ownedCompositionBow", {})
        vocabulary = composition_meta.get("vocabulary", {})
        id_to_name = list(vocabulary.get("idToName", []))
        unknown_index = int(vocabulary.get("nameToId", {}).get(vocabulary.get("unknownName", "<unk>"), 0))
        totals = owned_composition.to(torch.int64).sum(dim=(0, 1))
        unknown_total = int(owned_composition[:, :, unknown_index].sum().item()) if unknown_index < owned_composition.shape[2] else 0
        aggregates["owned_composition_totals"].update({int(idx): int(value) for idx, value in enumerate(totals.tolist()) if int(value)})
        owned_composition_summary = {
            "density": section_density["ownedCompositionBow"],
            "unknownBucketTotal": unknown_total,
            "topOccupiedSlots": top_indexed_values(totals, id_to_name=id_to_name, limit=top_k),
        }

    enemy_memory = feature_tensors.get("enemyMemoryBow")
    enemy_memory_tech = feature_tensors.get("enemyMemoryTechFlags")
    enemy_memory_summary: dict[str, Any] = {}
    if enemy_memory is not None:
        if int(torch.sum(enemy_memory < 0).item()):
            issue(issues, issue_type="negative_enemy_memory_count", shard=shard_name)
        diffs = enemy_memory[1:] - enemy_memory[:-1]
        decreasing = int(torch.sum(diffs < 0).item())
        if decreasing:
            issue(issues, issue_type="enemy_memory_not_monotonic", shard=shard_name, mismatchCount=decreasing)
        memory_meta = metadata["featureLayoutV1"].get("enemyMemoryBow", {})
        vocabulary = memory_meta.get("vocabulary", {})
        id_to_name = list(vocabulary.get("idToName", []))
        totals = enemy_memory.to(torch.int64).max(dim=0).values.sum(dim=0)
        aggregates["enemy_memory_totals"].update({int(idx): int(value) for idx, value in enumerate(totals.tolist()) if int(value)})
        enemy_memory_summary = {
            "density": section_density["enemyMemoryBow"],
            "topSeenTypes": top_indexed_values(totals, id_to_name=id_to_name, limit=top_k),
        }
    if enemy_memory_tech is not None:
        append_non_binary_issue(
            tensor=enemy_memory_tech,
            issues=issues,
            shard_name=shard_name,
            section_name="enemyMemoryTechFlags",
            group_name="featureTensors",
        )
        diffs = enemy_memory_tech[1:] - enemy_memory_tech[:-1]
        decreasing = int(torch.sum(diffs < 0).item())
        if decreasing:
            issue(issues, issue_type="enemy_memory_tech_not_monotonic", shard=shard_name, mismatchCount=decreasing)

    build_order_trace = feature_tensors.get("buildOrderTrace")
    build_order_summary: dict[str, Any] = {}
    if build_order_trace is not None:
        non_missing = int(torch.sum(build_order_trace != LABEL_LAYOUT_V1_MISSING_INT).item())
        aggregates["build_order_ids"].update(
            int(value) for value in build_order_trace.reshape(-1).tolist() if int(value) != LABEL_LAYOUT_V1_MISSING_INT
        )
        build_order_summary = {
            "density": section_density["buildOrderTrace"],
            "nonMissingCount": non_missing,
            "missingCount": int(torch.sum(build_order_trace == LABEL_LAYOUT_V1_MISSING_INT).item()),
        }

    tech_state = feature_tensors.get("techState")
    tech_state_summary: dict[str, Any] = {}
    if tech_state is not None:
        append_non_binary_issue(
            tensor=tech_state,
            issues=issues,
            shard_name=shard_name,
            section_name="techState",
            group_name="featureTensors",
        )
        tech_meta = metadata["featureLayoutV1"].get("techState", {})
        feature_names = list(tech_meta.get("featureNames", []))
        activations = tech_state.to(torch.int64).sum(dim=0)
        if feature_names:
            aggregates["tech_state_flags"].update(
                {feature_names[idx]: int(value) for idx, value in enumerate(activations.tolist()) if int(value)}
            )
            index_by_name = {name: idx for idx, name in enumerate(feature_names)}
            power_low_idx = index_by_name.get("power_low")
            power_satisfied_idx = index_by_name.get("power_satisfied")
            if power_low_idx is not None and power_satisfied_idx is not None:
                contradiction = int(torch.sum((tech_state[:, power_low_idx] == 1) & (tech_state[:, power_satisfied_idx] == 1)).item())
                if contradiction:
                    issue(issues, issue_type="tech_state_power_contradiction", shard=shard_name, mismatchCount=contradiction)
            prerequisite_pairs = [
                ("unlocks_infantry_production", "owned_has_barracks"),
                ("unlocks_vehicle_production", "owned_has_factory"),
                ("unlocks_air_production", "owned_has_airfield"),
                ("unlocks_naval_production", "owned_has_naval_yard"),
            ]
            for unlock_name, owned_name in prerequisite_pairs:
                unlock_idx = index_by_name.get(unlock_name)
                owned_idx = index_by_name.get(owned_name)
                if unlock_idx is None or owned_idx is None:
                    continue
                mismatch_count = int(torch.sum((tech_state[:, unlock_idx] == 1) & (tech_state[:, owned_idx] == 0)).item())
                if mismatch_count:
                    issue(
                        issues,
                        issue_type="tech_state_prerequisite_mismatch",
                        shard=shard_name,
                        unlockFlag=unlock_name,
                        requiredOwnedFlag=owned_name,
                        mismatchCount=mismatch_count,
                    )
        tech_state_summary = {
            "density": section_density["techState"],
            "topActiveFlags": top_counter_entries(aggregates["tech_state_flags"], key_name="name", limit=top_k),
        }

    production_state = feature_tensors.get("productionState")
    production_state_summary: dict[str, Any] = {}
    if production_state is not None:
        production_meta = metadata["featureLayoutV1"].get("productionState", {})
        feature_names = list(production_meta.get("featureNames", []))
        if feature_names:
            index_by_name = {name: idx for idx, name in enumerate(feature_names)}
            for name, idx in index_by_name.items():
                column = production_state[:, idx]
                if name.endswith("_progress"):
                    invalid = int(torch.sum((column < 0) | (column > 1)).item())
                    if invalid:
                        issue(
                            issues,
                            issue_type="production_progress_out_of_range",
                            shard=shard_name,
                            feature=name,
                            invalidCount=invalid,
                        )
                if name.endswith("_has_queue") or name.endswith("_has_items") or "_status_" in name:
                    append_non_binary_issue(
                        tensor=column,
                        issues=issues,
                        shard_name=shard_name,
                        section_name=f"productionState::{name}",
                        group_name="featureTensors",
                    )
            queue_type_names = list(production_meta.get("queueTypeNames", []))
            for queue_type_name in queue_type_names:
                prefix = f"production_{queue_type_name.lower()}"
                status_names = [
                    f"{prefix}_status_idle",
                    f"{prefix}_status_active",
                    f"{prefix}_status_on_hold",
                    f"{prefix}_status_ready",
                ]
                status_indices = [index_by_name[name] for name in status_names if name in index_by_name]
                if status_indices:
                    status_sum = production_state[:, status_indices].sum(dim=1)
                    invalid = int(torch.sum(status_sum > 1).item())
                    if invalid:
                        issue(
                            issues,
                            issue_type="production_multiple_status_flags",
                            shard=shard_name,
                            queueType=queue_type_name,
                            invalidCount=invalid,
                        )
            for idx, name in enumerate(feature_names):
                column = production_state[:, idx]
                if torch.count_nonzero(column):
                    aggregates["production_nonzero_features"].update({name: int(torch.count_nonzero(column).item())})
        production_state_summary = {
            "density": section_density["productionState"],
            "topNonzeroFeatures": top_counter_entries(aggregates["production_nonzero_features"], key_name="name", limit=top_k),
        }

    super_weapon_state = feature_tensors.get("superWeaponState")
    super_weapon_summary: dict[str, Any] = {}
    if super_weapon_state is not None:
        super_meta = metadata["featureLayoutV1"].get("superWeaponState", {})
        feature_names = list(super_meta.get("featureNames", []))
        if feature_names:
            index_by_name = {name: idx for idx, name in enumerate(feature_names)}
            for name, idx in index_by_name.items():
                column = super_weapon_state[:, idx]
                if name.endswith("_has") or "_status_" in name:
                    append_non_binary_issue(
                        tensor=column,
                        issues=issues,
                        shard_name=shard_name,
                        section_name=f"superWeaponState::{name}",
                        group_name="featureTensors",
                    )
                if name.endswith("_charge_progress_01"):
                    invalid = int(torch.sum((column < 0) | (column > 1)).item())
                    if invalid:
                        issue(
                            issues,
                            issue_type="super_weapon_progress_out_of_range",
                            shard=shard_name,
                            feature=name,
                            invalidCount=invalid,
                        )
        super_weapon_summary = {"density": section_density["superWeaponState"]}

    entity_features = feature_tensors.get("entityFeatures")
    scalar = feature_tensors.get("scalar")
    spatial = feature_tensors.get("spatial")
    minimap = feature_tensors.get("minimap")
    map_static = feature_tensors.get("mapStatic")
    entity_intent_summary: dict[str, Any] = {}

    if scalar is not None:
        scalar_names = list(metadata["schema"]["observation"].get("scalarFeatureNames", []))
        for name, idx in find_feature_indices(scalar_names, ["visible_tile_fraction"]).items():
            column = scalar[:, idx]
            invalid = int(torch.sum((column < 0) | (column > 1)).item())
            if invalid:
                issue(issues, issue_type="scalar_fraction_out_of_range", shard=shard_name, feature=name, invalidCount=invalid)

    if spatial is not None:
        invalid = int(torch.sum(spatial < 0).item())
        if invalid:
            issue(issues, issue_type="spatial_channel_negative", shard=shard_name, invalidCount=invalid)

    if minimap is not None:
        invalid = int(torch.sum(minimap < 0).item())
        if invalid:
            issue(issues, issue_type="minimap_channel_negative", shard=shard_name, invalidCount=invalid)

    if map_static is not None:
        invalid = int(torch.sum(map_static < 0).item())
        if invalid:
            issue(issues, issue_type="map_static_channel_negative", shard=shard_name, invalidCount=invalid)
        if map_static.shape[0] > 1:
            base = map_static[0:1]
            constant_mismatch = int(torch.sum(torch.abs(map_static - base) > 1e-6).item())
            if constant_mismatch:
                issue(issues, issue_type="map_static_not_constant_within_shard", shard=shard_name, mismatchCount=constant_mismatch)

    if entity_features is not None:
        entity_feature_names = list(metadata["schema"]["observation"].get("entityFeatureNames", []))
        intent_names = metadata["featureLayoutV1"].get("entityIntentSummary", {}).get("featureNames", [])
        intent_nonzero = Counter()
        for name, idx in find_feature_indices(entity_feature_names, list(intent_names)).items():
            column = entity_features[..., idx]
            nonzero = int(torch.count_nonzero(column).item())
            if nonzero:
                intent_nonzero[name] = nonzero
            if name not in {"intent_progress_01", "weapon_cooldown_progress_01", "intent_rally_distance_norm"}:
                append_non_binary_issue(
                    tensor=column,
                    issues=issues,
                    shard_name=shard_name,
                    section_name=f"entityFeatures::{name}",
                    group_name="featureTensors",
                )
            if name in {"intent_progress_01", "weapon_cooldown_progress_01", "intent_rally_distance_norm"}:
                invalid = int(torch.sum((column < 0) | (column > 1)).item())
                if invalid:
                    issue(issues, issue_type="entity_intent_value_out_of_range", shard=shard_name, feature=name, invalidCount=invalid)

        entity_intent_summary = {
            "entityFeatureDensity": section_density["entityFeatures"],
            "nonzeroIntentFeatures": top_counter_entries(intent_nonzero, key_name="name", limit=top_k),
        }

    shard_summary = {
        "shard": shard_name,
        "sampleCount": sample_count,
        "featureShape": list(feature_shape),
        "labelShape": list(label_shape),
        "issueCount": len(issues),
        "availableActionMask": available_action_summary,
        "ownedCompositionBow": owned_composition_summary,
        "buildOrderTrace": build_order_summary,
        "techState": tech_state_summary,
        "productionState": production_state_summary,
        "enemyMemoryBow": enemy_memory_summary,
        "superWeaponState": super_weapon_summary,
        "entityIntentSummary": entity_intent_summary,
    }
    return shard_summary, {"rawIssues": issues, **aggregates}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest_path = resolve_manifest_path(args.input)
    manifest = load_json(manifest_path)

    output_path = args.output.resolve() if args.output is not None else manifest_path.with_name("feature_tensor_audit.json")

    results = [result for result in manifest.get("results", []) if result.get("status") == "saved"]
    shard_summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    aggregate_counters: dict[str, Counter[Any]] = defaultdict(Counter)

    for result in results:
        shard_summary, shard_aggregates = audit_feature_shard(result=result, top_k=args.top_k)
        shard_summaries.append(shard_summary)
        issues.extend(shard_aggregates.pop("rawIssues", []))
        for key, counter in shard_aggregates.items():
            aggregate_counters[key].update(counter)

    report = {
        "createdAt": utc_now_iso(),
        "input": str(args.input.resolve()),
        "manifestPath": str(manifest_path),
        "runSummary": {
            "replayCount": int(manifest.get("replayCount", 0)),
            "savedShardCount": int(manifest.get("savedShardCount", 0)),
            "errorCount": int(manifest.get("errorCount", 0)),
        },
        "auditSummary": {"shardCount": len(shard_summaries), "issueCount": len(issues)},
        "aggregate": {
            "availableActionMask": {
                "topEnabledActions": top_counter_entries(aggregate_counters["available_enabled_total"], key_name="actionTypeId", limit=args.top_k),
                "topDisabledActions": top_counter_entries(aggregate_counters["available_disabled_total"], key_name="actionTypeId", limit=args.top_k),
            },
            "ownedCompositionBow": {
                "topOccupiedSlots": top_counter_entries(aggregate_counters["owned_composition_totals"], key_name="vocabId", limit=args.top_k),
            },
            "buildOrderTrace": {
                "topActionTypeIds": top_counter_entries(aggregate_counters["build_order_ids"], key_name="actionTypeId", limit=args.top_k),
            },
            "techState": {
                "topActiveFlags": top_counter_entries(aggregate_counters["tech_state_flags"], key_name="name", limit=args.top_k),
            },
            "productionState": {
                "topNonzeroFeatures": top_counter_entries(aggregate_counters["production_nonzero_features"], key_name="name", limit=args.top_k),
            },
            "enemyMemoryBow": {
                "topSeenTypes": top_counter_entries(aggregate_counters["enemy_memory_totals"], key_name="vocabId", limit=args.top_k),
            },
        },
        "shards": shard_summaries,
        "issues": issues,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Audited {len(shard_summaries)} shard(s).")
    print(f"Issues: {len(issues)}")
    print(f"Report: {output_path}")
    return 1 if args.strict and issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
