from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from transform_lib.action_labels import (
    get_action_family_name,
    get_legacy_label_tensors,
    get_order_type_name,
)
from transform_lib.common import LABEL_LAYOUT_V1_DELAY_BINS, LABEL_LAYOUT_V1_MISSING_INT, TransformConfig
from transform_lib.schema_utils import rebuild_flat_tensor


def stable_keep_fraction(
    *,
    replay_path: Path,
    player_name: str,
    sample: dict[str, Any],
    sample_index: int,
    filter_seed: int,
) -> float:
    legacy_label_tensors = get_legacy_label_tensors(sample)
    key = "|".join(
        [
            str(filter_seed),
            str(replay_path),
            player_name,
            str(sample.get("tick")),
            str(sample.get("playerId")),
            str(sample_index),
            str(legacy_label_tensors["rawActionId"][0]),
            str(sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64 - 1)


def get_sample_filter_bucket(sample: dict[str, Any], dataset: dict[str, Any]) -> tuple[str, float]:
    filter_config = dataset["filterConfig"]
    action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
    if isinstance(action_type_name, str):
        if action_type_name == "NoAction":
            return "no_action", float(filter_config["noActionKeepProb"])
        if action_type_name == "SelectUnits":
            return "select_units", float(filter_config["selectUnitsKeepProb"])
        if action_type_name.startswith("Order::"):
            parts = action_type_name.split("::")
            order_type_name = parts[1] if len(parts) > 1 else None
            if order_type_name in {"Move", "ForceMove"}:
                return "order_move", float(filter_config["moveOrderKeepProb"])
            if order_type_name == "Gather":
                return "order_gather", float(filter_config["gatherOrderKeepProb"])
            if order_type_name in {"Attack", "ForceAttack", "AttackMove"}:
                return "order_attack", float(filter_config["attackOrderKeepProb"])

    action_family = get_action_family_name(sample, dataset)

    if action_family == "no_action":
        return "no_action", float(filter_config["noActionKeepProb"])

    if action_family == "select_units":
        return "select_units", float(filter_config["selectUnitsKeepProb"])

    if action_family == "order_units":
        order_type_name = get_order_type_name(sample, dataset)
        if order_type_name in {"Move", "ForceMove"}:
            return "order_move", float(filter_config["moveOrderKeepProb"])
        if order_type_name == "Gather":
            return "order_gather", float(filter_config["gatherOrderKeepProb"])
        if order_type_name in {"Attack", "ForceAttack", "AttackMove"}:
            return "order_attack", float(filter_config["attackOrderKeepProb"])

    return "keep_all", 1.0


def build_player_action_counts(samples: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    raw_counts: dict[int, int] = {}
    family_counts: dict[str, int] = {}
    action_type_counts: dict[str, int] = {}

    for sample in samples:
        legacy_label_tensors = get_legacy_label_tensors(sample)
        raw_action_id = int(legacy_label_tensors["rawActionId"][0])
        raw_counts[raw_action_id] = raw_counts.get(raw_action_id, 0) + 1

        family_name = get_action_family_name(sample, dataset)
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
        action_type_name = sample.get("derivedLabelMetadata", {}).get("actionTypeNameV1")
        if isinstance(action_type_name, str):
            action_type_counts[action_type_name] = action_type_counts.get(action_type_name, 0) + 1

    return {
        "rawActionCounts": [
            {"rawActionId": raw_action_id, "count": count}
            for raw_action_id, count in sorted(raw_counts.items(), key=lambda item: item[0])
        ],
        "actionFamilyCounts": [
            {"actionFamily": family_name, "count": count}
            for family_name, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "actionTypeCountsV1": [
            {"actionTypeName": action_type_name, "count": count}
            for action_type_name, count in sorted(action_type_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def rewrite_temporal_context(samples: list[dict[str, Any]]) -> None:
    previous_tick: int | None = None
    previous_action_type_id_v1 = -1
    previous_queue = -1

    for sample in samples:
        legacy_label_tensors = get_legacy_label_tensors(sample)
        current_tick = int(sample["tick"])
        delay_from_previous = -1 if previous_tick is None else current_tick - previous_tick
        sample["featureTensors"]["lastActionContext"] = [
            delay_from_previous,
            previous_action_type_id_v1,
            previous_queue,
        ]

        previous_tick = current_tick
        previous_action_type_id_v1 = int(sample["labelTensors"]["actionTypeId"][0])
        previous_queue = int(legacy_label_tensors["queue"][0])

    for index, sample in enumerate(samples):
        legacy_label_tensors = get_legacy_label_tensors(sample)
        next_tick = int(samples[index + 1]["tick"]) if index + 1 < len(samples) else None
        delay_to_next = -1 if next_tick is None else next_tick - int(sample["tick"])
        legacy_label_tensors["delayToNextAction"] = [delay_to_next]
        sample["labelTensors"]["delayBin"] = [
            LABEL_LAYOUT_V1_MISSING_INT
            if delay_to_next < 0
            else min(delay_to_next, LABEL_LAYOUT_V1_DELAY_BINS - 1)
        ]


def build_filter_config(config: TransformConfig) -> dict[str, Any]:
    return {
        "profile": config.action_filter_profile,
        "filterSeed": config.filter_seed,
        "noActionKeepProb": config.no_action_keep_prob,
        "selectUnitsKeepProb": config.select_units_keep_prob,
        "moveOrderKeepProb": config.move_order_keep_prob,
        "gatherOrderKeepProb": config.gather_order_keep_prob,
        "attackOrderKeepProb": config.attack_order_keep_prob,
        "notes": [
            "This filter is applied in chronodivide-bot-sl after py-chronodivide extracts the action-aligned sample stream.",
            "Temporal fields are rewritten on the final kept stream so lastActionContext and delayToNextAction stay consistent.",
            "The default mas profile downsamples common RA2 action patterns analogously to mini-AlphaStar's replay filtering.",
        ],
    }


def apply_action_filter_profile(
    config: TransformConfig,
    dataset: dict[str, Any],
    replay_path: Path,
    player_name: str,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_count = len(samples)
    if config.action_filter_profile == "none":
        kept_samples = list(samples)
        if kept_samples:
            rewrite_temporal_context(kept_samples)
            for sample in kept_samples:
                sample["flatFeatureTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["featureSections"], "featureTensors")
                sample["flatLabelTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["labelSections"], "labelTensors")
        return kept_samples, {
            "profile": "none",
            "sourceSampleCount": source_count,
            "keptSampleCount": len(kept_samples),
            "droppedSampleCount": source_count - len(kept_samples),
            "bucketCounts": [],
        }

    bucket_counts: dict[str, dict[str, Any]] = {}
    kept_samples: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        bucket_name, keep_prob = get_sample_filter_bucket(sample, dataset)
        bucket_record = bucket_counts.setdefault(
            bucket_name,
            {
                "bucket": bucket_name,
                "keepProb": keep_prob,
                "before": 0,
                "kept": 0,
                "dropped": 0,
            },
        )
        bucket_record["before"] += 1

        keep = keep_prob >= 1.0
        if not keep:
            keep = stable_keep_fraction(
                replay_path=replay_path,
                player_name=player_name,
                sample=sample,
                sample_index=sample_index,
                filter_seed=config.filter_seed,
            ) < keep_prob

        if keep:
            bucket_record["kept"] += 1
            kept_samples.append(sample)
        else:
            bucket_record["dropped"] += 1

    if kept_samples:
        rewrite_temporal_context(kept_samples)
        for sample in kept_samples:
            sample["flatFeatureTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["featureSections"], "featureTensors")
            sample["flatLabelTensor"] = rebuild_flat_tensor(sample, dataset["schema"]["labelSections"], "labelTensors")

    return kept_samples, {
        "profile": config.action_filter_profile,
        "sourceSampleCount": source_count,
        "keptSampleCount": len(kept_samples),
        "droppedSampleCount": source_count - len(kept_samples),
        "bucketCounts": sorted(bucket_counts.values(), key=lambda item: item["bucket"]),
    }
