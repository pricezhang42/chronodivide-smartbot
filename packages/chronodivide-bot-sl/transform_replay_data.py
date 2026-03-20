#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transform Chronodivide replays into replay-level (features, labels) torch shards.

This is the SL-transformer-side bridge that mirrors the role of
mini-AlphaStar's `transform_replay_data.py`, while delegating replay parsing,
feature extraction, and action-label decoding to the reusable `py-chronodivide`
package.

Current behavior:
- iterate replay files
- call `py-chronodivide/extract_sl_tensors.mjs`
- group samples by player perspective
- derive static-dict-backed RA2 SL V1 action-type metadata
- apply mAS-style action filtering/downsampling on the action-aligned sample stream
- save replay-player `.pt` shards as `(features, labels)`
- save replay-player `.sections.pt` sidecars with structured feature/label tensors
- save replay-player `.training.pt` sidecars with model-ready derived targets and masks
- save sidecar metadata and a run manifest

Notes:
- feature tensors are stored as `float32`
- label tensors are stored as `int64`
- structured section tensors preserve the schema dtypes from `py-chronodivide`
- per-replay vocabularies still come from `py-chronodivide`, so schema metadata
  is saved per shard instead of globally
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from action_dict import STATIC_ACTION_DICT_VERSION
from transform_lib.action_labels import (
    augment_dataset_with_feature_context_v1,
    augment_dataset_with_label_layout_v1,
    build_global_action_type_vocabulary,
    register_dataset_action_types_globally,
)
from transform_lib.label_layout_v2 import (
    augment_dataset_with_label_layout_v2_preview,
    build_label_layout_v2_metadata,
    build_player_v2_canonical_samples,
)
from transform_lib.common import (
    PROJECT_ROOT,
    TransformConfig,
    TransformRunState,
    ensure_parent,
    iter_progress,
    list_replays,
    parse_args,
    player_output_stem,
    utc_now_iso,
    validate_config,
)
from transform_lib.filtering import (
    apply_action_filter_profile,
    build_filter_config,
    build_player_action_counts,
)
from transform_lib.feature_layout import (
    augment_dataset_with_available_action_mask,
    augment_dataset_with_build_order_trace,
    augment_dataset_with_current_selection_summary,
    augment_dataset_with_enemy_memory_bow,
    augment_dataset_with_entity_intent_summary,
    augment_dataset_with_map_static,
    augment_dataset_with_owned_composition_bow,
    augment_dataset_with_production_state,
    augment_dataset_with_scalar_core_identity,
    augment_dataset_with_super_weapon_state,
    augment_dataset_with_tech_state,
)
from transform_lib.schema_utils import (
    build_sample_context_tensors,
    build_section_offsets,
    build_structured_section_tensors,
    build_tensors,
    validate_flat_matches_sections,
    validate_flat_tensor_lengths,
    validate_section_shapes,
)
from transform_lib.training_targets import finalize_training_target_sidecars

EXTRACT_CACHE_VERSION = "v3_binary_flatten_fix"


def node_player_arg(player: str) -> str | None:
    normalized = player.strip()
    if normalized.lower() == "first":
        return None
    return normalized


def build_node_command(config: TransformConfig, replay_path: Path, output_path: Path) -> list[str]:
    command = [
        "node",
        str(config.py_chronodivide_script),
        "--replay",
        str(replay_path),
        "--data-dir",
        str(config.data_dir),
        "--max-entities",
        str(config.max_entities),
        "--max-selected-units",
        str(config.max_selected_units),
        "--spatial-size",
        str(config.spatial_size),
        "--minimap-size",
        str(config.minimap_size),
        "--output-format",
        "binary",
        "--output",
        str(output_path),
    ]

    player_arg = node_player_arg(config.player)
    if player_arg:
        command.extend(["--player", player_arg])
    if config.include_no_action:
        command.extend(["--include-no-action", "true"])
    if config.include_ui_actions:
        command.extend(["--include-ui-actions", "true"])
    if config.max_actions is not None:
        command.extend(["--max-actions", str(config.max_actions)])
    if config.max_tick is not None:
        command.extend(["--max-tick", str(config.max_tick)])
    return command


def build_extract_cache_path(config: TransformConfig, replay_path: Path) -> Path | None:
    if config.extract_cache_dir is None:
        return None

    extractor_dir = config.py_chronodivide_script.parent
    extractor_signature = []
    for source_path in sorted(extractor_dir.glob("*.mjs")):
        stat = source_path.stat()
        extractor_signature.append(
            {
                "path": str(source_path.resolve()),
                "mtimeNs": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )

    replay_stat = replay_path.stat()
    cache_key_payload = {
        "version": EXTRACT_CACHE_VERSION,
        "replayPath": str(replay_path.resolve()),
        "replaySize": replay_stat.st_size,
        "replayMtimeNs": replay_stat.st_mtime_ns,
        "dataDir": str(config.data_dir.resolve()),
        "player": node_player_arg(config.player),
        "includeNoAction": config.include_no_action,
        "includeUiActions": config.include_ui_actions,
        "maxActions": config.max_actions,
        "maxTick": config.max_tick,
        "maxEntities": config.max_entities,
        "maxSelectedUnits": config.max_selected_units,
        "spatialSize": config.spatial_size,
        "minimapSize": config.minimap_size,
        "extractorSignature": extractor_signature,
    }
    cache_hash = hashlib.sha1(json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return config.extract_cache_dir / f"{replay_path.stem}__{cache_hash}"


def numpy_dtype_for_schema(dtype_name: str) -> np.dtype[Any]:
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    try:
        return np.dtype(mapping[dtype_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported binary section dtype: {dtype_name}") from exc


def load_binary_section(
    dataset_dir: Path,
    sample_count: int,
    section: dict[str, Any],
    file_name: str,
) -> np.ndarray[Any, Any]:
    file_path = dataset_dir / file_name
    dtype = numpy_dtype_for_schema(str(section["dtype"]))
    shape = [int(dimension) for dimension in section["shape"]]
    values_per_sample = int(np.prod(shape, dtype=np.int64)) if shape else 1
    expected_value_count = sample_count * values_per_sample
    section_array = np.fromfile(file_path, dtype=dtype)
    if int(section_array.size) != expected_value_count:
        raise ValueError(
            f"Binary section {file_path} has {section_array.size} values, expected {expected_value_count}."
        )
    return section_array.reshape((sample_count, *shape))


def load_binary_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    support_path = dataset_dir / "support.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with support_path.open("r", encoding="utf-8") as handle:
        support_payload = json.load(handle)

    if manifest.get("format") != "sl_dataset_binary_v1":
        raise ValueError(f"Unsupported extractor binary format in {manifest_path}.")

    sample_count = int(manifest["sampleCount"])
    support_samples = list(support_payload.get("samples", []))
    if len(support_samples) != sample_count:
        raise ValueError(
            f"Support sample count mismatch for {dataset_dir}: {len(support_samples)} vs expected {sample_count}."
        )

    feature_sections = manifest["schema"]["featureSections"]
    label_sections = manifest["schema"]["labelSections"]
    feature_arrays = {
        section["name"]: load_binary_section(
            dataset_dir,
            sample_count,
            section,
            manifest["featureSectionFiles"][section["name"]],
        )
        for section in feature_sections
    }
    label_arrays = {
        section["name"]: load_binary_section(
            dataset_dir,
            sample_count,
            section,
            manifest["labelSectionFiles"][section["name"]],
        )
        for section in label_sections
    }

    samples: list[dict[str, Any]] = []
    for sample_index, support_sample in enumerate(support_samples):
        sample = dict(support_sample)
        sample["featureTensors"] = {
            section["name"]: feature_arrays[section["name"]][sample_index]
            for section in feature_sections
        }
        sample["labelTensors"] = {
            section["name"]: label_arrays[section["name"]][sample_index]
            for section in label_sections
        }
        samples.append(sample)

    return {
        "replay": manifest["replay"],
        "sampledPlayers": manifest["sampledPlayers"],
        "options": manifest["options"],
        "schema": manifest["schema"],
        "superWeaponSchema": manifest["superWeaponSchema"],
        "staticTechTree": manifest.get("staticTechTree"),
        "staticMapSchema": manifest["staticMapSchema"],
        "staticMapByPlayer": manifest["staticMapByPlayer"],
        "counts": manifest["counts"],
        "samples": samples,
    }


def run_py_chronodivide_extract(config: TransformConfig, replay_path: Path) -> dict[str, Any]:
    cache_path = build_extract_cache_path(config, replay_path)
    if cache_path is not None and cache_path.exists() and not config.refresh_extract_cache:
        return load_binary_dataset(cache_path)

    with tempfile.TemporaryDirectory(prefix="chronodivide_sl_") as temp_dir:
        temp_output = Path(temp_dir) / "dataset"
        command = build_node_command(config, replay_path, temp_output)
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "\n".join(
                    [
                        f"py-chronodivide extraction failed for {replay_path.name}",
                        f"command: {' '.join(command)}",
                        f"stdout tail: {completed.stdout[-4000:]}",
                        f"stderr tail: {completed.stderr[-4000:]}",
                    ]
                )
            )
        if not temp_output.exists():
            raise RuntimeError(f"Expected extractor output file was not created: {temp_output}")
        result_path = temp_output
        if cache_path is not None:
            if cache_path.exists():
                shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(temp_output, cache_path, dirs_exist_ok=True)
            result_path = cache_path
        return load_binary_dataset(result_path)


def group_samples_by_player(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        player_name = sample["playerName"]
        grouped.setdefault(player_name, []).append(sample)
    return grouped


def write_player_shard(
    config: TransformConfig,
    replay_path: Path,
    dataset: dict[str, Any],
    player_name: str,
    source_samples: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    filter_stats: dict[str, Any],
    player_counts: dict[str, Any],
) -> dict[str, Any]:
    output_stem = player_output_stem(replay_path, player_name)
    tensor_path = config.output_dir / f"{output_stem}.pt"
    structured_tensor_path = config.output_dir / f"{output_stem}.sections.pt"
    structured_v2_tensor_path = config.output_dir / f"{output_stem}.v2.sections.pt"
    metadata_path = config.output_dir / f"{output_stem}.meta.json"

    if tensor_path.exists() and structured_tensor_path.exists() and metadata_path.exists() and not config.overwrite:
        return {
            "status": "skipped",
            "replay": str(replay_path),
            "playerName": player_name,
            "tensorPath": str(tensor_path),
            "structuredTensorPath": str(structured_tensor_path),
            "metadataPath": str(metadata_path),
            "reason": "existing shard",
        }

    validate_flat_tensor_lengths(samples, dataset["schema"], replay_path.name, player_name)
    validate_section_shapes(samples, dataset["schema"]["featureSections"], "featureTensors", replay_path.name, player_name)
    validate_section_shapes(samples, dataset["schema"]["labelSections"], "labelTensors", replay_path.name, player_name)
    validate_flat_matches_sections(samples, dataset["schema"], replay_path.name, player_name)
    features, labels = build_tensors(samples)
    feature_section_tensors = build_structured_section_tensors(samples, dataset["schema"]["featureSections"], "featureTensors")
    label_section_tensors = build_structured_section_tensors(samples, dataset["schema"]["labelSections"], "labelTensors")
    sample_context_tensors = build_sample_context_tensors(samples)

    ensure_parent(tensor_path)
    torch.save((features, labels), tensor_path)
    torch.save(
        {
            "featureTensors": feature_section_tensors,
            "labelTensors": label_section_tensors,
            "sampleContext": sample_context_tensors,
        },
        structured_tensor_path,
    )

    v2_samples = build_player_v2_canonical_samples(source_samples, dataset)
    label_layout_v2 = build_label_layout_v2_metadata(dataset, player_name, len(v2_samples))
    structured_feature_shapes_v2: dict[str, list[int]] = {}
    structured_label_shapes_v2: dict[str, list[int]] = {}
    sample_context_shapes_v2: dict[str, list[int]] = {}
    if v2_samples:
        feature_section_tensors_v2 = build_structured_section_tensors(
            v2_samples,
            dataset["schema"]["featureSections"],
            "featureTensors",
        )
        label_section_tensors_v2 = build_structured_section_tensors(
            v2_samples,
            label_layout_v2["labelSections"],
            "labelTensors",
        )
        sample_context_tensors_v2 = build_sample_context_tensors(v2_samples)
        torch.save(
            {
                "featureTensors": feature_section_tensors_v2,
                "labelTensors": label_section_tensors_v2,
                "sampleContext": sample_context_tensors_v2,
            },
            structured_v2_tensor_path,
        )
        structured_feature_shapes_v2 = {
            name: list(tensor.shape) for name, tensor in feature_section_tensors_v2.items()
        }
        structured_label_shapes_v2 = {
            name: list(tensor.shape) for name, tensor in label_section_tensors_v2.items()
        }
        sample_context_shapes_v2 = {
            name: list(tensor.shape) for name, tensor in sample_context_tensors_v2.items()
        }

    metadata = {
        "createdAt": utc_now_iso(),
        "replay": dataset["replay"],
        "playerName": player_name,
        "sampleCount": len(samples),
        "featureShape": list(features.shape),
        "labelShape": list(labels.shape),
        "featureDType": str(features.dtype),
        "labelDType": str(labels.dtype),
        "schema": dataset["schema"],
        "legacyFeatureSchema": {
            "featureSections": dataset.get("legacyFeatureSections", []),
            "flatFeatureLength": dataset.get("legacyFlatFeatureLength"),
        },
        "legacyLabelSchema": {
            "labelSections": dataset.get("legacyLabelSections", []),
            "flatLabelLength": dataset.get("legacyFlatLabelLength"),
        },
        "featureSectionOffsets": build_section_offsets(dataset["schema"]["featureSections"]),
        "labelSectionOffsets": build_section_offsets(dataset["schema"]["labelSections"]),
        "structuredFeatureShapes": {
            name: list(tensor.shape) for name, tensor in feature_section_tensors.items()
        },
        "structuredLabelShapes": {
            name: list(tensor.shape) for name, tensor in label_section_tensors.items()
        },
        "sampleContextShapes": {
            name: list(tensor.shape) for name, tensor in sample_context_tensors.items()
        },
        "v2CommandSampleCount": len(v2_samples),
        "sourceOptions": dataset["options"],
        "sourceCounts": dataset["counts"],
        "playerCounts": player_counts,
        "actionFilter": filter_stats,
        "filterConfig": dataset["filterConfig"],
        "featureLayoutV1": dataset.get("featureLayoutV1"),
        "featureContextV1": dataset.get("featureContextV1"),
        "labelLayoutV1": dataset.get("labelLayoutV1"),
        "labelLayoutV2": label_layout_v2,
        "labelLayoutV2Preview": dataset.get("labelLayoutV2Preview", {}).get("byPlayer", {}).get(player_name),
        "staticMapPlayerMetadata": {
            "buildabilityReferenceName": dataset.get("staticMapByPlayer", {}).get(player_name, {}).get(
                "buildabilityReferenceName"
            ),
            "channelNames": dataset.get("staticMapSchema", {}).get("staticMapChannelNames"),
        },
        "tensorPath": str(tensor_path),
        "structuredTensorPath": str(structured_tensor_path),
        "structuredV2TensorPath": str(structured_v2_tensor_path) if v2_samples else None,
        "structuredFeatureShapesV2": structured_feature_shapes_v2,
        "structuredLabelShapesV2": structured_label_shapes_v2,
        "sampleContextShapesV2": sample_context_shapes_v2,
    }
    ensure_parent(metadata_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "status": "saved",
        "replay": str(replay_path),
        "playerName": player_name,
        "sampleCount": len(samples),
        "sourceSampleCount": int(filter_stats["sourceSampleCount"]),
        "droppedSampleCount": int(filter_stats["droppedSampleCount"]),
        "featureShape": list(features.shape),
        "labelShape": list(labels.shape),
        "tensorPath": str(tensor_path),
        "structuredTensorPath": str(structured_tensor_path),
        "structuredV2TensorPath": str(structured_v2_tensor_path) if v2_samples else None,
        "metadataPath": str(metadata_path),
    }


def transform_single_replay(config: TransformConfig, replay_path: Path, run_state: TransformRunState) -> list[dict[str, Any]]:
    dataset = run_py_chronodivide_extract(config, replay_path)
    augment_dataset_with_label_layout_v1(dataset)
    augment_dataset_with_label_layout_v2_preview(dataset)
    register_dataset_action_types_globally(dataset, run_state)
    augment_dataset_with_feature_context_v1(dataset)
    augment_dataset_with_current_selection_summary(dataset)
    augment_dataset_with_scalar_core_identity(dataset)
    augment_dataset_with_available_action_mask(dataset)
    augment_dataset_with_owned_composition_bow(dataset)
    augment_dataset_with_build_order_trace(dataset)
    augment_dataset_with_tech_state(dataset)
    augment_dataset_with_production_state(dataset)
    augment_dataset_with_super_weapon_state(dataset)
    augment_dataset_with_enemy_memory_bow(dataset)
    augment_dataset_with_entity_intent_summary(dataset)
    augment_dataset_with_map_static(dataset)
    dataset["filterConfig"] = build_filter_config(config)
    grouped_samples = group_samples_by_player(dataset.get("samples", []))
    if not grouped_samples:
        return [
            {
                "status": "empty",
                "replay": str(replay_path),
                "reason": "extractor returned no samples",
            }
        ]

    shard_results = []
    for player_name, player_samples in grouped_samples.items():
        filtered_samples, filter_stats = apply_action_filter_profile(config, dataset, replay_path, player_name, player_samples)
        if not filtered_samples:
            shard_results.append(
                {
                    "status": "empty",
                    "replay": str(replay_path),
                    "playerName": player_name,
                    "reason": "all samples were dropped by action filtering",
                    "sourceSampleCount": int(filter_stats["sourceSampleCount"]),
                    "droppedSampleCount": int(filter_stats["droppedSampleCount"]),
                }
            )
            continue

        player_counts = build_player_action_counts(filtered_samples, dataset)
        shard_results.append(
            write_player_shard(
                config,
                replay_path,
                dataset,
                player_name,
                player_samples,
                filtered_samples,
                filter_stats,
                player_counts,
            )
        )
    return shard_results


def write_manifest(
    config: TransformConfig,
    replay_paths: list[Path],
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    run_state: TransformRunState,
) -> Path:
    config_dict = asdict(config)
    config_dict["replay_dir"] = str(config.replay_dir)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["data_dir"] = str(config.data_dir)
    config_dict["py_chronodivide_script"] = str(config.py_chronodivide_script)
    config_dict["extract_cache_dir"] = str(config.extract_cache_dir) if config.extract_cache_dir is not None else None

    manifest = {
        "createdAt": utc_now_iso(),
        "config": config_dict,
        "extractCacheVersion": EXTRACT_CACHE_VERSION,
        "replayCount": len(replay_paths),
        "savedShardCount": sum(1 for result in results if result["status"] == "saved"),
        "skippedShardCount": sum(1 for result in results if result["status"] == "skipped"),
        "trainingTargetShardCount": sum(1 for result in results if result.get("trainingTargetTensorPath")),
        "emptyReplayCount": sum(1 for result in results if result["status"] == "empty"),
        "errorCount": len(errors),
        "staticActionDictVersion": STATIC_ACTION_DICT_VERSION,
        "labelLayoutV1GlobalActionVocabulary": build_global_action_type_vocabulary(run_state),
        "unseenObservedActionTypes": [
            {"name": name, "count": count}
            for name, count in sorted(run_state.unseen_action_type_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "results": results,
        "errors": errors,
    }
    manifest_path = config.output_dir / "manifest.json"
    ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_state = TransformRunState()

    replay_paths = list_replays(config)
    if not replay_paths:
        raise FileNotFoundError(
            f"No replay files matching {config.replay_glob!r} were found in {config.replay_dir}."
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for replay_path in iter_progress(replay_paths, "Transform replays"):
        try:
            results.extend(transform_single_replay(config, replay_path, run_state))
        except Exception as exc:  # pragma: no cover - failure path exercised by real runs
            errors.append(
                {
                    "replay": str(replay_path),
                    "errorType": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
            if config.fail_fast:
                raise

    finalize_training_target_sidecars(config, results, build_global_action_type_vocabulary(run_state))
    manifest_path = write_manifest(config, replay_paths, results, errors, run_state)
    print(f"Processed {len(replay_paths)} replay(s).")
    print(f"Saved {sum(1 for result in results if result['status'] == 'saved')} shard(s).")
    print(f"Training targets: {sum(1 for result in results if result.get('trainingTargetTensorPath'))} shard(s).")
    print(f"Manifest: {manifest_path}")

    if errors:
        print(f"Encountered {len(errors)} replay error(s). See manifest for details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
