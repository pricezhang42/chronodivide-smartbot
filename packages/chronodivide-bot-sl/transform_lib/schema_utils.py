from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from transform_lib.common import feature_dtype, label_dtype, schema_dtype_to_torch


def compute_flat_length(schema_sections: list[dict[str, Any]]) -> int:
    total = 0
    for section in schema_sections:
        length = 1
        for dimension in section["shape"]:
            length *= int(dimension)
        total += length
    return total


def append_schema_section(schema_sections: list[dict[str, Any]], *, name: str, shape: list[int], dtype: str) -> None:
    if any(section["name"] == name for section in schema_sections):
        return
    schema_sections.append({"name": name, "shape": shape, "dtype": dtype})


def get_section_shape(schema_sections: list[dict[str, Any]], section_name: str) -> list[int]:
    for section in schema_sections:
        if section["name"] == section_name:
            return list(section["shape"])
    raise KeyError(f"Section not found in schema: {section_name}")


def infer_nested_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        return tuple(int(dimension) for dimension in value.shape)
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)

    first_shape = infer_nested_shape(value[0])
    for item in value[1:]:
        item_shape = infer_nested_shape(item)
        if item_shape != first_shape:
            raise ValueError(f"Inconsistent nested shape: expected {first_shape}, got {item_shape}.")
    return (len(value), *first_shape)


def flatten_nested_values(value: Any) -> list[float | int]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        return [value]

    flattened: list[float | int] = []
    for item in value:
        flattened.extend(flatten_nested_values(item))
    return flattened


def flatten_js_nested_numbers(value: Any) -> list[float | int]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        return [value]

    flattened: list[float | int] = []
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            for item in reversed(current):
                stack.append(item)
            continue
        flattened.append(current)
    flattened.reverse()
    return flattened


def flatten_section_value(value: Any, shape: list[int]) -> list[float | int]:
    if len(shape) <= 1:
        return flatten_nested_values(value)
    return flatten_js_nested_numbers(value)


def values_match(expected: list[Any], actual: list[Any], tolerance: float = 1e-6) -> bool:
    if len(expected) != len(actual):
        return False

    for left, right in zip(expected, actual, strict=True):
        if isinstance(left, bool) or isinstance(right, bool):
            if bool(left) != bool(right):
                return False
            continue

        if isinstance(left, int) and isinstance(right, int):
            if left != right:
                return False
            continue

        try:
            if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
                return False
        except (TypeError, ValueError):
            if left != right:
                return False
    return True


def validate_flat_tensor_lengths(samples: list[dict[str, Any]], schema: dict[str, Any], replay_name: str, player_name: str) -> None:
    expected_feature_length = int(schema["flatFeatureLength"])
    expected_label_length = int(schema["flatLabelLength"])

    for index, sample in enumerate(samples):
        feature_length = len(sample.get("flatFeatureTensor", []))
        label_length = len(sample.get("flatLabelTensor", []))
        if feature_length != expected_feature_length:
            raise ValueError(
                f"{replay_name}/{player_name} sample {index} has flat feature length {feature_length}, "
                f"expected {expected_feature_length}."
            )
        if label_length != expected_label_length:
            raise ValueError(
                f"{replay_name}/{player_name} sample {index} has flat label length {label_length}, "
                f"expected {expected_label_length}."
            )


def validate_section_shapes(
    samples: list[dict[str, Any]],
    schema_sections: list[dict[str, Any]],
    tensor_key: str,
    replay_name: str,
    player_name: str,
) -> None:
    for sample_index, sample in enumerate(samples):
        tensor_sections = sample.get(tensor_key, {})
        for section in schema_sections:
            section_name = section["name"]
            if section_name not in tensor_sections:
                raise ValueError(
                    f"{replay_name}/{player_name} sample {sample_index} is missing {tensor_key}.{section_name}."
                )
            observed_shape = infer_nested_shape(tensor_sections[section_name])
            expected_shape = tuple(int(dimension) for dimension in section["shape"])
            if observed_shape != expected_shape:
                raise ValueError(
                    f"{replay_name}/{player_name} sample {sample_index} has shape {observed_shape} for "
                    f"{tensor_key}.{section_name}, expected {expected_shape}."
                )


def validate_flat_matches_sections(samples: list[dict[str, Any]], schema: dict[str, Any], replay_name: str, player_name: str) -> None:
    feature_sections = schema["featureSections"]
    label_sections = schema["labelSections"]

    for sample_index, sample in enumerate(samples):
        flattened_feature: list[Any] = []
        for section in feature_sections:
            flattened_feature.extend(flatten_section_value(sample["featureTensors"][section["name"]], section["shape"]))
        if not values_match(flattened_feature, sample["flatFeatureTensor"]):
            raise ValueError(
                f"{replay_name}/{player_name} sample {sample_index} flat feature tensor does not match structured sections."
            )

        flattened_label: list[Any] = []
        for section in label_sections:
            flattened_label.extend(flatten_section_value(sample["labelTensors"][section["name"]], section["shape"]))
        if not values_match(flattened_label, sample["flatLabelTensor"]):
            raise ValueError(
                f"{replay_name}/{player_name} sample {sample_index} flat label tensor does not match structured sections."
            )


def build_tensors(samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows = [sample["flatFeatureTensor"] for sample in samples]
    label_rows = [sample["flatLabelTensor"] for sample in samples]
    features = torch.from_numpy(np.asarray(feature_rows, dtype=np.float32)).to(dtype=feature_dtype())
    labels = torch.from_numpy(np.asarray(label_rows, dtype=np.int64)).to(dtype=label_dtype())
    return features, labels


def build_structured_section_tensors(
    samples: list[dict[str, Any]],
    schema_sections: list[dict[str, Any]],
    tensor_key: str,
) -> dict[str, torch.Tensor]:
    dtype_to_numpy = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    section_tensors: dict[str, torch.Tensor] = {}
    for section in schema_sections:
        section_name = section["name"]
        rows = [sample[tensor_key][section_name] for sample in samples]
        numpy_dtype = dtype_to_numpy[str(section["dtype"])]
        section_array = np.asarray(rows, dtype=numpy_dtype)
        section_tensors[section_name] = torch.from_numpy(section_array).to(dtype=schema_dtype_to_torch(str(section["dtype"])))
    return section_tensors


def build_sample_context_tensors(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    ticks = torch.tensor([sample["tick"] for sample in samples], dtype=torch.int32)
    player_ids = torch.tensor([sample["playerId"] for sample in samples], dtype=torch.int32)
    return {
        "ticks": ticks,
        "playerIds": player_ids,
    }


def build_section_offsets(schema_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets: list[dict[str, Any]] = []
    cursor = 0
    for section in schema_sections:
        length = 1
        for dimension in section["shape"]:
            length *= int(dimension)
        offsets.append(
            {
                "name": section["name"],
                "offset": cursor,
                "length": length,
                "shape": section["shape"],
                "dtype": section["dtype"],
            }
        )
        cursor += length
    return offsets


def rebuild_flat_tensor(sample: dict[str, Any], schema_sections: list[dict[str, Any]], tensor_key: str) -> list[float | int]:
    flattened: list[float | int] = []
    for section in schema_sections:
        flattened.extend(flatten_section_value(sample[tensor_key][section["name"]], section["shape"]))
    return flattened
