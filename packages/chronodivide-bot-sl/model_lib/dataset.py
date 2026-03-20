from __future__ import annotations

import bisect
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


def _normalize_filter_values(values: set[str] | list[str] | tuple[str, ...] | None) -> frozenset[str] | None:
    if values is None:
        return None
    normalized = {str(value) for value in values}
    return frozenset(normalized) if normalized else None


def _normalize_int_filter_values(values: set[int] | list[int] | tuple[int, ...] | None) -> frozenset[int] | None:
    if values is None:
        return None
    normalized = {int(value) for value in values}
    return frozenset(normalized) if normalized else None


@dataclass(frozen=True)
class ModelShardRecord:
    stem: str
    meta_path: Path
    sections_path: Path
    training_path: Path
    sections_path_v2: Path | None
    training_path_v2: Path | None
    flat_path: Path | None
    metadata: dict[str, Any]
    sample_count: int
    sample_count_v2: int | None
    replay_path: str
    replay_game_id: str
    map_name: str
    player_name: str
    player_country_name: str | None
    player_side_id: int | None


@dataclass(frozen=True)
class ModelShardFilter:
    map_names: frozenset[str] | None = None
    player_names: frozenset[str] | None = None
    player_country_names: frozenset[str] | None = None
    player_side_ids: frozenset[int] | None = None
    replay_game_ids: frozenset[str] | None = None
    replay_paths: frozenset[str] | None = None
    metadata_predicate: Callable[[ModelShardRecord], bool] | None = None

    @classmethod
    def create(
        cls,
        *,
        map_names: set[str] | list[str] | tuple[str, ...] | None = None,
        player_names: set[str] | list[str] | tuple[str, ...] | None = None,
        player_country_names: set[str] | list[str] | tuple[str, ...] | None = None,
        player_side_ids: set[int] | list[int] | tuple[int, ...] | None = None,
        replay_game_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        replay_paths: set[str] | list[str] | tuple[str, ...] | None = None,
        metadata_predicate: Callable[[ModelShardRecord], bool] | None = None,
    ) -> "ModelShardFilter":
        return cls(
            map_names=_normalize_filter_values(map_names),
            player_names=_normalize_filter_values(player_names),
            player_country_names=_normalize_filter_values(player_country_names),
            player_side_ids=_normalize_int_filter_values(player_side_ids),
            replay_game_ids=_normalize_filter_values(replay_game_ids),
            replay_paths=_normalize_filter_values(replay_paths),
            metadata_predicate=metadata_predicate,
        )

    def matches(self, record: ModelShardRecord) -> bool:
        if self.map_names is not None and record.map_name not in self.map_names:
            return False
        if self.player_names is not None and record.player_name not in self.player_names:
            return False
        if self.player_country_names is not None and record.player_country_name not in self.player_country_names:
            return False
        if self.player_side_ids is not None and record.player_side_id not in self.player_side_ids:
            return False
        if self.replay_game_ids is not None and record.replay_game_id not in self.replay_game_ids:
            return False
        if self.replay_paths is not None and record.replay_path not in self.replay_paths:
            return False
        if self.metadata_predicate is not None and not self.metadata_predicate(record):
            return False
        return True


@dataclass
class _LoadedShard:
    feature_sections: dict[str, torch.Tensor]
    label_sections: dict[str, torch.Tensor]
    training_targets: dict[str, torch.Tensor]
    training_masks: dict[str, torch.Tensor]
    sample_context: dict[str, torch.Tensor]


def _get_current_player_info(metadata: dict[str, Any]) -> tuple[str | None, int | None]:
    player_name = str(metadata.get("playerName"))
    replay_players = metadata.get("replay", {}).get("players", [])
    for player_info in replay_players:
        if str(player_info.get("name")) != player_name:
            continue
        country_name = player_info.get("countryName")
        side_id = player_info.get("sideId")
        return (None if country_name is None else str(country_name), None if side_id is None else int(side_id))
    return (None, None)


def _extract_v2_sample_count(metadata: dict[str, Any]) -> int | None:
    v2_sample_count = metadata.get("v2CommandSampleCount")
    if v2_sample_count is not None:
        return int(v2_sample_count)

    structured_feature_shapes_v2 = metadata.get("structuredFeatureShapesV2", {})
    scalar_shape = structured_feature_shapes_v2.get("scalar")
    if isinstance(scalar_shape, list) and scalar_shape:
        return int(scalar_shape[0])

    structured_label_shapes_v2 = metadata.get("structuredLabelShapesV2", {})
    action_family_shape = structured_label_shapes_v2.get("actionFamilyId")
    if isinstance(action_family_shape, list) and action_family_shape:
        return int(action_family_shape[0])

    return None


def _build_record(meta_path: Path) -> ModelShardRecord:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sections_path = Path(metadata["structuredTensorPath"])
    training_path = Path(metadata["trainingTargetTensorPath"])
    sections_path_v2_value = metadata.get("structuredV2TensorPath")
    training_path_v2_value = metadata.get("trainingTargetTensorPathV2")
    sections_path_v2 = Path(sections_path_v2_value) if sections_path_v2_value else None
    training_path_v2 = Path(training_path_v2_value) if training_path_v2_value else None
    flat_path_value = metadata.get("tensorPath")
    flat_path = Path(flat_path_value) if flat_path_value else None

    if not sections_path.exists():
        raise FileNotFoundError(f"Structured tensor sidecar is missing: {sections_path}")
    if not training_path.exists():
        raise FileNotFoundError(f"Training target sidecar is missing: {training_path}")
    if flat_path is not None and not flat_path.exists():
        raise FileNotFoundError(f"Flat tensor shard is missing: {flat_path}")
    if sections_path_v2 is not None and not sections_path_v2.exists():
        raise FileNotFoundError(f"Structured V2 tensor sidecar is missing: {sections_path_v2}")
    if training_path_v2 is not None and not training_path_v2.exists():
        raise FileNotFoundError(f"Training target V2 sidecar is missing: {training_path_v2}")

    country_name, side_id = _get_current_player_info(metadata)
    replay_info = metadata["replay"]
    return ModelShardRecord(
        stem=meta_path.name.removesuffix(".meta.json"),
        meta_path=meta_path,
        sections_path=sections_path,
        training_path=training_path,
        sections_path_v2=sections_path_v2,
        training_path_v2=training_path_v2,
        flat_path=flat_path,
        metadata=metadata,
        sample_count=int(metadata["sampleCount"]),
        sample_count_v2=_extract_v2_sample_count(metadata),
        replay_path=str(replay_info["path"]),
        replay_game_id=str(replay_info["gameId"]),
        map_name=str(replay_info["mapName"]),
        player_name=str(metadata["playerName"]),
        player_country_name=country_name,
        player_side_id=side_id,
    )


def discover_model_shards(root_dir: Path, shard_filter: ModelShardFilter | None = None) -> list[ModelShardRecord]:
    root_dir = root_dir.resolve()
    records: list[ModelShardRecord] = []
    for meta_path in sorted(root_dir.glob("*.meta.json")):
        record = _build_record(meta_path)
        if shard_filter is not None and not shard_filter.matches(record):
            continue
        records.append(record)
    return records


def summarize_model_shards(records: list[ModelShardRecord]) -> dict[str, Any]:
    map_counter = Counter(record.map_name for record in records)
    player_counter = Counter(record.player_name for record in records)
    country_counter = Counter(record.player_country_name or "<unknown>" for record in records)
    side_counter = Counter(str(record.player_side_id) if record.player_side_id is not None else "<unknown>" for record in records)
    total_samples = sum(int(record.sample_count) for record in records)
    return {
        "shardCount": len(records),
        "sampleCount": total_samples,
        "maps": dict(sorted(map_counter.items())),
        "players": dict(sorted(player_counter.items())),
        "playerCountries": dict(sorted(country_counter.items())),
        "playerSides": dict(sorted(side_counter.items())),
    }


class RA2SLSectionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        shard_records: list[ModelShardRecord],
        *,
        artifact_variant: str = "v1",
        cache_size: int | None = 2,
    ) -> None:
        if not shard_records:
            raise ValueError("RA2SLSectionDataset requires at least one shard record.")
        if artifact_variant not in {"v1", "v2"}:
            raise ValueError(f"artifact_variant must be 'v1' or 'v2', got {artifact_variant}.")
        self.shard_records = list(shard_records)
        self.artifact_variant = artifact_variant
        self.cache_size = cache_size
        self._cumulative_sizes: list[int] = []
        running_total = 0
        for record in self.shard_records:
            running_total += self._get_record_sample_count(record)
            self._cumulative_sizes.append(running_total)
        self._loaded_shards: "OrderedDict[int, _LoadedShard]" = OrderedDict()

    @classmethod
    def from_directory(
        cls,
        root_dir: Path,
        *,
        shard_filter: ModelShardFilter | None = None,
        artifact_variant: str = "v1",
        cache_size: int | None = 2,
    ) -> "RA2SLSectionDataset":
        records = discover_model_shards(root_dir, shard_filter=shard_filter)
        return cls(records, artifact_variant=artifact_variant, cache_size=cache_size)

    def _resolve_artifact_paths(self, record: ModelShardRecord) -> tuple[Path, Path]:
        if self.artifact_variant == "v1":
            return record.sections_path, record.training_path
        if record.sections_path_v2 is None or record.training_path_v2 is None:
            raise FileNotFoundError(
                f"{record.stem} does not have V2 sidecars, but artifact_variant='v2' was requested."
            )
        return record.sections_path_v2, record.training_path_v2

    def _get_record_sample_count(self, record: ModelShardRecord) -> int:
        if self.artifact_variant == "v1":
            return int(record.sample_count)
        if record.sample_count_v2 is None:
            raise ValueError(
                f"{record.stem} is missing v2CommandSampleCount metadata, but artifact_variant='v2' was requested."
            )
        return int(record.sample_count_v2)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def _locate_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Sample index out of range: {index}")

        shard_index = bisect.bisect_right(self._cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self._cumulative_sizes[shard_index - 1]
        local_index = index - shard_start
        return shard_index, local_index

    def _validate_loaded_shard(self, record: ModelShardRecord, loaded_shard: _LoadedShard) -> None:
        expected_samples = self._get_record_sample_count(record)
        for section_name, tensor in loaded_shard.feature_sections.items():
            if int(tensor.shape[0]) != expected_samples:
                raise ValueError(
                    f"{record.stem} feature section {section_name} has {tensor.shape[0]} rows, expected {expected_samples}."
                )
        for section_name, tensor in loaded_shard.label_sections.items():
            if int(tensor.shape[0]) != expected_samples:
                raise ValueError(
                    f"{record.stem} label section {section_name} has {tensor.shape[0]} rows, expected {expected_samples}."
                )
        for section_name, tensor in loaded_shard.training_targets.items():
            if int(tensor.shape[0]) != expected_samples:
                raise ValueError(
                    f"{record.stem} training target {section_name} has {tensor.shape[0]} rows, expected {expected_samples}."
                )
        for section_name, tensor in loaded_shard.training_masks.items():
            if int(tensor.shape[0]) != expected_samples:
                raise ValueError(
                    f"{record.stem} training mask {section_name} has {tensor.shape[0]} rows, expected {expected_samples}."
                )

    def _load_shard(self, shard_index: int) -> _LoadedShard:
        if shard_index in self._loaded_shards:
            loaded = self._loaded_shards.pop(shard_index)
            self._loaded_shards[shard_index] = loaded
            return loaded

        record = self.shard_records[shard_index]
        sections_path, training_path = self._resolve_artifact_paths(record)
        structured = torch.load(str(sections_path), map_location="cpu", weights_only=True)
        training = torch.load(str(training_path), map_location="cpu", weights_only=True)
        loaded = _LoadedShard(
            feature_sections=dict(structured["featureTensors"]),
            label_sections=dict(structured["labelTensors"]),
            training_targets=dict(training["trainingTargets"]),
            training_masks=dict(training["trainingMasks"]),
            sample_context=dict(structured["sampleContext"]),
        )
        self._validate_loaded_shard(record, loaded)

        self._loaded_shards[shard_index] = loaded
        if self.cache_size is not None and self.cache_size >= 0:
            while len(self._loaded_shards) > self.cache_size:
                self._loaded_shards.popitem(last=False)
        return loaded

    def _build_metadata(
        self,
        *,
        record: ModelShardRecord,
        shard_index: int,
        global_index: int | None,
        local_index: int | None = None,
        window_start_local_index: int | None = None,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        sections_path, training_path = self._resolve_artifact_paths(record)
        metadata: dict[str, Any] = {
            "global_index": global_index,
            "local_index": local_index,
            "shard_index": shard_index,
            "shard_stem": record.stem,
            "artifact_variant": self.artifact_variant,
            "meta_path": str(record.meta_path),
            "sections_path": str(sections_path),
            "training_path": str(training_path),
            "replay_path": record.replay_path,
            "replay_game_id": record.replay_game_id,
            "map_name": record.map_name,
            "player_name": record.player_name,
            "player_country_name": record.player_country_name,
            "player_side_id": record.player_side_id,
            "sample_count": self._get_record_sample_count(record),
            "sample_count_v1": int(record.sample_count),
            "sample_count_v2": None if record.sample_count_v2 is None else int(record.sample_count_v2),
        }
        if window_start_local_index is not None and window_size is not None:
            metadata["window_start_local_index"] = window_start_local_index
            metadata["window_end_local_index"] = window_start_local_index + window_size - 1
            metadata["window_size"] = window_size
        return metadata

    def _build_single_sample(
        self,
        *,
        record: ModelShardRecord,
        loaded: _LoadedShard,
        shard_index: int,
        local_index: int,
        global_index: int,
    ) -> dict[str, Any]:
        return {
            "feature_sections": {
                name: tensor[local_index]
                for name, tensor in loaded.feature_sections.items()
            },
            "label_sections": {
                name: tensor[local_index]
                for name, tensor in loaded.label_sections.items()
            },
            "training_targets": {
                name: tensor[local_index]
                for name, tensor in loaded.training_targets.items()
            },
            "training_masks": {
                name: tensor[local_index]
                for name, tensor in loaded.training_masks.items()
            },
            "sample_context": {
                name: tensor[local_index]
                for name, tensor in loaded.sample_context.items()
            },
            "metadata": self._build_metadata(
                record=record,
                shard_index=shard_index,
                global_index=global_index,
                local_index=local_index,
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local_index = self._locate_index(index)
        record = self.shard_records[shard_index]
        loaded = self._load_shard(shard_index)
        return self._build_single_sample(
            record=record,
            loaded=loaded,
            shard_index=shard_index,
            local_index=local_index,
            global_index=index,
        )


class RA2SLSequenceWindowDataset(RA2SLSectionDataset):
    def __init__(
        self,
        shard_records: list[ModelShardRecord],
        *,
        window_size: int,
        window_stride: int = 1,
        artifact_variant: str = "v1",
        cache_size: int | None = 2,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if window_stride <= 0:
            raise ValueError(f"window_stride must be positive, got {window_stride}.")
        super().__init__(shard_records, artifact_variant=artifact_variant, cache_size=cache_size)
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self._window_index: list[tuple[int, int]] = []
        for shard_index, record in enumerate(self.shard_records):
            sample_count = self._get_record_sample_count(record)
            if sample_count < self.window_size:
                continue
            max_start = sample_count - self.window_size
            for start in range(0, max_start + 1, self.window_stride):
                self._window_index.append((shard_index, start))
        if not self._window_index:
            raise ValueError(
                "RA2SLSequenceWindowDataset found no valid windows. "
                f"window_size={self.window_size} may be too large for the selected shards."
            )

    @classmethod
    def from_directory(
        cls,
        root_dir: Path,
        *,
        window_size: int,
        window_stride: int = 1,
        shard_filter: ModelShardFilter | None = None,
        artifact_variant: str = "v1",
        cache_size: int | None = 2,
    ) -> "RA2SLSequenceWindowDataset":
        records = discover_model_shards(root_dir, shard_filter=shard_filter)
        return cls(
            records,
            window_size=window_size,
            window_stride=window_stride,
            artifact_variant=artifact_variant,
            cache_size=cache_size,
        )

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Sequence window index out of range: {index}")

        shard_index, window_start = self._window_index[index]
        record = self.shard_records[shard_index]
        loaded = self._load_shard(shard_index)
        window_end = window_start + self.window_size

        return {
            "feature_sections": {
                name: tensor[window_start:window_end]
                for name, tensor in loaded.feature_sections.items()
            },
            "label_sections": {
                name: tensor[window_start:window_end]
                for name, tensor in loaded.label_sections.items()
            },
            "training_targets": {
                name: tensor[window_start:window_end]
                for name, tensor in loaded.training_targets.items()
            },
            "training_masks": {
                name: tensor[window_start:window_end]
                for name, tensor in loaded.training_masks.items()
            },
            "sample_context": {
                name: tensor[window_start:window_end]
                for name, tensor in loaded.sample_context.items()
            },
            "metadata": self._build_metadata(
                record=record,
                shard_index=shard_index,
                global_index=index,
                window_start_local_index=window_start,
                window_size=self.window_size,
            ),
        }
