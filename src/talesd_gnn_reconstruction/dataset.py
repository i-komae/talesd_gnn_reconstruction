from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from .constants import (
    DROPPED_NODE_FEATURE_COLUMNS,
    DROPPED_PULSE_FEATURE_COLUMNS,
    NODE_FEATURE_COLUMNS,
    PULSE_FEATURE_COLUMNS,
    WAVEFORM_FEATURE_CHANNELS,
    WAVEFORM_SCHEMA,
    WAVEFORM_TRACE_BINS,
)
from .progress import progress as _progress
from .progress import progress_bar as _progress_bar
from .progress import progress_interval_seconds as _progress_interval_seconds


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def identity(cls, n_features: int) -> "StandardScaler":
        return cls(np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32))

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "StandardScaler":
        if not arrays:
            raise ValueError("cannot fit scaler on empty data")
        data = np.concatenate(arrays, axis=0).astype(np.float32)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, value: np.ndarray) -> np.ndarray:
        return (value.astype(np.float32) - self.mean) / self.std

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        return value.astype(np.float32) * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardScaler":
        return cls(np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32))


class RunningFeatureStats:
    def __init__(self, n_features: int):
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[0] == 0:
            return

        batch_count = values.shape[0]
        batch_mean = np.mean(values, axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    def merge(self, other: "RunningFeatureStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = int(other.count)
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean = self.mean + delta * other.count / total
        self.m2 = self.m2 + other.m2 + delta**2 * self.count * other.count / total
        self.count = total

    def to_scaler(self) -> StandardScaler:
        if self.count == 0:
            return StandardScaler.identity(len(self.mean))
        variance = self.m2 / max(self.count, 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < 1.0e-6, 1.0, std)
        return StandardScaler(self.mean.astype(np.float32), std.astype(np.float32))


def _as_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, str | Path):
        return [Path(paths).expanduser()]
    return [Path(path).expanduser() for path in paths]


def _text_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _columns_from_handle(handle: h5py.File) -> dict[str, list[str]]:
    columns_text = _text_attr(handle.attrs.get("columns_json"), "{}")
    try:
        parsed = json.loads(columns_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    columns: dict[str, list[str]] = {}
    for key, values in parsed.items():
        if isinstance(values, list):
            columns[str(key)] = [str(value) for value in values]
    return columns


def _node_feature_selection_from_columns(
    columns: dict[str, list[str]],
    *,
    expected_columns: Sequence[str] | None = None,
    dropped_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    stored = list(columns.get("node_features") or [])
    if not stored:
        return None, []

    dropped = set(DROPPED_NODE_FEATURE_COLUMNS if dropped_columns is None else dropped_columns)
    kept_indices = [index for index, name in enumerate(stored) if name not in dropped]
    effective = [stored[index] for index in kept_indices]
    expected = list(NODE_FEATURE_COLUMNS if expected_columns is None else expected_columns)
    if effective != expected:
        raise ValueError(
            "stored node feature columns are incompatible with this code after dropping disabled columns: "
            f"stored={stored!r}, effective={effective!r}, expected={expected!r}. "
            "Re-export the graph HDF5 files or update the feature-column migration rule."
        )
    if len(kept_indices) == len(stored):
        return None, effective
    return np.asarray(kept_indices, dtype=np.int64), effective


def _pulse_feature_selection_from_columns(
    columns: dict[str, list[str]],
    *,
    expected_columns: Sequence[str] | None = None,
    dropped_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    stored = list(columns.get("pulse_features") or [])
    if not stored:
        return None, []

    dropped = set(DROPPED_PULSE_FEATURE_COLUMNS if dropped_columns is None else dropped_columns)
    kept_indices = [index for index, name in enumerate(stored) if name not in dropped]
    effective = [stored[index] for index in kept_indices]
    expected = list(PULSE_FEATURE_COLUMNS if expected_columns is None else expected_columns)
    if effective != expected:
        raise ValueError(
            "stored pulse feature columns are incompatible with this code after dropping disabled columns: "
            f"stored={stored!r}, effective={effective!r}, expected={expected!r}. "
            "Re-export the graph HDF5 files or update the feature-column migration rule."
        )
    if len(kept_indices) == len(stored):
        return None, effective
    return np.asarray(kept_indices, dtype=np.int64), effective


def _validate_waveform_schema(
    path: Path,
    handle: h5py.File,
    *,
    allowed_schemas: Sequence[str] | None = None,
    expected_feature_channels: Sequence[str] | None = None,
) -> None:
    stored_schema = _text_attr(handle.attrs.get("waveform_schema"), "")
    schemas = set([WAVEFORM_SCHEMA] if allowed_schemas is None else allowed_schemas)
    if stored_schema and stored_schema not in schemas:
        raise ValueError(
            f"{path} uses waveform_schema={stored_schema!r}, but this code expects one of {sorted(schemas)!r}. "
            "Re-export the graph HDF5 files from DST before training with this waveform definition."
        )

    columns = _columns_from_handle(handle)
    waveform_columns = columns.get("waveform_features")
    if waveform_columns is None:
        return
    expected = list(WAVEFORM_FEATURE_CHANNELS if expected_feature_channels is None else expected_feature_channels)
    if list(waveform_columns) != expected:
        raise ValueError(
            f"{path} stores waveform_features={list(waveform_columns)!r}, "
            f"but this code expects {expected!r}. "
            "Re-export the graph HDF5 files from DST; old compact or accepted-gapped waveforms cannot be "
            "interpreted as raw waveform plus accepted-pulse mask channels."
        )


class H5GraphDataset:
    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        require_target: bool = False,
        require_particle_label: bool = False,
        cache_size: int = 0,
        load_node_positions: bool = True,
        load_attrs: bool = True,
        load_particle_label: bool = False,
        load_detector_lids: bool = False,
        max_graphs: int | None = None,
        particle_filter: str = "all",
        show_progress: bool = False,
        max_open_files: int | None = None,
        expected_node_feature_columns: Sequence[str] | None = None,
        dropped_node_feature_columns: Sequence[str] | None = None,
        expected_pulse_feature_columns: Sequence[str] | None = None,
        dropped_pulse_feature_columns: Sequence[str] | None = None,
        allowed_waveform_schemas: Sequence[str] | None = None,
        expected_waveform_feature_channels: Sequence[str] | None = None,
    ):
        self.paths = _as_paths(path)
        self.require_target = require_target
        self.require_particle_label = require_particle_label
        self.cache_size = max(int(cache_size), 0)
        self.load_node_positions = load_node_positions
        self.load_attrs = load_attrs
        self.load_particle_label = load_particle_label or require_particle_label
        self.load_detector_lids = bool(load_detector_lids)
        self.max_graphs = None if max_graphs is None or max_graphs <= 0 else int(max_graphs)
        if max_open_files is None:
            max_open_files = int(os.environ.get("TALESD_GNN_H5_MAX_OPEN_FILES", "4"))
        self.max_open_files = max(int(max_open_files), 0)
        self.particle_filter = particle_filter.lower()
        if self.particle_filter not in {"all", "proton", "iron"}:
            raise ValueError("particle_filter must be 'all', 'proton', or 'iron'")
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._path_key_lists: list[list[str] | None] = []
        self._path_local_indices: list[list[int] | None] = []
        self._path_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        self._path_has_metadata: list[bool] = []
        self._node_feature_indices: list[np.ndarray | None] = []
        self._pulse_feature_indices: list[np.ndarray | None] = []
        self.node_feature_columns: list[str] = []
        self.pulse_feature_columns: list[str] = []
        self.columns_json = "{}"
        self.expected_node_feature_columns = (
            None if expected_node_feature_columns is None else list(expected_node_feature_columns)
        )
        self.dropped_node_feature_columns = (
            None if dropped_node_feature_columns is None else list(dropped_node_feature_columns)
        )
        self.expected_pulse_feature_columns = (
            None if expected_pulse_feature_columns is None else list(expected_pulse_feature_columns)
        )
        self.dropped_pulse_feature_columns = (
            None if dropped_pulse_feature_columns is None else list(dropped_pulse_feature_columns)
        )
        self.allowed_waveform_schemas = None if allowed_waveform_schemas is None else list(allowed_waveform_schemas)
        self.expected_waveform_feature_channels = (
            None if expected_waveform_feature_channels is None else list(expected_waveform_feature_channels)
        )

        remaining = self.max_graphs
        progress = _progress_bar("initialize graph shards", len(self.paths), enabled=show_progress)
        try:
            for path_index, graph_path in enumerate(self.paths):
                if remaining is not None and remaining <= 0:
                    break
                with h5py.File(graph_path, "r") as handle:
                    columns = _columns_from_handle(handle)
                    node_feature_indices, effective_node_columns = _node_feature_selection_from_columns(
                        columns,
                        expected_columns=self.expected_node_feature_columns,
                        dropped_columns=self.dropped_node_feature_columns,
                    )
                    pulse_feature_indices, effective_pulse_columns = _pulse_feature_selection_from_columns(
                        columns,
                        expected_columns=self.expected_pulse_feature_columns,
                        dropped_columns=self.dropped_pulse_feature_columns,
                    )
                    if path_index == 0:
                        effective_columns = dict(columns)
                        if effective_node_columns:
                            effective_columns["node_features"] = effective_node_columns
                            self.node_feature_columns = effective_node_columns
                        if effective_pulse_columns:
                            effective_columns["pulse_features"] = effective_pulse_columns
                            self.pulse_feature_columns = effective_pulse_columns
                        self.columns_json = (
                            json.dumps(effective_columns)
                            if effective_node_columns or effective_pulse_columns
                            else handle.attrs.get("columns_json", "{}")
                        )
                    elif effective_node_columns and self.node_feature_columns and effective_node_columns != self.node_feature_columns:
                        raise ValueError(
                            f"{graph_path} has node feature columns incompatible with the first shard: "
                            f"{effective_node_columns!r} != {self.node_feature_columns!r}"
                        )
                    elif effective_pulse_columns and self.pulse_feature_columns and effective_pulse_columns != self.pulse_feature_columns:
                        raise ValueError(
                            f"{graph_path} has pulse feature columns incompatible with the first shard: "
                            f"{effective_pulse_columns!r} != {self.pulse_feature_columns!r}"
                        )
                    self._node_feature_indices.append(node_feature_indices)
                    self._pulse_feature_indices.append(pulse_feature_indices)
                    _validate_waveform_schema(
                        graph_path,
                        handle,
                        allowed_schemas=self.allowed_waveform_schemas,
                        expected_feature_channels=self.expected_waveform_feature_channels,
                    )
                    events = handle["events"]
                    key_list_all = sorted(events.keys())
                    n_events = len(key_list_all)
                    dense_numeric_keys = n_events > 0 and all(
                        key == f"{index:08d}" for index, key in enumerate(key_list_all)
                    )
                    if dense_numeric_keys:
                        key_list = None
                    else:
                        key_list = key_list_all
                        n_events = len(key_list)
                    raw_n_events = n_events
                    metadata = handle.get("metadata")
                    selected_local_indices = None
                    if self.particle_filter != "all":
                        selected_local_indices = self._selected_particle_indices(
                            events=events,
                            metadata=metadata,
                            key_list=key_list,
                            n_events=raw_n_events,
                        )
                        n_events = len(selected_local_indices)
                    if remaining is not None:
                        n_events = min(n_events, remaining)
                        if selected_local_indices is not None:
                            selected_local_indices = selected_local_indices[:n_events]
                        if key_list is not None:
                            if selected_local_indices is None:
                                key_list = key_list[:n_events]
                        remaining -= n_events
                    self._path_key_lists.append(key_list)
                    self._path_local_indices.append(selected_local_indices)
                    self._path_lengths.append(n_events)
                    self._path_has_metadata.append(
                        metadata is not None
                        and "source_path" in metadata
                        and "particle_label" in metadata
                        and len(metadata["source_path"]) >= raw_n_events
                    )
                    total = n_events + (self._cumulative_lengths[-1] if self._cumulative_lengths else 0)
                    self._cumulative_lengths.append(total)
                progress.update(1)
        finally:
            progress.close()

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_cache"] = OrderedDict()
        return state

    def _handle(self, path_index: int) -> h5py.File:
        if path_index in self._handles:
            handle = self._handles.pop(path_index)
            self._handles[path_index] = handle
            return handle
        handle = h5py.File(self.paths[path_index], "r")
        self._handles[path_index] = handle
        if self.max_open_files > 0:
            while len(self._handles) > self.max_open_files:
                _old_path_index, old_handle = self._handles.popitem(last=False)
                old_handle.close()
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = OrderedDict()
        self._cache.clear()

    @staticmethod
    def _label_matches_filter(value: Any, particle_filter: str) -> bool:
        if value is None:
            return False
        label = float(value)
        if not np.isfinite(label):
            return False
        if particle_filter == "proton":
            return label < 0.5
        if particle_filter == "iron":
            return label >= 0.5
        return True

    @staticmethod
    def _group_particle_label(group: h5py.Group) -> float | None:
        value = None
        if "particle_label" in group:
            value = group["particle_label"][()]
        else:
            value = group.attrs.get("particle_label", group.attrs.get("particle_is_iron", None))
            if value is None:
                parttype = int(group.attrs.get("parttype", -1))
                if parttype == 14:
                    value = 0.0
                elif parttype == 5626:
                    value = 1.0
            if value is None:
                source_path = str(group.attrs.get("source_path", "")).lower()
                if "/proton/" in source_path or "tale_proton" in source_path:
                    value = 0.0
                elif "/iron/" in source_path or "tale_iron" in source_path:
                    value = 1.0
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    def _selected_particle_indices(
        self,
        *,
        events: h5py.Group,
        metadata: h5py.Group | None,
        key_list: list[str] | None,
        n_events: int,
    ) -> list[int]:
        if metadata is not None and "particle_label" in metadata and len(metadata["particle_label"]) >= n_events:
            labels = metadata["particle_label"][:n_events]
            finite = np.isfinite(labels)
            if self.particle_filter == "proton":
                mask = finite & (labels < 0.5)
            elif self.particle_filter == "iron":
                mask = finite & (labels >= 0.5)
            else:
                mask = finite
            return np.flatnonzero(mask).astype(np.int64).tolist()

        selected = []
        for local_index in range(n_events):
            key = f"{local_index:08d}" if key_list is None else key_list[local_index]
            label = self._group_particle_label(events[key])
            if self._label_matches_filter(label, self.particle_filter):
                selected.append(local_index)
        return selected

    def _locate(self, index: int) -> tuple[int, int, str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path_index = bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[path_index - 1] if path_index > 0 else 0
        local_index = index - previous
        selected_local_indices = self._path_local_indices[path_index]
        if selected_local_indices is not None:
            local_index = selected_local_indices[local_index]
        key_list = self._path_key_lists[path_index]
        key = f"{local_index:08d}" if key_list is None else key_list[local_index]
        return path_index, local_index, key

    def _metadata_value(self, path_index: int, local_index: int, name: str) -> Any:
        if not self._path_has_metadata[path_index]:
            return None
        value = self._handle(path_index)["metadata"][name][local_index]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def source_path(self, index: int) -> str:
        path_index, local_index, key = self._locate(index)
        value = self._metadata_value(path_index, local_index, "source_path")
        if value is not None:
            return str(value)
        return str(self._handle(path_index)["events"][key].attrs.get("source_path", ""))

    def target(self, index: int) -> np.ndarray | None:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        if "target" not in group:
            return None
        return group["target"][()].astype(np.float32)

    def particle_label(self, index: int) -> float | None:
        path_index, local_index, key = self._locate(index)
        value = self._metadata_value(path_index, local_index, "particle_label")
        if value is None:
            group = self._handle(path_index)["events"][key]
            return self._group_particle_label(group)
        return self._group_particle_label_from_value(value)

    @staticmethod
    def _group_particle_label_from_value(value: Any) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _parse_lids_attr(value: Any, n_nodes: int) -> np.ndarray:
        if value is None:
            return np.full(n_nodes, -1, dtype=np.int64)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray):
            lids = value.astype(np.int64, copy=False).reshape(-1)
        else:
            text = str(value).strip()
            if not text:
                return np.full(n_nodes, -1, dtype=np.int64)
            lids = np.fromiter((int(item) for item in text.split(",") if item.strip()), dtype=np.int64)
        if lids.shape[0] == n_nodes:
            return lids.astype(np.int64, copy=False)
        out = np.full(n_nodes, -1, dtype=np.int64)
        n_copy = min(n_nodes, int(lids.shape[0]))
        if n_copy > 0:
            out[:n_copy] = lids[:n_copy]
        return out

    @staticmethod
    def _group_detector_lids(group: h5py.Group, n_nodes: int) -> np.ndarray:
        if "node_lids" in group:
            lids = group["node_lids"][()].astype(np.int64, copy=False).reshape(-1)
            if lids.shape[0] == n_nodes:
                return lids
            out = np.full(n_nodes, -1, dtype=np.int64)
            n_copy = min(n_nodes, int(lids.shape[0]))
            if n_copy > 0:
                out[:n_copy] = lids[:n_copy]
            return out
        return H5GraphDataset._parse_lids_attr(group.attrs.get("lids", None), n_nodes)

    def detector_lids(self, index: int) -> np.ndarray:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        if "node_features" in group:
            n_nodes = int(group["node_features"].shape[0])
        else:
            n_nodes = 0
        return self._group_detector_lids(group, n_nodes)

    def _node_features_from_group(self, path_index: int, group: h5py.Group) -> np.ndarray:
        node_features = group["node_features"][()].astype(np.float32)
        indices = self._node_feature_indices[path_index] if path_index < len(self._node_feature_indices) else None
        if indices is not None:
            node_features = node_features[:, indices]
        return node_features

    def _pulse_features_from_group(self, path_index: int, group: h5py.Group) -> np.ndarray:
        if "pulse_features" not in group:
            return np.zeros((0, len(PULSE_FEATURE_COLUMNS)), dtype=np.float32)
        pulse_features = group["pulse_features"][()].astype(np.float32)
        indices = self._pulse_feature_indices[path_index] if path_index < len(self._pulse_feature_indices) else None
        if indices is not None:
            pulse_features = pulse_features[:, indices]
        return pulse_features

    def node_feature_dim_for_group(self, path_index: int, group: h5py.Group) -> int:
        indices = self._node_feature_indices[path_index] if path_index < len(self._node_feature_indices) else None
        if indices is not None:
            return int(indices.shape[0])
        return int(group["node_features"].shape[1])

    def pulse_feature_dim_for_group(self, path_index: int, group: h5py.Group) -> int:
        indices = self._pulse_feature_indices[path_index] if path_index < len(self._pulse_feature_indices) else None
        if indices is not None:
            return max(int(indices.shape[0]) - 1, 0)
        if "pulse_features" not in group:
            return 0
        return max(int(group["pulse_features"].shape[1]) - 1, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_size > 0 and index in self._cache:
            sample = self._cache.pop(index)
            self._cache[index] = sample
            return sample

        path_index, local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        target = group["target"][()].astype(np.float32) if "target" in group else None
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{key}")
        particle_label = self.particle_label(index) if self.load_particle_label else None
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no proton/iron label: {self.paths[path_index]}::{key}")
        node_features = self._node_features_from_group(path_index, group)
        sample: dict[str, Any] = {
            "node_features": node_features,
            "edge_index": group["edge_index"][()].astype(np.int64),
            "edge_features": group["edge_features"][()].astype(np.float32),
            "pulse_features": self._pulse_features_from_group(path_index, group),
            "waveform_features": group["waveform_features"][()].astype(np.float32)
            if "waveform_features" in group
            else np.zeros(
                (
                    int(node_features.shape[0]),
                    0,
                    0,
                ),
                dtype=np.float32,
            ),
            "target": target,
            "attrs": dict(group.attrs.items()) if self.load_attrs else {},
        }
        if self.load_detector_lids:
            sample["detector_lids"] = self._group_detector_lids(group, sample["node_features"].shape[0])
        if self.load_particle_label:
            sample["particle_label"] = particle_label
        if self.load_node_positions:
            sample["node_positions_km"] = group["node_positions_km"][()].astype(np.float32)
        if self.load_attrs:
            sample["event_id"] = str(sample["attrs"].get("event_id", key))
        else:
            event_id = self._metadata_value(path_index, local_index, "event_id")
            sample["event_id"] = str(event_id) if event_id is not None else key
        if self.cache_size > 0:
            self._cache[index] = sample
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return sample


def fit_scalers(
    dataset: H5GraphDataset,
    indices: list[int],
    show_progress: bool = False,
    workers: int = 0,
) -> dict[str, StandardScaler]:
    if not indices:
        raise ValueError("cannot fit scalers with no training indices")

    node_dim, edge_dim, pulse_dim, target_dim = _scaler_feature_dimensions(dataset, indices[0])
    node_stats = RunningFeatureStats(node_dim)
    edge_stats = RunningFeatureStats(edge_dim)
    pulse_stats = RunningFeatureStats(pulse_dim) if pulse_dim > 0 else None
    target_stats = RunningFeatureStats(target_dim)
    if int(workers) > 1 and len(indices) >= 1024:
        return _fit_scalers_parallel(dataset, indices, show_progress=show_progress, workers=int(workers))

    for index in _progress(indices, desc="fit scalers", total=len(indices), enabled=show_progress):
        _update_scaler_stats(dataset, index, node_stats, edge_stats, pulse_stats, target_stats)

    if target_stats.count == 0:
        raise ValueError("training graphs have no MC targets")
    scalers = {
        "node": node_stats.to_scaler(),
        "edge": edge_stats.to_scaler(),
        "target": target_stats.to_scaler(),
    }
    if pulse_stats is not None:
        scalers["pulse"] = pulse_stats.to_scaler()
    return scalers


def _scaler_feature_dimensions(dataset: H5GraphDataset, index: int) -> tuple[int, int, int, int]:
    path_index, _local_index, key = dataset._locate(index)
    group = dataset._handle(path_index)["events"][key]
    node_dim = dataset.node_feature_dim_for_group(path_index, group)
    edge_dim = int(group["edge_features"].shape[1])
    pulse_dim = dataset.pulse_feature_dim_for_group(path_index, group)
    target_dim = int(group["target"].shape[0]) if "target" in group else 0
    return node_dim, edge_dim, pulse_dim, target_dim


def _update_scaler_stats(
    dataset: H5GraphDataset,
    index: int,
    node_stats: RunningFeatureStats,
    edge_stats: RunningFeatureStats,
    pulse_stats: RunningFeatureStats | None,
    target_stats: RunningFeatureStats,
) -> None:
    path_index, _local_index, key = dataset._locate(index)
    group = dataset._handle(path_index)["events"][key]
    node_stats.update(dataset._node_features_from_group(path_index, group))
    edge_features = group["edge_features"][()].astype(np.float32)
    if edge_features.shape[0] > 0:
        edge_stats.update(edge_features)
    if pulse_stats is not None and "pulse_features" in group:
        pulse_features = dataset._pulse_features_from_group(path_index, group)
        if pulse_features.shape[0] > 0 and pulse_features.shape[1] > 1:
            pulse_stats.update(pulse_features[:, 1:])
    if "target" in group:
        target_stats.update(group["target"][()].astype(np.float32))


def _fit_scaler_chunk(
    payload: tuple[H5GraphDataset, list[int]],
) -> tuple[int, dict[str, RunningFeatureStats | None]]:
    dataset, indices = payload
    try:
        node_dim, edge_dim, pulse_dim, target_dim = _scaler_feature_dimensions(dataset, indices[0])
        node_stats = RunningFeatureStats(node_dim)
        edge_stats = RunningFeatureStats(edge_dim)
        pulse_stats = RunningFeatureStats(pulse_dim) if pulse_dim > 0 else None
        target_stats = RunningFeatureStats(target_dim)
        for index in indices:
            _update_scaler_stats(dataset, index, node_stats, edge_stats, pulse_stats, target_stats)
        return (
            len(indices),
            {
                "node": node_stats,
                "edge": edge_stats,
                "pulse": pulse_stats,
                "target": target_stats,
            },
        )
    finally:
        dataset.close()


def _fit_scalers_parallel(
    dataset: H5GraphDataset,
    indices: list[int],
    *,
    show_progress: bool,
    workers: int,
) -> dict[str, StandardScaler]:
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    node_dim, edge_dim, pulse_dim, target_dim = _scaler_feature_dimensions(dataset, indices[0])
    node_stats = RunningFeatureStats(node_dim)
    edge_stats = RunningFeatureStats(edge_dim)
    pulse_stats = RunningFeatureStats(pulse_dim) if pulse_dim > 0 else None
    target_stats = RunningFeatureStats(target_dim)

    worker_count = min(max(int(workers), 1), len(indices))
    chunk_size = max(1024, int(np.ceil(len(indices) / max(worker_count * 4, 1))))
    chunks = [indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size)]
    progress = _progress_bar("fit scalers", len(indices), enabled=show_progress)

    pending = set()
    chunk_iter = iter(chunks)
    max_pending = max(worker_count * 2, 1)
    pool = ProcessPoolExecutor(max_workers=worker_count)
    pool_closed = False

    def submit_next() -> bool:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return False
        pending.add(pool.submit(_fit_scaler_chunk, (dataset, chunk)))
        return True

    try:
        for _ in range(min(max_pending, len(chunks))):
            submit_next()
        while pending:
            done, pending = wait(pending, timeout=_progress_interval_seconds(), return_when=FIRST_COMPLETED)
            if not done:
                progress.update(0)
                continue
            for future in done:
                count, stats = future.result()
                node_stats.merge(stats["node"])  # type: ignore[arg-type]
                edge_stats.merge(stats["edge"])  # type: ignore[arg-type]
                if pulse_stats is not None and stats["pulse"] is not None:
                    pulse_stats.merge(stats["pulse"])  # type: ignore[arg-type]
                target_stats.merge(stats["target"])  # type: ignore[arg-type]
                progress.update(count)
                submit_next()
        pool.shutdown(wait=True)
        pool_closed = True
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        pool_closed = True
        raise
    finally:
        if not pool_closed:
            pool.shutdown(wait=False, cancel_futures=True)
        progress.close()

    if target_stats.count == 0:
        raise ValueError("training graphs have no MC targets")
    scalers = {
        "node": node_stats.to_scaler(),
        "edge": edge_stats.to_scaler(),
        "target": target_stats.to_scaler(),
    }
    if pulse_stats is not None:
        scalers["pulse"] = pulse_stats.to_scaler()
    return scalers


def _collate_graph_arrays_python(
    samples: list[dict[str, Any]],
    scalers: dict[str, StandardScaler] | None = None,
    require_target: bool = True,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("empty batch")

    scalers = scalers or {}
    node_scaler = scalers.get("node")
    edge_scaler = scalers.get("edge")
    pulse_scaler = scalers.get("pulse")
    target_scaler = scalers.get("target")

    node_arrays = []
    edge_arrays = []
    edge_indices = []
    pulse_arrays = []
    pulse_node_indices = []
    waveform_arrays = []
    detector_lid_arrays = []
    batch_index = []
    targets = []
    particle_labels = []
    event_ids = []
    attrs = []
    node_offset = 0
    pulse_dim = max(samples[0]["pulse_features"].shape[1] - 1, 0)

    for graph_index, sample in enumerate(samples):
        node = sample["node_features"]
        edge = sample["edge_features"]
        if node_scaler is not None:
            node = node_scaler.transform(node)
        if edge_scaler is not None and edge.shape[0] > 0:
            edge = edge_scaler.transform(edge)

        node_arrays.append(node)
        waveform = sample.get("waveform_features")
        if waveform is None:
            waveform = np.zeros((node.shape[0], 0, 0), dtype=np.float32)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 3 or waveform.shape[0] != node.shape[0]:
            waveform = np.zeros((node.shape[0], 0, 0), dtype=np.float32)
        waveform_arrays.append(waveform)
        detector_lids = sample.get("detector_lids")
        if detector_lids is None:
            detector_lids = np.full(node.shape[0], -1, dtype=np.int64)
        else:
            detector_lids = np.asarray(detector_lids, dtype=np.int64).reshape(-1)
            if detector_lids.shape[0] != node.shape[0]:
                out_lids = np.full(node.shape[0], -1, dtype=np.int64)
                n_copy = min(node.shape[0], int(detector_lids.shape[0]))
                if n_copy > 0:
                    out_lids[:n_copy] = detector_lids[:n_copy]
                detector_lids = out_lids
        detector_lid_arrays.append(detector_lids)
        edge_arrays.append(edge)
        edge_index = sample["edge_index"].copy()
        if edge_index.size > 0:
            edge_index = edge_index + node_offset
        edge_indices.append(edge_index)
        batch_index.append(np.full(node.shape[0], graph_index, dtype=np.int64))

        pulses = sample.get("pulse_features")
        if pulses is not None and pulses.shape[0] > 0 and pulses.shape[1] > 1:
            local_node_index = pulses[:, 0].astype(np.int64)
            valid = (local_node_index >= 0) & (local_node_index < node.shape[0])
            if np.any(valid):
                pulse_x = pulses[valid, 1:].astype(np.float32)
                if pulse_scaler is not None:
                    pulse_x = pulse_scaler.transform(pulse_x)
                pulse_arrays.append(pulse_x)
                pulse_node_indices.append(local_node_index[valid] + node_offset)

        node_offset += node.shape[0]

        target = sample["target"]
        if require_target and target is None:
            raise ValueError(f"sample has no target: {sample['event_id']}")
        if target is not None:
            if target_scaler is not None:
                target = target_scaler.transform(target[None, :])[0]
            targets.append(target)
        if "particle_label" in sample:
            particle_label = sample.get("particle_label")
            particle_labels.append(float("nan") if particle_label is None else float(particle_label))
        event_ids.append(sample["event_id"])
        attrs.append(sample["attrs"])

    edge_index_np = (
        np.concatenate(edge_indices, axis=1)
        if edge_indices and any(edge.shape[1] > 0 for edge in edge_indices)
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_attr_np = (
        np.concatenate(edge_arrays, axis=0)
        if edge_arrays and any(edge.shape[0] > 0 for edge in edge_arrays)
        else np.zeros((0, samples[0]["edge_features"].shape[1]), dtype=np.float32)
    )
    pulse_x_np = (
        np.concatenate(pulse_arrays, axis=0)
        if pulse_arrays and any(pulse.shape[0] > 0 for pulse in pulse_arrays)
        else np.zeros((0, pulse_dim), dtype=np.float32)
    )
    pulse_node_index_np = (
        np.concatenate(pulse_node_indices, axis=0)
        if pulse_node_indices and any(index.shape[0] > 0 for index in pulse_node_indices)
        else np.zeros((0,), dtype=np.int64)
    )

    batch = {
        "x": np.concatenate(node_arrays, axis=0).astype(np.float32, copy=False),
        "edge_index": edge_index_np.astype(np.int64, copy=False),
        "edge_attr": edge_attr_np.astype(np.float32, copy=False),
        "pulse_x": pulse_x_np.astype(np.float32, copy=False),
        "pulse_node_index": pulse_node_index_np.astype(np.int64, copy=False),
        "waveform_x": np.concatenate(waveform_arrays, axis=0).astype(np.float32, copy=False)
        if waveform_arrays and any(waveform.shape[1] > 0 and waveform.shape[2] > 0 for waveform in waveform_arrays)
        else np.zeros((np.concatenate(node_arrays, axis=0).shape[0], 0, 0), dtype=np.float32),
        "detector_lids": np.concatenate(detector_lid_arrays, axis=0).astype(np.int64, copy=False),
        "batch": np.concatenate(batch_index, axis=0).astype(np.int64, copy=False),
        "num_graphs": len(samples),
        "event_ids": event_ids,
        "attrs": attrs,
    }
    if targets:
        batch["y"] = np.stack(targets, axis=0).astype(np.float32, copy=False)
    if particle_labels:
        batch["mass_label"] = np.asarray(particle_labels, dtype=np.float32)
    return batch


def _scaler_arrays(
    scalers: dict[str, StandardScaler],
    name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    scaler = scalers.get(name)
    if scaler is None:
        return None, None
    return scaler.mean, scaler.std


def _collate_graph_arrays_cpp(
    samples: list[dict[str, Any]],
    scalers: dict[str, StandardScaler] | None = None,
    require_target: bool = True,
    num_threads: int = 0,
) -> dict[str, Any]:
    from . import _collate_ext

    scalers = scalers or {}
    node_mean, node_std = _scaler_arrays(scalers, "node")
    edge_mean, edge_std = _scaler_arrays(scalers, "edge")
    pulse_mean, pulse_std = _scaler_arrays(scalers, "pulse")
    target_mean, target_std = _scaler_arrays(scalers, "target")
    batch = dict(
        _collate_ext.collate_numeric(
            samples,
            node_mean,
            node_std,
            edge_mean,
            edge_std,
            pulse_mean,
            pulse_std,
            target_mean,
            target_std,
            require_target,
            int(num_threads),
        )
    )
    batch.pop("collate_threads", None)
    batch["num_graphs"] = len(samples)
    batch["event_ids"] = [sample["event_id"] for sample in samples]
    batch["attrs"] = [sample["attrs"] for sample in samples]
    return batch


def collate_graph_arrays(
    samples: list[dict[str, Any]],
    scalers: dict[str, StandardScaler] | None = None,
    require_target: bool = True,
    backend: str = "cpp",
    num_threads: int = 0,
) -> dict[str, Any]:
    if backend == "cpp":
        return _collate_graph_arrays_cpp(
            samples,
            scalers=scalers,
            require_target=require_target,
            num_threads=num_threads,
        )
    if backend == "python":
        return _collate_graph_arrays_python(samples, scalers=scalers, require_target=require_target)
    raise ValueError(f"unsupported collate backend: {backend}")


def collate_graphs(
    samples: list[dict[str, Any]],
    scalers: dict[str, StandardScaler] | None = None,
    device: str = "cpu",
    require_target: bool = True,
    backend: str = "cpp",
    num_threads: int = 0,
) -> dict[str, Any]:
    import torch

    arrays = collate_graph_arrays(
        samples,
        scalers=scalers,
        require_target=require_target,
        backend=backend,
        num_threads=num_threads,
    )
    tensor_keys = {"x", "edge_index", "edge_attr", "edge_dst_degree", "pulse_x", "pulse_node_index", "waveform_x", "detector_lids", "batch", "y"}
    batch: dict[str, Any] = {}
    for key, value in arrays.items():
        if key in tensor_keys and isinstance(value, np.ndarray):
            batch[key] = torch.as_tensor(value, device=device)
        else:
            batch[key] = value
    return batch
