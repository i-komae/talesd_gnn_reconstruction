from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np

from .constants import PULSE_FEATURE_COLUMNS


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


class H5GraphDataset:
    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        require_target: bool = False,
        cache_size: int = 0,
    ):
        self.paths = _as_paths(path)
        self.require_target = require_target
        self.cache_size = max(int(cache_size), 0)
        self._handles: dict[int, h5py.File] = {}
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.keys: list[tuple[int, str]] = []
        self.columns_json = "{}"

        for path_index, graph_path in enumerate(self.paths):
            with h5py.File(graph_path, "r") as handle:
                if path_index == 0:
                    self.columns_json = handle.attrs.get("columns_json", "{}")
                keys = sorted(handle["events"].keys())
                if require_target:
                    keys = [key for key in keys if "target" in handle["events"][key]]
                self.keys.extend((path_index, key) for key in keys)

    def __len__(self) -> int:
        return len(self.keys)

    def _handle(self, path_index: int) -> h5py.File:
        if path_index not in self._handles:
            self._handles[path_index] = h5py.File(self.paths[path_index], "r")
        return self._handles[path_index]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = {}
        self._cache.clear()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_size > 0 and index in self._cache:
            sample = self._cache.pop(index)
            self._cache[index] = sample
            return sample

        path_index, key = self.keys[index]
        group = self._handle(path_index)["events"][key]
        sample: dict[str, Any] = {
            "node_features": group["node_features"][()].astype(np.float32),
            "node_positions_km": group["node_positions_km"][()].astype(np.float32),
            "edge_index": group["edge_index"][()].astype(np.int64),
            "edge_features": group["edge_features"][()].astype(np.float32),
            "pulse_features": group["pulse_features"][()].astype(np.float32)
            if "pulse_features" in group
            else np.zeros((0, len(PULSE_FEATURE_COLUMNS)), dtype=np.float32),
            "target": group["target"][()].astype(np.float32) if "target" in group else None,
            "attrs": dict(group.attrs.items()),
        }
        sample["event_id"] = str(sample["attrs"].get("event_id", key))
        if self.cache_size > 0:
            self._cache[index] = sample
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return sample


def fit_scalers(dataset: H5GraphDataset, indices: list[int]) -> dict[str, StandardScaler]:
    if not indices:
        raise ValueError("cannot fit scalers with no training indices")

    first = dataset[indices[0]]
    node_stats = RunningFeatureStats(first["node_features"].shape[1])
    edge_stats = RunningFeatureStats(first["edge_features"].shape[1])
    pulse_dim = max(first["pulse_features"].shape[1] - 1, 0)
    pulse_stats = RunningFeatureStats(pulse_dim) if pulse_dim > 0 else None
    target_stats = RunningFeatureStats(first["target"].shape[0] if first["target"] is not None else 0)
    for index in indices:
        sample = dataset[index]
        node_stats.update(sample["node_features"])
        if sample["edge_features"].shape[0] > 0:
            edge_stats.update(sample["edge_features"])
        if pulse_stats is not None and sample["pulse_features"].shape[0] > 0:
            pulse_stats.update(sample["pulse_features"][:, 1:])
        if sample["target"] is not None:
            target_stats.update(sample["target"])

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


def collate_graphs(
    samples: list[dict[str, Any]],
    scalers: dict[str, StandardScaler] | None = None,
    device: str = "cpu",
    require_target: bool = True,
) -> dict[str, Any]:
    import torch

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
    batch_index = []
    targets = []
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
        "x": torch.as_tensor(np.concatenate(node_arrays, axis=0), dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(edge_index_np, dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(edge_attr_np, dtype=torch.float32, device=device),
        "pulse_x": torch.as_tensor(pulse_x_np, dtype=torch.float32, device=device),
        "pulse_node_index": torch.as_tensor(pulse_node_index_np, dtype=torch.long, device=device),
        "batch": torch.as_tensor(np.concatenate(batch_index, axis=0), dtype=torch.long, device=device),
        "num_graphs": len(samples),
        "event_ids": event_ids,
        "attrs": attrs,
    }
    if targets:
        batch["y"] = torch.as_tensor(np.stack(targets, axis=0), dtype=torch.float32, device=device)
    return batch
