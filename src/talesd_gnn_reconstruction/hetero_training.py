from __future__ import annotations

import os
import random
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import RunningFeatureStats, StandardScaler
from .diagnostics import save_training_diagnostics
from .hetero_data import EDGE_TYPE_BY_RELATION
from .hetero_data import hetero_sample_to_tensors
from .hetero_graph_io import H5HeteroGraphDataset, H5PyGHeteroGraphDataset, hetero_dataset_class_for_paths
from .hetero_model import MinimalHeteroTaleSdGNN
from .hetero_model import NODE_TYPE_BY_RELATION
from .metrics import binary_classification_metrics, reconstruction_metrics
from .progress import progress as _progress
from .train import (
    _error_prediction_loss,
    _loader_worker_init,
    _mass_classification_loss,
    _physical_error_predictions,
    _quality_prediction_loss,
    _reconstruction_training_loss,
    _split_model_output,
    _target_scaler_tensors,
    resolve_device,
    split_indices,
    split_indices_by_source_path,
    split_indices_by_stratified_source_path,
)


def _finite_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[0] == 0:
        return values
    return values[np.all(np.isfinite(values), axis=1)]


def fit_hetero_scalers(dataset: H5HeteroGraphDataset, indices: Sequence[int]) -> dict[str, StandardScaler]:
    if not indices:
        raise ValueError("cannot fit hetero scalers with no training indices")
    first = dataset.scaler_sample(int(indices[0]))
    detector_stats = RunningFeatureStats(int(first["detector_features"].shape[1]))
    detector_context_stats = RunningFeatureStats(int(first["detector_context_features"].shape[1]))
    pulse_stats = RunningFeatureStats(int(first["pulse_features"].shape[1]))
    target_stats = RunningFeatureStats(int(first["target"].shape[0]) if first["target"] is not None else 0)
    edge_stats = {
        relation: RunningFeatureStats(
            int(first["edge_features_by_type"][relation].shape[1])
            if first["edge_features_by_type"][relation].ndim == 2
            else 0
        )
        for relation in EDGE_TYPE_BY_RELATION
    }
    for index in indices:
        sample = dataset.scaler_sample(int(index))
        detector_stats.update(_finite_rows(sample["detector_features"]))
        detector_context_stats.update(_finite_rows(sample["detector_context_features"]))
        pulse_stats.update(_finite_rows(sample["pulse_features"]))
        if sample["target"] is not None:
            target_stats.update(_finite_rows(sample["target"]))
        for relation, features in sample["edge_features_by_type"].items():
            if relation in edge_stats and features.shape[0] > 0 and features.shape[1] > 0:
                edge_stats[relation].update(_finite_rows(features))
    if target_stats.count == 0:
        raise ValueError("training graphs have no MC targets")
    scalers = {
        "detector": detector_stats.to_scaler(),
        "detector_context": detector_context_stats.to_scaler(),
        "pulse": pulse_stats.to_scaler(),
        "target": target_stats.to_scaler(),
    }
    for relation, stats in edge_stats.items():
        if stats.mean.shape[0] > 0:
            scalers[f"edge:{relation}"] = stats.to_scaler()
    return scalers


def _resolve_waveform_shape(
    dataset: H5HeteroGraphDataset,
    indices: Sequence[int],
    *,
    waveform_length: int | None = None,
) -> tuple[int, int]:
    if not indices:
        raise ValueError("cannot resolve waveform shape with no training indices")
    requested_length = None if waveform_length is None else int(waveform_length)
    if requested_length is not None and requested_length <= 0:
        raise ValueError("waveform_length must be positive")
    waveform_channels: int | None = None
    max_length = 0
    for index in indices:
        waveform_shape = dataset.detector_waveform_shape(int(index))
        if len(waveform_shape) != 3:
            raise ValueError(
                "detector_waveforms must be 3D [detector, channel, time], "
                f"got shape={waveform_shape} at graph index {index}"
            )
        channels = int(waveform_shape[1])
        if waveform_channels is None:
            waveform_channels = channels
        elif waveform_channels != channels:
            raise ValueError(
                f"detector waveform channel mismatch: expected {waveform_channels}, "
                f"got {channels} at graph index {index}"
            )
        max_length = max(max_length, int(waveform_shape[2]))
    resolved_length = requested_length if requested_length is not None else max_length
    if waveform_channels is None or resolved_length <= 0:
        raise ValueError("training graphs have no detector waveform samples")
    return waveform_channels, resolved_length


def _scalers_to_dict(scalers: dict[str, StandardScaler]) -> dict[str, Any]:
    return {name: scaler.to_dict() for name, scaler in scalers.items()}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _split_dataset(
    dataset: H5HeteroGraphDataset,
    *,
    split_mode: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    source_val_fraction: float,
    source_test_fraction: float,
    show_progress: bool,
    split_workers: int,
) -> dict[str, list[int]]:
    if split_mode == "event":
        return split_indices(len(dataset), val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    if split_mode == "source-path":
        return split_indices_by_source_path(
            dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            show_progress=show_progress,
        )
    if split_mode == "source-stratified":
        return split_indices_by_stratified_source_path(
            dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            show_progress=show_progress,
            workers=max(int(split_workers), 0),
            source_val_fraction=source_val_fraction,
            source_test_fraction=source_test_fraction,
        )
    raise ValueError("split_mode must be 'event', 'source-path', or 'source-stratified'")


def _limit_split_for_debug(
    split: dict[str, list[int]],
    *,
    max_graphs: int | None,
    seed: int,
) -> dict[str, list[int]]:
    if max_graphs is None or int(max_graphs) <= 0:
        return split
    requested = max(int(max_graphs), 1)
    before = {name: len(values) for name, values in split.items()}
    total_before = sum(before.values())
    if total_before <= requested:
        print(
            "hetero_max_graphs "
            f"active=0 requested={requested} n_total_before={total_before} n_total_after={total_before}",
            flush=True,
        )
        return split
    rng = random.Random(int(seed) + 1000003)
    limited: dict[str, list[int]] = {}
    assigned = 0
    names = ["train", "val", "test"]
    nonempty_names = [name for name in names if split.get(name)]
    keep_one_per_nonempty_split = requested >= len(nonempty_names)
    for name in names:
        values = list(split.get(name, []))
        if not values:
            limited[name] = []
            continue
        quota = int(round(requested * (len(values) / float(total_before))))
        if keep_one_per_nonempty_split:
            quota = max(quota, 1)
        elif name == "train":
            quota = max(quota, 1)
        quota = min(max(quota, 0), len(values))
        limited[name] = sorted(rng.sample(values, quota)) if quota < len(values) else values
        assigned += len(limited[name])
    remaining = requested - assigned
    if remaining > 0:
        for name in names:
            if remaining <= 0:
                break
            selected = set(limited.get(name, []))
            candidates = [index for index in split.get(name, []) if index not in selected]
            take = min(remaining, len(candidates))
            if take > 0:
                limited[name].extend(rng.sample(candidates, take))
                limited[name].sort()
                remaining -= take
    total_after = sum(len(values) for values in limited.values())
    print(
        "hetero_max_graphs "
        f"active=1 requested={requested} n_total_before={total_before} n_total_after={total_after} "
        f"train_before={before.get('train', 0)} val_before={before.get('val', 0)} test_before={before.get('test', 0)} "
        f"train_after={len(limited.get('train', []))} val_after={len(limited.get('val', []))} "
        f"test_after={len(limited.get('test', []))} "
        "scope=debug_after_split_source_groups_preserved_across_splits",
        flush=True,
    )
    return limited


def _memory_bytes_from_slurm_env() -> int | None:
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node:
        try:
            return int(mem_per_node) * 1024 * 1024
        except ValueError:
            pass
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if mem_per_cpu and cpus_per_task:
        try:
            return int(mem_per_cpu) * int(cpus_per_task) * 1024 * 1024
        except ValueError:
            pass
    return None


def _cpu_worker_limit() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(int(slurm_cpus), 0)
        except ValueError:
            pass
    return max(os.cpu_count() or 1, 0)


def _sample_indices(indices: Sequence[int], *, max_samples: int) -> list[int]:
    if not indices or max_samples <= 0:
        return []
    if len(indices) <= max_samples:
        return [int(index) for index in indices]
    positions = np.linspace(0, len(indices) - 1, num=max_samples, dtype=np.int64)
    return [int(indices[int(position)]) for position in positions]


def _estimate_graph_bytes(
    dataset: H5HeteroGraphDataset,
    indices: Sequence[int],
    *,
    max_samples: int,
) -> dict[str, Any]:
    sampled = _sample_indices(indices, max_samples=max_samples)
    values = np.asarray([dataset.graph_nbytes(index) for index in sampled], dtype=np.float64)
    if values.size == 0:
        return {
            "sampled_graphs": 0,
            "mean_graph_bytes": 0,
            "p95_graph_bytes": 0,
            "max_graph_bytes": 0,
        }
    return {
        "sampled_graphs": int(values.size),
        "mean_graph_bytes": int(np.mean(values)),
        "p95_graph_bytes": int(np.percentile(values, 95)),
        "max_graph_bytes": int(np.max(values)),
    }


def _resolve_loader_settings(
    *,
    requested_workers: int,
    batch_size: int,
    prefetch_factor: int,
    pin_memory: bool,
    loader_memory_budget_gib: float | None,
    graph_byte_summary: dict[str, Any],
) -> dict[str, Any]:
    cpu_limit = _cpu_worker_limit()
    if requested_workers < 0:
        requested = cpu_limit
    else:
        requested = min(max(int(requested_workers), 0), cpu_limit)
    batch_size = max(int(batch_size), 1)
    prefetch_factor = max(int(prefetch_factor), 1)
    budget_bytes = None
    if loader_memory_budget_gib is not None and float(loader_memory_budget_gib) > 0:
        budget_bytes = int(float(loader_memory_budget_gib) * (1024**3))
    else:
        budget_bytes = _memory_bytes_from_slurm_env()
    p95_graph_bytes = max(int(graph_byte_summary.get("p95_graph_bytes", 0)), 1)
    pinned_copy_batches = 1 if pin_memory else 0
    memory_limited_workers = requested
    if budget_bytes is not None and budget_bytes > 0:
        per_batch_bytes = batch_size * p95_graph_bytes
        fixed_batches = 1 + pinned_copy_batches
        available_batches = (budget_bytes // max(per_batch_bytes, 1)) - fixed_batches
        memory_limited_workers = max(int(available_batches // prefetch_factor), 0)
    resolved_workers = min(requested, memory_limited_workers)
    held_batches = 1 + resolved_workers * prefetch_factor + pinned_copy_batches
    estimated_loader_bytes = held_batches * batch_size * p95_graph_bytes
    return {
        "requested_workers": int(requested_workers),
        "cpu_worker_limit": int(cpu_limit),
        "resolved_workers": int(resolved_workers),
        "prefetch_factor": int(prefetch_factor),
        "pin_memory": bool(pin_memory),
        "loader_memory_budget_bytes": None if budget_bytes is None else int(budget_bytes),
        "estimated_loader_bytes": int(estimated_loader_bytes),
        "held_batches_estimate": int(held_batches),
    }


class H5TensorHeteroGraphDataset:
    """Training-oriented hetero dataset that skips PyG object construction."""

    def __init__(
        self,
        *args: Any,
        scalers: dict[str, Any] | None = None,
        waveform_length: int | None = None,
        **kwargs: Any,
    ):
        dataset_class = hetero_dataset_class_for_paths(args[0])
        self.base = dataset_class(*args, load_attrs=False, **kwargs)
        self.scalers = scalers
        self.waveform_length = None if waveform_length is None else int(waveform_length)

    def __len__(self) -> int:
        return len(self.base)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "base": self.base.__getstate__(),
            "base_class": self.base.__class__.__name__,
            "scalers": self.scalers,
            "waveform_length": self.waveform_length,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        from .hetero_graph_io import H5FlatHeteroGraphDataset

        base_class = H5FlatHeteroGraphDataset if state.get("base_class") == "H5FlatHeteroGraphDataset" else H5HeteroGraphDataset
        self.base = base_class.__new__(base_class)
        self.base.__dict__.update(state["base"])
        self.scalers = state.get("scalers")
        self.waveform_length = state.get("waveform_length")

    def close(self) -> None:
        self.base.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        tensors = hetero_sample_to_tensors(
            self.base[int(index)],
            scalers=self.scalers,
            waveform_length=self.waveform_length,
        )
        # Keep only tensors used by training forward/loss. Diagnostics and
        # attention maps continue to use the PyG path with full metadata.
        return {
            "detector": {
                "x": tensors["detector"]["x"],
                "context": tensors["detector"]["context"],
                "waveform": tensors["detector"]["waveform"],
                "waveform_valid": tensors["detector"]["waveform_valid"],
            },
            "pulse": {
                "x": tensors["pulse"]["x"],
            },
            "edge_index_by_type": tensors["edge_index_by_type"],
            "edge_features_by_type": tensors["edge_features_by_type"],
            "target": tensors["target"],
            "particle_label": tensors["particle_label"],
            "num_graphs": 1,
        }


def _empty_edge_features(samples: Sequence[dict[str, Any]], relation: str) -> torch.Tensor:
    import torch

    for sample in samples:
        features = sample["edge_features_by_type"].get(relation)
        if features is not None:
            return torch.zeros((0, int(features.shape[1])), dtype=torch.float32)
    return torch.zeros((0, 0), dtype=torch.float32)


def _collate_tensor_hetero_graphs(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not samples:
        raise ValueError("cannot collate an empty hetero graph batch")
    detector_x_rows = []
    detector_context_rows = []
    detector_waveform_rows = []
    detector_valid_rows = []
    detector_batch_rows = []
    pulse_x_rows = []
    pulse_batch_rows = []
    target_rows = []
    label_rows = []
    edge_index_by_type = {relation: [] for relation in EDGE_TYPE_BY_RELATION}
    edge_features_by_type = {relation: [] for relation in EDGE_TYPE_BY_RELATION}
    node_offsets = {"detector": 0, "pulse": 0}
    for graph_index, sample in enumerate(samples):
        detector = sample["detector"]
        pulse = sample["pulse"]
        n_detector = int(detector["x"].shape[0])
        n_pulse = int(pulse["x"].shape[0])
        detector_x_rows.append(detector["x"])
        detector_context_rows.append(detector["context"])
        detector_waveform_rows.append(detector["waveform"])
        detector_valid_rows.append(detector["waveform_valid"].reshape(-1))
        detector_batch_rows.append(torch.full((n_detector,), int(graph_index), dtype=torch.long))
        pulse_x_rows.append(pulse["x"])
        pulse_batch_rows.append(torch.full((n_pulse,), int(graph_index), dtype=torch.long))
        if sample["target"] is not None:
            target_rows.append(sample["target"].reshape(1, -1))
        if sample["particle_label"] is not None:
            label_rows.append(sample["particle_label"].reshape(-1))
        for relation in EDGE_TYPE_BY_RELATION:
            edge_index = sample["edge_index_by_type"].get(relation)
            edge_features = sample["edge_features_by_type"].get(relation)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_type, dst_type = NODE_TYPE_BY_RELATION[relation]
            offset = torch.tensor(
                [[node_offsets[src_type]], [node_offsets[dst_type]]],
                dtype=torch.long,
            )
            edge_index_by_type[relation].append(edge_index.to(dtype=torch.long) + offset)
            edge_features_by_type[relation].append(edge_features.to(dtype=torch.float32))
        node_offsets["detector"] += n_detector
        node_offsets["pulse"] += n_pulse

    collated_edges = {}
    collated_edge_features = {}
    for relation in EDGE_TYPE_BY_RELATION:
        if edge_index_by_type[relation]:
            collated_edges[relation] = torch.cat(edge_index_by_type[relation], dim=1)
            collated_edge_features[relation] = torch.cat(edge_features_by_type[relation], dim=0)
        else:
            collated_edges[relation] = torch.zeros((2, 0), dtype=torch.long)
            collated_edge_features[relation] = _empty_edge_features(samples, relation)
    return {
        "detector": {
            "x": torch.cat(detector_x_rows, dim=0),
            "context": torch.cat(detector_context_rows, dim=0),
            "waveform": torch.cat(detector_waveform_rows, dim=0),
            "waveform_valid": torch.cat(detector_valid_rows, dim=0),
            "batch": torch.cat(detector_batch_rows, dim=0),
        },
        "pulse": {
            "x": torch.cat(pulse_x_rows, dim=0),
            "batch": torch.cat(pulse_batch_rows, dim=0),
        },
        "edge_index_by_type": collated_edges,
        "edge_features_by_type": collated_edge_features,
        "target": torch.cat(target_rows, dim=0) if target_rows else None,
        "particle_label": torch.cat(label_rows, dim=0) if label_rows else None,
        "num_graphs": int(len(samples)),
    }


def _batch_to_device(batch: Any, device: str, *, non_blocking: bool = False) -> Any:
    import torch

    if hasattr(batch, "to"):
        try:
            return batch.to(device, non_blocking=bool(non_blocking))
        except TypeError:
            return batch.to(device)
    if isinstance(batch, dict):
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device, non_blocking=bool(non_blocking))
            elif isinstance(value, dict):
                moved[key] = {
                    sub_key: sub_value.to(device, non_blocking=bool(non_blocking))
                    if isinstance(sub_value, torch.Tensor)
                    else sub_value
                    for sub_key, sub_value in value.items()
                }
            else:
                moved[key] = value
        return moved
    raise TypeError(f"unsupported hetero batch type: {type(batch).__name__}")


def _batch_tensor(batch: Any, name: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None) if hasattr(batch, name) else batch[name]


def _parse_relation_filter(value: str | None) -> set[str]:
    if value is None:
        value = os.environ.get("HETERO_RELATIONS", "all")
    text = str(value).strip()
    if not text or text.lower() == "all":
        return set(EDGE_TYPE_BY_RELATION)
    relations = {item.strip() for item in text.split(",") if item.strip()}
    unknown = sorted(relations.difference(EDGE_TYPE_BY_RELATION))
    if unknown:
        raise ValueError(f"unknown HETERO_RELATIONS entries: {unknown}")
    return relations


def _max_neighbors_by_relation() -> dict[str, int]:
    mapping = {
        "pulse__near_space__pulse": int(os.environ.get("PULSE_NEAR_SPACE_MAX_NEIGHBORS", "0") or 0),
        "pulse__time_causal__pulse": int(os.environ.get("PULSE_TIME_CAUSAL_MAX_NEIGHBORS", "0") or 0),
    }
    return {relation: value for relation, value in mapping.items() if value > 0}


def _empty_edge_like(edge_features: Any, edge_index: Any) -> tuple[Any, Any]:
    import torch

    empty_index = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
    feature_dim = int(edge_features.shape[1]) if edge_features is not None and edge_features.ndim == 2 else 0
    empty_features = torch.zeros((0, feature_dim), dtype=torch.float32, device=edge_index.device)
    if edge_features is not None:
        empty_features = empty_features.to(dtype=edge_features.dtype, device=edge_features.device)
    return empty_index, empty_features


def _cap_relation_neighbors(edge_index: Any, edge_features: Any, max_neighbors: int) -> tuple[Any, Any]:
    import torch

    if max_neighbors <= 0 or edge_index.numel() == 0:
        return edge_index, edge_features
    dst = edge_index[1].detach().cpu().numpy()
    keep = np.zeros(edge_index.shape[1], dtype=bool)
    counts: dict[int, int] = {}
    for edge_offset, dst_index in enumerate(dst.tolist()):
        count = counts.get(int(dst_index), 0)
        if count < max_neighbors:
            keep[edge_offset] = True
            counts[int(dst_index)] = count + 1
    keep_tensor = torch.as_tensor(keep, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, keep_tensor], edge_features[keep_tensor]


def _filter_batch_relations(
    batch: Any,
    *,
    enabled_relations: set[str],
    max_neighbors: dict[str, int],
) -> Any:
    if enabled_relations == set(EDGE_TYPE_BY_RELATION) and not max_neighbors:
        return batch
    if isinstance(batch, dict):
        for relation in EDGE_TYPE_BY_RELATION:
            edge_index = batch["edge_index_by_type"][relation]
            edge_features = batch["edge_features_by_type"][relation]
            if relation not in enabled_relations:
                empty_index, empty_features = _empty_edge_like(edge_features, edge_index)
                batch["edge_index_by_type"][relation] = empty_index
                batch["edge_features_by_type"][relation] = empty_features
            elif relation in max_neighbors:
                capped_index, capped_features = _cap_relation_neighbors(
                    edge_index,
                    edge_features,
                    int(max_neighbors[relation]),
                )
                batch["edge_index_by_type"][relation] = capped_index
                batch["edge_features_by_type"][relation] = capped_features
        return batch
    for relation, edge_type in EDGE_TYPE_BY_RELATION.items():
        edge_store = batch[edge_type]
        edge_index = edge_store.edge_index
        edge_features = edge_store.edge_attr
        if relation not in enabled_relations:
            empty_index, empty_features = _empty_edge_like(edge_features, edge_index)
            edge_store.edge_index = empty_index
            edge_store.edge_attr = empty_features
        elif relation in max_neighbors:
            edge_store.edge_index, edge_store.edge_attr = _cap_relation_neighbors(
                edge_index,
                edge_features,
                int(max_neighbors[relation]),
            )
    return batch


def _log_relation_stats(dataset: H5HeteroGraphDataset, indices: Sequence[int], *, max_samples: int = 512) -> None:
    sampled = _sample_indices(indices, max_samples=max_samples)
    if not sampled:
        return
    counts = {relation: [] for relation in EDGE_TYPE_BY_RELATION}
    for index in sampled:
        sample = dataset.scaler_sample(int(index))
        for relation, features in sample["edge_features_by_type"].items():
            if relation in counts:
                counts[relation].append(int(features.shape[0]))
    for relation in EDGE_TYPE_BY_RELATION:
        values = np.asarray(counts[relation], dtype=np.float64)
        if values.size == 0:
            continue
        print(
            "hetero_relation_stats "
            f"split=train relation={relation} "
            f"sampled_graphs={int(values.size)} "
            f"mean_edges={float(np.mean(values)):.6g} "
            f"p95_edges={float(np.percentile(values, 95)):.6g} "
            f"max_edges={int(np.max(values))}",
            flush=True,
        )


def _configure_torch_sharing_strategy(num_workers: int) -> str:
    if int(num_workers) <= 0:
        return "single-process"
    import torch.multiprocessing as torch_mp

    requested = (
        os.environ.get("TALESD_GNN_TORCH_SHARING_STRATEGY")
        or os.environ.get("TORCH_SHARING_STRATEGY")
        or "file_system"
    )
    requested = str(requested).strip()
    if not requested or requested == "default":
        return str(torch_mp.get_sharing_strategy())
    available = set(torch_mp.get_all_sharing_strategies())
    if requested not in available:
        raise ValueError(
            f"torch sharing strategy {requested!r} is not available; "
            f"available={sorted(available)}"
        )
    torch_mp.set_sharing_strategy(requested)
    return str(torch_mp.get_sharing_strategy())


def _open_fd_count() -> int:
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        return -1
    try:
        return len(list(fd_dir.iterdir()))
    except OSError:
        return -1


def _nofile_limit() -> tuple[int, int]:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return (-1, -1)
    return int(soft), int(hard)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _raise_nofile_limit(*, context: str, target: int | None = None) -> tuple[int, int, int, int]:
    before_soft, before_hard = _nofile_limit()
    after_soft, after_hard = before_soft, before_hard
    if not _env_flag("TALESD_GNN_NOFILE_RAISE", True):
        print(
            "hetero_nofile_limit "
            f"context={context} before_soft={before_soft} before_hard={before_hard} "
            f"after_soft={after_soft} after_hard={after_hard} raised=0 reason=disabled",
            flush=True,
        )
        return before_soft, before_hard, after_soft, after_hard
    try:
        import resource

        requested = (
            int(os.environ.get("TALESD_GNN_NOFILE_TARGET", "65535"))
            if target is None
            else int(target)
        )
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(max(int(requested), int(soft)), int(hard))
        if int(soft) < int(new_soft):
            resource.setrlimit(resource.RLIMIT_NOFILE, (int(new_soft), int(hard)))
        after_soft, after_hard = _nofile_limit()
        print(
            "hetero_nofile_limit "
            f"context={context} before_soft={before_soft} before_hard={before_hard} "
            f"after_soft={after_soft} after_hard={after_hard} raised={int(after_soft > before_soft)}",
            flush=True,
        )
    except Exception as exc:
        after_soft, after_hard = _nofile_limit()
        print(
            "hetero_nofile_limit "
            f"context={context} before_soft={before_soft} before_hard={before_hard} "
            f"after_soft={after_soft} after_hard={after_hard} raised=0 error={type(exc).__name__}:{exc}",
            flush=True,
        )
    return before_soft, before_hard, after_soft, after_hard


def _hetero_loader_worker_init(worker_id: int) -> None:
    _loader_worker_init(worker_id)
    _raise_nofile_limit(context=f"worker:{int(worker_id)}")
    strategy = _configure_torch_sharing_strategy(1)
    soft, hard = _nofile_limit()
    print(
        "hetero_loader_worker_init "
        f"worker_id={int(worker_id)} "
        f"torch_sharing_strategy={strategy} "
        f"fd_count={_open_fd_count()} "
        f"nofile_soft={soft} "
        f"nofile_hard={hard}",
        flush=True,
    )


def _make_hetero_loader(
    dataset: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    split_name: str,
    timeout_sec: float | None,
    data_format: str,
) -> Any:
    from torch.utils.data import Subset
    if data_format == "pyg":
        from torch_geometric.loader import DataLoader
        collate_fn = None
    elif data_format == "fast_tensor":
        from torch.utils.data import DataLoader
        collate_fn = _collate_tensor_hetero_graphs
    else:
        raise ValueError("HETERO_TRAINING_DATA_FORMAT must be fast_tensor or pyg")

    worker_count = min(max(int(num_workers), 0), max(len(indices), 1))
    context = os.environ.get("TALESD_GNN_DATALOADER_CONTEXT", "default").strip().lower()
    if context not in {"default", "fork", "forkserver", "spawn"}:
        raise ValueError("TALESD_GNN_DATALOADER_CONTEXT must be default, fork, forkserver, or spawn")
    effective_timeout = 0.0 if worker_count <= 0 else float(timeout_sec if timeout_sec is not None else 300.0)
    kwargs: dict[str, Any] = {
        "batch_size": max(int(batch_size), 1),
        "shuffle": bool(shuffle),
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "timeout": max(float(effective_timeout), 0.0),
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if worker_count > 0:
        if context != "default":
            kwargs["multiprocessing_context"] = context
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["worker_init_fn"] = _hetero_loader_worker_init
    print(
        "hetero_loader_config "
        f"split={split_name} "
        f"graphs={len(indices)} "
        f"batch_size={max(int(batch_size), 1)} "
        f"workers={worker_count} "
        f"persistent_workers={int(bool(persistent_workers) and worker_count > 0)} "
        f"prefetch_factor={max(int(prefetch_factor), 1) if worker_count > 0 else 0} "
        f"context={context} "
        f"timeout_sec={max(float(effective_timeout), 0.0):.6g} "
        f"pin_memory={int(bool(pin_memory))} "
        f"data_format={data_format}",
        flush=True,
    )
    return DataLoader(Subset(dataset, list(indices)), **kwargs)


def _next_loader_batch(
    iterator: Any,
    *,
    split_name: str,
    epoch: int,
    batch_index: int,
    total_batches: int,
    num_workers: int,
    persistent_workers: bool,
    timeout_sec: float,
    warn_sec: float,
) -> tuple[Any, float]:
    start = time.monotonic()
    try:
        batch = next(iterator)
    except Exception as exc:
        soft, hard = _nofile_limit()
        sharing_strategy = os.environ.get("TALESD_GNN_TORCH_SHARING_STRATEGY") or os.environ.get(
            "TORCH_SHARING_STRATEGY", "file_system"
        )
        context = os.environ.get("TALESD_GNN_DATALOADER_CONTEXT", "default")
        print(
            "hetero_dataloader_error "
            f"split={split_name} "
            f"epoch={int(epoch)} "
            f"batch={int(batch_index)}/{int(total_batches)} "
            f"num_workers={int(num_workers)} "
            f"persistent_workers={int(bool(persistent_workers))} "
            f"context={context} "
            f"timeout_sec={float(timeout_sec):.6g} "
            f"torch_sharing_strategy={sharing_strategy} "
            f"fd_count={_open_fd_count()} "
            f"nofile_soft={soft} "
            f"nofile_hard={hard} "
            f"error={type(exc).__name__}:{exc}",
            flush=True,
        )
        raise
    elapsed = time.monotonic() - start
    if warn_sec > 0.0 and elapsed >= warn_sec:
        print(
            "hetero_data_wait_warning "
            f"split={split_name} "
            f"epoch={int(epoch)} "
            f"batch={int(batch_index)}/{int(total_batches)} "
            f"data_wait_s={elapsed:.6g} "
            f"num_workers={int(num_workers)} "
            f"persistent_workers={int(bool(persistent_workers))}",
            flush=True,
        )
    return batch, elapsed


def _hetero_batch_loss(
    model: MinimalHeteroTaleSdGNN,
    batch: Any,
    *,
    target_dim: int,
    mass_classification: bool,
    scalers: dict[str, StandardScaler],
    device: str,
    loss_mode: str,
    energy_loss_weight: float,
    core_loss_weight: float,
    direction_loss_weight: float,
    core_loss_scale_km: float,
    angular_loss_scale_deg: float,
    energy_bias_loss_weight: float,
    energy_particle_bias_loss_weight: float,
    energy_bias_bin_width: float,
    energy_bias_min_bin_count: int,
    mass_loss_weight: float,
    mass_loss_mode: str,
    mass_focal_gamma: float,
    mass_ranking_weight: float,
    mass_ranking_margin: float,
    quality_prediction: bool,
    quality_loss_weight: float,
    quality_angular_scale_deg: float,
    quality_core_scale_km: float,
    quality_energy_scale: float,
    error_prediction: bool,
    error_loss_weight: float,
    error_angular_scale_deg: float,
    error_core_scale_km: float,
    error_energy_scale: float,
    nll_loss_weight: float,
    nll_sigma_energy_floor: float,
    nll_sigma_angle_floor_deg: float,
    nll_sigma_core_floor_km: float,
) -> tuple[Any, dict[str, Any]]:
    pred_all = model(batch)
    target_value = _batch_tensor(batch, "target")
    if target_value is None:
        raise ValueError("hetero training batch has no target")
    target = target_value.to(device=device, dtype=pred_all.dtype)
    pred_scaled, mass_logit, quality_logit, error_raw = _split_model_output(
        pred_all,
        target_dim,
        mass_classification,
        quality_prediction=quality_prediction,
        error_prediction=error_prediction,
    )
    target_mean, target_std = _target_scaler_tensors(scalers, device)
    labels = (
        _batch_tensor(batch, "particle_label").to(device=device, dtype=pred_all.dtype).reshape(-1)
        if mass_classification and _batch_tensor(batch, "particle_label") is not None
        else None
    )
    loss, components = _reconstruction_training_loss(
        pred_scaled,
        target,
        error_raw,
        labels,
        mode=loss_mode,
        target_mean=target_mean,
        target_std=target_std,
        energy_weight=energy_loss_weight,
        core_weight=core_loss_weight,
        direction_weight=direction_loss_weight,
        core_scale_km=core_loss_scale_km,
        angular_loss_scale_deg=angular_loss_scale_deg,
        nll_loss_weight=nll_loss_weight,
        error_angular_scale_deg=error_angular_scale_deg,
        error_core_scale_km=error_core_scale_km,
        error_energy_scale=error_energy_scale,
        nll_sigma_energy_floor=nll_sigma_energy_floor,
        nll_sigma_angle_floor_deg=nll_sigma_angle_floor_deg,
        nll_sigma_core_floor_km=nll_sigma_core_floor_km,
        energy_bias_loss_weight=energy_bias_loss_weight,
        energy_particle_bias_loss_weight=energy_particle_bias_loss_weight,
        energy_bias_bin_width=energy_bias_bin_width,
        energy_bias_min_bin_count=energy_bias_min_bin_count,
    )
    components["reconstruction"] = loss
    if quality_prediction and quality_logit is not None:
        quality_loss = _quality_prediction_loss(
            quality_logit,
            pred_scaled,
            target,
            target_mean=target_mean,
            target_std=target_std,
            angular_scale_deg=quality_angular_scale_deg,
            core_scale_km=quality_core_scale_km,
            energy_scale=quality_energy_scale,
        )
        loss = loss + float(quality_loss_weight) * quality_loss
        components["quality"] = quality_loss
    if error_prediction and error_raw is not None and float(error_loss_weight) > 0.0:
        error_loss = _error_prediction_loss(
            error_raw,
            pred_scaled,
            target,
            target_mean=target_mean,
            target_std=target_std,
            angular_scale_deg=error_angular_scale_deg,
            core_scale_km=error_core_scale_km,
            energy_scale=error_energy_scale,
        )
        loss = loss + float(error_loss_weight) * error_loss
        components["error"] = error_loss
    if mass_classification and mass_logit is not None and labels is not None:
        mass_loss = _mass_classification_loss(
            mass_logit,
            labels,
            mode=mass_loss_mode,
            pos_weight=None,
            focal_gamma=mass_focal_gamma,
            ranking_weight=mass_ranking_weight,
            ranking_margin=mass_ranking_margin,
        )
        loss = loss + float(mass_loss_weight) * mass_loss
        components["mass"] = mass_loss
    return loss, components


def _scale_gradients(model: MinimalHeteroTaleSdGNN, scale: float) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(float(scale))


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _predict_hetero_numpy(
    model: MinimalHeteroTaleSdGNN,
    loader: Any,
    scalers: dict[str, StandardScaler],
    device: str,
    *,
    target_dim: int,
    mass_classification: bool,
    quality_prediction: bool,
    error_prediction: bool,
    error_angular_scale_deg: float,
    error_core_scale_km: float,
    error_energy_scale: float,
    desc: str,
    show_progress: bool,
    enabled_relations: set[str] | None = None,
    max_neighbors: dict[str, int] | None = None,
    non_blocking: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    import torch

    model.eval()
    if len(loader) == 0:
        print(
            "hetero_split_warning "
            f"split={desc} graphs=0 metrics_skipped=1",
            flush=True,
        )
        empty_target = np.zeros((0, int(target_dim)), dtype=np.float32)
        return empty_target, empty_target, None, None, None, None
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mass_logit_rows: list[np.ndarray] = []
    mass_label_rows: list[np.ndarray] = []
    quality_score_rows: list[np.ndarray] = []
    error_prediction_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _progress(loader, desc=desc, total=len(loader), enabled=show_progress, leave=False):
            batch = _filter_batch_relations(
                batch,
                enabled_relations=set(EDGE_TYPE_BY_RELATION) if enabled_relations is None else enabled_relations,
                max_neighbors={} if max_neighbors is None else max_neighbors,
            )
            batch = _batch_to_device(batch, device, non_blocking=bool(non_blocking))
            pred_all = model(batch)
            pred_scaled, mass_logit, quality_logit, error_raw = _split_model_output(
                pred_all,
                target_dim,
                mass_classification,
                quality_prediction=quality_prediction,
                error_prediction=error_prediction,
            )
            pred_rows.append(scalers["target"].inverse_transform(pred_scaled.detach().cpu().numpy()))
            target_value = _batch_tensor(batch, "target")
            if target_value is None:
                raise ValueError("hetero prediction batch has no target")
            target_rows.append(scalers["target"].inverse_transform(target_value.detach().cpu().numpy()))
            label_value = _batch_tensor(batch, "particle_label")
            if mass_classification and mass_logit is not None and label_value is not None:
                mass_logit_rows.append(mass_logit.detach().cpu().numpy())
                mass_label_rows.append(label_value.detach().cpu().numpy())
            if quality_prediction and quality_logit is not None:
                quality_score_rows.append(torch.sigmoid(quality_logit).detach().cpu().numpy())
            if error_prediction and error_raw is not None:
                predicted_errors = _physical_error_predictions(
                    error_raw,
                    angular_scale_deg=error_angular_scale_deg,
                    core_scale_km=error_core_scale_km,
                    energy_scale=error_energy_scale,
                )
                error_prediction_rows.append(predicted_errors.detach().cpu().numpy())
    if not pred_rows or not target_rows:
        print(
            "hetero_split_warning "
            f"split={desc} graphs=0 metrics_skipped=1",
            flush=True,
        )
        empty_target = np.zeros((0, int(target_dim)), dtype=np.float32)
        return empty_target, empty_target, None, None, None, None
    return (
        np.concatenate(pred_rows, axis=0),
        np.concatenate(target_rows, axis=0),
        np.concatenate(mass_logit_rows, axis=0) if mass_logit_rows else None,
        np.concatenate(mass_label_rows, axis=0) if mass_label_rows else None,
        np.concatenate(quality_score_rows, axis=0) if quality_score_rows else None,
        np.concatenate(error_prediction_rows, axis=0) if error_prediction_rows else None,
    )


def _mean_tensor_value(value_sum: Any, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float((value_sum / float(count)).detach().cpu())


def _append_component_mean(row: dict[str, Any], prefix: str, component_sums: dict[str, Any], component_counts: dict[str, int]) -> None:
    for name, value_sum in component_sums.items():
        count = int(component_counts.get(name, 0))
        if count > 0:
            row[f"{prefix}_{name}_loss"] = _mean_tensor_value(value_sum, count)


def train_hetero_model(
    graphs_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.05,
    model_architecture: str = "hetero_attention",
    attention_heads: int = 4,
    readout_heads: int = 4,
    waveform_encoder: str = "cnn",
    waveform_embedding_dim: int = 64,
    waveform_length: int | None = None,
    waveform_transformer_heads: int = 4,
    waveform_transformer_layers: int = 1,
    waveform_transformer_max_tokens: int = 128,
    waveform_transformer_downsample: str = "adaptive_avg",
    loss_mode: str = "physics",
    energy_loss_weight: float = 1.0,
    core_loss_weight: float = 1.0,
    direction_loss_weight: float = 1.0,
    core_loss_scale_km: float = 0.05,
    angular_loss_scale_deg: float = 1.0,
    energy_bias_loss_weight: float = 0.0,
    energy_particle_bias_loss_weight: float = 0.0,
    energy_bias_bin_width: float = 0.1,
    energy_bias_min_bin_count: int = 8,
    mass_classification: bool = False,
    mass_loss_weight: float = 0.1,
    mass_loss_mode: str = "bce",
    mass_focal_gamma: float = 2.0,
    mass_ranking_weight: float = 0.0,
    mass_ranking_margin: float = 1.0,
    quality_prediction: bool = False,
    quality_loss_weight: float = 0.2,
    quality_angular_scale_deg: float = 1.0,
    quality_core_scale_km: float = 0.05,
    quality_energy_scale: float = 0.10,
    error_prediction: bool = False,
    error_loss_weight: float = 0.2,
    error_angular_scale_deg: float = 1.0,
    error_core_scale_km: float = 0.05,
    error_energy_scale: float = 0.10,
    nll_loss_weight: float = 0.2,
    nll_sigma_energy_floor: float = 0.01,
    nll_sigma_angle_floor_deg: float = 0.05,
    nll_sigma_core_floor_km: float = 0.005,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    source_val_fraction: float = 0.10,
    source_test_fraction: float = 0.20,
    split_mode: str = "event",
    seed: int = 12345,
    device: str = "auto",
    save_diagnostics: bool = False,
    diagnostic_energy_bin_width: float = 0.1,
    diagnostic_min_bin_count: int = 20,
    num_workers: int = -1,
    prefetch_factor: int = 2,
    persistent_workers: bool | None = None,
    val_num_workers: int | None = None,
    validate_every_n_epochs: int = 1,
    max_val_graphs: int | None = None,
    early_stopping_patience: int = 0,
    early_stopping_min_epochs: int = 1,
    checkpoint_milestones: Sequence[int] | None = None,
    checkpoint_milestone_full_eval: bool | None = None,
    allow_train_loss_checkpoint: bool | None = None,
    pin_memory: bool | None = None,
    loader_memory_budget_gib: float | None = None,
    loader_memory_estimate_samples: int = 512,
    split_workers: int = 0,
    amp: str = "off",
    profile: bool | None = None,
    max_graphs: int | None = None,
    training_data_format: str | None = None,
    final_eval_data_format: str | None = None,
    hetero_relations: str | None = None,
    dataloader_timeout_sec: float | None = None,
    data_wait_warn_sec: float | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)
    gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
    prefetch_factor = max(int(prefetch_factor), 1)
    if persistent_workers is None:
        persistent_workers = int(num_workers) != 0
    persistent_workers = bool(persistent_workers)
    val_num_workers = 0 if val_num_workers is None else int(val_num_workers)
    validate_every_n_epochs = int(validate_every_n_epochs)
    early_stopping_patience = max(int(early_stopping_patience), 0)
    early_stopping_min_epochs = max(int(early_stopping_min_epochs), 1)
    if checkpoint_milestones is None:
        milestone_epochs: tuple[int, ...] = ()
    else:
        milestone_epochs = tuple(sorted({int(epoch) for epoch in checkpoint_milestones if int(epoch) > 0}))
    if checkpoint_milestone_full_eval is None:
        checkpoint_milestone_full_eval = _env_flag("CHECKPOINT_MILESTONE_FULL_EVAL", False)
    checkpoint_milestone_full_eval = bool(checkpoint_milestone_full_eval)
    if allow_train_loss_checkpoint is None:
        allow_train_loss_checkpoint = _env_flag("ALLOW_TRAIN_LOSS_CHECKPOINT", False)
    allow_train_loss_checkpoint = bool(allow_train_loss_checkpoint)
    max_val_graphs = None if max_val_graphs is None or int(max_val_graphs) <= 0 else int(max_val_graphs)
    max_graphs = None if max_graphs is None or int(max_graphs) <= 0 else int(max_graphs)
    training_data_format = str(
        training_data_format or os.environ.get("HETERO_TRAINING_DATA_FORMAT", "fast_tensor")
    ).strip()
    if training_data_format not in {"fast_tensor", "pyg"}:
        raise ValueError("training_data_format must be fast_tensor or pyg")
    final_eval_data_format = str(
        final_eval_data_format or os.environ.get("HETERO_FINAL_EVAL_DATA_FORMAT", training_data_format)
    ).strip()
    if final_eval_data_format not in {"fast_tensor", "pyg"}:
        raise ValueError("final_eval_data_format must be fast_tensor or pyg")
    enabled_relations = _parse_relation_filter(hetero_relations)
    max_neighbors = _max_neighbors_by_relation()
    if dataloader_timeout_sec is None:
        dataloader_timeout_sec = float(os.environ.get("TALESD_GNN_DATALOADER_TIMEOUT_SEC", "300"))
    if data_wait_warn_sec is None:
        data_wait_warn_sec = float(os.environ.get("TALESD_GNN_DATA_WAIT_WARN_SEC", "30"))
    _raise_nofile_limit(context="main")
    pin_memory = device.startswith("cuda") if pin_memory is None else bool(pin_memory)
    loss_mode = str(loss_mode).lower()
    model_architecture = str(model_architecture)
    if model_architecture not in {"minimal_hetero", "hetero_attention"}:
        raise ValueError("model_architecture must be 'minimal_hetero' or 'hetero_attention'")
    waveform_transformer_downsample = str(waveform_transformer_downsample)
    if waveform_transformer_downsample not in {"adaptive_avg", "stride_conv"}:
        raise ValueError("waveform_transformer_downsample must be 'adaptive_avg' or 'stride_conv'")
    waveform_transformer_max_tokens = max(int(waveform_transformer_max_tokens), 1)
    amp_mode = str(amp).lower()
    if amp_mode not in {"off", "fp16", "bf16"}:
        raise ValueError("amp must be 'off', 'fp16', or 'bf16'")
    profile_enabled = (
        os.environ.get("TALESD_GNN_PROFILE", "0").strip().lower() not in {"", "0", "false", "no", "off"}
        if profile is None
        else bool(profile)
    )
    valid_loss_modes = {"scaled-mse", "weighted-scaled-mse", "hybrid-angle", "physics", "physics-nll", "nll"}
    if loss_mode not in valid_loss_modes:
        raise ValueError(
            "loss_mode must be 'scaled-mse', 'weighted-scaled-mse', 'hybrid-angle', "
            "'physics', 'physics-nll', or 'nll'"
        )
    if loss_mode in {"physics-nll", "nll"} and not error_prediction:
        error_prediction = True
        error_loss_weight = 0.0
    base_dataset_class = hetero_dataset_class_for_paths(graphs_path)
    base_dataset = base_dataset_class(
        graphs_path,
        require_target=True,
        require_particle_label=mass_classification,
    )
    if len(base_dataset) < 2:
        raise ValueError("hetero training needs at least two graphs with MC targets")
    split = _split_dataset(
        base_dataset,
        split_mode=split_mode,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        source_val_fraction=source_val_fraction,
        source_test_fraction=source_test_fraction,
        show_progress=show_progress,
        split_workers=split_workers,
    )
    split = _limit_split_for_debug(split, max_graphs=max_graphs, seed=seed)
    train_indices = split["train"]
    val_indices = split["val"]
    if max_val_graphs is not None and len(val_indices) > max_val_graphs:
        rng = random.Random(int(seed) + 2000003)
        val_indices_for_epoch = sorted(rng.sample(list(val_indices), int(max_val_graphs)))
    else:
        val_indices_for_epoch = list(val_indices)
    _log_relation_stats(base_dataset, train_indices, max_samples=max(int(loader_memory_estimate_samples), 1))
    graph_byte_summary = _estimate_graph_bytes(
        base_dataset,
        train_indices,
        max_samples=max(int(loader_memory_estimate_samples), 1),
    )
    loader_settings = _resolve_loader_settings(
        requested_workers=int(num_workers),
        batch_size=max(int(batch_size), 1),
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        loader_memory_budget_gib=loader_memory_budget_gib,
        graph_byte_summary=graph_byte_summary,
    )
    num_workers = int(loader_settings["resolved_workers"])
    torch_sharing_strategy = _configure_torch_sharing_strategy(num_workers)
    nofile_soft, nofile_hard = _nofile_limit()
    print(
        "hetero_loader_memory "
        f"sampled_graphs={graph_byte_summary['sampled_graphs']} "
        f"mean_graph_bytes={graph_byte_summary['mean_graph_bytes']} "
        f"p95_graph_bytes={graph_byte_summary['p95_graph_bytes']} "
        f"max_graph_bytes={graph_byte_summary['max_graph_bytes']} "
        f"batch_size={max(int(batch_size), 1)} "
        f"gradient_accumulation_steps={gradient_accumulation_steps} "
        f"effective_batch_size={max(int(batch_size), 1) * gradient_accumulation_steps} "
        f"requested_workers={loader_settings['requested_workers']} "
        f"resolved_workers={loader_settings['resolved_workers']} "
        f"cpu_worker_limit={loader_settings['cpu_worker_limit']} "
        f"prefetch_factor={loader_settings['prefetch_factor']} "
        f"pin_memory={int(loader_settings['pin_memory'])} "
        f"torch_sharing_strategy={torch_sharing_strategy} "
        f"main_fd_count={_open_fd_count()} "
        f"nofile_soft={nofile_soft} "
        f"nofile_hard={nofile_hard} "
        f"held_batches_estimate={loader_settings['held_batches_estimate']} "
        f"estimated_loader_bytes={loader_settings['estimated_loader_bytes']} "
        f"loader_memory_budget_bytes={loader_settings['loader_memory_budget_bytes']}"
    )
    print(
        "hetero_training_data_config "
        f"data_format={training_data_format} "
        f"enabled_relations={','.join(relation for relation in EDGE_TYPE_BY_RELATION if relation in enabled_relations)} "
        f"disabled_relations={','.join(relation for relation in EDGE_TYPE_BY_RELATION if relation not in enabled_relations)} "
        f"max_neighbors_json={json.dumps(max_neighbors, sort_keys=True)} "
        f"validate_every_n_epochs={int(validate_every_n_epochs)} "
        f"max_val_graphs={0 if max_val_graphs is None else int(max_val_graphs)} "
        f"early_stopping_patience={int(early_stopping_patience)} "
        f"early_stopping_min_epochs={int(early_stopping_min_epochs)} "
        f"checkpoint_milestones={','.join(str(epoch) for epoch in milestone_epochs)} "
        f"checkpoint_milestone_full_eval={int(checkpoint_milestone_full_eval)} "
        f"allow_train_loss_checkpoint={int(allow_train_loss_checkpoint)} "
        f"val_graphs_per_epoch={len(val_indices_for_epoch)} "
        f"val_num_workers={int(val_num_workers)} "
        f"final_eval_data_format={final_eval_data_format} "
        f"dataloader_timeout_sec={float(dataloader_timeout_sec):.6g} "
        f"data_wait_warn_sec={float(data_wait_warn_sec):.6g}",
        flush=True,
    )
    print(
        "hetero_checkpoint_milestones "
        f"enabled={int(bool(milestone_epochs))} "
        f"milestones={','.join(str(epoch) for epoch in milestone_epochs)} "
        f"full_eval={int(checkpoint_milestone_full_eval)}",
        flush=True,
    )
    _waveform_channels, resolved_waveform_length = _resolve_waveform_shape(
        base_dataset,
        train_indices,
        waveform_length=waveform_length,
    )
    scalers = fit_hetero_scalers(base_dataset, train_indices)
    first = base_dataset[train_indices[0]]
    target_dim = int(first["target"].shape[0])
    classification_dim = 1 if mass_classification else 0
    quality_dim = 1 if quality_prediction else 0
    error_dim = 3 if error_prediction else 0
    model = MinimalHeteroTaleSdGNN.from_sample(
        first,
        target_dim=target_dim,
        classification_dim=classification_dim,
        quality_dim=quality_dim,
        error_dim=error_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        waveform_encoder=waveform_encoder,
        waveform_embedding_dim=waveform_embedding_dim,
        waveform_length=resolved_waveform_length,
        waveform_transformer_heads=waveform_transformer_heads,
        waveform_transformer_layers=waveform_transformer_layers,
        waveform_transformer_max_tokens=waveform_transformer_max_tokens,
        waveform_transformer_downsample=waveform_transformer_downsample,
        architecture=model_architecture,
        attention_heads=attention_heads,
        readout_heads=readout_heads,
    ).to(device)
    amp_enabled = bool(device.startswith("cuda") and amp_mode != "off")
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=bool(amp_enabled and amp_dtype is torch.float16))
    print(
        "hetero_precision "
        f"amp={amp_mode} "
        f"enabled={int(amp_enabled)} "
        f"dtype={'none' if not amp_enabled else ('float16' if amp_dtype is torch.float16 else 'bfloat16')} "
        f"grad_scaler={int(grad_scaler.is_enabled())}",
        flush=True,
    )
    if str(waveform_encoder) == "transformer":
        print(
            "hetero_waveform_transformer_config "
            f"waveform_length={int(resolved_waveform_length)} "
            f"max_tokens={int(waveform_transformer_max_tokens)} "
            f"heads={int(waveform_transformer_heads)} "
            f"layers={int(waveform_transformer_layers)} "
            f"downsample={waveform_transformer_downsample}",
            flush=True,
        )
    base_dataset.close()

    dataset_class: Any = H5TensorHeteroGraphDataset if training_data_format == "fast_tensor" else H5PyGHeteroGraphDataset
    train_dataset = dataset_class(
        graphs_path,
        require_target=True,
        require_particle_label=mass_classification,
        scalers=scalers,
        waveform_length=resolved_waveform_length,
    )
    train_loader = _make_hetero_loader(
        train_dataset,
        train_indices,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        split_name="train",
        timeout_sec=dataloader_timeout_sec,
        data_format=training_data_format,
    )
    val_worker_count = min(max(int(val_num_workers), 0), max(len(val_indices_for_epoch), 1))
    val_loader = _make_hetero_loader(
        train_dataset,
        val_indices_for_epoch,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=val_worker_count,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and val_worker_count > 0),
        split_name="validation",
        timeout_sec=dataloader_timeout_sec,
        data_format=training_data_format,
    )
    eval_dataset_class: Any = H5TensorHeteroGraphDataset if final_eval_data_format == "fast_tensor" else H5PyGHeteroGraphDataset
    eval_dataset = eval_dataset_class(
        graphs_path,
        require_target=True,
        require_particle_label=mass_classification,
        scalers=scalers,
        waveform_length=resolved_waveform_length,
    )
    final_val_loader = _make_hetero_loader(
        eval_dataset,
        val_indices,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=0,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=False,
        split_name="validation_final",
        timeout_sec=0.0,
        data_format=final_eval_data_format,
    )
    test_loader = _make_hetero_loader(
        eval_dataset,
        split["test"],
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=0,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=False,
        split_name="test_final",
        timeout_sec=0.0,
        data_format=final_eval_data_format,
    )
    non_blocking_h2d = bool(pin_memory and str(device).startswith("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    train_progress_interval_sec = float(os.environ.get("TALESD_GNN_TRAIN_PROGRESS_INTERVAL_SEC", "60"))
    train_loader_batches = len(train_loader)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(f"{output}.metrics.json")

    def _split_payload() -> dict[str, Any]:
        return {
            "train_indices": np.asarray(train_indices, dtype=np.int64),
            "val_indices": np.asarray(val_indices, dtype=np.int64),
            "test_indices": np.asarray(split["test"], dtype=np.int64),
            "split_mode": split_mode,
            "n_train": int(len(train_indices)),
            "n_val": int(len(val_indices)),
            "n_test": int(len(split["test"])),
        }

    def _runtime_payload(
        *,
        checkpoint_epoch: int,
        completed: bool,
        checkpoint_kind: str,
        best_epoch: int,
        best_val_loss: float,
        best_checkpoint_score: float,
        best_checkpoint_metric: str,
        best_checkpoint_kind: str,
    ) -> dict[str, Any]:
        return {
            "graph_format": "hetero",
            "training_path": "hetero_smoke",
            "training_task": "reconstruction",
            "model_architecture": str(model_architecture),
            "loss_mode": str(loss_mode),
            "energy_loss_weight": float(energy_loss_weight),
            "core_loss_weight": float(core_loss_weight),
            "direction_loss_weight": float(direction_loss_weight),
            "core_loss_scale_km": float(core_loss_scale_km),
            "angular_loss_scale_deg": float(angular_loss_scale_deg),
            "energy_bias_loss_weight": float(energy_bias_loss_weight),
            "energy_particle_bias_loss_weight": float(energy_particle_bias_loss_weight),
            "energy_bias_bin_width": float(energy_bias_bin_width),
            "energy_bias_min_bin_count": int(energy_bias_min_bin_count),
            "epochs": int(epochs),
            "checkpoint_epoch": int(checkpoint_epoch),
            "best_epoch": int(best_epoch),
            "best_val_loss": None if not np.isfinite(best_val_loss) else float(best_val_loss),
            "best_checkpoint_score": None if not np.isfinite(best_checkpoint_score) else float(best_checkpoint_score),
            "best_checkpoint_metric": str(best_checkpoint_metric),
            "best_checkpoint_kind": str(best_checkpoint_kind),
            "completed": bool(completed),
            "checkpoint_kind": str(checkpoint_kind),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "hidden_dim": int(hidden_dim),
            "layers": int(num_layers),
            "dropout": float(dropout),
            "attention_heads": int(attention_heads),
            "readout_heads": int(readout_heads),
            "device": str(device),
            "mass_classification": bool(mass_classification),
            "mass_loss_mode": str(mass_loss_mode),
            "mass_loss_weight": float(mass_loss_weight),
            "mass_focal_gamma": float(mass_focal_gamma),
            "mass_ranking_weight": float(mass_ranking_weight),
            "mass_ranking_margin": float(mass_ranking_margin),
            "quality_prediction": bool(quality_prediction),
            "quality_loss_weight": float(quality_loss_weight),
            "quality_angular_scale_deg": float(quality_angular_scale_deg),
            "quality_core_scale_km": float(quality_core_scale_km),
            "quality_energy_scale": float(quality_energy_scale),
            "error_prediction": bool(error_prediction),
            "error_loss_weight": float(error_loss_weight),
            "error_angular_scale_deg": float(error_angular_scale_deg),
            "error_core_scale_km": float(error_core_scale_km),
            "error_energy_scale": float(error_energy_scale),
            "nll_loss_weight": float(nll_loss_weight),
            "nll_sigma_energy_floor": float(nll_sigma_energy_floor),
            "nll_sigma_angle_floor_deg": float(nll_sigma_angle_floor_deg),
            "nll_sigma_core_floor_km": float(nll_sigma_core_floor_km),
            "waveform_length": int(resolved_waveform_length),
            "waveform_transformer_heads": int(waveform_transformer_heads),
            "waveform_transformer_layers": int(waveform_transformer_layers),
            "waveform_transformer_max_tokens": int(waveform_transformer_max_tokens),
            "waveform_transformer_downsample": str(waveform_transformer_downsample),
            "amp": str(amp_mode),
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": "none" if not amp_enabled else ("float16" if amp_dtype is torch.float16 else "bfloat16"),
            "profile": bool(profile_enabled),
            "batch_size": int(max(int(batch_size), 1)),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(max(int(batch_size), 1) * gradient_accumulation_steps),
            "training_data_format": str(training_data_format),
            "enabled_relations": sorted(enabled_relations),
            "disabled_relations": sorted(set(EDGE_TYPE_BY_RELATION).difference(enabled_relations)),
            "max_graphs": None if max_graphs is None else int(max_graphs),
            "validate_every_n_epochs": int(validate_every_n_epochs),
            "max_val_graphs": None if max_val_graphs is None else int(max_val_graphs),
            "checkpoint_milestones": [int(epoch) for epoch in milestone_epochs],
            "checkpoint_milestone_full_eval": bool(checkpoint_milestone_full_eval),
            "allow_train_loss_checkpoint": bool(allow_train_loss_checkpoint),
            "final_eval_data_format": str(final_eval_data_format),
            "data_loader": {
                **loader_settings,
                **graph_byte_summary,
                "loader_memory_budget_gib": None
                if loader_memory_budget_gib is None
                else float(loader_memory_budget_gib),
                "loader_memory_estimate_samples": int(loader_memory_estimate_samples),
                "split_workers": int(split_workers),
                "training_data_format": str(training_data_format),
                "enabled_relations": sorted(enabled_relations),
                "disabled_relations": sorted(set(EDGE_TYPE_BY_RELATION).difference(enabled_relations)),
                "max_neighbors": dict(max_neighbors),
                "max_graphs": None if max_graphs is None else int(max_graphs),
                "validate_every_n_epochs": int(validate_every_n_epochs),
                "max_val_graphs": None if max_val_graphs is None else int(max_val_graphs),
                "early_stopping_patience": int(early_stopping_patience),
                "early_stopping_min_epochs": int(early_stopping_min_epochs),
                "checkpoint_milestones": [int(epoch) for epoch in milestone_epochs],
                "checkpoint_milestone_full_eval": bool(checkpoint_milestone_full_eval),
                "allow_train_loss_checkpoint": bool(allow_train_loss_checkpoint),
                "final_eval_data_format": str(final_eval_data_format),
                "val_num_workers": int(val_num_workers),
                "dataloader_timeout_sec": float(dataloader_timeout_sec),
                "data_wait_warn_sec": float(data_wait_warn_sec),
            },
        }

    def _save_checkpoint_and_metrics(
        *,
        checkpoint_epoch: int,
        completed: bool,
        checkpoint_kind: str,
        best_epoch: int,
        best_val_loss: float,
        best_checkpoint_score: float,
        best_checkpoint_metric: str,
        best_checkpoint_kind: str,
        metrics: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model.config,
            "hetero_scalers": _scalers_to_dict(scalers),
            "history": history,
            "metrics": {} if metrics is None else metrics,
            "diagnostics": {} if diagnostics is None else diagnostics,
            "split": _split_payload(),
            "runtime": _runtime_payload(
                checkpoint_epoch=checkpoint_epoch,
                completed=completed,
                checkpoint_kind=checkpoint_kind,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_checkpoint_score=best_checkpoint_score,
                best_checkpoint_metric=best_checkpoint_metric,
                best_checkpoint_kind=best_checkpoint_kind,
            ),
        }
        tmp_path = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, output)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        metrics_payload = {
            "checkpoint": str(output),
            "history": history,
            "metrics": checkpoint["metrics"],
            "diagnostics": checkpoint["diagnostics"],
            "split": {
                "split_mode": split_mode,
                "n_train": int(len(train_indices)),
                "n_val": int(len(val_indices)),
                "n_test": int(len(split["test"])),
            },
            "runtime": checkpoint["runtime"],
        }
        _write_json_atomic(metrics_path, metrics_payload)
        print(
            "hetero_checkpoint "
            f"stage={checkpoint_kind} "
            f"epoch={int(checkpoint_epoch)}/{int(epochs)} "
            f"best_epoch={int(best_epoch)} "
            f"best_val_loss={best_val_loss:.6g} "
            f"best_checkpoint_score={best_checkpoint_score:.6g} "
            f"best_checkpoint_metric={best_checkpoint_metric} "
            f"checkpoint={output} "
            f"metrics={metrics_path}",
            flush=True,
        )
        return checkpoint

    def _evaluate_current_model(
        *,
        desc_suffix: str,
    ) -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
        pred_val, target_val, mass_logit_val, mass_label_val, quality_val, error_val = _predict_hetero_numpy(
            model,
            final_val_loader,
            scalers,
            device,
            target_dim=target_dim,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            error_angular_scale_deg=error_angular_scale_deg,
            error_core_scale_km=error_core_scale_km,
            error_energy_scale=error_energy_scale,
            desc=f"hetero validation predict {desc_suffix}",
            show_progress=show_progress,
            enabled_relations=enabled_relations,
            max_neighbors=max_neighbors,
            non_blocking=non_blocking_h2d,
        )
        pred_test, target_test, mass_logit_test, mass_label_test, quality_test, error_test = _predict_hetero_numpy(
            model,
            test_loader,
            scalers,
            device,
            target_dim=target_dim,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            error_angular_scale_deg=error_angular_scale_deg,
            error_core_scale_km=error_core_scale_km,
            error_energy_scale=error_energy_scale,
            desc=f"hetero test predict {desc_suffix}",
            show_progress=show_progress,
            enabled_relations=enabled_relations,
            max_neighbors=max_neighbors,
            non_blocking=non_blocking_h2d,
        )

        def _add_reconstruction_metric(split_name: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any] | None:
            if int(pred.shape[0]) == 0 or int(target.shape[0]) == 0:
                print(
                    "hetero_split_warning "
                    f"split={split_name} graphs=0 metrics_skipped=1",
                    flush=True,
                )
                return None
            return reconstruction_metrics(pred, target)

        metrics: dict[str, Any] = {}
        val_metrics = _add_reconstruction_metric("validation", pred_val, target_val)
        test_metrics = _add_reconstruction_metric("test", pred_test, target_test)
        if val_metrics is not None:
            metrics["validation"] = val_metrics
        if test_metrics is not None:
            metrics["test"] = test_metrics
        if mass_classification and mass_logit_val is not None and mass_label_val is not None and mass_logit_val.shape[0] > 0:
            metrics["validation_mass"] = binary_classification_metrics(mass_logit_val, mass_label_val)
        if mass_classification and mass_logit_test is not None and mass_label_test is not None and mass_logit_test.shape[0] > 0:
            metrics["test_mass"] = binary_classification_metrics(mass_logit_test, mass_label_test)
        return (
            metrics,
            (pred_val, target_val, mass_logit_val, mass_label_val, quality_val, error_val),
            (pred_test, target_test, mass_logit_test, mass_label_test, quality_test, error_test),
        )

    def _save_milestone_checkpoint(epoch: int) -> None:
        if best_state is None or best_epoch <= 0:
            return
        metrics: dict[str, Any] = {}
        if checkpoint_milestone_full_eval:
            current_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
            metrics, _val_payload, _test_payload = _evaluate_current_model(desc_suffix=f"milestone_{int(epoch)}")
            model.load_state_dict({key: value.to(device) for key, value in current_state.items()})
        milestone_path = output.with_name(f"{output.stem}.best_through_epoch{int(epoch):04d}{output.suffix}")
        milestone_metrics_path = Path(f"{milestone_path}.metrics.json")
        checkpoint = {
            "model_state": best_state,
            "model_config": model.config,
            "hetero_scalers": _scalers_to_dict(scalers),
            "history": history,
            "metrics": metrics,
            "diagnostics": {},
            "split": _split_payload(),
            "runtime": {
                **_runtime_payload(
                    checkpoint_epoch=best_epoch,
                    completed=False,
                    checkpoint_kind=f"milestone_epoch_{int(epoch)}",
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    best_checkpoint_score=best_checkpoint_score,
                    best_checkpoint_metric=best_checkpoint_metric,
                    best_checkpoint_kind=best_checkpoint_kind,
                ),
                "milestone_epoch": int(epoch),
                "milestone_full_eval": bool(checkpoint_milestone_full_eval),
            },
        }
        tmp_path = milestone_path.with_name(f".{milestone_path.name}.tmp-{os.getpid()}")
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, milestone_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        _write_json_atomic(
            milestone_metrics_path,
            {
                "checkpoint": str(milestone_path),
                "history": history,
                "metrics": metrics,
                "diagnostics": {},
                "split": {
                    "split_mode": split_mode,
                    "n_train": int(len(train_indices)),
                    "n_val": int(len(val_indices)),
                    "n_test": int(len(split["test"])),
                },
                "runtime": checkpoint["runtime"],
            },
        )
        print(
            "hetero_checkpoint "
            f"stage=milestone epoch={int(epoch)}/{int(epochs)} "
            f"best_epoch={int(best_epoch)} "
            f"best_val_loss={best_val_loss:.6g} "
            f"best_checkpoint_score={best_checkpoint_score:.6g} "
            f"best_checkpoint_metric={best_checkpoint_metric} "
            f"full_eval={int(checkpoint_milestone_full_eval)} "
            f"checkpoint={milestone_path} "
            f"metrics={milestone_metrics_path}",
            flush=True,
        )

    best_val_loss = float("inf")
    best_checkpoint_score = float("inf")
    best_checkpoint_metric = "none"
    best_checkpoint_kind = "none"
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_best = 0

    def _profile_sync() -> None:
        if profile_enabled and device.startswith("cuda"):
            torch.cuda.synchronize()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_loss_count = 0
        train_component_sums: dict[str, torch.Tensor] = {}
        train_component_counts: dict[str, int] = {}
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation_steps = 0
        epoch_start = time.monotonic()
        last_progress = epoch_start
        profile_data_wait = 0.0
        profile_h2d = 0.0
        profile_forward = 0.0
        profile_backward = 0.0
        profile_optim = 0.0
        profile_valid_waveforms = 0
        profile_detector_nodes = 0
        if profile_enabled and device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        print(
            "hetero_train_epoch "
            f"stage=start epoch={epoch}/{int(epochs)} "
            f"train_batches={train_loader_batches} "
            f"train_graphs={len(train_indices)}"
        )
        train_iter = iter(train_loader)
        for batch_index in range(1, train_loader_batches + 1):
            batch, data_wait = _next_loader_batch(
                train_iter,
                split_name="train",
                epoch=epoch,
                batch_index=batch_index,
                total_batches=train_loader_batches,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                timeout_sec=float(dataloader_timeout_sec),
                warn_sec=float(data_wait_warn_sec),
            )
            if profile_enabled:
                profile_data_wait += data_wait
            h2d_start = time.perf_counter()
            batch = _filter_batch_relations(batch, enabled_relations=enabled_relations, max_neighbors=max_neighbors)
            batch = _batch_to_device(batch, device, non_blocking=non_blocking_h2d)
            _profile_sync()
            if profile_enabled:
                profile_h2d += time.perf_counter() - h2d_start
                detector_store = batch["detector"]
                valid = detector_store.get("waveform_valid") if isinstance(detector_store, dict) else getattr(
                    detector_store, "waveform_valid", None
                )
                if valid is not None:
                    profile_valid_waveforms += int((valid > 0.5).sum().detach().cpu())
                    profile_detector_nodes += int(valid.numel())
            if batch_index == 1 and str(waveform_encoder) == "transformer":
                detector_store = batch["detector"]
                valid = detector_store.get("waveform_valid") if isinstance(detector_store, dict) else getattr(
                    detector_store, "waveform_valid", None
                )
                valid_count = int((valid > 0.5).sum().detach().cpu()) if valid is not None else -1
                waveform_shape = tuple(int(value) for value in detector_store["waveform"].shape)
                tokens = min(int(waveform_transformer_max_tokens), int(waveform_shape[-1]) if waveform_shape else 0)
                print(
                    "hetero_waveform_encoder "
                    f"mode=transformer waveform_length={int(resolved_waveform_length)} "
                    f"input_waveform_shape={waveform_shape} "
                    f"transformer_tokens={tokens} "
                    f"valid_waveforms_first_batch={valid_count}",
                    flush=True,
                )
            forward_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                loss, components = _hetero_batch_loss(
                    model,
                    batch,
                    target_dim=target_dim,
                    mass_classification=mass_classification,
                    scalers=scalers,
                    device=device,
                    loss_mode=loss_mode,
                    energy_loss_weight=energy_loss_weight,
                    core_loss_weight=core_loss_weight,
                    direction_loss_weight=direction_loss_weight,
                    core_loss_scale_km=core_loss_scale_km,
                    angular_loss_scale_deg=angular_loss_scale_deg,
                    energy_bias_loss_weight=energy_bias_loss_weight,
                    energy_particle_bias_loss_weight=energy_particle_bias_loss_weight,
                    energy_bias_bin_width=energy_bias_bin_width,
                    energy_bias_min_bin_count=energy_bias_min_bin_count,
                    mass_loss_weight=mass_loss_weight,
                    mass_loss_mode=mass_loss_mode,
                    mass_focal_gamma=mass_focal_gamma,
                    mass_ranking_weight=mass_ranking_weight,
                    mass_ranking_margin=mass_ranking_margin,
                    quality_prediction=quality_prediction,
                    quality_loss_weight=quality_loss_weight,
                    quality_angular_scale_deg=quality_angular_scale_deg,
                    quality_core_scale_km=quality_core_scale_km,
                    quality_energy_scale=quality_energy_scale,
                    error_prediction=error_prediction,
                    error_loss_weight=error_loss_weight,
                    error_angular_scale_deg=error_angular_scale_deg,
                    error_core_scale_km=error_core_scale_km,
                    error_energy_scale=error_energy_scale,
                    nll_loss_weight=nll_loss_weight,
                    nll_sigma_energy_floor=nll_sigma_energy_floor,
                    nll_sigma_angle_floor_deg=nll_sigma_angle_floor_deg,
                    nll_sigma_core_floor_km=nll_sigma_core_floor_km,
                )
            _profile_sync()
            if profile_enabled:
                profile_forward += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            grad_scaler.scale(loss / float(gradient_accumulation_steps)).backward()
            _profile_sync()
            if profile_enabled:
                profile_backward += time.perf_counter() - backward_start
            pending_accumulation_steps += 1
            if pending_accumulation_steps >= gradient_accumulation_steps:
                optim_start = time.perf_counter()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _profile_sync()
                if profile_enabled:
                    profile_optim += time.perf_counter() - optim_start
                pending_accumulation_steps = 0
            train_loss_sum = train_loss_sum + loss.detach()
            train_loss_count += 1
            for name, value in components.items():
                detached = value.detach()
                if name not in train_component_sums:
                    train_component_sums[name] = torch.zeros((), dtype=detached.dtype, device=detached.device)
                    train_component_counts[name] = 0
                train_component_sums[name] = train_component_sums[name] + detached
                train_component_counts[name] += 1
            now = time.monotonic()
            if (
                train_progress_interval_sec > 0
                and (now - last_progress) >= train_progress_interval_sec
            ):
                elapsed = now - epoch_start
                rate = float(batch_index) / elapsed if elapsed > 0 else 0.0
                remaining = max(train_loader_batches - batch_index, 0)
                eta = float(remaining) / rate if rate > 0 else float("nan")
                print(
                    "hetero_train_progress "
                    f"epoch={epoch}/{int(epochs)} "
                    f"batch={batch_index}/{train_loader_batches} "
                    f"graphs={min(batch_index * max(int(batch_size), 1), len(train_indices))}/{len(train_indices)} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"rate={rate:.3g}/s "
                    f"eta={_format_duration(eta) if np.isfinite(eta) else 'unknown'} "
                    f"loss={_mean_tensor_value(train_loss_sum, train_loss_count):.6g}"
                )
                last_progress = now
        if pending_accumulation_steps > 0:
            optim_start = time.perf_counter()
            _scale_gradients(
                model,
                float(gradient_accumulation_steps) / float(pending_accumulation_steps),
            )
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            _profile_sync()
            if profile_enabled:
                profile_optim += time.perf_counter() - optim_start
        model.eval()
        val_start = time.perf_counter()
        val_loss_sum = torch.zeros((), device=device)
        val_loss_count = 0
        val_component_sums: dict[str, torch.Tensor] = {}
        val_component_counts: dict[str, int] = {}
        run_validation = bool(validate_every_n_epochs > 0 and epoch % int(validate_every_n_epochs) == 0)
        if run_validation:
            val_batches = len(val_loader)
            print(
                "hetero_validation_start "
                f"epoch={epoch}/{int(epochs)} val_batches={val_batches} val_graphs={len(val_indices_for_epoch)}",
                flush=True,
            )
            val_iter = iter(val_loader)
            with torch.no_grad():
                for val_batch_index in range(1, val_batches + 1):
                    batch, _data_wait = _next_loader_batch(
                        val_iter,
                        split_name="validation",
                        epoch=epoch,
                        batch_index=val_batch_index,
                        total_batches=val_batches,
                        num_workers=val_worker_count,
                        persistent_workers=bool(persistent_workers and val_worker_count > 0),
                        timeout_sec=float(dataloader_timeout_sec),
                        warn_sec=float(data_wait_warn_sec),
                    )
                    batch = _filter_batch_relations(
                        batch,
                        enabled_relations=enabled_relations,
                        max_neighbors=max_neighbors,
                    )
                    batch = _batch_to_device(batch, device, non_blocking=non_blocking_h2d)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        loss, components = _hetero_batch_loss(
                            model,
                            batch,
                            target_dim=target_dim,
                            mass_classification=mass_classification,
                            scalers=scalers,
                            device=device,
                            loss_mode=loss_mode,
                            energy_loss_weight=energy_loss_weight,
                            core_loss_weight=core_loss_weight,
                            direction_loss_weight=direction_loss_weight,
                            core_loss_scale_km=core_loss_scale_km,
                            angular_loss_scale_deg=angular_loss_scale_deg,
                            energy_bias_loss_weight=energy_bias_loss_weight,
                            energy_particle_bias_loss_weight=energy_particle_bias_loss_weight,
                            energy_bias_bin_width=energy_bias_bin_width,
                            energy_bias_min_bin_count=energy_bias_min_bin_count,
                            mass_loss_weight=mass_loss_weight,
                            mass_loss_mode=mass_loss_mode,
                            mass_focal_gamma=mass_focal_gamma,
                            mass_ranking_weight=mass_ranking_weight,
                            mass_ranking_margin=mass_ranking_margin,
                            quality_prediction=quality_prediction,
                            quality_loss_weight=quality_loss_weight,
                            quality_angular_scale_deg=quality_angular_scale_deg,
                            quality_core_scale_km=quality_core_scale_km,
                            quality_energy_scale=quality_energy_scale,
                            error_prediction=error_prediction,
                            error_loss_weight=error_loss_weight,
                            error_angular_scale_deg=error_angular_scale_deg,
                            error_core_scale_km=error_core_scale_km,
                            error_energy_scale=error_energy_scale,
                            nll_loss_weight=nll_loss_weight,
                            nll_sigma_energy_floor=nll_sigma_energy_floor,
                            nll_sigma_angle_floor_deg=nll_sigma_angle_floor_deg,
                            nll_sigma_core_floor_km=nll_sigma_core_floor_km,
                        )
                    val_loss_sum = val_loss_sum + loss.detach()
                    val_loss_count += 1
                    for name, value in components.items():
                        detached = value.detach()
                        if name not in val_component_sums:
                            val_component_sums[name] = torch.zeros((), dtype=detached.dtype, device=detached.device)
                            val_component_counts[name] = 0
                        val_component_sums[name] = val_component_sums[name] + detached
                        val_component_counts[name] += 1
        _profile_sync()
        validation_time = time.perf_counter() - val_start
        print(
            "hetero_validation_done "
            f"epoch={epoch}/{int(epochs)} ran={int(run_validation)} "
            f"elapsed={_format_duration(validation_time)} "
            f"val_batches={val_loss_count}",
            flush=True,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": _mean_tensor_value(train_loss_sum, train_loss_count),
            "val_loss": _mean_tensor_value(val_loss_sum, val_loss_count),
        }
        _append_component_mean(epoch_row, "train", train_component_sums, train_component_counts)
        _append_component_mean(epoch_row, "val", val_component_sums, val_component_counts)
        history.append(epoch_row)
        epoch_elapsed = time.monotonic() - epoch_start
        print(
            "hetero_train_epoch "
            f"stage=done epoch={epoch}/{int(epochs)} "
            f"elapsed={_format_duration(epoch_elapsed)} "
            f"train_loss={epoch_row['train_loss']:.6g} "
            f"val_loss={epoch_row['val_loss']:.6g}"
        )
        if profile_enabled:
            cuda_allocated_mb = float("nan")
            cuda_reserved_mb = float("nan")
            if device.startswith("cuda"):
                cuda_allocated_mb = torch.cuda.max_memory_allocated() / float(1024 * 1024)
                cuda_reserved_mb = torch.cuda.max_memory_reserved() / float(1024 * 1024)
            valid_fraction = (
                float(profile_valid_waveforms) / float(profile_detector_nodes)
                if profile_detector_nodes > 0
                else float("nan")
            )
            print(
                "hetero_epoch_profile "
                f"epoch={epoch}/{int(epochs)} "
                f"data_wait_s={profile_data_wait:.6g} "
                f"h2d_s={profile_h2d:.6g} "
                f"forward_s={profile_forward:.6g} "
                f"backward_s={profile_backward:.6g} "
                f"optim_s={profile_optim:.6g} "
                f"val_s={validation_time:.6g} "
                f"train_epoch_total_s={epoch_elapsed:.6g} "
                f"valid_waveforms={profile_valid_waveforms} "
                f"detector_waveforms={profile_detector_nodes} "
                f"valid_waveform_fraction={valid_fraction:.6g} "
                f"cuda_max_allocated_mb={cuda_allocated_mb:.6g} "
                f"cuda_max_reserved_mb={cuda_reserved_mb:.6g}",
                flush=True,
            )
        val_loss = float(epoch_row["val_loss"])
        train_loss = float(epoch_row["train_loss"])
        checkpoint_score = float("nan")
        checkpoint_metric = "none"
        checkpoint_kind = "best"
        if run_validation and np.isfinite(val_loss):
            checkpoint_score = val_loss
            checkpoint_metric = "validation_loss"
            checkpoint_kind = "best"
        elif allow_train_loss_checkpoint and np.isfinite(train_loss):
            checkpoint_score = train_loss
            checkpoint_metric = "train_loss_benchmark"
            checkpoint_kind = "train_loss_benchmark"
        else:
            reason = "validation_not_run" if not run_validation else "validation_loss_nonfinite"
            print(
                "hetero_checkpoint "
                f"stage=skip_no_validation epoch={int(epoch)}/{int(epochs)} "
                f"reason={reason} "
                f"allow_train_loss_checkpoint={int(allow_train_loss_checkpoint)} "
                f"best_epoch={int(best_epoch)} "
                f"best_val_loss={best_val_loss:.6g}",
                flush=True,
            )
        if np.isfinite(checkpoint_score) and (best_epoch == 0 or checkpoint_score < best_checkpoint_score):
            if checkpoint_metric == "validation_loss":
                best_val_loss = val_loss
            best_checkpoint_score = checkpoint_score
            best_checkpoint_metric = checkpoint_metric
            best_checkpoint_kind = checkpoint_kind
            best_epoch = int(epoch)
            epochs_since_best = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_checkpoint_and_metrics(
                checkpoint_epoch=epoch,
                completed=False,
                checkpoint_kind=checkpoint_kind,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_checkpoint_score=best_checkpoint_score,
                best_checkpoint_metric=best_checkpoint_metric,
                best_checkpoint_kind=best_checkpoint_kind,
            )
        elif np.isfinite(checkpoint_score):
            epochs_since_best += 1
            print(
                "hetero_checkpoint "
                f"stage=skip epoch={int(epoch)}/{int(epochs)} "
                f"best_epoch={int(best_epoch)} "
                f"best_val_loss={best_val_loss:.6g} "
                f"best_checkpoint_score={best_checkpoint_score:.6g} "
                f"checkpoint_score={checkpoint_score:.6g} "
                f"checkpoint_metric={checkpoint_metric} "
                f"val_loss={val_loss:.6g}",
                flush=True,
            )
        if int(epoch) in milestone_epochs:
            _save_milestone_checkpoint(int(epoch))
        if (
            early_stopping_patience > 0
            and int(epoch) >= early_stopping_min_epochs
            and epochs_since_best >= early_stopping_patience
        ):
            print(
                "hetero_early_stopping "
                f"epoch={int(epoch)}/{int(epochs)} "
                f"best_epoch={int(best_epoch)} "
                f"best_val_loss={best_val_loss:.6g} "
                f"epochs_since_best={int(epochs_since_best)} "
                f"patience={int(early_stopping_patience)}",
                flush=True,
            )
            break
    if best_state is None:
        best_epoch = int(history[-1]["epoch"]) if history else int(epochs)
        best_val_loss = float("nan")
        best_checkpoint_score = float("nan")
        best_checkpoint_metric = "none_no_validation"
        best_checkpoint_kind = "none_no_validation"
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            "hetero_checkpoint "
            f"stage=no_best_validation epoch={int(best_epoch)}/{int(epochs)} "
            "reason=no_validation_checkpoint_available use_current_state_for_final=1",
            flush=True,
        )
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    metrics, validation_payload, test_payload = _evaluate_current_model(desc_suffix="final")
    pred_val, target_val, mass_logit_val, mass_label_val, quality_val, error_val = validation_payload
    pred_test, target_test, mass_logit_test, mass_label_test, quality_test, error_test = test_payload
    diagnostics: dict[str, Any] = {}
    if save_diagnostics:
        diagnostics = save_training_diagnostics(
            output_path,
            history,
            validation=(pred_val, target_val),
            test=(pred_test, target_test),
            validation_mass=(mass_logit_val, mass_label_val)
            if mass_logit_val is not None and mass_label_val is not None
            else None,
            test_mass=(mass_logit_test, mass_label_test)
            if mass_logit_test is not None and mass_label_test is not None
            else None,
            validation_particle_labels=mass_label_val,
            test_particle_labels=mass_label_test,
            validation_quality=quality_val,
            test_quality=quality_test,
            validation_predicted_errors=error_val,
            test_predicted_errors=error_test,
            energy_bin_width=diagnostic_energy_bin_width,
            min_bin_count=diagnostic_min_bin_count,
            save_reconstruction=target_dim >= 6,
        )
    checkpoint = _save_checkpoint_and_metrics(
        checkpoint_epoch=best_epoch,
        completed=True,
        checkpoint_kind="final",
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_checkpoint_score=best_checkpoint_score,
        best_checkpoint_metric=best_checkpoint_metric,
        best_checkpoint_kind=best_checkpoint_kind,
        metrics=metrics,
        diagnostics=diagnostics,
    )
    train_dataset.close()
    eval_dataset.close()
    return {
        "checkpoint": str(output),
        "metrics_json": str(metrics_path),
        "history": history,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "split": checkpoint["split"],
    }
