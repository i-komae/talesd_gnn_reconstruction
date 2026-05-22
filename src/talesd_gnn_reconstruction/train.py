from __future__ import annotations

import json
import math
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections.abc import Sequence
from functools import partial
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .dataset import H5GraphDataset, StandardScaler, collate_graph_arrays, fit_scalers
from .diagnostics import require_matplotlib_latex, save_training_diagnostics
from .metrics import balanced_accuracy_threshold, binary_classification_metrics, direction_to_angles, reconstruction_metrics
from .progress import progress as _progress
from .progress import progress_bar as _progress_bar
from .progress import write as _progress_write

if TYPE_CHECKING:
    from .model import TaleSdGNN


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _batches(indices: list[int], batch_size: int, shuffle: bool) -> list[list[int]]:
    indices = list(indices)
    if shuffle:
        random.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def _collate_graph_batch(
    samples: list[dict[str, Any]],
    scalers: dict[str, StandardScaler],
    require_target: bool,
    backend: str,
    num_threads: int,
) -> dict[str, Any]:
    return collate_graph_arrays(
        samples,
        scalers=scalers,
        require_target=require_target,
        backend=backend,
        num_threads=num_threads,
    )


def _loader_worker_init(_worker_id: int) -> None:
    import torch

    torch.set_num_threads(1)


def _resolve_collate_backend(backend: str, *, n_graphs: int, num_workers: int) -> str:
    if backend == "auto":
        return "python" if n_graphs < 1024 and num_workers == 0 else "cpp"
    if backend in {"cpp", "python"}:
        return backend
    raise ValueError("collate_backend must be 'auto', 'cpp', or 'python'")


class LocalityBatchSampler:
    def __init__(self, indices: list[int], batch_size: int, shuffle_batches: bool, seed: int):
        self.indices = sorted(indices)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle_batches = shuffle_batches
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        indices = list(self.indices)
        if self.shuffle_batches:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return ceil(len(self.indices) / self.batch_size)


def _make_graph_loader(
    dataset: H5GraphDataset,
    indices: list[int],
    scalers: dict[str, StandardScaler],
    batch_size: int,
    shuffle: bool,
    require_target: bool,
    num_workers: int,
    prefetch_factor: int,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    collate_backend: str,
    collate_threads: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    worker_count = min(max(int(num_workers), 0), max(len(indices), 1))
    batch_sampler = LocalityBatchSampler(indices, batch_size=batch_size, shuffle_batches=shuffle, seed=seed)
    kwargs: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "num_workers": worker_count,
        "collate_fn": partial(
            _collate_graph_batch,
            scalers=scalers,
            require_target=require_target,
            backend=collate_backend,
            num_threads=collate_threads,
        ),
        "pin_memory": pin_memory,
    }
    if worker_count > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
        kwargs["worker_init_fn"] = _loader_worker_init
    return DataLoader(dataset, **kwargs)


def _batch_to_device(batch: dict[str, Any], device: str, non_blocking: bool = False) -> dict[str, Any]:
    import torch

    tensor_keys = {
        "x",
        "edge_index",
        "edge_attr",
        "edge_dst_degree",
        "pulse_x",
        "pulse_node_index",
        "waveform_x",
        "detector_lids",
        "batch",
        "y",
        "mass_label",
    }
    return {
        key: (
            value.to(device, non_blocking=non_blocking)
            if torch.is_tensor(value)
            else torch.as_tensor(value, device=device)
        )
        if key in tensor_keys
        else value
        for key, value in batch.items()
    }


def split_indices(
    n_items: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 12345,
) -> dict[str, list[int]]:
    if n_items <= 0:
        raise ValueError("no graphs available")
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if n_items < 3:
        return {"train": indices, "val": indices, "test": indices}

    n_test = max(1, int(round(n_items * test_fraction)))
    n_test = min(n_test, n_items - 2)
    remaining = n_items - n_test
    n_val = max(1, int(round(n_items * val_fraction)))
    n_val = min(n_val, remaining - 1)

    test_indices = indices[:n_test]
    val_indices = indices[n_test : n_test + n_val]
    train_indices = indices[n_test + n_val :]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def split_indices_by_source_path(
    dataset: H5GraphDataset,
    val_fraction: float = 0.1,
    test_fraction: float = 0.5,
    seed: int = 12345,
    show_progress: bool = True,
) -> dict[str, list[int]]:
    source_to_indices: dict[str, list[int]] = {}
    source_to_stratum: dict[str, str] = {}
    iterator = _progress(range(len(dataset)), desc="scan source paths", total=len(dataset), enabled=show_progress)
    for index in iterator:
        source_path = dataset.source_path(index) or f"unknown:{index}"
        source_to_indices.setdefault(source_path, []).append(index)
        source_to_stratum.setdefault(source_path, str(Path(source_path).parent))

    strata: dict[str, list[str]] = {}
    for source_path, stratum in source_to_stratum.items():
        strata.setdefault(stratum, []).append(source_path)

    split = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for stratum_sources in strata.values():
        sources = list(stratum_sources)
        rng.shuffle(sources)
        if len(sources) < 3:
            for source_path in sources:
                split["train"].extend(source_to_indices[source_path])
            continue
        n_test = max(1, int(round(len(sources) * test_fraction)))
        n_test = min(n_test, len(sources) - 2)
        remaining = len(sources) - n_test
        n_val = max(1, int(round(len(sources) * val_fraction)))
        n_val = min(n_val, remaining - 1)
        test_sources = sources[:n_test]
        val_sources = sources[n_test : n_test + n_val]
        train_sources = sources[n_test + n_val :]
        for source_path in train_sources:
            split["train"].extend(source_to_indices[source_path])
        for source_path in val_sources:
            split["val"].extend(source_to_indices[source_path])
        for source_path in test_sources:
            split["test"].extend(source_to_indices[source_path])

    for indices in split.values():
        indices.sort()
    if not split["train"] or not split["val"] or not split["test"]:
        raise ValueError("source-path split produced an empty train/validation/test split")
    return split


def _finite_bin(value: float, width: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return str(int(math.floor(value / max(float(width), 1.0e-12))))


def _source_stratification_keys(source_path: str, target: np.ndarray | None, particle_label: float | None) -> dict[str, tuple[str, ...]]:
    parent = str(Path(source_path).parent)
    particle = "unknown"
    if particle_label is not None and math.isfinite(float(particle_label)):
        particle = "iron" if float(particle_label) >= 0.5 else "proton"
    loge_bin = "nan"
    zenith_bin = "nan"
    if target is not None and target.shape[0] >= 7 and np.all(np.isfinite(target[[0, 4, 5, 6]])):
        loge_bin = _finite_bin(float(target[0]), 0.1)
        zenith, _azimuth = direction_to_angles(target[None, 4:7])
        zenith_bin = _finite_bin(float(zenith[0]), 10.0)
    return {
        "fine": (parent, particle, loge_bin, zenith_bin),
        "mid": (parent, particle, loge_bin),
        "coarse": (parent, particle),
        "broad": (parent,),
    }


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _stratified_source_scan_payloads(
    dataset: H5GraphDataset,
) -> list[tuple[str, int, int, list[int] | None, list[str] | None]]:
    payloads: list[tuple[str, int, int, list[int] | None, list[str] | None]] = []
    global_start = 0
    for path_index in range(len(dataset._path_lengths)):
        path = dataset.paths[path_index]
        n_events = int(dataset._path_lengths[path_index])
        if n_events <= 0:
            continue
        selected = dataset._path_local_indices[path_index]
        key_list = dataset._path_key_lists[path_index]
        payloads.append(
            (
                str(path),
                global_start,
                n_events,
                None if selected is None else list(selected[:n_events]),
                None if key_list is None else list(key_list),
            )
        )
        global_start += n_events
    return payloads


def _scan_stratified_source_shard(
    payload: tuple[str, int, int, list[int] | None, list[str] | None],
) -> tuple[int, dict[str, list[int]], dict[str, dict[str, Any]]]:
    path, global_start, n_events, selected_local_indices, key_list = payload
    source_to_indices: dict[str, list[int]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as h5:
        events = h5["events"]
        metadata = h5.get("metadata")
        source_values = None
        label_values = None
        if metadata is not None and "source_path" in metadata and len(metadata["source_path"]) > 0:
            source_values = metadata["source_path"][:]
        if metadata is not None and "particle_label" in metadata and len(metadata["particle_label"]) > 0:
            label_values = np.asarray(metadata["particle_label"][:], dtype=np.float64)

        for offset in range(n_events):
            local_index = (
                int(selected_local_indices[offset])
                if selected_local_indices is not None
                else offset
            )
            global_index = global_start + offset
            key = f"{local_index:08d}" if key_list is None else key_list[local_index]

            source_path = ""
            if source_values is not None and local_index < len(source_values):
                source_path = _decode_h5_string(source_values[local_index])
            group = events[key]
            if not source_path:
                source_path = str(group.attrs.get("source_path", ""))
            if not source_path:
                source_path = f"unknown:{global_index}"
            source_to_indices.setdefault(source_path, []).append(global_index)

            particle_label = None
            if label_values is not None and local_index < len(label_values):
                value = float(label_values[local_index])
                if np.isfinite(value):
                    particle_label = value
            if particle_label is None:
                particle_label = H5GraphDataset._group_particle_label(group)

            stats = source_stats.setdefault(
                source_path,
                {
                    "target_sum": np.zeros(7, dtype=np.float64),
                    "target_count": 0,
                    "particle_label": particle_label,
                },
            )
            if "target" in group:
                target = group["target"][()]
                if target.shape[0] >= 7 and np.all(np.isfinite(target[:7])):
                    stats["target_sum"] += target[:7].astype(np.float64)
                    stats["target_count"] += 1
            if stats["particle_label"] is None:
                stats["particle_label"] = particle_label
    return n_events, source_to_indices, source_stats


def _merge_stratified_source_scan(
    target_source_to_indices: dict[str, list[int]],
    target_source_stats: dict[str, dict[str, Any]],
    source_to_indices: dict[str, list[int]],
    source_stats: dict[str, dict[str, Any]],
) -> None:
    for source_path, indices in source_to_indices.items():
        target_source_to_indices.setdefault(source_path, []).extend(indices)
    for source_path, stats in source_stats.items():
        current = target_source_stats.setdefault(
            source_path,
            {
                "target_sum": np.zeros(7, dtype=np.float64),
                "target_count": 0,
                "particle_label": stats["particle_label"],
            },
        )
        current["target_sum"] += stats["target_sum"]
        current["target_count"] += int(stats["target_count"])
        if current["particle_label"] is None:
            current["particle_label"] = stats["particle_label"]


def _scan_stratified_source_paths_parallel(
    dataset: H5GraphDataset,
    *,
    show_progress: bool,
    workers: int,
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    payloads = _stratified_source_scan_payloads(dataset)
    worker_count = min(max(int(workers), 1), max(len(payloads), 1))
    source_to_indices: dict[str, list[int]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    progress = _progress_bar("scan stratified source paths", len(dataset), enabled=show_progress)
    pending = set()
    payload_iter = iter(payloads)
    max_pending = max(worker_count * 2, 1)
    pool = ProcessPoolExecutor(max_workers=worker_count)
    pool_closed = False

    def submit_next() -> bool:
        try:
            payload = next(payload_iter)
        except StopIteration:
            return False
        pending.add(pool.submit(_scan_stratified_source_shard, payload))
        return True

    try:
        for _ in range(min(max_pending, len(payloads))):
            submit_next()
        while pending:
            done, pending = wait(pending, timeout=progress.interval, return_when=FIRST_COMPLETED)
            if not done:
                progress.update(0)
                continue
            for future in done:
                count, shard_source_to_indices, shard_source_stats = future.result()
                _merge_stratified_source_scan(
                    source_to_indices,
                    source_stats,
                    shard_source_to_indices,
                    shard_source_stats,
                )
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
    return source_to_indices, source_stats


def _assign_source_group(
    split_sources: dict[str, list[str]],
    sources: list[str],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> None:
    sources = list(sources)
    rng.shuffle(sources)
    if len(sources) < 3:
        split_sources["train"].extend(sources)
        return
    n_test = int(round(len(sources) * test_fraction))
    n_val = int(round(len(sources) * val_fraction))
    if n_test + n_val >= len(sources):
        overflow = n_test + n_val - (len(sources) - 1)
        n_val = max(0, n_val - overflow)
        overflow = n_test + n_val - (len(sources) - 1)
        n_test = max(0, n_test - overflow)
    split_sources["test"].extend(sources[:n_test])
    split_sources["val"].extend(sources[n_test : n_test + n_val])
    split_sources["train"].extend(sources[n_test + n_val :])


def split_indices_by_stratified_source_path(
    dataset: H5GraphDataset,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 12345,
    show_progress: bool = True,
    min_group_sources: int = 10,
    workers: int = 0,
) -> dict[str, list[int]]:
    source_to_indices: dict[str, list[int]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    if int(workers) > 1 and len(dataset) >= 1024 and len(dataset._path_lengths) > 1:
        source_to_indices, source_stats = _scan_stratified_source_paths_parallel(
            dataset,
            show_progress=show_progress,
            workers=int(workers),
        )
    else:
        iterator = _progress(
            range(len(dataset)),
            desc="scan stratified source paths",
            total=len(dataset),
            enabled=show_progress,
        )
        for index in iterator:
            source_path = dataset.source_path(index) or f"unknown:{index}"
            source_to_indices.setdefault(source_path, []).append(index)
            stats = source_stats.setdefault(
                source_path,
                {
                    "target_sum": np.zeros(7, dtype=np.float64),
                    "target_count": 0,
                    "particle_label": dataset.particle_label(index),
                },
            )
            target = dataset.target(index)
            if target is not None and target.shape[0] >= 7 and np.all(np.isfinite(target[:7])):
                stats["target_sum"] += target[:7].astype(np.float64)
                stats["target_count"] += 1
            if stats["particle_label"] is None:
                stats["particle_label"] = dataset.particle_label(index)

    source_keys: dict[str, dict[str, tuple[str, ...]]] = {}
    for source_path, stats in source_stats.items():
        target = None
        if int(stats["target_count"]) > 0:
            target = (stats["target_sum"] / int(stats["target_count"])).astype(np.float64)
        source_keys[source_path] = _source_stratification_keys(
            source_path,
            target,
            stats["particle_label"],
        )

    rng = random.Random(seed)
    pending = list(source_to_indices)
    split_sources = {"train": [], "val": [], "test": []}
    for key_name in ("fine", "mid", "coarse", "broad"):
        groups: dict[tuple[str, ...], list[str]] = {}
        for source_path in pending:
            groups.setdefault(source_keys[source_path][key_name], []).append(source_path)
        next_pending: list[str] = []
        for sources in groups.values():
            if key_name == "broad" or len(sources) >= int(min_group_sources):
                _assign_source_group(split_sources, sources, val_fraction, test_fraction, rng)
            else:
                next_pending.extend(sources)
        pending = next_pending

    if pending:
        _assign_source_group(split_sources, pending, val_fraction, test_fraction, rng)

    if not split_sources["val"] or not split_sources["test"]:
        all_sources = list(source_to_indices)
        split_sources = {"train": [], "val": [], "test": []}
        _assign_source_group(split_sources, all_sources, val_fraction, test_fraction, rng)

    split = {
        name: sorted(index for source_path in sources for index in source_to_indices[source_path])
        for name, sources in split_sources.items()
    }
    if not split["train"] or not split["val"] or not split["test"]:
        raise ValueError("source-stratified split produced an empty train/validation/test split")
    return split


def _particle_labels_for_indices(
    dataset: H5GraphDataset,
    indices: list[int],
    show_progress: bool = True,
) -> np.ndarray:
    labels = []
    iterator = _progress(indices, desc="scan particle labels", total=len(indices), enabled=show_progress, leave=False)
    for index in iterator:
        label = dataset.particle_label(index)
        labels.append(np.nan if label is None else float(label))
    return np.asarray(labels, dtype=np.float32)


def _detector_lids_for_indices(
    dataset: H5GraphDataset,
    indices: list[int],
    show_progress: bool = True,
) -> list[int]:
    detector_lids: set[int] = set()
    iterator = _progress(indices, desc="scan detector IDs", total=len(indices), enabled=show_progress, leave=False)
    for index in iterator:
        lids = dataset.detector_lids(index)
        detector_lids.update(int(lid) for lid in lids if int(lid) >= 0)
    return sorted(detector_lids)


def _split_model_output(
    pred: Any,
    target_dim: int,
    mass_classification: bool,
    quality_prediction: bool = False,
) -> tuple[Any, Any | None, Any | None]:
    offset = int(target_dim)
    reconstruction = pred[:, :offset]
    mass_logit = None
    if mass_classification:
        mass_logit = pred[:, offset]
        offset += 1
    quality_logit = None
    if quality_prediction:
        quality_logit = pred[:, offset]
    return reconstruction, mass_logit, quality_logit


def _target_scaler_tensors(
    scalers: dict[str, StandardScaler],
    device: str,
) -> tuple[Any, Any]:
    import torch

    target_scaler = scalers["target"]
    mean = torch.as_tensor(target_scaler.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(target_scaler.std, dtype=torch.float32, device=device)
    return mean, std


def _inverse_scaled_target(values: Any, mean: Any, std: Any) -> Any:
    return values * std + mean


def _angular_loss_from_vectors(pred: Any, target: Any, *, angular_loss_scale_deg: float) -> Any:
    import torch
    import torch.nn.functional as F

    scale_rad = math.radians(max(float(angular_loss_scale_deg), 1.0e-6))
    pred_dir = F.normalize(pred[:, 4:7], dim=1, eps=1.0e-8)
    target_dir = F.normalize(target[:, 4:7], dim=1, eps=1.0e-8)
    dot = torch.sum(pred_dir * target_dir, dim=1).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    scaled_angle = torch.acos(dot) / scale_rad
    return F.smooth_l1_loss(scaled_angle, torch.zeros_like(scaled_angle), beta=1.0)


def _reconstruction_loss(
    pred_scaled: Any,
    target_scaled: Any,
    *,
    mode: str,
    target_mean: Any,
    target_std: Any,
    energy_weight: float,
    core_weight: float,
    direction_weight: float,
    core_scale_km: float,
    angular_loss_scale_deg: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    if mode == "scaled-mse":
        return F.mse_loss(pred_scaled, target_scaled)
    if mode == "weighted-scaled-mse":
        delta = pred_scaled - target_scaled
        energy_loss = torch.mean(delta[:, 0] * delta[:, 0])
        core_loss = torch.mean(torch.sum(delta[:, 1:4] * delta[:, 1:4], dim=1))
        direction_loss = torch.mean(torch.sum(delta[:, 4:7] * delta[:, 4:7], dim=1))
        return (
            float(energy_weight) * energy_loss
            + float(core_weight) * core_loss
            + float(direction_weight) * direction_loss
        ) / max(float(energy_weight) + float(core_weight) + float(direction_weight), 1.0e-12)

    pred = _inverse_scaled_target(pred_scaled, target_mean, target_std)
    target = _inverse_scaled_target(target_scaled, target_mean, target_std)
    if mode == "hybrid-angle":
        delta = pred_scaled - target_scaled
        energy_loss = torch.mean(delta[:, 0] * delta[:, 0])
        core_loss = torch.mean(torch.sum(delta[:, 1:4] * delta[:, 1:4], dim=1))
        direction_loss = _angular_loss_from_vectors(pred, target, angular_loss_scale_deg=angular_loss_scale_deg)
        return (
            float(energy_weight) * energy_loss
            + float(core_weight) * core_loss
            + float(direction_weight) * direction_loss
        ) / max(float(energy_weight) + float(core_weight) + float(direction_weight), 1.0e-12)
    if mode != "physics":
        raise ValueError("loss_mode must be 'scaled-mse', 'weighted-scaled-mse', 'hybrid-angle', or 'physics'")

    energy_loss = F.smooth_l1_loss(pred[:, 0] - target[:, 0], torch.zeros_like(target[:, 0]), beta=0.05)
    core_scale = max(float(core_scale_km), 1.0e-6)
    core_delta = (pred[:, 1:3] - target[:, 1:3]) / core_scale
    core_loss = torch.mean(torch.sum(core_delta * core_delta, dim=1))
    direction_loss = _angular_loss_from_vectors(pred, target, angular_loss_scale_deg=angular_loss_scale_deg)
    return (
        float(energy_weight) * energy_loss
        + float(core_weight) * core_loss
        + float(direction_weight) * direction_loss
    )


def _quality_targets_from_reconstruction(
    pred_scaled: Any,
    target_scaled: Any,
    *,
    target_mean: Any,
    target_std: Any,
    angular_scale_deg: float,
    core_scale_km: float,
    energy_scale: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    pred = _inverse_scaled_target(pred_scaled, target_mean, target_std)
    target = _inverse_scaled_target(target_scaled, target_mean, target_std)
    loge_delta = pred[:, 0] - target[:, 0]
    energy_score = torch.abs(torch.exp(loge_delta * math.log(10.0)) - 1.0) / max(float(energy_scale), 1.0e-6)
    core_score = torch.linalg.vector_norm(pred[:, 1:3] - target[:, 1:3], dim=1) / max(float(core_scale_km), 1.0e-6)
    pred_dir = F.normalize(pred[:, 4:7], dim=1, eps=1.0e-8)
    target_dir = F.normalize(target[:, 4:7], dim=1, eps=1.0e-8)
    dot = torch.sum(pred_dir * target_dir, dim=1).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    angle_score = torch.acos(dot) / math.radians(max(float(angular_scale_deg), 1.0e-6))
    score = (energy_score + core_score + angle_score) / 3.0
    return torch.exp(-score).clamp(0.0, 1.0)


def _quality_prediction_loss(
    quality_logit: Any,
    pred_scaled: Any,
    target_scaled: Any,
    *,
    target_mean: Any,
    target_std: Any,
    angular_scale_deg: float,
    core_scale_km: float,
    energy_scale: float,
) -> Any:
    import torch.nn.functional as F

    quality_target = _quality_targets_from_reconstruction(
        pred_scaled,
        target_scaled,
        target_mean=target_mean,
        target_std=target_std,
        angular_scale_deg=angular_scale_deg,
        core_scale_km=core_scale_km,
        energy_scale=energy_scale,
    ).detach()
    return F.binary_cross_entropy_with_logits(quality_logit.reshape(-1), quality_target)


def _mass_classification_loss(
    logits: Any,
    labels: Any,
    *,
    mode: str,
    pos_weight: Any | None,
    focal_gamma: float,
    ranking_weight: float = 0.0,
    ranking_margin: float = 1.0,
) -> Any:
    import torch
    import torch.nn.functional as F

    mode = str(mode).lower()
    if mode not in {"bce", "focal"}:
        raise ValueError("mass_loss_mode must be 'bce' or 'focal'")
    logits = logits.reshape(-1)
    labels = labels.reshape(-1)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    if mode == "bce":
        loss = torch.mean(bce)
    else:
        probs = torch.sigmoid(logits)
        p_t = torch.where(labels >= 0.5, probs, 1.0 - probs)
        focal = torch.pow((1.0 - p_t).clamp_min(0.0), max(float(focal_gamma), 0.0))
        loss = torch.mean(focal * bce)
    if float(ranking_weight) <= 0.0:
        return loss
    finite = torch.isfinite(logits) & torch.isfinite(labels)
    pos = logits[finite & (labels >= 0.5)]
    neg = logits[finite & (labels < 0.5)]
    if pos.numel() == 0 or neg.numel() == 0:
        return loss
    diff = pos[:, None] - neg[None, :]
    ranking = F.softplus(float(ranking_margin) - diff).mean()
    return loss + float(ranking_weight) * ranking


def _empty_binary_counts() -> dict[str, int | float]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "score_sum": 0.0, "score_sq_sum": 0.0}


def _update_binary_counts(counts: dict[str, int | float], logits: Any, labels: Any, *, logit_offset: float = 0.0) -> None:
    import torch

    labels = labels.reshape(-1)
    logits = logits.reshape(-1)
    mask = torch.isfinite(labels) & torch.isfinite(logits)
    if not torch.any(mask):
        return
    calibrated = logits[mask] - float(logit_offset)
    truth = labels[mask] >= 0.5
    pred = calibrated >= 0.0
    counts["tp"] += int(torch.sum(pred & truth).detach().cpu())
    counts["tn"] += int(torch.sum(~pred & ~truth).detach().cpu())
    counts["fp"] += int(torch.sum(pred & ~truth).detach().cpu())
    counts["fn"] += int(torch.sum(~pred & truth).detach().cpu())
    scores = torch.sigmoid(calibrated)
    counts["score_sum"] += float(torch.sum(scores).detach().cpu())
    counts["score_sq_sum"] += float(torch.sum(scores * scores).detach().cpu())


def _binary_count_metrics(counts: dict[str, int | float]) -> dict[str, float | int]:
    tp = int(counts["tp"])
    tn = int(counts["tn"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    n = tp + tn + fp + fn
    true_pos = tp + fn
    true_neg = tn + fp
    tpr = tp / true_pos if true_pos else float("nan")
    tnr = tn / true_neg if true_neg else float("nan")
    score_mean = float(counts["score_sum"]) / n if n else float("nan")
    score_var = float(counts["score_sq_sum"]) / n - score_mean * score_mean if n else float("nan")
    return {
        "n": n,
        "accuracy": float((tp + tn) / n) if n else float("nan"),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])) if n else float("nan"),
        "score_mean": score_mean,
        "score_std": math.sqrt(max(score_var, 0.0)) if n else float("nan"),
    }


def _predict_numpy(
    model: "TaleSdGNN",
    loader: Any,
    scalers: dict[str, StandardScaler],
    device: str,
    non_blocking: bool = False,
    desc: str = "predict",
    show_progress: bool = True,
    mass_classification: bool = False,
    quality_prediction: bool = False,
    target_dim: int = 7,
    mass_logit_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    model.eval()
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mass_logit_rows: list[np.ndarray] = []
    mass_label_rows: list[np.ndarray] = []
    quality_score_rows: list[np.ndarray] = []
    import torch

    with torch.no_grad():
        for batch_cpu in _progress(loader, desc=desc, total=len(loader), enabled=show_progress, leave=False):
            batch = _batch_to_device(batch_cpu, device, non_blocking=non_blocking)
            pred_all = model(batch)
            pred_scaled_tensor, mass_logit_tensor, quality_logit_tensor = _split_model_output(
                pred_all,
                target_dim,
                mass_classification,
                quality_prediction,
            )
            pred_scaled = pred_scaled_tensor.detach().cpu().numpy()
            target_scaled = batch["y"].detach().cpu().numpy()
            pred_rows.append(scalers["target"].inverse_transform(pred_scaled))
            target_rows.append(scalers["target"].inverse_transform(target_scaled))
            if mass_classification and mass_logit_tensor is not None:
                calibrated = mass_logit_tensor - float(mass_logit_offset)
                mass_logit_rows.append(calibrated.detach().cpu().numpy())
            if quality_prediction and quality_logit_tensor is not None:
                quality_score_rows.append(torch.sigmoid(quality_logit_tensor).detach().cpu().numpy())
            if "mass_label" in batch:
                mass_label_rows.append(batch["mass_label"].detach().cpu().numpy())
    mass_logits = np.concatenate(mass_logit_rows, axis=0) if mass_logit_rows else None
    mass_labels = np.concatenate(mass_label_rows, axis=0) if mass_label_rows else None
    quality_scores = np.concatenate(quality_score_rows, axis=0) if quality_score_rows else None
    return np.concatenate(pred_rows, axis=0), np.concatenate(target_rows, axis=0), mass_logits, mass_labels, quality_scores


def train_model(
    graphs_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.05,
    lr_scheduler: str = "none",
    lr_factor: float = 0.5,
    lr_patience: int = 2,
    early_stopping_patience: int = 0,
    model_architecture: str = "baseline",
    readout_heads: int = 4,
    classification_arch: str = "enhanced",
    detector_embedding_dim: int = 0,
    waveform_encoder: str = "none",
    waveform_embedding_dim: int = 64,
    waveform_transformer_heads: int = 4,
    waveform_transformer_layers: int = 1,
    loss_mode: str = "scaled-mse",
    energy_loss_weight: float = 1.0,
    core_loss_weight: float = 1.0,
    direction_loss_weight: float = 1.0,
    core_loss_scale_km: float = 0.12,
    angular_loss_scale_deg: float = 1.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_mode: str = "event",
    seed: int = 12345,
    device: str = "auto",
    sample_cache_size: int = 0,
    max_graphs: int | None = None,
    particle_filter: str = "all",
    num_workers: int = -1,
    preprocess_workers: int = 0,
    prefetch_factor: int = 2,
    collate_backend: str = "auto",
    collate_threads: int = 1,
    training_task: str = "reconstruction",
    mass_classification: bool = False,
    mass_loss_weight: float = 0.1,
    mass_loss_mode: str = "focal",
    mass_focal_gamma: float = 2.0,
    mass_pos_weight_mode: str = "none",
    mass_ranking_weight: float = 0.0,
    mass_ranking_margin: float = 1.0,
    mass_collapse_patience: int = 3,
    mass_collapse_score_std: float = 1.0e-3,
    mass_collapse_balanced_accuracy: float = 0.505,
    quality_prediction: bool = False,
    quality_loss_weight: float = 0.2,
    quality_angular_scale_deg: float = 1.0,
    quality_core_scale_km: float = 0.05,
    quality_energy_scale: float = 0.10,
    show_progress: bool = True,
    save_diagnostics: bool = True,
    diagnostic_energy_bin_width: float = 0.1,
    diagnostic_min_bin_count: int = 20,
) -> dict[str, Any]:
    stage_started = time.perf_counter()
    _progress_write("stage=start import_torch")
    import torch
    stage_seconds: dict[str, float] = {"import_torch": time.perf_counter() - stage_started}
    _progress_write(f"stage=done import_torch elapsed={stage_seconds['import_torch']:.1f}s")

    stage_started = time.perf_counter()
    _progress_write("stage=start import_model")
    from .model import PhysicsTaleSdGNN, TaleSdGNN
    stage_seconds["import_model"] = time.perf_counter() - stage_started
    _progress_write(f"stage=done import_model elapsed={stage_seconds['import_model']:.1f}s")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    training_task = str(training_task).lower()
    if training_task not in {"reconstruction", "mass"}:
        raise ValueError("training_task must be 'reconstruction' or 'mass'")
    if training_task == "mass":
        mass_classification = True
    classification_arch = str(classification_arch).lower()
    if classification_arch not in {"legacy", "enhanced"}:
        raise ValueError("classification_arch must be 'legacy' or 'enhanced'")
    mass_loss_mode = str(mass_loss_mode).lower()
    if mass_loss_mode not in {"bce", "focal"}:
        raise ValueError("mass_loss_mode must be 'bce' or 'focal'")
    mass_pos_weight_mode = str(mass_pos_weight_mode).lower()
    if mass_pos_weight_mode not in {"none", "auto"}:
        raise ValueError("mass_pos_weight_mode must be 'none' or 'auto'")
    mass_ranking_weight = max(float(mass_ranking_weight), 0.0)
    mass_ranking_margin = float(mass_ranking_margin)

    overall_started = time.perf_counter()
    stage_started = time.perf_counter()
    _progress_write(f"stage=start resolve_device requested={device}")
    device = resolve_device(device)
    stage_seconds["resolve_device"] = time.perf_counter() - stage_started
    _progress_write(f"stage=done resolve_device resolved={device} elapsed={stage_seconds['resolve_device']:.1f}s")
    prefetch_factor = max(int(prefetch_factor), 1)
    collate_threads = max(int(collate_threads), 0)
    pin_memory = device.startswith("cuda")
    if save_diagnostics:
        stage_started = time.perf_counter()
        _progress_write("stage=start latex_check")
        require_matplotlib_latex()
        stage_seconds["latex_check"] = time.perf_counter() - stage_started
        _progress_write(f"stage=done latex_check elapsed={stage_seconds['latex_check']:.1f}s")
    else:
        _progress_write("stage=skip latex_check save_diagnostics=0")
    stage_started = time.perf_counter()
    _progress_write("stage=start dataset_init")
    dataset = H5GraphDataset(
        graphs_path,
        require_target=True,
        require_particle_label=mass_classification,
        cache_size=sample_cache_size,
        load_node_positions=False,
        load_attrs=False,
        load_particle_label=True,
        load_detector_lids=int(detector_embedding_dim) > 0,
        max_graphs=max_graphs,
        particle_filter=particle_filter,
        show_progress=show_progress,
    )
    stage_seconds["dataset_init"] = time.perf_counter() - stage_started
    _progress_write(
        f"stage=done dataset_init graphs={len(dataset)} shards={len(dataset.paths)} "
        f"elapsed={stage_seconds['dataset_init']:.1f}s"
    )
    if len(dataset) < 2:
        raise ValueError("training needs at least two graphs with MC targets")
    requested_num_workers = int(num_workers)
    if requested_num_workers < 0:
        cpu_count = os.cpu_count() or 2
        num_workers = 0 if len(dataset) < 1024 else min(4, max(cpu_count // 2, 1))
    else:
        num_workers = max(requested_num_workers, 0)
    requested_collate_backend = collate_backend
    collate_backend = _resolve_collate_backend(collate_backend, n_graphs=len(dataset), num_workers=num_workers)
    preprocess_workers = max(int(preprocess_workers), 0)

    stage_started = time.perf_counter()
    _progress_write(f"stage=start split mode={split_mode} preprocess_workers={preprocess_workers}")
    if split_mode == "event":
        split = split_indices(
            len(dataset),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    elif split_mode == "source-path":
        split = split_indices_by_source_path(
            dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            show_progress=show_progress,
        )
    elif split_mode == "source-stratified":
        split = split_indices_by_stratified_source_path(
            dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            show_progress=show_progress,
            workers=preprocess_workers,
        )
    else:
        raise ValueError("split_mode must be 'event', 'source-path', or 'source-stratified'")
    stage_seconds["split"] = time.perf_counter() - stage_started
    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]
    _progress_write(
        f"stage=done split train={len(train_indices)} val={len(val_indices)} test={len(test_indices)} "
        f"elapsed={stage_seconds['split']:.1f}s"
    )

    detector_lids: list[int] = []
    if int(detector_embedding_dim) > 0:
        stage_started = time.perf_counter()
        detector_lids = _detector_lids_for_indices(dataset, train_indices, show_progress=show_progress)
        if not detector_lids:
            raise ValueError("detector embedding requested, but no detector IDs were found in training graphs")
        stage_seconds["scan_detector_ids"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    _progress_write(f"stage=start fit_scalers train_graphs={len(train_indices)} preprocess_workers={preprocess_workers}")
    scalers = fit_scalers(
        dataset,
        sorted(train_indices),
        show_progress=show_progress,
        workers=preprocess_workers,
    )
    stage_seconds["fit_scalers"] = time.perf_counter() - stage_started
    _progress_write(f"stage=done fit_scalers elapsed={stage_seconds['fit_scalers']:.1f}s")
    stage_started = time.perf_counter()
    first = dataset[train_indices[0]]
    model_kwargs = {
        "node_dim": first["node_features"].shape[1],
        "edge_dim": first["edge_features"].shape[1],
        "pulse_dim": max(first["pulse_features"].shape[1] - 1, 0),
        "waveform_channels": first["waveform_features"].shape[1]
        if first.get("waveform_features") is not None and first["waveform_features"].ndim == 3
        else 0,
        "waveform_length": first["waveform_features"].shape[2]
        if first.get("waveform_features") is not None and first["waveform_features"].ndim == 3
        else 0,
        "waveform_encoder": waveform_encoder,
        "waveform_embedding_dim": waveform_embedding_dim,
        "waveform_transformer_heads": waveform_transformer_heads,
        "waveform_transformer_layers": waveform_transformer_layers,
        "target_dim": first["target"].shape[0],
        "classification_dim": 1 if mass_classification else 0,
        "quality_dim": 1 if quality_prediction else 0,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "classification_arch": classification_arch,
        "detector_lids": detector_lids,
        "detector_embedding_dim": max(int(detector_embedding_dim), 0),
    }
    if model_architecture == "baseline":
        model = TaleSdGNN(**model_kwargs).to(device)
    elif model_architecture == "physics":
        model = PhysicsTaleSdGNN(**model_kwargs, readout_heads=readout_heads).to(device)
    else:
        raise ValueError("model_architecture must be 'baseline' or 'physics'")
    if num_workers > 0:
        dataset.close()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    lr_scheduler = str(lr_scheduler).lower()
    if lr_scheduler == "reduce-on-plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(lr_factor),
            patience=max(int(lr_patience), 0),
        )
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(learning_rate) * 0.1,
        )
    elif lr_scheduler != "none":
        raise ValueError("lr_scheduler must be 'none', 'reduce-on-plateau', or 'cosine'")
    target_dim = int(first["target"].shape[0])
    target_mean, target_std = _target_scaler_tensors(scalers, device)
    bce_loss_fn = None
    mass_pos_weight = 1.0
    mass_pos_weight_tensor = None
    if mass_classification:
        stage_started_labels = time.perf_counter()
        _progress_write(f"stage=start scan_particle_labels train_graphs={len(train_indices)}")
        train_mass_labels = _particle_labels_for_indices(dataset, train_indices, show_progress=show_progress)
        finite_labels = train_mass_labels[np.isfinite(train_mass_labels)]
        if finite_labels.size == 0:
            raise ValueError("mass classification requested, but training labels are missing")
        positives = float(np.sum(finite_labels >= 0.5))
        negatives = float(np.sum(finite_labels < 0.5))
        if mass_pos_weight_mode == "auto":
            mass_pos_weight = negatives / max(positives, 1.0)
            mass_pos_weight_tensor = torch.tensor(mass_pos_weight, dtype=torch.float32, device=device)
        bce_loss_fn = True
        stage_seconds["scan_particle_labels"] = time.perf_counter() - stage_started_labels
        _progress_write(
            f"stage=done scan_particle_labels proton={int(negatives)} iron={int(positives)} "
            f"elapsed={stage_seconds['scan_particle_labels']:.1f}s"
        )

    train_loader = _make_graph_loader(
        dataset,
        train_indices,
        scalers=scalers,
        batch_size=batch_size,
        shuffle=True,
        require_target=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=True,
        collate_backend=collate_backend,
        collate_threads=collate_threads,
    )
    val_loader = _make_graph_loader(
        dataset,
        val_indices,
        scalers=scalers,
        batch_size=batch_size,
        shuffle=False,
        require_target=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=True,
        collate_backend=collate_backend,
        collate_threads=collate_threads,
    )
    test_loader = _make_graph_loader(
        dataset,
        test_indices,
        scalers=scalers,
        batch_size=batch_size,
        shuffle=False,
        require_target=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=False,
        collate_backend=collate_backend,
        collate_threads=collate_threads,
    )
    stage_seconds["model_and_loaders"] = time.perf_counter() - stage_started
    _progress_write(
        f"device={device} data_loader_workers={num_workers} "
        f"preprocess_workers={preprocess_workers} "
        f"prefetch_factor={prefetch_factor} collate_backend={collate_backend} "
        f"collate_threads={collate_threads or 'auto'}"
        + f" split_mode={split_mode} model_architecture={model_architecture}"
        + f" classification_arch={classification_arch}"
        + f" detector_embedding_dim={max(int(detector_embedding_dim), 0)} detector_count={len(detector_lids)}"
        + f" waveform_encoder={waveform_encoder} waveform_channels={model_kwargs['waveform_channels']}"
        + f" training_task={training_task} loss_mode={loss_mode}"
        + f" angular_loss_scale_deg={angular_loss_scale_deg}"
        + f" mass_classification={mass_classification} quality_prediction={quality_prediction}"
        + (
            f" mass_loss_mode={mass_loss_mode} mass_pos_weight_mode={mass_pos_weight_mode}"
            f" mass_pos_weight={mass_pos_weight:.6g} mass_primary_threshold=0.5"
            f" mass_ranking_weight={mass_ranking_weight:.6g} mass_ranking_margin={mass_ranking_margin:.6g}"
            if mass_classification
            else ""
        )
        + f" particle_filter={particle_filter}"
        + (
            f" requested_collate_backend={requested_collate_backend}"
            if requested_collate_backend != collate_backend
            else ""
        )
    )

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    mass_collapse_epochs = 0
    history: list[dict[str, Any]] = []

    stage_started = time.perf_counter()
    epoch_iter = _progress(range(1, epochs + 1), desc="epochs", total=epochs, enabled=show_progress, position=0)
    for epoch in epoch_iter:
        epoch_started = time.perf_counter()
        model.train()
        train_losses = []
        train_reco_losses = []
        train_mass_losses = []
        train_quality_losses = []
        train_mass_counts = _empty_binary_counts()
        train_desc = f"epoch {epoch}/{epochs} train"
        train_started = time.perf_counter()
        for batch_cpu in _progress(
            train_loader,
            desc=train_desc,
            total=len(train_loader),
            enabled=show_progress,
            leave=False,
            position=1,
        ):
            batch = _batch_to_device(batch_cpu, device, non_blocking=pin_memory)
            pred_all = model(batch)
            pred, mass_logit, quality_logit = _split_model_output(
                pred_all,
                target_dim,
                mass_classification,
                quality_prediction,
            )
            reco_loss = None
            loss = None
            if training_task != "mass":
                reco_loss = _reconstruction_loss(
                    pred,
                    batch["y"],
                    mode=loss_mode,
                    target_mean=target_mean,
                    target_std=target_std,
                    energy_weight=energy_loss_weight,
                    core_weight=core_loss_weight,
                    direction_weight=direction_loss_weight,
                    core_scale_km=core_loss_scale_km,
                    angular_loss_scale_deg=angular_loss_scale_deg,
                )
                loss = reco_loss
            quality_loss = None
            if quality_prediction and quality_logit is not None and training_task != "mass":
                quality_loss = _quality_prediction_loss(
                    quality_logit,
                    pred,
                    batch["y"],
                    target_mean=target_mean,
                    target_std=target_std,
                    angular_scale_deg=quality_angular_scale_deg,
                    core_scale_km=quality_core_scale_km,
                    energy_scale=quality_energy_scale,
                )
                loss = quality_loss if loss is None else loss + float(quality_loss_weight) * quality_loss
            mass_loss = None
            if mass_classification and mass_logit is not None and bce_loss_fn is not None:
                labels = batch["mass_label"].to(dtype=mass_logit.dtype)
                mask = torch.isfinite(labels)
                if torch.any(mask):
                    mass_loss = _mass_classification_loss(
                        mass_logit[mask],
                        labels[mask],
                        mode=mass_loss_mode,
                        pos_weight=mass_pos_weight_tensor,
                        focal_gamma=mass_focal_gamma,
                        ranking_weight=mass_ranking_weight,
                        ranking_margin=mass_ranking_margin,
                    )
                    _update_binary_counts(train_mass_counts, mass_logit[mask], labels[mask], logit_offset=0.0)
                    if training_task == "mass":
                        loss = mass_loss
                    else:
                        loss = loss + float(mass_loss_weight) * mass_loss
            if loss is None:
                raise ValueError("no valid loss was computed for this batch")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if reco_loss is not None:
                train_reco_losses.append(float(reco_loss.detach().cpu()))
            if mass_loss is not None:
                train_mass_losses.append(float(mass_loss.detach().cpu()))
            if quality_loss is not None:
                train_quality_losses.append(float(quality_loss.detach().cpu()))
        train_seconds = time.perf_counter() - train_started

        model.eval()
        val_losses = []
        val_reco_losses = []
        val_mass_losses = []
        val_quality_losses = []
        val_mass_counts = _empty_binary_counts()
        val_started = time.perf_counter()
        with torch.no_grad():
            val_desc = f"epoch {epoch}/{epochs} val"
            for batch_cpu in _progress(
                val_loader,
                desc=val_desc,
                total=len(val_loader),
                enabled=show_progress,
                leave=False,
                position=1,
            ):
                batch = _batch_to_device(batch_cpu, device, non_blocking=pin_memory)
                pred_all = model(batch)
                pred, mass_logit, quality_logit = _split_model_output(
                    pred_all,
                    target_dim,
                    mass_classification,
                    quality_prediction,
                )
                reco_loss = None
                loss = None
                if training_task != "mass":
                    reco_loss = _reconstruction_loss(
                        pred,
                        batch["y"],
                        mode=loss_mode,
                        target_mean=target_mean,
                        target_std=target_std,
                        energy_weight=energy_loss_weight,
                        core_weight=core_loss_weight,
                        direction_weight=direction_loss_weight,
                        core_scale_km=core_loss_scale_km,
                        angular_loss_scale_deg=angular_loss_scale_deg,
                    )
                    loss = reco_loss
                quality_loss = None
                if quality_prediction and quality_logit is not None and training_task != "mass":
                    quality_loss = _quality_prediction_loss(
                        quality_logit,
                        pred,
                        batch["y"],
                        target_mean=target_mean,
                        target_std=target_std,
                        angular_scale_deg=quality_angular_scale_deg,
                        core_scale_km=quality_core_scale_km,
                        energy_scale=quality_energy_scale,
                    )
                    loss = quality_loss if loss is None else loss + float(quality_loss_weight) * quality_loss
                mass_loss = None
                if mass_classification and mass_logit is not None and bce_loss_fn is not None:
                    labels = batch["mass_label"].to(dtype=mass_logit.dtype)
                    mask = torch.isfinite(labels)
                    if torch.any(mask):
                        mass_loss = _mass_classification_loss(
                            mass_logit[mask],
                            labels[mask],
                            mode=mass_loss_mode,
                            pos_weight=mass_pos_weight_tensor,
                            focal_gamma=mass_focal_gamma,
                            ranking_weight=mass_ranking_weight,
                            ranking_margin=mass_ranking_margin,
                        )
                        _update_binary_counts(val_mass_counts, mass_logit[mask], labels[mask], logit_offset=0.0)
                        if training_task == "mass":
                            loss = mass_loss
                        else:
                            loss = loss + float(mass_loss_weight) * mass_loss
                if loss is None:
                    raise ValueError("no valid validation loss was computed for this batch")
                val_losses.append(float(loss.detach().cpu()))
                if reco_loss is not None:
                    val_reco_losses.append(float(reco_loss.detach().cpu()))
                if mass_loss is not None:
                    val_mass_losses.append(float(mass_loss.detach().cpu()))
                if quality_loss is not None:
                    val_quality_losses.append(float(quality_loss.detach().cpu()))
        val_seconds = time.perf_counter() - val_started
        epoch_seconds = time.perf_counter() - epoch_started

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": float(train_seconds),
            "val_seconds": float(val_seconds),
            "epoch_seconds": float(epoch_seconds),
        }
        if train_reco_losses:
            epoch_row["train_reconstruction_loss"] = float(np.mean(train_reco_losses))
        if val_reco_losses:
            epoch_row["val_reconstruction_loss"] = float(np.mean(val_reco_losses))
        if train_mass_losses:
            epoch_row["train_mass_loss"] = float(np.mean(train_mass_losses))
            train_mass_metrics = _binary_count_metrics(train_mass_counts)
            epoch_row["train_mass_accuracy"] = float(train_mass_metrics["accuracy"])
            epoch_row["train_mass_balanced_accuracy"] = float(train_mass_metrics["balanced_accuracy"])
            epoch_row["train_mass_score_mean"] = float(train_mass_metrics["score_mean"])
            epoch_row["train_mass_score_std"] = float(train_mass_metrics["score_std"])
        if val_mass_losses:
            epoch_row["val_mass_loss"] = float(np.mean(val_mass_losses))
            val_mass_metrics = _binary_count_metrics(val_mass_counts)
            epoch_row["val_mass_accuracy"] = float(val_mass_metrics["accuracy"])
            epoch_row["val_mass_balanced_accuracy"] = float(val_mass_metrics["balanced_accuracy"])
            epoch_row["val_mass_score_mean"] = float(val_mass_metrics["score_mean"])
            epoch_row["val_mass_score_std"] = float(val_mass_metrics["score_std"])
        if train_quality_losses:
            epoch_row["train_quality_loss"] = float(np.mean(train_quality_losses))
        if val_quality_losses:
            epoch_row["val_quality_loss"] = float(np.mean(val_quality_losses))
        history.append(epoch_row)
        if epoch_row["val_loss"] < best_val:
            best_val = epoch_row["val_loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if lr_scheduler == "reduce-on-plateau":
                scheduler.step(epoch_row["val_loss"])
            else:
                scheduler.step()
            epoch_row["next_lr"] = float(optimizer.param_groups[0]["lr"])

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            mass_text = f" mass_loss={epoch_row['val_mass_loss']:.6f}" if "val_mass_loss" in epoch_row else ""
            if "val_mass_accuracy" in epoch_row:
                mass_text += (
                    f" mass_acc={epoch_row['val_mass_accuracy']:.4f}"
                    f" mass_bal_acc={epoch_row['val_mass_balanced_accuracy']:.4f}"
                    f" mass_score_std={epoch_row['val_mass_score_std']:.4g}"
                )
            quality_text = f" quality_loss={epoch_row['val_quality_loss']:.6f}" if "val_quality_loss" in epoch_row else ""
            lr_text = f" lr={epoch_row['lr']:.3g}"
            if "next_lr" in epoch_row and epoch_row["next_lr"] != epoch_row["lr"]:
                lr_text += f" next_lr={epoch_row['next_lr']:.3g}"
            timing_text = (
                f" train_seconds={epoch_row['train_seconds']:.1f}"
                f" val_seconds={epoch_row['val_seconds']:.1f}"
                f" epoch_seconds={epoch_row['epoch_seconds']:.1f}"
            )
            _progress_write(
                f"epoch={epoch:04d} train_loss={epoch_row['train_loss']:.6f} "
                f"val_loss={epoch_row['val_loss']:.6f}{mass_text}{quality_text}{timing_text}{lr_text}",
            )
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                train_loss=f"{epoch_row['train_loss']:.4g}",
                val_loss=f"{epoch_row['val_loss']:.4g}",
            )
        if (
            training_task == "mass"
            and int(mass_collapse_patience) > 0
            and "val_mass_score_std" in epoch_row
            and "val_mass_balanced_accuracy" in epoch_row
        ):
            collapsed = (
                float(epoch_row["val_mass_score_std"]) <= float(mass_collapse_score_std)
                and float(epoch_row["val_mass_balanced_accuracy"]) <= float(mass_collapse_balanced_accuracy)
            )
            mass_collapse_epochs = mass_collapse_epochs + 1 if collapsed else 0
            if collapsed:
                _progress_write(
                    "mass classifier collapse warning: "
                    f"epoch={epoch:04d} consecutive={mass_collapse_epochs}/{int(mass_collapse_patience)} "
                    f"val_mass_score_std={epoch_row['val_mass_score_std']:.6g} "
                    f"val_mass_bal_acc={epoch_row['val_mass_balanced_accuracy']:.6g}"
                )
            if mass_collapse_epochs >= int(mass_collapse_patience):
                _progress_write(
                    "stopping mass training because the classifier is still a near-constant function; "
                    "fix the model/input/loss before spending more GPU time."
                )
                break
        if early_stopping_patience > 0 and epochs_without_improvement >= int(early_stopping_patience):
            _progress_write(
                f"early stopping at epoch={epoch:04d} "
                f"best_val_loss={best_val:.6f} patience={early_stopping_patience}",
            )
            break
    stage_seconds["epochs"] = time.perf_counter() - stage_started

    if best_state is not None:
        model.load_state_dict(best_state)

    stage_started = time.perf_counter()
    pred_val, target_val, mass_logit_val, mass_label_val, quality_val = _predict_numpy(
        model,
        val_loader,
        scalers,
        device,
        non_blocking=pin_memory,
        desc="validation predict",
        show_progress=show_progress,
        mass_classification=mass_classification,
        quality_prediction=quality_prediction,
        target_dim=target_dim,
        mass_logit_offset=0.0,
    )
    val_metrics = None if training_task == "mass" else reconstruction_metrics(pred_val, target_val)
    mass_threshold = 0.5
    tuned_mass_threshold = (
        balanced_accuracy_threshold(mass_logit_val, mass_label_val)
        if mass_classification and mass_logit_val is not None and mass_label_val is not None
        else 0.5
    )
    val_mass_metrics = (
        binary_classification_metrics(mass_logit_val, mass_label_val, threshold=mass_threshold)
        if mass_classification and mass_logit_val is not None and mass_label_val is not None
        else None
    )
    val_mass_tuned_metrics = (
        binary_classification_metrics(mass_logit_val, mass_label_val, threshold=tuned_mass_threshold)
        if mass_classification and mass_logit_val is not None and mass_label_val is not None
        else None
    )
    stage_seconds["validation_predict"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    pred_test, target_test, mass_logit_test, mass_label_test, quality_test = _predict_numpy(
        model,
        test_loader,
        scalers,
        device,
        non_blocking=pin_memory,
        desc="test predict",
        show_progress=show_progress,
        mass_classification=mass_classification,
        quality_prediction=quality_prediction,
        target_dim=target_dim,
        mass_logit_offset=0.0,
    )
    test_metrics = None if training_task == "mass" else reconstruction_metrics(pred_test, target_test)
    test_mass_metrics = (
        binary_classification_metrics(mass_logit_test, mass_label_test, threshold=mass_threshold)
        if mass_classification and mass_logit_test is not None and mass_label_test is not None
        else None
    )
    test_mass_tuned_metrics = (
        binary_classification_metrics(mass_logit_test, mass_label_test, threshold=tuned_mass_threshold)
        if mass_classification and mass_logit_test is not None and mass_label_test is not None
        else None
    )
    stage_seconds["test_predict"] = time.perf_counter() - stage_started
    if val_metrics is not None:
        _progress_write("validation metrics: " + json.dumps(val_metrics, sort_keys=True))
    if test_metrics is not None:
        _progress_write("test metrics: " + json.dumps(test_metrics, sort_keys=True))
    if val_mass_metrics is not None:
        _progress_write("validation mass metrics: " + json.dumps(val_mass_metrics, sort_keys=True))
    if val_mass_tuned_metrics is not None:
        _progress_write("validation mass tuned metrics: " + json.dumps(val_mass_tuned_metrics, sort_keys=True))
    if test_mass_metrics is not None:
        _progress_write("test mass metrics: " + json.dumps(test_mass_metrics, sort_keys=True))
    if test_mass_tuned_metrics is not None:
        _progress_write("test mass tuned metrics: " + json.dumps(test_mass_tuned_metrics, sort_keys=True))

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, Any] = {}
    if save_diagnostics:
        stage_started = time.perf_counter()
        diagnostics = save_training_diagnostics(
            output,
            history=history,
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
            energy_bin_width=diagnostic_energy_bin_width,
            min_bin_count=diagnostic_min_bin_count,
            save_reconstruction=training_task != "mass",
        )
        stage_seconds["diagnostics"] = time.perf_counter() - stage_started
    stage_seconds["total_before_save"] = time.perf_counter() - overall_started
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "scalers": {name: scaler.to_dict() for name, scaler in scalers.items()},
        "history": history,
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": val_mass_metrics,
            "test_mass": test_mass_metrics,
            "validation_mass_tuned": val_mass_tuned_metrics,
            "test_mass_tuned": test_mass_tuned_metrics,
        },
        "diagnostics": diagnostics,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "split": {
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "split_mode": split_mode,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(test_indices),
            "particle_filter": particle_filter,
        },
        "runtime": {
            "device": device,
            "num_workers": num_workers,
            "preprocess_workers": preprocess_workers,
            "prefetch_factor": prefetch_factor,
            "collate_backend": collate_backend,
            "requested_collate_backend": requested_collate_backend,
            "collate_threads": collate_threads,
            "training_task": training_task,
            "mass_classification": mass_classification,
            "mass_loss_weight": mass_loss_weight,
            "mass_loss_mode": mass_loss_mode,
            "mass_focal_gamma": mass_focal_gamma,
            "mass_pos_weight_mode": mass_pos_weight_mode,
            "mass_pos_weight": mass_pos_weight,
            "mass_primary_threshold": mass_threshold,
            "mass_tuned_threshold": tuned_mass_threshold,
            "mass_ranking_weight": mass_ranking_weight,
            "mass_ranking_margin": mass_ranking_margin,
            "mass_collapse_patience": mass_collapse_patience,
            "mass_collapse_score_std": mass_collapse_score_std,
            "mass_collapse_balanced_accuracy": mass_collapse_balanced_accuracy,
            "quality_prediction": quality_prediction,
            "quality_loss_weight": quality_loss_weight,
            "quality_angular_scale_deg": quality_angular_scale_deg,
            "quality_core_scale_km": quality_core_scale_km,
            "quality_energy_scale": quality_energy_scale,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "early_stopping_patience": early_stopping_patience,
            "hidden_dim": hidden_dim,
            "layers": num_layers,
            "dropout": dropout,
            "classification_arch": classification_arch,
            "detector_embedding_dim": max(int(detector_embedding_dim), 0),
            "detector_count": len(detector_lids),
            "waveform_encoder": waveform_encoder,
            "waveform_embedding_dim": waveform_embedding_dim,
            "waveform_transformer_heads": waveform_transformer_heads,
            "waveform_transformer_layers": waveform_transformer_layers,
            "waveform_channels": model_kwargs["waveform_channels"],
            "waveform_length": model_kwargs["waveform_length"],
            "loss_mode": loss_mode,
            "energy_loss_weight": energy_loss_weight,
            "core_loss_weight": core_loss_weight,
            "direction_loss_weight": direction_loss_weight,
            "core_loss_scale_km": core_loss_scale_km,
            "angular_loss_scale_deg": angular_loss_scale_deg,
            "max_graphs": max_graphs,
            "particle_filter": particle_filter,
            "stage_seconds": {name: round(value, 3) for name, value in stage_seconds.items()},
        },
    }
    stage_started = time.perf_counter()
    torch.save(checkpoint, output)
    stage_seconds["save_checkpoint"] = time.perf_counter() - stage_started
    checkpoint["runtime"]["stage_seconds"] = {name: round(value, 3) for name, value in stage_seconds.items()}

    stage_started = time.perf_counter()
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "metrics": {
                    "validation": val_metrics,
                    "test": test_metrics,
                    "validation_mass": val_mass_metrics,
                    "test_mass": test_mass_metrics,
                    "validation_mass_tuned": val_mass_tuned_metrics,
                    "test_mass_tuned": test_mass_tuned_metrics,
                },
                "split": checkpoint["split"],
                "runtime": checkpoint["runtime"],
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    stage_seconds["save_metrics"] = time.perf_counter() - stage_started
    stage_seconds["total"] = time.perf_counter() - overall_started
    _progress_write(
        "stage_seconds: "
        + json.dumps({name: round(value, 3) for name, value in stage_seconds.items()}, sort_keys=True)
    )
    dataset.close()
    return {
        "checkpoint": str(output),
        "metrics_path": str(metrics_path),
        "diagnostics": diagnostics,
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_mass": val_mass_metrics,
            "test_mass": test_mass_metrics,
            "validation_mass_tuned": val_mass_tuned_metrics,
            "test_mass_tuned": test_mass_tuned_metrics,
        },
    }
