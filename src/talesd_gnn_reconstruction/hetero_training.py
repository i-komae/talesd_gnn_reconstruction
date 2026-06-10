from __future__ import annotations

import os
import random
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import RunningFeatureStats, StandardScaler
from .diagnostics import save_training_diagnostics
from .hetero_data import EDGE_TYPE_BY_RELATION
from .hetero_graph_io import H5HeteroGraphDataset, H5PyGHeteroGraphDataset
from .hetero_model import MinimalHeteroTaleSdGNN
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


def _make_hetero_loader(
    dataset: H5PyGHeteroGraphDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> Any:
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    worker_count = min(max(int(num_workers), 0), max(len(indices), 1))
    kwargs: dict[str, Any] = {
        "batch_size": max(int(batch_size), 1),
        "shuffle": bool(shuffle),
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
    }
    if worker_count > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["worker_init_fn"] = _loader_worker_init
    return DataLoader(Subset(dataset, list(indices)), **kwargs)


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
    target = batch.target.to(device=device, dtype=pred_all.dtype)
    pred_scaled, mass_logit, quality_logit, error_raw = _split_model_output(
        pred_all,
        target_dim,
        mass_classification,
        quality_prediction=quality_prediction,
        error_prediction=error_prediction,
    )
    target_mean, target_std = _target_scaler_tensors(scalers, device)
    labels = (
        batch.particle_label.to(device=device, dtype=pred_all.dtype).reshape(-1)
        if mass_classification and "particle_label" in batch
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    import torch

    model.eval()
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mass_logit_rows: list[np.ndarray] = []
    mass_label_rows: list[np.ndarray] = []
    quality_score_rows: list[np.ndarray] = []
    error_prediction_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _progress(loader, desc=desc, total=len(loader), enabled=show_progress, leave=False):
            batch = batch.to(device)
            pred_all = model(batch)
            pred_scaled, mass_logit, quality_logit, error_raw = _split_model_output(
                pred_all,
                target_dim,
                mass_classification,
                quality_prediction=quality_prediction,
                error_prediction=error_prediction,
            )
            pred_rows.append(scalers["target"].inverse_transform(pred_scaled.detach().cpu().numpy()))
            target_rows.append(scalers["target"].inverse_transform(batch.target.detach().cpu().numpy()))
            if mass_classification and mass_logit is not None and "particle_label" in batch:
                mass_logit_rows.append(mass_logit.detach().cpu().numpy())
                mass_label_rows.append(batch.particle_label.detach().cpu().numpy())
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
    return (
        np.concatenate(pred_rows, axis=0),
        np.concatenate(target_rows, axis=0),
        np.concatenate(mass_logit_rows, axis=0) if mass_logit_rows else None,
        np.concatenate(mass_label_rows, axis=0) if mass_label_rows else None,
        np.concatenate(quality_score_rows, axis=0) if quality_score_rows else None,
        np.concatenate(error_prediction_rows, axis=0) if error_prediction_rows else None,
    )


def _append_component_mean(row: dict[str, Any], prefix: str, component_values: dict[str, list[float]]) -> None:
    for name, values in component_values.items():
        if values:
            row[f"{prefix}_{name}_loss"] = float(np.mean(values))


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
    persistent_workers: bool = False,
    pin_memory: bool | None = None,
    loader_memory_budget_gib: float | None = None,
    loader_memory_estimate_samples: int = 512,
    split_workers: int = 0,
    show_progress: bool = True,
) -> dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)
    gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
    prefetch_factor = max(int(prefetch_factor), 1)
    persistent_workers = bool(persistent_workers)
    pin_memory = device.startswith("cuda") if pin_memory is None else bool(pin_memory)
    loss_mode = str(loss_mode).lower()
    model_architecture = str(model_architecture)
    if model_architecture not in {"minimal_hetero", "hetero_attention"}:
        raise ValueError("model_architecture must be 'minimal_hetero' or 'hetero_attention'")
    valid_loss_modes = {"scaled-mse", "weighted-scaled-mse", "hybrid-angle", "physics", "physics-nll", "nll"}
    if loss_mode not in valid_loss_modes:
        raise ValueError(
            "loss_mode must be 'scaled-mse', 'weighted-scaled-mse', 'hybrid-angle', "
            "'physics', 'physics-nll', or 'nll'"
        )
    if loss_mode in {"physics-nll", "nll"} and not error_prediction:
        error_prediction = True
        error_loss_weight = 0.0
    base_dataset = H5HeteroGraphDataset(
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
    train_indices = split["train"]
    val_indices = split["val"]
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
        f"held_batches_estimate={loader_settings['held_batches_estimate']} "
        f"estimated_loader_bytes={loader_settings['estimated_loader_bytes']} "
        f"loader_memory_budget_bytes={loader_settings['loader_memory_budget_bytes']}"
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
        architecture=model_architecture,
        attention_heads=attention_heads,
        readout_heads=readout_heads,
    ).to(device)
    base_dataset.close()

    pyg_dataset = H5PyGHeteroGraphDataset(
        graphs_path,
        require_target=True,
        require_particle_label=mass_classification,
        scalers=scalers,
        waveform_length=resolved_waveform_length,
    )
    train_loader = _make_hetero_loader(
        pyg_dataset,
        train_indices,
        batch_size=max(int(batch_size), 1),
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = _make_hetero_loader(
        pyg_dataset,
        val_indices,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = _make_hetero_loader(
        pyg_dataset,
        split["test"],
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        train_components: dict[str, list[float]] = {}
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation_steps = 0
        for batch in train_loader:
            batch = batch.to(device)
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
            (loss / float(gradient_accumulation_steps)).backward()
            pending_accumulation_steps += 1
            if pending_accumulation_steps >= gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation_steps = 0
            train_losses.append(float(loss.detach().cpu()))
            for name, value in components.items():
                train_components.setdefault(name, []).append(float(value.detach().cpu()))
        if pending_accumulation_steps > 0:
            _scale_gradients(
                model,
                float(gradient_accumulation_steps) / float(pending_accumulation_steps),
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.eval()
        val_losses = []
        val_components: dict[str, list[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
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
                val_losses.append(float(loss.detach().cpu()))
                for name, value in components.items():
                    val_components.setdefault(name, []).append(float(value.detach().cpu()))
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
        }
        _append_component_mean(epoch_row, "train", train_components)
        _append_component_mean(epoch_row, "val", val_components)
        history.append(epoch_row)
    pred_val, target_val, mass_logit_val, mass_label_val, quality_val, error_val = _predict_hetero_numpy(
        model,
        val_loader,
        scalers,
        device,
        target_dim=target_dim,
        mass_classification=mass_classification,
        quality_prediction=quality_prediction,
        error_prediction=error_prediction,
        error_angular_scale_deg=error_angular_scale_deg,
        error_core_scale_km=error_core_scale_km,
        error_energy_scale=error_energy_scale,
        desc="hetero validation predict",
        show_progress=show_progress,
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
        desc="hetero test predict",
        show_progress=show_progress,
    )
    metrics: dict[str, Any] = {
        "validation": reconstruction_metrics(pred_val, target_val),
        "test": reconstruction_metrics(pred_test, target_test),
    }
    if mass_classification and mass_logit_val is not None and mass_label_val is not None:
        metrics["validation_mass"] = binary_classification_metrics(mass_logit_val, mass_label_val)
    if mass_classification and mass_logit_test is not None and mass_label_test is not None:
        metrics["test_mass"] = binary_classification_metrics(mass_logit_test, mass_label_test)
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
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "hetero_scalers": _scalers_to_dict(scalers),
        "history": history,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "split": {
            "train_indices": np.asarray(train_indices, dtype=np.int64),
            "val_indices": np.asarray(val_indices, dtype=np.int64),
            "test_indices": np.asarray(split["test"], dtype=np.int64),
            "split_mode": split_mode,
            "n_train": int(len(train_indices)),
            "n_val": int(len(val_indices)),
            "n_test": int(len(split["test"])),
        },
        "runtime": {
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
            "batch_size": int(max(int(batch_size), 1)),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(max(int(batch_size), 1) * gradient_accumulation_steps),
            "data_loader": {
                **loader_settings,
                **graph_byte_summary,
                "loader_memory_budget_gib": None
                if loader_memory_budget_gib is None
                else float(loader_memory_budget_gib),
                "loader_memory_estimate_samples": int(loader_memory_estimate_samples),
                "split_workers": int(split_workers),
            },
        },
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
        "metrics": metrics,
        "diagnostics": diagnostics,
        "split": {
            "split_mode": split_mode,
            "n_train": int(len(train_indices)),
            "n_val": int(len(val_indices)),
            "n_test": int(len(split["test"])),
        },
        "runtime": checkpoint["runtime"],
    }
    metrics_path = Path(f"{output}.metrics.json")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True, default=_json_default))
    pyg_dataset.close()
    return {
        "checkpoint": str(output),
        "metrics_json": str(metrics_path),
        "history": history,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "split": checkpoint["split"],
    }
