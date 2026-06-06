from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import RunningFeatureStats, StandardScaler
from .hetero_data import EDGE_TYPE_BY_RELATION
from .hetero_graph_io import H5HeteroGraphDataset, H5PyGHeteroGraphDataset
from .hetero_model import MinimalHeteroTaleSdGNN
from .train import split_indices, split_indices_by_source_path, split_indices_by_stratified_source_path, resolve_device


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
    first = dataset[int(indices[0])]
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
        sample = dataset[int(index)]
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
        sample = dataset[int(index)]
        waveforms = np.asarray(sample["detector_waveforms"])
        if waveforms.ndim != 3:
            raise ValueError(
                "detector_waveforms must be 3D [detector, channel, time], "
                f"got shape={waveforms.shape} at graph index {index}"
            )
        channels = int(waveforms.shape[1])
        if waveform_channels is None:
            waveform_channels = channels
        elif waveform_channels != channels:
            raise ValueError(
                f"detector waveform channel mismatch: expected {waveform_channels}, "
                f"got {channels} at graph index {index}"
            )
        max_length = max(max_length, int(waveforms.shape[2]))
    resolved_length = requested_length if requested_length is not None else max_length
    if waveform_channels is None or resolved_length <= 0:
        raise ValueError("training graphs have no detector waveform samples")
    return waveform_channels, resolved_length


def _scalers_to_dict(scalers: dict[str, StandardScaler]) -> dict[str, Any]:
    return {name: scaler.to_dict() for name, scaler in scalers.items()}


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
            workers=0,
            source_val_fraction=source_val_fraction,
            source_test_fraction=source_test_fraction,
        )
    raise ValueError("split_mode must be 'event', 'source-path', or 'source-stratified'")


def train_hetero_model(
    graphs_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.05,
    waveform_encoder: str = "cnn",
    waveform_embedding_dim: int = 64,
    waveform_length: int | None = None,
    mass_classification: bool = False,
    mass_loss_weight: float = 0.1,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    source_val_fraction: float = 0.10,
    source_test_fraction: float = 0.20,
    split_mode: str = "event",
    seed: int = 12345,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)
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
    )
    train_indices = split["train"]
    val_indices = split["val"]
    _waveform_channels, resolved_waveform_length = _resolve_waveform_shape(
        base_dataset,
        train_indices,
        waveform_length=waveform_length,
    )
    scalers = fit_hetero_scalers(base_dataset, train_indices)
    first = base_dataset[train_indices[0]]
    target_dim = int(first["target"].shape[0])
    classification_dim = 1 if mass_classification else 0
    model = MinimalHeteroTaleSdGNN.from_sample(
        first,
        target_dim=target_dim,
        classification_dim=classification_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        waveform_encoder=waveform_encoder,
        waveform_embedding_dim=waveform_embedding_dim,
        waveform_length=resolved_waveform_length,
    ).to(device)
    base_dataset.close()

    pyg_dataset = H5PyGHeteroGraphDataset(
        graphs_path,
        require_target=True,
        require_particle_label=mass_classification,
        scalers=scalers,
        waveform_length=resolved_waveform_length,
    )
    train_loader = DataLoader(Subset(pyg_dataset, train_indices), batch_size=max(int(batch_size), 1), shuffle=True)
    val_loader = DataLoader(Subset(pyg_dataset, val_indices), batch_size=max(int(batch_size), 1), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            pred_all = model(batch)
            target = batch.target.to(device=device, dtype=pred_all.dtype)
            loss = F.mse_loss(pred_all[:, :target_dim], target)
            if mass_classification and classification_dim > 0 and "particle_label" in batch:
                labels = batch.particle_label.to(device=device, dtype=pred_all.dtype).reshape(-1)
                mass_loss = F.binary_cross_entropy_with_logits(pred_all[:, target_dim], labels)
                loss = loss + float(mass_loss_weight) * mass_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_all = model(batch)
                target = batch.target.to(device=device, dtype=pred_all.dtype)
                val_losses.append(float(F.mse_loss(pred_all[:, :target_dim], target).detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            }
        )
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "hetero_scalers": _scalers_to_dict(scalers),
        "history": history,
        "split": {
            "train_indices": np.asarray(train_indices, dtype=np.int64),
            "val_indices": np.asarray(val_indices, dtype=np.int64),
            "test_indices": np.asarray(split["test"], dtype=np.int64),
            "split_mode": split_mode,
        },
        "runtime": {
            "graph_format": "hetero",
            "training_path": "hetero_smoke",
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "mass_classification": bool(mass_classification),
            "waveform_length": int(resolved_waveform_length),
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
    pyg_dataset.close()
    return {"checkpoint": str(output), "history": history, "split": checkpoint["split"]}
