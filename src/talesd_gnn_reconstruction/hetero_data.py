from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


EDGE_TYPE_BY_RELATION: dict[str, tuple[str, str, str]] = {
    "pulse__same_detector_next__pulse": ("pulse", "same_detector_next", "pulse"),
    "pulse__same_detector_prev__pulse": ("pulse", "same_detector_prev", "pulse"),
    "pulse__near_space__pulse": ("pulse", "near_space", "pulse"),
    "pulse__time_causal__pulse": ("pulse", "time_causal", "pulse"),
    "detector__near__detector": ("detector", "near", "detector"),
    "detector__observes__pulse": ("detector", "observes", "pulse"),
    "pulse__observed_by__detector": ("pulse", "observed_by", "detector"),
}

DETECTOR_WAVEFORM_VALID_COLUMN = "detector_waveform_valid"
V3_DETECTOR_FEATURE_COLUMNS = (
    # v3-compatible name kept by dstio: this is the detector node time,
    # defined as the first Ising-kept pulse onset on the pulse_arrival_usec_rel
    # axis. It is not the detector waveform start time.
    "detector_trigger_usec_rel",
    "log10_detector_max_pulse_rho",
    "log10_detector_sum_pulse_rho",
    "sqrt_detector_sum_pulse_rho",
    "detector_accepted_pulse_count",
    "detector_accepted_pulse_time_span_usec",
    "nearest_detector_distance_km",
    "mean3_detector_distance_km",
    "neighbor_count_1p5km",
    "local_detector_density_1p5km",
    "detector_has_signal",
    "detector_arrival_time_valid",
    "detector_live_status",
    "detector_waveform_valid",
    "detector_has_ising_kept_pulse",
    "detector_ising_kept_pulse_count",
    "detector_ising_removed_pulse_count",
)


def detector_feature_index(name: str) -> int | None:
    try:
        import dstio.tale.graph as tale_graph

        columns = list(tale_graph.graph_columns().get("detector_features", []))
    except Exception:
        columns = list(V3_DETECTOR_FEATURE_COLUMNS)
    try:
        return int(columns.index(str(name)))
    except ValueError:
        return None


class TorchGeometricUnavailableError(ImportError):
    """Raised when PyG conversion is requested without torch_geometric installed."""


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _long_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(value), dtype=torch.long, device=device)


def normalize_detector_waveforms(
    waveforms: torch.Tensor,
    waveform_length: int | None,
) -> torch.Tensor:
    if waveform_length is None:
        return waveforms
    waveform_length = int(waveform_length)
    if waveform_length <= 0:
        raise ValueError("waveform_length must be positive")
    if waveforms.ndim != 3:
        raise ValueError(f"detector_waveforms must be 3D [detector, channel, time], got shape={tuple(waveforms.shape)}")
    current_length = int(waveforms.shape[-1])
    if current_length == waveform_length:
        return waveforms
    if current_length > waveform_length:
        return waveforms[..., :waveform_length].contiguous()
    pad_shape = list(waveforms.shape)
    pad_shape[-1] = waveform_length - current_length
    padding = torch.zeros(pad_shape, dtype=waveforms.dtype, device=waveforms.device)
    return torch.cat([waveforms, padding], dim=-1)


def hetero_sample_to_tensors(
    sample: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    scalers: Mapping[str, Any] | None = None,
    waveform_length: int | None = None,
) -> dict[str, Any]:
    resolved_device = torch.device(device) if device is not None else None
    detector_features = _tensor(sample["detector_features"], dtype=torch.float32, device=resolved_device)
    waveform_valid_index = detector_feature_index(DETECTOR_WAVEFORM_VALID_COLUMN)
    if waveform_valid_index is not None and waveform_valid_index < detector_features.shape[1]:
        detector_waveform_valid = detector_features[:, waveform_valid_index].clamp(0.0, 1.0)
    else:
        detector_waveform_valid = torch.ones(detector_features.shape[0], dtype=torch.float32, device=resolved_device)
    pulse_features = _tensor(sample["pulse_features"], dtype=torch.float32, device=resolved_device)
    edge_index_by_type = {
        relation: _long_tensor(edge_index, device=resolved_device)
        for relation, edge_index in sample["edge_index_by_type"].items()
    }
    edge_features_by_type = {
        relation: _tensor(edge_features, dtype=torch.float32, device=resolved_device)
        for relation, edge_features in sample["edge_features_by_type"].items()
    }
    detector_features = _scale_tensor(detector_features, _scaler_for(scalers, "detector", "detector_features"))
    detector_context = _scale_tensor(
        _tensor(sample["detector_context_features"], dtype=torch.float32, device=resolved_device),
        _scaler_for(scalers, "detector_context", "detector_context_features"),
    )
    pulse_features = _scale_tensor(pulse_features, _scaler_for(scalers, "pulse", "pulse_features"))
    edge_features_by_type = {
        relation: _scale_tensor(
            edge_features,
            _scaler_for(scalers, f"edge:{relation}", relation),
        )
        for relation, edge_features in edge_features_by_type.items()
    }
    target = sample.get("target")
    target_tensor = None
    if target is not None:
        target_tensor = _tensor(target, dtype=torch.float32, device=resolved_device).reshape(1, -1)
        target_tensor = _scale_tensor(target_tensor, _scaler_for(scalers, "target"))
    particle_label = sample.get("particle_label")
    return {
        "detector": {
            "x": detector_features,
            "context": detector_context,
            "pos": _tensor(sample["detector_positions_km"], dtype=torch.float32, device=resolved_device),
            "lid": _long_tensor(sample["detector_lids"], device=resolved_device),
            "waveform": normalize_detector_waveforms(
                _tensor(sample["detector_waveforms"], dtype=torch.float32, device=resolved_device),
                waveform_length,
            ),
            "waveform_valid": detector_waveform_valid,
            "batch": torch.zeros(detector_features.shape[0], dtype=torch.long, device=resolved_device),
        },
        "pulse": {
            "x": pulse_features,
            "pos": _tensor(sample["pulse_positions_km"], dtype=torch.float32, device=resolved_device),
            "lid": _long_tensor(sample["pulse_lids"], device=resolved_device),
            "detector_index": _long_tensor(sample["pulse_detector_index"], device=resolved_device),
            "pulse_bounds": _tensor(sample["pulse_bounds"], dtype=torch.float32, device=resolved_device),
            "batch": torch.zeros(pulse_features.shape[0], dtype=torch.long, device=resolved_device),
        },
        "edge_index_by_type": edge_index_by_type,
        "edge_features_by_type": edge_features_by_type,
        "target": target_tensor,
        "particle_label": None
        if particle_label is None
        else _tensor([particle_label], dtype=torch.float32, device=resolved_device),
        "metadata": dict(sample.get("metadata", {})),
        "num_graphs": 1,
    }


def _scaler_for(scalers: Mapping[str, Any] | None, *keys: str) -> Any:
    if scalers is None:
        return None
    for key in keys:
        if key in scalers:
            return scalers[key]
    return None


def _scale_tensor(value: torch.Tensor, scaler: Any) -> torch.Tensor:
    if scaler is None or value.numel() == 0:
        return value
    if hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean = getattr(scaler, "mean")
        std = getattr(scaler, "std")
    elif isinstance(scaler, Mapping):
        mean = scaler.get("mean")
        std = scaler.get("std")
    else:
        return value
    if mean is None or std is None:
        return value
    mean_tensor = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
    std_tensor = torch.as_tensor(std, dtype=value.dtype, device=value.device).clamp_min(1.0e-6)
    if value.shape[-1] != mean_tensor.shape[-1]:
        raise ValueError(f"scaler dimension mismatch: value_dim={value.shape[-1]} scaler_dim={mean_tensor.shape[-1]}")
    return (value - mean_tensor) / std_tensor


def sample_to_hetero_data(
    sample: Mapping[str, Any],
    *,
    scalers: Mapping[str, Any] | None = None,
    waveform_length: int | None = None,
):
    try:
        from torch_geometric.data import HeteroData
    except ModuleNotFoundError as exc:
        raise TorchGeometricUnavailableError(
            "torch_geometric is not installed; install PyTorch Geometric before requesting HeteroData conversion"
        ) from exc

    tensors = hetero_sample_to_tensors(sample, scalers=scalers, waveform_length=waveform_length)
    data = HeteroData()
    data["detector"].x = tensors["detector"]["x"]
    data["detector"].context = tensors["detector"]["context"]
    data["detector"].pos = tensors["detector"]["pos"]
    data["detector"].lid = tensors["detector"]["lid"]
    data["detector"].waveform = tensors["detector"]["waveform"]
    data["detector"].waveform_valid = tensors["detector"]["waveform_valid"]

    data["pulse"].x = tensors["pulse"]["x"]
    data["pulse"].pos = tensors["pulse"]["pos"]
    data["pulse"].lid = tensors["pulse"]["lid"]
    data["pulse"].detector_index = tensors["pulse"]["detector_index"]
    data["pulse"].pulse_bounds = tensors["pulse"]["pulse_bounds"]

    for relation, edge_type in EDGE_TYPE_BY_RELATION.items():
        edge_index = tensors["edge_index_by_type"].get(relation)
        edge_attr = tensors["edge_features_by_type"].get(relation)
        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1], 0), dtype=torch.float32)
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr

    if tensors["target"] is not None:
        data.target = tensors["target"]
        data.y = tensors["target"]
    if tensors["particle_label"] is not None:
        data.particle_label = tensors["particle_label"]
    data.metadata = tensors["metadata"]
    return data


def _storage_batch(storage: Any, n_nodes: int, *, device: torch.device) -> torch.Tensor:
    if "batch" in storage:
        return storage["batch"].to(device=device, dtype=torch.long)
    return torch.zeros(n_nodes, dtype=torch.long, device=device)


def hetero_data_to_tensors(data: Any) -> dict[str, Any]:
    detector = data["detector"]
    pulse = data["pulse"]
    detector_x = detector["x"].to(dtype=torch.float32)
    pulse_x = pulse["x"].to(dtype=torch.float32)
    edge_index_by_type = {}
    edge_features_by_type = {}
    for relation, edge_type in EDGE_TYPE_BY_RELATION.items():
        edge_store = data[edge_type]
        edge_index_by_type[relation] = edge_store["edge_index"].to(dtype=torch.long)
        edge_attr = edge_store["edge_attr"] if "edge_attr" in edge_store else None
        if edge_attr is None:
            edge_attr = torch.zeros(
                edge_index_by_type[relation].shape[1],
                0,
                dtype=torch.float32,
                device=edge_index_by_type[relation].device,
            )
        edge_features_by_type[relation] = edge_attr.to(dtype=torch.float32)
    target = data["target"] if "target" in data else None
    particle_label = data["particle_label"] if "particle_label" in data else None
    return {
        "detector": {
            "x": detector_x,
            "context": detector["context"].to(dtype=torch.float32),
            "pos": detector["pos"].to(dtype=torch.float32),
            "lid": detector["lid"].to(dtype=torch.long),
            "waveform": detector["waveform"].to(dtype=torch.float32),
            "waveform_valid": detector["waveform_valid"].to(dtype=torch.float32)
            if "waveform_valid" in detector
            else torch.ones(detector_x.shape[0], dtype=torch.float32, device=detector_x.device),
            "batch": _storage_batch(detector, detector_x.shape[0], device=detector_x.device),
        },
        "pulse": {
            "x": pulse_x,
            "pos": pulse["pos"].to(dtype=torch.float32),
            "lid": pulse["lid"].to(dtype=torch.long),
            "detector_index": pulse["detector_index"].to(dtype=torch.long),
            "pulse_bounds": pulse["pulse_bounds"].to(dtype=torch.float32),
            "batch": _storage_batch(pulse, pulse_x.shape[0], device=pulse_x.device),
        },
        "edge_index_by_type": edge_index_by_type,
        "edge_features_by_type": edge_features_by_type,
        "target": target.to(dtype=torch.float32) if target is not None else None,
        "particle_label": particle_label.to(dtype=torch.float32) if particle_label is not None else None,
        "metadata": data["metadata"] if "metadata" in data else {},
        "num_graphs": int(getattr(data, "num_graphs", 1)),
    }
