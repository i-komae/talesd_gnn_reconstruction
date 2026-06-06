from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import StandardScaler
from .feature_analysis import expand_graph_paths, _metric_delta, _plot_feature_group_importance
from .hetero_data import sample_to_hetero_data
from .hetero_graph_io import EDGE_RELATIONS, H5HeteroGraphDataset, H5PyGHeteroGraphDataset
from .hetero_model import MinimalHeteroTaleSdGNN
from .hetero_training import _predict_hetero_numpy
from .metrics import binary_classification_metrics, reconstruction_metrics
from .progress import progress as _progress
from .train import resolve_device


def _columns_from_hetero_dataset(dataset: H5HeteroGraphDataset) -> dict[str, Any]:
    try:
        columns = json.loads(dataset.columns_json)
    except json.JSONDecodeError:
        columns = {}
    if not isinstance(columns, dict):
        columns = {}
    edge_columns = columns.get("edge_features_by_type")
    if not isinstance(edge_columns, Mapping):
        edge_columns = {relation: columns.get("edge_features", []) for relation in EDGE_RELATIONS}
    return {
        "detector_features": [str(value) for value in columns.get("detector_features", [])],
        "detector_context_features": [str(value) for value in columns.get("detector_context_features", [])],
        "pulse_features": [str(value) for value in columns.get("pulse_features", columns.get("node_features", []))],
        "detector_waveforms": [str(value) for value in columns.get("detector_waveforms", [])],
        "edge_features_by_type": {
            str(relation): [str(value) for value in edge_columns.get(relation, columns.get("edge_features", []))]
            for relation in EDGE_RELATIONS
        },
    }


def _present(columns: Sequence[str], names: Sequence[str]) -> list[str]:
    column_set = set(columns)
    return [name for name in names if name in column_set]


def default_hetero_feature_groups(columns: dict[str, Any]) -> dict[str, dict[str, Any]]:
    detector = columns["detector_features"]
    context = columns["detector_context_features"]
    pulse = columns["pulse_features"]
    edge_by_type = columns["edge_features_by_type"]

    groups: dict[str, dict[str, Any]] = {
        "detector_signal": {
            "detector": _present(
                detector,
                [
                    "detector_trigger_usec_rel",
                    "log10_detector_max_pulse_rho",
                    "log10_detector_sum_pulse_rho",
                    "sqrt_detector_sum_pulse_rho",
                    "detector_accepted_pulse_count",
                    "detector_accepted_pulse_time_span_usec",
                ],
            )
        },
        "detector_geometry": {
            "detector": _present(
                detector,
                [
                    "nearest_detector_distance_km",
                    "mean3_detector_distance_km",
                    "neighbor_count_1p5km",
                    "local_detector_density_1p5km",
                ],
            )
        },
        "detector_readout_context": {
            "detector_context": _present(
                context,
                ["detector_wf_segments", "detector_wf_length_usec", "log10_detector_fadc_peak"],
            )
        },
        "detector_pedestal": {
            "detector_context": _present(
                context,
                [
                    "detector_upper_ped",
                    "detector_lower_ped",
                    "detector_upper_ped_sigma",
                    "detector_lower_ped_sigma",
                ],
            )
        },
        "pulse_core_geometry": {
            "pulse": _present(
                pulse,
                [
                    "dx_from_reference_core_km",
                    "dy_from_reference_core_km",
                    "dz_from_reference_core_km",
                    "r_from_reference_core_km",
                ],
            )
        },
        "pulse_timing_signal": {
            "pulse": _present(
                pulse,
                [
                    "pulse_arrival_usec_rel",
                    "log10_pulse_rho",
                    "sqrt_pulse_rho",
                    "accepted_pulse_order",
                    "is_first_accepted_pulse",
                ],
            )
        },
        "pulse_ising": {
            "pulse": _present(pulse, ["ising_keep", "ising_removed", "ising_spin", "ising_support"])
        },
        "detector_waveform": {"waveform": list(columns["detector_waveforms"])},
    }
    edge_feature_groups = {
        "edge_geometry": ["dx_km", "dy_km", "dz_km", "distance_km"],
        "edge_timing": ["dt_usec", "abs_dt_usec", "dt_per_km"],
        "edge_signal": ["dlog10_pulse_rho"],
        "edge_ising": [
            "ising_weight",
            "ising_weight_raw",
            "ising_causal_excess_usec",
            "ising_spatial",
            "ising_causal",
        ],
    }
    for group_name, feature_names in edge_feature_groups.items():
        group_edges = {
            relation: _present(edge_by_type.get(relation, []), feature_names)
            for relation in EDGE_RELATIONS
        }
        if any(group_edges.values()):
            groups[group_name] = {"edge": group_edges}
    return {name: group for name, group in groups.items() if any(bool(value) for value in group.values())}


class _HeteroFeatureAblationDataset:
    def __init__(
        self,
        dataset: H5HeteroGraphDataset,
        *,
        columns: dict[str, Any],
        scalers: dict[str, StandardScaler],
        group: dict[str, Any],
        waveform_length: int,
    ) -> None:
        self.dataset = dataset
        self.columns = columns
        self.scalers = scalers
        self.group = group
        self.waveform_length = int(waveform_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def _ablate_array(
        self,
        sample: dict[str, Any],
        sample_key: str,
        column_key: str,
        scaler_key: str,
        names: Sequence[str],
    ) -> None:
        if not names or scaler_key not in self.scalers:
            return
        columns = self.columns[column_key]
        values = np.array(sample[sample_key], copy=True)
        mean = np.asarray(self.scalers[scaler_key].mean, dtype=values.dtype)
        for name in names:
            if name in columns:
                col = int(columns.index(name))
                if col < values.shape[1] and col < mean.shape[0]:
                    values[:, col] = mean[col]
        sample[sample_key] = values

    def __getitem__(self, index: int):
        sample = dict(self.dataset[index])
        self._ablate_array(
            sample,
            "detector_features",
            "detector_features",
            "detector",
            self.group.get("detector", []),
        )
        self._ablate_array(
            sample,
            "detector_context_features",
            "detector_context_features",
            "detector_context",
            self.group.get("detector_context", []),
        )
        self._ablate_array(sample, "pulse_features", "pulse_features", "pulse", self.group.get("pulse", []))
        for relation, feature_names in (self.group.get("edge", {}) or {}).items():
            scaler_key = f"edge:{relation}"
            if scaler_key not in self.scalers:
                continue
            columns = self.columns["edge_features_by_type"].get(relation, [])
            edge_features = np.array(sample["edge_features_by_type"][relation], copy=True)
            mean = np.asarray(self.scalers[scaler_key].mean, dtype=edge_features.dtype)
            for name in feature_names:
                if name in columns:
                    col = int(columns.index(name))
                    if col < edge_features.shape[1] and col < mean.shape[0]:
                        edge_features[:, col] = mean[col]
            sample["edge_features_by_type"] = dict(sample["edge_features_by_type"])
            sample["edge_features_by_type"][relation] = edge_features
        if "waveform" in self.group:
            sample["detector_waveforms"] = np.zeros_like(sample["detector_waveforms"])
        return sample_to_hetero_data(sample, scalers=self.scalers, waveform_length=self.waveform_length)


def _selected_hetero_checkpoint_indices(checkpoint: dict[str, Any], split: str) -> list[int]:
    split_info = dict(checkpoint.get("split", {}))
    key = {"validation": "val_indices", "val": "val_indices", "test": "test_indices", "train": "train_indices"}[split]
    if key not in split_info:
        raise ValueError(f"checkpoint split has no {key}")
    return [int(value) for value in np.asarray(split_info[key]).reshape(-1)]


def _scalers_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, StandardScaler]:
    return {name: StandardScaler.from_dict(data) for name, data in checkpoint["hetero_scalers"].items()}


def save_hetero_feature_group_importance(
    graphs_path: str | Path | Sequence[str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    max_graphs: int = 50000,
    batch_size: int = 256,
    device: str = "auto",
    seed: int = 12345,
    show_progress: bool = True,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    device = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device, weights_only=False)
    scalers = _scalers_from_checkpoint(checkpoint)
    model_config = dict(checkpoint["model_config"])
    architecture = str(model_config.pop("architecture", ""))
    if architecture != "minimal_hetero":
        raise ValueError(f"checkpoint architecture {architecture!r} is not supported for hetero feature importance")
    model = MinimalHeteroTaleSdGNN(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    runtime = dict(checkpoint.get("runtime", {}))
    target_dim = int(model_config.get("target_dim", 6))
    mass_classification = int(model_config.get("classification_dim", 0)) > 0
    quality_prediction = int(model_config.get("quality_dim", 0)) > 0
    error_prediction = int(model_config.get("error_dim", 0)) > 0
    waveform_length = int(model_config["waveform_length"])

    dataset = H5HeteroGraphDataset(
        expand_graph_paths(graphs_path),
        require_target=True,
        require_particle_label=mass_classification,
        load_attrs=False,
    )
    columns = _columns_from_hetero_dataset(dataset)
    indices = _selected_hetero_checkpoint_indices(checkpoint, split)
    if max_graphs > 0 and len(indices) > max_graphs:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_graphs))

    def predict_for(ds: Any, desc: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        loader = DataLoader(Subset(ds, indices), batch_size=max(int(batch_size), 1), shuffle=False)
        pred, target, mass_logit, mass_label, _quality, _errors = _predict_hetero_numpy(
            model,
            loader,
            scalers,
            device,
            target_dim=target_dim,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            error_angular_scale_deg=float(runtime.get("error_angular_scale_deg", runtime.get("quality_angular_scale_deg", 1.0))),
            error_core_scale_km=float(runtime.get("error_core_scale_km", runtime.get("quality_core_scale_km", 0.05))),
            error_energy_scale=float(runtime.get("error_energy_scale", runtime.get("quality_energy_scale", 0.10))),
            desc=desc,
            show_progress=show_progress,
        )
        reco = None if target_dim == 0 else reconstruction_metrics(pred, target)
        mass = (
            binary_classification_metrics(mass_logit, mass_label, threshold=0.5)
            if mass_classification and mass_logit is not None and mass_label is not None
            else None
        )
        return reco, mass

    base_dataset = H5PyGHeteroGraphDataset(
        expand_graph_paths(graphs_path),
        require_target=True,
        require_particle_label=mass_classification,
        scalers=scalers,
        waveform_length=waveform_length,
        load_attrs=False,
    )
    baseline_reco, baseline_mass = predict_for(base_dataset, "hetero feature importance baseline")
    groups = default_hetero_feature_groups(columns)
    rows = []
    for name, group in _progress(list(groups.items()), desc="hetero feature group ablation", total=len(groups), enabled=show_progress):
        ablated = _HeteroFeatureAblationDataset(
            dataset,
            columns=columns,
            scalers=scalers,
            group=group,
            waveform_length=waveform_length,
        )
        reco, mass = predict_for(ablated, f"ablate {name}")
        row: dict[str, Any] = {
            "group": name,
            "features": group,
            "reconstruction": reco,
            "reconstruction_delta": _metric_delta(baseline_reco, reco),
            "mass": mass,
        }
        if baseline_mass is not None and mass is not None:
            row["mass_delta"] = {
                "accuracy": float(mass["accuracy"]) - float(baseline_mass["accuracy"]),
                "balanced_accuracy": float(mass["balanced_accuracy"]) - float(baseline_mass["balanced_accuracy"]),
            }
        rows.append(row)

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "graphs": expand_graph_paths(graphs_path),
        "split": split,
        "n_graphs": len(indices),
        "columns": columns,
        "baseline": {"reconstruction": baseline_reco, "mass": baseline_mass},
        "groups": rows,
    }
    json_path = output / "feature_group_importance.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    _plot_feature_group_importance(result, output / "feature_group_importance.pdf")
    dataset.close()
    base_dataset.close()
    result["summary_json"] = str(json_path)
    return result
