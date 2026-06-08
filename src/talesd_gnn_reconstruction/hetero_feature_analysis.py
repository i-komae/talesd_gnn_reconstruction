from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import StandardScaler
from .feature_analysis import (
    expand_graph_paths,
    _metric_delta,
    _plot_feature_group,
    _plot_feature_group_importance,
    _reservoir_merge,
    _sample_indices,
    _summarize_values,
    _write_feature_group_importance_plot_data,
    _write_sample_values_artifacts,
)
from .hetero_data import sample_to_hetero_data
from .hetero_graph_io import EDGE_RELATIONS, H5HeteroGraphDataset, H5PyGHeteroGraphDataset
from .hetero_model import MinimalHeteroTaleSdGNN
from .hetero_training import _predict_hetero_numpy
from .metrics import binary_classification_metrics, reconstruction_metrics
from .progress import progress as _progress
from .progress import write as _progress_write
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
        "pulse_bounds": [str(value) for value in columns.get("pulse_bounds", [])],
        "detector_waveforms": [str(value) for value in columns.get("detector_waveforms", [])],
        "edge_features_by_type": {
            str(relation): [str(value) for value in edge_columns.get(relation, columns.get("edge_features", []))]
            for relation in EDGE_RELATIONS
        },
        "target": [str(value) for value in columns.get("target", [])],
    }


def _merge_feature_columns(
    samples: dict[str, np.ndarray | None],
    values: np.ndarray,
    columns: Sequence[str],
    *,
    cap: int,
    rng: np.random.Generator,
) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    for col, name in enumerate(columns[: array.shape[1]]):
        samples[name] = _reservoir_merge(samples.get(name), array[:, col], cap=cap, rng=rng)


def save_hetero_input_distributions(
    graphs_path: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    max_graphs: int = 100000,
    max_values_per_feature: int = 200000,
    seed: int = 12345,
    show_progress: bool = True,
) -> dict[str, Any]:
    paths = expand_graph_paths(graphs_path)
    if show_progress:
        _progress_write(f"stage=start hetero_input_distributions paths={len(paths)}")
        _progress_write("stage=start hetero_input_distributions dataset_init")
    dataset = H5HeteroGraphDataset(paths, require_target=False, require_particle_label=False, load_attrs=False)
    try:
        if show_progress:
            _progress_write(
                f"stage=done hetero_input_distributions dataset_init graphs={len(dataset)} shards={len(dataset.paths)}"
            )
        columns = _columns_from_hetero_dataset(dataset)
        if not columns["pulse_bounds"]:
            columns["pulse_bounds"] = ["upper_rise_bin", "upper_fall_bin", "lower_rise_bin", "lower_fall_bin"]
        if not columns["detector_waveforms"]:
            columns["detector_waveforms"] = ["upper_raw_vem", "lower_raw_vem"]
        indices = _sample_indices(len(dataset), int(max_graphs), int(seed))
        if show_progress:
            _progress_write(
                "stage=start hetero_input_distributions collect "
                f"sampled_graphs={len(indices)} total_graphs={len(dataset)} "
                f"max_values_per_feature={int(max_values_per_feature)}"
            )
        rng = np.random.default_rng(seed)
        samples: dict[str, dict[str, np.ndarray | None]] = {
            "event": {
                "n_detector_nodes": None,
                "n_pulse_nodes": None,
                "detector_waveform_length": None,
            },
            "detector": {name: None for name in columns["detector_features"]},
            "detector_context": {name: None for name in columns["detector_context_features"]},
            "pulse": {name: None for name in columns["pulse_features"]},
            "pulse_bounds": {name: None for name in columns["pulse_bounds"]},
            "waveform": {name: None for name in columns["detector_waveforms"]},
            "target": {name: None for name in columns["target"]},
        }
        edge_samples: dict[str, dict[str, np.ndarray | None]] = {
            relation: {name: None for name in columns["edge_features_by_type"].get(relation, [])}
            for relation in EDGE_RELATIONS
        }
        particle_labels: list[float] = []

        for index in _progress(indices, desc="collect hetero input distributions", total=len(indices), enabled=show_progress):
            sample = dataset[int(index)]
            detector_features = np.asarray(sample["detector_features"], dtype=np.float64)
            detector_context = np.asarray(sample["detector_context_features"], dtype=np.float64)
            pulse_features = np.asarray(sample["pulse_features"], dtype=np.float64)
            pulse_bounds = np.asarray(sample["pulse_bounds"], dtype=np.float64)
            waveforms = np.asarray(sample["detector_waveforms"], dtype=np.float64)
            _merge_feature_columns(
                samples["detector"],
                detector_features,
                columns["detector_features"],
                cap=max_values_per_feature,
                rng=rng,
            )
            _merge_feature_columns(
                samples["detector_context"],
                detector_context,
                columns["detector_context_features"],
                cap=max_values_per_feature,
                rng=rng,
            )
            _merge_feature_columns(
                samples["pulse"],
                pulse_features,
                columns["pulse_features"],
                cap=max_values_per_feature,
                rng=rng,
            )
            _merge_feature_columns(
                samples["pulse_bounds"],
                pulse_bounds,
                columns["pulse_bounds"],
                cap=max_values_per_feature,
                rng=rng,
            )
            samples["event"]["n_detector_nodes"] = _reservoir_merge(
                samples["event"]["n_detector_nodes"],
                np.asarray([detector_features.shape[0]], dtype=np.float64),
                cap=max_values_per_feature,
                rng=rng,
            )
            samples["event"]["n_pulse_nodes"] = _reservoir_merge(
                samples["event"]["n_pulse_nodes"],
                np.asarray([pulse_features.shape[0]], dtype=np.float64),
                cap=max_values_per_feature,
                rng=rng,
            )
            samples["event"]["detector_waveform_length"] = _reservoir_merge(
                samples["event"]["detector_waveform_length"],
                np.asarray([waveforms.shape[2] if waveforms.ndim == 3 else 0], dtype=np.float64),
                cap=max_values_per_feature,
                rng=rng,
            )
            if waveforms.ndim == 3:
                for channel, name in enumerate(columns["detector_waveforms"][: waveforms.shape[1]]):
                    samples["waveform"][name] = _reservoir_merge(
                        samples["waveform"].get(name),
                        waveforms[:, channel, :].reshape(-1),
                        cap=max_values_per_feature,
                        rng=rng,
                    )
            edge_features_by_type = sample["edge_features_by_type"]
            for relation in EDGE_RELATIONS:
                edge_features = np.asarray(edge_features_by_type.get(relation, np.zeros((0, 0))), dtype=np.float64)
                _merge_feature_columns(
                    edge_samples[relation],
                    edge_features,
                    columns["edge_features_by_type"].get(relation, []),
                    cap=max_values_per_feature,
                    rng=rng,
                )
            target = sample.get("target")
            if target is not None:
                _merge_feature_columns(samples["target"], np.asarray(target, dtype=np.float64), columns["target"], cap=max_values_per_feature, rng=rng)
            label = sample.get("particle_label")
            if label is not None and np.isfinite(float(label)):
                particle_labels.append(float(label))
        if show_progress:
            _progress_write("stage=done hetero_input_distributions collect")

        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        features = {
            group: {
                name: _summarize_values(values if values is not None else np.empty((0,)))
                for name, values in group_samples.items()
            }
            for group, group_samples in samples.items()
        }
        features["edge_features_by_type"] = {
            relation: {
                name: _summarize_values(values if values is not None else np.empty((0,)))
                for name, values in relation_samples.items()
            }
            for relation, relation_samples in edge_samples.items()
        }
        summary = {
            "graph_format": "hetero",
            "graphs": paths,
            "n_graphs_total": len(dataset),
            "n_graphs_sampled": len(indices),
            "columns": columns,
            "features": features,
            "particle_labels": {
                "n": len(particle_labels),
                "proton": int(np.sum(np.asarray(particle_labels) < 0.5)) if particle_labels else 0,
                "iron": int(np.sum(np.asarray(particle_labels) >= 0.5)) if particle_labels else 0,
            },
        }
        summary_path = output / "input_feature_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        artifacts = _write_sample_values_artifacts(
            {
                **samples,
                "edge_features_by_type": edge_samples,
            },
            output,
            stem="input_feature",
            extra_arrays={("particle_labels", "label"): np.asarray(particle_labels, dtype=np.float64)},
        )
        summary["redraw_artifacts"] = artifacts
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        for group, group_samples in samples.items():
            _plot_feature_group(
                {name: values for name, values in group_samples.items() if values is not None},
                output / f"{group}_features.pdf",
                f"hetero {group} feature distributions",
            )
        for relation, relation_samples in edge_samples.items():
            _plot_feature_group(
                {name: values for name, values in relation_samples.items() if values is not None},
                output / f"edge_{relation}_features.pdf",
                f"hetero {relation} feature distributions",
            )
        summary["summary_json"] = str(summary_path)
        if show_progress:
            _progress_write(f"stage=done hetero_input_distributions summary_json={summary_path}")
        return summary
    finally:
        dataset.close()


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
    if architecture not in {"minimal_hetero", "hetero_attention"}:
        raise ValueError(f"checkpoint architecture {architecture!r} is not supported for hetero feature importance")
    model = MinimalHeteroTaleSdGNN(architecture=architecture, **model_config).to(device)
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
    result["redraw_artifacts"] = _write_feature_group_importance_plot_data(result, output)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    _plot_feature_group_importance(result, output / "feature_group_importance.pdf")
    dataset.close()
    base_dataset.close()
    result["summary_json"] = str(json_path)
    return result
