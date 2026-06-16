from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EDGE_FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS, TARGET_COLUMNS, WAVEFORM_FEATURE_CHANNELS
from .dataset import H5GraphDataset, StandardScaler
from .diagnostics import FIGSIZE_SINGLE, FIGSIZE_STACKED, LINEWIDTH_THIN, _prepare_matplotlib, _save_pdf, _style_axes
from .metrics import binary_classification_metrics, energy_particle_bias_metrics, reconstruction_metrics
from .model import build_model_from_config
from .progress import progress as _progress
from .progress import write as _progress_write
from .train import _make_graph_loader, _predict_numpy, resolve_device


def expand_graph_paths(paths: str | Path | Sequence[str | Path]) -> list[str]:
    raw_paths = [paths] if isinstance(paths, str | Path) else list(paths)
    expanded: list[str] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            expanded.extend(str(match) for match in sorted(path.glob("*.h5")))
        elif path.exists():
            expanded.append(str(path))
        elif path.suffix == ".h5":
            expanded.extend(str(match) for match in sorted(path.parent.glob(f"{path.stem}_*.h5")))
        elif not path.suffix:
            expanded.extend(str(match) for match in sorted(path.parent.glob(f"{path.name}_*.h5")))
        else:
            expanded.append(str(path))
    return list(dict.fromkeys(expanded))


def _columns_from_dataset(dataset: H5GraphDataset) -> dict[str, list[str]]:
    try:
        columns = json.loads(dataset.columns_json)
    except json.JSONDecodeError:
        columns = {}
    if not isinstance(columns, dict):
        columns = {}
    return {
        "node_features": [str(value) for value in columns.get("node_features", dataset.node_feature_columns)],
        "edge_features": [str(value) for value in columns.get("edge_features", EDGE_FEATURE_COLUMNS)],
        "pulse_features": [str(value) for value in columns.get("pulse_features", dataset.pulse_feature_columns or PULSE_FEATURE_COLUMNS)],
        "waveform_features": [str(value) for value in columns.get("waveform_features", WAVEFORM_FEATURE_CHANNELS)],
        "target": [str(value) for value in columns.get("target", TARGET_COLUMNS)],
    }


def _sample_indices(n_items: int, max_items: int, seed: int) -> list[int]:
    if max_items <= 0 or n_items <= max_items:
        return list(range(n_items))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_items), max_items))


def _reservoir_merge(current: np.ndarray | None, values: np.ndarray, *, cap: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.empty((0,), dtype=np.float64) if current is None else current
    if current is None or current.size == 0:
        if values.size > cap:
            values = values[rng.choice(values.size, size=cap, replace=False)]
        return values.astype(np.float64, copy=False)
    merged = np.concatenate([current, values])
    if merged.size > cap:
        merged = merged[rng.choice(merged.size, size=cap, replace=False)]
    return merged.astype(np.float64, copy=False)


def _iter_sample_arrays(prefix: tuple[str, ...], value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_sample_arrays((*prefix, str(key)), child)
        return
    if value is None:
        return
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    yield prefix, array


def _write_sample_values_artifacts(
    samples: Mapping[str, Any],
    output_dir: Path,
    *,
    stem: str,
    extra_arrays: Mapping[tuple[str, ...], Any] | None = None,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    manifest_arrays: list[dict[str, Any]] = []
    for index, (path, values) in enumerate(_iter_sample_arrays((), samples)):
        key = f"arr_{index:04d}"
        arrays[key] = values.astype(np.float64, copy=False)
        manifest_arrays.append(
            {
                "key": key,
                "path": list(path),
                "n": int(values.size),
                "dtype": "float64",
            }
        )
    if extra_arrays:
        for path, values in extra_arrays.items():
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            array = array[np.isfinite(array)]
            key = f"arr_{len(arrays):04d}"
            arrays[key] = array.astype(np.float64, copy=False)
            manifest_arrays.append(
                {
                    "key": key,
                    "path": [str(part) for part in path],
                    "n": int(array.size),
                    "dtype": "float64",
                }
            )

    npz_path = output_dir / f"{stem}_sample_values.npz"
    manifest_path = output_dir / f"{stem}_sample_values_manifest.json"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "format": "sample_values_npz_v1",
        "npz": str(npz_path),
        "arrays": manifest_arrays,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"sample_values_npz": str(npz_path), "sample_values_manifest": str(manifest_path)}


def _summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    quantiles = np.percentile(values, [1, 5, 16, 50, 84, 95, 99])
    rounded = np.round(values, 6)
    unique_values, counts = np.unique(rounded, return_counts=True)
    dominant_index = int(np.argmax(counts))
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p16": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p84": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(np.max(values)),
        "dominant_value_6dp": float(unique_values[dominant_index]),
        "dominant_fraction_6dp": float(counts[dominant_index] / values.size),
        "n_unique_6dp": int(unique_values.size),
    }


def _escape_tex(text: str) -> str:
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _plot_feature_group(samples: dict[str, np.ndarray], output: Path, title: str) -> None:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    items = [(name, values[np.isfinite(values)]) for name, values in samples.items() if np.any(np.isfinite(values))]
    if not items:
        return
    ncols = 4
    nrows = int(math.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, values) in zip(axes.ravel(), items):
        ax.axis("on")
        lo, hi = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(values)), float(np.max(values))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        ax.hist(values, bins=np.linspace(lo, hi, 50), histtype="stepfilled", alpha=0.35)
        ax.set_title(_escape_tex(name), fontsize=9)
        ax.set_ylabel("events")
    fig.suptitle(_escape_tex(title))
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_input_distributions(
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
        _progress_write(f"stage=start input_distributions paths={len(paths)}")
        _progress_write("stage=start input_distributions dataset_init")
    dataset = H5GraphDataset(
        paths,
        require_target=False,
        load_attrs=False,
        load_node_positions=False,
        load_particle_label=True,
        show_progress=show_progress,
    )
    if show_progress:
        _progress_write(
            f"stage=done input_distributions dataset_init graphs={len(dataset)} shards={len(dataset.paths)}"
        )
    columns = _columns_from_dataset(dataset)
    indices = _sample_indices(len(dataset), int(max_graphs), int(seed))
    if show_progress:
        _progress_write(
            f"stage=start input_distributions collect sampled_graphs={len(indices)} total_graphs={len(dataset)} "
            f"max_values_per_feature={int(max_values_per_feature)}"
        )
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, np.ndarray | None]] = {
        "node": {name: None for name in columns["node_features"]},
        "edge": {name: None for name in columns["edge_features"]},
        "pulse": {name: None for name in columns["pulse_features"] if name != "node_index"},
        "waveform": {name: None for name in columns["waveform_features"]},
        "target": {name: None for name in columns["target"]},
    }
    particle_labels: list[float] = []

    for index in _progress(indices, desc="collect input distributions", total=len(indices), enabled=show_progress):
        sample = dataset[index]
        node = np.asarray(sample["node_features"], dtype=np.float64)
        for col, name in enumerate(columns["node_features"][: node.shape[1]]):
            samples["node"][name] = _reservoir_merge(samples["node"][name], node[:, col], cap=max_values_per_feature, rng=rng)
        edge = np.asarray(sample["edge_features"], dtype=np.float64)
        for col, name in enumerate(columns["edge_features"][: edge.shape[1]]):
            samples["edge"][name] = _reservoir_merge(samples["edge"][name], edge[:, col], cap=max_values_per_feature, rng=rng)
        pulse = np.asarray(sample["pulse_features"], dtype=np.float64)
        for col, name in enumerate(columns["pulse_features"][: pulse.shape[1]]):
            if name == "node_index":
                continue
            samples["pulse"][name] = _reservoir_merge(samples["pulse"][name], pulse[:, col], cap=max_values_per_feature, rng=rng)
        waveform = np.asarray(sample["waveform_features"], dtype=np.float64)
        if waveform.ndim == 3:
            for col, name in enumerate(columns["waveform_features"][: waveform.shape[1]]):
                samples["waveform"][name] = _reservoir_merge(
                    samples["waveform"][name],
                    waveform[:, col, :].reshape(-1),
                    cap=max_values_per_feature,
                    rng=rng,
                )
        target = sample.get("target")
        if target is not None:
            target = np.asarray(target, dtype=np.float64)
            for col, name in enumerate(columns["target"][: target.shape[0]]):
                samples["target"][name] = _reservoir_merge(samples["target"][name], target[col : col + 1], cap=max_values_per_feature, rng=rng)
        label = sample.get("particle_label")
        if label is not None and np.isfinite(float(label)):
            particle_labels.append(float(label))
    if show_progress:
        _progress_write("stage=done input_distributions collect")

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if show_progress:
        _progress_write(f"stage=start input_distributions summarize output={output}")
    summary = {
        "graphs": paths,
        "n_graphs_total": len(dataset),
        "n_graphs_sampled": len(indices),
        "columns": columns,
        "features": {
            group: {name: _summarize_values(values if values is not None else np.empty((0,))) for name, values in group_samples.items()}
            for group, group_samples in samples.items()
        },
        "particle_labels": {
            "n": len(particle_labels),
            "proton": int(np.sum(np.asarray(particle_labels) < 0.5)) if particle_labels else 0,
            "iron": int(np.sum(np.asarray(particle_labels) >= 0.5)) if particle_labels else 0,
        },
    }
    summary_path = output / "input_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    artifacts = _write_sample_values_artifacts(
        samples,
        output,
        stem="input_feature",
        extra_arrays={("particle_labels", "label"): np.asarray(particle_labels, dtype=np.float64)},
    )
    summary["redraw_artifacts"] = artifacts
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if show_progress:
        _progress_write(f"stage=done input_distributions summary_json={summary_path}")
    for group, group_samples in samples.items():
        if show_progress:
            _progress_write(f"stage=start input_distributions plot group={group}")
        _plot_feature_group(
            {name: values for name, values in group_samples.items() if values is not None},
            output / f"{group}_features.pdf",
            f"{group} feature distributions",
        )
        if show_progress:
            _progress_write(f"stage=done input_distributions plot group={group}")
    dataset.close()
    summary["summary_json"] = str(summary_path)
    if show_progress:
        _progress_write(f"stage=done input_distributions output={output}")
    return summary


def default_feature_groups(columns: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    groups = {
        "node_geometry": {
            "node": [
                "x_km",
                "y_km",
                "z_km",
                "nearest_detector_distance_km",
                "mean3_detector_distance_km",
                "neighbor_count_1p5km",
                "dx_from_signal_bary_km",
                "dy_from_signal_bary_km",
                "dz_from_signal_bary_km",
                "r_from_signal_bary_km",
            ]
        },
        "node_timing": {"node": ["pulse_arrival_usec_rel", "detector_trigger_usec_rel"]},
        "node_signal": {
            "node": [
                "log10_pulse_rho",
                "sqrt_pulse_rho",
                "log10_detector_max_pulse_rho",
                "log10_detector_sum_pulse_rho",
                "sqrt_detector_sum_pulse_rho",
                "detector_accepted_pulse_count",
                "detector_accepted_pulse_time_span_usec",
            ]
        },
        "node_waveform_context": {"node": ["detector_wf_segments", "detector_wf_length_usec", "log10_detector_fadc_peak"]},
        "node_pedestal": {
            "node": [
                "detector_upper_ped",
                "detector_lower_ped",
                "detector_upper_ped_sigma",
                "detector_lower_ped_sigma",
            ]
        },
        "node_order": {"node": ["accepted_pulse_order", "is_first_accepted_pulse"]},
        "edge_geometry": {"edge": ["dx_km", "dy_km", "dz_km", "distance_km"]},
        "edge_timing": {"edge": ["dt_usec", "abs_dt_usec", "dt_per_km"]},
        "edge_signal": {"edge": ["dlog10_pulse_rho"]},
        "edge_ising": {"edge": ["ising_weight", "ising_weight_raw", "ising_causal_excess_usec", "ising_spatial", "ising_causal"]},
        "waveform": {"waveform": list(columns["waveform_features"])},
    }
    pulse_inputs = [name for name in columns["pulse_features"] if name != "node_index"]
    if pulse_inputs:
        groups["pulse_features"] = {"pulse": pulse_inputs}
    return groups


class _FeatureAblationDataset:
    def __init__(
        self,
        dataset: H5GraphDataset,
        *,
        columns: dict[str, list[str]],
        scalers: dict[str, StandardScaler],
        group: dict[str, list[str]],
    ) -> None:
        self.dataset = dataset
        self.columns = columns
        self.scalers = scalers
        self.group = group

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        if "node" in self.group and "node" in self.scalers:
            node = np.array(sample["node_features"], copy=True)
            for name in self.group["node"]:
                if name in self.columns["node_features"]:
                    col = self.columns["node_features"].index(name)
                    if col < node.shape[1]:
                        node[:, col] = self.scalers["node"].mean[col]
            sample["node_features"] = node
        if "edge" in self.group and "edge" in self.scalers:
            edge = np.array(sample["edge_features"], copy=True)
            for name in self.group["edge"]:
                if name in self.columns["edge_features"]:
                    col = self.columns["edge_features"].index(name)
                    if col < edge.shape[1]:
                        edge[:, col] = self.scalers["edge"].mean[col]
            sample["edge_features"] = edge
        if "pulse" in self.group and "pulse" in self.scalers:
            pulse = np.array(sample["pulse_features"], copy=True)
            for name in self.group["pulse"]:
                if name in self.columns["pulse_features"]:
                    col = self.columns["pulse_features"].index(name)
                    scaler_col = col - 1
                    if col < pulse.shape[1] and scaler_col >= 0 and scaler_col < self.scalers["pulse"].mean.shape[0]:
                        pulse[:, col] = self.scalers["pulse"].mean[scaler_col]
            sample["pulse_features"] = pulse
        if "waveform" in self.group:
            sample["waveform_features"] = np.zeros_like(sample["waveform_features"])
        return sample


def _selected_checkpoint_indices(checkpoint: dict[str, Any], split: str) -> list[int]:
    key = {"validation": "val_indices", "val": "val_indices", "test": "test_indices", "train": "train_indices"}[split]
    return [int(value) for value in np.asarray(checkpoint[key]).reshape(-1)]


def _metric_delta(baseline: dict[str, Any] | None, changed: dict[str, Any] | None) -> dict[str, float]:
    if baseline is None or changed is None:
        return {}
    delta = {}
    skip_keys = {
        "energy_particle_bias_n_events",
        "energy_particle_bias_n_bins",
        "energy_particle_bias_bin_width",
        "energy_particle_bias_min_bin_count",
    }
    for key in sorted(set(baseline) & set(changed)):
        if key in skip_keys:
            continue
        try:
            baseline_value = float(baseline[key])
            changed_value = float(changed[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(baseline_value) and np.isfinite(changed_value):
            delta[key] = changed_value - baseline_value
    return delta


def reconstruction_metrics_with_particle_bias(
    pred: np.ndarray,
    target: np.ndarray,
    particle_labels: np.ndarray | None,
    *,
    energy_bin_width: float = 0.1,
    min_bin_count: int = 8,
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(reconstruction_metrics(pred, target))
    if particle_labels is not None:
        metrics.update(
            energy_particle_bias_metrics(
                pred,
                target,
                particle_labels,
                bin_width=energy_bin_width,
                min_bin_count=min_bin_count,
            )
        )
    return metrics


def _feature_importance_plot_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    reconstruction_labels = {
        "rmse_log10_energy": "energy RMSE loss",
        "energy_particle_bias_abs_mean_log10": "p/Fe energy bias gap loss",
        "angular_68_deg": "angle 68% loss",
        "core_68_km": "core 68% loss",
    }
    for metric in ("rmse_log10_energy", "energy_particle_bias_abs_mean_log10", "angular_68_deg", "core_68_km"):
        if any(metric in row.get("reconstruction_delta", {}) for row in rows):
            specs.append(
                {
                    "section": "reconstruction_delta",
                    "metric": metric,
                    "label": reconstruction_labels[metric],
                    "display_name": f"delta_{metric}",
                    "display_transform": "ablated_minus_baseline",
                    "sign": 1.0,
                    "interpretation": "positive means ablation worsens this error metric; negative means ablation improves it",
                }
            )
    if any("balanced_accuracy" in row.get("mass_delta", {}) for row in rows):
        specs.append(
            {
                "section": "mass_delta",
                "metric": "balanced_accuracy",
                "label": "mass accuracy loss",
                "display_name": "balanced_accuracy_drop",
                "display_transform": "baseline_minus_ablated",
                "sign": -1.0,
                "interpretation": "positive means ablation lowers balanced accuracy; negative means ablation improves it",
            }
        )
    return specs


def _finite_positive_metric(section: Mapping[str, Any] | None, metric: str) -> float | None:
    if section is None:
        return None
    try:
        value = float(section[metric])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _relative_performance_change(
    spec: Mapping[str, Any],
    baseline: Mapping[str, Any],
    row: Mapping[str, Any],
) -> float | None:
    metric = str(spec["metric"])
    section = str(spec["section"])
    if section == "reconstruction_delta":
        baseline_value = _finite_positive_metric(baseline.get("reconstruction"), metric)
        ablated_value = _finite_positive_metric(row.get("reconstruction"), metric)
        if baseline_value is None or ablated_value is None:
            return None
        return float(ablated_value / baseline_value - 1.0)
    if section == "mass_delta":
        baseline_value = _finite_positive_metric(baseline.get("mass"), metric)
        ablated_value = _finite_positive_metric(row.get("mass"), metric)
        if baseline_value is None or ablated_value is None:
            return None
        return float(baseline_value / ablated_value - 1.0)
    return None


def _feature_group_importance_plot_data(result: dict[str, Any]) -> dict[str, Any]:
    rows = list(result.get("groups", []))
    plot_specs = _feature_importance_plot_specs(rows)
    baseline = result.get("baseline", {})
    return {
        "format": "feature_group_importance_plot_data_v1",
        "split": result.get("split"),
        "n_graphs": result.get("n_graphs"),
        "display_convention": "Positive values mean ablation worsens performance. Negative values mean ablation improves that metric.",
        "relative_display_convention": (
            "Baseline performance is 0.0. Positive values mean ablation worsens performance; "
            "negative values mean ablation improves performance. Error metrics use ablated/baseline - 1; "
            "mass balanced accuracy uses baseline/ablated - 1."
        ),
        "groups": [str(row.get("group", "")) for row in rows],
        "plot_specs": [
            {
                **spec,
                "values": [
                    (
                        float(spec["sign"]) * float(row.get(spec["section"], {}).get(spec["metric"]))
                        if row.get(spec["section"], {}).get(spec["metric"]) is not None
                        else None
                    )
                    for row in rows
                ],
            }
            for spec in plot_specs
        ],
        "relative_plot_specs": [
            {
                **spec,
                "display_name": f"relative_{spec['display_name']}",
                "display_transform": (
                    "ablated_over_baseline_minus_one"
                    if spec["section"] == "reconstruction_delta"
                    else "baseline_over_ablated_minus_one"
                ),
                "baseline_value": 0.0,
                "unit": "dimensionless_fraction",
                "values": [_relative_performance_change(spec, baseline, row) for row in rows],
            }
            for spec in plot_specs
        ],
        "baseline": baseline,
        "rows": rows,
    }


def _write_feature_group_importance_plot_data(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    data = _feature_group_importance_plot_data(result)
    path = output_dir / "feature_group_importance_plot_data.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return {
        "plot_data_json": str(path),
        "feature_group_importance_pdf": str(output_dir / "feature_group_importance.pdf"),
        "feature_group_importance_relative_pdf": str(output_dir / "feature_group_importance_relative.pdf"),
    }


def save_feature_group_importance(
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

    device = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device, weights_only=False)
    scalers = {name: StandardScaler.from_dict(data) for name, data in checkpoint["scalers"].items()}
    model_config = dict(checkpoint["model_config"])
    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    runtime = dict(checkpoint.get("runtime", {}))
    mass_classification = int(model_config.get("classification_dim", 0)) > 0
    quality_prediction = int(model_config.get("quality_dim", 0)) > 0
    error_prediction = int(model_config.get("error_dim", 0)) > 0
    target_dim = int(model_config.get("target_dim", 7))
    load_detector_lids = int(model_config.get("detector_embedding_dim", 0)) > 0
    dataset = H5GraphDataset(
        expand_graph_paths(graphs_path),
        require_target=True,
        load_attrs=False,
        load_node_positions=False,
        load_particle_label=mass_classification,
        load_detector_lids=load_detector_lids,
    )
    columns = _columns_from_dataset(dataset)
    indices = _selected_checkpoint_indices(checkpoint, split)
    if max_graphs > 0 and len(indices) > max_graphs:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_graphs))

    def predict_for(ds: Any, desc: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        loader = _make_graph_loader(
            ds,
            indices,
            scalers=scalers,
            batch_size=batch_size,
            shuffle=False,
            require_target=True,
            num_workers=0,
            prefetch_factor=2,
            seed=seed,
            pin_memory=device.startswith("cuda"),
            persistent_workers=False,
            collate_backend="cpp",
            collate_threads=1,
        )
        pred, target, mass_logit, mass_label, _quality, _errors = _predict_numpy(
            model,
            loader,
            scalers,
            device,
            non_blocking=device.startswith("cuda"),
            desc=desc,
            show_progress=show_progress,
            mass_classification=mass_classification,
            quality_prediction=quality_prediction,
            error_prediction=error_prediction,
            target_dim=target_dim,
            error_angular_scale_deg=float(runtime.get("error_angular_scale_deg", runtime.get("quality_angular_scale_deg", 1.0))),
            error_core_scale_km=float(runtime.get("error_core_scale_km", runtime.get("quality_core_scale_km", 0.05))),
            error_energy_scale=float(runtime.get("error_energy_scale", runtime.get("quality_energy_scale", 0.10))),
        )
        reco = (
            None
            if target_dim == 0
            else reconstruction_metrics_with_particle_bias(
                pred,
                target,
                mass_label if mass_label is not None else None,
                energy_bin_width=float(runtime.get("energy_bias_bin_width", 0.1)),
                min_bin_count=int(runtime.get("energy_bias_min_bin_count", 8)),
            )
        )
        mass = (
            binary_classification_metrics(mass_logit, mass_label, threshold=0.5)
            if mass_classification and mass_logit is not None and mass_label is not None
            else None
        )
        return reco, mass

    baseline_reco, baseline_mass = predict_for(dataset, "feature importance baseline")
    groups = default_feature_groups(columns)
    rows = []
    for name, group in _progress(list(groups.items()), desc="feature group ablation", total=len(groups), enabled=show_progress):
        ablated = _FeatureAblationDataset(dataset, columns=columns, scalers=scalers, group=group)
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
    _plot_feature_group_importance_relative(result, output / "feature_group_importance_relative.pdf")
    dataset.close()
    result["summary_json"] = str(json_path)
    return result


def _plot_feature_group_importance(result: dict[str, Any], output: Path) -> None:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    rows = result.get("groups", [])
    if not rows:
        return
    names = [str(row["group"]) for row in rows]
    plot_specs = _feature_importance_plot_specs(list(rows))
    if not plot_specs:
        return

    figsize = FIGSIZE_SINGLE if len(plot_specs) == 1 else FIGSIZE_STACKED
    fig, axes = plt.subplots(len(plot_specs), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(names))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    for ax, spec in zip(axes, plot_specs, strict=True):
        section = str(spec["section"])
        metric = str(spec["metric"])
        values = [float(spec["sign"]) * float(row.get(section, {}).get(metric, np.nan)) for row in rows]
        ax.bar(x, values, color=colors[0], alpha=0.75)
        ax.axhline(0.0, color="0.25", linewidth=LINEWIDTH_THIN)
        ax.set_ylabel(_escape_tex(str(spec["label"])))
        _style_axes(ax)
    axes[0].set_title("feature group ablation: positive means ablation worsens performance")
    axes[-1].set_xticks(x, [_escape_tex(name) for name in names], rotation=35, ha="right")
    fig.tight_layout()
    _save_pdf(fig, output)


def _plot_feature_group_importance_relative(result: dict[str, Any], output: Path) -> None:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    rows = result.get("groups", [])
    if not rows:
        return
    names = [str(row["group"]) for row in rows]
    data = _feature_group_importance_plot_data(result)
    plot_specs = [spec for spec in data.get("relative_plot_specs", []) if any(value is not None for value in spec.get("values", []))]
    if not plot_specs:
        return

    figsize = FIGSIZE_SINGLE if len(plot_specs) == 1 else FIGSIZE_STACKED
    fig, axes = plt.subplots(len(plot_specs), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(names))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    for ax, spec in zip(axes, plot_specs, strict=True):
        values = [np.nan if value is None else float(value) for value in spec.get("values", [])]
        ax.bar(x, values, color=colors[1 % len(colors)], alpha=0.75)
        ax.axhline(0.0, color="0.25", linewidth=LINEWIDTH_THIN)
        ax.set_ylabel("relative change")
        ax.set_title(_escape_tex(f"{spec['label']} (baseline = 0)"), fontsize=10)
        _style_axes(ax)
    axes[0].figure.suptitle("feature group ablation: relative performance impact")
    axes[-1].set_xticks(x, [_escape_tex(name) for name in names], rotation=35, ha="right")
    fig.tight_layout()
    _save_pdf(fig, output)
