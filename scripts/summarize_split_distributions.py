#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.dataset import H5GraphDataset
from talesd_gnn_reconstruction.hetero_graph_io import FORMAT_NAME as HETERO_FORMAT_NAME
from talesd_gnn_reconstruction.hetero_graph_io import H5HeteroGraphDataset
from talesd_gnn_reconstruction.metrics import direction_columns_for_dim, direction_to_angles
from talesd_gnn_reconstruction.progress import progress
from talesd_gnn_reconstruction.progress import write as progress_write
from talesd_gnn_reconstruction.train import source_group_key, split_indices_by_stratified_source_path


GraphDataset = H5GraphDataset | H5HeteroGraphDataset


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "p16": None, "p84": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16.0)),
        "p84": float(np.percentile(arr, 84.0)),
    }


def _energy_bin(log10_energy: float, width: float) -> str:
    bin_width = float(width)
    center_index = math.floor(float(log10_energy) / bin_width + 0.5 + 1.0e-9)
    center = center_index * bin_width
    low = center - 0.5 * bin_width
    high = center + 0.5 * bin_width
    return f"{low:.2f}-{high:.2f}"


def _new_bucket() -> dict[str, Any]:
    return {
        "events": 0,
        "sources": set(),
        "particle_labels": [],
        "log10_energy": [],
        "core_x_km": [],
        "core_y_km": [],
        "core_radius_km": [],
        "zenith_deg": [],
        "azimuth_deg": [],
        "event_time_hour": [],
        "nodes": [],
        "detector_nodes": [],
        "pulse_nodes": [],
        "edges": [],
    }


def _hhmmss_to_hour(value: Any) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    time_value = int(number)
    hours = time_value // 10000
    minutes = (time_value // 100) % 100
    seconds = time_value % 100
    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59 or seconds < 0 or seconds > 59:
        return None
    return float(hours + minutes / 60.0 + seconds / 3600.0)


def _add(
    bucket: dict[str, Any],
    *,
    source_path: str,
    target: np.ndarray | None,
    particle_label: float | None,
    event_time: Any = None,
    n_nodes: int | None,
    n_edges: int | None,
    n_detector_nodes: int | None = None,
    n_pulse_nodes: int | None = None,
) -> None:
    bucket["events"] += 1
    bucket["sources"].add(source_path)
    if particle_label is not None and math.isfinite(float(particle_label)):
        bucket["particle_labels"].append(float(particle_label))
    event_hour = _hhmmss_to_hour(event_time)
    if event_hour is not None:
        bucket["event_time_hour"].append(event_hour)
    if n_nodes is not None:
        bucket["nodes"].append(float(n_nodes))
    if n_detector_nodes is not None:
        bucket["detector_nodes"].append(float(n_detector_nodes))
    if n_pulse_nodes is not None:
        bucket["pulse_nodes"].append(float(n_pulse_nodes))
    if n_edges is not None:
        bucket["edges"].append(float(n_edges))
    if target is None or target.shape[0] < 6 or not np.all(np.isfinite(target)):
        return
    log10_energy = float(target[0])
    core_x = float(target[1])
    core_y = float(target[2])
    bucket["log10_energy"].append(log10_energy)
    bucket["core_x_km"].append(core_x)
    bucket["core_y_km"].append(core_y)
    bucket["core_radius_km"].append(math.hypot(core_x, core_y))
    direction_slice = direction_columns_for_dim(target.shape[0])
    zenith, azimuth = direction_to_angles(target[None, direction_slice])
    bucket["zenith_deg"].append(float(zenith[0]))
    bucket["azimuth_deg"].append(float(azimuth[0]))


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(bucket["particle_labels"], dtype=np.float64)
    finite_labels = labels[np.isfinite(labels)]
    independent_showers = int(len(bucket["sources"]))
    return {
        "events": int(bucket["events"]),
        "sources": independent_showers,
        "independent_showers": independent_showers,
        "proton": int(np.sum(finite_labels < 0.5)),
        "iron": int(np.sum(finite_labels >= 0.5)),
        "unknown_particle": int(bucket["events"] - finite_labels.size),
        "iron_fraction": float(np.mean(finite_labels >= 0.5)) if finite_labels.size else None,
        "log10_energy": _stats(bucket["log10_energy"]),
        "core_x_km": _stats(bucket["core_x_km"]),
        "core_y_km": _stats(bucket["core_y_km"]),
        "core_radius_km": _stats(bucket["core_radius_km"]),
        "zenith_deg": _stats(bucket["zenith_deg"]),
        "azimuth_deg": _stats(bucket["azimuth_deg"]),
        "event_time_hour": _stats(bucket["event_time_hour"]),
        "nodes": _stats(bucket["nodes"]),
        "detector_nodes": _stats(bucket["detector_nodes"]),
        "pulse_nodes": _stats(bucket["pulse_nodes"]),
        "edges": _stats(bucket["edges"]),
    }


def _finite_array(values: list[float]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _hist_bins(arrays: list[np.ndarray], *, bins: int = 40) -> np.ndarray:
    combined = np.concatenate([arr for arr in arrays if arr.size]) if any(arr.size for arr in arrays) else np.empty((0,))
    if combined.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    low = float(np.min(combined))
    high = float(np.max(combined))
    if not math.isfinite(low) or not math.isfinite(high):
        return np.linspace(0.0, 1.0, bins + 1)
    if low == high:
        width = max(abs(low) * 0.05, 1.0)
        low -= width
        high += width
    return np.linspace(low, high, bins + 1)


def _plot_split_distributions(
    totals: dict[str, dict[str, Any]],
    by_energy: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    from talesd_gnn_reconstruction.diagnostics import (  # noqa: PLC0415
        FIGSIZE_GRID,
        FIGSIZE_PAIR,
        LINEWIDTH,
        MARKERSIZE,
        _prepare_matplotlib,
        _save_pdf,
        _style_axes,
    )

    _prepare_matplotlib()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    split_order = [name for name in ("train", "validation", "test") if name in totals]
    split_colors = dict(zip(split_order, plt.rcParams["axes.prop_cycle"].by_key().get("color", []), strict=False))
    pdf_files: list[str] = []
    plot_data: dict[str, Any] = {
        "format": "split_distribution_plot_data_v1",
        "split_order": split_order,
        "features": {},
        "energy_bin_counts": {},
        "count_definitions": {
            "independent_showers": "distinct source_group_key(source_path) values; DAT??????_gea_trg_XXX files are counted once per DAT?????? CORSIKA shower group",
            "sources": "backward-compatible alias for independent_showers in summary JSON",
        },
    }

    features = [
        ("log10_energy", r"$\log_{10}(E/\mathrm{eV})$"),
        ("core_x_km", "core x [km]"),
        ("core_y_km", "core y [km]"),
        ("zenith_deg", "zenith [deg]"),
        ("azimuth_deg", "azimuth [deg]"),
        ("detector_nodes", "detectors / event"),
        ("pulse_nodes", "pulses / event"),
        ("edges", "edges / event"),
        ("particle_labels", "particle label"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_GRID)
    for ax, (key, xlabel) in zip(axes.reshape(-1), features):
        arrays = [_finite_array(totals[name][key]) for name in split_order]
        if key == "particle_labels":
            bins = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
            ax.set_xticks([0.0, 1.0], ["p", "Fe"])
        else:
            bins = _hist_bins(arrays)
        feature_payload: dict[str, Any] = {
            "xlabel": xlabel,
            "bins": bins.astype(float).tolist(),
            "splits": {},
        }
        for name, arr in zip(split_order, arrays, strict=True):
            if arr.size == 0:
                feature_payload["splits"][name] = {"n": 0, "density": [], "counts": []}
                continue
            density, _ = np.histogram(arr, bins=bins, density=True)
            counts, _ = np.histogram(arr, bins=bins, density=False)
            feature_payload["splits"][name] = {
                "n": int(arr.size),
                "density": density.astype(float).tolist(),
                "counts": counts.astype(int).tolist(),
            }
            ax.hist(
                arr,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=LINEWIDTH,
                color=split_colors.get(name),
                label=f"{name} (n={arr.size})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        _style_axes(ax)
        plot_data["features"][key] = feature_payload
    for ax in axes.reshape(-1)[len(features) :]:
        ax.axis("off")
    axes.reshape(-1)[0].legend(frameon=False)
    fig.suptitle("Train/validation/test parameter distributions")
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, output_dir / "split_parameter_distributions.pdf"))

    energy_bins = sorted(by_energy, key=lambda item: float(item.split("-", maxsplit=1)[0]))
    if energy_bins:
        x = []
        labels = []
        for bin_key in energy_bins:
            low, high = bin_key.split("-", maxsplit=1)
            x.append((float(low) + float(high)) * 0.5)
            labels.append(bin_key)
        plot_data["energy_bin_counts"] = {
            "bin_keys": labels,
            "bin_centers": [float(value) for value in x],
            "splits": {name: {} for name in split_order},
        }
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
        for ax, value_key, ylabel in (
            (axes[0], "events", "events"),
            (axes[1], "independent_showers", "independent shower groups"),
        ):
            for name in split_order:
                y = [float(_finish_bucket(by_energy[bin_key][name])[value_key]) for bin_key in energy_bins]
                plot_data["energy_bin_counts"]["splits"][name][value_key] = [float(value) for value in y]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH,
                    color=split_colors.get(name),
                    label=name,
                )
            ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$ bin center")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x[:: max(len(x) // 8, 1)], labels[:: max(len(labels) // 8, 1)], rotation=45, ha="right")
            _style_axes(ax)
        axes[0].legend(frameon=False)
        fig.suptitle("Split counts by true-energy bin")
        fig.tight_layout()
        pdf_files.append(_save_pdf(fig, output_dir / "split_energy_bin_counts.pdf"))
    plot_data_path = output_dir / "split_distribution_plot_data.json"
    plot_data_path.write_text(json.dumps(plot_data, indent=2, sort_keys=True))
    return {"plot_files": pdf_files, "plot_data_json": str(plot_data_path)}


def redraw_split_distribution_plots(plot_data_json: Path, output_dir: Path | None = None) -> dict[str, Any]:
    from talesd_gnn_reconstruction.diagnostics import (  # noqa: PLC0415
        FIGSIZE_GRID,
        FIGSIZE_PAIR,
        LINEWIDTH,
        MARKERSIZE,
        _prepare_matplotlib,
        _save_pdf,
        _style_axes,
    )

    plot_data_json = Path(plot_data_json).expanduser()
    plot_data = json.loads(plot_data_json.read_text())
    if plot_data.get("format") != "split_distribution_plot_data_v1":
        raise ValueError(f"unsupported split plot data format: {plot_data.get('format')!r}")
    output_dir = Path(output_dir).expanduser() if output_dir is not None else plot_data_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _prepare_matplotlib()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    split_order = [str(name) for name in plot_data.get("split_order", [])]
    split_colors = dict(zip(split_order, plt.rcParams["axes.prop_cycle"].by_key().get("color", []), strict=False))
    pdf_files: list[str] = []

    feature_order = [
        "log10_energy",
        "core_x_km",
        "core_y_km",
        "zenith_deg",
        "azimuth_deg",
        "detector_nodes",
        "pulse_nodes",
        "edges",
        "particle_labels",
    ]
    features = plot_data.get("features", {})
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_GRID)
    for ax, key in zip(axes.reshape(-1), feature_order):
        payload = features.get(key)
        if payload is None:
            ax.axis("off")
            continue
        bins = np.asarray(payload.get("bins", []), dtype=np.float64)
        if bins.size < 2:
            ax.axis("off")
            continue
        if key == "particle_labels":
            ax.set_xticks([0.0, 1.0], ["p", "Fe"])
        for name in split_order:
            split_payload = payload.get("splits", {}).get(name, {})
            density = np.asarray(split_payload.get("density", []), dtype=np.float64)
            n = int(split_payload.get("n", 0))
            if density.size != bins.size - 1 or n <= 0:
                continue
            ax.stairs(
                density,
                bins,
                linewidth=LINEWIDTH,
                color=split_colors.get(name),
                label=f"{name} (n={n})",
            )
        ax.set_xlabel(str(payload.get("xlabel", key)))
        ax.set_ylabel("density")
        _style_axes(ax)
    axes.reshape(-1)[0].legend(frameon=False)
    fig.suptitle("Train/validation/test parameter distributions")
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, output_dir / "split_parameter_distributions.pdf"))

    energy_counts = plot_data.get("energy_bin_counts", {})
    bin_centers = np.asarray(energy_counts.get("bin_centers", []), dtype=np.float64)
    bin_keys = [str(key) for key in energy_counts.get("bin_keys", [])]
    if bin_centers.size:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
        for ax, value_key, ylabel in (
            (axes[0], "events", "events"),
            (axes[1], "independent_showers", "independent shower groups"),
        ):
            for name in split_order:
                y = np.asarray(
                    energy_counts.get("splits", {}).get(name, {}).get(value_key, []),
                    dtype=np.float64,
                )
                if y.size != bin_centers.size:
                    continue
                ax.plot(
                    bin_centers,
                    y,
                    marker="o",
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH,
                    color=split_colors.get(name),
                    label=name,
                )
            ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$ bin center")
            ax.set_ylabel(ylabel)
            ax.set_xticks(
                bin_centers[:: max(len(bin_centers) // 8, 1)],
                bin_keys[:: max(len(bin_keys) // 8, 1)],
                rotation=45,
                ha="right",
            )
            _style_axes(ax)
        axes[0].legend(frameon=False)
        fig.suptitle("Split counts by true-energy bin")
        fig.tight_layout()
        pdf_files.append(_save_pdf(fig, output_dir / "split_energy_bin_counts.pdf"))

    return {"plot_files": pdf_files, "plot_data_json": str(plot_data_json)}


def _shape_counts(dataset: GraphDataset, index: int) -> tuple[int | None, int | None, int | None, int | None]:
    path_index, _local_index, key = dataset._locate(index)  # noqa: SLF001 - summary script uses dataset internals for cheap shapes.
    group = dataset._handle(path_index)["events"][key]  # noqa: SLF001
    if isinstance(dataset, H5HeteroGraphDataset):
        n_detector_nodes = int(group["detector_features"].shape[0]) if "detector_features" in group else None
        n_pulse_nodes = int(group["pulse_features"].shape[0]) if "pulse_features" in group else None
        n_nodes = (
            (n_detector_nodes or 0) + (n_pulse_nodes or 0)
            if n_detector_nodes is not None or n_pulse_nodes is not None
            else None
        )
        n_edges = 0
        edge_group = group.get("edge_features_by_type")
        if edge_group is None:
            n_edges = None
        else:
            for relation in edge_group.keys():
                n_edges += int(edge_group[relation].shape[0])
        return n_nodes, n_edges, n_detector_nodes, n_pulse_nodes
    n_nodes = int(group["node_features"].shape[0]) if "node_features" in group else None
    n_edges = int(group["edge_features"].shape[0]) if "edge_features" in group else None
    return n_nodes, n_edges, None, n_nodes


def _event_attrs(dataset: GraphDataset, index: int) -> dict[str, Any]:
    path_index, _local_index, key = dataset._locate(index)  # noqa: SLF001 - summary script uses dataset internals for cheap attrs.
    group = dataset._handle(path_index)["events"][key]  # noqa: SLF001
    attrs = dict(group.attrs.items())
    metadata_json = attrs.get("metadata_json")
    if metadata_json is not None:
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode("utf-8", errors="replace")
        try:
            attrs.update(json.loads(str(metadata_json)))
        except json.JSONDecodeError:
            pass
    return attrs


def summarize(
    dataset: GraphDataset,
    *,
    val_fraction: float,
    test_fraction: float,
    source_val_fraction: float,
    source_test_fraction: float,
    seed: int,
    energy_bin_width: float,
    split_workers: int,
    show_progress: bool,
    plot_dir: Path | None = None,
) -> dict[str, Any]:
    progress_write(
        "stage=start split_distribution_summary split_assignment "
        f"graphs={len(dataset)} split_workers={split_workers}"
    )
    split = split_indices_by_stratified_source_path(
        dataset,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        source_val_fraction=source_val_fraction,
        source_test_fraction=source_test_fraction,
        seed=seed,
        show_progress=show_progress,
        workers=split_workers,
    )
    progress_write(
        "stage=done split_distribution_summary split_assignment "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )
    totals = {name: _new_bucket() for name in split}
    by_energy: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {name: _new_bucket() for name in split}
    )
    for split_name, indices in split.items():
        iterator = progress(
            indices,
            desc=f"summarize {split_name} split distributions",
            total=len(indices),
            enabled=show_progress,
        )
        for index in iterator:
            target = dataset.target(index)
            particle_label = dataset.particle_label(index)
            source_path = dataset.source_path(index) or f"unknown:{index}"
            source_group = source_group_key(source_path)
            attrs = _event_attrs(dataset, index)
            n_nodes, n_edges, n_detector_nodes, n_pulse_nodes = _shape_counts(dataset, index)
            _add(
                totals[split_name],
                source_path=source_group,
                target=target,
                particle_label=particle_label,
                event_time=attrs.get("time", attrs.get("hhmmss")),
                n_nodes=n_nodes,
                n_edges=n_edges,
                n_detector_nodes=n_detector_nodes,
                n_pulse_nodes=n_pulse_nodes,
            )
            if target is not None and target.shape[0] > 0 and math.isfinite(float(target[0])):
                bin_key = _energy_bin(float(target[0]), energy_bin_width)
                _add(
                    by_energy[bin_key][split_name],
                    source_path=source_group,
                    target=target,
                    particle_label=particle_label,
                    event_time=attrs.get("time", attrs.get("hhmmss")),
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    n_detector_nodes=n_detector_nodes,
                    n_pulse_nodes=n_pulse_nodes,
                )
    total_events = max(sum(len(indices) for indices in split.values()), 1)
    total_sources = sum(len(totals[name]["sources"]) for name in split)
    plot_result = _plot_split_distributions(totals, by_energy, plot_dir) if plot_dir is not None else {}
    plot_files = list(plot_result.get("plot_files", []))
    return {
        "config": {
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
            "train_fraction": float(1.0 - val_fraction - test_fraction),
            "source_val_fraction": float(source_val_fraction),
            "source_test_fraction": float(source_test_fraction),
            "source_train_fraction": float(1.0 - source_val_fraction - source_test_fraction),
            "seed": int(seed),
            "energy_bin_width": float(energy_bin_width),
            "split_mode": "source-stratified",
            "graph_format": "hetero" if isinstance(dataset, H5HeteroGraphDataset) else "homogeneous",
            "plot_files": plot_files,
            "redraw_artifacts": (
                {"split_distribution_plot_data_json": plot_result["plot_data_json"]}
                if "plot_data_json" in plot_result
                else {}
            ),
        },
        "totals": {
            name: {
                **_finish_bucket(bucket),
                "event_fraction": len(split[name]) / total_events,
                "source_fraction": len(bucket["sources"]) / max(total_sources, 1),
            }
            for name, bucket in totals.items()
        },
        "by_energy": {
            bin_key: {
                name: _finish_bucket(bucket)
                for name, bucket in split_buckets.items()
            }
            for bin_key, split_buckets in sorted(by_energy.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize source-stratified train/val/test distributions.")
    parser.add_argument("graphs", nargs="+", help="HDF5 shard, shard base path, or directory")
    parser.add_argument("-o", "--output", required=True, help="output JSON path")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--source-val-fraction", type=float, default=0.10)
    parser.add_argument("--source-test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--energy-bin-width", type=float, default=0.1)
    parser.add_argument("--split-workers", type=int, default=1)
    parser.add_argument("--plot-dir", default=None, help="optional output directory for train/val/test distribution PDFs")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    progress_write("stage=start split_distribution_summary")
    paths = _expand_h5_graph_paths(args.graphs)
    if not paths:
        raise SystemExit("no graph files matched")
    progress_write(f"stage=done split_distribution_summary expand_paths shards={len(paths)}")
    progress_write(f"stage=start split_distribution_summary detect_format first_path={paths[0]}")
    with h5py.File(paths[0], "r") as handle:
        is_hetero = str(handle.attrs.get("format", "")) == HETERO_FORMAT_NAME
    progress_write(f"stage=done split_distribution_summary detect_format hetero={int(is_hetero)}")
    progress_write("stage=start split_distribution_summary dataset_init")
    if is_hetero:
        dataset: GraphDataset = H5HeteroGraphDataset(
            paths,
            require_target=True,
            require_particle_label=True,
            load_attrs=False,
        )
    else:
        dataset = H5GraphDataset(
            paths,
            require_target=True,
            require_particle_label=True,
            load_node_positions=False,
            load_attrs=False,
            load_particle_label=True,
            show_progress=not args.no_progress,
        )
    progress_write(
        "stage=done split_distribution_summary dataset_init "
        f"graphs={len(dataset)} shards={len(paths)}"
    )
    try:
        payload = summarize(
            dataset,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            source_val_fraction=args.source_val_fraction,
            source_test_fraction=args.source_test_fraction,
            seed=args.seed,
            energy_bin_width=args.energy_bin_width,
            split_workers=args.split_workers,
            show_progress=not args.no_progress,
            plot_dir=Path(args.plot_dir).expanduser() if args.plot_dir else None,
        )
    finally:
        dataset.close()
        progress_write("stage=done split_distribution_summary dataset_close")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"split_distribution_summary={output}")
    progress_write(f"stage=done split_distribution_summary output={output}")


if __name__ == "__main__":
    main()
