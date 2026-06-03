#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.dataset import H5GraphDataset
from talesd_gnn_reconstruction.metrics import direction_columns_for_dim, direction_to_angles
from talesd_gnn_reconstruction.progress import progress
from talesd_gnn_reconstruction.train import split_indices_by_stratified_source_path


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
    bin_index = math.floor(float(log10_energy) / float(width))
    low = bin_index * float(width)
    high = low + float(width)
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
        "nodes": [],
        "edges": [],
    }


def _add(bucket: dict[str, Any], *, source_path: str, target: np.ndarray | None, particle_label: float | None, n_nodes: int | None, n_edges: int | None) -> None:
    bucket["events"] += 1
    bucket["sources"].add(source_path)
    if particle_label is not None and math.isfinite(float(particle_label)):
        bucket["particle_labels"].append(float(particle_label))
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
    if n_nodes is not None:
        bucket["nodes"].append(float(n_nodes))
    if n_edges is not None:
        bucket["edges"].append(float(n_edges))


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(bucket["particle_labels"], dtype=np.float64)
    finite_labels = labels[np.isfinite(labels)]
    return {
        "events": int(bucket["events"]),
        "sources": int(len(bucket["sources"])),
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
        "nodes": _stats(bucket["nodes"]),
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
) -> list[str]:
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

    features = [
        ("log10_energy", r"$\log_{10}(E/\mathrm{eV})$"),
        ("core_x_km", "core x [km]"),
        ("core_y_km", "core y [km]"),
        ("core_radius_km", "core radius [km]"),
        ("zenith_deg", "zenith [deg]"),
        ("azimuth_deg", "azimuth [deg]"),
        ("nodes", "accepted pulses / event"),
        ("edges", "edges / event"),
        ("particle_labels", "particle label"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_GRID)
    for ax, (key, xlabel) in zip(axes.reshape(-1), features, strict=True):
        arrays = [_finite_array(totals[name][key]) for name in split_order]
        if key == "particle_labels":
            bins = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
            ax.set_xticks([0.0, 1.0], ["p", "Fe"])
        else:
            bins = _hist_bins(arrays)
        for name, arr in zip(split_order, arrays, strict=True):
            if arr.size == 0:
                continue
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
    axes.reshape(-1)[0].legend(frameon=False)
    fig.suptitle("Train/validation/test parameter distributions")
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, output_dir / "split_parameter_distributions.pdf"))

    energy_bins = sorted(by_energy)
    if energy_bins:
        x = []
        labels = []
        for bin_key in energy_bins:
            low, high = bin_key.split("-", maxsplit=1)
            x.append((float(low) + float(high)) * 0.5)
            labels.append(bin_key)
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
        for ax, value_key, ylabel in (
            (axes[0], "events", "events"),
            (axes[1], "sources", "sources"),
        ):
            for name in split_order:
                y = [float(_finish_bucket(by_energy[bin_key][name])[value_key]) for bin_key in energy_bins]
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
    return pdf_files


def _shape_counts(dataset: H5GraphDataset, index: int) -> tuple[int | None, int | None]:
    path_index, _local_index, key = dataset._locate(index)  # noqa: SLF001 - summary script uses dataset internals for cheap shapes.
    group = dataset._handle(path_index)["events"][key]  # noqa: SLF001
    n_nodes = int(group["node_features"].shape[0]) if "node_features" in group else None
    n_edges = int(group["edge_features"].shape[0]) if "edge_features" in group else None
    return n_nodes, n_edges


def summarize(
    dataset: H5GraphDataset,
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
            n_nodes, n_edges = _shape_counts(dataset, index)
            _add(
                totals[split_name],
                source_path=source_path,
                target=target,
                particle_label=particle_label,
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
            if target is not None and target.shape[0] > 0 and math.isfinite(float(target[0])):
                bin_key = _energy_bin(float(target[0]), energy_bin_width)
                _add(
                    by_energy[bin_key][split_name],
                    source_path=source_path,
                    target=target,
                    particle_label=particle_label,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                )
    total_events = max(sum(len(indices) for indices in split.values()), 1)
    total_sources = sum(len(totals[name]["sources"]) for name in split)
    plot_files = _plot_split_distributions(totals, by_energy, plot_dir) if plot_dir is not None else []
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
            "plot_files": plot_files,
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

    paths = _expand_h5_graph_paths(args.graphs)
    if not paths:
        raise SystemExit("no graph files matched")
    dataset = H5GraphDataset(
        paths,
        require_target=True,
        require_particle_label=True,
        load_node_positions=False,
        load_attrs=False,
        load_particle_label=True,
        show_progress=not args.no_progress,
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
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"split_distribution_summary={output}")


if __name__ == "__main__":
    main()
