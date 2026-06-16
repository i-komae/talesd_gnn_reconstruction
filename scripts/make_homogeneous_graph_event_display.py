"""Render the DATA-sample homogeneous graph display from dstio GraphEvent.

This script intentionally follows dstio's ``plot_hetero_graph_3d`` drawing
style.  The homogeneous view keeps pulse nodes and pulse-pulse relations only:
detector nodes, detector-detector edges, and detector-pulse edges are omitted.
By default, Ising-rejected pulse candidates are shown as unused nodes with white
fill and black outline, and they are excluded from the time colorbar
normalization.  The script also writes a kept-only view where rejected pulse
candidates are not drawn at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import dstio
from dstio.tale.constants import EDGE_FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS
from dstio.tale.graph import GraphEvent
from dstio.tale.plotting import (
    _axis_equal_3d,
    _event_display_time_colormap,
    _plot_3d_edge,
    _sample_edge_indices,
    _tex_escape_text,
    apply_event_display_matplotlib_style,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DSTIO_ROOT = Path(dstio.__file__).resolve().parents[2]
DEFAULT_DATA_DST = DSTIO_ROOT / "test" / "data" / "tale_data_talesdcalibev_single_event.dst"
DEFAULT_CONST_DST = Path("/Users/ikomae/TALE/TASoft/development/data/SD/talesdconst_pass2.dst")
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "api" / "fig" / "homogeneous_graph_event_display.png"
DEFAULT_KEPT_ONLY_OUTPUT = (
    REPO_ROOT / "docs" / "api" / "fig" / "homogeneous_graph_event_display_ising_kept_only.png"
)


def _pulse_column(graph: GraphEvent, name: str) -> np.ndarray:
    return graph.pulse_features[:, PULSE_FEATURE_COLUMNS.index(name)]


def _used_time_limits(arrival: np.ndarray, used: np.ndarray) -> tuple[float, float]:
    used_times = np.asarray(arrival, dtype=np.float64)[np.asarray(used, dtype=bool)]
    used_times = used_times[np.isfinite(used_times)]
    if used_times.size:
        min_t = float(np.nanmin(used_times))
        max_t = float(np.nanmax(used_times))
        if not np.isfinite(min_t) or not np.isfinite(max_t):
            min_t = 0.0
            max_t = 1.0
        elif max_t <= min_t:
            max_t = min_t + 1.0
    else:
        min_t = 0.0
        max_t = 1.0
    return min_t, max_t


def _displayed_pulse_mask(used: np.ndarray, *, drop_rejected_pulses: bool) -> np.ndarray:
    used = np.asarray(used, dtype=bool)
    if drop_rejected_pulses:
        return used.copy()
    return np.ones(used.shape, dtype=bool)


def _load_data_graph(dst_path: Path, const_dst: Path, *, event_index: int = 0) -> GraphEvent:
    import dstio.tale.graph as tale_graph

    for index, graph in enumerate(
        tale_graph.iter_graphs(
            dst_path,
            kind="auto",
            cleaning="ising",
            node_policy="all_candidates_with_ising",
            const_dst=const_dst,
            require_reference_core=False,
        )
    ):
        if index == event_index:
            return graph
    raise ValueError(f"event_index {event_index} was not found in {dst_path}")


def plot_homogeneous_graph_from_hetero(
    graph: GraphEvent,
    *,
    title: str | None = None,
    pulse_relations: tuple[str, ...] = (
        "pulse__same_detector_next__pulse",
        "pulse__same_detector_prev__pulse",
        "pulse__near_space__pulse",
        "pulse__time_causal__pulse",
    ),
    max_edges_per_relation: int | None = 160,
    pulse_time_scale: float = 0.60,
    drop_rejected_pulses: bool = False,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError("plot_homogeneous_graph_from_hetero requires matplotlib") from exc

    apply_event_display_matplotlib_style()
    if graph.pulse_positions_km.shape[0] == 0:
        raise ValueError("graph has no pulse nodes")

    arrival = _pulse_column(graph, "pulse_arrival_usec_rel").astype(np.float64, copy=False)
    z_scale = max(float(pulse_time_scale), 1.0e-3)
    finite_arrival = arrival[np.isfinite(arrival)]
    z_origin = min(float(np.min(finite_arrival)), 0.0) if finite_arrival.size else 0.0
    pulse_z = 0.24 + z_scale * np.sqrt(np.clip(arrival - z_origin, 0.0, None))
    pulse_xyz = np.column_stack([graph.pulse_positions_km[:, :2].astype(np.float64, copy=False), pulse_z])

    rho = np.power(10.0, _pulse_column(graph, "log10_pulse_rho").astype(np.float64, copy=False))
    keep = _pulse_column(graph, "ising_keep").astype(np.float64, copy=False) > 0.5
    used = keep & np.isfinite(arrival)
    unused = ~used
    displayed = _displayed_pulse_mask(used, drop_rejected_pulses=drop_rejected_pulses)

    min_t, max_t = _used_time_limits(arrival, used)
    cmap = _event_display_time_colormap(max_t)
    norm = Normalize(vmin=min_t, vmax=max_t)

    fig = plt.figure(figsize=(10.2, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    if title:
        ax.set_title(_tex_escape_text(title), fontsize=13, pad=12)

    relation_style = {
        "pulse__same_detector_next__pulse": ("#009E73", "-", 0.035, 0.10),
        "pulse__same_detector_prev__pulse": ("#009E73", ":", -0.035, 0.09),
        "pulse__near_space__pulse": ("#7A7A7A", "-", 0.060, 0.16),
        "pulse__time_causal__pulse": ("#D55E00", "-", -0.070, 0.22),
    }
    for relation in pulse_relations:
        edge_index = graph.edge_index_by_type.get(relation)
        if edge_index is None or edge_index.shape[1] == 0:
            continue
        edge_used = used[edge_index[0]] & used[edge_index[1]]
        if not np.any(edge_used):
            continue
        original_indices = np.flatnonzero(edge_used)
        filtered_edges = edge_index[:, original_indices]
        features = graph.edge_features_by_type.get(relation)
        filtered_features = features[original_indices] if features is not None else None
        color, linestyle, base_offset, z_lift = relation_style.get(
            relation,
            ("0.45", "-", 0.03, 0.08),
        )
        if (
            filtered_features is not None
            and filtered_features.shape[1] > EDGE_FEATURE_COLUMNS.index("ising_weight_raw")
        ):
            weights = filtered_features[:, EDGE_FEATURE_COLUMNS.index("ising_weight_raw")]
            scale = max(float(np.nanquantile(np.abs(weights), 0.90)), 1.0e-6)
        else:
            weights = np.ones(filtered_edges.shape[1], dtype=np.float64)
            scale = 1.0
        for local_idx in _sample_edge_indices(filtered_edges.shape[1], max_edges_per_relation):
            src = int(filtered_edges[0, local_idx])
            dst = int(filtered_edges[1, local_idx])
            sign_offset = base_offset * (1.0 + 0.25 * ((local_idx % 3) - 1))
            frac = np.clip(abs(float(weights[local_idx])) / scale, 0.0, 1.0)
            _plot_3d_edge(
                ax,
                pulse_xyz[src],
                pulse_xyz[dst],
                color=color,
                alpha=0.08 + 0.30 * frac,
                linewidth=0.20 + 0.70 * np.sqrt(frac),
                offset=sign_offset,
                z_lift=z_lift,
                linestyle=linestyle,
            )

    pulse_size = np.clip(18.0 + 7.0 * np.sqrt(np.clip(rho, 0.0, None)), 20.0, 68.0)
    if np.any(used):
        ax.scatter(
            pulse_xyz[used, 0],
            pulse_xyz[used, 1],
            pulse_xyz[used, 2],
            c=arrival[used],
            cmap=cmap,
            norm=norm,
            s=pulse_size[used],
            marker="o",
            edgecolors="none",
            alpha=0.98,
            zorder=7,
        )
    if np.any(unused) and not drop_rejected_pulses:
        ax.scatter(
            pulse_xyz[unused, 0],
            pulse_xyz[unused, 1],
            pulse_xyz[unused, 2],
            s=pulse_size[unused],
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
            alpha=0.95,
            zorder=6,
        )

    axis_xyz = pulse_xyz[displayed] if np.any(displayed) else pulse_xyz
    if graph.detector_positions_km.shape[0] > 0:
        detector_xy = graph.detector_positions_km[:, :2].astype(np.float64, copy=False)
        detector_xyz = np.column_stack([detector_xy, np.zeros(detector_xy.shape[0], dtype=np.float64)])
        axis_xyz = np.vstack([detector_xyz, pulse_xyz])
    _axis_equal_3d(ax, axis_xyz)
    ax.set_xlabel(r"$x\ [{\rm km}]$")
    ax.set_ylabel(r"$y\ [{\rm km}]$")
    ax.set_zlabel("pulse time layer")
    ax.view_init(elev=24, azim=-58)
    ax.grid(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor=cmap(norm(0.5 * (min_t + max_t))),
            markersize=6,
            linewidth=0,
            label="used homogeneous node",
        ),
        Line2D([0], [0], color="#009E73", linewidth=1.2, label="same detector next/prev"),
        Line2D([0], [0], color="#7A7A7A", linewidth=1.2, label="near space"),
        Line2D([0], [0], color="#D55E00", linewidth=1.2, label="time causal"),
    ]
    if not drop_rejected_pulses:
        handles.insert(
            1,
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="white",
                markersize=6,
                linewidth=0,
                label="unused Ising-rejected pulse",
            ),
        )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"used node time $[\mu{\rm s}]$", shrink=0.62, pad=0.08)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DATA_DST)
    parser.add_argument("--const-dst", type=Path, default=DEFAULT_CONST_DST)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kept-only-output", type=Path, default=DEFAULT_KEPT_ONLY_OUTPUT)
    parser.add_argument(
        "--skip-kept-only-output",
        action="store_true",
        help="Do not write the additional view with Ising-rejected pulses removed.",
    )
    parser.add_argument("--max-edges-per-relation", type=int, default=160)
    args = parser.parse_args()

    graph = _load_data_graph(args.dst, args.const_dst, event_index=args.event_index)
    title = f"event {args.event_index}: {graph.event_id} homogeneous graph"
    fig = plot_homogeneous_graph_from_hetero(
        graph,
        title=title,
        max_edges_per_relation=args.max_edges_per_relation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white")
    print(f"wrote {args.output}")
    if not args.skip_kept_only_output:
        kept_title = f"event {args.event_index}: {graph.event_id} homogeneous graph, Ising-kept pulses only"
        kept_fig = plot_homogeneous_graph_from_hetero(
            graph,
            title=kept_title,
            max_edges_per_relation=args.max_edges_per_relation,
            drop_rejected_pulses=True,
        )
        args.kept_only_output.parent.mkdir(parents=True, exist_ok=True)
        kept_fig.savefig(args.kept_only_output, facecolor="white")
        print(f"wrote {args.kept_only_output}")


if __name__ == "__main__":
    main()
