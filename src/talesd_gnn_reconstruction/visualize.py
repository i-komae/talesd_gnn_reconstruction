from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import ListedColormap, Normalize

from .constants import NODE_FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS
from .dataset import H5GraphDataset
from .layout import default_const_dst_path, load_tale_const_positions
from .metrics import normalize_directions

MAP_X_MIN = -10.0
MAP_X_MAX = 0.0
MAP_Y_MIN = 11.0
MAP_Y_MAX = 21.0


def _columns(dataset: H5GraphDataset) -> dict[str, list[str]]:
    try:
        loaded = json.loads(dataset.columns_json)
        if isinstance(loaded, dict) and "node_features" in loaded:
            return loaded
    except Exception:
        pass
    return {"node_features": list(NODE_FEATURE_COLUMNS)}


def _feature(sample: dict[str, Any], columns: dict[str, list[str]], name: str, fallback: float = 0.0) -> np.ndarray:
    names = columns.get("node_features", [])
    if name not in names:
        return np.full(sample["node_features"].shape[0], fallback, dtype=np.float32)
    column_index = names.index(name)
    if column_index >= sample["node_features"].shape[1]:
        return np.full(sample["node_features"].shape[0], fallback, dtype=np.float32)
    return sample["node_features"][:, column_index]


def _feature_any(
    sample: dict[str, Any],
    columns: dict[str, list[str]],
    names: list[str],
    fallback: float = 0.0,
) -> np.ndarray:
    for name in names:
        values = _feature(sample, columns, name, fallback=math.nan)
        if not np.all(np.isnan(values)):
            return values
    return np.full(sample["node_features"].shape[0], fallback, dtype=np.float32)


def _node_sizes(log10_rho: np.ndarray) -> np.ndarray:
    rho = np.power(10.0, np.asarray(log10_rho, dtype=float))
    rho = np.where(np.isfinite(rho) & (rho > 0.0), rho, 0.0)
    sizes = np.sqrt(rho) * 20.0
    return np.clip(sizes, 25.0, 600.0)


def _pulse_columns(columns: dict[str, list[str]]) -> list[str]:
    return columns.get("pulse_features", list(PULSE_FEATURE_COLUMNS))


def _draw_secondary_pulses(
    ax,
    sample: dict[str, Any],
    columns: dict[str, list[str]],
    cmap: ListedColormap,
    norm: Normalize,
) -> None:
    pulse_features = sample.get("pulse_features")
    if pulse_features is None or pulse_features.shape[0] == 0:
        return
    names = _pulse_columns(columns)
    required = {"node_index", "arrival_usec_rel", "log10_rho", "pulse_order"}
    if not required.issubset(set(names)):
        return
    node_idx_col = names.index("node_index")
    arrival_col = names.index("arrival_usec_rel")
    rho_col = names.index("log10_rho")
    order_col = names.index("pulse_order")
    positions = sample["node_positions_km"]

    secondary = pulse_features[pulse_features[:, order_col] > 0.0]
    if secondary.shape[0] == 0:
        return
    node_idx = secondary[:, node_idx_col].astype(int)
    valid = (node_idx >= 0) & (node_idx < positions.shape[0])
    secondary = secondary[valid]
    node_idx = node_idx[valid]
    if secondary.shape[0] == 0:
        return

    edge_colors = cmap(norm(secondary[:, arrival_col]))
    sizes = np.maximum(_node_sizes(secondary[:, rho_col]) * 1.35, 80.0)
    ax.scatter(
        positions[node_idx, 0],
        positions[node_idx, 1],
        s=sizes,
        marker="o",
        facecolors="none",
        edgecolors=edge_colors,
        linewidths=1.2,
        alpha=0.95,
        zorder=5,
    )


def _hls_event_colormap(max_time_usec: float) -> ListedColormap:
    import colorsys

    n_colors = int(max(float(max_time_usec), 1.0) * 10.0) + 10
    hues = np.linspace(0.0, 1.0, n_colors, endpoint=False)
    colors = [colorsys.hls_to_rgb(float(h), 0.55, 0.80) for h in hues]
    colors.reverse()
    skip = int(len(colors) * 0.2)
    return ListedColormap(colors[skip:] or colors)


def _unique_edges(edge_index: np.ndarray) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    output: list[tuple[int, int]] = []
    for src, dst in edge_index.T:
        a = int(src)
        b = int(dst)
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            continue
        seen.add(key)
        output.append((a, b))
    return output


def find_event_index(dataset: H5GraphDataset, event_id: str) -> int:
    for index in range(len(dataset)):
        if dataset[index]["event_id"] == event_id:
            return index
    raise ValueError(f"event_id not found: {event_id}")


def _format_ta_time(date_val: Any, time_val: Any, usec_val: Any) -> str:
    try:
        d_val = int(date_val)
        t_val = int(time_val)
        u_val = int(usec_val)
        year = 2000 + d_val // 10000
        month = (d_val // 100) % 100
        day = d_val % 100
        hour = t_val // 10000
        minute = (t_val // 100) % 100
        second = t_val % 100
        return f"{year:04d}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}:{second:02d}.{u_val:06d}"
    except Exception:
        return f"{date_val}, {time_val}.{usec_val}"


def _setup_event_display_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "cm"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def _load_background_detectors(const_dst: str | Path | None) -> np.ndarray | None:
    if const_dst is None:
        const_dst = default_const_dst_path()
    if const_dst is None:
        return None
    try:
        positions = load_tale_const_positions(const_dst)
    except Exception:
        return None
    rows = [(det.x_km, det.y_km) for det in positions.values()]
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def _draw_background_sds(ax, detector_xy: np.ndarray | None) -> None:
    if detector_xy is None or detector_xy.size == 0:
        return
    handle = ax.scatter(
        0,
        0,
        marker="s",
        facecolor="white",
        edgecolor="black",
        s=20,
        linewidths=0.8,
        label="TALE SD",
    )
    ax.scatter(
        detector_xy[:, 0],
        detector_xy[:, 1],
        marker="s",
        facecolor="white",
        edgecolor="black",
        s=18,
        alpha=0.5,
        linewidths=0.8,
        zorder=1,
    )
    ax.legend(handles=[handle], loc="lower left", fontsize=10)


def _draw_truth_overlay(ax, target: np.ndarray | None) -> None:
    if target is None:
        return
    core_x = float(target[1])
    core_y = float(target[2])
    if not (math.isfinite(core_x) and math.isfinite(core_y)):
        return
    if not (MAP_X_MIN <= core_x <= MAP_X_MAX and MAP_Y_MIN <= core_y <= MAP_Y_MAX):
        return

    direction = normalize_directions(target[None, :])[0]
    arrow_half_len = 0.9
    cross_half_len = 0.25
    arrow_cos = float(direction[0])
    arrow_sin = float(direction[1])
    norm_xy = math.hypot(arrow_cos, arrow_sin)
    if norm_xy > 1.0e-9:
        arrow_cos /= norm_xy
        arrow_sin /= norm_xy

    ax.annotate(
        "",
        xy=(core_x - arrow_half_len * arrow_cos, core_y - arrow_half_len * arrow_sin),
        xytext=(core_x + arrow_half_len * arrow_cos, core_y + arrow_half_len * arrow_sin),
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black", linestyle="--"),
        zorder=6,
        annotation_clip=True,
    )
    ax.plot(
        [core_x + cross_half_len * arrow_sin, core_x - cross_half_len * arrow_sin],
        [core_y - cross_half_len * arrow_cos, core_y + cross_half_len * arrow_cos],
        color="black",
        linewidth=1.6,
        linestyle="--",
        zorder=6,
        clip_on=True,
    )


def _truth_text(target: np.ndarray | None) -> str | None:
    if target is None:
        return None
    direction = normalize_directions(target[None, :])[0]
    zenith = math.degrees(math.acos(float(np.clip(direction[2], -1.0, 1.0))))
    azimuth = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 360.0
    return (
        rf"$\log(E_{{\rm true}}/{{\rm eV}})={float(target[0]):.2f}$" + "\n"
        + rf"$\theta_{{\rm true}}={zenith:.1f}^\circ$" + "\n"
        + rf"$\phi_{{\rm true}}={azimuth:.1f}^\circ$"
    )


def plot_graph_sample(
    sample: dict[str, Any],
    columns: dict[str, list[str]],
    output_path: str | Path,
    detector_xy: np.ndarray | None = None,
    show_edges: bool = True,
    annotate_lids: bool = False,
    max_edges: int = 2000,
    dpi: int = 160,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_event_display_style()

    positions = sample["node_positions_km"]
    edge_index = sample["edge_index"]
    target = sample["target"]
    attrs = sample["attrs"]
    event_id = sample["event_id"]

    x = positions[:, 0]
    y = positions[:, 1]
    arrival_usec = _feature_any(sample, columns, ["pulse_arrival_usec_rel", "first_arrival_usec_rel", "arrival_usec_rel"])
    arrival_usec = arrival_usec - np.nanmin(arrival_usec) if arrival_usec.size else arrival_usec
    log10_rho = _feature_any(sample, columns, ["log10_pulse_rho", "log10_first_rho", "log10_rho"])
    sizes = _node_sizes(log10_rho)
    max_time = float(np.nanmax(arrival_usec)) if arrival_usec.size else 1.0
    if not math.isfinite(max_time) or max_time <= 0.0:
        max_time = 1.0
    cmap = _hls_event_colormap(max_time)
    norm = Normalize(vmin=0.0, vmax=max_time)

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_background_sds(ax, detector_xy)

    if show_edges and edge_index.size > 0:
        edges = _unique_edges(edge_index)[:max_edges]
        for src, dst in edges:
            ax.plot(
                [x[src], x[dst]],
                [y[src], y[dst]],
                color="0.45",
                linewidth=0.55,
                alpha=0.18,
                zorder=2,
            )

    scatter = ax.scatter(
        x,
        y,
        c=arrival_usec,
        s=sizes,
        cmap=cmap,
        norm=norm,
        marker="o",
        edgecolors="none",
        linewidths=0.0,
        alpha=1.0,
        zorder=4,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.015)
    cbar.set_label(r"Relative pulse onset time [$\mu$s]")
    _draw_secondary_pulses(ax, sample, columns, cmap, norm)

    if annotate_lids:
        lids = str(attrs.get("lids", "")).split(",")
        for i, lid in enumerate(lids[: len(x)]):
            ax.annotate(lid, (x[i], y[i]), xytext=(3, 3), textcoords="offset points", fontsize=7)

    _draw_truth_overlay(ax, target)

    n_nodes = int(attrs.get("n_nodes", positions.shape[0]))
    n_edges = int(attrs.get("n_edges", edge_index.shape[1]))
    n_pulses = int(attrs.get("n_pulses", sample.get("pulse_features", np.zeros((0, 0))).shape[0]))
    ax.set_title("TALE-SD GNN graph", fontsize=16)
    ax.set_xlabel(r"$x\ [\mathrm{km}]$")
    ax.set_ylabel(r"$y\ [\mathrm{km}]$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)
    ax.grid(False)

    info_lines = [
        f"Event: {event_id}",
        f"TALE: {_format_ta_time(attrs.get('date', 0), attrs.get('time', 0), attrs.get('usec', 0))}",
        f"N_SD={n_nodes}, N_pulse={n_pulses}, N_edge={n_edges}",
    ]
    truth = _truth_text(target)
    if truth is not None:
        info_lines.append(truth)
        info_lines.append("dashed: true")
    ax.text(
        0.05,
        0.95,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=10.5,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="gray"),
    )
    ax.text(
        0.95,
        0.05,
        "GNN INPUT",
        transform=ax.transAxes,
        fontsize=34,
        color="gray",
        weight="bold",
        ha="right",
        va="bottom",
        alpha=0.38,
    )

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, facecolor="white", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(output)


def visualize_graphs(
    graphs: list[str | Path],
    output: str | Path,
    index: int = 0,
    event_id: str | None = None,
    count: int = 1,
    show_edges: bool = True,
    annotate_lids: bool = False,
    max_edges: int = 2000,
    dpi: int = 160,
    const_dst: str | Path | None = None,
) -> list[str]:
    dataset = H5GraphDataset(graphs, require_target=False)
    try:
        columns = _columns(dataset)
        detector_xy = _load_background_detectors(const_dst)
        if event_id is not None:
            start = find_event_index(dataset, event_id)
        else:
            start = int(index)
        if start < 0 or start >= len(dataset):
            raise IndexError(f"graph index out of range: {start} (n={len(dataset)})")

        output_path = Path(output).expanduser()
        multi = count > 1 or output_path.suffix == ""
        written: list[str] = []
        for offset in range(max(int(count), 1)):
            sample_index = start + offset
            if sample_index >= len(dataset):
                break
            sample = dataset[sample_index]
            if multi:
                target_path = output_path / f"graph_{sample_index:08d}.pdf"
            else:
                target_path = output_path
            written.append(
                plot_graph_sample(
                    sample,
                    columns=columns,
                    output_path=target_path,
                    detector_xy=detector_xy,
                    show_edges=show_edges,
                    annotate_lids=annotate_lids,
                    max_edges=max_edges,
                    dpi=dpi,
                )
            )
        return written
    finally:
        dataset.close()
