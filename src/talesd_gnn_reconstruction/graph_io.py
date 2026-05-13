from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .event_graph import GraphEvent, graph_columns


def create_graph_file(path: str | Path, config: dict[str, Any] | None = None) -> h5py.File:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(output, "w")
    handle.attrs["format"] = "talesd_gnn_graphs"
    handle.attrs["format_version"] = "0.3"
    handle.attrs["charge_definition"] = "coincidence_onset_integral"
    handle.attrs["columns_json"] = json.dumps(graph_columns())
    if config:
        handle.attrs["config_json"] = json.dumps(config, sort_keys=True)
    handle.create_group("events")
    return handle


def write_graph(handle: h5py.File, index: int, graph: GraphEvent) -> None:
    group = handle["events"].create_group(f"{index:08d}")
    group.create_dataset("node_features", data=graph.node_features, compression="gzip", compression_opts=4)
    group.create_dataset("node_positions_km", data=graph.node_positions_km, compression="gzip", compression_opts=4)
    group.create_dataset("edge_index", data=graph.edge_index, compression="gzip", compression_opts=4)
    group.create_dataset("edge_features", data=graph.edge_features, compression="gzip", compression_opts=4)
    group.create_dataset("pulse_features", data=graph.pulse_features, compression="gzip", compression_opts=4)
    if graph.target is not None:
        group.create_dataset("target", data=graph.target.astype(np.float32))
    for key, value in graph.metadata.items():
        group.attrs[key] = value


def graph_count(path: str | Path) -> int:
    with h5py.File(Path(path).expanduser(), "r") as handle:
        return len(handle["events"])
