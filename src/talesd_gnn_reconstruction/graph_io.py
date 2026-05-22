from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import WAVEFORM_SCHEMA
from .event_graph import GraphEvent, graph_columns


def create_graph_file(path: str | Path, config: dict[str, Any] | None = None) -> h5py.File:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(output, "w")
    handle.attrs["format"] = "talesd_gnn_graphs"
    handle.attrs["format_version"] = "0.4"
    handle.attrs["charge_definition"] = "coincidence_onset_integral"
    handle.attrs["waveform_schema"] = WAVEFORM_SCHEMA
    handle.attrs["columns_json"] = json.dumps(graph_columns())
    if config:
        handle.attrs["config_json"] = json.dumps(config, sort_keys=True)
    handle.create_group("events")
    metadata = handle.create_group("metadata")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    metadata.create_dataset("event_id", shape=(0,), maxshape=(None,), chunks=True, dtype=string_dtype)
    metadata.create_dataset("source_path", shape=(0,), maxshape=(None,), chunks=True, dtype=string_dtype)
    metadata.create_dataset("source_index", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int64)
    metadata.create_dataset("parttype", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int32)
    metadata.create_dataset("particle_label", shape=(0,), maxshape=(None,), chunks=True, dtype=np.float32)
    return handle


def _append_metadata(handle: h5py.File, index: int, graph: GraphEvent) -> None:
    metadata = handle.get("metadata")
    if metadata is None:
        return
    size = int(index) + 1
    for dataset in metadata.values():
        if dataset.shape[0] < size:
            dataset.resize((size,))

    metadata["event_id"][index] = str(graph.event_id)
    metadata["source_path"][index] = str(graph.metadata.get("source_path", ""))
    metadata["source_index"][index] = int(graph.metadata.get("source_index", -1))
    metadata["parttype"][index] = int(graph.metadata.get("parttype", -1))
    label = graph.particle_label
    metadata["particle_label"][index] = np.nan if label is None else float(label)


def write_graph(handle: h5py.File, index: int, graph: GraphEvent) -> None:
    group = handle["events"].create_group(f"{index:08d}")
    group.create_dataset("node_features", data=graph.node_features, compression="gzip", compression_opts=4)
    group.create_dataset("node_positions_km", data=graph.node_positions_km, compression="gzip", compression_opts=4)
    group.create_dataset("node_lids", data=graph.node_lids.astype(np.int64), compression="gzip", compression_opts=4)
    group.create_dataset("edge_index", data=graph.edge_index, compression="gzip", compression_opts=4)
    group.create_dataset("edge_features", data=graph.edge_features, compression="gzip", compression_opts=4)
    group.create_dataset("pulse_features", data=graph.pulse_features, compression="gzip", compression_opts=4)
    group.create_dataset("waveform_features", data=graph.waveform_features, compression="gzip", compression_opts=4)
    if graph.target is not None:
        group.create_dataset("target", data=graph.target.astype(np.float32))
    if graph.particle_label is not None:
        group.create_dataset("particle_label", data=np.asarray(graph.particle_label, dtype=np.float32))
    for key, value in graph.metadata.items():
        group.attrs[key] = value
    _append_metadata(handle, index, graph)


def graph_count(path: str | Path) -> int:
    with h5py.File(Path(path).expanduser(), "r") as handle:
        return len(handle["events"])
