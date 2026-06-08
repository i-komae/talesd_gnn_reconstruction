from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FORMAT_NAME = "talesd_gnn_hetero_graphs"
FORMAT_VERSION = "0.1"
GRAPH_DEFINITION = "tale_sd_hetero_ising_pulse_detector_graph_v1"
WAVEFORM_SCHEMA = "detector_full_calibrated_vem_v1"

EDGE_RELATIONS = (
    "pulse__interacts__pulse",
    "detector__near__detector",
    "detector__observes__pulse",
)


def _graph_columns() -> dict[str, Any]:
    import dstio.tale.graph as tale_graph

    return tale_graph.graph_columns()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


def _metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, default=_json_default, sort_keys=True)


def _set_scalar_attrs(group: h5py.Group, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[key] = value


def create_hetero_graph_file(path: str | Path, config: dict[str, Any] | None = None) -> h5py.File:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(output, "w")
    handle.attrs["format"] = FORMAT_NAME
    handle.attrs["format_version"] = FORMAT_VERSION
    handle.attrs["graph_definition"] = GRAPH_DEFINITION
    handle.attrs["waveform_schema"] = WAVEFORM_SCHEMA
    handle.attrs["columns_json"] = json.dumps(_graph_columns(), sort_keys=True)
    if config:
        handle.attrs["config_json"] = json.dumps(config, default=_json_default, sort_keys=True)
    handle.create_group("events")
    metadata = handle.create_group("metadata")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    metadata.create_dataset("event_id", shape=(0,), maxshape=(None,), chunks=True, dtype=string_dtype)
    metadata.create_dataset("source_path", shape=(0,), maxshape=(None,), chunks=True, dtype=string_dtype)
    metadata.create_dataset("source_index", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int64)
    metadata.create_dataset("parttype", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int32)
    metadata.create_dataset("particle_label", shape=(0,), maxshape=(None,), chunks=True, dtype=np.float32)
    metadata.create_dataset("n_detector_nodes", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int32)
    metadata.create_dataset("n_pulse_nodes", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int32)
    metadata.create_dataset("metadata_json", shape=(0,), maxshape=(None,), chunks=True, dtype=string_dtype)
    return handle


def _append_metadata(handle: h5py.File, index: int, graph: Any) -> None:
    metadata_group = handle.get("metadata")
    if metadata_group is None:
        return
    size = int(index) + 1
    for dataset in metadata_group.values():
        if dataset.shape[0] < size:
            dataset.resize((size,))

    metadata = dict(graph.metadata)
    metadata_group["event_id"][index] = str(graph.event_id)
    metadata_group["source_path"][index] = str(metadata.get("source_path", ""))
    metadata_group["source_index"][index] = int(metadata.get("source_index", -1))
    metadata_group["parttype"][index] = int(metadata.get("parttype", -1))
    particle_label = graph.particle_label
    metadata_group["particle_label"][index] = np.nan if particle_label is None else float(particle_label)
    metadata_group["n_detector_nodes"][index] = int(graph.detector_features.shape[0])
    metadata_group["n_pulse_nodes"][index] = int(graph.pulse_features.shape[0])
    metadata_group["metadata_json"][index] = _metadata_json(metadata)


def _create_compressed(group: h5py.Group, name: str, data: Any) -> None:
    array = np.asarray(data)
    group.create_dataset(name, data=array, compression="gzip", compression_opts=4)


def write_hetero_graph(handle: h5py.File, index: int, graph: Any) -> None:
    group = handle["events"].create_group(f"{index:08d}")
    _create_compressed(group, "detector_features", graph.detector_features.astype(np.float32, copy=False))
    _create_compressed(
        group,
        "detector_context_features",
        graph.detector_context_features.astype(np.float32, copy=False),
    )
    _create_compressed(group, "detector_positions_km", graph.detector_positions_km.astype(np.float32, copy=False))
    _create_compressed(group, "detector_lids", graph.detector_lids.astype(np.int64, copy=False))
    _create_compressed(group, "detector_waveforms", graph.detector_waveforms.astype(np.float32, copy=False))
    _create_compressed(group, "pulse_features", graph.pulse_features.astype(np.float32, copy=False))
    _create_compressed(group, "pulse_positions_km", graph.pulse_positions_km.astype(np.float32, copy=False))
    _create_compressed(group, "pulse_lids", graph.pulse_lids.astype(np.int64, copy=False))
    _create_compressed(group, "pulse_detector_index", graph.pulse_detector_index.astype(np.int64, copy=False))
    _create_compressed(group, "pulse_bounds", graph.pulse_bounds.astype(np.float32, copy=False))

    edge_index_group = group.create_group("edge_index_by_type")
    edge_feature_group = group.create_group("edge_features_by_type")
    for relation in EDGE_RELATIONS:
        edge_index = graph.edge_index_by_type.get(relation)
        edge_features = graph.edge_features_by_type.get(relation)
        if edge_index is None:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        if edge_features is None:
            edge_features = np.zeros((0, 0), dtype=np.float32)
        _create_compressed(edge_index_group, relation, np.asarray(edge_index, dtype=np.int64))
        _create_compressed(edge_feature_group, relation, np.asarray(edge_features, dtype=np.float32))

    if graph.target is not None:
        group.create_dataset("target", data=np.asarray(graph.target, dtype=np.float32))
    if graph.particle_label is not None:
        group.create_dataset("particle_label", data=np.asarray(graph.particle_label, dtype=np.float32))
    metadata = dict(graph.metadata)
    group.attrs["metadata_json"] = _metadata_json(metadata)
    _set_scalar_attrs(group, metadata)
    _append_metadata(handle, index, graph)


def graph_event_to_sample(graph: Any, *, load_attrs: bool = True) -> dict[str, Any]:
    metadata = dict(graph.metadata)
    sample: dict[str, Any] = {
        "detector_features": graph.detector_features.astype(np.float32, copy=False),
        "detector_context_features": graph.detector_context_features.astype(np.float32, copy=False),
        "detector_positions_km": graph.detector_positions_km.astype(np.float32, copy=False),
        "detector_lids": graph.detector_lids.astype(np.int64, copy=False),
        "detector_waveforms": graph.detector_waveforms.astype(np.float32, copy=False),
        "pulse_features": graph.pulse_features.astype(np.float32, copy=False),
        "pulse_positions_km": graph.pulse_positions_km.astype(np.float32, copy=False),
        "pulse_lids": graph.pulse_lids.astype(np.int64, copy=False),
        "pulse_detector_index": graph.pulse_detector_index.astype(np.int64, copy=False),
        "pulse_bounds": graph.pulse_bounds.astype(np.float32, copy=False),
        "edge_index_by_type": {
            relation: np.asarray(graph.edge_index_by_type.get(relation, np.zeros((2, 0), dtype=np.int64)), dtype=np.int64)
            for relation in EDGE_RELATIONS
        },
        "edge_features_by_type": {
            relation: np.asarray(
                graph.edge_features_by_type.get(relation, np.zeros((0, 0), dtype=np.float32)),
                dtype=np.float32,
            )
            for relation in EDGE_RELATIONS
        },
        "target": None if graph.target is None else np.asarray(graph.target, dtype=np.float32),
        "particle_label": graph.particle_label,
        "metadata": metadata,
        "event_id": str(graph.event_id),
    }
    if load_attrs:
        attrs = dict(metadata)
        attrs["event_id"] = str(graph.event_id)
        sample["attrs"] = attrs
    return sample


def hetero_graph_count(path: str | Path) -> int:
    with h5py.File(Path(path).expanduser(), "r") as handle:
        return len(handle["events"])


class H5HeteroGraphDataset:
    def __init__(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        require_target: bool = False,
        require_particle_label: bool = False,
        load_attrs: bool = True,
    ):
        if isinstance(path, (str, Path)):
            self.paths = [Path(path).expanduser()]
        else:
            self.paths = [Path(item).expanduser() for item in path]
        self.require_target = bool(require_target)
        self.require_particle_label = bool(require_particle_label)
        self.load_attrs = bool(load_attrs)
        self._handles: dict[int, h5py.File] = {}
        self._path_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        self._path_local_indices: list[list[int] | None] = []
        self._path_key_lists: list[list[str] | None] = []
        self.columns_json = "{}"

        total = 0
        for path_index, graph_path in enumerate(self.paths):
            with h5py.File(graph_path, "r") as handle:
                if str(handle.attrs.get("format", "")) != FORMAT_NAME:
                    raise ValueError(f"{graph_path} is not a hetero graph HDF5 file")
                if str(handle.attrs.get("graph_definition", "")) != GRAPH_DEFINITION:
                    raise ValueError(f"{graph_path} stores unsupported graph_definition")
                if str(handle.attrs.get("waveform_schema", "")) != WAVEFORM_SCHEMA:
                    raise ValueError(f"{graph_path} stores unsupported waveform_schema")
                if path_index == 0:
                    self.columns_json = str(handle.attrs.get("columns_json", "{}"))
                n_events = len(handle["events"])
            total += n_events
            self._path_lengths.append(n_events)
            self._cumulative_lengths.append(total)
            self._path_local_indices.append(None)
            self._path_key_lists.append(None)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle(self, path_index: int) -> h5py.File:
        handle = self._handles.get(path_index)
        if handle is None:
            handle = h5py.File(self.paths[path_index], "r")
            self._handles[path_index] = handle
        return handle

    def _locate(self, index: int) -> tuple[int, int, str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path_index = bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[path_index - 1] if path_index > 0 else 0
        local_index = index - previous
        return path_index, local_index, f"{local_index:08d}"

    @staticmethod
    def _read_edge_group(group: h5py.Group) -> dict[str, np.ndarray]:
        return {relation: group[relation][()] for relation in group.keys()}

    @staticmethod
    def _dataset_nbytes(dataset: h5py.Dataset) -> int:
        dtype = np.dtype(dataset.dtype)
        if dtype.hasobject:
            return int(dataset.id.get_storage_size())
        return int(np.prod(dataset.shape, dtype=np.int64)) * int(dtype.itemsize)

    @classmethod
    def _group_nbytes(cls, group: h5py.Group) -> int:
        total = 0
        for value in group.values():
            if isinstance(value, h5py.Dataset):
                total += cls._dataset_nbytes(value)
            elif isinstance(value, h5py.Group):
                total += cls._group_nbytes(value)
        return int(total)

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _metadata_value(self, path_index: int, local_index: int, name: str) -> Any:
        metadata = self._handle(path_index).get("metadata")
        if metadata is None or name not in metadata or local_index >= len(metadata[name]):
            return None
        value = metadata[name][local_index]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def _particle_label_from_value(value: Any) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _particle_label_from_group(group: h5py.Group) -> float | None:
        if "particle_label" in group:
            return H5HeteroGraphDataset._particle_label_from_value(group["particle_label"][()])
        value = group.attrs.get("particle_label", group.attrs.get("particle_is_iron", None))
        if value is not None:
            return H5HeteroGraphDataset._particle_label_from_value(value)
        parttype = int(group.attrs.get("parttype", -1))
        if parttype == 14:
            return 0.0
        if parttype == 5626:
            return 1.0
        source_path = str(group.attrs.get("source_path", "")).lower()
        if "/proton/" in source_path or "tale_proton" in source_path:
            return 0.0
        if "/iron/" in source_path or "tale_iron" in source_path:
            return 1.0
        return None

    def source_path(self, index: int) -> str:
        path_index, local_index, key = self._locate(index)
        value = self._metadata_value(path_index, local_index, "source_path")
        if value is not None:
            return str(value)
        group = self._handle(path_index)["events"][key]
        if "source_path" in group.attrs:
            return str(group.attrs["source_path"])
        metadata = {}
        if "metadata_json" in group.attrs:
            metadata = json.loads(str(group.attrs["metadata_json"]))
        return str(metadata.get("source_path", ""))

    def target(self, index: int) -> np.ndarray | None:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        if "target" not in group:
            return None
        return group["target"][()].astype(np.float32)

    def particle_label(self, index: int) -> float | None:
        path_index, local_index, key = self._locate(index)
        value = self._metadata_value(path_index, local_index, "particle_label")
        if value is not None:
            label = self._particle_label_from_value(value)
            if label is not None:
                return label
        return self._particle_label_from_group(self._handle(path_index)["events"][key])

    def detector_waveform_shape(self, index: int) -> tuple[int, ...]:
        path_index, _local_index, key = self._locate(index)
        return tuple(int(value) for value in self._handle(path_index)["events"][key]["detector_waveforms"].shape)

    def graph_nbytes(self, index: int) -> int:
        path_index, _local_index, key = self._locate(index)
        return self._group_nbytes(self._handle(path_index)["events"][key])

    def scaler_sample(self, index: int) -> dict[str, Any]:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        target = group["target"][()].astype(np.float32) if "target" in group else None
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{key}")
        particle_label = self.particle_label(index)
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{key}")
        return {
            "detector_features": group["detector_features"][()].astype(np.float32),
            "detector_context_features": group["detector_context_features"][()].astype(np.float32),
            "pulse_features": group["pulse_features"][()].astype(np.float32),
            "edge_features_by_type": self._read_edge_group(group["edge_features_by_type"]),
            "target": target,
            "particle_label": particle_label,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        target = group["target"][()].astype(np.float32) if "target" in group else None
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{key}")
        particle_label = self.particle_label(index)
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{key}")
        sample: dict[str, Any] = {
            "detector_features": group["detector_features"][()].astype(np.float32),
            "detector_context_features": group["detector_context_features"][()].astype(np.float32),
            "detector_positions_km": group["detector_positions_km"][()].astype(np.float32),
            "detector_lids": group["detector_lids"][()].astype(np.int64),
            "detector_waveforms": group["detector_waveforms"][()].astype(np.float32),
            "pulse_features": group["pulse_features"][()].astype(np.float32),
            "pulse_positions_km": group["pulse_positions_km"][()].astype(np.float32),
            "pulse_lids": group["pulse_lids"][()].astype(np.int64),
            "pulse_detector_index": group["pulse_detector_index"][()].astype(np.int64),
            "pulse_bounds": group["pulse_bounds"][()].astype(np.float32),
            "edge_index_by_type": self._read_edge_group(group["edge_index_by_type"]),
            "edge_features_by_type": self._read_edge_group(group["edge_features_by_type"]),
            "target": target,
            "particle_label": particle_label,
        }
        if self.load_attrs:
            sample["attrs"] = dict(group.attrs.items())
            if "metadata_json" in sample["attrs"]:
                sample["metadata"] = json.loads(str(sample["attrs"]["metadata_json"]))
        return sample


class H5PyGHeteroGraphDataset:
    def __init__(
        self,
        *args: Any,
        scalers: dict[str, Any] | None = None,
        waveform_length: int | None = None,
        **kwargs: Any,
    ):
        self.base = H5HeteroGraphDataset(*args, **kwargs)
        self.scalers = scalers
        self.waveform_length = None if waveform_length is None else int(waveform_length)

    def __len__(self) -> int:
        return len(self.base)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "base": self.base.__getstate__(),
            "scalers": self.scalers,
            "waveform_length": self.waveform_length,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.base = H5HeteroGraphDataset.__new__(H5HeteroGraphDataset)
        self.base.__dict__.update(state["base"])
        self.scalers = state.get("scalers")
        self.waveform_length = state.get("waveform_length")

    def close(self) -> None:
        self.base.close()

    def source_path(self, index: int) -> str:
        return self.base.source_path(index)

    def target(self, index: int) -> np.ndarray | None:
        return self.base.target(index)

    def particle_label(self, index: int) -> float | None:
        return self.base.particle_label(index)

    def __getitem__(self, index: int):
        from .hetero_data import sample_to_hetero_data

        return sample_to_hetero_data(
            self.base[index],
            scalers=self.scalers,
            waveform_length=self.waveform_length,
        )
