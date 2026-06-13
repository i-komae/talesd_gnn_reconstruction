from __future__ import annotations

import json
import os
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import h5py
import numpy as np

from .core_coordinates import core_anchor_from_sample
from .core_coordinates import filter_feature_matrix
from .core_coordinates import filtered_columns
from .core_coordinates import normalize_coordinate_feature_mode
from .core_coordinates import normalize_core_target_mode
from .core_coordinates import parse_columns_json
from .core_coordinates import transform_core_target


FORMAT_NAME = "talesd_gnn_hetero_graphs"
FLAT_FORMAT_NAME = "talesd_gnn_hetero_graphs_flat"
FORMAT_VERSION = "0.1"
GRAPH_DEFINITION = "tale_sd_hetero_ising_pulse_detector_graph_v3"
WAVEFORM_SCHEMA = "detector_full_calibrated_vem_v1"

EDGE_RELATIONS = (
    "pulse__same_detector_next__pulse",
    "pulse__same_detector_prev__pulse",
    "pulse__near_space__pulse",
    "pulse__time_causal__pulse",
    "detector__near__detector",
    "detector__observes__pulse",
    "pulse__observed_by__detector",
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


def _ensure_metadata_size(metadata_group: h5py.Group, size: int) -> None:
    for dataset in metadata_group.values():
        if dataset.shape[0] < size:
            dataset.resize((size,))


def _metadata_dataset_value(source: h5py.File, source_local_index: int, name: str) -> Any | None:
    metadata_group = source.get("metadata")
    if metadata_group is None or name not in metadata_group:
        return None
    dataset = metadata_group[name]
    if source_local_index >= dataset.shape[0]:
        return None
    return dataset[source_local_index]


def copy_hetero_graph_group(
    source: h5py.File,
    source_key: str,
    source_local_index: int,
    target: h5py.File,
    target_index: int,
) -> None:
    """Copy one hetero event group and rebuild the flat metadata row."""

    target_key = f"{int(target_index):08d}"
    source.copy(source["events"][source_key], target["events"], name=target_key)

    metadata_group = target.get("metadata")
    if metadata_group is None:
        return
    _ensure_metadata_size(metadata_group, int(target_index) + 1)
    copied_group = target["events"][target_key]
    metadata: dict[str, Any] = {}
    if "metadata_json" in copied_group.attrs:
        try:
            metadata = json.loads(str(copied_group.attrs["metadata_json"]))
        except json.JSONDecodeError:
            metadata = {}

    for name in metadata_group.keys():
        value = _metadata_dataset_value(source, int(source_local_index), name)
        if value is None:
            if name == "event_id":
                value = metadata.get("event_id", str(copied_group.attrs.get("event_id", "")))
            elif name == "source_path":
                value = metadata.get("source_path", str(copied_group.attrs.get("source_path", "")))
            elif name == "source_index":
                value = int(metadata.get("source_index", copied_group.attrs.get("source_index", -1)))
            elif name == "parttype":
                value = int(metadata.get("parttype", copied_group.attrs.get("parttype", -1)))
            elif name == "particle_label":
                value = copied_group["particle_label"][()] if "particle_label" in copied_group else np.nan
            elif name == "n_detector_nodes":
                value = int(copied_group["detector_features"].shape[0])
            elif name == "n_pulse_nodes":
                value = int(copied_group["pulse_features"].shape[0])
            elif name == "metadata_json":
                value = str(copied_group.attrs.get("metadata_json", _metadata_json(metadata)))
            else:
                continue
        metadata_group[name][int(target_index)] = value


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
        graph_format = str(handle.attrs.get("format", ""))
        if graph_format == FLAT_FORMAT_NAME:
            if "detector_offsets" in handle:
                return int(handle["detector_offsets"].shape[0] - 1)
            return int(handle["offsets/detector"].shape[0] - 1)
        return len(handle["events"])


def hetero_h5_format(path: str | Path) -> str:
    with h5py.File(Path(path).expanduser(), "r") as handle:
        return str(handle.attrs.get("format", ""))


def _compression_name(dataset: h5py.Dataset | None) -> str:
    if dataset is None:
        return "unknown"
    compression = dataset.compression
    return "none" if compression in {None, ""} else str(compression)


def _grouped_h5_compression(handle: h5py.File) -> str:
    events = handle.get("events")
    if events is None or not events:
        return "unknown"
    first_key = sorted(events.keys())[0]
    group = events[first_key]
    dataset = group.get("detector_features")
    return _compression_name(dataset if isinstance(dataset, h5py.Dataset) else None)


def _flat_h5_compression(handle: h5py.File) -> str:
    dataset = handle.get("detector_features_all")
    if not isinstance(dataset, h5py.Dataset):
        dataset = handle.get("arrays/detector_features")
    return _compression_name(dataset if isinstance(dataset, h5py.Dataset) else None)


def _log_h5_layout(*, graph_path: Path, format_label: str, compression: str, n_events: int) -> None:
    print(
        "hetero_graph_io "
        f"format={format_label} "
        f"compression={compression} "
        f"n_events={int(n_events)} "
        f"path={graph_path}",
        flush=True,
    )


def _metadata_from_group(group: h5py.Group) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "metadata_json" in group.attrs:
        try:
            metadata.update(json.loads(str(group.attrs["metadata_json"])))
        except json.JSONDecodeError:
            pass
    for key, value in group.attrs.items():
        if key != "metadata_json":
            metadata[str(key)] = value
    return metadata


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _progress_interval_from_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    return float(text)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _maybe_log_flat_cache_progress(
    *,
    stage: str,
    index: int,
    total: int,
    start_time: float,
    last_log_time: float,
    interval_sec: float,
    force: bool = False,
    detector_nodes: int | None = None,
    pulse_nodes: int | None = None,
    edges: int | None = None,
) -> float:
    now = time.monotonic()
    if not force and (interval_sec <= 0.0 or (now - last_log_time) < interval_sec):
        return last_log_time
    elapsed = max(now - start_time, 0.0)
    rate = float(index) / elapsed if elapsed > 0.0 else 0.0
    remaining = max(int(total) - int(index), 0)
    eta = float(remaining) / rate if rate > 0.0 else float("nan")
    extras = ""
    if detector_nodes is not None:
        extras += f" detector_nodes={int(detector_nodes)}"
    if pulse_nodes is not None:
        extras += f" pulse_nodes={int(pulse_nodes)}"
    if edges is not None:
        extras += f" edges={int(edges)}"
    print(
        "hetero_flat_cache_progress "
        f"stage={stage} "
        f"index={int(index)}/{int(total)} "
        f"events_per_sec={rate:.6g}{extras} "
        f"elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta) if np.isfinite(eta) else 'unknown'}",
        flush=True,
    )
    return now


def hetero_dataset_class_for_paths(path: str | Path | Sequence[str | Path]) -> type:
    paths = [Path(path).expanduser()] if isinstance(path, (str, Path)) else [Path(item).expanduser() for item in path]
    formats = {hetero_h5_format(item) for item in paths}
    if formats == {FORMAT_NAME}:
        return H5HeteroGraphDataset
    if formats == {FLAT_FORMAT_NAME}:
        return H5FlatHeteroGraphDataset
    raise ValueError(f"mixed or unsupported hetero HDF5 formats: {sorted(formats)}")


def _pad_waveforms(waveforms: np.ndarray, waveform_length: int) -> np.ndarray:
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.shape[-1] == waveform_length:
        return waveforms
    if waveforms.shape[-1] > waveform_length:
        return waveforms[..., :waveform_length]
    pad_shape = list(waveforms.shape)
    pad_shape[-1] = int(waveform_length) - int(waveforms.shape[-1])
    return np.concatenate([waveforms, np.zeros(pad_shape, dtype=np.float32)], axis=-1)


def _verification_indices(n_events: int, verify_samples: int) -> list[int]:
    n_events = int(n_events)
    verify_samples = int(verify_samples)
    if n_events <= 0 or verify_samples <= 0:
        return []
    if n_events <= verify_samples:
        return list(range(n_events))
    positions = np.linspace(0, n_events - 1, num=verify_samples, dtype=np.int64)
    return [int(position) for position in positions]


def _assert_array_equal_or_close(name: str, left: np.ndarray, right: np.ndarray) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(f"flat cache verification failed for {name}: shape {left_array.shape} != {right_array.shape}")
    if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(right_array.dtype, np.floating):
        if not np.allclose(left_array, right_array, rtol=1.0e-6, atol=1.0e-6, equal_nan=True):
            raise ValueError(f"flat cache verification failed for {name}: values differ")
    elif not np.array_equal(left_array, right_array):
        raise ValueError(f"flat cache verification failed for {name}: values differ")


def _verify_flat_cache_samples(
    source: "H5HeteroGraphDataset",
    flat_path: Path,
    *,
    verify_samples: int,
    cache_mode: str = "training",
    core_anchor_mode: str = "absolute",
    progress_interval_sec: float = 60.0,
) -> int:
    indices = _verification_indices(len(source), verify_samples)
    if not indices:
        return 0
    flat = H5FlatHeteroGraphDataset(
        flat_path,
        require_target=source.require_target,
        load_attrs=False,
        core_target_mode=source.core_target_mode,
        core_anchor_mode=core_anchor_mode,
    )
    try:
        start = time.monotonic()
        last_log = start
        print(
            "hetero_flat_cache_verify_start "
            f"path={flat_path} samples={len(indices)} progress_interval_sec={float(progress_interval_sec):.6g}",
            flush=True,
        )
        for ordinal, index in enumerate(indices, start=1):
            if cache_mode == "training":
                grouped = source.training_sample(int(index))
                cached = flat.training_sample(int(index))
            else:
                grouped = source[int(index)]
                cached = flat[int(index)]
            for key in (
                "detector_features",
                "detector_context_features",
                "pulse_features",
                "pulse_detector_index",
                "pulse_bounds",
                "target",
                "core_anchor",
            ):
                _assert_array_equal_or_close(f"{key}[{index}]", grouped[key], cached[key])
            grouped_label = np.asarray(
                [np.nan if grouped["particle_label"] is None else float(grouped["particle_label"])],
                dtype=np.float32,
            )
            cached_label = np.asarray(
                [np.nan if cached["particle_label"] is None else float(cached["particle_label"])],
                dtype=np.float32,
            )
            _assert_array_equal_or_close(f"particle_label[{index}]", grouped_label, cached_label)
            for relation in EDGE_RELATIONS:
                _assert_array_equal_or_close(
                    f"edge_index_by_type/{relation}[{index}]",
                    grouped["edge_index_by_type"][relation],
                    cached["edge_index_by_type"][relation],
                )
                _assert_array_equal_or_close(
                    f"edge_features_by_type/{relation}[{index}]",
                    grouped["edge_features_by_type"][relation],
                    cached["edge_features_by_type"][relation],
                )
            grouped_waveform = np.asarray(grouped["detector_waveforms"], dtype=np.float32)
            cached_waveform = np.asarray(cached["detector_waveforms"], dtype=np.float32)
            if cached_waveform.shape[:2] != grouped_waveform.shape[:2] or cached_waveform.shape[2] < grouped_waveform.shape[2]:
                raise ValueError(
                    "flat cache verification failed for detector_waveforms: "
                    f"grouped_shape={grouped_waveform.shape} cached_shape={cached_waveform.shape}"
                )
            _assert_array_equal_or_close(
                f"detector_waveforms prefix[{index}]",
                grouped_waveform,
                cached_waveform[..., : grouped_waveform.shape[-1]],
            )
            last_log = _maybe_log_flat_cache_progress(
                stage="verify",
                index=ordinal,
                total=len(indices),
                start_time=start,
                last_log_time=last_log,
                interval_sec=float(progress_interval_sec),
                force=ordinal == len(indices),
            )
        print(
            "hetero_flat_cache_verify "
            f"samples={len(indices)} "
            f"elapsed_sec={time.monotonic() - start:.6g} "
            f"status=ok path={flat_path}",
            flush=True,
        )
        return len(indices)
    finally:
        flat.close()


def _flat_conversion_sample(source: Any, index: int, *, full_cache: bool) -> dict[str, Any]:
    if full_cache:
        return source[int(index)]
    if hasattr(source, "training_sample"):
        return source.training_sample(int(index))
    return source[int(index)]


def _dataset_axis_nbytes(dataset: h5py.Dataset, *, axis: int, count: int) -> int:
    count = max(int(count), 0)
    shape = list(dataset.shape)
    if not shape:
        return int(np.dtype(dataset.dtype).itemsize)
    axis = int(axis)
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f"axis {axis} is out of bounds for dataset shape={tuple(shape)}")
    shape[axis] = count
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dataset.dtype).itemsize)


def _dataset_rows_nbytes(dataset: h5py.Dataset, rows: int) -> int:
    return _dataset_axis_nbytes(dataset, axis=0, count=int(rows))


def convert_hetero_to_flat_cache(
    input_paths: str | Path | Sequence[str | Path],
    output_path: str | Path,
    *,
    compression: str | None = "none",
    cache_mode: str = "training",
    core_anchor_mode: str = "signal_bary_relative",
    max_graphs: int | None = None,
    verify_samples: int = 5,
    progress_interval_sec: float | None = None,
    allow_slow_cache: bool | None = None,
) -> dict[str, Any]:
    cache_mode = str(cache_mode).strip().lower()
    if cache_mode not in {"training", "full"}:
        raise ValueError(f"cache_mode must be training or full, got {cache_mode!r}")
    core_anchor_mode = normalize_core_target_mode(core_anchor_mode)
    if allow_slow_cache is None:
        allow_slow_cache = _env_flag("ALLOW_SLOW_CACHE", False)
    if progress_interval_sec is None:
        progress_interval_sec = _progress_interval_from_env("HETERO_FLAT_CACHE_PROGRESS_INTERVAL_SEC", 60.0)
    progress_interval_sec = float(progress_interval_sec)
    output = Path(output_path).expanduser()
    compression_label = "none" if compression in {None, "", "none"} else str(compression)
    total_start = time.monotonic()
    source = H5HeteroGraphDataset(
        input_paths,
        require_target=True,
        load_attrs=True,
        core_target_mode="absolute",
        core_anchor_mode=core_anchor_mode,
    )
    try:
        if len(source) == 0:
            raise ValueError("cannot convert empty hetero HDF5 dataset")
        n_events = int(len(source))
        if max_graphs is not None and int(max_graphs) > 0:
            n_events = min(n_events, int(max_graphs))
        full_cache = cache_mode == "full"
        print(
            "hetero_flat_cache_start "
            f"input={input_paths} output={output} compression={compression_label} "
            f"cache_mode={cache_mode} graphs={n_events} verify_samples={int(verify_samples)} "
            f"progress_interval_sec={progress_interval_sec:.6g}",
            flush=True,
        )
        if cache_mode == "training":
            print(
                "hetero_flat_cache_note "
                "cache_mode=training stores training tensors, pulse waveform references, and minimal split metadata; "
                "use grouped HDF5 for visualization and attention maps",
                flush=True,
            )
        if compression_label != "none":
            print(
                "WARNING: compressed post-hoc flat cache writes can be very slow; "
                "use --compression none for training caches.",
                flush=True,
            )
        detector_total = 0
        pulse_total = 0
        edge_totals = {relation: 0 for relation in EDGE_RELATIONS}
        waveform_channels = 0
        waveform_length = 0
        target_dim = 0
        detector_dim = 0
        detector_context_dim = 0
        pulse_dim = 0
        edge_dims = {relation: 0 for relation in EDGE_RELATIONS}
        count_start = time.monotonic()
        last_count_log = count_start
        print(
            "hetero_flat_cache_count_start "
            f"events={n_events}",
            flush=True,
        )
        for index in range(n_events):
            sample = _flat_conversion_sample(source, index, full_cache=full_cache)
            detector_total += int(sample["detector_features"].shape[0])
            pulse_total += int(sample["pulse_features"].shape[0])
            waveform_channels = max(waveform_channels, int(sample["detector_waveforms"].shape[1]))
            waveform_length = max(waveform_length, int(sample["detector_waveforms"].shape[2]))
            target_dim = max(target_dim, int(sample["target"].shape[0]) if sample["target"] is not None else 0)
            detector_dim = max(detector_dim, int(sample["detector_features"].shape[1]))
            detector_context_dim = max(detector_context_dim, int(sample["detector_context_features"].shape[1]))
            pulse_dim = max(pulse_dim, int(sample["pulse_features"].shape[1]))
            for relation in EDGE_RELATIONS:
                edge_totals[relation] += int(sample["edge_index_by_type"][relation].shape[1])
                edge_dims[relation] = max(edge_dims[relation], int(sample["edge_features_by_type"][relation].shape[1]))
            last_count_log = _maybe_log_flat_cache_progress(
                stage="count",
                index=index + 1,
                total=n_events,
                start_time=count_start,
                last_log_time=last_count_log,
                interval_sec=progress_interval_sec,
                force=(index + 1) == n_events,
                detector_nodes=detector_total,
                pulse_nodes=pulse_total,
                edges=sum(edge_totals.values()),
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        filter_kwargs = {} if compression in {None, "", "none"} else {"compression": compression}
        detector_offsets = np.zeros(n_events + 1, dtype=np.int64)
        pulse_offsets = np.zeros(n_events + 1, dtype=np.int64)
        edge_offsets = {relation: np.zeros(n_events + 1, dtype=np.int64) for relation in EDGE_RELATIONS}
        with h5py.File(output, "w") as handle:
            handle.attrs["format"] = FLAT_FORMAT_NAME
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["source_format"] = FORMAT_NAME
            handle.attrs["graph_definition"] = GRAPH_DEFINITION
            handle.attrs["waveform_schema"] = WAVEFORM_SCHEMA
            handle.attrs["columns_json"] = source.columns_json
            handle.attrs["cache_mode"] = cache_mode
            handle.attrs["core_anchor_mode"] = core_anchor_mode
            arrays = handle.create_group("arrays")
            offsets = handle.create_group("offsets")
            metadata = handle.create_group("metadata")
            arrays.create_dataset("detector_features", shape=(detector_total, detector_dim), dtype=np.float32, **filter_kwargs)
            arrays.create_dataset(
                "detector_context_features",
                shape=(detector_total, detector_context_dim),
                dtype=np.float32,
                **filter_kwargs,
            )
            if full_cache:
                arrays.create_dataset("detector_positions_km", shape=(detector_total, 3), dtype=np.float32, **filter_kwargs)
                arrays.create_dataset("detector_lids", shape=(detector_total,), dtype=np.int64, **filter_kwargs)
            arrays.create_dataset(
                "detector_waveforms",
                shape=(detector_total, waveform_channels, waveform_length),
                dtype=np.float32,
                **filter_kwargs,
            )
            arrays.create_dataset("pulse_features", shape=(pulse_total, pulse_dim), dtype=np.float32, **filter_kwargs)
            if full_cache:
                arrays.create_dataset("pulse_positions_km", shape=(pulse_total, 3), dtype=np.float32, **filter_kwargs)
                arrays.create_dataset("pulse_lids", shape=(pulse_total,), dtype=np.int64, **filter_kwargs)
            arrays.create_dataset("pulse_detector_index", shape=(pulse_total,), dtype=np.int64, **filter_kwargs)
            arrays.create_dataset("pulse_bounds", shape=(pulse_total, 4), dtype=np.float32, **filter_kwargs)
            arrays.create_dataset("target", shape=(n_events, target_dim), dtype=np.float32, **filter_kwargs)
            arrays.create_dataset("core_anchor", shape=(n_events, 2), dtype=np.float32, **filter_kwargs)
            arrays.create_dataset("particle_label", shape=(n_events,), dtype=np.float32, **filter_kwargs)
            edge_index_group = arrays.create_group("edge_index_by_type")
            edge_feature_group = arrays.create_group("edge_features_by_type")
            for relation in EDGE_RELATIONS:
                edge_index_group.create_dataset(
                    relation,
                    shape=(2, edge_totals[relation]),
                    dtype=np.int64,
                    **filter_kwargs,
                )
                edge_feature_group.create_dataset(
                    relation,
                    shape=(edge_totals[relation], edge_dims[relation]),
                    dtype=np.float32,
                    **filter_kwargs,
                )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            metadata.create_dataset("event_id", shape=(n_events,), dtype=string_dtype)
            metadata.create_dataset("source_path", shape=(n_events,), dtype=string_dtype)
            metadata.create_dataset("source_index", shape=(n_events,), dtype=np.int64)
            if full_cache:
                metadata.create_dataset("metadata_json", shape=(n_events,), dtype=string_dtype)

            # Root-level names are the public training-cache schema. The
            # arrays/offsets groups are kept as hard-link aliases so older
            # local cache readers can still open files created during the
            # transition.
            handle["detector_features_all"] = arrays["detector_features"]
            handle["detector_context_features_all"] = arrays["detector_context_features"]
            if full_cache:
                handle["detector_positions_km_all"] = arrays["detector_positions_km"]
                handle["detector_lids_all"] = arrays["detector_lids"]
            handle["detector_waveforms_all"] = arrays["detector_waveforms"]
            handle["pulse_features_all"] = arrays["pulse_features"]
            if full_cache:
                handle["pulse_positions_km_all"] = arrays["pulse_positions_km"]
                handle["pulse_lids_all"] = arrays["pulse_lids"]
            handle["pulse_detector_index_all"] = arrays["pulse_detector_index"]
            handle["pulse_bounds_all"] = arrays["pulse_bounds"]
            handle["target_all"] = arrays["target"]
            handle["core_anchor_all"] = arrays["core_anchor"]
            handle["particle_label_all"] = arrays["particle_label"]
            handle["edge_index_all_by_relation"] = edge_index_group
            handle["edge_features_all_by_relation"] = edge_feature_group

            detector_cursor = 0
            pulse_cursor = 0
            edge_cursors = {relation: 0 for relation in EDGE_RELATIONS}
            write_start = time.monotonic()
            last_write_log = write_start
            print(
                "hetero_flat_cache_write_start "
                f"events={n_events} detector_nodes={int(detector_total)} "
                f"pulse_nodes={int(pulse_total)} edges={int(sum(edge_totals.values()))}",
                flush=True,
            )
            for index in range(n_events):
                sample = _flat_conversion_sample(source, index, full_cache=full_cache)
                n_detector = int(sample["detector_features"].shape[0])
                n_pulse = int(sample["pulse_features"].shape[0])
                detector_slice = slice(detector_cursor, detector_cursor + n_detector)
                pulse_slice = slice(pulse_cursor, pulse_cursor + n_pulse)
                arrays["detector_features"][detector_slice] = sample["detector_features"]
                arrays["detector_context_features"][detector_slice] = sample["detector_context_features"]
                if full_cache:
                    arrays["detector_positions_km"][detector_slice] = sample["detector_positions_km"]
                    arrays["detector_lids"][detector_slice] = sample["detector_lids"]
                arrays["detector_waveforms"][detector_slice] = _pad_waveforms(
                    sample["detector_waveforms"],
                    waveform_length,
                )
                arrays["pulse_features"][pulse_slice] = sample["pulse_features"]
                if full_cache:
                    arrays["pulse_positions_km"][pulse_slice] = sample["pulse_positions_km"]
                    arrays["pulse_lids"][pulse_slice] = sample["pulse_lids"]
                arrays["pulse_detector_index"][pulse_slice] = sample["pulse_detector_index"]
                arrays["pulse_bounds"][pulse_slice] = sample["pulse_bounds"]
                arrays["target"][index] = sample["target"]
                core_anchor = np.asarray(sample.get("core_anchor", np.zeros((2,), dtype=np.float32)), dtype=np.float32)
                arrays["core_anchor"][index] = core_anchor.reshape(-1)[:2]
                arrays["particle_label"][index] = (
                    np.nan if sample["particle_label"] is None else float(sample["particle_label"])
                )
                for relation in EDGE_RELATIONS:
                    edge_index = sample["edge_index_by_type"][relation]
                    edge_features = sample["edge_features_by_type"][relation]
                    n_edges = int(edge_index.shape[1])
                    edge_slice = slice(edge_cursors[relation], edge_cursors[relation] + n_edges)
                    edge_index_group[relation][:, edge_slice] = edge_index
                    edge_feature_group[relation][edge_slice] = edge_features
                    edge_cursors[relation] += n_edges
                    edge_offsets[relation][index + 1] = edge_cursors[relation]
                detector_cursor += n_detector
                pulse_cursor += n_pulse
                detector_offsets[index + 1] = detector_cursor
                pulse_offsets[index + 1] = pulse_cursor
                metadata_dict = sample.get("metadata", {})
                event_id_value = metadata_dict.get("event_id", sample.get("event_id"))
                if event_id_value is None and hasattr(source, "_metadata_value"):
                    path_index, local_index, _key = source._locate(index)
                    event_id_value = source._metadata_value(path_index, local_index, "event_id")
                metadata["event_id"][index] = str(event_id_value if event_id_value is not None else f"{index:08d}")
                metadata["source_path"][index] = str(metadata_dict.get("source_path", source.source_path(index)))
                source_index_value = metadata_dict.get("source_index")
                if source_index_value is None and hasattr(source, "_metadata_value"):
                    path_index, local_index, _key = source._locate(index)
                    source_index_value = source._metadata_value(path_index, local_index, "source_index")
                metadata["source_index"][index] = -1 if source_index_value is None else int(source_index_value)
                if full_cache:
                    metadata["metadata_json"][index] = _metadata_json(metadata_dict)
                last_write_log = _maybe_log_flat_cache_progress(
                    stage="write",
                    index=index + 1,
                    total=n_events,
                    start_time=write_start,
                    last_log_time=last_write_log,
                    interval_sec=progress_interval_sec,
                    force=(index + 1) == n_events,
                    detector_nodes=detector_cursor,
                    pulse_nodes=pulse_cursor,
                    edges=sum(edge_cursors.values()),
                )
                write_elapsed = max(time.monotonic() - write_start, 0.0)
                if index + 1 > 0 and write_elapsed > 0.0:
                    write_rate = float(index + 1) / write_elapsed
                    eta_sec = float(n_events - (index + 1)) / write_rate if write_rate > 0.0 else float("inf")
                    if eta_sec > 1800.0 and (index + 1) in {1, 10, 100}:
                        print(
                            "WARNING: hetero_flat_cache_write_slow "
                            f"stage=write index={index + 1}/{n_events} events_per_sec={write_rate:.6g} "
                            f"eta={_format_duration(eta_sec)} cache_mode={cache_mode} compression={compression_label}",
                            flush=True,
                        )
                    if (
                        _env_flag("SPEED_BENCHMARK", False)
                        and eta_sec > 600.0
                        and not bool(allow_slow_cache)
                        and index + 1 >= 10
                    ):
                        raise RuntimeError(
                            "flat cache write ETA exceeds 10 minutes in SPEED_BENCHMARK; "
                            "rerun with PREPARE_FAST_CACHE=0 or ALLOW_SLOW_CACHE=1"
                        )
            offsets.create_dataset("detector", data=detector_offsets)
            offsets.create_dataset("pulse", data=pulse_offsets)
            edge_offsets_group = offsets.create_group("edge_by_type")
            for relation in EDGE_RELATIONS:
                edge_offsets_group.create_dataset(relation, data=edge_offsets[relation])
            handle["detector_offsets"] = offsets["detector"]
            handle["pulse_offsets"] = offsets["pulse"]
            handle["edge_offsets_by_relation"] = edge_offsets_group
        verified_samples = _verify_flat_cache_samples(
            source,
            output,
            verify_samples=int(verify_samples),
            cache_mode=cache_mode,
            core_anchor_mode=core_anchor_mode,
            progress_interval_sec=progress_interval_sec,
        )
        _log_h5_layout(
            graph_path=output,
            format_label="flat_hdf5",
            compression=compression_label,
            n_events=n_events,
        )
        elapsed_sec = time.monotonic() - total_start
        print(
            "hetero_flat_cache_done "
            f"output={output} graphs={n_events} detector_nodes={int(detector_total)} "
            f"pulse_nodes={int(pulse_total)} edges={int(sum(edge_totals.values()))} "
            f"compression={compression_label} cache_mode={cache_mode} core_anchor_mode={core_anchor_mode} "
            f"verified_samples={int(verified_samples)} "
            f"elapsed_sec={elapsed_sec:.6g} events_per_sec={float(n_events) / elapsed_sec if elapsed_sec > 0 else 0.0:.6g}",
            flush=True,
        )
        return {
            "output": str(output),
            "format": FLAT_FORMAT_NAME,
            "graphs": n_events,
            "detector_nodes": int(detector_total),
            "pulse_nodes": int(pulse_total),
            "waveform_channels": int(waveform_channels),
            "waveform_length": int(waveform_length),
            "compression": compression_label,
            "cache_mode": cache_mode,
            "core_anchor_mode": core_anchor_mode,
            "verified_samples": int(verified_samples),
            "elapsed_sec": float(elapsed_sec),
            "events_per_sec": float(n_events) / elapsed_sec if elapsed_sec > 0 else 0.0,
        }
    finally:
        source.close()


class H5HeteroGraphDataset:
    def __init__(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        require_target: bool = False,
        require_particle_label: bool = False,
        load_attrs: bool = True,
        core_target_mode: str = "absolute",
        coordinate_feature_mode: str = "absolute_and_relative",
        core_anchor_mode: str | None = None,
    ):
        if isinstance(path, (str, Path)):
            self.paths = [Path(path).expanduser()]
        else:
            self.paths = [Path(item).expanduser() for item in path]
        self.require_target = bool(require_target)
        self.require_particle_label = bool(require_particle_label)
        self.load_attrs = bool(load_attrs)
        self.core_target_mode = normalize_core_target_mode(core_target_mode)
        self.coordinate_feature_mode = normalize_coordinate_feature_mode(coordinate_feature_mode)
        self.core_anchor_mode = normalize_core_target_mode(core_anchor_mode or self.core_target_mode)
        self._handles: dict[int, h5py.File] = {}
        self._path_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        self._path_local_indices: list[list[int] | None] = []
        self._path_key_lists: list[list[str] | None] = []
        self.columns_json = "{}"
        self.columns: dict[str, Any] = {}

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
                compression = _grouped_h5_compression(handle)
                _log_h5_layout(
                    graph_path=graph_path,
                    format_label="grouped_hdf5",
                    compression=compression,
                    n_events=int(n_events),
                )
            total += n_events
            self._path_lengths.append(n_events)
            self._cumulative_lengths.append(total)
            self._path_local_indices.append(None)
            self._path_key_lists.append(None)
        self.columns = parse_columns_json(self.columns_json)

    def _filter_detector_features(self, values: np.ndarray) -> np.ndarray:
        return filter_feature_matrix(
            values,
            list(self.columns.get("detector_features", [])),
            self.coordinate_feature_mode,
        )

    def _filter_pulse_features(self, values: np.ndarray) -> np.ndarray:
        return filter_feature_matrix(
            values,
            list(self.columns.get("pulse_features", [])),
            self.coordinate_feature_mode,
        )

    def _effective_detector_columns(self) -> list[str]:
        return filtered_columns(list(self.columns.get("detector_features", [])), self.coordinate_feature_mode)

    def _effective_pulse_columns(self) -> list[str]:
        return filtered_columns(list(self.columns.get("pulse_features", [])), self.coordinate_feature_mode)

    def _anchor_from_sample(self, sample: dict[str, Any]) -> np.ndarray:
        return core_anchor_from_sample(
            sample,
            columns=self.columns,
            core_anchor_mode=self.core_anchor_mode,
        )

    def _prepare_target(self, target: np.ndarray | None, core_anchor: np.ndarray) -> np.ndarray | None:
        return transform_core_target(target, core_anchor, self.core_target_mode)

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

    def graph_training_nbytes(self, index: int) -> int:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        total = 0
        for name in (
            "detector_features",
            "detector_context_features",
            "detector_waveforms",
            "pulse_features",
        ):
            total += self._dataset_nbytes(group[name])
        if "target" in group:
            total += self._dataset_nbytes(group["target"])
        if "particle_label" in group:
            total += self._dataset_nbytes(group["particle_label"])
        for relation in EDGE_RELATIONS:
            total += self._dataset_nbytes(group["edge_index_by_type"][relation])
            total += self._dataset_nbytes(group["edge_features_by_type"][relation])
        return int(total)

    def scaler_sample(self, index: int) -> dict[str, Any]:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        target = group["target"][()].astype(np.float32) if "target" in group else None
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{key}")
        particle_label = self.particle_label(index)
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{key}")
        raw_pulse_features = group["pulse_features"][()].astype(np.float32)
        core_anchor = self._anchor_from_sample(
            {
                "pulse_features": raw_pulse_features,
                "pulse_positions_km": group["pulse_positions_km"][()].astype(np.float32),
                "metadata": _metadata_from_group(group),
            }
        )
        return {
            "detector_features": self._filter_detector_features(group["detector_features"][()].astype(np.float32)),
            "detector_context_features": group["detector_context_features"][()].astype(np.float32),
            "pulse_features": self._filter_pulse_features(raw_pulse_features),
            "edge_features_by_type": self._read_edge_group(group["edge_features_by_type"]),
            "target": self._prepare_target(target, core_anchor),
            "core_anchor": core_anchor,
            "particle_label": particle_label,
            "detector_feature_columns": self._effective_detector_columns(),
            "pulse_feature_columns": self._effective_pulse_columns(),
        }

    def training_sample(self, index: int) -> dict[str, Any]:
        path_index, _local_index, key = self._locate(index)
        group = self._handle(path_index)["events"][key]
        target = group["target"][()].astype(np.float32) if "target" in group else None
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{key}")
        particle_label = self.particle_label(index)
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{key}")
        raw_pulse_features = group["pulse_features"][()].astype(np.float32)
        core_anchor = self._anchor_from_sample(
            {
                "pulse_features": raw_pulse_features,
                "pulse_positions_km": group["pulse_positions_km"][()].astype(np.float32),
                "metadata": _metadata_from_group(group),
            }
        )
        return {
            "detector_features": self._filter_detector_features(group["detector_features"][()].astype(np.float32)),
            "detector_context_features": group["detector_context_features"][()].astype(np.float32),
            "detector_waveforms": group["detector_waveforms"][()].astype(np.float32),
            "pulse_features": self._filter_pulse_features(raw_pulse_features),
            "pulse_detector_index": group["pulse_detector_index"][()].astype(np.int64),
            "pulse_bounds": group["pulse_bounds"][()].astype(np.float32),
            "edge_index_by_type": self._read_edge_group(group["edge_index_by_type"]),
            "edge_features_by_type": self._read_edge_group(group["edge_features_by_type"]),
            "target": self._prepare_target(target, core_anchor),
            "core_anchor": core_anchor,
            "particle_label": particle_label,
            "detector_feature_columns": self._effective_detector_columns(),
            "pulse_feature_columns": self._effective_pulse_columns(),
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
        raw_pulse_features = group["pulse_features"][()].astype(np.float32)
        pulse_positions = group["pulse_positions_km"][()].astype(np.float32)
        metadata_for_anchor = _metadata_from_group(group)
        core_anchor = self._anchor_from_sample(
            {
                "pulse_features": raw_pulse_features,
                "pulse_positions_km": pulse_positions,
                "metadata": metadata_for_anchor,
            }
        )
        sample: dict[str, Any] = {
            "detector_features": self._filter_detector_features(group["detector_features"][()].astype(np.float32)),
            "detector_context_features": group["detector_context_features"][()].astype(np.float32),
            "detector_positions_km": group["detector_positions_km"][()].astype(np.float32),
            "detector_lids": group["detector_lids"][()].astype(np.int64),
            "detector_waveforms": group["detector_waveforms"][()].astype(np.float32),
            "pulse_features": self._filter_pulse_features(raw_pulse_features),
            "pulse_positions_km": pulse_positions,
            "pulse_lids": group["pulse_lids"][()].astype(np.int64),
            "pulse_detector_index": group["pulse_detector_index"][()].astype(np.int64),
            "pulse_bounds": group["pulse_bounds"][()].astype(np.float32),
            "edge_index_by_type": self._read_edge_group(group["edge_index_by_type"]),
            "edge_features_by_type": self._read_edge_group(group["edge_features_by_type"]),
            "target": self._prepare_target(target, core_anchor),
            "core_anchor": core_anchor,
            "particle_label": particle_label,
            "detector_feature_columns": self._effective_detector_columns(),
            "pulse_feature_columns": self._effective_pulse_columns(),
        }
        if self.load_attrs:
            sample["attrs"] = dict(group.attrs.items())
            sample["metadata"] = metadata_for_anchor
        return sample


class H5FlatHeteroGraphDataset:
    def __init__(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        require_target: bool = False,
        require_particle_label: bool = False,
        load_attrs: bool = True,
        core_target_mode: str = "absolute",
        coordinate_feature_mode: str = "absolute_and_relative",
        core_anchor_mode: str | None = None,
    ):
        if isinstance(path, (str, Path)):
            self.paths = [Path(path).expanduser()]
        else:
            self.paths = [Path(item).expanduser() for item in path]
        self.require_target = bool(require_target)
        self.require_particle_label = bool(require_particle_label)
        self.load_attrs = bool(load_attrs)
        self.core_target_mode = normalize_core_target_mode(core_target_mode)
        self.coordinate_feature_mode = normalize_coordinate_feature_mode(coordinate_feature_mode)
        self.core_anchor_mode = normalize_core_target_mode(core_anchor_mode or self.core_target_mode)
        self._handles: dict[int, h5py.File] = {}
        self._path_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        self._path_local_indices: list[list[int] | None] = []
        self._path_key_lists: list[list[str] | None] = []
        self._path_cache_modes: list[str] = []
        self._path_scan_chunkable = True
        self.columns_json = "{}"
        self.columns: dict[str, Any] = {}
        total = 0
        for path_index, graph_path in enumerate(self.paths):
            with h5py.File(graph_path, "r") as handle:
                if str(handle.attrs.get("format", "")) != FLAT_FORMAT_NAME:
                    raise ValueError(f"{graph_path} is not a flat hetero graph HDF5 cache")
                if str(handle.attrs.get("graph_definition", "")) != GRAPH_DEFINITION:
                    raise ValueError(f"{graph_path} stores unsupported graph_definition")
                if str(handle.attrs.get("waveform_schema", "")) != WAVEFORM_SCHEMA:
                    raise ValueError(f"{graph_path} stores unsupported waveform_schema")
                if path_index == 0:
                    self.columns_json = str(handle.attrs.get("columns_json", "{}"))
                stored_anchor_mode = str(handle.attrs.get("core_anchor_mode", "absolute")).strip().lower() or "absolute"
                stored_anchor_mode = normalize_core_target_mode(stored_anchor_mode)
                if self.core_target_mode != "absolute" and stored_anchor_mode != self.core_target_mode:
                    raise ValueError(
                        f"{graph_path} stores core_anchor_mode={stored_anchor_mode!r}, "
                        f"but core_target_mode={self.core_target_mode!r} was requested; "
                        "rebuild the flat cache with matching --core-anchor-mode"
                    )
                n_events = int(self._offset_dataset(handle, "detector").shape[0] - 1)
                cache_mode = str(handle.attrs.get("cache_mode", "full")).strip().lower() or "full"
                compression = _flat_h5_compression(handle)
                _log_h5_layout(
                    graph_path=graph_path,
                    format_label="flat_hdf5",
                    compression=compression,
                    n_events=int(n_events),
                )
            total += n_events
            self._path_lengths.append(n_events)
            self._cumulative_lengths.append(total)
            self._path_local_indices.append(None)
            self._path_key_lists.append(None)
            self._path_cache_modes.append(cache_mode)
        self.columns = parse_columns_json(self.columns_json)

    def _filter_detector_features(self, values: np.ndarray) -> np.ndarray:
        return filter_feature_matrix(
            values,
            list(self.columns.get("detector_features", [])),
            self.coordinate_feature_mode,
        )

    def _filter_pulse_features(self, values: np.ndarray) -> np.ndarray:
        return filter_feature_matrix(
            values,
            list(self.columns.get("pulse_features", [])),
            self.coordinate_feature_mode,
        )

    def _effective_detector_columns(self) -> list[str]:
        return filtered_columns(list(self.columns.get("detector_features", [])), self.coordinate_feature_mode)

    def _effective_pulse_columns(self) -> list[str]:
        return filtered_columns(list(self.columns.get("pulse_features", [])), self.coordinate_feature_mode)

    def _core_anchor(self, handle: h5py.File, local_index: int) -> np.ndarray:
        if self.core_anchor_mode == "absolute":
            return np.zeros((2,), dtype=np.float32)
        if "core_anchor_all" in handle:
            return handle["core_anchor_all"][local_index].astype(np.float32).reshape(-1)[:2]
        arrays = handle.get("arrays")
        if arrays is not None and "core_anchor" in arrays:
            return arrays["core_anchor"][local_index].astype(np.float32).reshape(-1)[:2]
        if self.core_target_mode != "absolute":
            raise ValueError(
                "relative core target mode needs core_anchor_all in flat HDF5 cache; "
                "rebuild the flat cache from grouped HDF5 with the current converter"
            )
        return np.zeros((2,), dtype=np.float32)

    def _prepare_target(self, target: np.ndarray | None, core_anchor: np.ndarray) -> np.ndarray | None:
        return transform_core_target(target, core_anchor, self.core_target_mode)

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

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path_index = bisect_right(self._cumulative_lengths, index)
        previous = self._cumulative_lengths[path_index - 1] if path_index > 0 else 0
        return path_index, index - previous

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _slice_offsets(handle: h5py.File, name: str, local_index: int) -> slice:
        offsets = H5FlatHeteroGraphDataset._offset_dataset(handle, name)
        return slice(int(offsets[local_index]), int(offsets[local_index + 1]))

    @staticmethod
    def _array_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
        public_name = f"{name}_all"
        if public_name in handle:
            return handle[public_name]
        return handle[f"arrays/{name}"]

    @staticmethod
    def _has_array_dataset(handle: h5py.File, name: str) -> bool:
        public_name = f"{name}_all"
        arrays = handle.get("arrays")
        return public_name in handle or (arrays is not None and name in arrays)

    @staticmethod
    def _offset_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
        public_name = f"{name}_offsets"
        if public_name in handle:
            return handle[public_name]
        return handle[f"offsets/{name}"]

    @staticmethod
    def _edge_index_dataset(handle: h5py.File, relation: str) -> h5py.Dataset:
        if "edge_index_all_by_relation" in handle:
            return handle["edge_index_all_by_relation"][relation]
        return handle["arrays/edge_index_by_type"][relation]

    @staticmethod
    def _edge_feature_dataset(handle: h5py.File, relation: str) -> h5py.Dataset:
        if "edge_features_all_by_relation" in handle:
            return handle["edge_features_all_by_relation"][relation]
        return handle["arrays/edge_features_by_type"][relation]

    @staticmethod
    def _edge_offset_dataset(handle: h5py.File, relation: str) -> h5py.Dataset:
        if "edge_offsets_by_relation" in handle:
            return handle["edge_offsets_by_relation"][relation]
        return handle["offsets/edge_by_type"][relation]

    def source_path(self, index: int) -> str:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        return self._decode_text(handle["metadata/source_path"][local_index])

    def target(self, index: int) -> np.ndarray | None:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        arrays = handle.get("arrays")
        if "target_all" not in handle and (arrays is None or "target" not in arrays):
            return None
        return self._array_dataset(handle, "target")[local_index].astype(np.float32)

    def particle_label(self, index: int) -> float | None:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        arrays = handle.get("arrays")
        if "particle_label_all" not in handle and (arrays is None or "particle_label" not in arrays):
            return None
        value = float(self._array_dataset(handle, "particle_label")[local_index])
        return value if np.isfinite(value) else None

    def detector_waveform_shape(self, index: int) -> tuple[int, ...]:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        detector_slice = self._slice_offsets(handle, "detector", local_index)
        n_detector = int(detector_slice.stop - detector_slice.start)
        waveform = self._array_dataset(handle, "detector_waveforms")
        return (n_detector, int(waveform.shape[1]), int(waveform.shape[2]))

    def graph_nbytes(self, index: int) -> int:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        detector_slice = self._slice_offsets(handle, "detector", local_index)
        pulse_slice = self._slice_offsets(handle, "pulse", local_index)
        n_detector = int(detector_slice.stop - detector_slice.start)
        n_pulse = int(pulse_slice.stop - pulse_slice.start)
        total = 0
        for name, rows in (
            ("detector_features", n_detector),
            ("detector_context_features", n_detector),
            ("detector_positions_km", n_detector),
            ("detector_lids", n_detector),
            ("detector_waveforms", n_detector),
            ("pulse_features", n_pulse),
            ("pulse_positions_km", n_pulse),
            ("pulse_lids", n_pulse),
            ("pulse_detector_index", n_pulse),
            ("pulse_bounds", n_pulse),
        ):
            if not self._has_array_dataset(handle, name):
                continue
            dataset = self._array_dataset(handle, name)
            total += _dataset_rows_nbytes(dataset, rows)
        arrays = handle.get("arrays")
        if "target_all" in handle or (arrays is not None and "target" in arrays):
            total += _dataset_rows_nbytes(self._array_dataset(handle, "target"), 1)
        if "core_anchor_all" in handle or (arrays is not None and "core_anchor" in arrays):
            total += _dataset_rows_nbytes(self._array_dataset(handle, "core_anchor"), 1)
        if "particle_label_all" in handle or (arrays is not None and "particle_label" in arrays):
            total += _dataset_rows_nbytes(self._array_dataset(handle, "particle_label"), 1)
        for relation in EDGE_RELATIONS:
            offsets = self._edge_offset_dataset(handle, relation)
            n_edges = int(offsets[local_index + 1] - offsets[local_index])
            total += _dataset_axis_nbytes(self._edge_index_dataset(handle, relation), axis=1, count=n_edges)
            total += _dataset_rows_nbytes(self._edge_feature_dataset(handle, relation), n_edges)
        return int(total)

    def graph_training_nbytes(self, index: int) -> int:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        detector_slice = self._slice_offsets(handle, "detector", local_index)
        pulse_slice = self._slice_offsets(handle, "pulse", local_index)
        n_detector = int(detector_slice.stop - detector_slice.start)
        n_pulse = int(pulse_slice.stop - pulse_slice.start)
        total = 0
        for name, rows in (
            ("detector_features", n_detector),
            ("detector_context_features", n_detector),
            ("detector_waveforms", n_detector),
            ("pulse_features", n_pulse),
            ("pulse_detector_index", n_pulse),
            ("pulse_bounds", n_pulse),
        ):
            if not self._has_array_dataset(handle, name):
                continue
            dataset = self._array_dataset(handle, name)
            total += _dataset_rows_nbytes(dataset, rows)
        arrays = handle.get("arrays")
        if "target_all" in handle or (arrays is not None and "target" in arrays):
            total += _dataset_rows_nbytes(self._array_dataset(handle, "target"), 1)
        if "core_anchor_all" in handle or (arrays is not None and "core_anchor" in arrays):
            total += _dataset_rows_nbytes(self._array_dataset(handle, "core_anchor"), 1)
        if "particle_label_all" in handle or (arrays is not None and "particle_label" in arrays):
            total += _dataset_rows_nbytes(self._array_dataset(handle, "particle_label"), 1)
        for relation in EDGE_RELATIONS:
            offsets = self._edge_offset_dataset(handle, relation)
            n_edges = int(offsets[local_index + 1] - offsets[local_index])
            total += _dataset_axis_nbytes(self._edge_index_dataset(handle, relation), axis=1, count=n_edges)
            total += _dataset_rows_nbytes(self._edge_feature_dataset(handle, relation), n_edges)
        return int(total)

    def scaler_sample(self, index: int) -> dict[str, Any]:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        detector_slice = self._slice_offsets(handle, "detector", local_index)
        pulse_slice = self._slice_offsets(handle, "pulse", local_index)
        edge_features_by_type = {}
        for relation in EDGE_RELATIONS:
            offsets = self._edge_offset_dataset(handle, relation)
            edge_slice = slice(int(offsets[local_index]), int(offsets[local_index + 1]))
            edge_features_by_type[relation] = self._edge_feature_dataset(handle, relation)[edge_slice].astype(np.float32)
        arrays = handle.get("arrays")
        target = (
            self._array_dataset(handle, "target")[local_index].astype(np.float32)
            if "target_all" in handle or (arrays is not None and "target" in arrays)
            else None
        )
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{local_index}")
        particle_label = self.particle_label(index)
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{local_index}")
        core_anchor = self._core_anchor(handle, local_index)
        return {
            "detector_features": self._filter_detector_features(
                self._array_dataset(handle, "detector_features")[detector_slice].astype(np.float32)
            ),
            "detector_context_features": self._array_dataset(handle, "detector_context_features")[detector_slice].astype(np.float32),
            "pulse_features": self._filter_pulse_features(
                self._array_dataset(handle, "pulse_features")[pulse_slice].astype(np.float32)
            ),
            "edge_features_by_type": edge_features_by_type,
            "target": self._prepare_target(target, core_anchor),
            "core_anchor": core_anchor,
            "particle_label": particle_label,
            "detector_feature_columns": self._effective_detector_columns(),
            "pulse_feature_columns": self._effective_pulse_columns(),
        }

    def training_sample(self, index: int) -> dict[str, Any]:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        detector_slice = self._slice_offsets(handle, "detector", local_index)
        pulse_slice = self._slice_offsets(handle, "pulse", local_index)
        edge_index_by_type = {}
        edge_features_by_type = {}
        for relation in EDGE_RELATIONS:
            offsets = self._edge_offset_dataset(handle, relation)
            edge_slice = slice(int(offsets[local_index]), int(offsets[local_index + 1]))
            edge_index_by_type[relation] = self._edge_index_dataset(handle, relation)[:, edge_slice].astype(np.int64)
            edge_features_by_type[relation] = self._edge_feature_dataset(handle, relation)[edge_slice].astype(np.float32)
        arrays = handle.get("arrays")
        target = (
            self._array_dataset(handle, "target")[local_index].astype(np.float32)
            if "target_all" in handle or (arrays is not None and "target" in arrays)
            else None
        )
        particle_label = self.particle_label(index)
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{local_index}")
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{local_index}")
        core_anchor = self._core_anchor(handle, local_index)
        sample = {
            "detector_features": self._filter_detector_features(
                self._array_dataset(handle, "detector_features")[detector_slice].astype(np.float32)
            ),
            "detector_context_features": self._array_dataset(handle, "detector_context_features")[detector_slice].astype(np.float32),
            "detector_waveforms": self._array_dataset(handle, "detector_waveforms")[detector_slice].astype(np.float32),
            "pulse_features": self._filter_pulse_features(
                self._array_dataset(handle, "pulse_features")[pulse_slice].astype(np.float32)
            ),
            "edge_index_by_type": edge_index_by_type,
            "edge_features_by_type": edge_features_by_type,
            "target": self._prepare_target(target, core_anchor),
            "core_anchor": core_anchor,
            "particle_label": particle_label,
            "detector_feature_columns": self._effective_detector_columns(),
            "pulse_feature_columns": self._effective_pulse_columns(),
        }
        if self._has_array_dataset(handle, "pulse_detector_index"):
            sample["pulse_detector_index"] = self._array_dataset(handle, "pulse_detector_index")[pulse_slice].astype(np.int64)
        if self._has_array_dataset(handle, "pulse_bounds"):
            sample["pulse_bounds"] = self._array_dataset(handle, "pulse_bounds")[pulse_slice].astype(np.float32)
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        path_index, local_index = self._locate(index)
        handle = self._handle(path_index)
        full_required = (
            "detector_positions_km",
            "detector_lids",
            "pulse_positions_km",
            "pulse_lids",
            "pulse_detector_index",
            "pulse_bounds",
        )
        missing = [name for name in full_required if not self._has_array_dataset(handle, name)]
        if missing:
            cache_mode = self._path_cache_modes[path_index] if path_index < len(self._path_cache_modes) else "training"
            raise ValueError(
                "flat hetero cache does not contain full graph metadata "
                f"(cache_mode={cache_mode}, missing={','.join(missing)}). "
                "Use H5TensorHeteroGraphDataset/training_sample() for training, "
                "or rebuild the cache with --cache-mode full for PyG/visualization access."
            )
        detector_slice = self._slice_offsets(handle, "detector", local_index)
        pulse_slice = self._slice_offsets(handle, "pulse", local_index)
        edge_index_by_type = {}
        edge_features_by_type = {}
        for relation in EDGE_RELATIONS:
            offsets = self._edge_offset_dataset(handle, relation)
            edge_slice = slice(int(offsets[local_index]), int(offsets[local_index + 1]))
            edge_index_by_type[relation] = self._edge_index_dataset(handle, relation)[:, edge_slice].astype(np.int64)
            edge_features_by_type[relation] = self._edge_feature_dataset(handle, relation)[edge_slice].astype(np.float32)
        arrays = handle.get("arrays")
        target = (
            self._array_dataset(handle, "target")[local_index].astype(np.float32)
            if "target_all" in handle or (arrays is not None and "target" in arrays)
            else None
        )
        particle_label = self.particle_label(index)
        if self.require_target and target is None:
            raise ValueError(f"graph has no target: {self.paths[path_index]}::{local_index}")
        if self.require_particle_label and particle_label is None:
            raise ValueError(f"graph has no particle label: {self.paths[path_index]}::{local_index}")
        core_anchor = self._core_anchor(handle, local_index)
        sample: dict[str, Any] = {
            "detector_features": self._filter_detector_features(
                self._array_dataset(handle, "detector_features")[detector_slice].astype(np.float32)
            ),
            "detector_context_features": self._array_dataset(handle, "detector_context_features")[detector_slice].astype(np.float32),
            "detector_positions_km": self._array_dataset(handle, "detector_positions_km")[detector_slice].astype(np.float32),
            "detector_lids": self._array_dataset(handle, "detector_lids")[detector_slice].astype(np.int64),
            "detector_waveforms": self._array_dataset(handle, "detector_waveforms")[detector_slice].astype(np.float32),
            "pulse_features": self._filter_pulse_features(
                self._array_dataset(handle, "pulse_features")[pulse_slice].astype(np.float32)
            ),
            "pulse_positions_km": self._array_dataset(handle, "pulse_positions_km")[pulse_slice].astype(np.float32),
            "pulse_lids": self._array_dataset(handle, "pulse_lids")[pulse_slice].astype(np.int64),
            "pulse_detector_index": self._array_dataset(handle, "pulse_detector_index")[pulse_slice].astype(np.int64),
            "pulse_bounds": self._array_dataset(handle, "pulse_bounds")[pulse_slice].astype(np.float32),
            "edge_index_by_type": edge_index_by_type,
            "edge_features_by_type": edge_features_by_type,
            "target": self._prepare_target(target, core_anchor),
            "core_anchor": core_anchor,
            "particle_label": particle_label,
            "detector_feature_columns": self._effective_detector_columns(),
            "pulse_feature_columns": self._effective_pulse_columns(),
        }
        if self.load_attrs:
            metadata = {
                "event_id": self._decode_text(handle["metadata/event_id"][local_index]),
                "source_path": self._decode_text(handle["metadata/source_path"][local_index]),
                "source_index": int(handle["metadata/source_index"][local_index]),
            }
            if "metadata_json" in handle["metadata"]:
                metadata_json = self._decode_text(handle["metadata/metadata_json"][local_index])
                if metadata_json:
                    try:
                        metadata.update(json.loads(metadata_json))
                    except json.JSONDecodeError:
                        pass
            sample["metadata"] = metadata
            sample["attrs"] = dict(metadata)
        return sample


class H5PyGHeteroGraphDataset:
    def __init__(
        self,
        *args: Any,
        scalers: dict[str, Any] | None = None,
        waveform_length: int | None = None,
        **kwargs: Any,
    ):
        dataset_class = hetero_dataset_class_for_paths(args[0])
        self.base = dataset_class(*args, **kwargs)
        self.scalers = scalers
        self.waveform_length = None if waveform_length is None else int(waveform_length)

    def __len__(self) -> int:
        return len(self.base)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "base": self.base.__getstate__(),
            "base_class": self.base.__class__.__name__,
            "scalers": self.scalers,
            "waveform_length": self.waveform_length,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        base_class = H5FlatHeteroGraphDataset if state.get("base_class") == "H5FlatHeteroGraphDataset" else H5HeteroGraphDataset
        self.base = base_class.__new__(base_class)
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
