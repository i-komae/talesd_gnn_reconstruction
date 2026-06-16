from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .constants import (
    EDGE_FEATURE_COLUMNS,
    GEOM_MIN_POINTS,
    NODE_FEATURE_COLUMNS,
    PULSE_FEATURE_COLUMNS,
    WAVEFORM_FEATURE_CHANNELS,
    WAVEFORM_RISE_ANCHOR_BIN,
    WAVEFORM_TRACE_BINS,
)
from .event_graph import GraphEvent
from .event_graph import _build_edges
from .event_graph import _copy_rise_aligned_window
from .event_graph import _local_detector_context
from .graph_io import create_graph_file, write_graph
from .hetero_graph_io import H5HeteroGraphDataset, hetero_dataset_class_for_paths


def _column_index(columns: Sequence[str], name: str) -> int | None:
    try:
        return list(columns).index(name)
    except ValueError:
        return None


def _column_value(
    values: np.ndarray,
    columns: Sequence[str],
    name: str,
    *,
    row: int | None = None,
    default: float = 0.0,
) -> float:
    index = _column_index(columns, name)
    if index is None:
        return float(default)
    if row is None:
        value = values[index]
    else:
        value = values[row, index]
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _selected_pulse_mask(sample: dict[str, Any], pulse_columns: Sequence[str], pulse_mask: str) -> np.ndarray:
    pulse_features = np.asarray(sample["pulse_features"], dtype=np.float32)
    n_pulse = int(pulse_features.shape[0])
    if pulse_mask == "all":
        return np.ones(n_pulse, dtype=bool)
    if pulse_mask == "valid":
        log_index = _column_index(pulse_columns, "log10_pulse_rho")
        if log_index is None:
            return np.ones(n_pulse, dtype=bool)
        return np.isfinite(pulse_features[:, log_index])
    if pulse_mask == "ising_kept":
        keep_index = _column_index(pulse_columns, "ising_keep")
        if keep_index is None:
            raise ValueError("pulse_mask='ising_kept' requires pulse feature column 'ising_keep'")
        return np.isfinite(pulse_features[:, keep_index]) & (pulse_features[:, keep_index] >= 0.5)
    raise ValueError(f"unsupported pulse_mask: {pulse_mask!r}")


def _copy_pulse_mask_window(
    pulse_bounds: np.ndarray,
    *,
    channel: str,
    rise_anchor_bin: int,
    source_length: int,
) -> np.ndarray:
    out = np.zeros(WAVEFORM_TRACE_BINS, dtype=np.float32)
    if pulse_bounds.size == 0:
        return out
    if channel == "upper":
        start_col, end_col = 0, 1
    elif channel == "lower":
        start_col, end_col = 2, 3
    else:
        raise ValueError(f"unknown channel: {channel}")
    window_start = int(rise_anchor_bin) - int(WAVEFORM_RISE_ANCHOR_BIN)
    window_end = window_start + int(WAVEFORM_TRACE_BINS)
    for bounds in np.asarray(pulse_bounds, dtype=np.float32):
        if bounds.shape[0] < 4 or not np.all(np.isfinite(bounds[:4])):
            continue
        start = int(round(float(bounds[start_col])))
        end = int(round(float(bounds[end_col])))
        src_start = max(start, window_start, 0)
        src_end = min(end, window_end, int(source_length))
        if src_end <= src_start:
            continue
        dst_start = src_start - window_start
        dst_end = src_end - window_start
        out[dst_start:dst_end] = 1.0
    return out


def _waveform_features_from_detector_waveform(
    detector_waveform: np.ndarray,
    pulse_bounds: np.ndarray,
    selected_bounds_for_detector: np.ndarray,
) -> np.ndarray:
    detector_waveform = np.asarray(detector_waveform, dtype=np.float32)
    if detector_waveform.ndim != 2 or detector_waveform.shape[0] < 2:
        return np.zeros((len(WAVEFORM_FEATURE_CHANNELS), WAVEFORM_TRACE_BINS), dtype=np.float32)
    if pulse_bounds.shape[0] < 4 or not np.all(np.isfinite(pulse_bounds[:4])):
        return np.zeros((len(WAVEFORM_FEATURE_CHANNELS), WAVEFORM_TRACE_BINS), dtype=np.float32)

    upper = detector_waveform[0]
    lower = detector_waveform[1]
    rise_anchor = int(round(float(min(pulse_bounds[0], pulse_bounds[2]))))
    upper_window = _copy_rise_aligned_window(upper, rise_anchor)
    lower_window = _copy_rise_aligned_window(lower, rise_anchor)
    upper_mask = _copy_pulse_mask_window(
        selected_bounds_for_detector,
        channel="upper",
        rise_anchor_bin=rise_anchor,
        source_length=int(upper.shape[0]),
    )
    lower_mask = _copy_pulse_mask_window(
        selected_bounds_for_detector,
        channel="lower",
        rise_anchor_bin=rise_anchor,
        source_length=int(lower.shape[0]),
    )
    return np.stack([upper_window, lower_window, upper_mask, lower_mask], axis=0).astype(np.float32, copy=False)


def _event_id_from_sample(sample: dict[str, Any], index: int) -> str:
    metadata = dict(sample.get("metadata") or sample.get("attrs") or {})
    value = metadata.get("event_id")
    if value is None:
        return f"converted_{int(index):08d}"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def hetero_sample_to_homogeneous_graph(
    sample: dict[str, Any],
    *,
    columns: dict[str, Any],
    index: int,
    pulse_mask: str = "ising_kept",
) -> GraphEvent | None:
    detector_features = np.asarray(sample["detector_features"], dtype=np.float32)
    detector_context = np.asarray(sample["detector_context_features"], dtype=np.float32)
    detector_positions = np.asarray(sample["detector_positions_km"], dtype=np.float32)
    detector_waveforms = np.asarray(sample["detector_waveforms"], dtype=np.float32)
    detector_lids = np.asarray(sample["detector_lids"], dtype=np.int64)
    pulse_features_all = np.asarray(sample["pulse_features"], dtype=np.float32)
    pulse_positions_all = np.asarray(sample["pulse_positions_km"], dtype=np.float32)
    pulse_lids_all = np.asarray(sample["pulse_lids"], dtype=np.int64)
    pulse_detector_index_all = np.asarray(sample["pulse_detector_index"], dtype=np.int64)
    pulse_bounds_all = np.asarray(sample["pulse_bounds"], dtype=np.float32)

    detector_columns = list(columns.get("detector_features", []))
    detector_context_columns = list(columns.get("detector_context_features", []))
    pulse_columns = list(columns.get("pulse_features", []))

    keep = _selected_pulse_mask(sample, pulse_columns, pulse_mask)
    if not np.any(keep):
        return None
    selected_indices = np.flatnonzero(keep)
    positions = pulse_positions_all[selected_indices].astype(np.float32, copy=False)
    detector_ids = [int(value) for value in pulse_lids_all[selected_indices]]
    unique_detector_count = len(set(detector_ids))
    if positions.shape[0] < GEOM_MIN_POINTS or unique_detector_count < GEOM_MIN_POINTS:
        return None

    selected_pulse_features = pulse_features_all[selected_indices]
    selected_detector_index = pulse_detector_index_all[selected_indices]
    selected_bounds = pulse_bounds_all[selected_indices]
    local_context = _local_detector_context(positions, detector_ids)

    log_rho_col = _column_index(pulse_columns, "log10_pulse_rho")
    sqrt_rho_col = _column_index(pulse_columns, "sqrt_pulse_rho")
    if log_rho_col is not None:
        log_rho = selected_pulse_features[:, log_rho_col].astype(np.float32)
        rho = np.power(np.float32(10.0), log_rho).astype(np.float32, copy=False)
    elif sqrt_rho_col is not None:
        sqrt_rho = selected_pulse_features[:, sqrt_rho_col].astype(np.float32)
        rho = np.square(np.maximum(sqrt_rho, 0.0)).astype(np.float32)
        log_rho = np.log10(np.maximum(rho, 1.0e-6)).astype(np.float32)
    else:
        rho = np.ones(selected_pulse_features.shape[0], dtype=np.float32)
        log_rho = np.zeros(selected_pulse_features.shape[0], dtype=np.float32)
    sqrt_rho = np.sqrt(np.maximum(rho, 0.0)).astype(np.float32)

    weights = np.maximum(rho, 0.0) + 1.0e-6
    bary = np.sum(positions * weights[:, None], axis=0) / np.sum(weights)
    delta = positions - bary[None, :]
    radius = np.linalg.norm(delta, axis=1)

    arrival = np.asarray(
        [
            _column_value(selected_pulse_features[i], pulse_columns, "pulse_arrival_usec_rel", default=0.0)
            for i in range(selected_pulse_features.shape[0])
        ],
        dtype=np.float32,
    )
    arrival = arrival - float(np.min(arrival))

    bounds_by_detector: dict[int, np.ndarray] = {}
    for detector_index in sorted(set(int(value) for value in selected_detector_index)):
        bounds_by_detector[detector_index] = selected_bounds[selected_detector_index == detector_index]

    node_rows: list[list[float]] = []
    pulse_rows: list[list[float]] = []
    waveform_rows: list[np.ndarray] = []
    for row, pulse_index in enumerate(selected_indices):
        detector_index = int(pulse_detector_index_all[pulse_index])
        if detector_index < 0 or detector_index >= detector_features.shape[0]:
            return None
        det_features = detector_features[detector_index]
        det_context = detector_context[detector_index] if detector_index < detector_context.shape[0] else np.zeros(0)
        detector_waveform = (
            detector_waveforms[detector_index]
            if detector_index < detector_waveforms.shape[0]
            else np.zeros((2, WAVEFORM_TRACE_BINS), dtype=np.float32)
        )

        node_rows.append(
            [
                float(positions[row, 0]),
                float(positions[row, 1]),
                float(positions[row, 2]),
                float(local_context[row, 0]),
                float(local_context[row, 1]),
                float(local_context[row, 2]),
                float(delta[row, 0]),
                float(delta[row, 1]),
                float(delta[row, 2]),
                float(radius[row]),
                float(arrival[row]),
                _column_value(det_features, detector_columns, "detector_trigger_usec_rel", default=0.0),
                float(log_rho[row]),
                float(sqrt_rho[row]),
                _column_value(det_features, detector_columns, "log10_detector_max_pulse_rho", default=0.0),
                _column_value(det_features, detector_columns, "log10_detector_sum_pulse_rho", default=0.0),
                _column_value(det_features, detector_columns, "sqrt_detector_sum_pulse_rho", default=0.0),
                _column_value(det_features, detector_columns, "detector_accepted_pulse_count", default=1.0),
                _column_value(det_features, detector_columns, "detector_accepted_pulse_time_span_usec", default=0.0),
                _column_value(det_context, detector_context_columns, "detector_wf_segments", default=1.0),
                _column_value(det_context, detector_context_columns, "detector_wf_length_usec", default=0.0),
                _column_value(det_context, detector_context_columns, "log10_detector_fadc_peak", default=0.0),
                _column_value(det_context, detector_context_columns, "detector_upper_ped", default=0.0),
                _column_value(det_context, detector_context_columns, "detector_lower_ped", default=0.0),
                _column_value(det_context, detector_context_columns, "detector_upper_ped_sigma", default=0.0),
                _column_value(det_context, detector_context_columns, "detector_lower_ped_sigma", default=0.0),
                _column_value(selected_pulse_features[row], pulse_columns, "accepted_pulse_order", default=float(row)),
                _column_value(
                    selected_pulse_features[row],
                    pulse_columns,
                    "is_first_accepted_pulse",
                    default=1.0 if row == 0 else 0.0,
                ),
            ]
        )
        pulse_rows.append([float(row)])
        waveform_rows.append(
            _waveform_features_from_detector_waveform(
                detector_waveform,
                pulse_bounds_all[pulse_index],
                bounds_by_detector.get(detector_index, np.zeros((0, 4), dtype=np.float32)),
            )
        )

    node_features = np.asarray(node_rows, dtype=np.float32)
    pulse_features = np.asarray(pulse_rows, dtype=np.float32)
    waveform_features = np.asarray(waveform_rows, dtype=np.float32)
    edge_index, edge_features = _build_edges(positions, arrival, log_rho, rho, detector_ids)

    metadata = dict(sample.get("metadata") or sample.get("attrs") or {})
    event_id = _event_id_from_sample(sample, index)
    metadata.update(
        {
            "event_id": event_id,
            "n_nodes": int(node_features.shape[0]),
            "n_sd": int(unique_detector_count),
            "n_pulses": int(node_features.shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "lids": ",".join(str(value) for value in pulse_lids_all[selected_indices].tolist()),
            "unique_lids": ",".join(str(value) for value in sorted(set(detector_ids))),
            "graph_definition": "hetero_v3_to_homogeneous_ising_pulse_graph_v1",
            "source_graph_definition": str(metadata.get("graph_definition", "")),
            "homogeneous_conversion_pulse_mask": str(pulse_mask),
            "hetero_n_detector_nodes": int(detector_features.shape[0]),
            "hetero_n_pulse_nodes": int(pulse_features_all.shape[0]),
            "hetero_selected_pulse_nodes": int(node_features.shape[0]),
            "hetero_dropped_pulse_nodes": int(pulse_features_all.shape[0] - node_features.shape[0]),
            "homogeneous_waveform_reconstructed_from_detector_waveform": True,
        }
    )
    particle_label = sample.get("particle_label")
    if particle_label is not None:
        try:
            metadata["particle_label"] = float(particle_label)
            metadata["particle_name"] = "iron" if float(particle_label) >= 0.5 else "proton"
        except Exception:
            pass
    target = sample.get("target")
    return GraphEvent(
        event_id=event_id,
        node_features=node_features,
        node_positions_km=positions,
        node_lids=pulse_lids_all[selected_indices].astype(np.int64, copy=False),
        edge_index=edge_index,
        edge_features=edge_features,
        pulse_features=pulse_features,
        waveform_features=waveform_features,
        target=None if target is None else np.asarray(target, dtype=np.float32),
        particle_label=None if particle_label is None else float(particle_label),
        metadata=metadata,
    )


def _shard_path(output: Path, shard_index: int) -> Path:
    if output.suffix == ".h5":
        if shard_index == 0:
            return output
        return output.with_name(f"{output.stem}_{shard_index:04d}{output.suffix}")
    return output / f"graphs_{shard_index:04d}.h5"


def _worker_shard_path(output: Path, worker_index: int, shard_index: int) -> Path:
    return output / f"worker_{int(worker_index):04d}_{int(shard_index):04d}.h5"


def _cleanup_output_h5_files(output: Path, *, overwrite: bool) -> None:
    if not overwrite or output.suffix == ".h5" or not output.exists():
        return
    for stale in output.glob("*.h5"):
        stale.unlink()


def _hetero_event_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("format", "")) == "talesd_gnn_hetero_graphs":
            return int(len(handle["events"]))
        if str(handle.attrs.get("format", "")) == "talesd_gnn_hetero_flat_cache":
            return int(handle["target_all"].shape[0])
    raise ValueError(f"{path} is not a supported hetero graph HDF5 file")


def _convert_worker(payload: dict[str, Any]) -> dict[str, Any]:
    input_paths = [Path(path).expanduser() for path in payload["input_paths"]]
    output_path = Path(payload["output"]).expanduser()
    worker_index = int(payload["worker_index"])
    pulse_mask = str(payload["pulse_mask"])
    shard_size = max(int(payload["shard_size"]), 1)
    max_events = payload.get("max_events")
    overwrite = bool(payload["overwrite"])
    progress_interval_sec = float(payload["progress_interval_sec"])

    dataset_cls = hetero_dataset_class_for_paths(input_paths)
    dataset = dataset_cls(
        input_paths,
        require_target=True,
        require_particle_label=True,
        load_attrs=True,
        core_target_mode="absolute",
        coordinate_feature_mode="absolute_and_relative",
    )
    total = len(dataset) if max_events is None or int(max_events) <= 0 else min(len(dataset), int(max_events))
    print(
        "convert hetero to homogeneous worker: "
        f"stage=start worker={worker_index} inputs={len(input_paths)} events={total} "
        f"output={output_path}",
        flush=True,
    )
    written = 0
    skipped = 0
    shard_index = 0
    handle: h5py.File | None = None
    last_log = time.monotonic()
    start = last_log
    shard_paths: list[str] = []

    def open_shard(index: int) -> h5py.File:
        if output_path.suffix == ".h5":
            path = _shard_path(output_path, index)
        else:
            path = _worker_shard_path(output_path, worker_index, index)
        if path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        shard_paths.append(str(path))
        return create_graph_file(
            path,
            config={
                "source_format": "talesd_gnn_hetero_graphs",
                "conversion": "hetero_to_homogeneous",
                "pulse_mask": pulse_mask,
                "input": [str(path) for path in input_paths],
                "conversion_worker_index": worker_index,
            },
        )

    try:
        handle = open_shard(shard_index)
        for index in range(total):
            sample = dataset[index]
            graph = hetero_sample_to_homogeneous_graph(
                sample,
                columns=dataset.columns,
                index=index,
                pulse_mask=pulse_mask,
            )
            if graph is None:
                skipped += 1
            else:
                if written > 0 and written % shard_size == 0:
                    handle.close()
                    shard_index += 1
                    handle = open_shard(shard_index)
                write_graph(handle, written % shard_size, graph)
                written += 1

            now = time.monotonic()
            if progress_interval_sec > 0 and (now - last_log) >= progress_interval_sec:
                elapsed = max(now - start, 1.0e-9)
                rate = (index + 1) / elapsed
                remaining = max(total - index - 1, 0)
                eta = remaining / rate if rate > 0.0 else float("nan")
                print(
                    "convert hetero to homogeneous worker: "
                    f"worker={worker_index} processed={index + 1}/{total} "
                    f"written={written} skipped={skipped} rate={rate:.6g}/s eta={eta:.0f}s",
                    flush=True,
                )
                last_log = now
        if handle is not None:
            handle.close()
            handle = None
    finally:
        if handle is not None:
            handle.close()
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    return {
        "worker_index": worker_index,
        "input": [str(path) for path in input_paths],
        "processed": int(total),
        "written": int(written),
        "skipped": int(skipped),
        "shards": int(len(shard_paths)),
        "shard_paths": shard_paths,
    }


def convert_hetero_to_homogeneous(
    paths: Sequence[str | Path],
    output: str | Path,
    *,
    pulse_mask: str = "ising_kept",
    shard_size: int = 100000,
    max_events: int | None = None,
    overwrite: bool = False,
    progress_interval_sec: float = 30.0,
    workers: int = 1,
) -> dict[str, Any]:
    input_paths = [Path(path).expanduser() for path in paths]
    if not input_paths:
        raise ValueError("at least one hetero HDF5 input path is required")
    output_path = Path(output).expanduser()
    if output_path.exists() and output_path.is_file() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite")
    if output_path.suffix == ".h5":
        if int(workers) > 1:
            raise ValueError("parallel conversion requires a directory output, not a single .h5 file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    _cleanup_output_h5_files(output_path, overwrite=overwrite)

    workers = min(max(int(workers), 1), len(input_paths))
    if workers > 1:
        path_limits: list[tuple[Path, int | None]] = []
        remaining = None if max_events is None or int(max_events) <= 0 else int(max_events)
        for path in input_paths:
            if remaining is None:
                path_limits.append((path, None))
                continue
            if remaining <= 0:
                break
            count = _hetero_event_count(path)
            take = min(count, remaining)
            path_limits.append((path, take))
            remaining -= take
        payloads = [
            {
                "worker_index": index,
                "input_paths": [str(path)],
                "output": str(output_path),
                "pulse_mask": pulse_mask,
                "shard_size": shard_size,
                "max_events": path_max_events,
                "overwrite": overwrite,
                "progress_interval_sec": progress_interval_sec,
            }
            for index, (path, path_max_events) in enumerate(path_limits)
        ]
        print(
            "convert hetero to homogeneous: "
            f"stage=start mode=parallel input_shards={len(payloads)} workers={workers} "
            f"output={output_path}",
            flush=True,
        )
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_convert_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    "convert hetero to homogeneous: "
                    f"stage=done_worker worker={result['worker_index']} processed={result['processed']} "
                    f"written={result['written']} skipped={result['skipped']} shards={result['shards']}",
                    flush=True,
                )
        ordered = sorted(results, key=lambda item: int(item["worker_index"]))
        summary = {
            "format": "talesd_gnn_homogeneous_from_hetero_v1",
            "input": [str(path) for path in input_paths],
            "output": str(output_path),
            "pulse_mask": pulse_mask,
            "processed": int(sum(int(item["processed"]) for item in ordered)),
            "written": int(sum(int(item["written"]) for item in ordered)),
            "skipped": int(sum(int(item["skipped"]) for item in ordered)),
            "workers": int(workers),
            "shards": int(sum(int(item["shards"]) for item in ordered)),
            "shard_paths": [path for item in ordered for path in item["shard_paths"]],
            "worker_results": ordered,
            "node_features": list(NODE_FEATURE_COLUMNS),
            "edge_features": list(EDGE_FEATURE_COLUMNS),
            "pulse_features": list(PULSE_FEATURE_COLUMNS),
            "waveform_features": list(WAVEFORM_FEATURE_CHANNELS),
        }
        summary_dir = output_path / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with (summary_dir / "hetero_to_homogeneous_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
        print(
            "convert hetero to homogeneous: "
            f"stage=done mode=parallel processed={summary['processed']} written={summary['written']} "
            f"skipped={summary['skipped']} workers={workers} shards={summary['shards']} output={output_path}",
            flush=True,
        )
        return summary

    print(
        "convert hetero to homogeneous: "
        f"stage=start mode=serial input_shards={len(input_paths)} workers=1 output={output_path}",
        flush=True,
    )
    dataset_cls = hetero_dataset_class_for_paths(input_paths)
    dataset = dataset_cls(
        input_paths,
        require_target=True,
        require_particle_label=True,
        load_attrs=True,
        core_target_mode="absolute",
        coordinate_feature_mode="absolute_and_relative",
    )
    if not isinstance(dataset, H5HeteroGraphDataset):
        # Flat caches are supported only when they were created in full mode.
        # Accessing __getitem__ below will raise a clear error if metadata is absent.
        pass

    total = len(dataset) if max_events is None or int(max_events) <= 0 else min(len(dataset), int(max_events))
    shard_size = max(int(shard_size), 1)
    written = 0
    skipped = 0
    shard_index = 0
    handle: h5py.File | None = None
    last_log = time.monotonic()
    start = last_log
    shard_paths: list[str] = []

    def open_shard(index: int) -> h5py.File:
        path = _shard_path(output_path, index)
        if path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        shard_paths.append(str(path))
        return create_graph_file(
            path,
            config={
                "source_format": "talesd_gnn_hetero_graphs",
                "conversion": "hetero_to_homogeneous",
                "pulse_mask": pulse_mask,
                "input": [str(path) for path in input_paths],
            },
        )

    try:
        handle = open_shard(shard_index)
        for index in range(total):
            sample = dataset[index]
            graph = hetero_sample_to_homogeneous_graph(
                sample,
                columns=dataset.columns,
                index=index,
                pulse_mask=pulse_mask,
            )
            if graph is None:
                skipped += 1
            else:
                if written > 0 and written % shard_size == 0:
                    handle.close()
                    shard_index += 1
                    handle = open_shard(shard_index)
                write_graph(handle, written % shard_size, graph)
                written += 1

            now = time.monotonic()
            if progress_interval_sec > 0 and (now - last_log) >= progress_interval_sec:
                elapsed = max(now - start, 1.0e-9)
                rate = (index + 1) / elapsed
                remaining = max(total - index - 1, 0)
                eta = remaining / rate if rate > 0.0 else float("nan")
                print(
                    "convert hetero to homogeneous: "
                    f"processed={index + 1}/{total} written={written} skipped={skipped} "
                    f"rate={rate:.6g}/s eta={eta:.0f}s",
                    flush=True,
                )
                last_log = now
        if handle is not None:
            handle.close()
            handle = None
    finally:
        if handle is not None:
            handle.close()
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    summary = {
        "format": "talesd_gnn_homogeneous_from_hetero_v1",
        "input": [str(path) for path in input_paths],
        "output": str(output_path),
        "pulse_mask": pulse_mask,
        "processed": int(total),
        "written": int(written),
        "skipped": int(skipped),
        "workers": 1,
        "shards": int(len(shard_paths)),
        "shard_paths": shard_paths,
        "node_features": list(NODE_FEATURE_COLUMNS),
        "edge_features": list(EDGE_FEATURE_COLUMNS),
        "pulse_features": list(PULSE_FEATURE_COLUMNS),
        "waveform_features": list(WAVEFORM_FEATURE_CHANNELS),
    }
    if output_path.suffix != ".h5":
        summary_dir = output_path / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        with (summary_dir / "hetero_to_homogeneous_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(
        "convert hetero to homogeneous: "
        f"done processed={total} written={written} skipped={skipped} shards={len(shard_paths)} output={output_path}",
        flush=True,
    )
    return summary
