#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talesd_gnn_reconstruction.train import source_group_key  # noqa: E402


ARRAY_DATASETS = (
    "node_features",
    "node_positions_km",
    "node_lids",
    "edge_index",
    "edge_features",
    "pulse_features",
    "waveform_features",
    "target",
    "particle_label",
)


@dataclasses.dataclass(frozen=True, slots=True)
class EventEntry:
    h5_path: str
    local_index: int
    event_key: str
    event_id: str
    source_path: str
    source_index: int
    source_group: str


def _expand_h5_paths(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            expanded.extend(str(match) for match in sorted(path.rglob("*.h5")))
            continue
        if path.exists():
            expanded.append(str(path))
            continue
        if path.suffix == ".h5":
            matches = sorted(path.parent.glob(f"{path.stem}_*{path.suffix}"))
        elif path.suffix:
            matches = []
        else:
            matches = sorted(path.parent.glob(f"{path.name}_*.h5"))
        expanded.extend(str(match) for match in matches)
    return list(dict.fromkeys(expanded))


def _decode_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _metadata_value(handle: h5py.File, local_index: int, name: str) -> Any | None:
    metadata = handle.get("metadata")
    if metadata is None or name not in metadata:
        return None
    dataset = metadata[name]
    if local_index >= int(dataset.shape[0]):
        return None
    return dataset[local_index]


def _event_keys(handle: h5py.File) -> list[str]:
    events = handle.get("events")
    if events is None:
        raise ValueError("HDF5 file has no events group")
    keys = sorted(events.keys())
    if not keys:
        return []
    expected = [f"{index:08d}" for index in range(len(keys))]
    if keys == expected:
        return expected
    return keys


def _event_entry(handle: h5py.File, h5_path: str, local_index: int, event_key: str) -> EventEntry:
    group = handle["events"][event_key]
    event_id_value = _metadata_value(handle, local_index, "event_id")
    source_path_value = _metadata_value(handle, local_index, "source_path")
    source_index_value = _metadata_value(handle, local_index, "source_index")
    event_id = _decode_text(event_id_value, str(group.attrs.get("event_id", event_key)))
    source_path = _decode_text(source_path_value, str(group.attrs.get("source_path", "")))
    try:
        source_index = int(source_index_value) if source_index_value is not None else int(group.attrs.get("source_index", local_index))
    except Exception:
        source_index = int(local_index)
    source_group = source_group_key(source_path)
    return EventEntry(
        h5_path=str(h5_path),
        local_index=int(local_index),
        event_key=str(event_key),
        event_id=str(event_id),
        source_path=str(source_path),
        source_index=int(source_index),
        source_group=str(source_group),
    )


def _match_key(entry: EventEntry, mode: str) -> str:
    if mode == "event_id":
        return entry.event_id
    if mode == "source_path_index":
        return f"{entry.source_path}\0{entry.source_index}"
    if mode == "source_group_index":
        return f"{entry.source_group}\0{entry.source_index}"
    if mode == "source_group_event_id":
        return f"{entry.source_group}\0{entry.event_id}"
    raise ValueError(f"unknown match key mode: {mode}")


def _score_key(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{key}".encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _progress(message: str) -> None:
    print(message, flush=True)


def _build_reference_index(
    paths: list[str],
    *,
    match_key: str,
    progress_interval_sec: float,
) -> tuple[dict[str, EventEntry], dict[str, Any]]:
    start = time.monotonic()
    last = start
    index: dict[str, EventEntry] = {}
    duplicate_examples: list[dict[str, Any]] = []
    duplicate_count = 0
    total_events = 0
    for path_index, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            keys = _event_keys(handle)
            for local_index, event_key in enumerate(keys):
                entry = _event_entry(handle, path, local_index, event_key)
                key = _match_key(entry, match_key)
                total_events += 1
                if key in index:
                    duplicate_count += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append(
                            {
                                "key": key,
                                "first": dataclasses.asdict(index[key]),
                                "duplicate": dataclasses.asdict(entry),
                            }
                        )
                    continue
                index[key] = entry
                now = time.monotonic()
                if progress_interval_sec > 0 and now - last >= progress_interval_sec:
                    elapsed = max(now - start, 1e-9)
                    _progress(
                        "audit_h5_inputs stage=scan_reference "
                        f"files={path_index + 1}/{len(paths)} events={total_events} "
                        f"unique_keys={len(index)} duplicates={duplicate_count} "
                        f"elapsed={elapsed:.1f}s rate={total_events / elapsed:.1f}/s"
                    )
                    last = now
    elapsed = max(time.monotonic() - start, 1e-9)
    return index, {
        "files": len(paths),
        "events": total_events,
        "unique_keys": len(index),
        "duplicates": duplicate_count,
        "duplicate_examples": duplicate_examples,
        "elapsed_sec": elapsed,
    }


def _sample_candidate_overlaps(
    paths: list[str],
    reference_index: dict[str, EventEntry],
    *,
    match_key: str,
    sample_size: int,
    seed: int,
    progress_interval_sec: float,
) -> tuple[list[tuple[str, EventEntry, EventEntry]], set[str], dict[str, Any]]:
    start = time.monotonic()
    last = start
    total_events = 0
    overlap_count = 0
    matched_reference_keys: set[str] = set()
    heap: list[tuple[int, str, EventEntry, EventEntry]] = []
    missing_reference_examples: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            keys = _event_keys(handle)
            for local_index, event_key in enumerate(keys):
                candidate = _event_entry(handle, path, local_index, event_key)
                key = _match_key(candidate, match_key)
                total_events += 1
                reference = reference_index.get(key)
                if reference is None:
                    if len(missing_reference_examples) < 20:
                        missing_reference_examples.append({"key": key, "candidate": dataclasses.asdict(candidate)})
                else:
                    overlap_count += 1
                    matched_reference_keys.add(key)
                    if sample_size > 0:
                        score = _score_key(seed, key)
                        item = (-score, key, reference, candidate)
                        if len(heap) < sample_size:
                            heapq.heappush(heap, item)
                        elif item[0] > heap[0][0]:
                            heapq.heapreplace(heap, item)
                now = time.monotonic()
                if progress_interval_sec > 0 and now - last >= progress_interval_sec:
                    elapsed = max(now - start, 1e-9)
                    _progress(
                        "audit_h5_inputs stage=scan_candidate "
                        f"files={path_index + 1}/{len(paths)} events={total_events} "
                        f"overlap={overlap_count} sampled={len(heap)} "
                        f"elapsed={elapsed:.1f}s rate={total_events / elapsed:.1f}/s"
                    )
                    last = now
    samples = [(key, reference, candidate) for _score, key, reference, candidate in sorted(heap, reverse=True)]
    elapsed = max(time.monotonic() - start, 1e-9)
    return samples, matched_reference_keys, {
        "files": len(paths),
        "events": total_events,
        "overlap_keys": overlap_count,
        "candidate_without_reference_examples": missing_reference_examples,
        "elapsed_sec": elapsed,
    }


def _read_event_arrays(entry: EventEntry, dataset_names: list[str]) -> dict[str, np.ndarray | None]:
    with h5py.File(entry.h5_path, "r") as handle:
        group = handle["events"][entry.event_key]
        return {name: (group[name][()] if name in group else None) for name in dataset_names}


def _finite_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _finite_number(value.item())
    return value


def _array_diff(reference: np.ndarray | None, candidate: np.ndarray | None) -> dict[str, Any]:
    if reference is None or candidate is None:
        return {
            "reference_missing": reference is None,
            "candidate_missing": candidate is None,
            "shape_match": False,
            "dtype_reference": None if reference is None else str(reference.dtype),
            "dtype_candidate": None if candidate is None else str(candidate.dtype),
        }
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": tuple(reference.shape) == tuple(candidate.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
    }
    if tuple(reference.shape) != tuple(candidate.shape):
        return result
    if reference.size == 0 and candidate.size == 0:
        result.update({"exact_equal": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "mismatch_count": 0})
        return result
    if np.issubdtype(reference.dtype, np.number) and np.issubdtype(candidate.dtype, np.number):
        ref = np.asarray(reference)
        cand = np.asarray(candidate)
        if np.issubdtype(ref.dtype, np.floating) or np.issubdtype(cand.dtype, np.floating):
            diff = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
            finite = np.isfinite(diff)
            max_abs = float(np.max(diff[finite])) if np.any(finite) else None
            mean_abs = float(np.mean(diff[finite])) if np.any(finite) else None
            mismatch_count = int(np.count_nonzero(~np.isclose(ref, cand, rtol=1e-6, atol=1e-7, equal_nan=True)))
        else:
            diff = ref != cand
            max_abs = float(np.max(np.abs(ref.astype(np.int64) - cand.astype(np.int64)))) if ref.size else 0.0
            mean_abs = float(np.mean(np.abs(ref.astype(np.int64) - cand.astype(np.int64)))) if ref.size else 0.0
            mismatch_count = int(np.count_nonzero(diff))
        result.update(
            {
                "exact_equal": bool(np.array_equal(reference, candidate, equal_nan=True)),
                "max_abs_diff": _finite_number(max_abs),
                "mean_abs_diff": _finite_number(mean_abs),
                "mismatch_count": mismatch_count,
            }
        )
    else:
        equal = bool(np.array_equal(reference, candidate))
        result.update({"exact_equal": equal, "mismatch_count": 0 if equal else int(reference.size)})
    return result


def _merge_array_stats(stats: dict[str, Any], dataset_name: str, diff: dict[str, Any]) -> None:
    item = stats.setdefault(
        dataset_name,
        {
            "compared": 0,
            "reference_missing": 0,
            "candidate_missing": 0,
            "shape_mismatch": 0,
            "dtype_mismatch": 0,
            "exact_equal": 0,
            "any_value_mismatch": 0,
            "max_abs_diff": None,
            "mean_abs_diff_max": None,
            "mismatch_count_total": 0,
        },
    )
    item["compared"] += 1
    item["reference_missing"] += int(bool(diff.get("reference_missing", False)))
    item["candidate_missing"] += int(bool(diff.get("candidate_missing", False)))
    item["shape_mismatch"] += int(not bool(diff.get("shape_match", False)))
    item["dtype_mismatch"] += int(
        diff.get("dtype_reference") is not None
        and diff.get("dtype_candidate") is not None
        and diff.get("dtype_reference") != diff.get("dtype_candidate")
    )
    item["exact_equal"] += int(bool(diff.get("exact_equal", False)))
    mismatch_count = int(diff.get("mismatch_count", 0) or 0)
    item["mismatch_count_total"] += mismatch_count
    if mismatch_count or not bool(diff.get("shape_match", False)):
        item["any_value_mismatch"] += 1
    for source_key, target_key in (("max_abs_diff", "max_abs_diff"), ("mean_abs_diff", "mean_abs_diff_max")):
        value = diff.get(source_key)
        if value is None:
            continue
        item[target_key] = float(value) if item[target_key] is None else max(float(item[target_key]), float(value))


def _compare_samples(
    samples: list[tuple[str, EventEntry, EventEntry]],
    *,
    skip_waveforms: bool,
    examples_output: str | None,
) -> dict[str, Any]:
    dataset_names = [name for name in ARRAY_DATASETS if not (skip_waveforms and name == "waveform_features")]
    array_stats: dict[str, Any] = {}
    metadata_mismatches: list[dict[str, Any]] = []
    array_mismatch_examples: list[dict[str, Any]] = []
    examples_handle = open(examples_output, "w", encoding="utf-8") if examples_output else None
    try:
        for sample_index, (key, reference, candidate) in enumerate(samples):
            metadata_diff: dict[str, Any] = {}
            for name in ("event_id", "source_path", "source_index", "source_group"):
                ref_value = getattr(reference, name)
                cand_value = getattr(candidate, name)
                if ref_value != cand_value:
                    metadata_diff[name] = {"reference": ref_value, "candidate": cand_value}
            if metadata_diff and len(metadata_mismatches) < 20:
                metadata_mismatches.append(
                    {
                        "sample_index": sample_index,
                        "key": key,
                        "reference": dataclasses.asdict(reference),
                        "candidate": dataclasses.asdict(candidate),
                        "diff": metadata_diff,
                    }
                )
            pair_record: dict[str, Any] = {
                "sample_index": sample_index,
                "key": key,
                "reference": dataclasses.asdict(reference),
                "candidate": dataclasses.asdict(candidate),
                "metadata_diff": metadata_diff,
                "arrays": {},
            }
            reference_arrays = _read_event_arrays(reference, dataset_names)
            candidate_arrays = _read_event_arrays(candidate, dataset_names)
            for dataset_name in dataset_names:
                ref_array = reference_arrays[dataset_name]
                cand_array = candidate_arrays[dataset_name]
                diff = _array_diff(ref_array, cand_array)
                _merge_array_stats(array_stats, dataset_name, diff)
                pair_record["arrays"][dataset_name] = diff
                if (
                    len(array_mismatch_examples) < 20
                    and (not bool(diff.get("shape_match", False)) or int(diff.get("mismatch_count", 0) or 0) > 0)
                ):
                    array_mismatch_examples.append(
                        {
                            "sample_index": sample_index,
                            "key": key,
                            "dataset": dataset_name,
                            "reference": dataclasses.asdict(reference),
                            "candidate": dataclasses.asdict(candidate),
                            "diff": diff,
                        }
                    )
            if examples_handle is not None:
                examples_handle.write(json.dumps(pair_record, default=_json_default, sort_keys=True) + "\n")
            if sample_index == 0 or (sample_index + 1) % 100 == 0:
                _progress(
                    "audit_h5_inputs stage=compare_samples "
                    f"sampled={sample_index + 1}/{len(samples)}"
                )
    finally:
        if examples_handle is not None:
            examples_handle.close()
    return {
        "sampled_pairs": len(samples),
        "skip_waveforms": bool(skip_waveforms),
        "metadata_mismatch_examples": metadata_mismatches,
        "array_stats": array_stats,
        "array_mismatch_examples": array_mismatch_examples,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def run_audit(
    *,
    reference: list[str],
    candidate: list[str],
    match_key: str,
    sample_size: int,
    seed: int,
    skip_waveforms: bool,
    progress_interval_sec: float,
    examples_output: str | None = None,
) -> dict[str, Any]:
    reference_paths = _expand_h5_paths(reference)
    candidate_paths = _expand_h5_paths(candidate)
    if not reference_paths:
        raise FileNotFoundError(f"no reference HDF5 files found from: {reference}")
    if not candidate_paths:
        raise FileNotFoundError(f"no candidate HDF5 files found from: {candidate}")
    _progress(
        "audit_h5_inputs stage=start "
        f"reference_files={len(reference_paths)} candidate_files={len(candidate_paths)} "
        f"match_key={match_key} sample_size={sample_size}"
    )
    reference_index, reference_summary = _build_reference_index(
        reference_paths,
        match_key=match_key,
        progress_interval_sec=progress_interval_sec,
    )
    _progress(
        "audit_h5_inputs stage=done_reference "
        f"events={reference_summary['events']} unique_keys={reference_summary['unique_keys']} "
        f"duplicates={reference_summary['duplicates']}"
    )
    samples, matched_reference_keys, candidate_summary = _sample_candidate_overlaps(
        candidate_paths,
        reference_index,
        match_key=match_key,
        sample_size=sample_size,
        seed=seed,
        progress_interval_sec=progress_interval_sec,
    )
    _progress(
        "audit_h5_inputs stage=done_candidate "
        f"events={candidate_summary['events']} overlap={candidate_summary['overlap_keys']} "
        f"sampled={len(samples)}"
    )
    comparison = _compare_samples(samples, skip_waveforms=skip_waveforms, examples_output=examples_output)
    reference_only = int(reference_summary["unique_keys"]) - len(matched_reference_keys)
    overlap_fraction_reference = len(matched_reference_keys) / max(int(reference_summary["unique_keys"]), 1)
    overlap_fraction_candidate = int(candidate_summary["overlap_keys"]) / max(int(candidate_summary["events"]), 1)
    payload = {
        "schema": "talesd_gnn_homogeneous_input_parity_audit_v1",
        "match_key": match_key,
        "seed": int(seed),
        "reference_paths": reference_paths,
        "candidate_paths": candidate_paths,
        "reference": reference_summary,
        "candidate": candidate_summary,
        "overlap": {
            "matched_reference_keys": len(matched_reference_keys),
            "reference_only_keys": reference_only,
            "candidate_events_with_reference": int(candidate_summary["overlap_keys"]),
            "fraction_of_reference_unique_keys": overlap_fraction_reference,
            "fraction_of_candidate_events": overlap_fraction_candidate,
        },
        "comparison": comparison,
    }
    _progress(
        "audit_h5_inputs stage=done "
        f"matched_reference_keys={len(matched_reference_keys)} "
        f"reference_only_keys={reference_only} "
        f"candidate_overlap_fraction={overlap_fraction_candidate:.6g}"
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare old and current homogeneous graph HDF5 inputs on matched events."
    )
    parser.add_argument("--reference", nargs="+", required=True, help="reference HDF5 file/directory")
    parser.add_argument("--candidate", nargs="+", required=True, help="candidate HDF5 file/directory")
    parser.add_argument("-o", "--output", required=True, help="summary JSON output")
    parser.add_argument("--examples-output", default=None, help="optional JSONL per-sampled-pair output")
    parser.add_argument(
        "--match-key",
        choices=("event_id", "source_path_index", "source_group_index", "source_group_event_id"),
        default="source_group_index",
        help="event identity key used to match reference and candidate graphs",
    )
    parser.add_argument("--sample-size", type=int, default=1000, help="deterministic matched-event sample size")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip-waveforms", action="store_true", help="do not read waveform_features in the sampled comparison")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.examples_output:
        Path(args.examples_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    payload = run_audit(
        reference=args.reference,
        candidate=args.candidate,
        match_key=args.match_key,
        sample_size=max(int(args.sample_size), 0),
        seed=int(args.seed),
        skip_waveforms=bool(args.skip_waveforms),
        progress_interval_sec=float(args.progress_interval_sec),
        examples_output=str(Path(args.examples_output).expanduser()) if args.examples_output else None,
    )
    output.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _progress(f"audit_h5_inputs summary={output}")


if __name__ == "__main__":
    main()
