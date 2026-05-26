#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.progress import progress_bar, write as progress_write


@dataclass(frozen=True)
class SelectedEvent:
    rank: float
    path_index: int
    local_index: int
    event_key: str
    log_energy: float
    energy_bin: int
    particle_label: float
    event_id: str
    source_path: str
    source_index: int
    parttype: int


def _decode(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    return str(value)


def _metadata_value(metadata: h5py.Group | None, name: str, index: int) -> Any:
    if metadata is None or name not in metadata:
        return None
    dataset = metadata[name]
    if index < 0 or index >= len(dataset):
        return None
    return dataset[index]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(np.asarray(value).reshape(()))
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _particle_label(metadata: h5py.Group | None, group: h5py.Group, local_index: int) -> float | None:
    value = _finite_float(_metadata_value(metadata, "particle_label", local_index))
    if value is not None:
        return value
    if "particle_label" in group:
        value = _finite_float(group["particle_label"][()])
        if value is not None:
            return value
    value = _finite_float(group.attrs.get("particle_label", group.attrs.get("particle_is_iron", None)))
    if value is not None:
        return value
    parttype = int(group.attrs.get("parttype", -1))
    if parttype == 14:
        return 0.0
    if parttype == 5626:
        return 1.0
    source_path = _decode(group.attrs.get("source_path", "")).lower()
    if "/proton/" in source_path or "tale_proton" in source_path:
        return 0.0
    if "/iron/" in source_path or "tale_iron" in source_path:
        return 1.0
    return None


def _event_metadata(metadata: h5py.Group | None, group: h5py.Group, local_index: int, fallback_key: str) -> tuple[str, str, int, int]:
    event_id = _decode(_metadata_value(metadata, "event_id", local_index), "")
    if not event_id:
        event_id = _decode(group.attrs.get("event_id", fallback_key), fallback_key)
    source_path = _decode(_metadata_value(metadata, "source_path", local_index), "")
    if not source_path:
        source_path = _decode(group.attrs.get("source_path", ""))
    source_index = _finite_float(_metadata_value(metadata, "source_index", local_index))
    if source_index is None:
        source_index = _finite_float(group.attrs.get("source_index", -1))
    parttype = _finite_float(_metadata_value(metadata, "parttype", local_index))
    if parttype is None:
        parttype = _finite_float(group.attrs.get("parttype", -1))
    return event_id, source_path, int(source_index if source_index is not None else -1), int(parttype if parttype is not None else -1)


def _rank(seed: int, path_index: int, local_index: int, event_id: str) -> float:
    text = f"{seed}|{path_index}|{local_index}|{event_id}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(text, digest_size=8, person=b"talesdgnn").digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _particle_allowed(label: float, particle_filter: str) -> bool:
    if particle_filter == "proton":
        return label < 0.5
    if particle_filter == "iron":
        return label >= 0.5
    return True


def _dense_or_sorted_keys(events: h5py.Group) -> list[str]:
    keys = sorted(events.keys())
    if keys and all(key == f"{index:08d}" for index, key in enumerate(keys)):
        return [f"{index:08d}" for index in range(len(keys))]
    return keys


def _progress_interval_seconds() -> float:
    try:
        return max(float(os.environ.get("TALESD_GNN_PROGRESS_INTERVAL", "30")), 1.0)
    except ValueError:
        return 30.0


def _new_stats() -> dict[str, Any]:
    return {
        "input_files": 0,
        "input_events": 0,
        "missing_target": 0,
        "missing_particle_label": 0,
        "filtered_particle": 0,
        "nonfinite_energy": 0,
        "candidate_events": 0,
    }


def _merge_stats(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, (int, np.integer)):
            total[key] = int(total.get(key, 0)) + int(value)


def _merge_reservoirs(
    total: dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]],
    part: dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]],
    *,
    per_bin: int,
) -> None:
    for stratum, heap in part.items():
        target = total.setdefault(stratum, [])
        for item in heap:
            if len(target) < per_bin:
                heapq.heappush(target, item)
            elif item[0] > target[0][0]:
                heapq.heapreplace(target, item)


def _scan_path_selected(
    path: str,
    path_index: int,
    *,
    per_bin: int,
    energy_bin_width: float,
    stratify_particle: bool,
    particle_filter: str,
    seed: int,
) -> tuple[dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]], dict[str, Any]]:
    reservoirs: dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]] = {}
    stats = _new_stats()
    stats["input_files"] = 1
    with h5py.File(path, "r") as h5:
        events = h5["events"]
        metadata = h5.get("metadata")
        stats["input_events"] = int(len(events))
        for local_index, event_key in enumerate(_dense_or_sorted_keys(events)):
            group = events[event_key]
            if "target" not in group:
                stats["missing_target"] += 1
                continue
            target = np.asarray(group["target"][()], dtype=np.float64).reshape(-1)
            if target.size == 0 or not np.isfinite(target[0]):
                stats["nonfinite_energy"] += 1
                continue
            label = _particle_label(metadata, group, local_index)
            if label is None:
                stats["missing_particle_label"] += 1
                continue
            if not _particle_allowed(label, particle_filter):
                stats["filtered_particle"] += 1
                continue
            event_id, source_path, source_index, parttype = _event_metadata(
                metadata,
                group,
                local_index,
                event_key,
            )
            log_energy = float(target[0])
            energy_bin = int(math.floor(log_energy / energy_bin_width))
            particle_key = int(label >= 0.5) if stratify_particle else "all"
            stratum = (particle_key, energy_bin)
            rank = _rank(seed, path_index, local_index, event_id)
            event = SelectedEvent(
                rank=rank,
                path_index=path_index,
                local_index=local_index,
                event_key=event_key,
                log_energy=log_energy,
                energy_bin=energy_bin,
                particle_label=float(label),
                event_id=event_id,
                source_path=source_path,
                source_index=source_index,
                parttype=parttype,
            )
            heap = reservoirs.setdefault(stratum, [])
            item = (-rank, path_index, local_index, event)
            if len(heap) < per_bin:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
            stats["candidate_events"] += 1
    return reservoirs, stats


def _scan_paths_selected(
    items: list[tuple[int, str]],
    *,
    per_bin: int,
    energy_bin_width: float,
    stratify_particle: bool,
    particle_filter: str,
    seed: int,
) -> tuple[dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]], dict[str, Any]]:
    reservoirs: dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]] = {}
    stats = _new_stats()
    for path_index, path in items:
        part_reservoirs, part_stats = _scan_path_selected(
            path,
            path_index,
            per_bin=per_bin,
            energy_bin_width=energy_bin_width,
            stratify_particle=stratify_particle,
            particle_filter=particle_filter,
            seed=seed,
        )
        _merge_reservoirs(reservoirs, part_reservoirs, per_bin=per_bin)
        _merge_stats(stats, part_stats)
    return reservoirs, stats


def _chunk_items(items: list[tuple[int, str]], chunk_count: int) -> list[list[tuple[int, str]]]:
    if not items:
        return []
    chunk_count = max(min(chunk_count, len(items)), 1)
    chunk_size = int(math.ceil(len(items) / chunk_count))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _chunk_events(items: list[SelectedEvent], chunk_count: int) -> list[list[SelectedEvent]]:
    if not items:
        return []
    chunk_count = max(min(chunk_count, len(items)), 1)
    chunk_size = int(math.ceil(len(items) / chunk_count))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _chunk_events_by_source(items: list[SelectedEvent], chunk_count: int) -> list[list[SelectedEvent]]:
    if not items:
        return []
    groups: dict[int, list[SelectedEvent]] = {}
    for event in items:
        groups.setdefault(event.path_index, []).append(event)
    source_groups = [groups[path_index] for path_index in sorted(groups)]
    chunk_count = max(min(chunk_count, len(source_groups)), 1)
    if chunk_count == len(source_groups):
        return source_groups

    buckets: list[list[SelectedEvent]] = [[] for _ in range(chunk_count)]
    bucket_sizes = [0 for _ in range(chunk_count)]
    for group in sorted(source_groups, key=len, reverse=True):
        bucket_index = min(range(chunk_count), key=lambda index: bucket_sizes[index])
        buckets[bucket_index].extend(group)
        bucket_sizes[bucket_index] += len(group)
    return [bucket for bucket in buckets if bucket]


def _shard_path(output: str | Path, shard_index: int) -> Path:
    base = Path(output).expanduser()
    suffix = base.suffix if base.suffix else ".h5"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}_{shard_index:04d}{suffix}")


def _existing_output_paths(output: Path) -> list[Path]:
    existing: list[Path] = []
    if output.exists():
        existing.append(output)
    if not output.parent.exists():
        return existing
    suffix = output.suffix if output.suffix else ".h5"
    stem = output.stem if output.suffix else output.name
    existing.extend(sorted(path for path in output.parent.glob(f"{stem}_*{suffix}") if path not in existing))
    existing.extend(sorted(path for path in output.parent.glob(f"{stem}_*{suffix}.summary.json") if path not in existing))
    summary = output.with_suffix(output.suffix + ".summary.json")
    if summary.exists() and summary not in existing:
        existing.append(summary)
    return existing


def _extra_output_path(output: Path, per_bin: int) -> Path:
    suffix = output.suffix if output.suffix else ".h5"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}-perbin{per_bin}{suffix}")


def _stratum_key(event: SelectedEvent, *, stratify_particle: bool) -> tuple[Any, int]:
    particle_key = int(event.particle_label >= 0.5) if stratify_particle else "all"
    return particle_key, event.energy_bin


def _subset_selected_per_bin(
    selected: list[SelectedEvent],
    *,
    per_bin: int,
    stratify_particle: bool,
) -> list[SelectedEvent]:
    counts: dict[tuple[Any, int], int] = {}
    subset: list[SelectedEvent] = []
    for event in selected:
        key = _stratum_key(event, stratify_particle=stratify_particle)
        count = counts.get(key, 0)
        if count >= per_bin:
            continue
        counts[key] = count + 1
        subset.append(event)
    return subset


def _strata_counts(selected: list[SelectedEvent], *, stratify_particle: bool) -> dict[str, int]:
    counts: dict[tuple[Any, int], int] = {}
    for event in selected:
        key = _stratum_key(event, stratify_particle=stratify_particle)
        counts[key] = counts.get(key, 0) + 1
    return {
        f"{stratum[0]}:{stratum[1]}": count
        for stratum, count in sorted(counts.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    }


def _resolve_output_shards(
    *,
    requested_shards: int,
    selected_events: int,
    target_events_per_shard: int,
) -> int:
    if selected_events <= 0:
        return 1
    if requested_shards > 0:
        return max(min(requested_shards, selected_events), 1)
    return max(math.ceil(selected_events / target_events_per_shard), 1)


def _scan_selected(
    paths: list[Path],
    *,
    per_bin: int,
    max_total: int | None,
    energy_bin_width: float,
    stratify_particle: bool,
    particle_filter: str,
    seed: int,
    scan_workers: int,
    show_progress: bool,
) -> tuple[list[SelectedEvent], dict[str, Any]]:
    reservoirs: dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]] = {}
    stats = _new_stats()
    workers = max(min(int(scan_workers), len(paths)), 1)
    workers_used = workers
    items = [(path_index, str(path)) for path_index, path in enumerate(paths)]
    progress_write(f"scan setup: input_files={len(paths)} scan_workers={workers} progress_unit=files")
    bar = progress_bar("scan small graph files", len(paths), enabled=show_progress)
    try:
        def scan_serial() -> None:
            for item in items:
                part_reservoirs, part_stats = _scan_paths_selected(
                    [item],
                    per_bin=per_bin,
                    energy_bin_width=energy_bin_width,
                    stratify_particle=stratify_particle,
                    particle_filter=particle_filter,
                    seed=seed,
                )
                _merge_reservoirs(reservoirs, part_reservoirs, per_bin=per_bin)
                _merge_stats(stats, part_stats)
                bar.update(int(part_stats["input_files"]))

        if workers == 1:
            scan_serial()
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    chunks = _chunk_items(items, chunk_count=workers * 4)
                    futures = [
                        executor.submit(
                            _scan_paths_selected,
                            chunk,
                            per_bin=per_bin,
                            energy_bin_width=energy_bin_width,
                            stratify_particle=stratify_particle,
                            particle_filter=particle_filter,
                            seed=seed,
                        )
                        for chunk in chunks
                    ]
                    pending = set(futures)
                    completed_chunks = 0
                    last_wait_report = time.perf_counter()
                    while pending:
                        done, pending = wait(
                            pending,
                            timeout=_progress_interval_seconds(),
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            progress_write(
                                "scan small graph files: "
                                f"completed_chunks={completed_chunks}/{len(futures)} "
                                f"files={bar.count}/{len(paths)} "
                                f"events_scanned={stats['input_events']} "
                                f"pending_chunks={len(pending)}"
                            )
                            last_wait_report = time.perf_counter()
                            continue
                        for future in done:
                            part_reservoirs, part_stats = future.result()
                            _merge_reservoirs(reservoirs, part_reservoirs, per_bin=per_bin)
                            _merge_stats(stats, part_stats)
                            bar.update(int(part_stats["input_files"]))
                            completed_chunks += 1
                        now = time.perf_counter()
                        if now - last_wait_report >= _progress_interval_seconds():
                            progress_write(
                                "scan small graph files: "
                                f"completed_chunks={completed_chunks}/{len(futures)} "
                                f"files={bar.count}/{len(paths)} "
                                f"events_scanned={stats['input_events']} "
                                f"pending_chunks={len(pending)}"
                            )
                            last_wait_report = now
            except PermissionError as exc:
                progress_write(f"scan workers unavailable ({exc}); falling back to scan_workers=1")
                workers_used = 1
                reservoirs.clear()
                stats.clear()
                stats.update(_new_stats())
                scan_serial()
    finally:
        bar.close()

    selected = [event for heap in reservoirs.values() for _neg_rank, _path_index, _local_index, event in heap]
    selected.sort(key=lambda event: event.rank)
    if max_total is not None and max_total > 0:
        selected = selected[: int(max_total)]
    stats["selected_events"] = len(selected)
    stats["scan_workers"] = workers_used
    stats["strata"] = {
        f"{stratum[0]}:{stratum[1]}": len(heap)
        for stratum, heap in sorted(reservoirs.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    }
    return selected, stats


def _copy_attrs(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _write_selected_shard(
    payload: tuple[int, str, list[str], list[SelectedEvent], dict[str, Any], int],
) -> dict[str, Any]:
    shard_index, output_text, path_texts, selected, config, output_shards = payload
    paths = [Path(path) for path in path_texts]
    output = Path(output_text)
    output_path = output if output_shards == 1 else _shard_path(output, shard_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles: OrderedDict[int, h5py.File] = OrderedDict()

    def handle(path_index: int) -> h5py.File:
        if path_index in handles:
            h5 = handles.pop(path_index)
            handles[path_index] = h5
            return h5
        h5 = h5py.File(paths[path_index], "r")
        handles[path_index] = h5
        while len(handles) > 4:
            _old_index, old = handles.popitem(last=False)
            old.close()
        return h5

    try:
        with h5py.File(paths[0], "r") as first, h5py.File(output_path, "w") as out:
            _copy_attrs(first, out)
            attr_config = dict(config)
            input_graphs = list(attr_config.pop("input_graphs", []))
            attr_config["input_graph_count"] = len(input_graphs)
            attr_config["input_graphs_preview"] = input_graphs[:10]
            attr_config["shard_index"] = shard_index if output_shards > 1 else None
            attr_config["output_shards"] = output_shards
            attr_config["output_base"] = str(output)
            out.attrs["small_dataset_config_json"] = json.dumps(attr_config, sort_keys=True)
            events_out = out.create_group("events")
            metadata = out.create_group("metadata")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            n_selected = len(selected)
            metadata.create_dataset("event_id", shape=(n_selected,), dtype=string_dtype)
            metadata.create_dataset("source_path", shape=(n_selected,), dtype=string_dtype)
            metadata.create_dataset("source_index", shape=(n_selected,), dtype=np.int64)
            metadata.create_dataset("parttype", shape=(n_selected,), dtype=np.int32)
            metadata.create_dataset("particle_label", shape=(n_selected,), dtype=np.float32)

            for output_index, event in enumerate(selected):
                src = handle(event.path_index)
                src.copy(src["events"][event.event_key], events_out, name=f"{output_index:08d}")
                metadata["event_id"][output_index] = event.event_id
                metadata["source_path"][output_index] = event.source_path
                metadata["source_index"][output_index] = event.source_index
                metadata["parttype"][output_index] = event.parttype
                metadata["particle_label"][output_index] = event.particle_label
    finally:
        for h5 in handles.values():
            h5.close()
    return {"shard_index": shard_index, "path": str(output_path), "events": len(selected)}


def _write_selected(
    paths: list[Path],
    selected: list[SelectedEvent],
    output: Path,
    *,
    config: dict[str, Any],
    show_progress: bool,
    write_workers: int,
    output_shards: int,
    group_by_source: bool,
) -> tuple[list[Path], list[list[SelectedEvent]]]:
    output_shards = max(min(int(output_shards), len(selected)), 1)
    write_workers = max(min(int(write_workers), output_shards), 1)
    if group_by_source:
        chunks = _chunk_events_by_source(selected, output_shards)
    else:
        chunks = _chunk_events(selected, output_shards)
    path_texts = [str(path) for path in paths]
    payloads = [
        (shard_index, str(output), path_texts, chunk, config, len(chunks))
        for shard_index, chunk in enumerate(chunks)
    ]
    output_paths = [output if len(chunks) == 1 else _shard_path(output, index) for index in range(len(chunks))]
    progress_write(
        f"write setup: selected_events={len(selected)} output_shards={len(chunks)} "
        f"write_workers={write_workers}"
    )

    def write_serial() -> None:
        bar = progress_bar("write small graph shards", len(payloads), enabled=show_progress)
        try:
            for payload in payloads:
                _write_selected_shard(payload)
                bar.update(1)
        finally:
            bar.close()

    if write_workers == 1:
        write_serial()
        return output_paths, chunks

    bar = None
    bar_closed = False
    pending: set[Future[dict[str, Any]]] = set()
    payload_iter = iter(payloads)
    completed_shards = 0
    written_events = 0
    pool: ProcessPoolExecutor | None = None
    pool_closed = False

    def close_parallel_bar() -> None:
        nonlocal bar_closed
        if bar is not None and not bar_closed:
            bar.close()
            bar_closed = True

    def submit_next() -> bool:
        try:
            payload = next(payload_iter)
        except StopIteration:
            return False
        assert pool is not None
        pending.add(pool.submit(_write_selected_shard, payload))
        return True

    try:
        pool = ProcessPoolExecutor(max_workers=write_workers)
        bar = progress_bar("write small graph shards", len(payloads), enabled=show_progress)
        for _ in range(min(write_workers * 2, len(payloads))):
            submit_next()
        last_wait_report = time.perf_counter()
        while pending:
            done, pending = wait(
                pending,
                timeout=_progress_interval_seconds(),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                progress_write(
                    "write small graph shards: "
                    f"completed_shards={completed_shards}/{len(payloads)} "
                    f"written_events={written_events}/{len(selected)} "
                    f"pending_shards={len(pending)}"
                )
                last_wait_report = time.perf_counter()
                continue
            for future in done:
                result = future.result()
                completed_shards += 1
                written_events += int(result["events"])
                assert bar is not None
                bar.update(1)
                submit_next()
            now = time.perf_counter()
            if now - last_wait_report >= _progress_interval_seconds():
                progress_write(
                    "write small graph shards: "
                    f"completed_shards={completed_shards}/{len(payloads)} "
                    f"written_events={written_events}/{len(selected)} "
                    f"pending_shards={len(pending)}"
                )
                last_wait_report = now
        pool.shutdown(wait=True)
        pool_closed = True
    except PermissionError as exc:
        for future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            pool_closed = True
        progress_write(f"write workers unavailable ({exc}); falling back to write_workers=1")
        close_parallel_bar()
        write_serial()
    except BaseException:
        for future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            pool_closed = True
        raise
    finally:
        if pool is not None and not pool_closed:
            pool.shutdown(wait=False, cancel_futures=True)
        close_parallel_bar()
    return output_paths, chunks


def _remap_selected_to_output_shards(
    chunks: list[list[SelectedEvent]],
) -> list[SelectedEvent]:
    remapped: list[SelectedEvent] = []
    for shard_index, chunk in enumerate(chunks):
        for local_index, event in enumerate(chunk):
            remapped.append(
                replace(
                    event,
                    path_index=shard_index,
                    local_index=local_index,
                    event_key=f"{local_index:08d}",
                )
            )
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small tuning HDF5 graph dataset from existing large HDF5 shards.")
    parser.add_argument("--graphs", nargs="+", required=True, help="input HDF5 shard, shard base, or directory")
    parser.add_argument("-o", "--output", required=True, help="output small HDF5 base path")
    parser.add_argument("--per-bin", type=int, default=2000, help="maximum events per energy bin or particle/energy bin")
    parser.add_argument(
        "--also-per-bin",
        type=int,
        nargs="*",
        default=[],
        help="also write smaller datasets with these per-bin limits from the same scan",
    )
    parser.add_argument("--max-total", type=int, default=None, help="optional total cap after per-bin sampling")
    parser.add_argument("--energy-bin-width", type=float, default=0.1, help="true log10(E/eV) bin width")
    parser.add_argument("--stratify-particle", dest="stratify_particle", action="store_true", default=True)
    parser.add_argument("--no-stratify-particle", dest="stratify_particle", action="store_false")
    parser.add_argument("--particle-filter", choices=["all", "proton", "iron"], default="all")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--scan-workers", type=int, default=1, help="parallel file scan workers")
    parser.add_argument("--write-workers", type=int, default=1, help="parallel output shard write workers")
    parser.add_argument("--output-shards", type=int, default=0, help="number of output HDF5 shards; 0 chooses from selected event count")
    parser.add_argument("--target-events-per-shard", type=int, default=20000, help="target events per output shard when --output-shards=0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.per_bin <= 0:
        raise SystemExit("--per-bin must be positive")
    extra_per_bins = sorted(set(int(value) for value in args.also_per_bin), reverse=True)
    if any(value <= 0 for value in extra_per_bins):
        raise SystemExit("--also-per-bin values must be positive")
    if any(value >= args.per_bin for value in extra_per_bins):
        raise SystemExit("--also-per-bin values must be smaller than --per-bin")
    if args.energy_bin_width <= 0:
        raise SystemExit("--energy-bin-width must be positive")
    if args.scan_workers <= 0:
        raise SystemExit("--scan-workers must be positive")
    if args.write_workers <= 0:
        raise SystemExit("--write-workers must be positive")
    if args.output_shards < 0:
        raise SystemExit("--output-shards must be non-negative")
    if args.target_events_per_shard <= 0:
        raise SystemExit("--target-events-per-shard must be positive")

    paths = [Path(path).expanduser() for path in _expand_h5_graph_paths(args.graphs)]
    if not paths:
        raise SystemExit("no input graph HDF5 files matched --graphs")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input graph HDF5 file: {missing[0]}")

    output = Path(args.output).expanduser()
    output_jobs = [(args.per_bin, output)]
    output_jobs.extend((per_bin, _extra_output_path(output, per_bin)) for per_bin in extra_per_bins)
    existing_outputs: list[Path] = []
    for _per_bin, job_output in output_jobs:
        existing_outputs.extend(_existing_output_paths(job_output))
    if existing_outputs and not args.overwrite:
        raise SystemExit(f"output already exists; pass --overwrite to replace: {existing_outputs[0]}")

    config = {
        "input_graphs": [str(path) for path in paths],
        "output": str(output),
        "per_bin": args.per_bin,
        "also_per_bin": extra_per_bins,
        "max_total": args.max_total,
        "energy_bin_width": args.energy_bin_width,
        "stratify_particle": args.stratify_particle,
        "particle_filter": args.particle_filter,
        "seed": args.seed,
        "scan_workers": args.scan_workers,
        "write_workers": args.write_workers,
        "output_shards": args.output_shards,
        "target_events_per_shard": args.target_events_per_shard,
    }
    selected, stats = _scan_selected(
        paths,
        per_bin=args.per_bin,
        max_total=args.max_total,
        energy_bin_width=args.energy_bin_width,
        stratify_particle=args.stratify_particle,
        particle_filter=args.particle_filter,
        seed=args.seed,
        scan_workers=args.scan_workers,
        show_progress=not args.no_progress,
    )
    if not selected:
        raise SystemExit("no events selected")
    if existing_outputs:
        for path in existing_outputs:
            path.unlink()
    if args.output_shards > 0:
        main_output_shards = min(args.output_shards, len(selected))
    else:
        main_output_shards = min(args.write_workers, len(selected))
    main_write_workers = min(args.write_workers, main_output_shards)
    stats["write_workers"] = main_write_workers
    stats["output_shards"] = main_output_shards
    main_output_paths, main_chunks = _write_selected(
        paths,
        selected,
        output,
        config=config,
        show_progress=not args.no_progress,
        write_workers=main_write_workers,
        output_shards=main_output_shards,
        group_by_source=True,
    )
    summary = {"config": config, "stats": stats}
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output: {output}")
    if main_output_shards > 1:
        print(f"output_shards: {main_output_shards}")
    print(f"summary: {summary_path}")
    print(f"selected_events: {stats['selected_events']}")

    if extra_per_bins:
        derived_paths = [Path(path) for path in main_output_paths]
        derived_selected = _remap_selected_to_output_shards(main_chunks)
        for output_per_bin, job_output in output_jobs[1:]:
            output_selected = _subset_selected_per_bin(
                derived_selected,
                per_bin=output_per_bin,
                stratify_particle=args.stratify_particle,
            )
            output_stats = {
                "input_files": len(derived_paths),
                "input_events": len(derived_selected),
                "candidate_events": len(derived_selected),
                "selected_events": len(output_selected),
                "strata": _strata_counts(output_selected, stratify_particle=args.stratify_particle),
                "scan_workers": 0,
                "derived_from": str(output),
            }
            output_config = dict(config)
            output_config["input_graphs"] = [str(path) for path in derived_paths]
            output_config["output"] = str(job_output)
            output_config["per_bin"] = output_per_bin
            output_config["source_per_bin"] = args.per_bin
            output_shards = _resolve_output_shards(
                requested_shards=args.output_shards,
                selected_events=len(output_selected),
                target_events_per_shard=args.target_events_per_shard,
            )
            write_workers = min(args.write_workers, output_shards)
            output_stats["write_workers"] = write_workers
            output_stats["output_shards"] = output_shards
            _write_selected(
                derived_paths,
                output_selected,
                job_output,
                config=output_config,
                show_progress=not args.no_progress,
                write_workers=write_workers,
                output_shards=output_shards,
                group_by_source=False,
            )
            summary = {"config": output_config, "stats": output_stats}
            summary_path = job_output.with_suffix(job_output.suffix + ".summary.json")
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            print(f"output: {job_output}")
            if output_shards > 1:
                print(f"output_shards: {output_shards}")
            print(f"summary: {summary_path}")
            print(f"selected_events: {output_stats['selected_events']}")


if __name__ == "__main__":
    main()
