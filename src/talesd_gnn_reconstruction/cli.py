from __future__ import annotations

import argparse
import hashlib
import heapq
import math
import os
import random
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from .layout import default_const_dst_path, load_tale_const_positions
from .progress import progress as _progress
from .progress import progress_bar as _progress_bar
from .progress import write as _progress_write

MAX_CONFIG_PATHS = 200
DEFAULT_WORKER_MAX_FILES = 0
DEFAULT_TRAIN_WORKERS = -1
SelectedEntry = tuple[int | tuple[str, int], str, str, int, float, int, int]


class DstUnitExhaustionError(RuntimeError):
    pass


def _stop_process_pool(pool: ProcessPoolExecutor) -> None:
    terminate = getattr(pool, "terminate_workers", None)
    if callable(terminate):
        terminate()
    else:
        pool.shutdown(wait=False, cancel_futures=True)


def _make_process_pool(workers: int, max_tasks_per_child: int | None = None) -> ProcessPoolExecutor:
    kwargs: dict[str, Any] = {}
    if max_tasks_per_child is not None and max_tasks_per_child > 0:
        kwargs["max_tasks_per_child"] = int(max_tasks_per_child)
    return ProcessPoolExecutor(max_workers=workers, **kwargs)


def _is_dst_unit_exhaustion(exc: BaseException) -> bool:
    if isinstance(exc, DstUnitExhaustionError):
        return True
    message = str(exc)
    return "unit 1024" in message or "out of allowed range [0-1023]" in message


def _raise_dst_unit_exhaustion(exc: BaseException) -> None:
    raise DstUnitExhaustionError(
        "DST unit handles were exhausted inside a worker. "
        "Update and rebuild dstio so closed DST units are reused, or set --worker-max-files "
        "to a positive value as a temporary workaround."
    ) from exc


def _iter_process_pool(
    payloads: list[Any],
    worker_fn: Any,
    workers: int,
    desc: str,
    max_tasks_per_child: int | None = None,
) -> Iterator[Any]:
    if workers <= 1:
        for payload in _progress(payloads, desc=desc, total=len(payloads)):
            yield worker_fn(payload)
        return

    pool = _make_process_pool(workers, max_tasks_per_child=max_tasks_per_child)
    progress = _progress_bar(desc, len(payloads))
    pending = set()
    payload_iter = iter(payloads)
    max_pending = max(workers * 2, 1)
    pool_closed = False

    def submit_next() -> bool:
        try:
            payload = next(payload_iter)
        except StopIteration:
            return False
        pending.add(pool.submit(worker_fn, payload))
        return True

    try:
        for _ in range(min(max_pending, len(payloads))):
            submit_next()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                progress.update(1)
                yield future.result()
                submit_next()
        pool.shutdown(wait=True)
        pool_closed = True
    except BaseException:
        for future in pending:
            future.cancel()
        _stop_process_pool(pool)
        pool_closed = True
        raise
    finally:
        if not pool_closed:
            _stop_process_pool(pool)
        progress.close()


def _chunked(iterable: Iterable[Any], chunk_size: int) -> Iterator[list[Any]]:
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _build_graph_chunk(payload: tuple[list[Any], dict[int, Any] | None]) -> list[Any]:
    from .event_graph import build_graph_event

    records, detector_positions = payload
    return [
        build_graph_event(
            record,
            detector_positions=detector_positions,
        )
        for record in records
    ]


def _sample_key_from_parts(seed: int, event_id: str, source_path: str, source_index: int) -> float:
    text = f"{seed}:{event_id}:{source_path}:{source_index}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _candidate_event_id(path: str, source_index: int, event: dict[str, Any]) -> str:
    rusdraw = event.get("rusdraw") or {}
    rusdmc = event.get("rusdmc") or {}
    event_num = int(rusdmc.get("event_num", rusdraw.get("event_num", -1)) or -1)
    date = int(rusdraw.get("yymmdd", 0) or 0)
    time = int(rusdraw.get("hhmmss", 0) or 0)
    usec = int(rusdraw.get("usec", 0) or 0)
    if event_num >= 0:
        return f"MC{event_num}_{date:06d}_{time:06d}_{usec:06d}_{source_index:06d}"
    return f"{Path(path).name}_{source_index:06d}_{date:06d}_{time:06d}_{usec:06d}"


def _particle_stratum_from_parttype(parttype: Any) -> str:
    try:
        parttype_int = int(parttype)
    except Exception:
        return "unknown"
    if parttype_int == 14:
        return "proton"
    if parttype_int == 5626:
        return "iron"
    return "unknown"


def _particle_stratum_from_graph(graph: Any) -> str:
    label = getattr(graph, "particle_label", None)
    try:
        label_float = float(label)
        if math.isfinite(label_float):
            return "iron" if label_float >= 0.5 else "proton"
    except Exception:
        pass
    return _particle_stratum_from_parttype(graph.metadata.get("parttype", -1))


def _energy_sample_bin_key(
    log10_energy: float,
    bin_width: float,
    *,
    particle: str | None = None,
) -> int | tuple[str, int]:
    energy_bin = _energy_bin_index(log10_energy, bin_width)
    if particle is None:
        return energy_bin
    return (particle, energy_bin)


def _energy_sample_bin_label(bin_key: int | tuple[str, int], bin_width: float) -> str:
    if isinstance(bin_key, tuple):
        particle, energy_bin = bin_key
        return f"{particle}:{energy_bin * bin_width:.3f}_{(energy_bin + 1) * bin_width:.3f}"
    return f"{bin_key * bin_width:.3f}_{(bin_key + 1) * bin_width:.3f}"


def _scan_energy_candidates_for_file(
    payload: tuple[str, float, int, int, bool, int, float, bool, int | None, str | None, bool]
) -> dict[str, Any]:
    import dstio

    (
        path,
        bin_width,
        seed,
        per_bin_limit,
        skip_errors,
        open_retries,
        open_retry_delay,
        stratify_particle,
        min_event_date,
        mc_calib_dir,
        skip_missing_mc_calibration,
    ) = payload
    mc_calibration = None
    if mc_calib_dir and skip_missing_mc_calibration:
        from .mc_calibration import get_cached_mc_calibration_db

        mc_calibration = get_cached_mc_calibration_db(Path(mc_calib_dir))
    reservoirs: dict[int | tuple[str, int], list[tuple[float, str, int, float, int, int]]] = {}
    seen_by_bin: dict[int | tuple[str, int], int] = {}
    raw_events = 0
    hit_events = 0
    missing_calibration_events = 0
    try:
        dst_handle = None
        last_exc: Exception | None = None
        for attempt in range(max(int(open_retries), 1)):
            try:
                dst_handle = dstio.open(path, banks=["rusdraw", "rusdmc"])
                break
            except Exception as exc:
                if _is_dst_unit_exhaustion(exc):
                    _raise_dst_unit_exhaustion(exc)
                last_exc = exc
                if attempt + 1 < max(int(open_retries), 1):
                    time.sleep(max(float(open_retry_delay), 0.0) * (attempt + 1))
        if dst_handle is None:
            if last_exc is not None:
                raise last_exc
            raise OSError(f"failed to open DST: {path}")
        with dst_handle as dst:
            for source_index, event in enumerate(dst):
                raw_events += 1
                rusdraw = event.get("rusdraw") or {}
                date = int(rusdraw.get("yymmdd", 0) or 0)
                if min_event_date is not None and (date <= 0 or date < int(min_event_date)):
                    continue
                time_value = int(rusdraw.get("hhmmss", 0) or 0)
                if mc_calibration is not None and not mc_calibration.has_calibration_time(date, time_value):
                    missing_calibration_events += 1
                    continue
                xxyy = rusdraw.get("xxyy", [])
                if len(xxyy) <= 0:
                    continue
                rusdmc = event.get("rusdmc") or {}
                energy_eev = float(rusdmc.get("energy", 0.0) or 0.0)
                if energy_eev <= 0.0 or not math.isfinite(energy_eev):
                    continue
                hit_events += 1
                log10_energy = math.log10(energy_eev * 1.0e18)
                particle = _particle_stratum_from_parttype(rusdmc.get("parttype", -1)) if stratify_particle else None
                bin_key = _energy_sample_bin_key(log10_energy, bin_width, particle=particle)
                seen_by_bin[bin_key] = seen_by_bin.get(bin_key, 0) + 1
                event_id = _candidate_event_id(path, source_index, event)
                key = _sample_key_from_parts(seed, event_id, path, source_index)
                entry = (-key, f"{path}:{source_index}", int(source_index), float(log10_energy), date, time_value)
                bucket = reservoirs.setdefault(bin_key, [])
                if len(bucket) < per_bin_limit:
                    heapq.heappush(bucket, entry)
                elif entry[0] > bucket[0][0]:
                    heapq.heapreplace(bucket, entry)
    except Exception as exc:
        if _is_dst_unit_exhaustion(exc):
            _raise_dst_unit_exhaustion(exc)
        if not skip_errors:
            raise
        return {
            "path": path,
            "reservoirs": {},
            "seen_by_bin": {},
            "raw_events": raw_events,
            "hit_events": hit_events,
            "missing_calibration_events": missing_calibration_events,
            "error": str(exc),
        }
    return {
        "path": path,
        "reservoirs": reservoirs,
        "seen_by_bin": seen_by_bin,
        "raw_events": raw_events,
        "hit_events": hit_events,
        "missing_calibration_events": missing_calibration_events,
        "error": None,
    }


def _build_graphs_for_file(
    payload: tuple[
        str,
        dict[int, Any] | None,
        str,
        bool,
        bool,
        set[int] | None,
        int,
        float,
        int | None,
        str | None,
        int | None,
        bool,
    ]
) -> dict[str, Any]:
    from .dst_reader import iter_dst_banks
    from .event_graph import build_graph_event

    (
        path,
        detector_positions,
        kind,
        require_trigger_mode0,
        skip_errors,
        source_indices,
        open_retries,
        open_retry_delay,
        max_events_per_file,
        mc_calib_dir,
        min_event_date,
        skip_missing_mc_calibration,
    ) = payload
    graphs = []
    skipped = 0
    records = 0
    for record in iter_dst_banks(
        [path],
        detector_positions=detector_positions,
        kind=kind,
        require_trigger_mode0=require_trigger_mode0,
        skip_errors=skip_errors,
        source_indices=source_indices,
        open_retries=open_retries,
        open_retry_delay=open_retry_delay,
        mc_calib_dir=mc_calib_dir,
        min_event_date=min_event_date,
        skip_missing_mc_calibration=skip_missing_mc_calibration,
    ):
        records += 1
        graph = build_graph_event(record, detector_positions=detector_positions)
        if graph is None:
            skipped += 1
            if max_events_per_file is not None and records >= max_events_per_file:
                break
            continue
        graphs.append(graph)
        if max_events_per_file is not None and records >= max_events_per_file:
            break
    return {
        "path": path,
        "graphs": graphs,
        "skipped": skipped,
        "records": records,
    }


def _selected_path_chunks(
    inputs: list[str],
    selected_indices_by_path: dict[str, set[int]],
    shard_size: int,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_count = 0
    target_size = max(int(shard_size), 1)
    for path in inputs:
        selected = selected_indices_by_path.get(path)
        if not selected:
            continue
        selected_count = len(selected)
        if current and current_count + selected_count > target_size:
            chunks.append(current)
            current = []
            current_count = 0
        current.append(path)
        current_count += selected_count
    if current:
        chunks.append(current)
    return chunks


def _interleaved_selected_entries(
    entries: list[SelectedEntry],
    *,
    seed: int,
    locality_run_size: int,
) -> list[SelectedEntry]:
    run_size = max(int(locality_run_size), 1)
    by_bin_and_path: dict[int | tuple[str, int], dict[str, list[SelectedEntry]]] = {}
    for entry in entries:
        bin_key = entry[0]
        path = entry[2]
        by_bin_and_path.setdefault(bin_key, {}).setdefault(path, []).append(entry)

    runs_by_bin: dict[int | tuple[str, int], deque[list[SelectedEntry]]] = {}
    for bin_key, by_path in by_bin_and_path.items():
        runs: list[list[SelectedEntry]] = []
        for path_entries in by_path.values():
            path_entries = sorted(path_entries, key=lambda entry: entry[3])
            for start in range(0, len(path_entries), run_size):
                runs.append(path_entries[start : start + run_size])
        runs.sort(
            key=lambda run: _sample_key_from_parts(
                int(seed) + 104729,
                run[0][1],
                run[0][2],
                run[0][3],
            )
        )
        runs_by_bin[bin_key] = deque(runs)

    bin_order = list(runs_by_bin)
    bin_order.sort(key=lambda bin_key: _sample_key_from_parts(int(seed) + 15485863, str(bin_key), "", 0))

    ordered: list[SelectedEntry] = []
    while bin_order:
        next_bin_order: list[int | tuple[str, int]] = []
        for bin_key in bin_order:
            runs = runs_by_bin[bin_key]
            if runs:
                ordered.extend(runs.popleft())
            if runs:
                next_bin_order.append(bin_key)
        bin_order = next_bin_order
    return ordered


def _selected_entry_chunks(entries: list[SelectedEntry], shard_size: int) -> list[list[SelectedEntry]]:
    target_size = max(int(shard_size), 1)
    return [entries[start : start + target_size] for start in range(0, len(entries), target_size)]


def _write_selected_graph_shard(
    payload: tuple[
        int,
        list[str],
        dict[str, set[int]],
        dict[int, Any] | None,
        str,
        bool,
        bool,
        int,
        float,
        int | None,
        str | None,
        int | None,
        bool,
        str,
        dict[str, Any],
        float,
        bool,
    ]
) -> dict[str, Any]:
    from .dst_reader import iter_dst_banks
    from .event_graph import build_graph_event
    from .graph_io import create_graph_file, write_graph

    (
        shard_index,
        paths,
        selected_indices_by_path,
        detector_positions,
        kind,
        require_trigger_mode0,
        skip_errors,
        open_retries,
        open_retry_delay,
        max_events_per_file,
        mc_calib_dir,
        min_event_date,
        skip_missing_mc_calibration,
        output,
        config,
        energy_bin_width,
        stratify_particle,
    ) = payload

    output_path = _shard_path(output, shard_index)
    shard_config = dict(config)
    shard_config["shard_index"] = shard_index
    handle = None
    written = 0
    skipped = 0
    records = 0
    processed_files = 0
    graph_seen_by_bin: dict[int | tuple[str, int], int] = {}
    total_files = len(paths)
    selected_total = sum(len(selected_indices_by_path.get(path, ())) for path in paths)
    interval = max(float(os.environ.get("TALESD_GNN_PROGRESS_INTERVAL", "30")), 1.0)
    last_report = time.perf_counter()

    _progress_write(
        f"export/write shard {shard_index:04d}: start files={total_files} "
        f"selected={selected_total} output={output_path.name}"
    )

    try:
        for file_number, path in enumerate(paths, start=1):
            selected = selected_indices_by_path.get(path)
            if not selected:
                continue
            file_records = 0
            for record in iter_dst_banks(
                [path],
                detector_positions=detector_positions,
                kind=kind,
                require_trigger_mode0=require_trigger_mode0,
                skip_errors=skip_errors,
                source_indices=selected,
                open_retries=open_retries,
                open_retry_delay=open_retry_delay,
                mc_calib_dir=mc_calib_dir,
                min_event_date=min_event_date,
                skip_missing_mc_calibration=skip_missing_mc_calibration,
            ):
                records += 1
                file_records += 1
                graph = build_graph_event(record, detector_positions=detector_positions)
                if graph is None:
                    skipped += 1
                    if max_events_per_file is not None and file_records >= max_events_per_file:
                        break
                    continue
                if graph.target is None or graph.target.shape[0] == 0 or not math.isfinite(float(graph.target[0])):
                    skipped += 1
                    if max_events_per_file is not None and file_records >= max_events_per_file:
                        break
                    continue
                particle = _particle_stratum_from_graph(graph) if stratify_particle else None
                bin_key = _energy_sample_bin_key(float(graph.target[0]), float(energy_bin_width), particle=particle)
                graph_seen_by_bin[bin_key] = graph_seen_by_bin.get(bin_key, 0) + 1
                if handle is None:
                    handle = create_graph_file(output_path, config=shard_config)
                write_graph(handle, written, graph)
                written += 1
                if max_events_per_file is not None and file_records >= max_events_per_file:
                    break
            processed_files += 1
            now = time.perf_counter()
            if now - last_report >= interval:
                _progress_write(
                    f"export/write shard {shard_index:04d}: files={file_number}/{total_files} "
                    f"records={records} written={written} skipped={skipped}"
                )
                last_report = now
    except BaseException as exc:
        _progress_write(
            f"export/write shard {shard_index:04d}: failed files={processed_files}/{total_files} "
            f"records={records} written={written} skipped={skipped} error={exc}"
        )
        raise
    finally:
        if handle is not None:
            handle.close()

    _progress_write(
        f"export/write shard {shard_index:04d}: done files={processed_files}/{total_files} "
        f"records={records} written={written} skipped={skipped} output={output_path.name if written > 0 else '(empty)'}"
    )

    return {
        "shard_index": shard_index,
        "path": str(output_path) if written > 0 else None,
        "written": written,
        "skipped": skipped,
        "records": records,
        "graph_seen_by_bin": graph_seen_by_bin,
    }


def _write_ordered_selected_graph_shard(
    payload: tuple[
        int,
        list[SelectedEntry],
        dict[int, Any] | None,
        str,
        bool,
        bool,
        int,
        float,
        int | None,
        str | None,
        int | None,
        bool,
        str,
        dict[str, Any],
        float,
        bool,
        int,
    ]
) -> dict[str, Any]:
    from .dst_reader import iter_dst_banks
    from .event_graph import build_graph_event
    from .graph_io import create_graph_file, write_graph

    (
        shard_index,
        entries,
        detector_positions,
        kind,
        require_trigger_mode0,
        skip_errors,
        open_retries,
        open_retry_delay,
        max_events_per_file,
        mc_calib_dir,
        min_event_date,
        skip_missing_mc_calibration,
        output,
        config,
        energy_bin_width,
        stratify_particle,
        write_block_size,
    ) = payload

    output_path = _shard_path(output, shard_index)
    shard_config = dict(config)
    shard_config["shard_index"] = shard_index
    handle = None
    written = 0
    skipped = 0
    records = 0
    graph_seen_by_bin: dict[int | tuple[str, int], int] = {}
    interval = max(float(os.environ.get("TALESD_GNN_PROGRESS_INTERVAL", "30")), 1.0)
    last_report = time.perf_counter()
    block_size = max(int(write_block_size), 1)

    _progress_write(
        f"export/write shard {shard_index:04d}: start ordered_events={len(entries)} "
        f"block_size={block_size} output={output_path.name}"
    )

    try:
        for block_number, block in enumerate(_chunked(entries, block_size), start=1):
            wanted_by_path: dict[str, set[int]] = {}
            for _bin_key, _unique_id, path, source_index, _log10_energy, _date, _time_value in block:
                wanted_by_path.setdefault(path, set()).add(int(source_index))

            graph_by_key: dict[tuple[str, int], Any] = {}
            for path, source_indices in wanted_by_path.items():
                file_records = 0
                for record in iter_dst_banks(
                    [path],
                    detector_positions=detector_positions,
                    kind=kind,
                    require_trigger_mode0=require_trigger_mode0,
                    skip_errors=skip_errors,
                    source_indices=source_indices,
                    open_retries=open_retries,
                    open_retry_delay=open_retry_delay,
                    mc_calib_dir=mc_calib_dir,
                    min_event_date=min_event_date,
                    skip_missing_mc_calibration=skip_missing_mc_calibration,
                ):
                    records += 1
                    file_records += 1
                    graph = build_graph_event(record, detector_positions=detector_positions)
                    if graph is not None:
                        graph_by_key[(path, int(record.source_index))] = graph
                    if max_events_per_file is not None and file_records >= max_events_per_file:
                        break

            for _bin_key, _unique_id, path, source_index, _log10_energy, _date, _time_value in block:
                graph = graph_by_key.get((path, int(source_index)))
                if graph is None:
                    skipped += 1
                    continue
                if graph.target is None or graph.target.shape[0] == 0 or not math.isfinite(float(graph.target[0])):
                    skipped += 1
                    continue
                particle = _particle_stratum_from_graph(graph) if stratify_particle else None
                bin_key = _energy_sample_bin_key(float(graph.target[0]), float(energy_bin_width), particle=particle)
                graph_seen_by_bin[bin_key] = graph_seen_by_bin.get(bin_key, 0) + 1
                if handle is None:
                    handle = create_graph_file(output_path, config=shard_config)
                write_graph(handle, written, graph)
                written += 1

            now = time.perf_counter()
            if now - last_report >= interval:
                _progress_write(
                    f"export/write shard {shard_index:04d}: blocks={block_number} "
                    f"records={records} written={written} skipped={skipped}"
                )
                last_report = now
    except BaseException as exc:
        _progress_write(
            f"export/write shard {shard_index:04d}: failed ordered_events={len(entries)} "
            f"records={records} written={written} skipped={skipped} error={exc}"
        )
        raise
    finally:
        if handle is not None:
            handle.close()

    _progress_write(
        f"export/write shard {shard_index:04d}: done ordered_events={len(entries)} "
        f"records={records} written={written} skipped={skipped} output={output_path.name if written > 0 else '(empty)'}"
    )

    return {
        "shard_index": shard_index,
        "path": str(output_path) if written > 0 else None,
        "written": written,
        "skipped": skipped,
        "records": records,
        "graph_seen_by_bin": graph_seen_by_bin,
    }


def _iter_selected_shard_write_results(
    inputs: list[str],
    args: argparse.Namespace,
    detector_positions: dict[int, Any] | None,
    selected_indices_by_path: dict[str, set[int]],
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    chunks = _selected_path_chunks(inputs, selected_indices_by_path, int(args.shard_size))
    selected_files = len(selected_indices_by_path)
    selected_events = sum(len(indices) for indices in selected_indices_by_path.values())
    payloads = [
        (
            shard_index,
            paths,
            {path: selected_indices_by_path[path] for path in paths},
            detector_positions,
            args.kind,
            not args.keep_non_mode0,
            args.skip_errors,
            int(args.open_retries),
            float(args.open_retry_delay),
            None if args.max_events_per_file is None or int(args.max_events_per_file) <= 0 else int(args.max_events_per_file),
            str(Path(args.mc_calib_dir).expanduser()) if args.mc_calib_dir else None,
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            bool(args.skip_errors and args.kind == "mc"),
            str(Path(args.output).expanduser()),
            config,
            float(args.energy_bin_width),
            bool(args.energy_sample_stratify_particle),
        )
        for shard_index, paths in enumerate(chunks)
    ]
    workers = min(max(int(args.workers), 1), len(payloads)) if payloads else 1
    _progress_write(
        f"export/write shards: start shards={len(payloads)} workers={workers} "
        f"selected_files={selected_files} selected_events={selected_events}"
    )
    yield from _iter_process_pool(
        payloads,
        _write_selected_graph_shard,
        workers,
        "export/write shards",
        max_tasks_per_child=max(int(args.worker_max_files), 0),
    )


def _iter_ordered_selected_shard_write_results(
    args: argparse.Namespace,
    detector_positions: dict[int, Any] | None,
    selected_entries: list[SelectedEntry],
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    chunks = _selected_entry_chunks(selected_entries, int(args.shard_size))
    payloads = [
        (
            shard_index,
            entries,
            detector_positions,
            args.kind,
            not args.keep_non_mode0,
            args.skip_errors,
            int(args.open_retries),
            float(args.open_retry_delay),
            None if args.max_events_per_file is None or int(args.max_events_per_file) <= 0 else int(args.max_events_per_file),
            str(Path(args.mc_calib_dir).expanduser()) if args.mc_calib_dir else None,
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            bool(args.skip_errors and args.kind == "mc"),
            str(Path(args.output).expanduser()),
            config,
            float(args.energy_bin_width),
            bool(args.energy_sample_stratify_particle),
            int(args.write_block_size),
        )
        for shard_index, entries in enumerate(chunks)
    ]
    workers = min(max(int(args.workers), 1), len(payloads)) if payloads else 1
    _progress_write(
        f"export/write ordered shards: start shards={len(payloads)} workers={workers} "
        f"selected_events={len(selected_entries)} block_size={int(args.write_block_size)}"
    )
    yield from _iter_process_pool(
        payloads,
        _write_ordered_selected_graph_shard,
        workers,
        "export/write ordered shards",
        max_tasks_per_child=max(int(args.worker_max_files), 0),
    )


def _iter_graphs(
    records: Iterable[Any],
    args: argparse.Namespace,
    detector_positions: dict[int, Any] | None,
) -> Iterator[Any]:
    from .event_graph import build_graph_event

    workers = max(int(args.workers), 1)
    if workers == 1:
        for record in _progress(records, desc="export graphs", total=args.max_events):
            yield build_graph_event(
                record,
                detector_positions=detector_positions,
            )
        return

    max_pending = max(workers * 2, 2)
    try:
        with _make_process_pool(workers) as pool:
            pending: deque[Any] = deque()
            chunk_size = max(int(args.chunk_size), 1)
            total_chunks = math.ceil(args.max_events / chunk_size) if args.max_events is not None else None
            for chunk in _progress(_chunked(records, chunk_size), desc="export chunks", total=total_chunks):
                pending.append(
                    pool.submit(
                        _build_graph_chunk,
                        (chunk, detector_positions),
                    )
                )
                while len(pending) >= max_pending:
                    for graph in pending.popleft().result():
                        yield graph
            while pending:
                for graph in pending.popleft().result():
                    yield graph
    except (OSError, PermissionError) as exc:
        _progress_write(f"warning: worker export failed ({exc}); falling back to single-process export")
        for record in _progress(records, desc="export graphs"):
            yield build_graph_event(
                record,
                detector_positions=detector_positions,
            )


def _iter_file_results(
    inputs: list[str],
    args: argparse.Namespace,
    detector_positions: dict[int, Any] | None,
    selected_indices_by_path: dict[str, set[int]] | None = None,
) -> Iterator[dict[str, Any]]:
    payloads = [
        (
            path,
            detector_positions,
            args.kind,
            not args.keep_non_mode0,
            args.skip_errors,
            selected_indices_by_path.get(path) if selected_indices_by_path is not None else None,
            int(args.open_retries),
            float(args.open_retry_delay),
            None if args.max_events_per_file is None or int(args.max_events_per_file) <= 0 else int(args.max_events_per_file),
            str(Path(args.mc_calib_dir).expanduser()) if args.mc_calib_dir else None,
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            bool(args.skip_errors and args.kind == "mc"),
        )
        for path in inputs
        if selected_indices_by_path is None or path in selected_indices_by_path
    ]
    workers = max(int(args.workers), 1)
    if workers == 1:
        for payload in _progress(payloads, desc="export files", total=len(payloads)):
            yield _build_graphs_for_file(payload)
        return

    try:
        yield from _iter_process_pool(
            payloads,
            _build_graphs_for_file,
            workers,
            "export files",
            max_tasks_per_child=max(int(args.worker_max_files), 0),
        )
    except (OSError, PermissionError) as exc:
        _progress_write(f"warning: file-parallel export failed ({exc}); falling back to single-process export")
        for payload in _progress(payloads, desc="export files", total=len(payloads)):
            yield _build_graphs_for_file(payload)


def _iter_scan_results(
    inputs: list[str],
    args: argparse.Namespace,
    preselect_per_bin: int,
) -> Iterator[dict[str, Any]]:
    payloads = [
        (
            path,
            float(args.energy_bin_width),
            int(args.seed),
            int(preselect_per_bin),
            bool(args.skip_errors),
            int(args.open_retries),
            float(args.open_retry_delay),
            bool(args.energy_sample_stratify_particle),
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            str(Path(args.mc_calib_dir).expanduser()) if args.mc_calib_dir else None,
            bool(args.skip_errors and args.kind == "mc"),
        )
        for path in inputs
    ]
    workers = max(int(args.workers), 1)
    if workers == 1:
        for payload in _progress(payloads, desc="scan files", total=len(payloads)):
            yield _scan_energy_candidates_for_file(payload)
        return

    try:
        yield from _iter_process_pool(
            payloads,
            _scan_energy_candidates_for_file,
            workers,
            "scan files",
            max_tasks_per_child=max(int(args.worker_max_files), 0),
        )
    except (OSError, PermissionError) as exc:
        _progress_write(f"warning: file-parallel scan failed ({exc}); falling back to single-process scan")
        for payload in _progress(payloads, desc="scan files", total=len(payloads)):
            yield _scan_energy_candidates_for_file(payload)


def _shard_path(output: str | Path, shard_index: int) -> Path:
    base = Path(output).expanduser()
    suffix = base.suffix if base.suffix else ".h5"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}_{shard_index:04d}{suffix}")


def _read_path_list(list_path: str | Path) -> list[str]:
    path = Path(list_path).expanduser()
    paths: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line).expanduser()
        if not item.is_absolute():
            item = path.parent / item
        paths.append(str(item))
    return paths


def _read_input_dirs(input_dirs: list[str] | None) -> list[str]:
    paths: list[str] = []
    for raw_dir in input_dirs or []:
        directory = Path(raw_dir).expanduser()
        for path in sorted(directory.rglob("*.dst.gz")):
            if "broken" in path.parts:
                continue
            paths.append(str(path))
    return paths


def _resolve_path_args(paths: list[str], list_paths: list[str] | None, label: str) -> list[str]:
    resolved: list[str] = [str(Path(path).expanduser()) for path in paths]
    for list_path in list_paths or []:
        resolved.extend(_read_path_list(list_path))

    deduped = list(dict.fromkeys(resolved))
    if not deduped:
        raise SystemExit(f"{label} files are required; pass paths directly or use --{label}-list")
    return deduped


def _expand_h5_graph_paths(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            matches = sorted(path.glob("*.h5"))
            expanded.extend(str(match) for match in matches)
            continue
        if path.exists():
            expanded.append(str(path))
            continue

        patterns: list[str] = []
        if path.suffix == ".h5":
            patterns.append(f"{path.stem}_*{path.suffix}")
        elif not path.suffix:
            patterns.append(f"{path.name}_*.h5")

        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(sorted(path.parent.glob(pattern)))
        if matches:
            expanded.extend(str(match) for match in matches)
        else:
            expanded.append(str(path))
    return list(dict.fromkeys(expanded))


def _resolve_graph_args(paths: list[str], list_paths: list[str] | None = None) -> list[str]:
    if not paths and not list_paths:
        raise SystemExit("graphs files are required; pass paths with --graphs")
    return _expand_h5_graph_paths(_resolve_path_args(paths, list_paths, "graphs"))


def _resolve_input_args(paths: list[str], list_paths: list[str] | None, input_dirs: list[str] | None) -> list[str]:
    resolved = _resolve_path_args(paths + _read_input_dirs(input_dirs), list_paths, "input")
    return resolved


def _paths_for_config(paths: list[str]) -> dict[str, Any]:
    return {
        "input_count": len(paths),
        "input": paths[:MAX_CONFIG_PATHS],
        "input_truncated": len(paths) > MAX_CONFIG_PATHS,
    }


def _write_graph_iterable(
    graphs: Iterable[Any],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[int, list[Path]]:
    from .graph_io import create_graph_file, write_graph

    written_total = 0
    written_in_file = 0
    shard_index = 0
    shard_size = max(int(args.shard_size), 0)
    written_paths: list[Path] = []
    handle = None
    if shard_size == 0:
        output_path = Path(args.output).expanduser()
        handle = create_graph_file(output_path, config=config)
        written_paths.append(output_path)

    try:
        total = len(graphs) if hasattr(graphs, "__len__") else None
        for graph in _progress(graphs, desc="write graphs", total=total):
            if handle is None or (shard_size > 0 and written_in_file >= shard_size):
                if handle is not None:
                    handle.close()
                    shard_index += 1
                output_path = _shard_path(args.output, shard_index) if shard_size > 0 else Path(args.output).expanduser()
                shard_config = dict(config)
                shard_config["shard_index"] = shard_index if shard_size > 0 else None
                handle = create_graph_file(output_path, config=shard_config)
                written_paths.append(output_path)
                written_in_file = 0
            write_graph(handle, written_in_file, graph)
            written_total += 1
            written_in_file += 1
    finally:
        if handle is not None:
            handle.close()
    return written_total, written_paths


def _write_hetero_graph_iterable(
    graphs: Iterable[Any],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[int, list[Path]]:
    from .hetero_graph_io import create_hetero_graph_file, write_hetero_graph

    written_total = 0
    written_in_file = 0
    shard_index = 0
    shard_size = max(int(args.shard_size), 0)
    written_paths: list[Path] = []
    handle = None
    if shard_size == 0:
        output_path = Path(args.output).expanduser()
        handle = create_hetero_graph_file(output_path, config=config)
        written_paths.append(output_path)

    try:
        total = len(graphs) if hasattr(graphs, "__len__") else None
        for graph in _progress(graphs, desc="write hetero graphs", total=total):
            if handle is None or (shard_size > 0 and written_in_file >= shard_size):
                if handle is not None:
                    handle.close()
                    shard_index += 1
                output_path = _shard_path(args.output, shard_index) if shard_size > 0 else Path(args.output).expanduser()
                shard_config = dict(config)
                shard_config["shard_index"] = shard_index if shard_size > 0 else None
                handle = create_hetero_graph_file(output_path, config=shard_config)
                written_paths.append(output_path)
                written_in_file = 0
            write_hetero_graph(handle, written_in_file, graph)
            written_total += 1
            written_in_file += 1
    finally:
        if handle is not None:
            handle.close()
    return written_total, written_paths


def _energy_bin_index(log10_energy: float, bin_width: float) -> int:
    return int(math.floor(float(log10_energy) / float(bin_width)))


def _energy_sample_key(graph: Any, seed: int) -> float:
    source_path = graph.metadata.get("source_path", "")
    source_index = graph.metadata.get("source_index", "")
    return _sample_key_from_parts(int(seed), graph.event_id, str(source_path), int(source_index))


def _add_energy_sample(
    graph: Any,
    reservoirs: dict[int | tuple[str, int], list[tuple[float, str, Any]]],
    seen_by_bin: dict[int | tuple[str, int], int],
    per_bin: int,
    bin_width: float,
    seed: int,
    stratify_particle: bool,
) -> bool:
    if graph.target is None or graph.target.shape[0] == 0 or not math.isfinite(float(graph.target[0])):
        return False
    particle = _particle_stratum_from_graph(graph) if stratify_particle else None
    bin_key = _energy_sample_bin_key(float(graph.target[0]), bin_width, particle=particle)
    seen_by_bin[bin_key] = seen_by_bin.get(bin_key, 0) + 1
    key = _energy_sample_key(graph, seed)
    unique_id = f"{graph.event_id}:{graph.metadata.get('source_path', '')}:{graph.metadata.get('source_index', '')}"
    entry = (-key, unique_id, graph)
    bucket = reservoirs.setdefault(bin_key, [])
    if len(bucket) < per_bin:
        heapq.heappush(bucket, entry)
        return True
    if entry[0] > bucket[0][0]:
        heapq.heapreplace(bucket, entry)
    return True


def _sampled_graphs_from_reservoirs(
    reservoirs: dict[int | tuple[str, int], list[tuple[float, str, Any]]],
    seed: int,
) -> list[Any]:
    sampled: list[Any] = []
    for bin_index in sorted(reservoirs):
        sampled.extend(entry[2] for entry in sorted(reservoirs[bin_index], reverse=True))
    random.Random(seed).shuffle(sampled)
    return sampled


def _sample_energy_flat_from_graphs(
    graphs: Iterable[Any],
    per_bin: int,
    bin_width: float,
    seed: int,
    stratify_particle: bool,
) -> tuple[list[Any], int, dict[int | tuple[str, int], int]]:
    reservoirs: dict[int | tuple[str, int], list[tuple[float, str, Any]]] = {}
    seen_by_bin: dict[int | tuple[str, int], int] = {}
    skipped = 0
    for graph in graphs:
        if graph is None:
            skipped += 1
            continue
        if not _add_energy_sample(graph, reservoirs, seen_by_bin, per_bin, bin_width, seed, stratify_particle):
            skipped += 1
    return _sampled_graphs_from_reservoirs(reservoirs, seed), skipped, seen_by_bin


def _merge_candidate_reservoirs(
    scan_results: Iterable[dict[str, Any]],
    per_bin_limit: int,
) -> tuple[
    dict[str, set[int]],
    list[SelectedEntry],
    dict[int | tuple[str, int], int],
    dict[int | tuple[str, int], int],
    dict[int, int],
    int,
    int,
    int,
]:
    merged: dict[int | tuple[str, int], list[tuple[float, str, str, int, float, int, int]]] = {}
    seen_by_bin: dict[int | tuple[str, int], int] = {}
    raw_events = 0
    hit_events = 0
    missing_calibration_events = 0

    for result in scan_results:
        if result.get("error"):
            _progress_write(f"warning: skipping unreadable DST {result['path']}: {result['error']}")
        raw_events += int(result.get("raw_events", 0))
        hit_events += int(result.get("hit_events", 0))
        missing_calibration_events += int(result.get("missing_calibration_events", 0))
        for bin_key, count in result.get("seen_by_bin", {}).items():
            seen_by_bin[bin_key] = seen_by_bin.get(bin_key, 0) + int(count)
        for bin_key, entries in result.get("reservoirs", {}).items():
            bucket = merged.setdefault(bin_key, [])
            for neg_key, unique_id, source_index, log10_energy, date, time_value in entries:
                entry = (
                    float(neg_key),
                    str(unique_id),
                    str(result["path"]),
                    int(source_index),
                    float(log10_energy),
                    int(date),
                    int(time_value),
                )
                if len(bucket) < per_bin_limit:
                    heapq.heappush(bucket, entry)
                elif entry[0] > bucket[0][0]:
                    heapq.heapreplace(bucket, entry)

    selected_by_path: dict[str, set[int]] = {}
    selected_entries: list[SelectedEntry] = []
    selected_by_bin: dict[int | tuple[str, int], int] = {}
    selected_event_dates: dict[int, int] = {}
    for bin_key, entries in merged.items():
        selected_by_bin[bin_key] = len(entries)
        for _neg_key, unique_id, path, source_index, log10_energy, date, time_value in entries:
            selected_by_path.setdefault(path, set()).add(int(source_index))
            selected_event_dates[int(date)] = selected_event_dates.get(int(date), 0) + 1
            selected_entries.append(
                (
                    bin_key,
                    unique_id,
                    path,
                    int(source_index),
                    float(log10_energy),
                    int(date),
                    int(time_value),
                )
            )
    return (
        selected_by_path,
        selected_entries,
        seen_by_bin,
        selected_by_bin,
        selected_event_dates,
        missing_calibration_events,
        raw_events,
        hit_events,
    )


def _validate_mc_calibration_dates(calib_dir: Path, event_dates: dict[int, int], *, context: str) -> None:
    if not event_dates:
        return
    from .mc_calibration import TaleMcCalibrationDB

    calibration = TaleMcCalibrationDB(calib_dir)
    missing = sorted(date for date in event_dates if not calibration.has_calibration_source(date, 0))
    if not missing:
        return
    examples = ", ".join(f"{date:06d}({event_dates[date]})" for date in missing[:20])
    extra = "" if len(missing) <= 20 else f", ... +{len(missing) - 20} more"
    raise SystemExit(
        f"TALE MC calibration source is missing for {len(missing)} selected event date(s) "
        f"during {context}: {examples}{extra}\n"
        f"calib_dir: {calib_dir}\n"
        "Add the corresponding talesdcalib_pass2_YYMMDD.dst(.gz) files, "
        "or explicitly provide a physically justified talesdcalib_pass2_typical.dst(.gz)."
    )


def _cmd_export(args: argparse.Namespace) -> None:
    from .dst_reader import iter_dst_banks

    inputs = _resolve_input_args(args.input, args.input_list, args.input_dir)
    const_dst = Path(args.const_dst).expanduser() if args.const_dst else default_const_dst_path()
    mc_calib_dir = Path(args.mc_calib_dir).expanduser() if args.mc_calib_dir else None
    if args.kind == "mc" and mc_calib_dir is None:
        raise ValueError("MC export requires --mc-calib-dir. Use the directory containing talesdcalib_pass2 files.")
    detector_positions = None
    if args.kind == "mc":
        detector_positions = load_tale_const_positions(const_dst)
    elif args.kind == "auto" and const_dst is not None:
        detector_positions = load_tale_const_positions(const_dst)
    config = {
        **_paths_for_config(inputs),
        "input_list": [str(Path(path).expanduser()) for path in args.input_list],
        "input_dir": [str(Path(path).expanduser()) for path in args.input_dir],
        "kind": args.kind,
        "const_dst": str(const_dst) if const_dst is not None else None,
        "mc_calib_dir": str(mc_calib_dir) if mc_calib_dir is not None else None,
        "max_events": args.max_events,
        "max_events_per_file": args.max_events_per_file,
        "graph_definition": "coincidence_analysis_ising_pulse_graph",
        "energy_sample_per_bin": args.energy_sample_per_bin,
        "energy_sample_stratify_particle": bool(args.energy_sample_stratify_particle),
        "energy_bin_width": args.energy_bin_width,
        "energy_oversample_factor": args.energy_oversample_factor,
        "output_order": args.output_order,
        "output_locality_run_size": args.output_locality_run_size,
        "write_block_size": args.write_block_size,
        "seed": args.seed,
        "workers": args.workers,
        "worker_max_files": args.worker_max_files,
        "chunk_size": args.chunk_size,
        "shard_size": args.shard_size,
        "open_retries": args.open_retries,
        "open_retry_delay": args.open_retry_delay,
        "min_event_date": args.min_event_date,
    }
    skipped = 0

    if args.max_events is not None:
        records = iter_dst_banks(
            inputs,
            detector_positions=detector_positions,
            kind=args.kind,
            max_events=args.max_events,
            require_trigger_mode0=not args.keep_non_mode0,
            skip_errors=args.skip_errors,
            open_retries=args.open_retries,
            open_retry_delay=args.open_retry_delay,
            mc_calib_dir=mc_calib_dir,
            min_event_date=None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            skip_missing_mc_calibration=bool(args.skip_errors and args.kind == "mc"),
        )
        graph_iter = _iter_graphs(records, args, detector_positions)
        if args.energy_sample_per_bin is not None:
            sampled_graphs, skipped, seen_by_bin = _sample_energy_flat_from_graphs(
                graph_iter,
                per_bin=max(int(args.energy_sample_per_bin), 1),
                bin_width=float(args.energy_bin_width),
                seed=int(args.seed),
                stratify_particle=bool(args.energy_sample_stratify_particle),
            )
            config["energy_seen_by_bin"] = {
                _energy_sample_bin_label(bin_key, float(args.energy_bin_width)): count
                for bin_key, count in sorted(seen_by_bin.items(), key=lambda item: str(item[0]))
            }
            written_total, written_paths = _write_graph_iterable(sampled_graphs, args, config)
        else:
            def non_null_graphs() -> Iterator[Any]:
                nonlocal skipped
                for graph in graph_iter:
                    if graph is None:
                        skipped += 1
                        continue
                    yield graph

            written_total, written_paths = _write_graph_iterable(non_null_graphs(), args, config)
    else:
        if args.energy_sample_per_bin is not None:
            preselect_per_bin = max(
                int(math.ceil(max(int(args.energy_sample_per_bin), 1) * max(float(args.energy_oversample_factor), 1.0))),
                1,
            )
            (
                selected_by_path,
                selected_entries,
                seen_by_bin,
                selected_by_bin,
                selected_event_dates,
                missing_calibration_events,
                raw_events,
                hit_events,
            ) = _merge_candidate_reservoirs(
                _iter_scan_results(inputs, args, preselect_per_bin=preselect_per_bin),
                per_bin_limit=preselect_per_bin,
            )
            if args.kind == "mc" and mc_calib_dir is not None:
                _validate_mc_calibration_dates(mc_calib_dir, selected_event_dates, context="energy-flat preselection")
            config["energy_seen_by_bin"] = {
                _energy_sample_bin_label(bin_key, float(args.energy_bin_width)): count
                for bin_key, count in sorted(seen_by_bin.items(), key=lambda item: str(item[0]))
            }
            config["energy_preselected_by_bin"] = {
                _energy_sample_bin_label(bin_key, float(args.energy_bin_width)): count
                for bin_key, count in sorted(selected_by_bin.items(), key=lambda item: str(item[0]))
            }
            config["scan_raw_events"] = raw_events
            config["scan_hit_events"] = hit_events
            config["scan_missing_calibration_events"] = missing_calibration_events
            config["scan_selected_files"] = len(selected_by_path)
            config["scan_selected_event_dates"] = {
                f"{date:06d}": count for date, count in sorted(selected_event_dates.items())
            }
            selected_event_count = sum(len(indices) for indices in selected_by_path.values())
            _progress_write(
                "energy-flat preselection: "
                f"selected_events={selected_event_count} selected_files={len(selected_by_path)} "
                f"bins={len(selected_by_bin)} raw_events={raw_events} hit_events={hit_events} "
                f"missing_calibration_events={missing_calibration_events}"
            )
            if selected_event_count != len(selected_entries):
                raise RuntimeError(
                    "energy-flat preselection bookkeeping mismatch: "
                    f"selected_by_path={selected_event_count} selected_entries={len(selected_entries)}"
                )

            file_results = _iter_file_results(inputs, args, detector_positions, selected_indices_by_path=selected_by_path)
            graph_seen_by_bin: dict[int | tuple[str, int], int] = {}
            per_bin = max(int(args.energy_sample_per_bin), 1)
            if preselect_per_bin <= per_bin and int(args.shard_size) > 0 and int(args.workers) > 1:
                output_order = str(args.output_order).lower()
                config["energy_sample_output_order"] = output_order
                config["energy_sample_output_locality_run_size"] = int(args.output_locality_run_size)
                config["energy_sample_write_block_size"] = int(args.write_block_size)
                if output_order == "interleaved":
                    selected_entries = _interleaved_selected_entries(
                        selected_entries,
                        seed=int(args.seed),
                        locality_run_size=int(args.output_locality_run_size),
                    )
                    config["energy_sample_parallel_ordered_shard_write"] = True
                    shard_results = _iter_ordered_selected_shard_write_results(
                        args,
                        detector_positions,
                        selected_entries,
                        config,
                    )
                else:
                    config["energy_sample_parallel_shard_write"] = True
                    shard_results = _iter_selected_shard_write_results(
                        inputs,
                        args,
                        detector_positions,
                        selected_by_path,
                        config,
                    )
                written_total = 0
                written_path_items: list[tuple[int, Path]] = []
                for result in shard_results:
                    skipped += int(result["skipped"])
                    written_total += int(result["written"])
                    if result.get("path"):
                        written_path_items.append((int(result["shard_index"]), Path(str(result["path"]))))
                    for bin_key, count in result.get("graph_seen_by_bin", {}).items():
                        graph_seen_by_bin[bin_key] = graph_seen_by_bin.get(bin_key, 0) + int(count)
                written_paths = [path for _index, path in sorted(written_path_items, key=lambda item: item[0])]
            elif preselect_per_bin <= per_bin:
                config["energy_sample_streaming_preselected"] = True

                def selected_graphs_from_files() -> Iterator[Any]:
                    nonlocal skipped
                    for result in file_results:
                        skipped += int(result["skipped"])
                        for graph in result["graphs"]:
                            if graph.target is None or graph.target.shape[0] == 0 or not math.isfinite(float(graph.target[0])):
                                skipped += 1
                                continue
                            particle = _particle_stratum_from_graph(graph) if bool(args.energy_sample_stratify_particle) else None
                            bin_key = _energy_sample_bin_key(float(graph.target[0]), float(args.energy_bin_width), particle=particle)
                            graph_seen_by_bin[bin_key] = graph_seen_by_bin.get(bin_key, 0) + 1
                            yield graph

                written_total, written_paths = _write_graph_iterable(selected_graphs_from_files(), args, config)
            else:
                reservoirs: dict[int | tuple[str, int], list[tuple[float, str, Any]]] = {}
                for result in file_results:
                    skipped += int(result["skipped"])
                    for graph in result["graphs"]:
                        if not _add_energy_sample(
                            graph,
                            reservoirs,
                            graph_seen_by_bin,
                            per_bin=per_bin,
                            bin_width=float(args.energy_bin_width),
                            seed=int(args.seed),
                            stratify_particle=bool(args.energy_sample_stratify_particle),
                        ):
                            skipped += 1
                sampled_graphs = _sampled_graphs_from_reservoirs(reservoirs, seed=int(args.seed))
                written_total, written_paths = _write_graph_iterable(sampled_graphs, args, config)
            config["energy_graph_seen_by_bin"] = {
                _energy_sample_bin_label(bin_key, float(args.energy_bin_width)): count
                for bin_key, count in sorted(graph_seen_by_bin.items(), key=lambda item: str(item[0]))
            }
        else:
            file_results = _iter_file_results(inputs, args, detector_positions)

            def graphs_from_files() -> Iterator[Any]:
                nonlocal skipped
                for result in file_results:
                    skipped += int(result["skipped"])
                    yield from result["graphs"]

            written_total, written_paths = _write_graph_iterable(graphs_from_files(), args, config)

    targets = ", ".join(str(path) for path in written_paths) if written_paths else str(args.output)
    print(f"wrote {written_total} graphs to {targets} (skipped {skipped} events)")
    if args.energy_sample_per_bin is not None:
        print(f"energy-flat bins: {len(config.get('energy_graph_seen_by_bin', config.get('energy_seen_by_bin', {})))}")


def _cmd_export_hetero(args: argparse.Namespace) -> None:
    import dstio.tale.graph as tale_graph

    inputs = _resolve_input_args(args.input, args.input_list, args.input_dir)
    const_dst = Path(args.const_dst).expanduser() if args.const_dst else None
    mc_calib_dir = Path(args.mc_calib_dir).expanduser() if args.mc_calib_dir else None
    min_event_date = None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date)
    config = {
        **_paths_for_config(inputs),
        "input_list": [str(Path(path).expanduser()) for path in args.input_list],
        "input_dir": [str(Path(path).expanduser()) for path in args.input_dir],
        "kind": args.kind,
        "const_dst": str(const_dst) if const_dst is not None else None,
        "mc_calib_dir": str(mc_calib_dir) if mc_calib_dir is not None else None,
        "max_events": args.max_events,
        "graph_definition": "tale_sd_hetero_ising_pulse_detector_graph_v1",
        "cleaning": args.cleaning,
        "node_policy": args.node_policy,
        "require_reference_core": bool(args.require_reference_core),
        "shard_size": args.shard_size,
        "open_retries": args.open_retries,
        "open_retry_delay": args.open_retry_delay,
        "min_event_date": min_event_date,
        "skip_missing_mc_calibration": bool(args.skip_missing_mc_calibration),
    }
    graphs = tale_graph.iter_graphs(
        inputs,
        kind=args.kind,
        cleaning=args.cleaning,
        node_policy=args.node_policy,
        const_dst=const_dst,
        mc_calib_dir=mc_calib_dir,
        max_events=args.max_events,
        require_trigger_mode0=not args.keep_non_mode0,
        require_reference_core=bool(args.require_reference_core),
        skip_errors=bool(args.skip_errors),
        skip_missing_mc_calibration=bool(args.skip_missing_mc_calibration),
        min_event_date=min_event_date,
        open_retries=args.open_retries,
        open_retry_delay=args.open_retry_delay,
    )
    written_total, written_paths = _write_hetero_graph_iterable(graphs, args, config)
    targets = ", ".join(str(path) for path in written_paths) if written_paths else str(args.output)
    print(f"wrote {written_total} hetero graphs to {targets}")


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import train_model

    graphs = _resolve_graph_args(args.graphs)
    result = train_model(
        graphs_path=graphs,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        model_architecture=args.model_architecture,
        readout_heads=args.readout_heads,
        classification_arch=args.classification_arch,
        detector_embedding_dim=args.detector_embedding_dim,
        waveform_encoder=args.waveform_encoder,
        waveform_embedding_dim=args.waveform_embedding_dim,
        waveform_transformer_heads=args.waveform_transformer_heads,
        waveform_transformer_layers=args.waveform_transformer_layers,
        loss_mode=args.loss_mode,
        energy_loss_weight=args.energy_loss_weight,
        core_loss_weight=args.core_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        core_loss_scale_km=args.core_loss_scale_km,
        angular_loss_scale_deg=args.angular_loss_scale_deg,
        energy_bias_loss_weight=args.energy_bias_loss_weight,
        energy_particle_bias_loss_weight=args.energy_particle_bias_loss_weight,
        energy_bias_bin_width=args.energy_bias_bin_width,
        energy_bias_min_bin_count=args.energy_bias_min_bin_count,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        source_val_fraction=args.source_val_fraction,
        source_test_fraction=args.source_test_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
        device=args.device,
        sample_cache_size=args.sample_cache_size,
        max_graphs=args.max_graphs,
        particle_filter=args.particle_filter,
        pin_memory=None if not args.no_pin_memory else False,
        num_workers=args.num_workers,
        preprocess_workers=args.preprocess_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        collate_backend=args.collate_backend,
        collate_threads=args.collate_threads,
        training_task=args.training_task,
        mass_classification=args.mass_classification,
        mass_loss_weight=args.mass_loss_weight,
        mass_loss_mode=args.mass_loss_mode,
        mass_focal_gamma=args.mass_focal_gamma,
        mass_pos_weight_mode=args.mass_pos_weight_mode,
        mass_ranking_weight=args.mass_ranking_weight,
        mass_ranking_margin=args.mass_ranking_margin,
        mass_collapse_patience=args.mass_collapse_patience,
        mass_collapse_score_std=args.mass_collapse_score_std,
        mass_collapse_balanced_accuracy=args.mass_collapse_balanced_accuracy,
        quality_prediction=args.quality_prediction,
        quality_loss_weight=args.quality_loss_weight,
        quality_angular_scale_deg=args.quality_angular_scale_deg,
        quality_core_scale_km=args.quality_core_scale_km,
        quality_energy_scale=args.quality_energy_scale,
        error_prediction=args.error_prediction,
        error_loss_weight=args.error_loss_weight,
        error_angular_scale_deg=args.error_angular_scale_deg,
        error_core_scale_km=args.error_core_scale_km,
        error_energy_scale=args.error_energy_scale,
        nll_loss_weight=args.nll_loss_weight,
        nll_sigma_energy_floor=args.nll_sigma_energy_floor,
        nll_sigma_angle_floor_deg=args.nll_sigma_angle_floor_deg,
        nll_sigma_core_floor_km=args.nll_sigma_core_floor_km,
        show_progress=not args.no_progress,
        save_diagnostics=not args.no_diagnostics,
        update_learning_curve_each_epoch=not args.no_epoch_learning_curve,
        best_diagnostics=not args.no_best_diagnostics,
        best_diagnostic_max_graphs=args.best_diagnostic_max_graphs,
        diagnostic_energy_bin_width=args.diagnostic_energy_bin_width,
        diagnostic_min_bin_count=args.diagnostic_min_bin_count,
    )
    print(f"checkpoint: {result['checkpoint']}")
    if result.get("metrics_json"):
        print(f"metrics: {result['metrics_json']}")
    print(f"metrics: {result['metrics_path']}")
    diagnostics = result.get("diagnostics") or {}
    if diagnostics:
        print(f"learning curve: {diagnostics.get('learning_curve_pdf')}")
        print(f"diagnostics summary: {diagnostics.get('summary_json')}")
        for split_name in ("validation", "test"):
            split_info = diagnostics.get(split_name) or {}
            if split_info.get("directory"):
                print(f"{split_name} diagnostics: {split_info['directory']}")
            elif split_info.get("pdf"):
                print(f"{split_name} diagnostics: {split_info['pdf']}")
        for split_name in ("validation_mass", "test_mass"):
            split_info = diagnostics.get(split_name) or {}
            if split_info.get("directory"):
                print(f"{split_name} diagnostics: {split_info['directory']}")
            elif split_info.get("pdfs"):
                print(f"{split_name} diagnostics: {', '.join(split_info['pdfs'])}")


def _cmd_train_hetero(args: argparse.Namespace) -> None:
    from .hetero_training import train_hetero_model

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    result = train_hetero_model(
        graphs_path=graphs,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        waveform_encoder=args.waveform_encoder,
        waveform_embedding_dim=args.waveform_embedding_dim,
        waveform_length=args.waveform_length,
        loss_mode=args.loss_mode,
        energy_loss_weight=args.energy_loss_weight,
        core_loss_weight=args.core_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        core_loss_scale_km=args.core_loss_scale_km,
        angular_loss_scale_deg=args.angular_loss_scale_deg,
        energy_bias_loss_weight=args.energy_bias_loss_weight,
        energy_particle_bias_loss_weight=args.energy_particle_bias_loss_weight,
        energy_bias_bin_width=args.energy_bias_bin_width,
        energy_bias_min_bin_count=args.energy_bias_min_bin_count,
        mass_classification=args.mass_classification,
        mass_loss_weight=args.mass_loss_weight,
        mass_loss_mode=args.mass_loss_mode,
        mass_focal_gamma=args.mass_focal_gamma,
        mass_ranking_weight=args.mass_ranking_weight,
        mass_ranking_margin=args.mass_ranking_margin,
        quality_prediction=args.quality_prediction,
        quality_loss_weight=args.quality_loss_weight,
        quality_angular_scale_deg=args.quality_angular_scale_deg,
        quality_core_scale_km=args.quality_core_scale_km,
        quality_energy_scale=args.quality_energy_scale,
        error_prediction=args.error_prediction,
        error_loss_weight=args.error_loss_weight,
        error_angular_scale_deg=args.error_angular_scale_deg,
        error_core_scale_km=args.error_core_scale_km,
        error_energy_scale=args.error_energy_scale,
        nll_loss_weight=args.nll_loss_weight,
        nll_sigma_energy_floor=args.nll_sigma_energy_floor,
        nll_sigma_angle_floor_deg=args.nll_sigma_angle_floor_deg,
        nll_sigma_core_floor_km=args.nll_sigma_core_floor_km,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        source_val_fraction=args.source_val_fraction,
        source_test_fraction=args.source_test_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
        device=args.device,
        save_diagnostics=args.diagnostics,
        diagnostic_energy_bin_width=args.diagnostic_energy_bin_width,
        diagnostic_min_bin_count=args.diagnostic_min_bin_count,
        show_progress=not args.no_progress,
    )
    print(f"checkpoint: {result['checkpoint']}")
    if result.get("history"):
        last = result["history"][-1]
        print(f"last epoch: {last['epoch']} train_loss={last['train_loss']:.6g} val_loss={last['val_loss']:.6g}")
    metrics = result.get("metrics") or {}
    if metrics.get("validation"):
        validation = metrics["validation"]
        print(
            "validation: "
            f"rmse_log10_energy={validation.get('rmse_log10_energy', float('nan')):.6g} "
            f"core_68_km={validation.get('core_68_km', float('nan')):.6g} "
            f"angular_68_deg={validation.get('angular_68_deg', float('nan')):.6g}"
        )
    if metrics.get("test"):
        test = metrics["test"]
        print(
            "test: "
            f"rmse_log10_energy={test.get('rmse_log10_energy', float('nan')):.6g} "
            f"core_68_km={test.get('core_68_km', float('nan')):.6g} "
            f"angular_68_deg={test.get('angular_68_deg', float('nan')):.6g}"
        )


def _cmd_reconstruct_dst(args: argparse.Namespace) -> None:
    from .hetero_predict import reconstruct_dst

    inputs = _resolve_input_args(args.input, args.input_list, args.input_dir)
    result = reconstruct_dst(
        inputs,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        kind=args.kind,
        const_dst=args.const_dst,
        mc_calib_dir=args.mc_calib_dir,
        batch_size=args.batch_size,
        max_events=args.max_events,
        device=args.device,
        cleaning=args.cleaning,
        node_policy=args.node_policy,
        require_reference_core=not args.allow_missing_reference_core,
        skip_errors=args.skip_errors,
        skip_missing_mc_calibration=args.skip_missing_mc_calibration,
        open_retries=args.open_retries,
        open_retry_delay=args.open_retry_delay,
    )
    print(f"wrote {result['events_written']} DST reconstructions to {result['output']}")


def _cmd_predict(args: argparse.Namespace) -> None:
    from .predict import predict_graphs

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    output = predict_graphs(
        graphs_path=graphs,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device,
        include_truth=not args.no_truth,
    )
    print(f"wrote predictions to {output}")


def _cmd_input_distributions(args: argparse.Namespace) -> None:
    from .feature_analysis import save_input_distributions

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    summary = save_input_distributions(
        graphs,
        args.output,
        max_graphs=args.max_graphs,
        max_values_per_feature=args.max_values_per_feature,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(f"input feature summary: {summary['summary_json']}")


def _cmd_feature_importance(args: argparse.Namespace) -> None:
    from .feature_analysis import save_feature_group_importance

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    summary = save_feature_group_importance(
        graphs,
        args.checkpoint,
        args.output,
        split=args.split,
        max_graphs=args.max_graphs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(f"feature group importance: {summary['summary_json']}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from .visualize import visualize_graphs

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    outputs = visualize_graphs(
        graphs=graphs,
        output=args.output,
        index=args.index,
        event_id=args.event_id,
        count=args.count,
        show_edges=not args.no_edges,
        annotate_lids=args.annotate_lids,
        max_edges=args.max_edges,
        dpi=args.dpi,
        const_dst=args.const_dst,
    )
    for output in outputs:
        print(f"wrote graph visualization to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talesd-gnn",
        description="TALE-SD GNN reconstruction: DST export, training, and prediction",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="DSTをGNN用HDF5グラフへ変換")
    export.add_argument("input", nargs="*", help="入力DSTファイル。複数指定可")
    export.add_argument("--input-list", action="append", default=[], help="入力DSTパスを1行1ファイルで書いたリスト。複数指定可")
    export.add_argument("--input-dir", action="append", default=[], help="入力DSTディレクトリ。*.dst.gzを再帰的に読む。複数指定可")
    export.add_argument("-o", "--output", required=True, help="出力HDF5グラフファイル")
    export.add_argument("--kind", choices=["auto", "data", "mc"], default="auto", help="入力DSTの種類")
    export.add_argument("--const-dst", default=None, help="TALE-SD detector geometry DST talesdconst_pass2.dst。MC入力で必須")
    export.add_argument(
        "--mc-calib-dir",
        default=None,
        help="MC rusdrawをJava解析相当のtalesdcalibevへ変換するための校正ディレクトリ。talesdcalib_pass2_*.dst(.gz)を読む",
    )
    export.add_argument("--max-events", type=int, default=None, help="読み込む最大イベント数")
    export.add_argument("--max-events-per-file", type=int, default=None, help="ファイル単位export時に各DSTから読む最大イベント数。source-path splitの小規模試験用")
    export.add_argument("--min-event-date", type=int, default=None, help="YYMMDD形式。この日付より前のeventをDST読み込み時に除外する")
    export.add_argument("--energy-sample-per-bin", type=int, default=None, help="log10(E/eV) binごとに残す最大グラフ数。reservoir samplingで時刻順バイアスを避ける")
    export.add_argument("--energy-sample-stratify-particle", action="store_true", help="energy-flat samplingをproton/iron別のlog10(E/eV) binで行う")
    export.add_argument("--energy-bin-width", type=float, default=0.1, help="energy-flat samplingのlog10(E/eV) bin幅")
    export.add_argument("--energy-oversample-factor", type=float, default=2.0, help="先行metadata scanで各energy binからgraph化対象として余分に残す倍率")
    export.add_argument("--seed", type=int, default=12345, help="energy-flat samplingの乱数seed")
    export.add_argument("--workers", type=int, default=1, help="DST読み込みとグラフ構築に使うファイル単位worker数。--max-events指定時だけイベントchunk単位")
    export.add_argument("--worker-max-files", type=int, default=DEFAULT_WORKER_MAX_FILES, help="ファイル単位workerをNファイル処理ごとに再起動する。0なら無効")
    export.add_argument("--chunk-size", type=int, default=128, help="--max-events指定時にworkerへ渡すイベントchunkサイズ")
    export.add_argument("--shard-size", type=int, default=0, help="NグラフごとにHDF5を分割する。0なら分割しない")
    export.add_argument("--output-order", choices=["source", "interleaved"], default="interleaved", help="energy-flat出力時のHDF5内event順。interleavedは粒子種・energy binを短いrun単位で混ぜる")
    export.add_argument("--output-locality-run-size", type=int, default=32, help="--output-order=interleavedで同一source/binから連続して書く最大event数")
    export.add_argument("--write-block-size", type=int, default=2048, help="ordered shard書き出しで一度にgraph化して並べ替えるevent数")
    export.add_argument("--open-retries", type=int, default=3, help="DST open失敗時の再試行回数")
    export.add_argument("--open-retry-delay", type=float, default=1.0, help="DST open再試行の待ち時間。試行ごとに線形に増やす")
    export.add_argument("--keep-non-mode0", action="store_true", help="trgMode != 0 も残す")
    export.add_argument("--skip-errors", action="store_true", help="読めないDSTを警告してスキップする")
    export.set_defaults(func=_cmd_export)

    export_hetero = sub.add_parser("export-hetero", help="dstio.tale.graph schema のheterogeneous HDF5グラフへ変換")
    export_hetero.add_argument("input", nargs="*", help="入力DSTファイル。複数指定可")
    export_hetero.add_argument("--input-list", action="append", default=[], help="入力DSTパスを1行1ファイルで書いたリスト。複数指定可")
    export_hetero.add_argument("--input-dir", action="append", default=[], help="入力DSTディレクトリ。*.dst.gzを再帰的に読む。複数指定可")
    export_hetero.add_argument("-o", "--output", required=True, help="出力heterogeneous HDF5グラフファイル")
    export_hetero.add_argument("--kind", choices=["auto", "data", "mc"], default="auto", help="入力DSTの種類")
    export_hetero.add_argument("--const-dst", default=None, help="TALE-SD detector geometry DST。省略時はdstio TALE config/envを使う")
    export_hetero.add_argument(
        "--mc-calib-dir",
        default=None,
        help="MC rusdrawをtalesdcalibev相当へ変換するための校正ディレクトリ。省略時はdstio TALE config/envを使う",
    )
    export_hetero.add_argument("--max-events", type=int, default=None, help="読み込む最大イベント数")
    export_hetero.add_argument("--min-event-date", type=int, default=None, help="YYMMDD形式。この日付より前のeventをDST読み込み時に除外する")
    export_hetero.add_argument("--cleaning", choices=["ising", "none"], default="ising", help="dstio.tale.graph cleaning mode")
    export_hetero.add_argument(
        "--node-policy",
        choices=["all_candidates_with_ising", "all_candidates", "ising_kept"],
        default="all_candidates_with_ising",
        help="pulse node policy。ML graphではall_candidates_with_isingを基本にする",
    )
    export_hetero.add_argument("--require-reference-core", action="store_true", help="Ising reference core があるgraphだけを書き出す")
    export_hetero.add_argument("--shard-size", type=int, default=0, help="NグラフごとにHDF5を分割する。0なら分割しない")
    export_hetero.add_argument("--open-retries", type=int, default=3, help="DST open失敗時の再試行回数")
    export_hetero.add_argument("--open-retry-delay", type=float, default=1.0, help="DST open再試行の待ち時間。試行ごとに線形に増やす")
    export_hetero.add_argument("--keep-non-mode0", action="store_true", help="trgMode != 0 も残す")
    export_hetero.add_argument("--skip-errors", action="store_true", help="読めないDSTを警告してスキップする")
    export_hetero.add_argument("--skip-missing-mc-calibration", action="store_true", help="MC calibration が見つからないeventをスキップする")
    export_hetero.set_defaults(func=_cmd_export_hetero)

    train = sub.add_parser("train", help="MC truth付きグラフでGNNを学習")
    train.add_argument("--graphs", nargs="*", default=[], help="exportで作成したMC HDF5グラフ。shard、shard base、またはHDF5ディレクトリを指定可")
    train.add_argument("-o", "--output", required=True, help="出力checkpoint .pt")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--lr-scheduler", choices=["none", "reduce-on-plateau", "cosine"], default="none")
    train.add_argument("--lr-factor", type=float, default=0.5, help="reduce-on-plateauでLRを下げる倍率")
    train.add_argument("--lr-patience", type=int, default=2, help="reduce-on-plateauで何epoch改善なしを待つか")
    train.add_argument("--early-stopping-patience", type=int, default=0, help="0なら無効。指定epoch数validation改善なしで停止")
    train.add_argument("--early-stopping-min-epochs", type=int, default=0, help="early stoppingを有効にする前に最低限走らせるepoch数")
    train.add_argument("--model-architecture", choices=["baseline", "physics"], default="baseline")
    train.add_argument("--readout-heads", type=int, default=4, help="physics architectureのattention readout head数")
    train.add_argument(
        "--classification-arch",
        choices=["legacy", "enhanced"],
        default="enhanced",
        help="mass分類head。enhancedは初期node表現、最終GNN表現、hit/edge数を使う分類専用head",
    )
    train.add_argument("--detector-embedding-dim", type=int, default=0, help="検出器LIDごとのlearnable embedding次元。0なら無効")
    train.add_argument("--waveform-encoder", choices=["none", "cnn", "cnn-gru", "transformer"], default="none", help="nodeごとの波形trace encoder")
    train.add_argument("--waveform-embedding-dim", type=int, default=64, help="波形encoder出力次元")
    train.add_argument("--waveform-transformer-heads", type=int, default=4)
    train.add_argument("--waveform-transformer-layers", type=int, default=1)
    train.add_argument(
        "--loss-mode",
        choices=["scaled-mse", "weighted-scaled-mse", "hybrid-angle", "physics", "physics-nll", "nll"],
        default="scaled-mse",
    )
    train.add_argument("--energy-loss-weight", type=float, default=1.0)
    train.add_argument("--core-loss-weight", type=float, default=1.0)
    train.add_argument("--direction-loss-weight", type=float, default=1.0)
    train.add_argument("--core-loss-scale-km", type=float, default=0.05)
    train.add_argument("--angular-loss-scale-deg", type=float, default=1.0, help="角度lossをこの角度[deg]で正規化する")
    train.add_argument("--energy-bias-loss-weight", type=float, default=0.0, help="true energy binごとの平均logE residualを0に寄せるloss重み")
    train.add_argument(
        "--energy-particle-bias-loss-weight",
        type=float,
        default=0.0,
        help="同じtrue energy bin内でproton/ironの平均logE residual差を0に寄せるloss重み",
    )
    train.add_argument("--energy-bias-bin-width", type=float, default=0.1, help="energy bias lossのtrue log10(E/eV) bin幅")
    train.add_argument("--energy-bias-min-bin-count", type=int, default=8, help="energy bias lossで1 bin/classに必要な最小event数")
    train.add_argument("--val-fraction", type=float, default=0.05, help="validation event fraction")
    train.add_argument("--test-fraction", type=float, default=0.10, help="test event fraction")
    train.add_argument("--source-val-fraction", type=float, default=0.10, help="source-stratified splitでvalidationに割り当てるsource fraction")
    train.add_argument("--source-test-fraction", type=float, default=0.20, help="source-stratified splitでtestに割り当てるsource fraction")
    train.add_argument(
        "--split-mode",
        choices=["event", "source-path", "source-stratified"],
        default="event",
        help="event単位、source group単位、またはsource groupを保った物理パラメーター層化でtrain/validation/testを分ける",
    )
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--device", default="auto", help="auto, cpu, mps, cuda など")
    train.add_argument("--sample-cache-size", type=int, default=0, help="学習中にLRU cacheするグラフ数。0で無効")
    train.add_argument("--max-graphs", type=int, default=None, help="trainで読むgraph数の上限。速度試験用。未指定または0なら全件")
    train.add_argument(
        "--particle-filter",
        choices=["all", "proton", "iron"],
        default="all",
        help="学習に使う核種を絞る。allはproton/iron混合、protonはrusdmc.parttype=14、ironは5626のみ",
    )
    train.add_argument("--no-pin-memory", action="store_true", help="CUDA転送用のpinned memoryを使わない。大きいHDF5でCPU RSSを抑えたい場合に使う")
    train.add_argument("--num-workers", type=int, default=DEFAULT_TRAIN_WORKERS, help="学習DataLoaderのworker数。-1でauto、0で単一process")
    train.add_argument("--preprocess-workers", type=int, default=0, help="split scanとscaler fitに使う前処理worker数。0/1で単一process")
    train.add_argument("--prefetch-factor", type=int, default=2, help="各DataLoader workerが先読みするbatch数")
    train.add_argument("--persistent-workers", action="store_true", help="DataLoader workerをepoch間で保持する。大きいHDF5ではメモリを残しやすいので既定では無効")
    train.add_argument("--collate-backend", choices=["auto", "cpp", "python"], default="auto", help="batch構築backend。autoは小規模入力ではpython、大規模/worker利用時はcppを選ぶ")
    train.add_argument("--collate-threads", type=int, default=1, help="C++ collate内部のthread数。0ならautoまたはTALESD_GNN_COLLATE_THREADS")
    train.add_argument("--training-task", choices=["reconstruction", "mass"], default="reconstruction", help="reconstructionは幾何/エネルギー再構成、massはproton/iron分類のみを学習する")
    train.add_argument("--mass-classification", action="store_true", help="rusdmc.parttype由来のproton/iron分類headも同時に学習する")
    train.add_argument("--mass-loss-weight", type=float, default=0.1, help="proton/iron分類lossを再構成lossに足す重み")
    train.add_argument("--mass-loss-mode", choices=["bce", "focal"], default="focal", help="proton/iron分類loss")
    train.add_argument("--mass-focal-gamma", type=float, default=2.0, help="--mass-loss-mode focal のgamma")
    train.add_argument("--mass-pos-weight-mode", choices=["none", "auto"], default="none", help="autoならtrain proton/iron比からBCE pos_weightを使う")
    train.add_argument("--mass-ranking-weight", type=float, default=0.0, help="batch内のiron logitをproton logitより大きくするranking lossの重み")
    train.add_argument("--mass-ranking-margin", type=float, default=1.0, help="mass ranking lossで要求するlogit margin")
    train.add_argument("--mass-collapse-patience", type=int, default=3, help="mass-onlyでscoreが定数化したepochが続いたら停止。0で無効")
    train.add_argument("--mass-collapse-score-std", type=float, default=1.0e-3, help="定数化判定に使うP(iron)標準偏差")
    train.add_argument("--mass-collapse-balanced-accuracy", type=float, default=0.505, help="定数化判定に使うbalanced accuracy上限")
    train.add_argument("--quality-prediction", action="store_true", help="再構成の信頼度を0から1で返すquality headも同時に学習する")
    train.add_argument("--quality-loss-weight", type=float, default=0.2, help="quality lossを再構成lossに足す重み")
    train.add_argument("--quality-angular-scale-deg", type=float, default=1.0, help="quality教師値で1/eに近づく角度誤差スケール[deg]")
    train.add_argument("--quality-core-scale-km", type=float, default=0.05, help="quality教師値で1/eに近づくcore誤差スケール[km]")
    train.add_argument("--quality-energy-scale", type=float, default=0.10, help="quality教師値で1/eに近づく相対エネルギー誤差")
    train.add_argument("--error-prediction", action="store_true", help="energy/angle/coreのevent-wise予想誤差headも同時に学習する")
    train.add_argument("--error-loss-weight", type=float, default=0.2, help="予想誤差lossを再構成lossに足す重み")
    train.add_argument("--error-angular-scale-deg", type=float, default=1.0, help="予想角度誤差headの教師値スケール[deg]")
    train.add_argument("--error-core-scale-km", type=float, default=0.05, help="予想core誤差headの教師値スケール[km]")
    train.add_argument("--error-energy-scale", type=float, default=0.10, help="予想相対エネルギー誤差headの教師値スケール")
    train.add_argument("--nll-loss-weight", type=float, default=0.2, help="physics-nllでGaussian NLLをphysics lossに足す重み")
    train.add_argument("--nll-sigma-energy-floor", type=float, default=0.01, help="Gaussian NLLで使う相対エネルギーsigmaの下限")
    train.add_argument("--nll-sigma-angle-floor-deg", type=float, default=0.05, help="Gaussian NLLで使う角度sigma下限[deg]")
    train.add_argument("--nll-sigma-core-floor-km", type=float, default=0.005, help="Gaussian NLLで使うcore sigma下限[km]")
    train.add_argument("--no-progress", action="store_true", help="学習中のprogress barを表示しない")
    train.add_argument("--no-diagnostics", action="store_true", help="学習後のPDF診断図を保存しない")
    train.add_argument("--no-epoch-learning-curve", action="store_true", help="epochごとのlearning curve更新を止める")
    train.add_argument("--no-best-diagnostics", action="store_true", help="validation loss最良更新時の軽量診断図更新を止める")
    train.add_argument("--best-diagnostic-max-graphs", type=int, default=20000, help="最良更新時の診断に使うvalidation graph数の上限。0ならvalidation全件")
    train.add_argument("--diagnostic-energy-bin-width", type=float, default=0.1, help="診断図で使うtrue log10(E/eV) bin幅")
    train.add_argument("--diagnostic-min-bin-count", type=int, default=20, help="energy bin別診断に使う最小event数")
    train.set_defaults(func=_cmd_train)

    train_hetero = sub.add_parser(
        "train-hetero",
        help="new dstio.tale.graph heterogeneous HDF5で最小hetero GNNを学習する",
    )
    train_hetero.add_argument("--graphs", nargs="*", default=[], help="export-heteroで作成したhetero HDF5 graph")
    train_hetero.add_argument("--graphs-list", action="append", default=[], help="hetero HDF5 shard path list")
    train_hetero.add_argument("-o", "--output", required=True, help="出力checkpoint .pt")
    train_hetero.add_argument("--epochs", type=int, default=1)
    train_hetero.add_argument("--batch-size", type=int, default=8)
    train_hetero.add_argument("--lr", type=float, default=1.0e-3)
    train_hetero.add_argument("--weight-decay", type=float, default=0.0)
    train_hetero.add_argument("--hidden-dim", type=int, default=128)
    train_hetero.add_argument("--layers", type=int, default=2)
    train_hetero.add_argument("--dropout", type=float, default=0.05)
    train_hetero.add_argument("--waveform-encoder", choices=["none", "cnn", "cnn-gru", "transformer"], default="cnn")
    train_hetero.add_argument("--waveform-embedding-dim", type=int, default=64)
    train_hetero.add_argument(
        "--waveform-length",
        type=int,
        default=None,
        help="detector waveform の固定入力長。未指定ならtrain split内の最大長を使う",
    )
    train_hetero.add_argument(
        "--loss-mode",
        choices=["scaled-mse", "weighted-scaled-mse", "hybrid-angle", "physics", "physics-nll", "nll"],
        default="physics",
    )
    train_hetero.add_argument("--energy-loss-weight", type=float, default=1.0)
    train_hetero.add_argument("--core-loss-weight", type=float, default=1.0)
    train_hetero.add_argument("--direction-loss-weight", type=float, default=1.0)
    train_hetero.add_argument("--core-loss-scale-km", type=float, default=0.05)
    train_hetero.add_argument("--angular-loss-scale-deg", type=float, default=1.0)
    train_hetero.add_argument("--energy-bias-loss-weight", type=float, default=0.0)
    train_hetero.add_argument("--energy-particle-bias-loss-weight", type=float, default=0.0)
    train_hetero.add_argument("--energy-bias-bin-width", type=float, default=0.1)
    train_hetero.add_argument("--energy-bias-min-bin-count", type=int, default=8)
    train_hetero.add_argument("--mass-classification", action="store_true")
    train_hetero.add_argument("--mass-loss-weight", type=float, default=0.1)
    train_hetero.add_argument("--mass-loss-mode", choices=["bce", "focal"], default="bce")
    train_hetero.add_argument("--mass-focal-gamma", type=float, default=2.0)
    train_hetero.add_argument("--mass-ranking-weight", type=float, default=0.0)
    train_hetero.add_argument("--mass-ranking-margin", type=float, default=1.0)
    train_hetero.add_argument("--quality-prediction", action="store_true", help="quality headも同時に学習する")
    train_hetero.add_argument("--quality-loss-weight", type=float, default=0.2)
    train_hetero.add_argument("--quality-angular-scale-deg", type=float, default=1.0)
    train_hetero.add_argument("--quality-core-scale-km", type=float, default=0.05)
    train_hetero.add_argument("--quality-energy-scale", type=float, default=0.10)
    train_hetero.add_argument("--error-prediction", action="store_true", help="event-wise predicted error headを同時に学習する")
    train_hetero.add_argument("--error-loss-weight", type=float, default=0.2)
    train_hetero.add_argument("--error-angular-scale-deg", type=float, default=1.0)
    train_hetero.add_argument("--error-core-scale-km", type=float, default=0.05)
    train_hetero.add_argument("--error-energy-scale", type=float, default=0.10)
    train_hetero.add_argument("--nll-loss-weight", type=float, default=0.2)
    train_hetero.add_argument("--nll-sigma-energy-floor", type=float, default=0.01)
    train_hetero.add_argument("--nll-sigma-angle-floor-deg", type=float, default=0.05)
    train_hetero.add_argument("--nll-sigma-core-floor-km", type=float, default=0.005)
    train_hetero.add_argument("--val-fraction", type=float, default=0.1)
    train_hetero.add_argument("--test-fraction", type=float, default=0.1)
    train_hetero.add_argument("--source-val-fraction", type=float, default=0.10)
    train_hetero.add_argument("--source-test-fraction", type=float, default=0.20)
    train_hetero.add_argument(
        "--split-mode",
        choices=["event", "source-path", "source-stratified"],
        default="event",
    )
    train_hetero.add_argument("--seed", type=int, default=12345)
    train_hetero.add_argument("--device", default="auto")
    train_hetero.add_argument("--diagnostics", action="store_true", help="training後に既存diagnostics PDF/JSONを生成する")
    train_hetero.add_argument("--diagnostic-energy-bin-width", type=float, default=0.1)
    train_hetero.add_argument("--diagnostic-min-bin-count", type=int, default=20)
    train_hetero.add_argument("--no-progress", action="store_true")
    train_hetero.set_defaults(func=_cmd_train_hetero)

    reconstruct_dst = sub.add_parser(
        "reconstruct-dst",
        help="hetero checkpointを使いDSTをH5なしで直接再構成する",
    )
    reconstruct_dst.add_argument("input", nargs="*", help="入力DSTファイル")
    reconstruct_dst.add_argument("--input-list", action="append", default=[], help="DST path list")
    reconstruct_dst.add_argument("--input-dir", action="append", default=[], help="DSTを再帰検索するdirectory")
    reconstruct_dst.add_argument("--checkpoint", required=True, help="train-heteroで作成したhetero checkpoint")
    reconstruct_dst.add_argument("-o", "--output", required=True, help="出力CSV")
    reconstruct_dst.add_argument("--kind", choices=["auto", "data", "mc"], default="auto")
    reconstruct_dst.add_argument("--const-dst", default=None)
    reconstruct_dst.add_argument("--mc-calib-dir", default=None)
    reconstruct_dst.add_argument("--batch-size", type=int, default=128)
    reconstruct_dst.add_argument("--max-events", type=int, default=None)
    reconstruct_dst.add_argument("--device", default="auto")
    reconstruct_dst.add_argument("--cleaning", choices=["ising"], default="ising")
    reconstruct_dst.add_argument(
        "--node-policy",
        choices=["all_candidates_with_ising", "ising_kept"],
        default="all_candidates_with_ising",
    )
    reconstruct_dst.add_argument(
        "--allow-missing-reference-core",
        action="store_true",
        help="reference coreが無いgraphも推論する。通常は使わない",
    )
    reconstruct_dst.add_argument("--skip-errors", action="store_true")
    reconstruct_dst.add_argument("--skip-missing-mc-calibration", action="store_true")
    reconstruct_dst.add_argument("--open-retries", type=int, default=1)
    reconstruct_dst.add_argument("--open-retry-delay", type=float, default=0.0)
    reconstruct_dst.set_defaults(func=_cmd_reconstruct_dst)

    predict = sub.add_parser("predict", help="学習済みGNNで再構成結果CSVを作成")
    predict.add_argument("--graphs", nargs="*", default=[], help="exportで作成したHDF5グラフ。shardを複数指定可")
    predict.add_argument("--graphs-list", action="append", default=[], help="HDF5 shardパスを1行1ファイルで書いたリスト。複数指定可")
    predict.add_argument("--checkpoint", required=True, help="trainで作成したcheckpoint .pt")
    predict.add_argument("-o", "--output", required=True, help="出力CSV")
    predict.add_argument("--batch-size", type=int, default=64)
    predict.add_argument("--device", default="auto", help="auto, cpu, mps, cuda など")
    predict.add_argument("--no-truth", action="store_true", help="truth列を出力しない")
    predict.set_defaults(func=_cmd_predict)

    input_dist = sub.add_parser("input-distributions", help="HDF5グラフ入力特徴量の分布図と要約JSONを作成")
    input_dist.add_argument("--graphs", nargs="*", default=[], help="HDF5グラフ。shard、shard base、またはHDF5ディレクトリを指定可")
    input_dist.add_argument("--graphs-list", action="append", default=[], help="HDF5 shardパスを1行1ファイルで書いたリスト。複数指定可")
    input_dist.add_argument("-o", "--output", required=True, help="出力ディレクトリ")
    input_dist.add_argument("--max-graphs", type=int, default=100000, help="分布作成に使う最大graph数。0なら全件")
    input_dist.add_argument("--max-values-per-feature", type=int, default=200000, help="各特徴量で保持する最大値数")
    input_dist.add_argument("--seed", type=int, default=12345)
    input_dist.add_argument("--no-progress", action="store_true")
    input_dist.set_defaults(func=_cmd_input_distributions)

    importance = sub.add_parser("feature-importance", help="学習済みcheckpointに対する特徴量group ablation重要度を評価")
    importance.add_argument("--graphs", nargs="*", default=[], help="HDF5グラフ。checkpoint作成時と同じgraph集合を指定する")
    importance.add_argument("--graphs-list", action="append", default=[], help="HDF5 shardパスを1行1ファイルで書いたリスト。複数指定可")
    importance.add_argument("--checkpoint", required=True, help="評価するcheckpoint .pt")
    importance.add_argument("-o", "--output", required=True, help="出力ディレクトリ")
    importance.add_argument("--split", choices=["validation", "val", "test", "train"], default="validation", help="checkpoint内のどのsplitで評価するか")
    importance.add_argument("--max-graphs", type=int, default=50000, help="評価に使う最大graph数。0ならsplit全件")
    importance.add_argument("--batch-size", type=int, default=256)
    importance.add_argument("--device", default="auto")
    importance.add_argument("--seed", type=int, default=12345)
    importance.add_argument("--no-progress", action="store_true")
    importance.set_defaults(func=_cmd_feature_importance)

    visualize = sub.add_parser("visualize", help="HDF5グラフをPDFとして描画")
    visualize.add_argument("--graphs", nargs="*", default=[], help="exportで作成したHDF5グラフ。shardを複数指定可")
    visualize.add_argument("--graphs-list", action="append", default=[], help="HDF5 shardパスを1行1ファイルで書いたリスト。複数指定可")
    visualize.add_argument("-o", "--output", required=True, help="出力PDF。複数描画時または拡張子なしなら出力ディレクトリ")
    visualize.add_argument("--index", type=int, default=0, help="描画するグラフindex")
    visualize.add_argument("--event-id", default=None, help="event_idで選ぶ。指定時は--indexより優先")
    visualize.add_argument("--const-dst", default=None, help="背景SD配置に使うTALE-SD detector geometry DST")
    visualize.add_argument("--count", type=int, default=1, help="連続して描画するイベント数")
    visualize.add_argument("--no-edges", action="store_true", help="GNN edgeを描画しない")
    visualize.add_argument("--annotate-lids", action="store_true", help="各ノードにSD lidを表示")
    visualize.add_argument("--max-edges", type=int, default=2000, help="描画する最大edge数")
    visualize.add_argument("--dpi", type=int, default=160, help="出力PDF内のraster要素DPI")
    visualize.set_defaults(func=_cmd_visualize)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
