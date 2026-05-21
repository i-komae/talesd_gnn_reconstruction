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
DEFAULT_WORKER_MAX_FILES = 200
DEFAULT_TRAIN_WORKERS = -1


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
        "Use a smaller --worker-max-files value, or keep the default worker recycling enabled."
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


def _scan_energy_candidates_for_file(payload: tuple[str, float, int, int, bool, int, float, bool]) -> dict[str, Any]:
    import dstio

    path, bin_width, seed, per_bin_limit, skip_errors, open_retries, open_retry_delay, stratify_particle = payload
    reservoirs: dict[int | tuple[str, int], list[tuple[float, str, int, float]]] = {}
    seen_by_bin: dict[int | tuple[str, int], int] = {}
    raw_events = 0
    hit_events = 0
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
                entry = (-key, f"{path}:{source_index}", int(source_index), float(log10_energy))
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
            "error": str(exc),
        }
    return {
        "path": path,
        "reservoirs": reservoirs,
        "seen_by_bin": seen_by_bin,
        "raw_events": raw_events,
        "hit_events": hit_events,
        "error": None,
    }


def _build_graphs_for_file(
    payload: tuple[str, dict[int, Any] | None, str, bool, bool, set[int] | None, int, float, int | None]
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
) -> tuple[dict[str, set[int]], dict[int | tuple[str, int], int], dict[int | tuple[str, int], int], int, int]:
    merged: dict[int | tuple[str, int], list[tuple[float, str, str, int, float]]] = {}
    seen_by_bin: dict[int | tuple[str, int], int] = {}
    raw_events = 0
    hit_events = 0

    for result in scan_results:
        if result.get("error"):
            _progress_write(f"warning: skipping unreadable DST {result['path']}: {result['error']}")
        raw_events += int(result.get("raw_events", 0))
        hit_events += int(result.get("hit_events", 0))
        for bin_key, count in result.get("seen_by_bin", {}).items():
            seen_by_bin[bin_key] = seen_by_bin.get(bin_key, 0) + int(count)
        for bin_key, entries in result.get("reservoirs", {}).items():
            bucket = merged.setdefault(bin_key, [])
            for neg_key, unique_id, source_index, log10_energy in entries:
                entry = (float(neg_key), str(unique_id), str(result["path"]), int(source_index), float(log10_energy))
                if len(bucket) < per_bin_limit:
                    heapq.heappush(bucket, entry)
                elif entry[0] > bucket[0][0]:
                    heapq.heapreplace(bucket, entry)

    selected_by_path: dict[str, set[int]] = {}
    selected_by_bin: dict[int | tuple[str, int], int] = {}
    for bin_key, entries in merged.items():
        selected_by_bin[bin_key] = len(entries)
        for _neg_key, _unique_id, path, source_index, _log10_energy in entries:
            selected_by_path.setdefault(path, set()).add(int(source_index))
    return selected_by_path, seen_by_bin, selected_by_bin, raw_events, hit_events


def _cmd_export(args: argparse.Namespace) -> None:
    from .dst_reader import iter_dst_banks

    inputs = _resolve_input_args(args.input, args.input_list, args.input_dir)
    const_dst = Path(args.const_dst).expanduser() if args.const_dst else default_const_dst_path()
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
        "max_events": args.max_events,
        "max_events_per_file": args.max_events_per_file,
        "graph_definition": "coincidence_analysis_ising_pulse_graph",
        "energy_sample_per_bin": args.energy_sample_per_bin,
        "energy_sample_stratify_particle": bool(args.energy_sample_stratify_particle),
        "energy_bin_width": args.energy_bin_width,
        "energy_oversample_factor": args.energy_oversample_factor,
        "seed": args.seed,
        "workers": args.workers,
        "worker_max_files": args.worker_max_files,
        "chunk_size": args.chunk_size,
        "shard_size": args.shard_size,
        "open_retries": args.open_retries,
        "open_retry_delay": args.open_retry_delay,
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
            selected_by_path, seen_by_bin, selected_by_bin, raw_events, hit_events = _merge_candidate_reservoirs(
                _iter_scan_results(inputs, args, preselect_per_bin=preselect_per_bin),
                per_bin_limit=preselect_per_bin,
            )
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
            config["scan_selected_files"] = len(selected_by_path)

            file_results = _iter_file_results(inputs, args, detector_positions, selected_indices_by_path=selected_by_path)
            graph_seen_by_bin: dict[int | tuple[str, int], int] = {}
            per_bin = max(int(args.energy_sample_per_bin), 1)
            if preselect_per_bin <= per_bin:
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
        model_architecture=args.model_architecture,
        readout_heads=args.readout_heads,
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
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
        device=args.device,
        sample_cache_size=args.sample_cache_size,
        max_graphs=args.max_graphs,
        particle_filter=args.particle_filter,
        num_workers=args.num_workers,
        preprocess_workers=args.preprocess_workers,
        prefetch_factor=args.prefetch_factor,
        collate_backend=args.collate_backend,
        collate_threads=args.collate_threads,
        mass_classification=args.mass_classification,
        mass_loss_weight=args.mass_loss_weight,
        show_progress=not args.no_progress,
        save_diagnostics=not args.no_diagnostics,
        diagnostic_energy_bin_width=args.diagnostic_energy_bin_width,
        diagnostic_min_bin_count=args.diagnostic_min_bin_count,
    )
    print(f"checkpoint: {result['checkpoint']}")
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
    export.add_argument("--const-dst", default=None, help="TALE-SD calibration DST talesdconst_pass2.dst。MC入力で必須")
    export.add_argument("--max-events", type=int, default=None, help="読み込む最大イベント数")
    export.add_argument("--max-events-per-file", type=int, default=None, help="ファイル単位export時に各DSTから読む最大イベント数。source-path splitの小規模試験用")
    export.add_argument("--energy-sample-per-bin", type=int, default=None, help="log10(E/eV) binごとに残す最大グラフ数。reservoir samplingで時刻順バイアスを避ける")
    export.add_argument("--energy-sample-stratify-particle", action="store_true", help="energy-flat samplingをproton/iron別のlog10(E/eV) binで行う")
    export.add_argument("--energy-bin-width", type=float, default=0.1, help="energy-flat samplingのlog10(E/eV) bin幅")
    export.add_argument("--energy-oversample-factor", type=float, default=2.0, help="先行metadata scanで各energy binからgraph化対象として余分に残す倍率")
    export.add_argument("--seed", type=int, default=12345, help="energy-flat samplingの乱数seed")
    export.add_argument("--workers", type=int, default=1, help="DST読み込みとグラフ構築に使うファイル単位worker数。--max-events指定時だけイベントchunk単位")
    export.add_argument("--worker-max-files", type=int, default=DEFAULT_WORKER_MAX_FILES, help="ファイル単位workerをNファイル処理ごとに再起動する。0なら無効")
    export.add_argument("--chunk-size", type=int, default=128, help="--max-events指定時にworkerへ渡すイベントchunkサイズ")
    export.add_argument("--shard-size", type=int, default=0, help="NグラフごとにHDF5を分割する。0なら分割しない")
    export.add_argument("--open-retries", type=int, default=3, help="DST open失敗時の再試行回数")
    export.add_argument("--open-retry-delay", type=float, default=1.0, help="DST open再試行の待ち時間。試行ごとに線形に増やす")
    export.add_argument("--keep-non-mode0", action="store_true", help="trgMode != 0 も残す")
    export.add_argument("--skip-errors", action="store_true", help="読めないDSTを警告してスキップする")
    export.set_defaults(func=_cmd_export)

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
    train.add_argument("--model-architecture", choices=["baseline", "physics"], default="baseline")
    train.add_argument("--readout-heads", type=int, default=4, help="physics architectureのattention readout head数")
    train.add_argument("--detector-embedding-dim", type=int, default=0, help="検出器LIDごとのlearnable embedding次元。0なら無効")
    train.add_argument("--waveform-encoder", choices=["none", "cnn", "cnn-gru", "transformer"], default="none", help="nodeごとの波形trace encoder")
    train.add_argument("--waveform-embedding-dim", type=int, default=64, help="波形encoder出力次元")
    train.add_argument("--waveform-transformer-heads", type=int, default=4)
    train.add_argument("--waveform-transformer-layers", type=int, default=1)
    train.add_argument(
        "--loss-mode",
        choices=["scaled-mse", "weighted-scaled-mse", "hybrid-angle", "physics"],
        default="scaled-mse",
    )
    train.add_argument("--energy-loss-weight", type=float, default=1.0)
    train.add_argument("--core-loss-weight", type=float, default=1.0)
    train.add_argument("--direction-loss-weight", type=float, default=1.0)
    train.add_argument("--core-loss-scale-km", type=float, default=0.12)
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--test-fraction", type=float, default=0.1)
    train.add_argument(
        "--split-mode",
        choices=["event", "source-path", "source-stratified"],
        default="event",
        help="event単位、元DST source_path単位、またはsource_pathを保った物理パラメーター層化でtrain/validation/testを分ける",
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
    train.add_argument("--num-workers", type=int, default=DEFAULT_TRAIN_WORKERS, help="学習DataLoaderのworker数。-1でauto、0で単一process")
    train.add_argument("--preprocess-workers", type=int, default=0, help="split scanとscaler fitに使う前処理worker数。0/1で単一process")
    train.add_argument("--prefetch-factor", type=int, default=2, help="各DataLoader workerが先読みするbatch数")
    train.add_argument("--collate-backend", choices=["auto", "cpp", "python"], default="auto", help="batch構築backend。autoは小規模入力ではpython、大規模/worker利用時はcppを選ぶ")
    train.add_argument("--collate-threads", type=int, default=1, help="C++ collate内部のthread数。0ならautoまたはTALESD_GNN_COLLATE_THREADS")
    train.add_argument("--mass-classification", action="store_true", help="rusdmc.parttype由来のproton/iron分類headも同時に学習する")
    train.add_argument("--mass-loss-weight", type=float, default=0.1, help="proton/iron分類lossを再構成lossに足す重み")
    train.add_argument("--no-progress", action="store_true", help="学習中のprogress barを表示しない")
    train.add_argument("--no-diagnostics", action="store_true", help="学習後のPDF診断図を保存しない")
    train.add_argument("--diagnostic-energy-bin-width", type=float, default=0.1, help="診断図で使うtrue log10(E/eV) bin幅")
    train.add_argument("--diagnostic-min-bin-count", type=int, default=20, help="energy bin別診断に使う最小event数")
    train.set_defaults(func=_cmd_train)

    predict = sub.add_parser("predict", help="学習済みGNNで再構成結果CSVを作成")
    predict.add_argument("--graphs", nargs="*", default=[], help="exportで作成したHDF5グラフ。shardを複数指定可")
    predict.add_argument("--graphs-list", action="append", default=[], help="HDF5 shardパスを1行1ファイルで書いたリスト。複数指定可")
    predict.add_argument("--checkpoint", required=True, help="trainで作成したcheckpoint .pt")
    predict.add_argument("-o", "--output", required=True, help="出力CSV")
    predict.add_argument("--batch-size", type=int, default=64)
    predict.add_argument("--device", default="auto", help="auto, cpu, mps, cuda など")
    predict.add_argument("--no-truth", action="store_true", help="truth列を出力しない")
    predict.set_defaults(func=_cmd_predict)

    visualize = sub.add_parser("visualize", help="HDF5グラフをPDFとして描画")
    visualize.add_argument("--graphs", nargs="*", default=[], help="exportで作成したHDF5グラフ。shardを複数指定可")
    visualize.add_argument("--graphs-list", action="append", default=[], help="HDF5 shardパスを1行1ファイルで書いたリスト。複数指定可")
    visualize.add_argument("-o", "--output", required=True, help="出力PDF。複数描画時または拡張子なしなら出力ディレクトリ")
    visualize.add_argument("--index", type=int, default=0, help="描画するグラフindex")
    visualize.add_argument("--event-id", default=None, help="event_idで選ぶ。指定時は--indexより優先")
    visualize.add_argument("--const-dst", default=None, help="背景SD配置に使うTALE-SD calibration DST")
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
