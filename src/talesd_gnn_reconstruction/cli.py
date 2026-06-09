from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import re
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
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
_DAT_TAG_RE = re.compile(r"(DAT\d{6})", re.IGNORECASE)
_GEA_TRG_RE = re.compile(r"_gea_trg_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class HeteroSelectionCandidate:
    bin_key: int | tuple[str, int]
    unique_id: str
    source_path: str
    source_group: str
    source_index: int
    log10_energy: float
    particle: str
    zenith_deg: float
    azimuth_deg: float
    core_x_km: float
    core_y_km: float
    date: int
    time_value: int
    sort_key: float
    balance_key: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HeteroSourceFileManifest:
    path: str
    source_group: str
    dat_tag: str
    energy_bin_code: str
    particle: str
    gea_trg_index: int
    source_zenith_deg: float
    eligible_event_count: int
    date_counts: dict[str, int]
    cell_counts: dict[tuple[str, ...], int]


@dataclass(frozen=True, slots=True)
class HeteroSourceGroupManifest:
    source_group: str
    dat_tag: str
    energy_bin_code: str
    particle: str
    source_zenith_deg: float
    eligible_event_count: int
    files: tuple[HeteroSourceFileManifest, ...]
    date_counts: dict[str, int]
    cell_counts: dict[tuple[str, ...], int]


@dataclass(frozen=True, slots=True)
class HeteroH5EventEntry:
    bin_key: int | tuple[str, int]
    unique_id: str
    h5_path: str
    local_index: int
    source_path: str
    source_index: int


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

    try:
        pool = _make_process_pool(workers, max_tasks_per_child=max_tasks_per_child)
    except (OSError, PermissionError) as exc:
        _progress_write(f"warning: process pool unavailable for {desc}; falling back to serial: {exc}")
        for payload in _progress(payloads, desc=desc, total=len(payloads)):
            yield worker_fn(payload)
        return
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


def _nested_float(value: Any, *keys: int, default: float = float("nan")) -> float:
    try:
        obj = value
        for key in keys:
            obj = obj[key]
        return float(obj)
    except Exception:
        return float(default)


def _time_seconds_from_hhmmss(value: int) -> int:
    value = max(int(value), 0)
    hh = value // 10000
    mm = (value // 100) % 100
    ss = value % 100
    if hh > 23 or mm > 59 or ss > 59:
        return 0
    return hh * 3600 + mm * 60 + ss


def _finite_bin_label(value: float, width: float, *, circular: float | None = None) -> str:
    value = float(value)
    width = max(float(width), 1.0e-12)
    if not math.isfinite(value):
        return "nan"
    if circular is not None:
        value = value % float(circular)
    return str(int(math.floor(value / width)))


def _source_group_key_for_path(source_path: str) -> str:
    from .train import source_group_key

    return source_group_key(source_path)


def _dat_tag_from_path(source_path: str) -> str:
    match = _DAT_TAG_RE.search(Path(source_path).name)
    if match is None:
        raise ValueError(f"DST filename does not contain DAT?????? tag: {source_path}")
    return match.group(1).upper()


def _energy_bin_code_from_dat_tag(dat_tag: str) -> str:
    digits = str(dat_tag).upper().replace("DAT", "")
    if len(digits) < 2 or not digits[-2:].isdigit():
        raise ValueError(f"DAT tag does not contain a two-digit energy code: {dat_tag}")
    return digits[-2:]


def _gea_trg_index_from_path(source_path: str) -> int:
    match = _GEA_TRG_RE.search(Path(source_path).name)
    return -1 if match is None else int(match.group(1))


def _particle_stratum_from_path(source_path: str) -> str:
    parts = [part.lower() for part in Path(source_path).parts]
    joined = "/".join(parts)
    if "proton" in joined:
        return "proton"
    if "iron" in joined:
        return "iron"
    return "unknown"


def _source_group_bin_key(group: HeteroSourceGroupManifest, *, stratify_particle: bool) -> str:
    if stratify_particle:
        return f"{group.particle}:{group.energy_bin_code}"
    return group.energy_bin_code


def _path_energy_code_int(source_path: str) -> int:
    try:
        return int(_energy_bin_code_from_dat_tag(_dat_tag_from_path(source_path)))
    except ValueError:
        return -1


def _hetero_output_bin_key_from_path(source_path: str, *, stratify_particle: bool) -> int | tuple[str, int]:
    energy_code = _path_energy_code_int(source_path)
    if stratify_particle:
        return (_particle_stratum_from_path(source_path), energy_code)
    return energy_code


def _hetero_bin_key_label(bin_key: int | tuple[str, int] | str) -> str:
    if isinstance(bin_key, tuple):
        return f"{bin_key[0]}:{int(bin_key[1])}"
    return str(bin_key)


def _merge_selected_indices(target: dict[str, set[int]], source: dict[str, set[int]]) -> None:
    for path, indices in source.items():
        target.setdefault(path, set()).update(int(index) for index in indices)


def _merge_count_dict(target: dict[str, int], source: dict[Any, Any]) -> None:
    for key, value in source.items():
        label = _hetero_bin_key_label(key)
        target[label] = target.get(label, 0) + int(value)


def _hetero_missing_bin_counts(
    desired_by_bin: dict[str, int],
    written_by_bin: dict[str, int],
) -> dict[str, int]:
    return {
        str(bin_key): int(target) - int(written_by_bin.get(str(bin_key), 0))
        for bin_key, target in desired_by_bin.items()
        if int(target) > int(written_by_bin.get(str(bin_key), 0))
    }


def _hetero_refill_bin_targets(
    desired_by_bin: dict[str, int],
    written_by_bin: dict[str, int],
    selected_by_bin: dict[str, int],
    *,
    safety_factor: float,
    min_efficiency: float,
) -> dict[str, int]:
    safety = max(float(safety_factor), 1.0)
    min_eff = min(max(float(min_efficiency), 1.0e-9), 1.0)
    refill_targets: dict[str, int] = {}
    for bin_key, missing in _hetero_missing_bin_counts(desired_by_bin, written_by_bin).items():
        selected = max(int(selected_by_bin.get(bin_key, 0)), 0)
        written = max(int(written_by_bin.get(bin_key, 0)), 0)
        efficiency = float(written) / float(selected) if selected > 0 else 0.0
        effective_efficiency = max(efficiency, min_eff)
        refill_targets[bin_key] = int(math.ceil(float(missing) * safety / effective_efficiency))
    return refill_targets


def _ordered_selected_entries(
    entries: list[SelectedEntry],
    *,
    output_order: str,
    seed: int,
    locality_run_size: int,
) -> list[SelectedEntry]:
    if output_order == "source":
        return list(entries)
    if output_order == "random":
        return sorted(
            entries,
            key=lambda entry: _sample_key_from_parts(int(seed) + 32452843, entry[1], entry[2], int(entry[3])),
        )
    if output_order == "interleaved":
        return _interleaved_selected_entries(entries, seed=seed, locality_run_size=locality_run_size)
    raise ValueError(f"unsupported output_order: {output_order}")


def _ordered_hetero_h5_entries(
    entries: list[HeteroH5EventEntry],
    *,
    output_order: str,
    seed: int,
    locality_run_size: int,
) -> list[HeteroH5EventEntry]:
    if output_order == "source":
        return list(entries)
    if output_order == "random":
        return sorted(
            entries,
            key=lambda entry: _sample_key_from_parts(
                int(seed) + 49979687,
                entry.unique_id,
                entry.source_path,
                int(entry.source_index),
            ),
        )
    if output_order != "interleaved":
        raise ValueError(f"unsupported output_order: {output_order}")

    run_size = max(int(locality_run_size), 1)
    by_bin_and_source: dict[int | tuple[str, int], dict[str, list[HeteroH5EventEntry]]] = {}
    for entry in entries:
        by_bin_and_source.setdefault(entry.bin_key, {}).setdefault(entry.source_path, []).append(entry)

    runs_by_bin: dict[int | tuple[str, int], deque[list[HeteroH5EventEntry]]] = {}
    for bin_key, by_source in by_bin_and_source.items():
        runs: list[list[HeteroH5EventEntry]] = []
        for source_entries in by_source.values():
            source_entries = sorted(source_entries, key=lambda entry: entry.source_index)
            for start in range(0, len(source_entries), run_size):
                runs.append(source_entries[start : start + run_size])
        runs.sort(
            key=lambda run: _sample_key_from_parts(
                int(seed) + 67867967,
                run[0].unique_id,
                run[0].source_path,
                int(run[0].source_index),
            )
        )
        runs_by_bin[bin_key] = deque(runs)

    bin_order = list(runs_by_bin)
    bin_order.sort(key=lambda bin_key: _sample_key_from_parts(int(seed) + 86028121, str(bin_key), "", 0))
    ordered: list[HeteroH5EventEntry] = []
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


def _sample_hetero_h5_entries_by_bin(
    entries: list[HeteroH5EventEntry],
    *,
    per_bin: int,
    seed: int,
) -> tuple[list[HeteroH5EventEntry], dict[str, Any]]:
    target = max(int(per_bin), 1)
    by_bin: dict[int | tuple[str, int], list[HeteroH5EventEntry]] = {}
    for entry in entries:
        by_bin.setdefault(entry.bin_key, []).append(entry)

    selected: list[HeteroH5EventEntry] = []
    by_bin_summary: dict[str, Any] = {}
    for bin_key, bucket in sorted(by_bin.items(), key=lambda item: str(item[0])):
        by_source_group: dict[str, list[HeteroH5EventEntry]] = {}
        for entry in bucket:
            by_source_group.setdefault(_source_group_key_for_path(entry.source_path), []).append(entry)
        for source_entries in by_source_group.values():
            source_entries.sort(
                key=lambda entry: _sample_key_from_parts(
                    int(seed) + 982451653,
                    entry.unique_id,
                    entry.source_path,
                    int(entry.source_index),
                )
            )
        source_order = sorted(
            by_source_group,
            key=lambda source_group: _sample_key_from_parts(
                int(seed) + 961748941,
                source_group,
                str(bin_key),
                0,
            ),
        )
        chosen: list[HeteroH5EventEntry] = []
        while source_order and len(chosen) < target:
            next_order: list[str] = []
            for source_group in source_order:
                if len(chosen) >= target:
                    break
                source_entries = by_source_group[source_group]
                if source_entries:
                    chosen.append(source_entries.pop(0))
                if source_entries:
                    next_order.append(source_group)
            source_order = next_order
        selected.extend(chosen)
        by_bin_summary[str(bin_key)] = {
            "input_events": len(bucket),
            "selected_events": len(chosen),
            "source_groups": len(by_source_group),
            "short_by_availability": max(target - len(chosen), 0),
        }

    return selected, {
        "energy_sample_per_bin": target,
        "input_events": len(entries),
        "selected_events": len(selected),
        "by_bin": by_bin_summary,
    }


def _candidate_balance_key(
    *,
    zenith_deg: float,
    azimuth_deg: float,
    core_x_km: float,
    core_y_km: float,
    time_value: int,
    zenith_bin_width_deg: float,
    azimuth_bin_width_deg: float,
    core_bin_width_km: float,
    time_bin_width_sec: int,
) -> tuple[str, ...]:
    time_bin_width = max(int(time_bin_width_sec), 1)
    return (
        _finite_bin_label(zenith_deg, zenith_bin_width_deg),
        _finite_bin_label(azimuth_deg, azimuth_bin_width_deg, circular=360.0),
        _finite_bin_label(core_x_km, core_bin_width_km),
        _finite_bin_label(core_y_km, core_bin_width_km),
        str(_time_seconds_from_hhmmss(time_value) // time_bin_width),
    )


def _event_balance_key_from_mc_event(
    event: dict[str, Any],
    *,
    azimuth_bin_width_deg: float,
    core_bin_width_km: float,
    time_bin_width_sec: int,
) -> tuple[str, ...]:
    rusdraw = event.get("rusdraw") or {}
    rusdmc = event.get("rusdmc") or {}
    phi = float(rusdmc.get("phi", float("nan")) or float("nan")) + math.pi
    azimuth_deg = math.degrees(phi) % 360.0 if math.isfinite(phi) else float("nan")
    core_x_km = _nested_float(rusdmc.get("corexyz"), 0, default=float("nan")) / 1.0e5
    core_y_km = _nested_float(rusdmc.get("corexyz"), 1, default=float("nan")) / 1.0e5
    time_value = int(rusdraw.get("hhmmss", 0) or 0)
    return (
        _finite_bin_label(azimuth_deg, azimuth_bin_width_deg, circular=360.0),
        _finite_bin_label(core_x_km, core_bin_width_km),
        _finite_bin_label(core_y_km, core_bin_width_km),
        str(_time_seconds_from_hhmmss(time_value) // max(int(time_bin_width_sec), 1)),
    )


def _candidate_from_mc_event(
    path: str,
    source_index: int,
    event: dict[str, Any],
    *,
    bin_width: float,
    seed: int,
    stratify_particle: bool,
    zenith_bin_width_deg: float,
    azimuth_bin_width_deg: float,
    core_bin_width_km: float,
    time_bin_width_sec: int,
) -> HeteroSelectionCandidate | None:
    rusdraw = event.get("rusdraw") or {}
    rusdmc = event.get("rusdmc") or {}
    energy_eev = float(rusdmc.get("energy", 0.0) or 0.0)
    if energy_eev <= 0.0 or not math.isfinite(energy_eev):
        return None
    theta = float(rusdmc.get("theta", float("nan")) or float("nan"))
    phi = float(rusdmc.get("phi", float("nan")) or float("nan")) + math.pi
    zenith_deg = math.degrees(theta) if math.isfinite(theta) else float("nan")
    azimuth_deg = math.degrees(phi) % 360.0 if math.isfinite(phi) else float("nan")
    core_x_km = _nested_float(rusdmc.get("corexyz"), 0, default=float("nan")) / 1.0e5
    core_y_km = _nested_float(rusdmc.get("corexyz"), 1, default=float("nan")) / 1.0e5
    date = int(rusdraw.get("yymmdd", 0) or 0)
    time_value = int(rusdraw.get("hhmmss", 0) or 0)
    log10_energy = math.log10(energy_eev * 1.0e18)
    particle = _particle_stratum_from_parttype(rusdmc.get("parttype", -1))
    bin_particle = particle if stratify_particle else None
    bin_key = _energy_sample_bin_key(log10_energy, bin_width, particle=bin_particle)
    event_id = _candidate_event_id(path, source_index, event)
    sort_key = _sample_key_from_parts(seed, event_id, path, source_index)
    source_group = _source_group_key_for_path(path)
    return HeteroSelectionCandidate(
        bin_key=bin_key,
        unique_id=f"{path}:{source_index}",
        source_path=path,
        source_group=source_group,
        source_index=int(source_index),
        log10_energy=float(log10_energy),
        particle=particle,
        zenith_deg=float(zenith_deg),
        azimuth_deg=float(azimuth_deg),
        core_x_km=float(core_x_km),
        core_y_km=float(core_y_km),
        date=date,
        time_value=time_value,
        sort_key=float(sort_key),
        balance_key=_candidate_balance_key(
            zenith_deg=zenith_deg,
            azimuth_deg=azimuth_deg,
            core_x_km=core_x_km,
            core_y_km=core_y_km,
            time_value=time_value,
            zenith_bin_width_deg=zenith_bin_width_deg,
            azimuth_bin_width_deg=azimuth_bin_width_deg,
            core_bin_width_km=core_bin_width_km,
            time_bin_width_sec=time_bin_width_sec,
        ),
    )


def _cap_hetero_candidates_by_cell(
    candidates: Iterable[HeteroSelectionCandidate],
    *,
    cap: int,
) -> list[HeteroSelectionCandidate]:
    cap = max(int(cap), 1)
    heaps: dict[tuple[int | tuple[str, int], str, tuple[str, ...]], list[tuple[float, str, HeteroSelectionCandidate]]] = {}
    for candidate in candidates:
        key = (candidate.bin_key, candidate.source_group, candidate.balance_key)
        entry = (-candidate.sort_key, candidate.unique_id, candidate)
        bucket = heaps.setdefault(key, [])
        if len(bucket) < cap:
            heapq.heappush(bucket, entry)
        elif entry[0] > bucket[0][0]:
            heapq.heapreplace(bucket, entry)
    merged: list[HeteroSelectionCandidate] = []
    for entries in heaps.values():
        merged.extend(entry[2] for entry in entries)
    return merged


def _interleaved_candidates_by_cell(
    candidates: list[HeteroSelectionCandidate],
    *,
    seed: int,
) -> list[HeteroSelectionCandidate]:
    cells: dict[tuple[str, ...], list[HeteroSelectionCandidate]] = {}
    for candidate in candidates:
        cells.setdefault(candidate.balance_key, []).append(candidate)
    for bucket in cells.values():
        bucket.sort(key=lambda item: (item.sort_key, item.unique_id))
    cell_order = sorted(
        cells,
        key=lambda key: _sample_key_from_parts(seed, "|".join(key), "cell", 0),
    )
    out: list[HeteroSelectionCandidate] = []
    while cell_order:
        next_order: list[tuple[str, ...]] = []
        for key in cell_order:
            bucket = cells[key]
            if bucket:
                out.append(bucket.pop(0))
            if bucket:
                next_order.append(key)
        cell_order = next_order
    return out


def _select_balanced_hetero_candidates(
    candidates: Iterable[HeteroSelectionCandidate],
    *,
    per_bin: int,
    seed: int,
) -> list[HeteroSelectionCandidate]:
    by_bin: dict[int | tuple[str, int], dict[str, list[HeteroSelectionCandidate]]] = {}
    for candidate in candidates:
        by_bin.setdefault(candidate.bin_key, {}).setdefault(candidate.source_group, []).append(candidate)

    selected: list[HeteroSelectionCandidate] = []
    for bin_key in sorted(by_bin, key=lambda key: str(key)):
        source_candidates = {
            source_group: _interleaved_candidates_by_cell(bucket, seed=seed)
            for source_group, bucket in by_bin[bin_key].items()
        }
        source_order = sorted(
            source_candidates,
            key=lambda source_group: _sample_key_from_parts(seed, source_group, str(bin_key), 0),
        )
        selected_in_bin = 0
        while source_order and selected_in_bin < int(per_bin):
            next_order: list[str] = []
            for source_group in source_order:
                bucket = source_candidates[source_group]
                if bucket and selected_in_bin < int(per_bin):
                    selected.append(bucket.pop(0))
                    selected_in_bin += 1
                if bucket:
                    next_order.append(source_group)
            source_order = next_order
    random.Random(seed).shuffle(selected)
    return selected


def _count_by(items: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _hetero_selection_summary(
    candidates: list[HeteroSelectionCandidate],
    *,
    bin_width: float,
    zenith_bin_width_deg: float,
    azimuth_bin_width_deg: float,
    core_bin_width_km: float,
    time_bin_width_sec: int,
) -> dict[str, Any]:
    source_counts = _count_by(candidate.source_group for candidate in candidates)
    top_sources = sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    time_bin_width = max(int(time_bin_width_sec), 1)
    return {
        "events": len(candidates),
        "source_groups": len(source_counts),
        "by_energy_bin": _count_by(_energy_sample_bin_label(candidate.bin_key, bin_width) for candidate in candidates),
        "by_particle": _count_by(candidate.particle for candidate in candidates),
        "by_zenith_bin": _count_by(
            _finite_bin_label(candidate.zenith_deg, zenith_bin_width_deg) for candidate in candidates
        ),
        "by_azimuth_bin": _count_by(
            _finite_bin_label(candidate.azimuth_deg, azimuth_bin_width_deg, circular=360.0)
            for candidate in candidates
        ),
        "by_core_x_bin": _count_by(
            _finite_bin_label(candidate.core_x_km, core_bin_width_km) for candidate in candidates
        ),
        "by_core_y_bin": _count_by(
            _finite_bin_label(candidate.core_y_km, core_bin_width_km) for candidate in candidates
        ),
        "by_time_bin": _count_by(
            str(_time_seconds_from_hhmmss(candidate.time_value) // time_bin_width) for candidate in candidates
        ),
        "by_date": _count_by(f"{candidate.date:06d}" for candidate in candidates),
        "top_source_groups": dict(top_sources),
    }


def _selected_by_path_from_candidates(candidates: Iterable[HeteroSelectionCandidate]) -> dict[str, set[int]]:
    selected_by_path: dict[str, set[int]] = {}
    for candidate in candidates:
        selected_by_path.setdefault(candidate.source_path, set()).add(int(candidate.source_index))
    return selected_by_path


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


def _scan_hetero_selection_candidates_for_file(
    payload: tuple[
        str,
        float,
        int,
        int,
        bool,
        int,
        float,
        bool,
        int | None,
        str | None,
        bool,
        int | None,
        float,
        float,
        float,
        int,
    ],
) -> dict[str, Any]:
    import dstio

    (
        path,
        bin_width,
        seed,
        cell_cap,
        skip_errors,
        open_retries,
        open_retry_delay,
        stratify_particle,
        min_event_date,
        mc_calib_dir,
        skip_missing_mc_calibration,
        max_events,
        zenith_bin_width_deg,
        azimuth_bin_width_deg,
        core_bin_width_km,
        time_bin_width_sec,
    ) = payload
    mc_calibration = None
    if mc_calib_dir and skip_missing_mc_calibration:
        from .mc_calibration import get_cached_mc_calibration_db

        mc_calibration = get_cached_mc_calibration_db(Path(mc_calib_dir))
    candidates: list[HeteroSelectionCandidate] = []
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
                if max_events is not None and raw_events >= int(max_events):
                    break
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
                candidate = _candidate_from_mc_event(
                    path,
                    source_index,
                    event,
                    bin_width=bin_width,
                    seed=seed,
                    stratify_particle=stratify_particle,
                    zenith_bin_width_deg=zenith_bin_width_deg,
                    azimuth_bin_width_deg=azimuth_bin_width_deg,
                    core_bin_width_km=core_bin_width_km,
                    time_bin_width_sec=time_bin_width_sec,
                )
                if candidate is None:
                    continue
                hit_events += 1
                seen_by_bin[candidate.bin_key] = seen_by_bin.get(candidate.bin_key, 0) + 1
                candidates.append(candidate)
    except Exception as exc:
        if _is_dst_unit_exhaustion(exc):
            _raise_dst_unit_exhaustion(exc)
        if not skip_errors:
            raise
        return {
            "path": path,
            "candidates": [],
            "seen_by_bin": {},
            "raw_events": raw_events,
            "hit_events": hit_events,
            "missing_calibration_events": missing_calibration_events,
            "error": str(exc),
        }
    candidates = _cap_hetero_candidates_by_cell(candidates, cap=cell_cap)
    return {
        "path": path,
        "candidates": candidates,
        "seen_by_bin": seen_by_bin,
        "raw_events": raw_events,
        "hit_events": hit_events,
        "missing_calibration_events": missing_calibration_events,
        "error": None,
    }


def _scan_hetero_source_file_manifest(
    payload: tuple[
        str,
        bool,
        int,
        float,
        int | None,
        str | None,
        bool,
        int | None,
        float,
        float,
        int,
    ],
) -> dict[str, Any]:
    import dstio

    (
        path,
        skip_errors,
        open_retries,
        open_retry_delay,
        min_event_date,
        mc_calib_dir,
        skip_missing_mc_calibration,
        max_events,
        azimuth_bin_width_deg,
        core_bin_width_km,
        time_bin_width_sec,
    ) = payload
    dat_tag = _dat_tag_from_path(path)
    energy_bin_code = _energy_bin_code_from_dat_tag(dat_tag)
    source_group = _source_group_key_for_path(path)
    gea_trg_index = _gea_trg_index_from_path(path)
    particle = _particle_stratum_from_path(path)
    mc_calibration = None
    if mc_calib_dir and skip_missing_mc_calibration:
        from .mc_calibration import get_cached_mc_calibration_db

        mc_calibration = get_cached_mc_calibration_db(Path(mc_calib_dir))
    raw_events = 0
    missing_calibration_events = 0
    eligible_event_count = 0
    source_zenith_deg = float("nan")
    date_counts: dict[str, int] = {}
    cell_counts: dict[tuple[str, ...], int] = {}
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
            for _source_index, event in enumerate(dst):
                if max_events is not None and raw_events >= int(max_events):
                    break
                raw_events += 1
                rusdraw = event.get("rusdraw") or {}
                rusdmc = event.get("rusdmc") or {}
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
                theta = float(rusdmc.get("theta", float("nan")) or float("nan"))
                if not math.isfinite(theta):
                    continue
                if particle == "unknown":
                    particle = _particle_stratum_from_parttype(rusdmc.get("parttype", -1))
                if not math.isfinite(source_zenith_deg):
                    source_zenith_deg = math.degrees(theta)
                eligible_event_count += 1
                date_key = f"{date:06d}"
                date_counts[date_key] = date_counts.get(date_key, 0) + 1
                cell_key = _event_balance_key_from_mc_event(
                    event,
                    azimuth_bin_width_deg=azimuth_bin_width_deg,
                    core_bin_width_km=core_bin_width_km,
                    time_bin_width_sec=time_bin_width_sec,
                )
                cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
    except Exception as exc:
        if _is_dst_unit_exhaustion(exc):
            _raise_dst_unit_exhaustion(exc)
        if not skip_errors:
            raise
        return {
            "path": path,
            "manifest": None,
            "raw_events": raw_events,
            "missing_calibration_events": missing_calibration_events,
            "error": str(exc),
        }
    manifest = HeteroSourceFileManifest(
        path=path,
        source_group=source_group,
        dat_tag=dat_tag,
        energy_bin_code=energy_bin_code,
        particle=particle,
        gea_trg_index=gea_trg_index,
        source_zenith_deg=float(source_zenith_deg),
        eligible_event_count=int(eligible_event_count),
        date_counts=date_counts,
        cell_counts=cell_counts,
    )
    return {
        "path": path,
        "manifest": manifest,
        "raw_events": raw_events,
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


def _selected_entries_from_path_indices(
    inputs: list[str],
    selected_indices_by_path: dict[str, set[int]],
    *,
    stratify_particle: bool = False,
) -> list[SelectedEntry]:
    entries: list[SelectedEntry] = []
    for path in inputs:
        bin_key = _hetero_output_bin_key_from_path(path, stratify_particle=stratify_particle)
        for source_index in sorted(selected_indices_by_path.get(path, ())):
            unique_id = f"{path}:{int(source_index)}"
            entries.append((bin_key, unique_id, path, int(source_index), float("nan"), 0, 0))
    return entries


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


def _write_selected_hetero_graph_shard(
    payload: tuple[
        int,
        list[SelectedEntry],
        str,
        str,
        str,
        str | None,
        str | None,
        bool,
        bool,
        bool,
        bool,
        int | None,
        int,
        float,
        str,
        dict[str, Any],
        int,
    ]
) -> dict[str, Any]:
    import dstio.tale.graph as tale_graph

    from .hetero_graph_io import create_hetero_graph_file, write_hetero_graph

    (
        shard_index,
        entries,
        kind,
        cleaning,
        node_policy,
        const_dst,
        mc_calib_dir,
        require_trigger_mode0,
        require_reference_core,
        skip_errors,
        skip_missing_mc_calibration,
        min_event_date,
        open_retries,
        open_retry_delay,
        output,
        config,
        write_block_size,
    ) = payload

    output_path = _shard_path(output, shard_index)
    shard_config = dict(config)
    shard_config["shard_index"] = shard_index
    handle = None
    written = 0
    skipped = 0
    graph_seen_by_bin: dict[str, int] = {}
    processed_blocks = 0
    processed_files = 0
    selected_total = len(entries)
    interval = max(float(os.environ.get("TALESD_GNN_PROGRESS_INTERVAL", "30")), 1.0)
    last_report = time.perf_counter()

    _progress_write(
        f"hetero export/write shard {shard_index:04d}: start ordered_events={selected_total} "
        f"selected={selected_total} output={output_path.name}"
    )
    try:
        for block in _chunked(entries, max(int(write_block_size), 1)):
            processed_blocks += 1
            wanted_by_path: dict[str, set[int]] = {}
            graph_by_key: dict[tuple[str, int], Any] = {}
            for _bin_key, _unique_id, path, source_index, _log10_energy, _date, _time_value in block:
                wanted_by_path.setdefault(path, set()).add(int(source_index))
            for path, selected in wanted_by_path.items():
                if not selected:
                    continue
                for graph in tale_graph.iter_graphs(
                    path,
                    kind=kind,
                    cleaning=cleaning,
                    node_policy=node_policy,
                    const_dst=const_dst,
                    mc_calib_dir=mc_calib_dir,
                    max_events=None,
                    require_trigger_mode0=require_trigger_mode0,
                    require_reference_core=require_reference_core,
                    skip_errors=skip_errors,
                    skip_missing_mc_calibration=skip_missing_mc_calibration,
                    source_indices=selected,
                    min_event_date=min_event_date,
                    open_retries=open_retries,
                    open_retry_delay=open_retry_delay,
                ):
                    try:
                        graph_source_index = int(graph.metadata.get("source_index", -1))
                    except Exception:
                        graph_source_index = -1
                    if graph.target is None or graph.target.shape[0] == 0:
                        continue
                    graph_by_key[(path, graph_source_index)] = graph
                processed_files += 1

            for _bin_key, _unique_id, path, source_index, _log10_energy, _date, _time_value in block:
                graph = graph_by_key.get((path, int(source_index)))
                if graph is None:
                    skipped += 1
                    continue
                if handle is None:
                    handle = create_hetero_graph_file(output_path, config=shard_config)
                write_hetero_graph(handle, written, graph)
                written += 1
                bin_label = _hetero_bin_key_label(_bin_key)
                graph_seen_by_bin[bin_label] = graph_seen_by_bin.get(bin_label, 0) + 1
            now = time.perf_counter()
            if now - last_report >= interval:
                _progress_write(
                    f"hetero export/write shard {shard_index:04d}: blocks={processed_blocks} "
                    f"files={processed_files} "
                    f"written={written} skipped={skipped}"
                )
                last_report = now
    except BaseException as exc:
        _progress_write(
            f"hetero export/write shard {shard_index:04d}: failed blocks={processed_blocks} files={processed_files} "
            f"written={written} skipped={skipped} error={exc}"
        )
        raise
    finally:
        if handle is not None:
            handle.close()

    _progress_write(
        f"hetero export/write shard {shard_index:04d}: done blocks={processed_blocks} files={processed_files} "
        f"written={written} skipped={skipped} output={output_path.name if written > 0 else '(empty)'}"
    )
    return {
        "shard_index": shard_index,
        "path": str(output_path) if written > 0 else None,
        "written": written,
        "skipped": skipped,
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


def _iter_selected_hetero_shard_write_results(
    inputs: list[str],
    args: argparse.Namespace,
    const_dst: Path | None,
    mc_calib_dir: Path | None,
    selected_indices_by_path: dict[str, set[int]],
    config: dict[str, Any],
    *,
    shard_start_index: int = 0,
) -> Iterator[dict[str, Any]]:
    selected_entries = _selected_entries_from_path_indices(
        inputs,
        selected_indices_by_path,
        stratify_particle=bool(args.energy_sample_stratify_particle),
    )
    selected_entries = _ordered_selected_entries(
        selected_entries,
        output_order=str(args.output_order),
        seed=int(args.seed),
        locality_run_size=int(args.output_locality_run_size),
    )
    chunks = _selected_entry_chunks(selected_entries, int(args.shard_size))
    payloads = [
        (
            int(shard_start_index) + shard_index,
            entries,
            args.kind,
            args.cleaning,
            args.node_policy,
            str(const_dst) if const_dst is not None else None,
            str(mc_calib_dir) if mc_calib_dir is not None else None,
            not args.keep_non_mode0,
            bool(args.require_reference_core),
            bool(args.skip_errors),
            bool(args.skip_missing_mc_calibration),
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            int(args.open_retries),
            float(args.open_retry_delay),
            str(Path(args.output).expanduser()),
            config,
            int(args.write_block_size),
        )
        for shard_index, entries in enumerate(chunks)
    ]
    workers = min(max(int(args.workers), 1), len(payloads)) if payloads else 1
    _progress_write(
        f"hetero export/write shards: start shards={len(payloads)} workers={workers} "
        f"selected_files={len(selected_indices_by_path)} selected_events={len(selected_entries)}"
    )
    yield from _iter_process_pool(
        payloads,
        _write_selected_hetero_graph_shard,
        workers,
        "hetero export/write shards",
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


def _iter_hetero_selection_scan_results(
    inputs: list[str],
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    payloads = [
        (
            path,
            float(args.energy_bin_width),
            int(args.seed),
            int(args.balance_cell_preselect),
            bool(args.skip_errors),
            int(args.open_retries),
            float(args.open_retry_delay),
            bool(args.energy_sample_stratify_particle),
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            str(Path(args.mc_calib_dir).expanduser()) if args.mc_calib_dir else None,
            bool(args.skip_missing_mc_calibration and args.kind == "mc"),
            None if args.max_events is None or int(args.max_events) <= 0 else int(args.max_events),
            float(args.balance_zenith_bin_width_deg),
            float(args.balance_azimuth_bin_width_deg),
            float(args.balance_core_bin_width_km),
            int(args.balance_time_bin_width_sec),
        )
        for path in inputs
    ]
    workers = max(int(args.workers), 1)
    if workers == 1:
        for payload in _progress(payloads, desc="scan hetero candidates", total=len(payloads)):
            yield _scan_hetero_selection_candidates_for_file(payload)
        return

    try:
        yield from _iter_process_pool(
            payloads,
            _scan_hetero_selection_candidates_for_file,
            workers,
            "scan hetero candidates",
            max_tasks_per_child=max(int(args.worker_max_files), 0),
        )
    except (OSError, PermissionError) as exc:
        _progress_write(f"warning: file-parallel hetero scan failed ({exc}); falling back to single-process scan")
        for payload in _progress(payloads, desc="scan hetero candidates", total=len(payloads)):
            yield _scan_hetero_selection_candidates_for_file(payload)


def _iter_hetero_source_file_manifest_results(
    inputs: list[str],
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    payloads = [
        (
            path,
            bool(args.skip_errors),
            int(args.open_retries),
            float(args.open_retry_delay),
            None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date),
            str(Path(args.mc_calib_dir).expanduser()) if args.mc_calib_dir else None,
            bool(args.skip_missing_mc_calibration and args.kind == "mc"),
            None if args.max_events is None or int(args.max_events) <= 0 else int(args.max_events),
            float(args.balance_azimuth_bin_width_deg),
            float(args.balance_core_bin_width_km),
            int(args.balance_time_bin_width_sec),
        )
        for path in inputs
    ]
    workers = max(int(args.workers), 1)
    if workers == 1:
        for payload in _progress(payloads, desc="scan hetero source files", total=len(payloads)):
            yield _scan_hetero_source_file_manifest(payload)
        return

    try:
        yield from _iter_process_pool(
            payloads,
            _scan_hetero_source_file_manifest,
            workers,
            "scan hetero source files",
            max_tasks_per_child=max(int(args.worker_max_files), 0),
        )
    except (OSError, PermissionError) as exc:
        _progress_write(f"warning: file-parallel hetero source scan failed ({exc}); falling back to single-process scan")
        for payload in _progress(payloads, desc="scan hetero source files", total=len(payloads)):
            yield _scan_hetero_source_file_manifest(payload)


def _iter_selected_hetero_graphs(
    inputs: list[str],
    args: argparse.Namespace,
    const_dst: Path | None,
    mc_calib_dir: Path | None,
    selected_indices_by_path: dict[str, set[int]],
) -> Iterator[Any]:
    import dstio.tale.graph as tale_graph

    min_event_date = None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date)
    selected_inputs = [path for path in inputs if selected_indices_by_path.get(path)]
    iterator = _progress(selected_inputs, desc="build selected hetero graphs", total=len(selected_inputs))
    for path in iterator:
        selected_indices = selected_indices_by_path.get(path)
        if not selected_indices:
            continue
        yield from tale_graph.iter_graphs(
            path,
            kind=args.kind,
            cleaning=args.cleaning,
            node_policy=args.node_policy,
            const_dst=const_dst,
            mc_calib_dir=mc_calib_dir,
            max_events=None,
            require_trigger_mode0=not args.keep_non_mode0,
            require_reference_core=bool(args.require_reference_core),
            skip_errors=bool(args.skip_errors),
            skip_missing_mc_calibration=bool(args.skip_missing_mc_calibration),
            source_indices=selected_indices,
            min_event_date=min_event_date,
            open_retries=args.open_retries,
            open_retry_delay=args.open_retry_delay,
        )


def _iter_ordered_selected_hetero_graphs(
    entries: list[SelectedEntry],
    args: argparse.Namespace,
    const_dst: Path | None,
    mc_calib_dir: Path | None,
) -> Iterator[Any]:
    import dstio.tale.graph as tale_graph

    min_event_date = None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date)
    total_blocks = math.ceil(len(entries) / max(int(args.write_block_size), 1)) if entries else 0
    for block in _progress(
        _chunked(entries, max(int(args.write_block_size), 1)),
        desc="build ordered selected hetero graphs",
        total=total_blocks,
    ):
        wanted_by_path: dict[str, set[int]] = {}
        graph_by_key: dict[tuple[str, int], Any] = {}
        for _bin_key, _unique_id, path, source_index, _log10_energy, _date, _time_value in block:
            wanted_by_path.setdefault(path, set()).add(int(source_index))
        for path, selected_indices in wanted_by_path.items():
            for graph in tale_graph.iter_graphs(
                path,
                kind=args.kind,
                cleaning=args.cleaning,
                node_policy=args.node_policy,
                const_dst=const_dst,
                mc_calib_dir=mc_calib_dir,
                max_events=None,
                require_trigger_mode0=not args.keep_non_mode0,
                require_reference_core=bool(args.require_reference_core),
                skip_errors=bool(args.skip_errors),
                skip_missing_mc_calibration=bool(args.skip_missing_mc_calibration),
                source_indices=selected_indices,
                min_event_date=min_event_date,
                open_retries=args.open_retries,
                open_retry_delay=args.open_retry_delay,
            ):
                try:
                    graph_source_index = int(graph.metadata.get("source_index", -1))
                except Exception:
                    graph_source_index = -1
                graph_by_key[(path, graph_source_index)] = graph

        for _bin_key, _unique_id, path, source_index, _log10_energy, _date, _time_value in block:
            graph = graph_by_key.get((path, int(source_index)))
            if graph is not None:
                yield graph


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


def _decode_h5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _h5_metadata_value(handle: Any, local_index: int, name: str) -> Any | None:
    metadata = handle.get("metadata")
    if metadata is None or name not in metadata:
        return None
    dataset = metadata[name]
    if local_index >= dataset.shape[0]:
        return None
    return dataset[local_index]


def _scan_hetero_h5_event_entries(
    payload: tuple[int, str, bool],
) -> dict[str, Any]:
    import h5py

    path_index, h5_path, stratify_particle = payload
    entries: list[HeteroH5EventEntry] = []
    with h5py.File(Path(h5_path).expanduser(), "r") as handle:
        if str(handle.attrs.get("format", "")) != "talesd_gnn_hetero_graphs":
            raise ValueError(f"{h5_path} is not a hetero graph HDF5 file")
        n_events = len(handle["events"])
        for local_index in range(n_events):
            key = f"{local_index:08d}"
            event_group = handle["events"][key]
            event_id_value = _h5_metadata_value(handle, local_index, "event_id")
            source_path_value = _h5_metadata_value(handle, local_index, "source_path")
            source_index_value = _h5_metadata_value(handle, local_index, "source_index")
            event_id = _decode_h5_text(event_id_value) if event_id_value is not None else str(event_group.attrs.get("event_id", key))
            source_path = (
                _decode_h5_text(source_path_value)
                if source_path_value is not None
                else str(event_group.attrs.get("source_path", ""))
            )
            try:
                source_index = int(source_index_value) if source_index_value is not None else int(event_group.attrs.get("source_index", local_index))
            except Exception:
                source_index = int(local_index)
            bin_key = _hetero_output_bin_key_from_path(source_path, stratify_particle=bool(stratify_particle))
            unique_id = f"{event_id}:{source_path}:{source_index}"
            entries.append(
                HeteroH5EventEntry(
                    bin_key=bin_key,
                    unique_id=unique_id,
                    h5_path=str(Path(h5_path).expanduser()),
                    local_index=int(local_index),
                    source_path=source_path,
                    source_index=int(source_index),
                )
            )
    return {"path_index": int(path_index), "entries": entries}


def _hetero_h5_event_entries(paths: list[str], *, stratify_particle: bool, workers: int = 1) -> list[HeteroH5EventEntry]:
    payloads = [(index, path, bool(stratify_particle)) for index, path in enumerate(paths)]
    worker_count = min(max(int(workers), 1), len(payloads)) if payloads else 1
    _progress_write(
        f"scan hetero HDF5 metadata: start shards={len(payloads)} workers={worker_count}"
    )
    results: list[dict[str, Any]] = []
    if worker_count <= 1:
        for payload in _progress(payloads, desc="scan hetero HDF5 metadata", total=len(payloads)):
            results.append(_scan_hetero_h5_event_entries(payload))
    else:
        results.extend(
            _iter_process_pool(
                payloads,
                _scan_hetero_h5_event_entries,
                worker_count,
                "scan hetero HDF5 metadata",
            )
        )
    entries: list[HeteroH5EventEntry] = []
    for result in sorted(results, key=lambda item: int(item["path_index"])):
        entries.extend(result["entries"])
    return entries


def _write_resharded_hetero_h5_shard(
    payload: tuple[int, list[HeteroH5EventEntry], str, dict[str, Any], bool],
) -> dict[str, Any]:
    import h5py

    from .hetero_graph_io import copy_hetero_graph_group, create_hetero_graph_file

    shard_index, entries, output, config, overwrite = payload
    output_path = _shard_path(output, shard_index)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace")

    shard_config = dict(config)
    shard_config["shard_index"] = int(shard_index)
    handle_cache: dict[str, h5py.File] = {}
    written = 0
    try:
        with create_hetero_graph_file(output_path, config=shard_config) as target:
            for output_index, entry in enumerate(entries):
                source = handle_cache.get(entry.h5_path)
                if source is None:
                    source = h5py.File(entry.h5_path, "r")
                    handle_cache[entry.h5_path] = source
                copy_hetero_graph_group(
                    source,
                    f"{entry.local_index:08d}",
                    int(entry.local_index),
                    target,
                    int(output_index),
                )
                written += 1
    finally:
        for source in handle_cache.values():
            source.close()
    return {
        "shard_index": int(shard_index),
        "path": str(output_path),
        "written": int(written),
    }


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


def _merge_counter(target: dict[str, int], source: dict[str, int]) -> None:
    for key, count in source.items():
        target[key] = target.get(key, 0) + int(count)


def _cell_label(cell: tuple[str, ...]) -> str:
    return "|".join(str(item) for item in cell)


def _merge_hetero_source_file_manifests(
    scan_results: Iterable[dict[str, Any]],
) -> tuple[dict[str, HeteroSourceGroupManifest], dict[str, Any]]:
    files_by_group: dict[str, list[HeteroSourceFileManifest]] = {}
    raw_events = 0
    missing_calibration_events = 0
    unreadable_files = 0
    skipped_empty_files = 0
    for result in scan_results:
        raw_events += int(result.get("raw_events", 0))
        missing_calibration_events += int(result.get("missing_calibration_events", 0))
        if result.get("error"):
            unreadable_files += 1
            _progress_write(f"warning: skipping unreadable DST {result['path']}: {result['error']}")
            continue
        manifest = result.get("manifest")
        if manifest is None or int(manifest.eligible_event_count) <= 0:
            skipped_empty_files += 1
            continue
        files_by_group.setdefault(manifest.source_group, []).append(manifest)

    groups: dict[str, HeteroSourceGroupManifest] = {}
    for source_group, files in files_by_group.items():
        files = sorted(files, key=lambda item: (item.gea_trg_index, item.path))
        dat_tags = {item.dat_tag for item in files}
        energy_codes = {item.energy_bin_code for item in files}
        particles = {item.particle for item in files}
        if len(dat_tags) != 1:
            raise ValueError(f"source group {source_group} mixes DAT tags: {sorted(dat_tags)}")
        if len(energy_codes) != 1:
            raise ValueError(f"source group {source_group} mixes filename energy codes: {sorted(energy_codes)}")
        known_particles = {particle for particle in particles if particle != "unknown"}
        if len(known_particles) > 1:
            raise ValueError(f"source group {source_group} mixes particles: {sorted(known_particles)}")
        zenith_values = [
            float(item.source_zenith_deg)
            for item in files
            if math.isfinite(float(item.source_zenith_deg))
        ]
        if not zenith_values:
            continue
        if max(zenith_values) - min(zenith_values) > 1.0e-6:
            raise ValueError(
                f"source group {source_group} has inconsistent source zenith values: "
                f"{min(zenith_values):.9f}..{max(zenith_values):.9f}"
            )
        date_counts: dict[str, int] = {}
        cell_counts: dict[tuple[str, ...], int] = {}
        eligible_event_count = 0
        for item in files:
            eligible_event_count += int(item.eligible_event_count)
            _merge_counter(date_counts, item.date_counts)
            for cell, count in item.cell_counts.items():
                cell_counts[cell] = cell_counts.get(cell, 0) + int(count)
        groups[source_group] = HeteroSourceGroupManifest(
            source_group=source_group,
            dat_tag=next(iter(dat_tags)),
            energy_bin_code=next(iter(energy_codes)),
            particle=next(iter(known_particles)) if known_particles else "unknown",
            source_zenith_deg=float(zenith_values[0]),
            eligible_event_count=int(eligible_event_count),
            files=tuple(files),
            date_counts=date_counts,
            cell_counts=cell_counts,
        )

    by_bin: dict[str, dict[str, int]] = {}
    for group in groups.values():
        for stratify_particle in (False, True):
            key = _source_group_bin_key(group, stratify_particle=stratify_particle)
            bucket = by_bin.setdefault(
                ("particle:" if stratify_particle else "energy:") + key,
                {"source_groups": 0, "eligible_events": 0},
            )
            bucket["source_groups"] += 1
            bucket["eligible_events"] += int(group.eligible_event_count)
    summary = {
        "input_files": sum(len(files) for files in files_by_group.values()) + unreadable_files + skipped_empty_files,
        "unreadable_files": unreadable_files,
        "skipped_empty_files": skipped_empty_files,
        "raw_events": raw_events,
        "missing_calibration_events": missing_calibration_events,
        "source_groups": len(groups),
        "source_files": sum(len(group.files) for group in groups.values()),
        "eligible_events": sum(int(group.eligible_event_count) for group in groups.values()),
        "by_bin": by_bin,
    }
    return groups, summary


def _allocate_hetero_source_group_quotas(
    groups: dict[str, HeteroSourceGroupManifest],
    *,
    per_bin: int,
    seed: int,
    stratify_particle: bool,
    bin_targets: dict[str, int] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    groups_by_bin: dict[str, list[HeteroSourceGroupManifest]] = {}
    for group in groups.values():
        if int(group.eligible_event_count) <= 0:
            continue
        groups_by_bin.setdefault(
            _source_group_bin_key(group, stratify_particle=stratify_particle),
            [],
        ).append(group)

    quotas: dict[str, int] = {}
    by_bin: dict[str, Any] = {}
    default_target = max(int(per_bin), 1)
    for bin_key, bin_groups in sorted(groups_by_bin.items()):
        target = default_target
        if bin_targets is not None:
            target = max(int(bin_targets.get(str(bin_key), 0)), 0)
            if target <= 0:
                continue
        ordered = sorted(bin_groups, key=lambda item: item.source_group)
        base = target // max(len(ordered), 1)
        remainder = target % max(len(ordered), 1)
        remainder_groups = {
            item.source_group
            for item in sorted(
                ordered,
                key=lambda item: _sample_key_from_parts(int(seed), item.source_group, "quota-remainder", 0),
            )[:remainder]
        }
        assigned = 0
        short_by_availability = 0
        for group in ordered:
            requested = base + (1 if group.source_group in remainder_groups else 0)
            quota = min(int(group.eligible_event_count), int(requested))
            quotas[group.source_group] = quota
            assigned += quota
            short_by_availability += int(requested) - quota
        by_bin[bin_key] = {
            "target_events": target,
            "source_groups": len(ordered),
            "eligible_events": sum(int(group.eligible_event_count) for group in ordered),
            "base_quota_per_source_group": base,
            "remainder_source_groups": remainder,
            "assigned_events": assigned,
            "short_by_availability": short_by_availability,
        }
    return quotas, {"by_bin": by_bin, "events": sum(quotas.values()), "source_groups": len(quotas)}


def _allocate_cell_quotas(
    cell_counts: dict[tuple[str, ...], int],
    *,
    target: int,
    seed: int,
    source_group: str,
) -> dict[tuple[str, ...], int]:
    target = max(int(target), 0)
    items = [(cell, int(count)) for cell, count in cell_counts.items() if int(count) > 0]
    total = sum(count for _cell, count in items)
    if target <= 0 or total <= 0:
        return {}
    target = min(target, total)
    quotas: dict[tuple[str, ...], int] = {}
    fractions: list[tuple[float, float, tuple[str, ...]]] = []
    assigned = 0
    for cell, count in items:
        exact = float(target) * float(count) / float(total)
        quota = min(int(math.floor(exact)), count)
        quotas[cell] = quota
        assigned += quota
        if quota < count:
            fractions.append(
                (
                    exact - math.floor(exact),
                    _sample_key_from_parts(seed, source_group, _cell_label(cell), 0),
                    cell,
                )
            )
    remaining = target - assigned
    fractions.sort(key=lambda item: (-item[0], item[1], _cell_label(item[2])))
    while remaining > 0 and fractions:
        next_fractions: list[tuple[float, float, tuple[str, ...]]] = []
        for fraction, key, cell in fractions:
            if remaining <= 0:
                next_fractions.append((fraction, key, cell))
                continue
            count = int(cell_counts[cell])
            if quotas[cell] < count:
                quotas[cell] += 1
                remaining -= 1
            if quotas[cell] < count:
                next_fractions.append((fraction, key, cell))
        if len(next_fractions) == len(fractions) and remaining > 0:
            break
        fractions = next_fractions
    return {cell: quota for cell, quota in quotas.items() if quota > 0}


def _select_hetero_events_for_source_group(
    group: HeteroSourceGroupManifest,
    *,
    quota: int,
    args: argparse.Namespace,
    excluded_by_path: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    import dstio

    quota = max(int(quota), 0)
    cell_quotas = _allocate_cell_quotas(
        group.cell_counts,
        target=quota,
        seed=int(args.seed),
        source_group=group.source_group,
    )
    selected_heaps: dict[tuple[str, ...], list[tuple[float, str, str, int, str]]] = {
        cell: [] for cell in cell_quotas
    }
    if quota <= 0 or not cell_quotas:
        return {
            "source_group": group.source_group,
            "selected_by_path": {},
            "selected_event_dates": {},
            "selected_cell_counts": {},
            "selected_events": 0,
        }
    min_event_date = None if args.min_event_date is None or int(args.min_event_date) <= 0 else int(args.min_event_date)
    mc_calibration = None
    if args.mc_calib_dir and bool(args.skip_missing_mc_calibration and args.kind == "mc"):
        from .mc_calibration import get_cached_mc_calibration_db

        mc_calibration = get_cached_mc_calibration_db(Path(args.mc_calib_dir).expanduser())
    for file_info in group.files:
        dst_handle = None
        last_exc: Exception | None = None
        for attempt in range(max(int(args.open_retries), 1)):
            try:
                dst_handle = dstio.open(file_info.path, banks=["rusdraw", "rusdmc"])
                break
            except Exception as exc:
                if _is_dst_unit_exhaustion(exc):
                    _raise_dst_unit_exhaustion(exc)
                last_exc = exc
                if attempt + 1 < max(int(args.open_retries), 1):
                    time.sleep(max(float(args.open_retry_delay), 0.0) * (attempt + 1))
        if dst_handle is None:
            if last_exc is not None:
                if args.skip_errors:
                    _progress_write(f"warning: skipping unreadable DST {file_info.path}: {last_exc}")
                    continue
                raise last_exc
            raise OSError(f"failed to open DST: {file_info.path}")
        with dst_handle as dst:
            raw_events = 0
            excluded = excluded_by_path.get(file_info.path, set()) if excluded_by_path is not None else set()
            for source_index, event in enumerate(dst):
                if args.max_events is not None and raw_events >= int(args.max_events):
                    break
                raw_events += 1
                if int(source_index) in excluded:
                    continue
                rusdraw = event.get("rusdraw") or {}
                rusdmc = event.get("rusdmc") or {}
                date = int(rusdraw.get("yymmdd", 0) or 0)
                if min_event_date is not None and (date <= 0 or date < int(min_event_date)):
                    continue
                time_value = int(rusdraw.get("hhmmss", 0) or 0)
                if mc_calibration is not None and not mc_calibration.has_calibration_time(date, time_value):
                    continue
                xxyy = rusdraw.get("xxyy", [])
                if len(xxyy) <= 0:
                    continue
                theta = float(rusdmc.get("theta", float("nan")) or float("nan"))
                if not math.isfinite(theta):
                    continue
                cell = _event_balance_key_from_mc_event(
                    event,
                    azimuth_bin_width_deg=float(args.balance_azimuth_bin_width_deg),
                    core_bin_width_km=float(args.balance_core_bin_width_km),
                    time_bin_width_sec=int(args.balance_time_bin_width_sec),
                )
                cell_quota = int(cell_quotas.get(cell, 0))
                if cell_quota <= 0:
                    continue
                event_id = _candidate_event_id(file_info.path, source_index, event)
                score = _sample_key_from_parts(int(args.seed), event_id, file_info.path, int(source_index))
                entry = (-score, event_id, file_info.path, int(source_index), f"{date:06d}")
                bucket = selected_heaps[cell]
                if len(bucket) < cell_quota:
                    heapq.heappush(bucket, entry)
                elif entry[0] > bucket[0][0]:
                    heapq.heapreplace(bucket, entry)

    selected_by_path: dict[str, set[int]] = {}
    selected_event_dates: dict[str, int] = {}
    selected_cell_counts: dict[str, int] = {}
    selected_events = 0
    for cell, entries in selected_heaps.items():
        selected_cell_counts[_cell_label(cell)] = len(entries)
        selected_events += len(entries)
        for _neg_score, _event_id, path, source_index, date_key in entries:
            selected_by_path.setdefault(path, set()).add(int(source_index))
            selected_event_dates[date_key] = selected_event_dates.get(date_key, 0) + 1
    return {
        "source_group": group.source_group,
        "selected_by_path": selected_by_path,
        "selected_event_dates": selected_event_dates,
        "selected_cell_counts": selected_cell_counts,
        "selected_events": selected_events,
    }


def _build_source_group_balanced_hetero_selection(
    inputs: list[str],
    args: argparse.Namespace,
    *,
    bin_targets: dict[str, int] | None = None,
    excluded_by_path: dict[str, set[int]] | None = None,
    seed_offset: int = 0,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    groups, scan_summary = _merge_hetero_source_file_manifests(
        _iter_hetero_source_file_manifest_results(inputs, args)
    )
    quotas, quota_summary = _allocate_hetero_source_group_quotas(
        groups,
        per_bin=max(int(args.energy_sample_per_bin), 1),
        seed=int(args.seed) + int(seed_offset),
        stratify_particle=bool(args.energy_sample_stratify_particle),
        bin_targets=bin_targets,
    )
    selected_by_path: dict[str, set[int]] = {}
    selected_event_dates: dict[str, int] = {}
    selected_by_bin: dict[str, int] = {}
    selected_by_particle: dict[str, int] = {}
    selected_cell_counts: dict[str, int] = {}
    selected_source_groups: set[str] = set()
    iterator = _progress(
        sorted(groups.values(), key=lambda item: item.source_group),
        desc="select hetero event indices",
        total=len(groups),
    )
    for group in iterator:
        result = _select_hetero_events_for_source_group(
            group,
            quota=int(quotas.get(group.source_group, 0)),
            args=args,
            excluded_by_path=excluded_by_path,
        )
        for path, indices in result["selected_by_path"].items():
            selected_by_path.setdefault(path, set()).update(indices)
        _merge_counter(selected_event_dates, result["selected_event_dates"])
        _merge_counter(selected_cell_counts, result["selected_cell_counts"])
        selected_count = int(result["selected_events"])
        if selected_count > 0:
            selected_source_groups.add(group.source_group)
        bin_key = _source_group_bin_key(group, stratify_particle=bool(args.energy_sample_stratify_particle))
        selected_by_bin[bin_key] = selected_by_bin.get(bin_key, 0) + selected_count
        selected_by_particle[group.particle] = selected_by_particle.get(group.particle, 0) + selected_count

    selected_events = sum(len(indices) for indices in selected_by_path.values())
    source_counts_by_bin: dict[str, int] = {}
    for group in groups.values():
        key = _source_group_bin_key(group, stratify_particle=bool(args.energy_sample_stratify_particle))
        source_counts_by_bin[key] = source_counts_by_bin.get(key, 0) + 1
    selection_summary = {
        "config": {
            "selection_strategy": "source_group_manifest_filename_energy_v1",
            "energy_sample_per_bin": int(args.energy_sample_per_bin),
            "energy_bin_source": "DAT tag suffix",
            "energy_sample_stratify_particle": bool(args.energy_sample_stratify_particle),
            "balance_azimuth_bin_width_deg": float(args.balance_azimuth_bin_width_deg),
            "balance_core_bin_width_km": float(args.balance_core_bin_width_km),
            "balance_time_bin_width_sec": int(args.balance_time_bin_width_sec),
            "seed": int(args.seed),
            "seed_offset": int(seed_offset),
            "bin_targets": bin_targets,
            "excluded_events": sum(len(indices) for indices in (excluded_by_path or {}).values()),
        },
        "scan": scan_summary,
        "quota": quota_summary,
        "source_groups_by_bin": dict(sorted(source_counts_by_bin.items())),
        "selected": {
            "events": selected_events,
            "files": len(selected_by_path),
            "source_groups": len(selected_source_groups),
            "by_filename_energy_bin": dict(sorted(selected_by_bin.items())),
            "by_particle": dict(sorted(selected_by_particle.items())),
            "by_date": dict(sorted(selected_event_dates.items())),
            "by_core_azimuth_time_cell": dict(sorted(selected_cell_counts.items())),
        },
    }
    return selected_by_path, selection_summary


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
        "energy_sample_per_bin": args.energy_sample_per_bin,
        "energy_sample_stratify_particle": bool(args.energy_sample_stratify_particle),
        "energy_bin_width": args.energy_bin_width,
        "balanced_selection": bool(args.energy_sample_per_bin is not None),
        "balance_cell_preselect": args.balance_cell_preselect,
        "balance_zenith_bin_width_deg": args.balance_zenith_bin_width_deg,
        "balance_azimuth_bin_width_deg": args.balance_azimuth_bin_width_deg,
        "balance_core_bin_width_km": args.balance_core_bin_width_km,
        "balance_time_bin_width_sec": args.balance_time_bin_width_sec,
        "seed": args.seed,
        "output_order": args.output_order,
        "output_locality_run_size": args.output_locality_run_size,
        "write_block_size": args.write_block_size,
        "workers": args.workers,
        "worker_max_files": args.worker_max_files,
        "refill_attempts": args.refill_attempts,
        "refill_safety_factor": args.refill_safety_factor,
        "refill_min_efficiency": args.refill_min_efficiency,
        "shard_size": args.shard_size,
        "open_retries": args.open_retries,
        "open_retry_delay": args.open_retry_delay,
        "min_event_date": min_event_date,
        "skip_missing_mc_calibration": bool(args.skip_missing_mc_calibration),
    }
    if args.energy_sample_per_bin is not None:
        if args.kind not in {"mc", "auto"}:
            raise ValueError("balanced energy sampling for export-hetero currently requires MC rusdraw/rusdmc input")
        selected_by_path, selection_summary = _build_source_group_balanced_hetero_selection(inputs, args)
        desired_by_bin = {
            str(bin_key): int(row.get("target_events", int(args.energy_sample_per_bin)))
            for bin_key, row in selection_summary["quota"]["by_bin"].items()
        }
        selected_event_dates = selection_summary["selected"]["by_date"]
        if args.kind == "mc" and mc_calib_dir is not None:
            _validate_mc_calibration_dates(
                mc_calib_dir,
                {int(date): count for date, count in selected_event_dates.items()},
                context="hetero balanced preselection",
            )
        scan_summary = selection_summary["scan"]
        config["balanced_selection_summary"] = selection_summary
        config["scan_raw_events"] = int(scan_summary.get("raw_events", 0))
        config["scan_hit_events"] = int(scan_summary.get("eligible_events", 0))
        config["scan_missing_calibration_events"] = int(scan_summary.get("missing_calibration_events", 0))
        config["scan_selected_files"] = len(selected_by_path)
        selected_event_count = sum(len(indices) for indices in selected_by_path.values())
        _progress_write(
            "hetero balanced preselection: "
            f"selected_events={selected_event_count} selected_files={len(selected_by_path)} "
            f"source_groups={selection_summary['selected']['source_groups']} "
            f"raw_events={config['scan_raw_events']} eligible_events={config['scan_hit_events']} "
            f"missing_calibration_events={config['scan_missing_calibration_events']}"
        )
        if args.dry_run_selection:
            if args.selection_summary:
                summary_path = Path(args.selection_summary).expanduser()
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(selection_summary, indent=2, sort_keys=True) + "\n")
                _progress_write(f"hetero selection summary: {summary_path}")
            print("dry-run selection complete; no hetero HDF5 was written")
            return
        if int(args.shard_size) > 0 and int(args.workers) > 1:
            config["balanced_selection_parallel_shard_write"] = True
            config["balanced_selection_output_ordered"] = True
            config["balanced_refill_enabled"] = True
            written_total = 0
            skipped_total = 0
            written_path_items: list[tuple[int, Path]] = []
            written_by_bin: dict[str, int] = {}
            selected_by_bin_total: dict[str, int] = {}
            selected_dates_total: dict[str, int] = {}
            attempted_by_path: dict[str, set[int]] = {}
            refill_history: list[dict[str, Any]] = []
            selection_attempts: list[dict[str, Any]] = [selection_summary]
            current_selected_by_path = selected_by_path
            current_selection_summary = selection_summary
            current_bin_targets: dict[str, int] | None = None
            shard_start_index = 0

            for attempt_index in range(max(int(args.refill_attempts), 0) + 1):
                if attempt_index > 0:
                    current_selection_summary = selection_attempts[-1]
                _merge_selected_indices(attempted_by_path, current_selected_by_path)
                _merge_count_dict(selected_by_bin_total, current_selection_summary["selected"]["by_filename_energy_bin"])
                _merge_count_dict(selected_dates_total, current_selection_summary["selected"]["by_date"])

                attempt_config = dict(config)
                attempt_config["balanced_selection_attempt"] = int(attempt_index)
                attempt_config["balanced_selection_summary"] = current_selection_summary
                attempt_config["balanced_refill_bin_targets"] = current_bin_targets
                attempt_written_by_bin: dict[str, int] = {}
                attempt_written_total = 0
                attempt_skipped_total = 0
                last_shard_index = shard_start_index - 1
                for result in _iter_selected_hetero_shard_write_results(
                    inputs,
                    args,
                    const_dst,
                    mc_calib_dir,
                    current_selected_by_path,
                    attempt_config,
                    shard_start_index=shard_start_index,
                ):
                    result_written = int(result["written"])
                    result_skipped = int(result["skipped"])
                    attempt_written_total += result_written
                    attempt_skipped_total += result_skipped
                    written_total += result_written
                    skipped_total += result_skipped
                    last_shard_index = max(last_shard_index, int(result["shard_index"]))
                    _merge_count_dict(attempt_written_by_bin, result.get("graph_seen_by_bin", {}))
                    if result.get("path"):
                        written_path_items.append((int(result["shard_index"]), Path(str(result["path"]))))
                shard_start_index = max(shard_start_index, last_shard_index + 1)
                _merge_count_dict(written_by_bin, attempt_written_by_bin)
                missing_by_bin = _hetero_missing_bin_counts(desired_by_bin, written_by_bin)
                refill_history.append(
                    {
                        "attempt": int(attempt_index),
                        "bin_targets": current_bin_targets,
                        "selected_events": sum(len(indices) for indices in current_selected_by_path.values()),
                        "selected_by_bin": current_selection_summary["selected"]["by_filename_energy_bin"],
                        "written_events": attempt_written_total,
                        "skipped_events": attempt_skipped_total,
                        "written_by_bin": dict(sorted(attempt_written_by_bin.items())),
                        "cumulative_written_by_bin": dict(sorted(written_by_bin.items())),
                        "missing_by_bin": dict(sorted(missing_by_bin.items())),
                    }
                )
                _progress_write(
                    "hetero balanced write attempt "
                    f"{attempt_index}: written={attempt_written_total} skipped={attempt_skipped_total} "
                    f"missing_bins={len(missing_by_bin)}"
                )
                if not missing_by_bin:
                    break
                if attempt_index >= max(int(args.refill_attempts), 0):
                    break
                current_bin_targets = _hetero_refill_bin_targets(
                    desired_by_bin,
                    written_by_bin,
                    selected_by_bin_total,
                    safety_factor=float(args.refill_safety_factor),
                    min_efficiency=float(args.refill_min_efficiency),
                )
                _progress_write(
                    "hetero balanced refill selection: "
                    f"attempt={attempt_index + 1} bins={len(current_bin_targets)} "
                    f"targets={dict(sorted(current_bin_targets.items()))}"
                )
                current_selected_by_path, next_selection_summary = _build_source_group_balanced_hetero_selection(
                    inputs,
                    args,
                    bin_targets=current_bin_targets,
                    excluded_by_path=attempted_by_path,
                    seed_offset=attempt_index + 1,
                )
                if args.kind == "mc" and mc_calib_dir is not None:
                    _validate_mc_calibration_dates(
                        mc_calib_dir,
                        {int(date): count for date, count in next_selection_summary["selected"]["by_date"].items()},
                        context=f"hetero balanced refill attempt {attempt_index + 1}",
                    )
                selection_attempts.append(next_selection_summary)
                if sum(len(indices) for indices in current_selected_by_path.values()) <= 0:
                    _progress_write("hetero balanced refill selection returned no new events")
                    break

            final_missing_by_bin = _hetero_missing_bin_counts(desired_by_bin, written_by_bin)
            final_selection_summary = dict(selection_summary)
            final_selection_summary["combined_selected"] = {
                "events": sum(len(indices) for indices in attempted_by_path.values()),
                "files": len(attempted_by_path),
                "by_filename_energy_bin": dict(sorted(selected_by_bin_total.items())),
                "by_date": dict(sorted(selected_dates_total.items())),
            }
            final_selection_summary["write"] = {
                "desired_by_bin": dict(sorted(desired_by_bin.items())),
                "written_by_bin": dict(sorted(written_by_bin.items())),
                "missing_by_bin": dict(sorted(final_missing_by_bin.items())),
                "written_events": int(written_total),
                "skipped_events": int(skipped_total),
                "refill_history": refill_history,
            }
            final_selection_summary["selection_attempts"] = selection_attempts
            config["balanced_selection_summary"] = final_selection_summary
            config["balanced_graph_seen_by_bin"] = dict(sorted(written_by_bin.items()))
            config["balanced_missing_by_bin"] = dict(sorted(final_missing_by_bin.items()))
            if args.selection_summary:
                summary_path = Path(args.selection_summary).expanduser()
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(final_selection_summary, indent=2, sort_keys=True) + "\n")
                _progress_write(f"hetero selection summary: {summary_path}")
            if final_missing_by_bin:
                raise SystemExit(
                    "hetero balanced export is short after refill attempts: "
                    + json.dumps(dict(sorted(final_missing_by_bin.items())), sort_keys=True)
                )
            written_paths = [path for _index, path in sorted(written_path_items, key=lambda item: item[0])]
            targets = ", ".join(str(path) for path in written_paths) if written_paths else str(args.output)
            print(f"wrote {written_total} hetero graphs to {targets} (skipped {skipped_total} selected events)")
            return
        if max(int(args.refill_attempts), 0) > 0:
            raise SystemExit(
                "hetero balanced refill requires --shard-size > 0 and --workers > 1; "
                "single-file export cannot verify written graph counts by bin before success"
            )
        selected_entries = _selected_entries_from_path_indices(
            inputs,
            selected_by_path,
            stratify_particle=bool(args.energy_sample_stratify_particle),
        )
        selected_entries = _ordered_selected_entries(
            selected_entries,
            output_order=str(args.output_order),
            seed=int(args.seed),
            locality_run_size=int(args.output_locality_run_size),
        )
        config["balanced_selection_output_ordered"] = True
        graphs = _iter_ordered_selected_hetero_graphs(selected_entries, args, const_dst, mc_calib_dir)
    else:
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


def _cmd_reshard_hetero(args: argparse.Namespace) -> None:
    import h5py

    from .hetero_graph_io import copy_hetero_graph_group, create_hetero_graph_file

    paths = _resolve_graph_args(args.graphs, args.graphs_list)
    output = Path(args.output).expanduser()
    input_paths = {Path(path).expanduser().resolve() for path in paths}
    if output.resolve() in input_paths:
        raise SystemExit("output must not overwrite an input HDF5 file")

    entries = _hetero_h5_event_entries(
        paths,
        stratify_particle=bool(args.energy_sample_stratify_particle),
        workers=int(args.workers),
    )
    downsample_summary: dict[str, Any] | None = None
    if args.energy_sample_per_bin is not None:
        entries, downsample_summary = _sample_hetero_h5_entries_by_bin(
            entries,
            per_bin=max(int(args.energy_sample_per_bin), 1),
            seed=int(args.seed),
        )
        _progress_write(
            "reshard hetero HDF5 downsample: "
            f"input_events={downsample_summary['input_events']} "
            f"selected_events={downsample_summary['selected_events']} "
            f"per_bin={downsample_summary['energy_sample_per_bin']}"
        )
    ordered_entries = _ordered_hetero_h5_entries(
        entries,
        output_order=str(args.output_order),
        seed=int(args.seed),
        locality_run_size=int(args.output_locality_run_size),
    )
    shard_size = max(int(args.shard_size), 0)
    if shard_size > 0:
        chunks = _chunked(ordered_entries, shard_size)
        total_chunks = math.ceil(len(ordered_entries) / shard_size)
    else:
        chunks = iter([ordered_entries])
        total_chunks = 1 if ordered_entries else 0

    config = {
        "operation": "reshard_hetero",
        "input_count": len(paths),
        "input": paths[:MAX_CONFIG_PATHS],
        "input_truncated": len(paths) > MAX_CONFIG_PATHS,
        "output_order": args.output_order,
        "output_locality_run_size": args.output_locality_run_size,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "workers": args.workers,
        "energy_sample_stratify_particle": bool(args.energy_sample_stratify_particle),
        "energy_sample_per_bin": args.energy_sample_per_bin,
        "downsample_summary": downsample_summary,
    }
    written_paths: list[Path] = []
    written_total = 0

    if shard_size > 0:
        payloads = [
            (shard_index, chunk, str(output), config, bool(args.overwrite))
            for shard_index, chunk in enumerate(chunks)
        ]
        workers = min(max(int(args.workers), 1), len(payloads)) if payloads else 1
        _progress_write(
            f"reshard hetero HDF5: start shards={len(payloads)} workers={workers} "
            f"events={len(ordered_entries)}"
        )
        written_items: list[tuple[int, Path]] = []
        for result in _iter_process_pool(
            payloads,
            _write_resharded_hetero_h5_shard,
            workers,
            "reshard hetero HDF5",
        ):
            written_total += int(result["written"])
            written_items.append((int(result["shard_index"]), Path(str(result["path"]))))
        written_paths = [path for _index, path in sorted(written_items, key=lambda item: item[0])]
    else:
        for shard_index, chunk in enumerate(_progress(chunks, desc="reshard hetero HDF5", total=total_chunks)):
            output_path = output
            if output_path.exists() and not args.overwrite:
                raise SystemExit(f"output already exists: {output_path}; pass --overwrite to replace")
            shard_config = dict(config)
            shard_config["shard_index"] = None
            handle_cache: dict[str, h5py.File] = {}
            try:
                with create_hetero_graph_file(output_path, config=shard_config) as target:
                    for output_index, entry in enumerate(chunk):
                        source = handle_cache.get(entry.h5_path)
                        if source is None:
                            source = h5py.File(entry.h5_path, "r")
                            handle_cache[entry.h5_path] = source
                        copy_hetero_graph_group(
                            source,
                            f"{entry.local_index:08d}",
                            int(entry.local_index),
                            target,
                            int(output_index),
                        )
                        written_total += 1
            finally:
                for source in handle_cache.values():
                    source.close()
            written_paths.append(output_path)

    targets = ", ".join(str(path) for path in written_paths) if written_paths else str(output)
    print(f"wrote {written_total} reshuffled hetero graphs to {targets}")


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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        model_architecture=args.model_architecture,
        attention_heads=args.attention_heads,
        readout_heads=args.readout_heads,
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
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=None if not args.no_pin_memory else False,
        loader_memory_budget_gib=args.loader_memory_budget_gib,
        loader_memory_estimate_samples=args.loader_memory_estimate_samples,
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
    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    paths = _expand_h5_graph_paths(graphs)
    graph_format = ""
    if paths:
        import h5py

        with h5py.File(paths[0], "r") as handle:
            graph_format = str(handle.attrs.get("format", ""))
    if graph_format == "talesd_gnn_hetero_graphs":
        from .hetero_feature_analysis import save_hetero_input_distributions

        save = save_hetero_input_distributions
    else:
        from .feature_analysis import save_input_distributions

        save = save_input_distributions
    summary = save(
        graphs,
        args.output,
        max_graphs=args.max_graphs,
        max_values_per_feature=args.max_values_per_feature,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(f"input feature summary: {summary['summary_json']}")


def _cmd_feature_importance(args: argparse.Namespace) -> None:
    import torch

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=False)
    model_config = dict(checkpoint.get("model_config", {}))
    runtime = dict(checkpoint.get("runtime", {}))
    if runtime.get("graph_format") == "hetero" or model_config.get("architecture") in {
        "minimal_hetero",
        "hetero_attention",
    }:
        from .hetero_feature_analysis import save_hetero_feature_group_importance

        save = save_hetero_feature_group_importance
    else:
        from .feature_analysis import save_feature_group_importance

        save = save_feature_group_importance
    summary = save(
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


def _parse_index_list(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    indices: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [part.strip() for part in token.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError(f"invalid index range: {token!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(token))
    return indices


def _cmd_attention_maps(args: argparse.Namespace) -> None:
    from .hetero_attention_analysis import save_hetero_attention_maps

    graphs = _resolve_graph_args(args.graphs, args.graphs_list)
    summary = save_hetero_attention_maps(
        graphs,
        args.checkpoint,
        args.output,
        split=args.split,
        max_graphs=args.max_graphs,
        indices=_parse_index_list(args.indices),
        device=args.device,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    print(f"attention maps: {summary['summary_json']}")
    print(f"attention arrays: {summary['array_file']}")


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
    export_hetero.add_argument("--energy-sample-per-bin", type=int, default=None, help="DAT filename energy codeごとに残す最大hetero graph数。source group均等割りを使う")
    export_hetero.add_argument("--energy-sample-stratify-particle", action="store_true", help="energy samplingをproton/iron別のDAT filename energy codeで行う")
    export_hetero.add_argument("--energy-bin-width", type=float, default=0.1, help="旧event-level sampling用。source-group manifest方式ではselectionに使わない")
    export_hetero.add_argument("--refill-attempts", type=int, default=2, help="balanced exportで実際に書けたgraph数が不足したbinを追加選択する回数")
    export_hetero.add_argument("--refill-safety-factor", type=float, default=1.25, help="不足binの追加選択数に掛ける安全係数")
    export_hetero.add_argument("--refill-min-efficiency", type=float, default=0.01, help="追加選択数を見積もる時の最小 graph 化効率")
    export_hetero.add_argument("--seed", type=int, default=12345, help="balanced selectionのdeterministic seed")
    export_hetero.add_argument("--workers", type=int, default=1, help="DST metadata scanに使うfile worker数")
    export_hetero.add_argument("--worker-max-files", type=int, default=DEFAULT_WORKER_MAX_FILES, help="scan workerをNファイル処理ごとに再起動する。0なら無効")
    export_hetero.add_argument("--balance-cell-preselect", type=int, default=8, help="旧event-level candidate scan用。source-group manifest方式ではselectionに使わない")
    export_hetero.add_argument("--balance-zenith-bin-width-deg", type=float, default=10.0, help="旧event-level selection用。source-group manifest方式ではselectionに使わない")
    export_hetero.add_argument("--balance-azimuth-bin-width-deg", type=float, default=30.0, help="balanced selection用azimuth bin幅")
    export_hetero.add_argument("--balance-core-bin-width-km", type=float, default=0.5, help="balanced selection用core x/y bin幅")
    export_hetero.add_argument("--balance-time-bin-width-sec", type=int, default=3600, help="balanced selection用時刻bin幅")
    export_hetero.add_argument("--selection-summary", default=None, help="balanced preselection summary JSONの出力先")
    export_hetero.add_argument("--dry-run-selection", action="store_true", help="balanced preselection summaryだけ作成し、HDF5は書かない")
    export_hetero.add_argument(
        "--output-order",
        choices=["source", "random", "interleaved"],
        default="interleaved",
        help="balanced export のHDF5内event順。interleavedはparticle/energy/sourceが連続しないように混ぜる",
    )
    export_hetero.add_argument(
        "--output-locality-run-size",
        type=int,
        default=32,
        help="--output-order=interleavedで同一source/binから連続して書く最大event数",
    )
    export_hetero.add_argument(
        "--write-block-size",
        type=int,
        default=2048,
        help="ordered hetero shard書き出しで一度にgraph化して並べ替えるevent数",
    )
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

    reshard_hetero = sub.add_parser("reshard-hetero", help="既存hetero HDF5をDST再読込なしで並べ替え・再shard化する")
    reshard_hetero.add_argument("--graphs", nargs="*", default=[], help="入力hetero HDF5。shard、shard base、またはHDF5ディレクトリを指定可")
    reshard_hetero.add_argument("--graphs-list", action="append", default=[], help="入力hetero HDF5 shard path list")
    reshard_hetero.add_argument("-o", "--output", required=True, help="出力hetero HDF5 base path")
    reshard_hetero.add_argument(
        "--output-order",
        choices=["source", "random", "interleaved"],
        default="interleaved",
        help="出力HDF5内event順。interleavedはparticle/energy/sourceが連続しないように混ぜる",
    )
    reshard_hetero.add_argument(
        "--output-locality-run-size",
        type=int,
        default=32,
        help="--output-order=interleavedで同一source/binから連続して書く最大event数",
    )
    reshard_hetero.add_argument("--seed", type=int, default=12345)
    reshard_hetero.add_argument("--shard-size", type=int, default=100000, help="N graphごとに出力HDF5を分割する。0なら単一ファイル")
    reshard_hetero.add_argument("--workers", type=int, default=1, help="出力shardコピーに使うworker数。shard-size > 0 の時だけ並列化する")
    reshard_hetero.add_argument("--energy-sample-stratify-particle", action="store_true", help="interleaved order のbinを particle:DAT energy code にする")
    reshard_hetero.add_argument(
        "--energy-sample-per-bin",
        type=int,
        default=None,
        help="既存hetero HDF5からDAT filename energy binごとに最大N graphを選んでコピーする",
    )
    reshard_hetero.add_argument("--overwrite", action="store_true", help="既存出力HDF5を上書きする")
    reshard_hetero.set_defaults(func=_cmd_reshard_hetero)

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
    train_hetero.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="GPU micro-batchを小さくした時にeffective batch sizeを保つための累積step数",
    )
    train_hetero.add_argument("--lr", type=float, default=1.0e-3)
    train_hetero.add_argument("--weight-decay", type=float, default=0.0)
    train_hetero.add_argument("--hidden-dim", type=int, default=128)
    train_hetero.add_argument("--layers", type=int, default=2)
    train_hetero.add_argument("--dropout", type=float, default=0.05)
    train_hetero.add_argument(
        "--model-architecture",
        choices=["minimal_hetero", "hetero_attention"],
        default="hetero_attention",
        help="hetero model architecture。通常は relation attention を持つ hetero_attention を使う",
    )
    train_hetero.add_argument("--attention-heads", type=int, default=4, help="hetero relation attention head数")
    train_hetero.add_argument("--readout-heads", type=int, default=4, help="detector/pulse type別 attention readout head数")
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
    train_hetero.add_argument("--num-workers", type=int, default=DEFAULT_TRAIN_WORKERS, help="hetero DataLoader worker数。-1ならCPUとメモリ見積もりから決める")
    train_hetero.add_argument("--prefetch-factor", type=int, default=2, help="各hetero DataLoader workerが先読みするbatch数")
    train_hetero.add_argument("--persistent-workers", action="store_true", help="hetero DataLoader workerをepoch間で保持する")
    train_hetero.add_argument("--no-pin-memory", action="store_true", help="CUDA転送用のpinned memoryを使わない")
    train_hetero.add_argument(
        "--loader-memory-budget-gib",
        type=float,
        default=None,
        help="DataLoader prefetchに使うCPU memory上限GiB。省略時はSlurmの割当memoryから読む",
    )
    train_hetero.add_argument("--loader-memory-estimate-samples", type=int, default=512)
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

    attention_maps = sub.add_parser("attention-maps", help="hetero_attention checkpointのattention mapを保存")
    attention_maps.add_argument("--graphs", nargs="*", default=[], help="checkpoint作成時と同じhetero HDF5 graph集合")
    attention_maps.add_argument("--graphs-list", action="append", default=[], help="hetero HDF5 shard path list")
    attention_maps.add_argument("--checkpoint", required=True, help="train-heteroで作成したhetero_attention checkpoint .pt")
    attention_maps.add_argument("-o", "--output", required=True, help="出力ディレクトリ")
    attention_maps.add_argument(
        "--split",
        choices=["validation", "val", "test", "train"],
        default="validation",
        help="checkpoint内のどのsplitからeventを選ぶか",
    )
    attention_maps.add_argument("--max-graphs", type=int, default=16, help="保存する最大event数。0ならsplit全件")
    attention_maps.add_argument(
        "--indices",
        default=None,
        help="保存するglobal graph index。例: 10,42,100:110。指定時は--splitの選択を上書きする",
    )
    attention_maps.add_argument("--device", default="auto")
    attention_maps.add_argument("--seed", type=int, default=12345)
    attention_maps.add_argument("--no-progress", action="store_true")
    attention_maps.set_defaults(func=_cmd_attention_maps)

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
