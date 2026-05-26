#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.progress import progress_bar


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


def _count_events(paths: list[Path]) -> list[int]:
    counts = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            counts.append(int(len(h5["events"])))
    return counts


def _scan_selected(
    paths: list[Path],
    *,
    per_bin: int,
    max_total: int | None,
    energy_bin_width: float,
    stratify_particle: bool,
    particle_filter: str,
    seed: int,
    show_progress: bool,
) -> tuple[list[SelectedEvent], dict[str, Any]]:
    counts = _count_events(paths)
    total_events = int(sum(counts))
    reservoirs: dict[tuple[Any, int], list[tuple[float, int, int, SelectedEvent]]] = {}
    stats: dict[str, Any] = {
        "input_files": len(paths),
        "input_events": total_events,
        "missing_target": 0,
        "missing_particle_label": 0,
        "filtered_particle": 0,
        "nonfinite_energy": 0,
        "candidate_events": 0,
    }
    bar = progress_bar("scan small candidates", total_events, enabled=show_progress)
    try:
        for path_index, path in enumerate(paths):
            with h5py.File(path, "r") as h5:
                events = h5["events"]
                metadata = h5.get("metadata")
                for local_index, event_key in enumerate(_dense_or_sorted_keys(events)):
                    group = events[event_key]
                    if "target" not in group:
                        stats["missing_target"] += 1
                        bar.update(1)
                        continue
                    target = np.asarray(group["target"][()], dtype=np.float64).reshape(-1)
                    if target.size == 0 or not np.isfinite(target[0]):
                        stats["nonfinite_energy"] += 1
                        bar.update(1)
                        continue
                    label = _particle_label(metadata, group, local_index)
                    if label is None:
                        stats["missing_particle_label"] += 1
                        bar.update(1)
                        continue
                    if not _particle_allowed(label, particle_filter):
                        stats["filtered_particle"] += 1
                        bar.update(1)
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
                    bar.update(1)
    finally:
        bar.close()

    selected = [event for heap in reservoirs.values() for _neg_rank, _path_index, _local_index, event in heap]
    selected.sort(key=lambda event: event.rank)
    if max_total is not None and max_total > 0:
        selected = selected[: int(max_total)]
    stats["selected_events"] = len(selected)
    stats["strata"] = {
        f"{stratum[0]}:{stratum[1]}": len(heap)
        for stratum, heap in sorted(reservoirs.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    }
    return selected, stats


def _copy_attrs(src: h5py.File, dst: h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _write_selected(
    paths: list[Path],
    selected: list[SelectedEvent],
    output: Path,
    *,
    config: dict[str, Any],
    show_progress: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
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
        with h5py.File(paths[0], "r") as first, h5py.File(output, "w") as out:
            _copy_attrs(first, out)
            attr_config = dict(config)
            input_graphs = list(attr_config.pop("input_graphs", []))
            attr_config["input_graph_count"] = len(input_graphs)
            attr_config["input_graphs_preview"] = input_graphs[:10]
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

            bar = progress_bar("write small graphs", n_selected, enabled=show_progress)
            try:
                for output_index, event in enumerate(selected):
                    src = handle(event.path_index)
                    src.copy(src["events"][event.event_key], events_out, name=f"{output_index:08d}")
                    metadata["event_id"][output_index] = event.event_id
                    metadata["source_path"][output_index] = event.source_path
                    metadata["source_index"][output_index] = event.source_index
                    metadata["parttype"][output_index] = event.parttype
                    metadata["particle_label"][output_index] = event.particle_label
                    bar.update(1)
            finally:
                bar.close()
    finally:
        for h5 in handles.values():
            h5.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small tuning HDF5 graph dataset from existing large HDF5 shards.")
    parser.add_argument("--graphs", nargs="+", required=True, help="input HDF5 shard, shard base, or directory")
    parser.add_argument("-o", "--output", required=True, help="output small HDF5 path")
    parser.add_argument("--per-bin", type=int, default=2000, help="maximum events per energy bin or particle/energy bin")
    parser.add_argument("--max-total", type=int, default=None, help="optional total cap after per-bin sampling")
    parser.add_argument("--energy-bin-width", type=float, default=0.1, help="true log10(E/eV) bin width")
    parser.add_argument("--stratify-particle", dest="stratify_particle", action="store_true", default=True)
    parser.add_argument("--no-stratify-particle", dest="stratify_particle", action="store_false")
    parser.add_argument("--particle-filter", choices=["all", "proton", "iron"], default="all")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.per_bin <= 0:
        raise SystemExit("--per-bin must be positive")
    if args.energy_bin_width <= 0:
        raise SystemExit("--energy-bin-width must be positive")

    paths = [Path(path).expanduser() for path in _expand_h5_graph_paths(args.graphs)]
    if not paths:
        raise SystemExit("no input graph HDF5 files matched --graphs")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input graph HDF5 file: {missing[0]}")

    output = Path(args.output).expanduser()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists; pass --overwrite to replace: {output}")

    config = {
        "input_graphs": [str(path) for path in paths],
        "output": str(output),
        "per_bin": args.per_bin,
        "max_total": args.max_total,
        "energy_bin_width": args.energy_bin_width,
        "stratify_particle": args.stratify_particle,
        "particle_filter": args.particle_filter,
        "seed": args.seed,
    }
    selected, stats = _scan_selected(
        paths,
        per_bin=args.per_bin,
        max_total=args.max_total,
        energy_bin_width=args.energy_bin_width,
        stratify_particle=args.stratify_particle,
        particle_filter=args.particle_filter,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    if not selected:
        raise SystemExit("no events selected")
    if output.exists():
        output.unlink()
    _write_selected(paths, selected, output, config=config, show_progress=not args.no_progress)
    summary = {"config": config, "stats": stats}
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output: {output}")
    print(f"summary: {summary_path}")
    print(f"selected_events: {stats['selected_events']}")


if __name__ == "__main__":
    main()
