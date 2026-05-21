#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.progress import progress
from talesd_gnn_reconstruction.progress import progress_bar


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _read_string_array(dataset: h5py.Dataset) -> list[str]:
    data = dataset[:]
    return [_decode(value) for value in data]


def _summarize_one(payload: tuple[int, str]) -> tuple[int, dict[str, Any], set[str]]:
    index, path_text = payload
    path = Path(path_text)
    size = int(path.stat().st_size)
    source_paths: set[str] = set()
    with h5py.File(path, "r") as h5:
        n_graphs = int(len(h5["events"]))
        row: dict[str, Any] = {
            "path": str(path),
            "graphs": n_graphs,
            "bytes": size,
            "proton": 0,
            "iron": 0,
            "unknown_particle": 0,
        }
        metadata = h5.get("metadata")
        if metadata is not None and "particle_label" in metadata:
            labels = np.asarray(metadata["particle_label"][:], dtype=np.float64)
            finite = labels[np.isfinite(labels)]
            row["proton"] = int(np.sum(finite < 0.5))
            row["iron"] = int(np.sum(finite >= 0.5))
            row["unknown_particle"] = int(n_graphs - finite.size)
        else:
            row["unknown_particle"] = n_graphs
        if metadata is not None and "source_path" in metadata:
            paths_in_shard = _read_string_array(metadata["source_path"])
            source_paths.update(paths_in_shard)
            row["unique_source_paths"] = len(source_paths)
    return index, row, source_paths


def _merge_summary_row(
    row: dict[str, Any],
    source_paths: set[str],
    totals: dict[str, Any],
    rows: dict[int, dict[str, Any]],
    index: int,
) -> None:
    totals["graphs"] += int(row["graphs"])
    totals["bytes"] += int(row["bytes"])
    totals["proton"] += int(row.get("proton", 0))
    totals["iron"] += int(row.get("iron", 0))
    totals["unknown_particle"] += int(row.get("unknown_particle", 0))
    totals["source_paths"].update(source_paths)
    rows[index] = row


def summarize(paths: list[Path], *, show_progress: bool = True, workers: int = 1) -> dict[str, Any]:
    worker_count = min(max(int(workers), 1), max(len(paths), 1))
    totals: dict[str, Any] = {
        "graphs": 0,
        "bytes": 0,
        "proton": 0,
        "iron": 0,
        "unknown_particle": 0,
        "source_paths": set(),
    }
    rows: dict[int, dict[str, Any]] = {}

    if worker_count > 1 and len(paths) > 1:
        payloads = [(index, str(path)) for index, path in enumerate(paths)]
        progress_handle = progress_bar("summarize graph shards", len(paths), enabled=show_progress)
        pending = set()
        payload_iter = iter(payloads)
        max_pending = max(worker_count * 2, 1)
        pool = ProcessPoolExecutor(max_workers=worker_count)
        pool_closed = False

        def submit_next() -> bool:
            try:
                payload = next(payload_iter)
            except StopIteration:
                return False
            pending.add(pool.submit(_summarize_one, payload))
            return True

        try:
            for _ in range(min(max_pending, len(payloads))):
                submit_next()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index, row, source_set = future.result()
                    _merge_summary_row(row, source_set, totals, rows, index)
                    progress_handle.update(1)
                    submit_next()
            pool.shutdown(wait=True)
            pool_closed = True
        except BaseException:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            pool_closed = True
            raise
        finally:
            if not pool_closed:
                pool.shutdown(wait=False, cancel_futures=True)
            progress_handle.close()
    else:
        iterator: Any = progress(paths, desc="summarize graph shards", total=len(paths), enabled=show_progress)
        for index, path in enumerate(iterator):
            row_index, row, source_set = _summarize_one((index, str(path)))
            _merge_summary_row(row, source_set, totals, rows, row_index)

    return {
        "shards": len(paths),
        "graphs": totals["graphs"],
        "bytes": totals["bytes"],
        "gib": totals["bytes"] / 1024.0**3,
        "proton": totals["proton"],
        "iron": totals["iron"],
        "unknown_particle": totals["unknown_particle"],
        "unique_source_paths": len(totals["source_paths"]),
        "shard_rows": [rows[index] for index in sorted(rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TALE-SD GNN HDF5 graph shards.")
    parser.add_argument("graphs", nargs="+", help="HDF5 shard, shard base path, or directory")
    parser.add_argument("-o", "--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="HDF5 shard summary worker count. Use >1 to scan shards in parallel.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    paths = _expand_h5_graph_paths(args.graphs)
    if not paths:
        raise SystemExit("no graph files matched")
    payload = summarize(paths, show_progress=not args.no_progress, workers=args.workers)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
