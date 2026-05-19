#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths
from talesd_gnn_reconstruction.progress import progress


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _read_string_array(dataset: h5py.Dataset) -> list[str]:
    data = dataset[:]
    return [_decode(value) for value in data]


def summarize(paths: list[Path], *, show_progress: bool = True) -> dict[str, Any]:
    iterator: Any = progress(paths, desc="summarize graph shards", total=len(paths), enabled=show_progress)

    total_graphs = 0
    total_bytes = 0
    proton = 0
    iron = 0
    unknown_particle = 0
    source_paths: set[str] = set()
    shard_rows: list[dict[str, Any]] = []

    for path in iterator:
        path = Path(path)
        total_bytes += path.stat().st_size
        with h5py.File(path, "r") as h5:
            n_graphs = int(len(h5["events"]))
            total_graphs += n_graphs
            row: dict[str, Any] = {
                "path": str(path),
                "graphs": n_graphs,
                "bytes": int(path.stat().st_size),
            }
            metadata = h5.get("metadata")
            if metadata is not None and "particle_label" in metadata:
                labels = np.asarray(metadata["particle_label"][:], dtype=np.float64)
                finite = labels[np.isfinite(labels)]
                shard_proton = int(np.sum(finite < 0.5))
                shard_iron = int(np.sum(finite >= 0.5))
                shard_unknown = int(n_graphs - finite.size)
                proton += shard_proton
                iron += shard_iron
                unknown_particle += shard_unknown
                row.update(
                    {
                        "proton": shard_proton,
                        "iron": shard_iron,
                        "unknown_particle": shard_unknown,
                    }
                )
            else:
                unknown_particle += n_graphs
                row["unknown_particle"] = n_graphs
            if metadata is not None and "source_path" in metadata:
                paths_in_shard = _read_string_array(metadata["source_path"])
                source_paths.update(paths_in_shard)
                row["unique_source_paths"] = len(set(paths_in_shard))
            shard_rows.append(row)

    return {
        "shards": len(paths),
        "graphs": total_graphs,
        "bytes": total_bytes,
        "gib": total_bytes / 1024.0**3,
        "proton": proton,
        "iron": iron,
        "unknown_particle": unknown_particle,
        "unique_source_paths": len(source_paths),
        "shard_rows": shard_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TALE-SD GNN HDF5 graph shards.")
    parser.add_argument("graphs", nargs="+", help="HDF5 shard, shard base path, or directory")
    parser.add_argument("-o", "--output", default=None, help="Optional JSON output path")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    paths = _expand_h5_graph_paths(args.graphs)
    if not paths:
        raise SystemExit("no graph files matched")
    payload = summarize(paths, show_progress=not args.no_progress)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
