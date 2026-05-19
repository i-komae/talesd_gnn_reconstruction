#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _run_key(name: str) -> str:
    if name.endswith(".pt.metrics.json"):
        name = name[: -len(".pt.metrics.json")]
    markers = ["_h128_", "_h160_", "_h192_", "_h256_", "_proton_", "_iron_"]
    for marker in markers:
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def _row(metrics_path: Path) -> dict[str, str | float | int]:
    stem = metrics_path.name[: -len(".pt.metrics.json")]
    checkpoint = metrics_path.with_name(stem + ".pt")
    diagnostics = metrics_path.with_name(stem + ".pt.diagnostics")
    log = ""
    data = _read_json(metrics_path)
    test = (data.get("metrics") or {}).get("test") or {}
    split = data.get("split") or {}
    runtime = data.get("runtime") or {}
    return {
        "run_key": _run_key(metrics_path.name),
        "config": stem,
        "checkpoint": str(checkpoint),
        "metrics": str(metrics_path),
        "diagnostics": str(diagnostics) if diagnostics.exists() else "",
        "log": log,
        "split_mode": split.get("split_mode", ""),
        "particle_filter": split.get("particle_filter", runtime.get("particle_filter", "")),
        "n_train": split.get("n_train", ""),
        "n_val": split.get("n_val", ""),
        "n_test": split.get("n_test", ""),
        "test_angular_68_deg": test.get("angular_68_deg", ""),
        "test_core_xy_68_km": test.get("core_xy_68_km", ""),
        "test_relative_energy_central68_half_width": test.get("relative_energy_central68_half_width", ""),
        "test_median_relative_energy": test.get("median_relative_energy", ""),
        "test_median_abs_relative_energy": test.get("median_abs_relative_energy", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index legacy flat output layout without moving files.")
    parser.add_argument(
        "--output-root",
        default="~/TALE/gnn/outputs/talesd_gnn_reconstruction",
        help="legacy output root containing models/logs/sweeps",
    )
    parser.add_argument("-o", "--output", default="", help="index CSV path")
    args = parser.parse_args()

    root = Path(args.output_root).expanduser()
    models = root / "models"
    rows = [_row(path) for path in sorted(models.glob("*.pt.metrics.json"))]

    log_dir = root / "logs"
    logs = list(log_dir.glob("*.log"))
    for row in rows:
        config = str(row["config"])
        matching = [path for path in logs if config in path.name]
        if matching:
            row["log"] = str(sorted(matching)[-1])

    output = Path(args.output).expanduser() if args.output else root / "legacy_index.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["run_key"])
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
