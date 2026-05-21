#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _get(mapping: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _best_epoch(history: list[dict[str, Any]]) -> tuple[Any, Any]:
    if not history:
        return "", ""
    row = min(history, key=lambda item: float(item.get("val_loss", float("inf"))))
    return row.get("epoch", ""), row.get("val_loss", "")


def _row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    history = data.get("history") or []
    best_epoch, best_val_loss = _best_epoch(history)
    runtime = data.get("runtime") or {}
    split = data.get("split") or {}
    stage_seconds = runtime.get("stage_seconds") or {}
    return {
        "metrics_path": str(path),
        "checkpoint": str(path).removesuffix(".metrics.json"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": history[-1].get("epoch", "") if history else "",
        "last_train_loss": history[-1].get("train_loss", "") if history else "",
        "last_val_loss": history[-1].get("val_loss", "") if history else "",
        "test_angular_68_deg": _get(data, "metrics", "test", "angular_68_deg"),
        "test_angular_median_deg": _get(data, "metrics", "test", "angular_median_deg"),
        "test_core_median_km": _get(data, "metrics", "test", "core_median_km"),
        "test_core_68_km": _get(data, "metrics", "test", "core_68_km"),
        "test_core_xy_median_km": _get(data, "metrics", "test", "core_xy_median_km"),
        "test_core_xy_68_km": _get(data, "metrics", "test", "core_xy_68_km"),
        "test_core_rmse_km": _get(data, "metrics", "test", "core_rmse_km"),
        "test_median_abs_log10_energy": _get(data, "metrics", "test", "median_abs_log10_energy"),
        "test_median_relative_energy": _get(data, "metrics", "test", "median_relative_energy"),
        "test_mean_relative_energy": _get(data, "metrics", "test", "mean_relative_energy"),
        "test_median_abs_relative_energy": _get(data, "metrics", "test", "median_abs_relative_energy"),
        "test_abs_relative_energy_68": _get(data, "metrics", "test", "abs_relative_energy_68"),
        "test_relative_energy_central68_width": _get(data, "metrics", "test", "relative_energy_central68_width"),
        "test_relative_energy_central68_half_width": _get(data, "metrics", "test", "relative_energy_central68_half_width"),
        "test_rmse_log10_energy": _get(data, "metrics", "test", "rmse_log10_energy"),
        "test_mass_accuracy": _get(data, "metrics", "test_mass", "accuracy"),
        "test_mass_auc": _get(data, "metrics", "test_mass", "auc"),
        "test_mass_balanced_accuracy": _get(data, "metrics", "test_mass", "balanced_accuracy"),
        "val_angular_68_deg": _get(data, "metrics", "validation", "angular_68_deg"),
        "val_core_median_km": _get(data, "metrics", "validation", "core_median_km"),
        "val_core_xy_68_km": _get(data, "metrics", "validation", "core_xy_68_km"),
        "val_median_abs_log10_energy": _get(data, "metrics", "validation", "median_abs_log10_energy"),
        "val_median_relative_energy": _get(data, "metrics", "validation", "median_relative_energy"),
        "val_median_abs_relative_energy": _get(data, "metrics", "validation", "median_abs_relative_energy"),
        "val_relative_energy_central68_half_width": _get(data, "metrics", "validation", "relative_energy_central68_half_width"),
        "val_mass_accuracy": _get(data, "metrics", "validation_mass", "accuracy"),
        "val_mass_auc": _get(data, "metrics", "validation_mass", "auc"),
        "val_mass_balanced_accuracy": _get(data, "metrics", "validation_mass", "balanced_accuracy"),
        "n_train": split.get("n_train", ""),
        "n_val": split.get("n_val", ""),
        "n_test": split.get("n_test", ""),
        "split_mode": split.get("split_mode", ""),
        "particle_filter": split.get("particle_filter", runtime.get("particle_filter", "")),
        "learning_rate": runtime.get("learning_rate", ""),
        "weight_decay": runtime.get("weight_decay", ""),
        "lr_scheduler": runtime.get("lr_scheduler", ""),
        "hidden_dim": runtime.get("hidden_dim", ""),
        "layers": runtime.get("layers", ""),
        "dropout": runtime.get("dropout", ""),
        "device": runtime.get("device", ""),
        "training_task": runtime.get("training_task", ""),
        "num_workers": runtime.get("num_workers", ""),
        "collate_backend": runtime.get("collate_backend", ""),
        "stage_epochs_sec": stage_seconds.get("epochs", ""),
        "stage_total_sec": stage_seconds.get("total", stage_seconds.get("total_before_save", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect talesd-gnn metrics JSON files into one CSV.")
    parser.add_argument("metrics", nargs="+", help="*.metrics.json files")
    parser.add_argument("-o", "--output", required=True, help="output CSV")
    args = parser.parse_args()

    rows = [_row(Path(path).expanduser()) for path in args.metrics if Path(path).expanduser().exists()]
    if not rows:
        raise SystemExit("no metrics files found")
    rows.sort(
        key=lambda row: (
            float(row["test_angular_68_deg"])
            if row["test_angular_68_deg"] != ""
            else -float(row["test_mass_accuracy"])
            if row["test_mass_accuracy"] != ""
            else float("inf")
        )
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
