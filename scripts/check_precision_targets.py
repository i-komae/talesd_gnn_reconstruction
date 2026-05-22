#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = {
    "energy_bias_abs": 0.05,
    "energy_central68_half_width": 0.25,
    "angular_68_deg": 1.0,
    "core_xy_68_km": 0.05,
}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _status(value: float | None, limit: float, *, absolute: bool = False) -> str:
    if value is None:
        return "MISSING"
    checked = abs(value) if absolute else value
    return "PASS" if checked <= limit else "FAIL"


def _line(name: str, value: float | None, limit: float, *, absolute: bool = False, unit: str = "") -> str:
    status = _status(value, limit, absolute=absolute)
    rendered_value = "missing" if value is None else f"{value:.6g}{unit}"
    rendered_limit = f"{limit:.6g}{unit}"
    op = "|x| <= " if absolute else "<= "
    return f"{status:7s} {name:34s} {rendered_value:>14s}   target {op}{rendered_limit}"


def _diagnostics(payload: dict[str, Any], split_name: str) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics") or {}
    return diagnostics.get(split_name) or {}


def _check_energy_bins(rows: list[dict[str, Any]], min_bin_count: int, targets: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        n = int(row.get("n") or 0)
        if n < min_bin_count:
            continue
        low = _finite(row.get("log10_energy_low"))
        high = _finite(row.get("log10_energy_high"))
        label = f"{low:.2f}-{high:.2f}" if low is not None and high is not None else "unknown-bin"
        if row.get("fit_ok") is False:
            fit_error = row.get("fit_error") or "fit rejected"
            lines.append(f"MISSING energy fit bin {label:>12s} n={n:<8d} {fit_error}")
            continue
        mu = _finite(row.get("mu"))
        c68 = _finite(row.get("central68"))
        if _status(mu, targets["energy_bias_abs"], absolute=True) == "MISSING":
            lines.append(f"MISSING energy bias bin {label:>11s} n={n:<8d} mu=missing")
            continue
        if _status(mu, targets["energy_bias_abs"], absolute=True) == "FAIL":
            lines.append(f"FAIL    energy bias bin {label:>11s} n={n:<8d} mu={mu:.6g}")
        if _status(c68, targets["energy_central68_half_width"]) == "FAIL":
            lines.append(f"FAIL    energy spread bin {label:>9s} n={n:<8d} central68={c68:.6g}")
    return lines


def _check_resolution_bins(rows: list[dict[str, Any]], min_bin_count: int, targets: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        n = int(row.get("n") or 0)
        if n < min_bin_count:
            continue
        low = _finite(row.get("log10_energy_low"))
        high = _finite(row.get("log10_energy_high"))
        label = f"{low:.2f}-{high:.2f}" if low is not None and high is not None else "unknown-bin"
        opening = _finite(row.get("opening_angle_68_deg"))
        core = _finite(row.get("core_xy_68_km"))
        if _status(opening, targets["angular_68_deg"]) == "FAIL":
            lines.append(f"FAIL    angular 68 bin {label:>12s} n={n:<8d} angle={opening:.6g} deg")
        if _status(core, targets["core_xy_68_km"]) == "FAIL":
            lines.append(f"FAIL    core 68 bin {label:>15s} n={n:<8d} core={1000.0 * core:.6g} m")
    return lines


def _quality_cut_report(rows: list[dict[str, Any]], min_bin_count: int, targets: dict[str, float]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    passed = False
    candidates = [row for row in rows if int(row.get("n") or 0) >= int(min_bin_count)]
    if not candidates:
        return False, ["MISSING no quality cut has enough entries"]
    lines.append(f"Quality-cut candidates with n >= {min_bin_count}:")
    for row in candidates:
        threshold = _finite(row.get("quality_threshold"))
        survival = _finite(row.get("survival_fraction"))
        n = int(row.get("n") or 0)
        energy_bias = _finite(row.get("energy_mu"))
        if energy_bias is None:
            energy_bias = _finite(row.get("energy_median"))
        energy_spread = _finite(row.get("energy_central68"))
        angular = _finite(row.get("opening_angle_68_deg"))
        core = _finite(row.get("core_xy_68_km"))
        row_pass = (
            _status(energy_bias, targets["energy_bias_abs"], absolute=True) == "PASS"
            and _status(energy_spread, targets["energy_central68_half_width"]) == "PASS"
            and _status(angular, targets["angular_68_deg"]) == "PASS"
            and _status(core, targets["core_xy_68_km"]) == "PASS"
        )
        passed = passed or row_pass
        rendered = {
            "threshold": "missing" if threshold is None else f"{threshold:.4f}",
            "survival": "missing" if survival is None else f"{100.0 * survival:.2f}%",
            "energy_bias": "missing" if energy_bias is None else f"{energy_bias:.4g}",
            "energy_spread": "missing" if energy_spread is None else f"{energy_spread:.4g}",
            "angular": "missing" if angular is None else f"{angular:.4g} deg",
            "core": "missing" if core is None else f"{1000.0 * core:.4g} m",
        }
        lines.append(
            f"{'PASS' if row_pass else 'FAIL':7s} q>={rendered['threshold']:>8s} "
            f"survival={rendered['survival']:>8s} n={n:<8d} "
            f"energy_bias={rendered['energy_bias']:>10s} "
            f"energy_spread={rendered['energy_spread']:>10s} "
            f"angle={rendered['angular']:>12s} core={rendered['core']:>10s}"
        )
    return passed, lines


def build_report(payload: dict[str, Any], *, split_name: str, min_bin_count: int, targets: dict[str, float]) -> tuple[bool, str]:
    metrics = ((payload.get("metrics") or {}).get(split_name) or {})
    diagnostics = _diagnostics(payload, split_name)
    energy_diag = diagnostics.get("energy_relative_error") or {}

    energy_bias = _finite(metrics.get("median_relative_energy"))
    if energy_bias is None:
        energy_bias = _finite(energy_diag.get("mu"))
    energy_spread = _finite(metrics.get("relative_energy_central68_half_width"))
    if energy_spread is None:
        energy_spread = _finite(energy_diag.get("central68"))
    angular = _finite(metrics.get("angular_68_deg"))
    if angular is None:
        angular = _finite(diagnostics.get("opening_angle_68_deg"))
    core = _finite(metrics.get("core_xy_68_km"))
    if core is None:
        core = _finite(diagnostics.get("core_xy_68_km"))

    lines = [
        f"Precision target report: split={split_name}",
        _line("energy median bias", energy_bias, targets["energy_bias_abs"], absolute=True),
        _line("energy central 68% half-width", energy_spread, targets["energy_central68_half_width"]),
        _line("opening angle 68%", angular, targets["angular_68_deg"], unit=" deg"),
        _line("core xy 68%", 1000.0 * core if core is not None else None, 1000.0 * targets["core_xy_68_km"], unit=" m"),
    ]

    failures = [line for line in lines if line.startswith("FAIL")]
    energy_bin_failures = _check_energy_bins(
        list(diagnostics.get("energy_bins") or []),
        min_bin_count,
        targets,
    )
    resolution_bin_failures = _check_resolution_bins(
        list(diagnostics.get("resolution_bins") or []),
        min_bin_count,
        targets,
    )
    if energy_bin_failures or resolution_bin_failures:
        lines.append("")
        lines.append(f"Energy-bin failures with n >= {min_bin_count}:")
        lines.extend(energy_bin_failures)
        lines.extend(resolution_bin_failures)
        failures.extend(energy_bin_failures)
        failures.extend(resolution_bin_failures)
    all_event_passed = not failures and not any(line.startswith("MISSING") for line in lines)
    quality_rows = list((diagnostics.get("quality") or {}).get("cut_rows") or [])
    quality_passed = False
    if quality_rows:
        quality_passed, quality_lines = _quality_cut_report(quality_rows, min_bin_count, targets)
        lines.append("")
        lines.extend(quality_lines)
    passed = all_event_passed or quality_passed
    lines.append("")
    lines.append("OVERALL " + ("PASS" if passed else "FAIL"))
    return passed, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check reconstruction metrics against minimum physics targets.")
    parser.add_argument("metrics_json", help="*.pt.metrics.json")
    parser.add_argument("-o", "--output", default=None, help="Optional text report path")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--min-bin-count", type=int, default=1000)
    parser.add_argument("--energy-bias-abs", type=float, default=DEFAULT_TARGETS["energy_bias_abs"])
    parser.add_argument("--energy-central68-half-width", type=float, default=DEFAULT_TARGETS["energy_central68_half_width"])
    parser.add_argument("--angular-68-deg", type=float, default=DEFAULT_TARGETS["angular_68_deg"])
    parser.add_argument("--core-xy-68-km", type=float, default=DEFAULT_TARGETS["core_xy_68_km"])
    args = parser.parse_args()

    targets = {
        "energy_bias_abs": float(args.energy_bias_abs),
        "energy_central68_half_width": float(args.energy_central68_half_width),
        "angular_68_deg": float(args.angular_68_deg),
        "core_xy_68_km": float(args.core_xy_68_km),
    }
    payload = json.loads(Path(args.metrics_json).expanduser().read_text())
    passed, report = build_report(payload, split_name=args.split, min_bin_count=args.min_bin_count, targets=targets)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n")
    print(report)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
