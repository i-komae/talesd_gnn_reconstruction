from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import balanced_accuracy_threshold, binary_classification_metrics
from .metrics import angular_error_deg

PROTON_COLOR = "#d62728"
IRON_COLOR = "#1f77b4"
NEUTRAL_COLOR = "0.25"
GRID_COLOR = "0.88"
FIGSIZE_SINGLE = (6.4, 4.4)
FIGSIZE_ENERGY = (7.0, 4.6)
FIGSIZE_PAIR = (10.4, 4.4)
FIGSIZE_TRIPLE = (13.2, 4.0)
FIGSIZE_STACKED = (7.0, 7.0)
FIGSIZE_GRID = (11.0, 7.0)
LINEWIDTH = 1.4
LINEWIDTH_THIN = 1.0
MARKERSIZE = 4.0
CAPSIZE = 2.5
ENERGY_MU_TARGET = 0.05
ENERGY_SIGMA_TARGET = 0.20
ENERGY_SIGMA_STRETCH_TARGET = 0.15
ANGULAR_TARGET_DEG = 1.0
CORE_TARGET_KM = 0.05
QUALITY_THRESHOLD_KEEP_FRACTIONS = (0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10, 0.05)
QUALITY_CUT_KEEP_FRACTIONS = (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.10, 0.05)
QUALITY_ENERGY_KEEP_FRACTIONS = (1.00, 0.95, 0.90, 0.80, 0.50, 0.10)
QUALITY_MARKER_KEEP_FRACTIONS = (0.95, 0.90, 0.80, 0.50, 0.10)
ERROR_SCATTER_MAX_POINTS = 50000
BALANCED_ACCURACY_PLOT_MIN_DELTA = 0.01


def require_matplotlib_latex() -> None:
    missing = [cmd for cmd in ("latex", "dvipng", "kpsewhich") if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(
            "matplotlib diagnostics require TeX because text.usetex=True. "
            f"Missing command(s): {', '.join(missing)}"
        )

    missing_files = []
    for tex_file in ("amsmath.sty",):
        result = subprocess.run(
            ["kpsewhich", tex_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            missing_files.append(tex_file)
    if missing_files:
        raise RuntimeError(
            "matplotlib diagnostics require TeX packages because text.usetex=True. "
            f"Missing file(s): {', '.join(missing_files)}"
        )


def _prepare_matplotlib() -> None:
    require_matplotlib_latex()
    cache_dir = Path(tempfile.gettempdir()) / "talesd_gnn_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "cm"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def _to_float(value: float) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _none_to_cache_array(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float32)
    return np.asarray(values)


def save_prediction_cache(
    output_path: str | Path,
    *,
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    validation_mass: tuple[np.ndarray, np.ndarray] | None = None,
    test_mass: tuple[np.ndarray, np.ndarray] | None = None,
    validation_quality: np.ndarray | None = None,
    test_quality: np.ndarray | None = None,
    validation_predicted_errors: np.ndarray | None = None,
    test_predicted_errors: np.ndarray | None = None,
) -> Path:
    cache_path = _diagnostics_root(Path(output_path).expanduser()) / "prediction_cache.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        pred_val=np.asarray(validation[0]),
        target_val=np.asarray(validation[1]),
        mass_logit_val=_none_to_cache_array(validation_mass[0] if validation_mass is not None else None),
        mass_label_val=_none_to_cache_array(validation_mass[1] if validation_mass is not None else None),
        quality_val=_none_to_cache_array(validation_quality),
        predicted_error_val=_none_to_cache_array(validation_predicted_errors),
        pred_test=np.asarray(test[0]),
        target_test=np.asarray(test[1]),
        mass_logit_test=_none_to_cache_array(test_mass[0] if test_mass is not None else None),
        mass_label_test=_none_to_cache_array(test_mass[1] if test_mass is not None else None),
        quality_test=_none_to_cache_array(test_quality),
        predicted_error_test=_none_to_cache_array(test_predicted_errors),
    )
    return cache_path


def _finite_mask(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2 or pred.shape != target.shape:
        return np.zeros(0, dtype=bool)
    mask = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(target), axis=1)
    return mask


def _finite_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = _finite_mask(pred, target)
    if mask.shape[0] != pred.shape[0] or mask.shape[0] != target.shape[0]:
        return pred[:0], target[:0]
    return pred[mask], target[mask]


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _central68_half_width(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [16.0, 84.0])
    return float(0.5 * (hi - lo))


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0.0:
        scale = _central68_half_width(values)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = max(float(np.ptp(values)) / 6.0, 1.0e-12)
    return median, scale


def _robust_window(values: np.ndarray, center: float, scale: float, *, nsigma: float, min_half_width: float) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        return values, float("nan"), float("nan")
    half_width = max(float(nsigma) * float(scale), float(min_half_width))
    low = float(center - half_width)
    high = float(center + half_width)
    return values[(values >= low) & (values <= high)], low, high


def _diagnostics_root(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".diagnostics")


def _save_pdf(fig: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return str(path)


class _SplitPdfWriter:
    def __init__(self, directory: Path, page_names: list[str], written: list[str]):
        self.directory = directory
        self.page_names = list(page_names)
        self.written = written
        self.index = 0

    def __enter__(self) -> "_SplitPdfWriter":
        self.directory.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None

    def savefig(self, fig: Any, name: str | None = None) -> None:
        if name is not None:
            name = name
        elif self.index < len(self.page_names):
            name = self.page_names[self.index]
        else:
            name = f"figure_{self.index:02d}"
        path = self.directory / f"{self.index:02d}_{name}.pdf"
        fig.savefig(path)
        self.written.append(str(path))
        self.index += 1


def _bootstrap_percentile_se(
    values: np.ndarray,
    q: float,
    *,
    n_resamples: int = 200,
    max_samples: int = 5000,
    seed: int = 12345,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    original_n = int(values.size)
    if values.size > max_samples:
        values = rng.choice(values, size=max_samples, replace=False)
    n = int(values.size)
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        samples[index] = np.percentile(rng.choice(values, size=n, replace=True), q)
    se = float(np.std(samples, ddof=1))
    if original_n > n:
        se *= math.sqrt(n / original_n)
    return se


def _gaussian_curve(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    sigma = abs(float(sigma))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return np.zeros_like(x)
    z = (x - float(mu)) / sigma
    return float(amplitude) * np.exp(-0.5 * z * z)


def _fit_gaussian_hist(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    robust_mu, robust_sigma = _robust_location_scale(values)
    central68 = _central68_half_width(values)
    abs68 = _percentile(np.abs(values), 68.0)
    fit_values, fit_low, fit_high = _robust_window(values, robust_mu, robust_sigma, nsigma=5.0, min_half_width=0.05)

    def failed(error: str) -> dict[str, Any]:
        return {
            "n": int(values.size),
            "fit_n": int(fit_values.size),
            "mu": None,
            "mu_err": None,
            "sigma": None,
            "sigma_err": None,
            "amplitude": None,
            "amplitude_err": None,
            "central68": _to_float(central68),
            "abs68": _to_float(abs68),
            "robust_median": _to_float(robust_mu),
            "robust_sigma": _to_float(robust_sigma),
            "fit_window_low": _to_float(fit_low),
            "fit_window_high": _to_float(fit_high),
            "fit_ok": False,
            "fit_method": "robust_window_scipy_curve_fit",
            "fit_error": error,
        }

    if values.size == 0:
        return failed("empty sample")
    if values.size < 3 or float(np.ptp(values)) <= 0.0:
        return failed("too few distinct values")
    if fit_values.size < 3 or float(np.ptp(fit_values)) <= 0.0:
        return failed("too few distinct values inside robust fit window")

    from scipy.optimize import curve_fit

    mu0, sigma0 = _robust_location_scale(fit_values)
    sigma0 = max(float(sigma0), 1.0e-12)
    bins = min(max(int(math.sqrt(fit_values.size)), 12), 80)
    density, edges = np.histogram(fit_values, bins=bins, range=(fit_low, fit_high), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = np.isfinite(density) & np.isfinite(centers) & (density > 0.0)
    density = density[mask]
    centers = centers[mask]
    if density.size < 3:
        return failed("too few populated histogram bins inside robust fit window")
    amplitude0 = float(np.max(density)) if density.size else 1.0 / (sigma0 * math.sqrt(2.0 * math.pi))

    fit_ok = True
    fit_error = None
    amplitude = amplitude0
    mu = float(mu0)
    sigma = sigma0
    amplitude_err = None
    mu_err = None
    sigma_err = None
    try:
        mu_half_width = max(4.0 * float(robust_sigma), 0.05)
        sigma_upper = max(5.0 * float(robust_sigma), 0.05)
        sigma_lower = max(float(robust_sigma) / 20.0, 1.0e-12)
        mu_lower = float(robust_mu) - mu_half_width
        mu_upper = float(robust_mu) + mu_half_width
        mu0 = float(np.clip(mu0, mu_lower, mu_upper))
        sigma0 = float(np.clip(sigma0, sigma_lower, sigma_upper))
        popt, pcov = curve_fit(
            _gaussian_curve,
            centers,
            density,
            p0=[amplitude0, mu0, sigma0],
            bounds=(
                [0.0, mu_lower, sigma_lower],
                [np.inf, mu_upper, sigma_upper],
            ),
            maxfev=20000,
        )
        amplitude = float(popt[0])
        mu = float(popt[1])
        sigma = abs(float(popt[2]))
        if not (math.isfinite(amplitude) and math.isfinite(mu) and math.isfinite(sigma)):
            raise RuntimeError("fit returned non-finite parameter")
        if abs(mu - float(robust_mu)) > mu_half_width:
            raise RuntimeError(f"fit mu outside robust window: mu={mu:g}, median={robust_mu:g}")
        if sigma < sigma_lower or sigma > sigma_upper:
            raise RuntimeError(f"fit sigma outside robust window: sigma={sigma:g}, robust_sigma={robust_sigma:g}")
        if pcov is not None and np.shape(pcov) == (3, 3) and np.all(np.isfinite(pcov)):
            errors = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
            amplitude_err = float(errors[0])
            mu_err = float(errors[1])
            sigma_err = float(errors[2])
    except Exception as exc:
        fit_ok = False
        fit_error = str(exc)

    return {
        "n": int(values.size),
        "fit_n": int(fit_values.size),
        "mu": _to_float(mu) if fit_ok else None,
        "mu_err": _to_float(mu_err) if mu_err is not None else None,
        "sigma": _to_float(sigma) if fit_ok else None,
        "sigma_err": _to_float(sigma_err) if sigma_err is not None else None,
        "amplitude": _to_float(amplitude) if fit_ok else None,
        "amplitude_err": _to_float(amplitude_err) if amplitude_err is not None else None,
        "central68": _to_float(central68),
        "abs68": _to_float(abs68),
        "robust_median": _to_float(robust_mu),
        "robust_sigma": _to_float(robust_sigma),
        "fit_window_low": _to_float(fit_low),
        "fit_window_high": _to_float(fit_high),
        "fit_ok": fit_ok,
        "fit_method": "robust_window_scipy_curve_fit",
        "fit_error": fit_error,
    }


def _hist_bins(values: np.ndarray, default: int = 60) -> int | str:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return max(5, min(default, int(values.size)))
    return "auto"


def _style_axes(ax: Any) -> None:
    ax.tick_params(direction="in")


def _add_energy_bias_guides(ax: Any) -> None:
    ax.axhspan(-ENERGY_MU_TARGET, ENERGY_MU_TARGET, color="0.93", zorder=0)
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=LINEWIDTH_THIN)
    ax.axhline(ENERGY_MU_TARGET, color="0.45", linestyle="--", linewidth=LINEWIDTH_THIN)
    ax.axhline(-ENERGY_MU_TARGET, color="0.45", linestyle="--", linewidth=LINEWIDTH_THIN)
    ax.axhline(ENERGY_SIGMA_STRETCH_TARGET, color="0.65", linestyle=":", linewidth=LINEWIDTH_THIN)
    ax.axhline(-ENERGY_SIGMA_STRETCH_TARGET, color="0.65", linestyle=":", linewidth=LINEWIDTH_THIN)
    ax.axhline(ENERGY_SIGMA_TARGET, color="0.65", linestyle="-.", linewidth=LINEWIDTH_THIN)
    ax.axhline(-ENERGY_SIGMA_TARGET, color="0.65", linestyle="-.", linewidth=LINEWIDTH_THIN)


def _add_energy_sigma_guides(ax: Any) -> None:
    ax.axhline(ENERGY_SIGMA_STRETCH_TARGET, color="0.45", linestyle=":", linewidth=LINEWIDTH_THIN)
    ax.axhline(ENERGY_SIGMA_TARGET, color="0.45", linestyle="--", linewidth=LINEWIDTH_THIN)


def _add_angular_target(ax: Any) -> None:
    ax.axhline(ANGULAR_TARGET_DEG, color="0.45", linestyle="--", linewidth=LINEWIDTH_THIN)


def _add_core_target(ax: Any, *, unit: str) -> None:
    scale = 1000.0 if unit == "m" else 1.0
    ax.axhline(CORE_TARGET_KM * scale, color="0.45", linestyle="--", linewidth=LINEWIDTH_THIN)


def _draw_gaussian_hist(
    ax: Any,
    values: np.ndarray,
    title: str,
    xlabel: str,
    color: str = "#4c78a8",
    show_central68: bool = True,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    stats = _fit_gaussian_hist(values)
    if values.size == 0:
        ax.text(0.5, 0.5, "no entries", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        _style_axes(ax)
        return stats

    display_mu = stats.get("robust_median")
    display_sigma = stats.get("robust_sigma")
    display_values = values
    display_low = None
    display_high = None
    if display_mu is not None and display_sigma is not None and float(display_sigma) > 0.0:
        display_values, display_low, display_high = _robust_window(
            values,
            float(display_mu),
            float(display_sigma),
            nsigma=6.0,
            min_half_width=0.08,
        )
        if display_values.size < max(5, int(0.2 * values.size)):
            display_values = values
            display_low = None
            display_high = None

    ax.hist(
        display_values,
        bins=_hist_bins(display_values),
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=color,
        edgecolor=color,
    )
    mu = stats["mu"]
    sigma = stats["sigma"]
    if stats.get("fit_ok", False) and mu is not None and sigma is not None and sigma > 0.0:
        if display_low is not None and display_high is not None:
            x_min = float(display_low)
            x_max = float(display_high)
        else:
            x_min, x_max = np.percentile(values, [0.5, 99.5])
        pad = 0.15 * max(x_max - x_min, sigma)
        x = np.linspace(x_min - pad, x_max + pad, 300)
        amplitude = stats.get("amplitude")
        if amplitude is None:
            amplitude = 1.0 / (float(sigma) * math.sqrt(2.0 * math.pi))
        ax.plot(x, _gaussian_curve(x, float(amplitude), float(mu), float(sigma)), color=PROTON_COLOR, linewidth=LINEWIDTH)
        ax.axvline(float(mu), color=PROTON_COLOR, linewidth=LINEWIDTH_THIN)
    if display_low is not None and display_high is not None:
        ax.set_xlim(float(display_low), float(display_high))
    c68 = stats["central68"] if show_central68 else None
    if c68 is not None:
        ax.axvline(c68, color="#2ca02c", linestyle="--", linewidth=LINEWIDTH_THIN)
        ax.axvline(-c68, color="#2ca02c", linestyle="--", linewidth=LINEWIDTH_THIN)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    label = f"n={stats['n']}"
    if display_values.size != values.size:
        label += f"\nshown={display_values.size}"
    if stats.get("fit_ok", False) and mu is not None and sigma is not None:
        label += f"\nmu={mu:.4g}\nsigma={sigma:.4g}"
    elif stats.get("robust_median") is not None and stats.get("robust_sigma") is not None:
        label += f"\nmedian={stats['robust_median']:.4g}\nrobust sigma={stats['robust_sigma']:.4g}"
        label += "\nfit failed"
    if c68 is not None:
        label += f"\ncentral 68\\%={c68:.4g}"
    ax.text(0.98, 0.96, label, ha="right", va="top", transform=ax.transAxes)
    _style_axes(ax)
    return stats


def _energy_bin_edges(log10_energy: np.ndarray, bin_width: float) -> np.ndarray:
    values = np.asarray(log10_energy, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    width = max(float(bin_width), 1.0e-6)
    # Center bins on exact width ticks: for width=0.1, bins are
    # 15.95-16.05, 16.05-16.15, ... instead of 15.9-16.0, 16.0-16.1.
    offset = 0.5 * width
    lo = math.floor((float(values.min()) - offset) / width) * width + offset
    hi = math.ceil((float(values.max()) - offset) / width) * width + offset
    if hi <= lo:
        hi = lo + width
    return np.arange(lo, hi + 0.5 * width, width, dtype=np.float64)


def _energy_bin_table(
    true_log10_energy: np.ndarray,
    rel_error: np.ndarray,
    bin_width: float,
) -> list[dict[str, Any]]:
    edges = _energy_bin_edges(true_log10_energy, bin_width)
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.size - 2:
            mask = (true_log10_energy >= low) & (true_log10_energy <= high)
        else:
            mask = (true_log10_energy >= low) & (true_log10_energy < high)
        values = rel_error[mask]
        stats = _fit_gaussian_hist(values)
        rows.append(
            {
                "log10_energy_low": _to_float(low),
                "log10_energy_high": _to_float(high),
                "log10_energy_center": _to_float(0.5 * (low + high)),
                **stats,
            }
        )
    return rows


def _resolution_bin_table(
    true_log10_energy: np.ndarray,
    opening_deg: np.ndarray,
    core_xy_km: np.ndarray,
    bin_width: float,
) -> list[dict[str, Any]]:
    edges = _energy_bin_edges(true_log10_energy, bin_width)
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.size - 2:
            mask = (true_log10_energy >= low) & (true_log10_energy <= high)
        else:
            mask = (true_log10_energy >= low) & (true_log10_energy < high)
        opening = opening_deg[mask]
        core = core_xy_km[mask]
        seed_base = 12345 + index * 17
        rows.append(
            {
                "log10_energy_low": _to_float(low),
                "log10_energy_high": _to_float(high),
                "log10_energy_center": _to_float(0.5 * (low + high)),
                "n": int(np.sum(mask)),
                "opening_angle_68_deg": _to_float(_percentile(opening, 68.0)),
                "opening_angle_68_deg_err": _to_float(_bootstrap_percentile_se(opening, 68.0, seed=seed_base + 1)),
                "opening_angle_median_deg": _to_float(_percentile(opening, 50.0)),
                "opening_angle_median_deg_err": _to_float(_bootstrap_percentile_se(opening, 50.0, seed=seed_base + 2)),
                "core_xy_68_km": _to_float(_percentile(core, 68.0)),
                "core_xy_68_km_err": _to_float(_bootstrap_percentile_se(core, 68.0, seed=seed_base + 3)),
                "core_xy_median_km": _to_float(_percentile(core, 50.0)),
                "core_xy_median_km_err": _to_float(_bootstrap_percentile_se(core, 50.0, seed=seed_base + 4)),
            }
        )
    return rows


def _prediction_quantities(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred, target = _finite_pair(pred, target)
    dx = pred[:, 1] - target[:, 1]
    dy = pred[:, 2] - target[:, 2]
    core_xy = np.sqrt(dx * dx + dy * dy)
    loge_delta = pred[:, 0] - target[:, 0]
    rel_energy = np.power(10.0, loge_delta) - 1.0
    return {
        "pred": pred,
        "target": target,
        "true_log10_energy": target[:, 0],
        "opening_deg": angular_error_deg(pred, target),
        "core_dx_km": dx,
        "core_dy_km": dy,
        "core_xy_km": core_xy,
        "rel_energy": rel_energy,
    }


def _quality_for_finite_pair(pred: np.ndarray, target: np.ndarray, quality: np.ndarray | None) -> np.ndarray | None:
    if quality is None:
        return None
    values = np.asarray(quality, dtype=np.float64).reshape(-1)
    mask = _finite_mask(pred, target)
    if mask.size == 0 or values.shape[0] != mask.shape[0]:
        return None
    values = values[mask]
    values = np.clip(values, 0.0, 1.0)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    return values


def _predicted_errors_for_finite_pair(
    pred: np.ndarray,
    target: np.ndarray,
    predicted_errors: np.ndarray | None,
) -> np.ndarray | None:
    if predicted_errors is None:
        return None
    values = np.asarray(predicted_errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        return None
    mask = _finite_mask(pred, target)
    if mask.size == 0 or values.shape[0] != mask.shape[0]:
        return None
    values = values[mask, :3]
    values = np.where(np.isfinite(values), np.clip(values, 0.0, None), np.nan)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    return values


def _quality_cut_mask(quality: np.ndarray, keep_fraction: float) -> tuple[np.ndarray, float]:
    values = np.asarray(quality, dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(values, dtype=bool), float("nan")
    if keep_fraction >= 1.0:
        threshold = float(np.nanmin(values[valid]))
        return valid, threshold
    threshold = float(np.percentile(values[valid], 100.0 * (1.0 - float(keep_fraction))))
    return valid & (values >= threshold), threshold


def _quality_cut_directory_name(keep_fraction: float) -> str:
    return f"keep_{int(round(100.0 * float(keep_fraction))):02d}pct"


def _subset_prediction_quantities(q: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    subset: dict[str, np.ndarray] = {}
    for key, values in q.items():
        array = np.asarray(values)
        subset[key] = array[mask] if array.shape[:1] == mask.shape[:1] else array
    return subset


def _quality_threshold_summary(quality: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(quality, dtype=np.float64)
    values = values[np.isfinite(values)]
    rows: list[dict[str, Any]] = []
    if values.size == 0:
        return rows
    for keep_fraction in QUALITY_THRESHOLD_KEEP_FRACTIONS:
        threshold = float(np.percentile(values, 100.0 * (1.0 - keep_fraction)))
        rows.append(
            {
                "keep_fraction": _to_float(keep_fraction),
                "quality_threshold": _to_float(threshold),
                "n_kept": int(np.sum(values >= threshold)),
                "n_total": int(values.size),
            }
        )
    return rows


def _quality_cut_rows(q: dict[str, np.ndarray], quality: np.ndarray, min_bin_count: int) -> list[dict[str, Any]]:
    values = np.asarray(quality, dtype=np.float64)
    valid = np.isfinite(values)
    rows: list[dict[str, Any]] = []
    if not np.any(valid):
        return rows
    total = int(np.sum(valid))
    for keep_fraction in QUALITY_CUT_KEEP_FRACTIONS:
        mask, threshold = _quality_cut_mask(values, keep_fraction)
        n = int(np.sum(mask))
        energy_stats = _fit_gaussian_hist(q["rel_energy"][mask])
        rows.append(
            {
                "quality_threshold": _to_float(threshold),
                "requested_keep_fraction": _to_float(keep_fraction),
                "survival_fraction": _to_float(n / max(total, 1)),
                "n": n,
                "passes_min_count": bool(n >= int(min_bin_count)),
                "opening_angle_68_deg": _to_float(_percentile(q["opening_deg"][mask], 68.0)),
                "opening_angle_median_deg": _to_float(_percentile(q["opening_deg"][mask], 50.0)),
                "core_xy_68_km": _to_float(_percentile(q["core_xy_km"][mask], 68.0)),
                "core_xy_median_km": _to_float(_percentile(q["core_xy_km"][mask], 50.0)),
                "energy_mu": energy_stats.get("mu") if energy_stats.get("fit_ok", False) else None,
                "energy_sigma": energy_stats.get("sigma") if energy_stats.get("fit_ok", False) else None,
                "energy_central68": energy_stats.get("central68"),
                "energy_abs68": energy_stats.get("abs68"),
            }
        )
    return rows


def _quality_energy_rows(
    q: dict[str, np.ndarray],
    quality: np.ndarray,
    *,
    keep_fraction: float,
    energy_bin_width: float,
) -> dict[str, list[dict[str, Any]] | float]:
    values = np.asarray(quality, dtype=np.float64)
    mask, threshold = _quality_cut_mask(values, keep_fraction)
    return {
        "keep_fraction": float(keep_fraction),
        "quality_threshold": threshold,
        "energy_rows": _energy_bin_table(q["true_log10_energy"][mask], q["rel_energy"][mask], energy_bin_width),
        "resolution_rows": _resolution_bin_table(
            q["true_log10_energy"][mask],
            q["opening_deg"][mask],
            q["core_xy_km"][mask],
            energy_bin_width,
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_correlation(x: np.ndarray, y: np.ndarray, *, method: str) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.nanstd(x) <= 0.0 or np.nanstd(y) <= 0.0:
        return None
    if method == "spearman":
        x = _average_ranks(x)
        y = _average_ranks(y)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if math.isfinite(corr) else None


def _error_correlation_summary_from_variables(variables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(variables)
    rows: list[dict[str, Any]] = []
    for y_index, y_name in enumerate(names):
        for x_name in names[:y_index]:
            x = variables[x_name]["values"]
            y = variables[y_name]["values"]
            mask = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "x": x_name,
                    "y": y_name,
                    "n": int(np.sum(mask)),
                    "pearson": _safe_correlation(x, y, method="pearson"),
                    "spearman": _safe_correlation(x, y, method="spearman"),
                }
            )
    return {
        "variables": [
            {
                "name": name,
                "label": str(info["label"]),
                "unit": str(info["unit"]),
                "n": int(np.sum(np.isfinite(info["values"]))),
                "median": _to_float(np.nanmedian(info["values"])),
                "p68": _to_float(_percentile(info["values"], 68.0)),
                "p95": _to_float(_percentile(info["values"], 95.0)),
            }
            for name, info in variables.items()
        ],
        "pairs": rows,
    }


def _actual_error_correlation_inputs(q: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        "energy_abs_relative_error": {
            "values": np.abs(q["rel_energy"]),
            "label": r"$|E_{\mathrm{rec}}/E_{\mathrm{true}}-1|$",
            "unit": "",
        },
        "opening_angle_deg": {
            "values": q["opening_deg"],
            "label": "opening angle [deg]",
            "unit": "deg",
        },
        "core_xy_error_m": {
            "values": q["core_xy_km"] * 1000.0,
            "label": "core error [m]",
            "unit": "m",
        },
    }


def _error_correlation_summary(q: dict[str, np.ndarray]) -> dict[str, Any]:
    return _error_correlation_summary_from_variables(_actual_error_correlation_inputs(q))


def _predicted_error_correlation_inputs(predicted_errors: np.ndarray) -> dict[str, dict[str, Any]]:
    values = np.asarray(predicted_errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        values = np.full((0, 3), np.nan, dtype=np.float64)
    return {
        "predicted_energy_abs_relative_error": {
            "values": values[:, 0],
            "label": r"pred. $|E_{\mathrm{rec}}/E_{\mathrm{true}}-1|$",
            "unit": "",
        },
        "predicted_opening_angle_deg": {
            "values": values[:, 1],
            "label": "pred. opening angle [deg]",
            "unit": "deg",
        },
        "predicted_core_xy_error_m": {
            "values": values[:, 2] * 1000.0,
            "label": "pred. core error [m]",
            "unit": "m",
        },
    }


def _predicted_actual_error_inputs(
    q: dict[str, np.ndarray],
    predicted_errors: np.ndarray,
) -> dict[str, dict[str, Any]]:
    values = np.asarray(predicted_errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        values = np.full((0, 3), np.nan, dtype=np.float64)
    actual = _actual_error_correlation_inputs(q)
    return {
        "energy_abs_relative_error": {
            "actual": actual["energy_abs_relative_error"]["values"],
            "predicted": values[:, 0],
            "label": r"$|E_{\mathrm{rec}}/E_{\mathrm{true}}-1|$",
            "unit": "",
        },
        "opening_angle_deg": {
            "actual": actual["opening_angle_deg"]["values"],
            "predicted": values[:, 1],
            "label": "opening angle [deg]",
            "unit": "deg",
        },
        "core_xy_error_m": {
            "actual": actual["core_xy_error_m"]["values"],
            "predicted": values[:, 2] * 1000.0,
            "label": "core error [m]",
            "unit": "m",
        },
    }


def _predicted_actual_error_summary(calibration: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, info in calibration.items():
        actual = np.asarray(info["actual"], dtype=np.float64)
        predicted = np.asarray(info["predicted"], dtype=np.float64)
        mask = np.isfinite(actual) & np.isfinite(predicted)
        rows.append(
            {
                "name": name,
                "label": str(info["label"]),
                "unit": str(info["unit"]),
                "n": int(np.sum(mask)),
                "pearson": _safe_correlation(predicted, actual, method="pearson"),
                "spearman": _safe_correlation(predicted, actual, method="spearman"),
                "actual_median": _to_float(np.nanmedian(actual[mask])) if np.any(mask) else None,
                "predicted_median": _to_float(np.nanmedian(predicted[mask])) if np.any(mask) else None,
                "actual_p68": _to_float(_percentile(actual[mask], 68.0)) if np.any(mask) else None,
                "predicted_p68": _to_float(_percentile(predicted[mask], 68.0)) if np.any(mask) else None,
            }
        )
    return {"variables": rows}


def _save_error_correlation_scatter(
    variables: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    title: str,
    rng_seed: int = 314159,
) -> Any:
    import matplotlib.pyplot as plt

    names = list(variables)
    arrays = [np.asarray(variables[name]["values"], dtype=np.float64).reshape(-1) for name in names]
    finite = np.ones(arrays[0].shape[0], dtype=bool)
    for values in arrays:
        finite &= np.isfinite(values)
    indices = np.flatnonzero(finite)
    if indices.size == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "no finite reconstruction errors", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig
    if indices.size > ERROR_SCATTER_MAX_POINTS:
        rng = np.random.default_rng(rng_seed)
        indices = np.sort(rng.choice(indices, size=ERROR_SCATTER_MAX_POINTS, replace=False))

    limits: list[tuple[float, float]] = []
    for values in arrays:
        finite_values = values[np.isfinite(values)]
        high = _percentile(finite_values, 99.5)
        if not math.isfinite(high) or high <= 0.0:
            high = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        high = max(float(high), 1.0e-12)
        limits.append((0.0, high))

    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.0))
    pair_rows = {
        (str(row["y"]), str(row["x"])): row
        for row in summary.get("pairs", [])
    }
    for row_index, y_name in enumerate(names):
        for col_index, x_name in enumerate(names):
            ax = axes[row_index, col_index]
            x = np.asarray(variables[x_name]["values"], dtype=np.float64)
            y = np.asarray(variables[y_name]["values"], dtype=np.float64)
            if col_index > row_index:
                ax.axis("off")
                continue
            if row_index == col_index:
                values = x[np.isfinite(x)]
                ax.hist(
                    values,
                    bins=np.linspace(limits[col_index][0], limits[col_index][1], 41),
                    histtype="stepfilled",
                    color="#4c78a8",
                    alpha=0.35,
                    edgecolor="#4c78a8",
                )
                ax.set_xlim(*limits[col_index])
            else:
                ax.scatter(
                    x[indices],
                    y[indices],
                    s=2.0,
                    alpha=0.12,
                    color=NEUTRAL_COLOR,
                    rasterized=True,
                    linewidths=0,
                )
                ax.set_xlim(*limits[col_index])
                ax.set_ylim(*limits[row_index])
                row = pair_rows.get((y_name, x_name), {})
                pearson = row.get("pearson")
                spearman = row.get("spearman")
                label_parts = []
                if pearson is not None:
                    label_parts.append(f"r={float(pearson):.3f}")
                if spearman is not None:
                    label_parts.append(r"$\rho$=" + f"{float(spearman):.3f}")
                if label_parts:
                    ax.text(
                        0.04,
                        0.94,
                        "\n".join(label_parts),
                        ha="left",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=9,
                    )
            if row_index == len(names) - 1:
                ax.set_xlabel(str(variables[x_name]["label"]))
            else:
                ax.set_xticklabels([])
            if col_index == 0:
                ax.set_ylabel(str(variables[y_name]["label"]) if row_index != col_index else "events")
            else:
                ax.set_yticklabels([])
            _style_axes(ax)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _save_predicted_actual_error_scatter(
    calibration: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    title: str,
    rng_seed: int = 271828,
) -> Any:
    import matplotlib.pyplot as plt

    names = list(calibration)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    rows = {str(row["name"]): row for row in summary.get("variables", [])}
    for axis_index, name in enumerate(names):
        ax = axes[axis_index]
        actual = np.asarray(calibration[name]["actual"], dtype=np.float64).reshape(-1)
        predicted = np.asarray(calibration[name]["predicted"], dtype=np.float64).reshape(-1)
        finite = np.isfinite(actual) & np.isfinite(predicted)
        indices = np.flatnonzero(finite)
        if indices.size > ERROR_SCATTER_MAX_POINTS:
            rng = np.random.default_rng(rng_seed + axis_index)
            indices = np.sort(rng.choice(indices, size=ERROR_SCATTER_MAX_POINTS, replace=False))
        finite_actual = actual[finite]
        finite_predicted = predicted[finite]
        if finite_actual.size == 0:
            ax.text(0.5, 0.5, "no finite entries", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        high = max(_percentile(finite_actual, 99.5), _percentile(finite_predicted, 99.5))
        if not math.isfinite(high) or high <= 0.0:
            high = max(float(np.nanmax(finite_actual)), float(np.nanmax(finite_predicted)), 1.0)
        high = max(float(high), 1.0e-12)
        ax.scatter(
            predicted[indices],
            actual[indices],
            s=2.0,
            alpha=0.12,
            color=NEUTRAL_COLOR,
            rasterized=True,
            linewidths=0,
        )
        ax.plot([0.0, high], [0.0, high], color=PROTON_COLOR, linewidth=LINEWIDTH_THIN, label="ideal")
        row = rows.get(name, {})
        pearson = row.get("pearson")
        spearman = row.get("spearman")
        text_lines = []
        if pearson is not None:
            text_lines.append(f"r={float(pearson):.3f}")
        if spearman is not None:
            text_lines.append(r"$\rho$=" + f"{float(spearman):.3f}")
        if text_lines:
            ax.text(0.04, 0.94, "\n".join(text_lines), ha="left", va="top", transform=ax.transAxes, fontsize=9)
        ax.set_xlim(0.0, high)
        ax.set_ylim(0.0, high)
        ax.set_xlabel("predicted " + str(calibration[name]["label"]))
        ax.set_ylabel("actual " + str(calibration[name]["label"]))
        ax.legend(frameon=False)
        _style_axes(ax)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _save_learning_curve(output_path: Path, history: list[dict[str, Any]]) -> Path:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    path = _diagnostics_root(output_path) / "learning_curve.pdf"
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    has_reconstruction = "train_reconstruction_loss" in history[0] if history else False
    has_mass = "train_mass_loss" in history[0] if history else False
    has_mass_accuracy = "train_mass_accuracy" in history[0] and "val_mass_accuracy" in history[0] if history else False
    has_mass_balanced_accuracy = False
    if history and "train_mass_balanced_accuracy" in history[0] and "val_mass_balanced_accuracy" in history[0]:
        train_delta = np.max(
            np.abs(
                np.asarray([row["train_mass_accuracy"] for row in history], dtype=float)
                - np.asarray([row["train_mass_balanced_accuracy"] for row in history], dtype=float)
            )
        )
        val_delta = np.max(
            np.abs(
                np.asarray([row["val_mass_accuracy"] for row in history], dtype=float)
                - np.asarray([row["val_mass_balanced_accuracy"] for row in history], dtype=float)
            )
        )
        has_mass_balanced_accuracy = max(float(train_delta), float(val_delta)) >= BALANCED_ACCURACY_PLOT_MIN_DELTA
    mass_loss_duplicates_total = bool(
        has_mass
        and not has_reconstruction
        and all(
            np.isclose(row["train_loss"], row["train_mass_loss"])
            and np.isclose(row["val_loss"], row["val_mass_loss"])
            for row in history
        )
    )
    show_separate_mass_loss = bool(has_mass and not mass_loss_duplicates_total)

    if has_mass:
        ncols = 1 + int(show_separate_mass_loss) + int(has_mass_accuracy) + int(has_mass_balanced_accuracy)
        fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.2))
        axes = np.atleast_1d(axes)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        axes = [ax]
    ax.plot(epochs, train, marker="o", markersize=2.5, linewidth=1.4, label="train")
    ax.plot(epochs, val, marker="s", markersize=2.5, linewidth=1.4, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss" if mass_loss_duplicates_total else "loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style_axes(ax)
    panel_index = 1
    if show_separate_mass_loss:
        ax = axes[panel_index]
        panel_index += 1
        ax.plot(epochs, [row["train_mass_loss"] for row in history], marker="o", markersize=2.5, linewidth=1.4, label="train")
        ax.plot(epochs, [row["val_mass_loss"] for row in history], marker="s", markersize=2.5, linewidth=1.4, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        _style_axes(ax)
    if has_mass_accuracy:
        ax = axes[panel_index]
        panel_index += 1
        ax.plot(epochs, [row["train_mass_accuracy"] for row in history], marker="o", markersize=2.5, linewidth=1.4, label="train accuracy")
        ax.plot(epochs, [row["val_mass_accuracy"] for row in history], marker="s", markersize=2.5, linewidth=1.4, label="validation accuracy")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False)
        _style_axes(ax)
    if has_mass_balanced_accuracy:
        ax = axes[panel_index]
        ax.plot(
            epochs,
            [row["train_mass_balanced_accuracy"] for row in history],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label="train balanced accuracy",
        )
        ax.plot(
            epochs,
            [row["val_mass_balanced_accuracy"] for row in history],
            marker="s",
            markersize=2.5,
            linewidth=1.4,
            label="validation balanced accuracy",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("balanced accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False)
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    pos = values >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_values = np.exp(values[~pos])
    out[~pos] = exp_values / (1.0 + exp_values)
    return out


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    return fpr, tpr


def _mass_energy_bin_table(
    true_log10_energy: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    bin_width: float,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    edges = _energy_bin_edges(true_log10_energy, bin_width)
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    predictions = scores >= float(threshold)
    truth = labels >= 0.5
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.size - 2:
            mask = (true_log10_energy >= low) & (true_log10_energy <= high)
        else:
            mask = (true_log10_energy >= low) & (true_log10_energy < high)
        n = int(np.sum(mask))
        if n == 0:
            rows.append(
                {
                    "log10_energy_low": _to_float(low),
                    "log10_energy_high": _to_float(high),
                    "log10_energy_center": _to_float(0.5 * (low + high)),
                    "n": 0,
                    "accuracy": None,
                    "proton_accuracy": None,
                    "iron_accuracy": None,
                    "n_proton": 0,
                    "n_iron": 0,
                }
            )
            continue
        true_bin = truth[mask]
        pred_bin = predictions[mask]
        proton_mask = ~true_bin
        iron_mask = true_bin
        rows.append(
            {
                "log10_energy_low": _to_float(low),
                "log10_energy_high": _to_float(high),
                "log10_energy_center": _to_float(0.5 * (low + high)),
                "n": n,
                "accuracy": _to_float(np.mean(pred_bin == true_bin)),
                "proton_accuracy": _to_float(np.mean(~pred_bin[proton_mask])) if np.any(proton_mask) else None,
                "iron_accuracy": _to_float(np.mean(pred_bin[iron_mask])) if np.any(iron_mask) else None,
                "n_proton": int(np.sum(proton_mask)),
                "n_iron": int(np.sum(iron_mask)),
            }
        )
    return rows


def _plot_mass_accuracy_by_energy(
    ax: Any,
    energy_rows: list[dict[str, Any]],
    *,
    min_bin_count: int,
    split_name: str,
) -> bool:
    valid_rows = [row for row in energy_rows if int(row["n"] or 0) >= int(min_bin_count)]
    if not valid_rows:
        ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{split_name}: mass accuracy by true energy")
        ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        ax.set_ylabel("accuracy")
        _style_axes(ax)
        return False

    x = np.asarray([row["log10_energy_center"] for row in valid_rows], dtype=float)
    all_accuracy = np.asarray([row["accuracy"] for row in valid_rows], dtype=float)
    proton = np.asarray([np.nan if row["proton_accuracy"] is None else row["proton_accuracy"] for row in valid_rows])
    iron = np.asarray([np.nan if row["iron_accuracy"] is None else row["iron_accuracy"] for row in valid_rows])
    ax.plot(x, all_accuracy, "o-", color=NEUTRAL_COLOR, label="all")
    ax.plot(x, proton, "s--", color=PROTON_COLOR, label="proton")
    ax.plot(x, iron, "^--", color=IRON_COLOR, label="iron")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{split_name}: mass accuracy by true energy")
    ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False)
    _style_axes(ax)
    return True


def _species_resolution_rows(
    true_log10_energy: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    species_label: float,
    bin_width: float,
    q: float = 68.0,
) -> list[dict[str, Any]]:
    mask = np.isfinite(labels) & (labels >= 0.5 if species_label >= 0.5 else labels < 0.5)
    edges = _energy_bin_edges(true_log10_energy[mask], bin_width)
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.size - 2:
            bin_mask = mask & (true_log10_energy >= low) & (true_log10_energy <= high)
        else:
            bin_mask = mask & (true_log10_energy >= low) & (true_log10_energy < high)
        bin_values = values[bin_mask]
        rows.append(
            {
                "log10_energy_low": _to_float(low),
                "log10_energy_high": _to_float(high),
                "log10_energy_center": _to_float(0.5 * (low + high)),
                "n": int(np.sum(bin_mask)),
                "p68": _to_float(_percentile(bin_values, q)),
                "p68_err": _to_float(_bootstrap_percentile_se(bin_values, q, seed=23456 + index)),
                "median": _to_float(_percentile(bin_values, 50.0)),
                "median_err": _to_float(_bootstrap_percentile_se(bin_values, 50.0, seed=34567 + index)),
            }
        )
    return rows


def _species_energy_rows(
    true_log10_energy: np.ndarray,
    rel_energy: np.ndarray,
    labels: np.ndarray,
    species_label: float,
    bin_width: float,
    edges: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    mask = np.isfinite(labels) & (labels >= 0.5 if species_label >= 0.5 else labels < 0.5)
    if edges is None:
        edges = _energy_bin_edges(true_log10_energy[mask], bin_width)
    else:
        edges = np.asarray(edges, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.size - 2:
            bin_mask = mask & (true_log10_energy >= low) & (true_log10_energy <= high)
        else:
            bin_mask = mask & (true_log10_energy >= low) & (true_log10_energy < high)
        stats = _fit_gaussian_hist(rel_energy[bin_mask])
        rows.append(
            {
                "log10_energy_low": _to_float(low),
                "log10_energy_high": _to_float(high),
                "log10_energy_center": _to_float(0.5 * (low + high)),
                **stats,
            }
        )
    return rows


def _valid_energy_fit_rows(rows: list[dict[str, Any]], min_bin_count: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if int(row["n"] or 0) < int(min_bin_count):
            continue
        if not row.get("fit_ok", False):
            continue
        if row.get("mu") is None or row.get("sigma") is None:
            continue
        out.append(row)
    return out


def _rows_array(rows: list[dict[str, Any]], key: str, default: float = float("nan")) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, default)
        values.append(default if value is None else value)
    return np.asarray(values, dtype=float)


def _plot_energy_mu_sigma_rows(
    ax: Any,
    row_sets: list[tuple[str, list[dict[str, Any]], str]],
    *,
    min_bin_count: int,
    title: str,
) -> bool:
    plotted = False
    for label, rows, color in row_sets:
        valid_rows = _valid_energy_fit_rows(rows, min_bin_count)
        if not valid_rows:
            continue
        x = _rows_array(valid_rows, "log10_energy_center")
        mu = _rows_array(valid_rows, "mu")
        sigma = _rows_array(valid_rows, "sigma")
        ax.errorbar(
            x,
            mu,
            yerr=sigma,
            fmt="o",
            linestyle="none",
            color=color,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            label=label,
        )
        plotted = True
    if plotted:
        ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=LINEWIDTH_THIN)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("relative energy error")
    _style_axes(ax)
    return plotted


def _plot_species_energy_fit_parameter_rows(
    axes: Any,
    row_sets: list[tuple[str, list[dict[str, Any]], str]],
    *,
    min_bin_count: int,
    split_name: str,
) -> bool:
    plotted = False
    for label, rows, color in row_sets:
        valid_rows = _valid_energy_fit_rows(rows, min_bin_count)
        if not valid_rows:
            continue
        x = _rows_array(valid_rows, "log10_energy_center")
        mu = _rows_array(valid_rows, "mu")
        mu_err = _rows_array(valid_rows, "mu_err")
        sigma = _rows_array(valid_rows, "sigma")
        sigma_err = _rows_array(valid_rows, "sigma_err")
        axes[0].errorbar(
            x,
            mu,
            yerr=mu_err if np.any(np.isfinite(mu_err)) else None,
            fmt="o-",
            color=color,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            label=label,
        )
        axes[1].errorbar(
            x,
            sigma,
            yerr=sigma_err if np.any(np.isfinite(sigma_err)) else None,
            fmt="o-",
            color=color,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            label=label,
        )
        plotted = True
    if not plotted:
        for ax in axes:
            ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
    axes[0].axhline(0.0, color=NEUTRAL_COLOR, linewidth=LINEWIDTH_THIN)
    axes[0].set_title(f"{split_name}: Gaussian mu by true energy and species")
    axes[0].set_ylabel("Gaussian mu")
    axes[1].set_title(f"{split_name}: Gaussian sigma by true energy and species")
    axes[1].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    axes[1].set_ylabel("Gaussian sigma")
    for ax in axes:
        if plotted:
            ax.legend()
        _style_axes(ax)
    return plotted


def _plot_species_rows(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    error_key: str | None,
    label: str,
    color: str,
    min_bin_count: int,
    linestyle: str = "-",
) -> None:
    valid_rows = [row for row in rows if int(row["n"] or 0) >= int(min_bin_count)]
    if not valid_rows:
        return
    x = np.asarray([row["log10_energy_center"] for row in valid_rows], dtype=float)
    y = np.asarray([row[value_key] for row in valid_rows], dtype=float)
    if error_key is None:
        ax.plot(x, y, marker="o", linestyle=linestyle, color=color, linewidth=1.4, markersize=4, label=label)
    else:
        yerr = np.asarray([row[error_key] for row in valid_rows], dtype=float)
        ax.errorbar(x, y, yerr=yerr, fmt="o", linestyle=linestyle, color=color, linewidth=1.4, markersize=4, capsize=2.5, label=label)


def _save_species_reconstruction_outputs(
    output_path: Path,
    split_name: str,
    pred: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    energy_bin_width: float,
    min_bin_count: int,
) -> dict[str, Any] | None:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    mask = (
        pred.ndim == 2
        and target.ndim == 2
        and pred.shape[0] == target.shape[0]
        and pred.shape[0] == labels.shape[0]
    )
    if not mask:
        return None
    valid = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(target), axis=1) & np.isfinite(labels)
    pred = pred[valid]
    target = target[valid]
    labels = labels[valid]
    if labels.size == 0:
        return None
    q = _prediction_quantities(pred, target)
    proton_mask = labels < 0.5
    iron_mask = labels >= 0.5
    split_dir = _diagnostics_root(output_path) / split_name
    pdf_files: list[str] = []

    species = [
        ("proton", proton_mask, PROTON_COLOR, 0.0),
        ("iron", iron_mask, IRON_COLOR, 1.0),
    ]
    energy_edges = _energy_bin_edges(q["true_log10_energy"], energy_bin_width)
    species_energy_rows = {
        name: _species_energy_rows(
            q["true_log10_energy"],
            q["rel_energy"],
            labels,
            label_value,
            energy_bin_width,
            edges=energy_edges,
        )
        for name, _species_mask, _color, label_value in species
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for name, species_mask, color, _label_value in species:
        if not np.any(species_mask):
            continue
        axes[0].hist(q["rel_energy"][species_mask], bins=_hist_bins(q["rel_energy"][species_mask]), histtype="step", linewidth=1.6, color=color, label=name)
        axes[1].hist(q["opening_deg"][species_mask], bins=_hist_bins(q["opening_deg"][species_mask]), histtype="step", linewidth=1.6, color=color, label=name)
        axes[2].hist(q["core_xy_km"][species_mask], bins=_hist_bins(q["core_xy_km"][species_mask]), histtype="step", linewidth=1.6, color=color, label=name)
    axes[0].set_title(f"{split_name}: energy error by species")
    axes[0].set_xlabel(r"$(E_{\mathrm{rec}}-E_{\mathrm{true}})/E_{\mathrm{true}}$")
    axes[1].set_title(f"{split_name}: opening angle by species")
    axes[1].set_xlabel("opening angle [deg]")
    axes[2].set_title(f"{split_name}: core error by species")
    axes[2].set_xlabel("core position error [km]")
    for ax in axes:
        ax.set_ylabel("events")
        ax.legend(frameon=False)
        _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "03_species_error_histograms.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, _species_mask, color, _label_value in species:
        rows = _valid_energy_fit_rows(species_energy_rows[name], min_bin_count)
        if rows:
            x = _rows_array(rows, "log10_energy_center")
            mu = _rows_array(rows, "mu")
            sigma = _rows_array(rows, "sigma")
            ax.errorbar(
                x,
                mu,
                yerr=sigma,
                fmt="o",
                color=color,
                capsize=2.5,
                label=name,
            )
    if ax.has_data():
        _add_energy_bias_guides(ax)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{split_name}: energy resolution by true energy and species")
    ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("relative energy error")
    _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "04_species_energy_resolution_by_true_energy.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
    for name, _species_mask, color, _label_value in species:
        rows = _valid_energy_fit_rows(species_energy_rows[name], min_bin_count)
        if not rows:
            continue
        x = _rows_array(rows, "log10_energy_center")
        mu = _rows_array(rows, "mu")
        mu_err = _rows_array(rows, "mu_err")
        sigma = _rows_array(rows, "sigma")
        sigma_err = _rows_array(rows, "sigma_err")
        axes[0].errorbar(
            x,
            mu,
            yerr=mu_err if np.any(np.isfinite(mu_err)) else None,
            fmt="o-",
            color=color,
            linewidth=1.3,
            markersize=4,
            capsize=2.5,
            label=name,
        )
        axes[1].errorbar(
            x,
            sigma,
            yerr=sigma_err if np.any(np.isfinite(sigma_err)) else None,
            fmt="o-",
            color=color,
            linewidth=1.3,
            markersize=4,
            capsize=2.5,
            label=name,
        )
    _add_energy_bias_guides(axes[0])
    _add_energy_sigma_guides(axes[1])
    axes[0].set_title(f"{split_name}: Gaussian mu by true energy and species")
    axes[0].set_ylabel("Gaussian mu")
    axes[1].set_title(f"{split_name}: Gaussian sigma by true energy and species")
    axes[1].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    axes[1].set_ylabel("Gaussian sigma")
    for ax in axes:
        ax.legend(frameon=False)
        _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "04b_species_energy_fit_parameters_by_true_energy.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for name, _species_mask, color, label_value in species:
        opening_rows = _species_resolution_rows(q["true_log10_energy"], q["opening_deg"], labels, label_value, energy_bin_width)
        core_rows = _species_resolution_rows(q["true_log10_energy"], q["core_xy_km"], labels, label_value, energy_bin_width)
        _plot_species_rows(
            axes[0],
            opening_rows,
            value_key="p68",
            error_key="p68_err",
            label=f"{name} 68\\%",
            color=color,
            min_bin_count=min_bin_count,
        )
        _plot_species_rows(
            axes[1],
            core_rows,
            value_key="p68",
            error_key="p68_err",
            label=f"{name} 68\\%",
            color=color,
            min_bin_count=min_bin_count,
        )
    axes[0].set_title(f"{split_name}: angular resolution by true energy and species")
    axes[0].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    axes[0].set_ylabel("opening angle [deg]")
    axes[1].set_title(f"{split_name}: core resolution by true energy and species")
    axes[1].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    axes[1].set_ylabel("core position error [km]")
    _add_angular_target(axes[0])
    _add_core_target(axes[1], unit="km")
    for ax in axes:
        ax.legend(frameon=False)
        _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "05_species_angular_core_resolution_by_true_energy.pdf"))
    plt.close(fig)

    return {
        "directory": str(split_dir),
        "pdfs": pdf_files,
        "n_proton": int(np.sum(proton_mask)),
        "n_iron": int(np.sum(iron_mask)),
        "energy_bins": species_energy_rows,
    }


def _save_mass_pdf(
    output_path: Path,
    split_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    energy_bin_width: float,
    min_bin_count: int,
    threshold: float = 0.5,
    tuned_threshold: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[0] != logits.shape[0]:
        target = np.full((logits.shape[0], 1), np.nan, dtype=np.float64)
    mask = np.isfinite(logits) & np.isfinite(labels)
    logits = logits[mask]
    labels = labels[mask]
    target = target[mask]
    scores = _sigmoid(logits)
    truth = labels >= 0.5
    predictions = scores >= float(threshold)
    metrics = binary_classification_metrics(logits, labels, threshold=threshold)
    tuned_metrics = (
        binary_classification_metrics(logits, labels, threshold=tuned_threshold)
        if tuned_threshold is not None
        else None
    )
    energy_rows = _mass_energy_bin_table(target[:, 0], scores, labels, energy_bin_width, threshold=threshold)
    summary: dict[str, Any] = {
        **metrics,
        "primary_threshold": float(threshold),
        "tuned_threshold": None if tuned_threshold is None else float(tuned_threshold),
        "tuned_metrics": tuned_metrics,
        "energy_bins": energy_rows,
    }

    split_dir = _diagnostics_root(output_path) / split_name
    pdf_files: list[str] = []

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    matrix = np.asarray(
        [
            [metrics["tn_proton"], metrics["fp_iron"]],
            [metrics["fn_iron"], metrics["tp_iron"]],
        ],
        dtype=float,
    )
    total_entries = float(np.sum(matrix))
    matrix_percent = np.zeros_like(matrix)
    if total_entries > 0:
        matrix_percent = 100.0 * matrix / total_entries
    summary["confusion_counts"] = {
        "tn_proton": int(matrix[0, 0]),
        "fp_iron": int(matrix[0, 1]),
        "fn_iron": int(matrix[1, 0]),
        "tp_iron": int(matrix[1, 1]),
    }
    summary["confusion_percent_of_all"] = {
        "tn_proton": _to_float(matrix_percent[0, 0]),
        "fp_iron": _to_float(matrix_percent[0, 1]),
        "fn_iron": _to_float(matrix_percent[1, 0]),
        "tp_iron": _to_float(matrix_percent[1, 1]),
    }
    image = ax.imshow(matrix_percent, cmap="Blues", vmin=0.0)
    ax.set_xticks([0, 1], labels=["pred proton", "pred iron"])
    ax.set_yticks([0, 1], labels=["true proton", "true iron"])
    cell_names = np.asarray([["TN", "FP"], ["FN", "TP"]])
    threshold_percent = 0.45 * float(np.nanmax(matrix_percent)) if np.any(np.isfinite(matrix_percent)) else 0.0
    for (row, col), value in np.ndenumerate(matrix):
        percent = matrix_percent[row, col]
        text_color = "white" if percent >= threshold_percent and threshold_percent > 0.0 else "black"
        ax.text(
            col,
            row,
            f"{cell_names[row, col]}\n{int(value):,}\n{percent:.1f}\\%",
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": (0, 0, 0, 0.18) if text_color == "white" else (1, 1, 1, 0.72), "edgecolor": "none"},
        )
    ax.set_title(f"{split_name}: mass confusion matrix")
    ax.text(0.5, -0.18, "percentages are fractions of all events", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("events [\\%]")
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "mass_confusion_matrix.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    proton_scores = scores[~truth]
    iron_scores = scores[truth]
    bins = np.linspace(0.0, 1.0, 41)
    if proton_scores.size:
        ax.hist(proton_scores, bins=bins, histtype="step", linewidth=1.6, label="true proton", color=PROTON_COLOR)
    if iron_scores.size:
        ax.hist(iron_scores, bins=bins, histtype="step", linewidth=1.6, label="true iron", color=IRON_COLOR)
    ax.axvline(float(threshold), color="0.25", linestyle="--", linewidth=1.1, label=f"threshold={threshold:.3g}")
    if tuned_threshold is not None and abs(float(tuned_threshold) - float(threshold)) > 1.0e-6:
        ax.axvline(
            float(tuned_threshold),
            color="0.5",
            linestyle=":",
            linewidth=1.1,
            label=f"tuned={tuned_threshold:.3g}",
        )
    ax.set_title(f"{split_name}: predicted iron probability")
    ax.set_xlabel("P(iron)")
    ax.set_ylabel("events")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "mass_score_distribution.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    fpr, tpr = _roc_points(scores, truth.astype(np.int8))
    ax.plot(fpr, tpr, color="#54a24b", linewidth=1.8)
    ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1.0)
    auc = metrics.get("auc")
    ax.set_title(f"{split_name}: ROC" + (f" AUC={auc:.3g}" if auc is not None else ""))
    ax.set_xlabel("false iron rate")
    ax.set_ylabel("true iron rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "mass_roc.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE_ENERGY)
    _plot_mass_accuracy_by_energy(ax, energy_rows, min_bin_count=min_bin_count, split_name=split_name)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "mass_accuracy_by_true_energy.pdf"))
    plt.close(fig)

    summary["directory"] = str(split_dir)
    summary["pdfs"] = pdf_files
    return split_dir, summary


def _save_reconstruction_cut_pages(
    directory: Path,
    split_label: str,
    q: dict[str, np.ndarray],
    energy_bin_width: float,
    min_bin_count: int,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    pdf_files: list[str] = []
    bin_rows = _energy_bin_table(q["true_log10_energy"], q["rel_energy"], energy_bin_width)
    resolution_rows = _resolution_bin_table(
        q["true_log10_energy"],
        q["opening_deg"],
        q["core_xy_km"],
        energy_bin_width,
    )
    summary: dict[str, Any] = {
        "directory": str(directory),
        "n": int(q["target"].shape[0]),
        "opening_angle_68_deg": _to_float(_percentile(q["opening_deg"], 68.0)),
        "core_xy_68_km": _to_float(_percentile(q["core_xy_km"], 68.0)),
        "energy_relative_error": _fit_gaussian_hist(q["rel_energy"]),
        "energy_bins": bin_rows,
        "resolution_bins": resolution_rows,
        "pdfs": pdf_files,
    }
    page_names = [
        "opening_angle",
        "core_coordinate_residuals",
        "core_position_error",
        "energy_relative_error",
        "energy_resolution_by_true_energy",
        "angular_core_resolution_by_true_energy",
    ]
    with _SplitPdfWriter(directory, page_names, pdf_files) as pdf:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.hist(q["opening_deg"], bins=_hist_bins(q["opening_deg"]), histtype="stepfilled", color="#4c78a8", alpha=0.4, edgecolor="#4c78a8")
        open68 = summary["opening_angle_68_deg"]
        if open68 is not None:
            ax.axvline(open68, color=PROTON_COLOR, linewidth=LINEWIDTH, label=f"68\\% = {open68:.3g} deg")
            ax.legend(frameon=False)
        ax.set_title(f"{split_label}: opening angle")
        ax.set_xlabel("opening angle [deg]")
        ax.set_ylabel("events")
        _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
        for ax, values, label, color in [
            (axes[0], q["core_dx_km"], "core x residual [km]", "#4c78a8"),
            (axes[1], q["core_dy_km"], "core y residual [km]", "#f58518"),
        ]:
            ax.hist(values, bins=_hist_bins(values), histtype="stepfilled", color=color, alpha=0.4, edgecolor=color)
            c68 = _central68_half_width(values)
            if math.isfinite(c68):
                ax.axvline(c68, color="#2ca02c", linestyle="--", linewidth=LINEWIDTH_THIN)
                ax.axvline(-c68, color="#2ca02c", linestyle="--", linewidth=LINEWIDTH_THIN)
                ax.text(0.98, 0.96, f"central 68\\%={c68:.3g} km", ha="right", va="top", transform=ax.transAxes, fontsize=9)
            ax.axvline(0.0, color=NEUTRAL_COLOR, linewidth=LINEWIDTH_THIN)
            ax.set_xlabel(label)
            _style_axes(ax)
        axes[0].set_ylabel("events")
        fig.suptitle(f"{split_label}: core coordinate residuals")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.hist(q["core_xy_km"], bins=_hist_bins(q["core_xy_km"]), histtype="stepfilled", color="#54a24b", alpha=0.4, edgecolor="#54a24b")
        core68 = summary["core_xy_68_km"]
        if core68 is not None:
            ax.axvline(core68, color=PROTON_COLOR, linewidth=LINEWIDTH, label=f"68\\% = {core68:.3g} km")
            ax.legend(frameon=False)
        ax.set_title(f"{split_label}: core position error")
        ax.set_xlabel(r"$\sqrt{\Delta x^2 + \Delta y^2}$ [km]")
        ax.set_ylabel("events")
        _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        summary["energy_relative_error"] = _draw_gaussian_hist(
            ax,
            q["rel_energy"],
            f"{split_label}: relative energy error",
            r"$(E_{\mathrm{rec}}-E_{\mathrm{true}})/E_{\mathrm{true}}$",
            color="#b279a2",
            show_central68=False,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        valid_rows = _valid_energy_fit_rows(bin_rows, min_bin_count)
        display_rows = [row for row in bin_rows if int(row["n"] or 0) >= int(min_bin_count)]
        fig, ax = plt.subplots(figsize=FIGSIZE_ENERGY)
        if valid_rows:
            x = _rows_array(valid_rows, "log10_energy_center")
            mu = _rows_array(valid_rows, "mu")
            sigma = _rows_array(valid_rows, "sigma")
            ax.errorbar(x, mu, yerr=sigma, fmt="o", color=PROTON_COLOR, capsize=CAPSIZE, label=r"Gaussian $\mu\pm\sigma$")
            _add_energy_bias_guides(ax)
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{split_label}: energy resolution by true energy")
        ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        ax.set_ylabel("relative energy error")
        _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        valid_resolution_rows = [row for row in resolution_rows if int(row["n"] or 0) >= int(min_bin_count)]
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR)
        if valid_resolution_rows:
            x = _rows_array(valid_resolution_rows, "log10_energy_center")
            axes[0].errorbar(
                x,
                _rows_array(valid_resolution_rows, "opening_angle_68_deg"),
                yerr=_rows_array(valid_resolution_rows, "opening_angle_68_deg_err"),
                fmt="o-",
                color="#4c78a8",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                label=r"68\%",
            )
            axes[0].errorbar(
                x,
                _rows_array(valid_resolution_rows, "opening_angle_median_deg"),
                yerr=_rows_array(valid_resolution_rows, "opening_angle_median_deg_err"),
                fmt="s--",
                color="#72b7b2",
                linewidth=LINEWIDTH_THIN,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                label="median",
            )
            axes[1].errorbar(
                x,
                _rows_array(valid_resolution_rows, "core_xy_68_km"),
                yerr=_rows_array(valid_resolution_rows, "core_xy_68_km_err"),
                fmt="o-",
                color="#54a24b",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                label=r"68\%",
            )
            axes[1].errorbar(
                x,
                _rows_array(valid_resolution_rows, "core_xy_median_km"),
                yerr=_rows_array(valid_resolution_rows, "core_xy_median_km_err"),
                fmt="s--",
                color="#b79a20",
                linewidth=LINEWIDTH_THIN,
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                label="median",
            )
            _add_angular_target(axes[0])
            _add_core_target(axes[1], unit="km")
            for ax in axes:
                ax.legend(frameon=False)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
        axes[0].set_title(f"{split_label}: angular resolution by true energy")
        axes[0].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        axes[0].set_ylabel("opening angle [deg]")
        axes[1].set_title(f"{split_label}: core resolution by true energy")
        axes[1].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        axes[1].set_ylabel("core position error [km]")
        for ax in axes:
            _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        rows_per_page = 6
        for start in range(0, len(display_rows), rows_per_page):
            rows = display_rows[start : start + rows_per_page]
            fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID)
            axes_flat = axes.ravel()
            for ax in axes_flat:
                ax.axis("off")
            for ax, row in zip(axes_flat, rows):
                ax.axis("on")
                low = float(row["log10_energy_low"])
                high = float(row["log10_energy_high"])
                if high == bin_rows[-1]["log10_energy_high"]:
                    mask = (q["true_log10_energy"] >= low) & (q["true_log10_energy"] <= high)
                else:
                    mask = (q["true_log10_energy"] >= low) & (q["true_log10_energy"] < high)
                _draw_gaussian_hist(
                    ax,
                    q["rel_energy"][mask],
                    f"{low:.1f} <= logE < {high:.1f}",
                    r"$(E_{\mathrm{rec}}-E_{\mathrm{true}})/E_{\mathrm{true}}$",
                    color="#b279a2",
                    show_central68=False,
                )
            fig.suptitle(f"{split_label}: relative energy error by true energy")
            fig.tight_layout()
            pdf.savefig(fig, name=f"energy_relative_error_bins_{start // rows_per_page:02d}")
            plt.close(fig)

    return summary


def _save_reconstruction_pdf(
    output_path: Path,
    split_name: str,
    pred: np.ndarray,
    target: np.ndarray,
    energy_bin_width: float,
    min_bin_count: int,
    quality: np.ndarray | None = None,
    predicted_errors: np.ndarray | None = None,
) -> tuple[Path, dict[str, Any]]:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    q = _prediction_quantities(pred, target)
    quality_values = _quality_for_finite_pair(pred, target, quality)
    predicted_error_values = _predicted_errors_for_finite_pair(pred, target, predicted_errors)
    bin_rows = _energy_bin_table(q["true_log10_energy"], q["rel_energy"], energy_bin_width)
    resolution_rows = _resolution_bin_table(
        q["true_log10_energy"],
        q["opening_deg"],
        q["core_xy_km"],
        energy_bin_width,
    )
    summary: dict[str, Any] = {
        "n": int(q["target"].shape[0]),
        "opening_angle_68_deg": _to_float(_percentile(q["opening_deg"], 68.0)),
        "core_dx_central68_km": _to_float(_central68_half_width(q["core_dx_km"])),
        "core_dy_central68_km": _to_float(_central68_half_width(q["core_dy_km"])),
        "core_xy_68_km": _to_float(_percentile(q["core_xy_km"], 68.0)),
        "energy_relative_error": _fit_gaussian_hist(q["rel_energy"]),
        "error_correlations": _error_correlation_summary(q),
        "energy_bins": bin_rows,
        "resolution_bins": resolution_rows,
    }
    if predicted_error_values is not None:
        predicted_variables = _predicted_error_correlation_inputs(predicted_error_values)
        calibration_variables = _predicted_actual_error_inputs(q, predicted_error_values)
        summary["predicted_error_correlations"] = _error_correlation_summary_from_variables(predicted_variables)
        summary["predicted_actual_error"] = _predicted_actual_error_summary(calibration_variables)
    if quality_values is not None:
        quality_cut_rows = _quality_cut_rows(q, quality_values, min_bin_count)
        quality_energy_dependence = [
            _quality_energy_rows(q, quality_values, keep_fraction=fraction, energy_bin_width=energy_bin_width)
            for fraction in QUALITY_ENERGY_KEEP_FRACTIONS
        ]
        summary["quality"] = {
            "n": int(np.sum(np.isfinite(quality_values))),
            "mean": _to_float(np.nanmean(quality_values)),
            "median": _to_float(np.nanmedian(quality_values)),
            "thresholds": _quality_threshold_summary(quality_values),
            "cut_rows": quality_cut_rows,
            "energy_dependence": quality_energy_dependence,
        }

    split_dir = _diagnostics_root(output_path) / split_name
    pdf_files: list[str] = []
    page_names = [
        "opening_angle",
        "core_coordinate_residuals",
        "core_position_error",
        "energy_relative_error",
        "energy_resolution_by_true_energy",
        "angular_core_resolution_by_true_energy",
        "actual_error_correlations",
    ]
    if predicted_error_values is not None:
        page_names.extend(["predicted_error_correlations", "predicted_vs_actual_error"])
    if quality_values is not None:
        page_names.extend(["quality_histograms", "quality_cut_performance", "quality_energy_dependence"])
    with _SplitPdfWriter(split_dir, page_names, pdf_files) as pdf:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.hist(q["opening_deg"], bins=_hist_bins(q["opening_deg"]), histtype="stepfilled", color="#4c78a8", alpha=0.4, edgecolor="#4c78a8")
        open68 = summary["opening_angle_68_deg"]
        if open68 is not None:
            ax.axvline(open68, color="#d62728", linewidth=1.8, label=f"68\\% = {open68:.3g} deg")
            ax.legend(frameon=False)
        ax.set_title(f"{split_name}: opening angle")
        ax.set_xlabel("opening angle [deg]")
        ax.set_ylabel("events")
        _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
        for ax, values, label, color in [
            (axes[0], q["core_dx_km"], "core x residual [km]", "#4c78a8"),
            (axes[1], q["core_dy_km"], "core y residual [km]", "#f58518"),
        ]:
            ax.hist(values, bins=_hist_bins(values), histtype="stepfilled", color=color, alpha=0.4, edgecolor=color)
            c68 = _central68_half_width(values)
            if math.isfinite(c68):
                ax.axvline(c68, color="#2ca02c", linestyle="--", linewidth=1.2)
                ax.axvline(-c68, color="#2ca02c", linestyle="--", linewidth=1.2)
                ax.text(0.98, 0.96, f"central 68\\%={c68:.3g} km", ha="right", va="top", transform=ax.transAxes, fontsize=9)
            ax.axvline(0.0, color="0.25", linewidth=1.0)
            ax.set_xlabel(label)
            _style_axes(ax)
        axes[0].set_ylabel("events")
        fig.suptitle(f"{split_name}: core coordinate residuals")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.hist(q["core_xy_km"], bins=_hist_bins(q["core_xy_km"]), histtype="stepfilled", color="#54a24b", alpha=0.4, edgecolor="#54a24b")
        core68 = summary["core_xy_68_km"]
        if core68 is not None:
            ax.axvline(core68, color="#d62728", linewidth=1.8, label=f"68\\% = {core68:.3g} km")
            ax.legend(frameon=False)
        ax.set_title(f"{split_name}: core position error")
        ax.set_xlabel(r"$\sqrt{\Delta x^2 + \Delta y^2}$ [km]")
        ax.set_ylabel("events")
        _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        summary["energy_relative_error"] = _draw_gaussian_hist(
            ax,
            q["rel_energy"],
            f"{split_name}: relative energy error",
            r"$(E_{\mathrm{rec}}-E_{\mathrm{true}})/E_{\mathrm{true}}$",
            color="#b279a2",
            show_central68=False,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        valid_rows = _valid_energy_fit_rows(bin_rows, min_bin_count)
        display_rows = [row for row in bin_rows if int(row["n"] or 0) >= int(min_bin_count)]
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        if valid_rows:
            x = np.asarray([row["log10_energy_center"] for row in valid_rows], dtype=float)
            mu = np.asarray([row["mu"] for row in valid_rows], dtype=float)
            sigma = np.asarray([row["sigma"] for row in valid_rows], dtype=float)
            ax.errorbar(x, mu, yerr=sigma, fmt="o", color="#d62728", capsize=2.5, label=r"Gaussian $\mu\pm\sigma$")
            _add_energy_bias_guides(ax)
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{split_name}: energy resolution by true energy")
        ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        ax.set_ylabel("relative energy error")
        _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        valid_resolution_rows = [row for row in resolution_rows if int(row["n"] or 0) >= int(min_bin_count)]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
        if valid_resolution_rows:
            x = np.asarray([row["log10_energy_center"] for row in valid_resolution_rows], dtype=float)
            opening68 = np.asarray([row["opening_angle_68_deg"] for row in valid_resolution_rows], dtype=float)
            opening68_err = np.asarray([row["opening_angle_68_deg_err"] for row in valid_resolution_rows], dtype=float)
            opening50 = np.asarray([row["opening_angle_median_deg"] for row in valid_resolution_rows], dtype=float)
            opening50_err = np.asarray([row["opening_angle_median_deg_err"] for row in valid_resolution_rows], dtype=float)
            core68 = np.asarray([row["core_xy_68_km"] for row in valid_resolution_rows], dtype=float)
            core68_err = np.asarray([row["core_xy_68_km_err"] for row in valid_resolution_rows], dtype=float)
            core50 = np.asarray([row["core_xy_median_km"] for row in valid_resolution_rows], dtype=float)
            core50_err = np.asarray([row["core_xy_median_km_err"] for row in valid_resolution_rows], dtype=float)
            axes[0].errorbar(x, opening68, yerr=opening68_err, fmt="o-", color="#4c78a8", linewidth=1.4, markersize=4, capsize=2.5, label=r"68\%")
            axes[0].errorbar(x, opening50, yerr=opening50_err, fmt="s--", color="#72b7b2", linewidth=1.2, markersize=3.5, capsize=2.5, label="median")
            axes[1].errorbar(x, core68, yerr=core68_err, fmt="o-", color="#54a24b", linewidth=1.4, markersize=4, capsize=2.5, label=r"68\%")
            axes[1].errorbar(x, core50, yerr=core50_err, fmt="s--", color="#b79a20", linewidth=1.2, markersize=3.5, capsize=2.5, label="median")
            _add_angular_target(axes[0])
            _add_core_target(axes[1], unit="km")
            for ax in axes:
                ax.legend(frameon=False)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
        axes[0].set_title(f"{split_name}: angular resolution by true energy")
        axes[0].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        axes[0].set_ylabel("opening angle [deg]")
        axes[1].set_title(f"{split_name}: core resolution by true energy")
        axes[1].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
        axes[1].set_ylabel("core position error [km]")
        for ax in axes:
            _style_axes(ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = _save_error_correlation_scatter(
            _actual_error_correlation_inputs(q),
            summary["error_correlations"],
            f"{split_name}: actual reconstruction error correlations",
        )
        pdf.savefig(fig)
        plt.close(fig)

        if predicted_error_values is not None:
            predicted_variables = _predicted_error_correlation_inputs(predicted_error_values)
            fig = _save_error_correlation_scatter(
                predicted_variables,
                summary["predicted_error_correlations"],
                f"{split_name}: predicted error correlations",
            )
            pdf.savefig(fig)
            plt.close(fig)

            calibration_variables = _predicted_actual_error_inputs(q, predicted_error_values)
            fig = _save_predicted_actual_error_scatter(
                calibration_variables,
                summary["predicted_actual_error"],
                f"{split_name}: predicted vs actual error",
            )
            pdf.savefig(fig)
            plt.close(fig)

        if quality_values is not None:
            finite_quality = quality_values[np.isfinite(quality_values)]
            fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
            ax = axes[0]
            ax.hist(finite_quality, bins=np.linspace(0.0, 1.0, 51), histtype="stepfilled", color="#4c78a8", alpha=0.4, edgecolor="#4c78a8")
            for row in summary["quality"]["thresholds"]:
                keep_fraction = float(row["keep_fraction"])
                if keep_fraction in QUALITY_MARKER_KEEP_FRACTIONS:
                    threshold = float(row["quality_threshold"])
                    ax.axvline(threshold, color=NEUTRAL_COLOR, linestyle="--", linewidth=LINEWIDTH_THIN)
                    ax.text(threshold, 0.96, f"{100.0 * keep_fraction:.0f}\\%", rotation=90, ha="right", va="top", transform=ax.get_xaxis_transform())
            ax.set_title(f"{split_name}: quality score")
            ax.set_xlabel("quality")
            ax.set_ylabel("events")
            _style_axes(ax)

            ax = axes[1]
            thresholds = np.linspace(0.0, 1.0, 201)
            survival = np.asarray([np.mean(finite_quality >= threshold) for threshold in thresholds], dtype=float)
            ax.plot(thresholds, survival, color="#54a24b", linewidth=LINEWIDTH)
            for row in summary["quality"]["thresholds"]:
                keep_fraction = float(row["keep_fraction"])
                if keep_fraction in QUALITY_MARKER_KEEP_FRACTIONS:
                    threshold = float(row["quality_threshold"])
                    ax.plot([threshold], [keep_fraction], marker="o", color=PROTON_COLOR, markersize=MARKERSIZE)
                    ax.text(threshold, keep_fraction, f" {100.0 * keep_fraction:.0f}\\%", va="center")
            ax.set_title(f"{split_name}: cumulative survival")
            ax.set_xlabel("quality threshold")
            ax.set_ylabel("survival fraction")
            ax.set_ylim(-0.02, 1.02)
            _style_axes(ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            valid_quality_rows = [
                row for row in summary["quality"]["cut_rows"] if int(row["n"] or 0) >= int(min_bin_count)
            ]
            fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
            if valid_quality_rows:
                x = _rows_array(valid_quality_rows, "survival_fraction")
                opening = _rows_array(valid_quality_rows, "opening_angle_68_deg")
                core = _rows_array(valid_quality_rows, "core_xy_68_km") * 1000.0
                energy_sigma = _rows_array(valid_quality_rows, "energy_sigma")
                energy_c68 = _rows_array(valid_quality_rows, "energy_central68")
                energy = np.where(np.isfinite(energy_sigma), energy_sigma, energy_c68)
                axes[0].plot(x, opening, marker="o", color="#4c78a8", linewidth=LINEWIDTH, markersize=MARKERSIZE)
                axes[1].plot(x, core, marker="o", color="#54a24b", linewidth=LINEWIDTH, markersize=MARKERSIZE)
                axes[2].plot(x, energy, marker="o", color="#b279a2", linewidth=LINEWIDTH, markersize=MARKERSIZE)
                _add_angular_target(axes[0])
                _add_core_target(axes[1], unit="m")
                _add_energy_sigma_guides(axes[2])
                for ax in axes:
                    ax.set_xlim(1.02, max(0.0, float(np.nanmin(x)) - 0.02))
            else:
                for ax in axes:
                    ax.text(0.5, 0.5, "no quality cut has enough entries", ha="center", va="center", transform=ax.transAxes)
            axes[0].set_title(f"{split_name}: angular vs quality cut")
            axes[0].set_ylabel("opening angle 68\\% [deg]")
            axes[1].set_title(f"{split_name}: core vs quality cut")
            axes[1].set_ylabel("core 68\\% [m]")
            axes[2].set_title(f"{split_name}: energy vs quality cut")
            axes[2].set_ylabel("energy resolution")
            for ax in axes:
                ax.set_xlabel("survival fraction")
                _style_axes(ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True)
            plotted = [False, False, False]
            for item in summary["quality"]["energy_dependence"]:
                keep_fraction = float(item["keep_fraction"])
                label = f"top {100.0 * keep_fraction:.0f}\\%"
                resolution_valid = [
                    row for row in item["resolution_rows"] if int(row["n"] or 0) >= int(min_bin_count)
                ]
                energy_valid = _valid_energy_fit_rows(item["energy_rows"], min_bin_count)
                if resolution_valid:
                    x = _rows_array(resolution_valid, "log10_energy_center")
                    axes[0].plot(x, _rows_array(resolution_valid, "opening_angle_68_deg"), marker="o", linewidth=LINEWIDTH_THIN, markersize=MARKERSIZE, label=label)
                    axes[1].plot(x, _rows_array(resolution_valid, "core_xy_68_km") * 1000.0, marker="o", linewidth=LINEWIDTH_THIN, markersize=MARKERSIZE, label=label)
                    plotted[0] = True
                    plotted[1] = True
                if energy_valid:
                    x = _rows_array(energy_valid, "log10_energy_center")
                    axes[2].plot(x, _rows_array(energy_valid, "sigma"), marker="o", linewidth=LINEWIDTH_THIN, markersize=MARKERSIZE, label=label)
                    plotted[2] = True
            axes[0].set_title(f"{split_name}: angular resolution by true energy and quality")
            axes[0].set_ylabel("opening angle 68\\% [deg]")
            axes[1].set_title(f"{split_name}: core resolution by true energy and quality")
            axes[1].set_ylabel("core 68\\% [m]")
            axes[2].set_title(f"{split_name}: energy resolution by true energy and quality")
            axes[2].set_ylabel("Gaussian sigma")
            axes[2].set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
            _add_angular_target(axes[0])
            _add_core_target(axes[1], unit="m")
            _add_energy_sigma_guides(axes[2])
            for index, ax in enumerate(axes):
                if plotted[index]:
                    ax.legend(frameon=False)
                else:
                    ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
                _style_axes(ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        rows_per_page = 6
        for start in range(0, len(display_rows), rows_per_page):
            rows = display_rows[start : start + rows_per_page]
            fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
            axes_flat = axes.ravel()
            for ax in axes_flat:
                ax.axis("off")
            for ax, row in zip(axes_flat, rows):
                ax.axis("on")
                low = float(row["log10_energy_low"])
                high = float(row["log10_energy_high"])
                if high == bin_rows[-1]["log10_energy_high"]:
                    mask = (q["true_log10_energy"] >= low) & (q["true_log10_energy"] <= high)
                else:
                    mask = (q["true_log10_energy"] >= low) & (q["true_log10_energy"] < high)
                values = q["rel_energy"][mask]
                _draw_gaussian_hist(
                    ax,
                    values,
                    f"{low:.1f} <= logE < {high:.1f}",
                    r"$(E_{\mathrm{rec}}-E_{\mathrm{true}})/E_{\mathrm{true}}$",
                    color="#b279a2",
                    show_central68=False,
                )
            fig.suptitle(f"{split_name}: relative energy error by true energy")
            fig.tight_layout()
            pdf.savefig(fig, name=f"energy_relative_error_bins_{start // rows_per_page:02d}")
            plt.close(fig)

    summary["directory"] = str(split_dir)
    summary["pdfs"] = pdf_files
    return split_dir, summary


def save_training_diagnostics(
    output_path: str | Path,
    history: list[dict[str, Any]],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    validation_mass: tuple[np.ndarray, np.ndarray] | None = None,
    test_mass: tuple[np.ndarray, np.ndarray] | None = None,
    validation_particle_labels: np.ndarray | None = None,
    test_particle_labels: np.ndarray | None = None,
    validation_quality: np.ndarray | None = None,
    test_quality: np.ndarray | None = None,
    validation_predicted_errors: np.ndarray | None = None,
    test_predicted_errors: np.ndarray | None = None,
    energy_bin_width: float = 0.1,
    min_bin_count: int = 20,
    save_reconstruction: bool = True,
) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = _diagnostics_root(output)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    learning_curve = _save_learning_curve(output, history)
    prediction_cache = save_prediction_cache(
        output,
        validation=validation,
        test=test,
        validation_mass=validation_mass,
        test_mass=test_mass,
        validation_quality=validation_quality,
        test_quality=test_quality,
        validation_predicted_errors=validation_predicted_errors,
        test_predicted_errors=test_predicted_errors,
    )
    diagnostics: dict[str, Any] = {
        "directory": str(diagnostics_dir),
        "learning_curve_pdf": str(learning_curve),
        "prediction_cache": str(prediction_cache),
    }
    if save_reconstruction:
        for split_name, pair, quality, predicted_errors in [
            ("validation", validation, validation_quality, validation_predicted_errors),
            ("test", test, test_quality, test_predicted_errors),
        ]:
            _pdf, summary = _save_reconstruction_pdf(
                output,
                split_name,
                pred=pair[0],
                target=pair[1],
                energy_bin_width=energy_bin_width,
                min_bin_count=min_bin_count,
                quality=quality,
                predicted_errors=predicted_errors,
            )
            diagnostics[split_name] = summary
        for split_name, labels, pair in [
            ("validation", validation_particle_labels, validation),
            ("test", test_particle_labels, test),
        ]:
            if labels is None:
                continue
            summary = _save_species_reconstruction_outputs(
                output,
                split_name,
                pred=pair[0],
                target=pair[1],
                labels=labels,
                energy_bin_width=energy_bin_width,
                min_bin_count=min_bin_count,
            )
            if summary is not None:
                diagnostics[f"{split_name}_species"] = summary
    mass_threshold = 0.5
    tuned_mass_threshold = (
        balanced_accuracy_threshold(validation_mass[0], validation_mass[1]) if validation_mass is not None else None
    )
    for split_name, mass_pair, reco_pair in [
        ("validation", validation_mass, validation),
        ("test", test_mass, test),
    ]:
        if mass_pair is None:
            continue
        _pdf, summary = _save_mass_pdf(
            output,
            split_name,
            logits=mass_pair[0],
            labels=mass_pair[1],
            target=reco_pair[1],
            energy_bin_width=energy_bin_width,
            min_bin_count=min_bin_count,
            threshold=mass_threshold,
            tuned_threshold=tuned_mass_threshold,
        )
        diagnostics[f"{split_name}_mass"] = summary

    summary_path = diagnostics_dir / "summary.json"
    diagnostics["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics
