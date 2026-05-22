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

from .metrics import binary_classification_metrics
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


def _finite_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(target), axis=1)
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


def _save_learning_curve(output_path: Path, history: list[dict[str, Any]]) -> Path:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    path = _diagnostics_root(output_path) / "learning_curve.pdf"
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    has_reconstruction = "train_reconstruction_loss" in history[0] if history else False
    has_mass = "train_mass_loss" in history[0] if history else False

    if has_mass:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        axes = [ax]
    ax.plot(epochs, train, marker="o", markersize=2.5, linewidth=1.4, label="train")
    ax.plot(epochs, val, marker="s", markersize=2.5, linewidth=1.4, label="validation")
    if has_reconstruction:
        ax.plot(
            epochs,
            [row["train_reconstruction_loss"] for row in history],
            linestyle="--",
            linewidth=1.0,
            label="train reconstruction",
        )
        ax.plot(
            epochs,
            [row["val_reconstruction_loss"] for row in history],
            linestyle=":",
            linewidth=1.2,
            label="validation reconstruction",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _style_axes(ax)
    if has_mass:
        ax = axes[1]
        ax.plot(epochs, [row["train_mass_loss"] for row in history], marker="o", markersize=2.5, linewidth=1.4, label="train")
        ax.plot(epochs, [row["val_mass_loss"] for row in history], marker="s", markersize=2.5, linewidth=1.4, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.set_yscale("log")
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
) -> list[dict[str, Any]]:
    edges = _energy_bin_edges(true_log10_energy, bin_width)
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    predictions = scores >= 0.5
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
    pdf_files.append(_save_pdf(fig, split_dir / "species_error_histograms.pdf"))
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
        ax.axhline(0.0, color="0.25", linewidth=1.0)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{split_name}: energy resolution by true energy and species")
    ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("relative energy error")
    _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "04_species_energy_resolution_by_true_energy.pdf"))
    pdf_files.append(_save_pdf(fig, split_dir / "species_energy_resolution_by_true_energy.pdf"))
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
    axes[0].axhline(0.0, color="0.25", linewidth=1.0)
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
    pdf_files.append(_save_pdf(fig, split_dir / "species_energy_fit_parameters_by_true_energy.pdf"))
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
    for ax in axes:
        ax.legend(frameon=False)
        _style_axes(ax)
    fig.tight_layout()
    pdf_files.append(_save_pdf(fig, split_dir / "05_species_angular_core_resolution_by_true_energy.pdf"))
    pdf_files.append(_save_pdf(fig, split_dir / "species_angular_core_resolution_by_true_energy.pdf"))
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
    predictions = scores >= 0.5
    metrics = binary_classification_metrics(logits, labels)
    energy_rows = _mass_energy_bin_table(target[:, 0], scores, labels, energy_bin_width)
    summary: dict[str, Any] = {
        **metrics,
        "energy_bins": energy_rows,
    }

    split_dir = _diagnostics_root(output_path) / split_name
    pdf_path = split_dir / "mass_classification.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    ax = axes[0, 0]
    matrix = np.asarray(
        [
            [metrics["tn_proton"], metrics["fp_iron"]],
            [metrics["fn_iron"], metrics["tp_iron"]],
        ],
        dtype=float,
    )
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred proton", "pred iron"])
    ax.set_yticks([0, 1], labels=["true proton", "true iron"])
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{int(value)}", ha="center", va="center", color="black")
    ax.set_title(f"{split_name}: mass confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    proton_scores = scores[~truth]
    iron_scores = scores[truth]
    bins = np.linspace(0.0, 1.0, 41)
    if proton_scores.size:
        ax.hist(proton_scores, bins=bins, histtype="step", linewidth=1.6, label="true proton", color="#4c78a8")
    if iron_scores.size:
        ax.hist(iron_scores, bins=bins, histtype="step", linewidth=1.6, label="true iron", color="#d62728")
    ax.axvline(0.5, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_title(f"{split_name}: predicted iron probability")
    ax.set_xlabel("P(iron)")
    ax.set_ylabel("events")
    ax.legend(frameon=False)
    _style_axes(ax)

    ax = axes[1, 0]
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

    ax = axes[1, 1]
    valid_rows = [row for row in energy_rows if int(row["n"] or 0) >= int(min_bin_count)]
    if valid_rows:
        x = np.asarray([row["log10_energy_center"] for row in valid_rows], dtype=float)
        ax.plot(x, [row["accuracy"] for row in valid_rows], "o-", color="#4c78a8", label="all")
        proton = np.asarray([np.nan if row["proton_accuracy"] is None else row["proton_accuracy"] for row in valid_rows])
        iron = np.asarray([np.nan if row["iron_accuracy"] is None else row["iron_accuracy"] for row in valid_rows])
        ax.plot(x, proton, "s--", color="#72b7b2", label="proton")
        ax.plot(x, iron, "^--", color="#d62728", label="iron")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "no energy bin has enough entries", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{split_name}: mass accuracy by true energy")
    ax.set_xlabel(r"true $\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("accuracy")
    _style_axes(ax)

    fig.tight_layout()
    _save_pdf(fig, pdf_path)
    plt.close(fig)

    summary["directory"] = str(split_dir)
    summary["pdfs"] = [str(pdf_path)]
    return split_dir, summary


def _save_reconstruction_pdf(
    output_path: Path,
    split_name: str,
    pred: np.ndarray,
    target: np.ndarray,
    energy_bin_width: float,
    min_bin_count: int,
) -> tuple[Path, dict[str, Any]]:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    q = _prediction_quantities(pred, target)
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
        "energy_bins": bin_rows,
        "resolution_bins": resolution_rows,
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
    ]
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
            ax.axhline(0.0, color="0.25", linewidth=1.0)
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
    energy_bin_width: float = 0.1,
    min_bin_count: int = 20,
    save_reconstruction: bool = True,
) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = _diagnostics_root(output)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    learning_curve = _save_learning_curve(output, history)
    diagnostics: dict[str, Any] = {"directory": str(diagnostics_dir), "learning_curve_pdf": str(learning_curve)}
    if save_reconstruction:
        for split_name, pair in [("validation", validation), ("test", test)]:
            _pdf, summary = _save_reconstruction_pdf(
                output,
                split_name,
                pred=pair[0],
                target=pair[1],
                energy_bin_width=energy_bin_width,
                min_bin_count=min_bin_count,
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
        )
        diagnostics[f"{split_name}_mass"] = summary

    summary_path = diagnostics_dir / "summary.json"
    diagnostics["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics
