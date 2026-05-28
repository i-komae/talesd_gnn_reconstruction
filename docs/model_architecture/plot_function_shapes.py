from pathlib import Path
import os
import tempfile

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "talesd_gnn_matplotlib"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "cm"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def smooth_l1(r: np.ndarray, beta: float) -> np.ndarray:
    return np.where(np.abs(r) < beta, 0.5 * r * r / beta, np.abs(r) - 0.5 * beta)


def bce_with_logits(z: np.ndarray, y: float) -> np.ndarray:
    # Numerically stable form of -y log(sigmoid(z)) - (1-y) log(1-sigmoid(z)).
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


def main() -> None:
    out = Path(__file__).resolve().parent / "fig" / "function_shapes.pdf"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    z = np.linspace(-8.0, 8.0, 800)
    axes[0, 0].plot(z, sigmoid(z), color="#1f77b4", lw=2.2)
    axes[0, 0].axhline(0.5, color="0.55", lw=0.8, ls="--")
    axes[0, 0].axvline(0.0, color="0.55", lw=0.8, ls="--")
    axes[0, 0].set_title(r"$\sigma(z)=1/(1+\exp(-z))$")
    axes[0, 0].set_xlabel(r"$z$")
    axes[0, 0].set_ylabel(r"$\sigma(z)$")
    axes[0, 0].set_ylim(-0.05, 1.05)

    p = np.linspace(1e-3, 1.0 - 1e-3, 800)
    axes[0, 1].plot(p, logit(p), color="#d62728", lw=2.2)
    axes[0, 1].axhline(0.0, color="0.55", lw=0.8, ls="--")
    axes[0, 1].axvline(0.5, color="0.55", lw=0.8, ls="--")
    axes[0, 1].set_title(r"$\mathrm{logit}(p)=\log[p/(1-p)]$")
    axes[0, 1].set_xlabel(r"$p$")
    axes[0, 1].set_ylabel(r"$\mathrm{logit}(p)$")
    axes[0, 1].set_ylim(-8.0, 8.0)

    r = np.linspace(-3.0, 3.0, 800)
    axes[1, 0].plot(r, smooth_l1(r, beta=1.0), color="#2ca02c", lw=2.2, label=r"$\mathrm{SmoothL1}_{\beta=1}$")
    axes[1, 0].plot(r, 0.5 * r * r, color="0.45", lw=1.0, ls="--", label=r"$0.5r^2$")
    axes[1, 0].plot(r, np.abs(r) - 0.5, color="0.25", lw=1.0, ls=":", label=r"$|r|-0.5$")
    axes[1, 0].axvline(-1.0, color="0.75", lw=0.8)
    axes[1, 0].axvline(1.0, color="0.75", lw=0.8)
    axes[1, 0].set_title(r"$\mathrm{SmoothL1}_{\beta=1}(r)$")
    axes[1, 0].set_xlabel(r"$r$")
    axes[1, 0].set_ylabel(r"$L$")
    axes[1, 0].set_ylim(-0.05, 2.6)
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(z, bce_with_logits(z, y=1.0), color="#9467bd", lw=2.2, label=r"$y=1$")
    axes[1, 1].plot(z, bce_with_logits(z, y=0.0), color="#ff7f0e", lw=2.2, label=r"$y=0$")
    axes[1, 1].axvline(0.0, color="0.55", lw=0.8, ls="--")
    axes[1, 1].set_title(r"$\mathrm{BCEWithLogits}(z,y)$")
    axes[1, 1].set_xlabel(r"$z$")
    axes[1, 1].set_ylabel(r"$L$")
    axes[1, 1].set_ylim(-0.1, 8.2)
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, color="0.88", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
