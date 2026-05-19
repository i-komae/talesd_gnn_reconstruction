from __future__ import annotations

import numpy as np


def normalize_directions(values: np.ndarray) -> np.ndarray:
    direction = values[:, 4:7].astype(np.float64)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.clip(norm, 1.0e-12, None)


def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_dir = normalize_directions(pred)
    target_dir = normalize_directions(target)
    dot = np.sum(pred_dir * target_dir, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def reconstruction_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    energy_delta = pred[:, 0] - target[:, 0]
    core_delta = pred[:, 1:4] - target[:, 1:4]
    core_xy_delta = pred[:, 1:3] - target[:, 1:3]
    rel_energy_delta = np.power(10.0, energy_delta) - 1.0
    rel_energy_q16, rel_energy_q84 = np.percentile(rel_energy_delta, [16.0, 84.0])
    angular = angular_error_deg(pred, target)
    return {
        "rmse_log10_energy": float(np.sqrt(np.mean(energy_delta**2))),
        "median_abs_log10_energy": float(np.median(np.abs(energy_delta))),
        "median_relative_energy": float(np.median(rel_energy_delta)),
        "mean_relative_energy": float(np.mean(rel_energy_delta)),
        "median_abs_relative_energy": float(np.median(np.abs(rel_energy_delta))),
        "abs_relative_energy_68": float(np.percentile(np.abs(rel_energy_delta), 68.0)),
        "relative_energy_q16": float(rel_energy_q16),
        "relative_energy_q84": float(rel_energy_q84),
        "relative_energy_central68_width": float(rel_energy_q84 - rel_energy_q16),
        "relative_energy_central68_half_width": float(0.5 * (rel_energy_q84 - rel_energy_q16)),
        "core_rmse_km": float(np.sqrt(np.mean(np.sum(core_delta**2, axis=1)))),
        "core_median_km": float(np.median(np.linalg.norm(core_delta, axis=1))),
        "core_68_km": float(np.percentile(np.linalg.norm(core_delta, axis=1), 68.0)),
        "core_xy_rmse_km": float(np.sqrt(np.mean(np.sum(core_xy_delta**2, axis=1)))),
        "core_xy_median_km": float(np.median(np.linalg.norm(core_xy_delta, axis=1))),
        "core_xy_68_km": float(np.percentile(np.linalg.norm(core_xy_delta, axis=1), 68.0)),
        "angular_median_deg": float(np.median(angular)),
        "angular_68_deg": float(np.percentile(angular, 68.0)),
    }


def binary_classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    mask = np.isfinite(logits) & np.isfinite(labels)
    logits = logits[mask]
    labels = labels[mask]
    if labels.size == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "auc": float("nan"),
            "true_proton": 0,
            "true_iron": 0,
            "pred_proton": 0,
            "pred_iron": 0,
            "tp_iron": 0,
            "tn_proton": 0,
            "fp_iron": 0,
            "fn_iron": 0,
        }
    y = labels >= 0.5
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    pred = probs >= 0.5
    tp = int(np.sum(pred & y))
    tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    true_pos = int(np.sum(y))
    true_neg = int(np.sum(~y))
    tpr = tp / true_pos if true_pos else float("nan")
    tnr = tn / true_neg if true_neg else float("nan")
    if true_pos and true_neg:
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, probs.size + 1, dtype=np.float64)
        auc = (float(np.sum(ranks[y])) - true_pos * (true_pos + 1) / 2.0) / (true_pos * true_neg)
    else:
        auc = float("nan")
    return {
        "n": int(labels.size),
        "accuracy": float((tp + tn) / labels.size),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "auc": float(auc),
        "true_proton": true_neg,
        "true_iron": true_pos,
        "pred_proton": int(np.sum(~pred)),
        "pred_iron": int(np.sum(pred)),
        "tp_iron": tp,
        "tn_proton": tn,
        "fp_iron": fp,
        "fn_iron": fn,
    }


def direction_to_angles(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-12, None)
    zenith = np.degrees(np.arccos(np.clip(unit[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(unit[:, 1], unit[:, 0])) % 360.0
    return zenith, azimuth
