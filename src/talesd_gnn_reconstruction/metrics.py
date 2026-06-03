from __future__ import annotations

import numpy as np


def direction_columns_for_dim(target_dim: int) -> slice:
    target_dim = int(target_dim)
    if target_dim >= 7:
        return slice(4, 7)
    if target_dim >= 6:
        return slice(3, 6)
    raise ValueError("reconstruction targets must contain logE, core_x, core_y, dir_x, dir_y, dir_z")


def normalize_directions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    direction = values[:, direction_columns_for_dim(values.shape[1])].astype(np.float64)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.clip(norm, 1.0e-12, None)


def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_dir = normalize_directions(pred)
    target_dir = normalize_directions(target)
    dot = np.sum(pred_dir * target_dir, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def reconstruction_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    energy_delta = pred[:, 0] - target[:, 0]
    core_xy_delta = pred[:, 1:3] - target[:, 1:3]
    core_delta = core_xy_delta
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


def _finite_binary_inputs(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    mask = np.isfinite(logits) & np.isfinite(labels)
    return logits[mask], labels[mask]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def balanced_accuracy_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
    logits, labels = _finite_binary_inputs(logits, labels)
    if labels.size == 0:
        return 0.5
    y = labels >= 0.5
    true_pos = int(np.sum(y))
    true_neg = int(np.sum(~y))
    if true_pos == 0 or true_neg == 0:
        return 0.5

    probs = _sigmoid(logits)
    order = np.argsort(-probs, kind="mergesort")
    sorted_probs = probs[order]
    sorted_y = y[order]
    ends = np.flatnonzero(np.concatenate((sorted_probs[1:] != sorted_probs[:-1], [True])))

    tp = np.cumsum(sorted_y, dtype=np.float64)[ends]
    fp = np.cumsum(~sorted_y, dtype=np.float64)[ends]
    tpr = tp / float(true_pos)
    tnr = (float(true_neg) - fp) / float(true_neg)
    balanced = 0.5 * (tpr + tnr)

    thresholds = sorted_probs[ends]
    candidate_thresholds = np.concatenate(([np.nextafter(sorted_probs[0], np.inf)], thresholds))
    candidate_balanced = np.concatenate(([0.5], balanced))
    best_value = float(np.max(candidate_balanced))
    best_indices = np.flatnonzero(np.isclose(candidate_balanced, best_value, rtol=0.0, atol=1.0e-12))
    best_index = int(best_indices[np.argmin(np.abs(candidate_thresholds[best_indices] - 0.5))])
    return float(candidate_thresholds[best_index])


def binary_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    logits, labels = _finite_binary_inputs(logits, labels)
    if labels.size == 0:
        return {
            "n": 0,
            "threshold": float(threshold),
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
    probs = _sigmoid(logits)
    pred = probs >= float(threshold)
    tp = int(np.sum(pred & y))
    tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    true_pos = int(np.sum(y))
    true_neg = int(np.sum(~y))
    tpr = tp / true_pos if true_pos else float("nan")
    tnr = tn / true_neg if true_neg else float("nan")
    if true_pos and true_neg:
        ranks = _average_ranks(probs)
        auc = (float(np.sum(ranks[y])) - true_pos * (true_pos + 1) / 2.0) / (true_pos * true_neg)
    else:
        auc = float("nan")
    return {
        "n": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / labels.size),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "auc": float(auc),
        "score_median_proton": float(np.median(probs[~y])) if true_neg else float("nan"),
        "score_median_iron": float(np.median(probs[y])) if true_pos else float("nan"),
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
