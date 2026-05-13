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
    angular = angular_error_deg(pred, target)
    return {
        "rmse_log10_energy": float(np.sqrt(np.mean(energy_delta**2))),
        "median_abs_log10_energy": float(np.median(np.abs(energy_delta))),
        "core_rmse_km": float(np.sqrt(np.mean(np.sum(core_delta**2, axis=1)))),
        "core_median_km": float(np.median(np.linalg.norm(core_delta, axis=1))),
        "angular_median_deg": float(np.median(angular)),
        "angular_68_deg": float(np.percentile(angular, 68.0)),
    }


def direction_to_angles(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = direction / np.clip(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-12, None)
    zenith = np.degrees(np.arccos(np.clip(unit[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(unit[:, 1], unit[:, 0])) % 360.0
    return zenith, azimuth
