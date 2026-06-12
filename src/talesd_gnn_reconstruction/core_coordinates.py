from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np


CORE_TARGET_MODES = ("absolute", "signal_bary_relative", "fit_core_relative")
COORDINATE_FEATURE_MODES = ("absolute_and_relative", "relative_only", "absolute_only")
CORE_ANCHOR_COLUMNS = ("core_anchor_x_km", "core_anchor_y_km")
RAW_TARGET_COLUMNS = ("log10_energy_eV", "core_x_km", "core_y_km", "dir_x", "dir_y", "dir_z")

ABSOLUTE_COORDINATE_COLUMNS = {
    "x_km",
    "y_km",
    "z_km",
    "detector_x_km",
    "detector_y_km",
    "detector_z_km",
    "pulse_x_km",
    "pulse_y_km",
    "pulse_z_km",
}
RELATIVE_COORDINATE_MARKERS = (
    "dx_",
    "dy_",
    "dz_",
    "dr_",
    "r_from_",
    "_rel_x",
    "_rel_y",
    "_rel_z",
)


def normalize_core_target_mode(value: str | None) -> str:
    mode = str(value or "absolute").strip().lower()
    if mode not in CORE_TARGET_MODES:
        raise ValueError(f"core_target_mode must be one of {CORE_TARGET_MODES}, got {value!r}")
    return mode


def normalize_coordinate_feature_mode(value: str | None) -> str:
    mode = str(value or "absolute_and_relative").strip().lower()
    if mode not in COORDINATE_FEATURE_MODES:
        raise ValueError(f"coordinate_feature_mode must be one of {COORDINATE_FEATURE_MODES}, got {value!r}")
    return mode


def parse_columns_json(value: str | bytes | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return dict(value)


def target_columns_for_mode(core_target_mode: str) -> tuple[str, ...]:
    mode = normalize_core_target_mode(core_target_mode)
    if mode == "absolute":
        return RAW_TARGET_COLUMNS
    return (
        "log10_energy_eV",
        "delta_core_x_km",
        "delta_core_y_km",
        "dir_x",
        "dir_y",
        "dir_z",
    )


def _is_absolute_coordinate_column(name: str) -> bool:
    text = str(name).strip()
    return text in ABSOLUTE_COORDINATE_COLUMNS


def _is_relative_coordinate_column(name: str) -> bool:
    text = str(name).strip()
    return any(marker in text for marker in RELATIVE_COORDINATE_MARKERS)


def feature_keep_mask(columns: list[str] | tuple[str, ...], coordinate_feature_mode: str) -> np.ndarray:
    mode = normalize_coordinate_feature_mode(coordinate_feature_mode)
    keep = np.ones(len(columns), dtype=bool)
    if mode == "absolute_and_relative":
        return keep
    for index, name in enumerate(columns):
        if mode == "relative_only" and _is_absolute_coordinate_column(name):
            keep[index] = False
        elif mode == "absolute_only" and _is_relative_coordinate_column(name):
            keep[index] = False
    return keep


def filter_feature_matrix(
    values: np.ndarray,
    columns: list[str] | tuple[str, ...],
    coordinate_feature_mode: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        return array
    mask = feature_keep_mask(columns, coordinate_feature_mode)
    if mask.size != array.shape[1]:
        return array
    return array[:, mask].astype(np.float32, copy=False)


def filtered_columns(columns: list[str] | tuple[str, ...], coordinate_feature_mode: str) -> list[str]:
    mask = feature_keep_mask(columns, coordinate_feature_mode)
    return [str(column) for column, keep in zip(columns, mask) if bool(keep)]


def _pulse_weights_from_features(
    pulse_features: np.ndarray,
    pulse_feature_columns: list[str] | tuple[str, ...],
) -> np.ndarray:
    features = np.asarray(pulse_features, dtype=np.float32)
    n_pulse = int(features.shape[0]) if features.ndim == 2 else 0
    if n_pulse == 0:
        return np.zeros((0,), dtype=np.float32)
    columns = [str(column) for column in pulse_feature_columns]
    if "log10_pulse_rho" in columns:
        values = features[:, columns.index("log10_pulse_rho")]
        weights = np.power(np.float32(10.0), np.clip(values, -20.0, 20.0)).astype(np.float32)
    elif "sqrt_pulse_rho" in columns:
        values = features[:, columns.index("sqrt_pulse_rho")]
        weights = np.square(np.maximum(values, 0.0), dtype=np.float32)
    else:
        weights = np.ones((n_pulse,), dtype=np.float32)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones((n_pulse,), dtype=np.float32)
    return weights


def signal_barycenter_anchor(
    *,
    pulse_positions_km: np.ndarray,
    pulse_features: np.ndarray,
    pulse_feature_columns: list[str] | tuple[str, ...],
) -> np.ndarray:
    positions = np.asarray(pulse_positions_km, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] < 2:
        raise ValueError("signal_bary_relative core target needs pulse_positions_km with x/y columns")
    weights = _pulse_weights_from_features(pulse_features, pulse_feature_columns)
    if weights.shape[0] != positions.shape[0]:
        raise ValueError("pulse feature/position row mismatch while computing signal-bary core anchor")
    finite = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1]) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        raise ValueError("cannot compute signal-bary core anchor: no finite weighted pulse positions")
    weighted = positions[finite, :2] * weights[finite, None]
    return (np.sum(weighted, axis=0) / np.sum(weights[finite])).astype(np.float32)


def fit_core_anchor_from_metadata(metadata: Mapping[str, Any] | None) -> np.ndarray:
    if not metadata:
        raise ValueError("fit_core_relative core target needs metadata with reference_core_km")
    value = None
    for key in ("reference_core_km", "ising_core_km", "fit_core_km"):
        if key in metadata and metadata[key] is not None:
            value = metadata[key]
            break
    if value is None:
        raise ValueError("fit_core_relative core target needs metadata reference_core_km")
    anchor = np.asarray(value, dtype=np.float32).reshape(-1)
    if anchor.shape[0] < 2 or not np.all(np.isfinite(anchor[:2])):
        raise ValueError("fit_core_relative core anchor is not finite")
    return anchor[:2].astype(np.float32)


def core_anchor_from_sample(
    sample: Mapping[str, Any],
    *,
    columns: Mapping[str, Any] | None,
    core_anchor_mode: str,
) -> np.ndarray:
    mode = normalize_core_target_mode(core_anchor_mode)
    if mode == "absolute":
        return np.zeros((2,), dtype=np.float32)
    if mode == "signal_bary_relative":
        column_map = dict(columns or {})
        pulse_columns = list(column_map.get("pulse_features", []))
        return signal_barycenter_anchor(
            pulse_positions_km=np.asarray(sample.get("pulse_positions_km"), dtype=np.float32),
            pulse_features=np.asarray(sample.get("pulse_features"), dtype=np.float32),
            pulse_feature_columns=pulse_columns,
        )
    if mode == "fit_core_relative":
        return fit_core_anchor_from_metadata(sample.get("metadata"))
    raise AssertionError(mode)


def transform_core_target(target: np.ndarray | None, core_anchor: np.ndarray, core_target_mode: str) -> np.ndarray | None:
    if target is None:
        return None
    mode = normalize_core_target_mode(core_target_mode)
    transformed = np.asarray(target, dtype=np.float32).copy()
    if transformed.shape[0] >= 3 and mode != "absolute":
        anchor = np.asarray(core_anchor, dtype=np.float32).reshape(-1)
        if anchor.shape[0] < 2 or not np.all(np.isfinite(anchor[:2])):
            raise ValueError("relative core target needs finite core_anchor x/y")
        transformed[1:3] = transformed[1:3] - anchor[:2]
    return transformed


def inverse_transform_core_target(values: np.ndarray, core_anchor: np.ndarray | None, core_target_mode: str) -> np.ndarray:
    mode = normalize_core_target_mode(core_target_mode)
    restored = np.asarray(values, dtype=np.float32).copy()
    if mode == "absolute" or restored.shape[0] == 0 or restored.shape[1] < 3:
        return restored
    if core_anchor is None:
        raise ValueError("relative core prediction needs core_anchor for absolute metric conversion")
    anchor = np.asarray(core_anchor, dtype=np.float32)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
    if anchor.shape[0] != restored.shape[0] or anchor.shape[1] < 2:
        raise ValueError("core_anchor shape does not match prediction rows")
    restored[:, 1:3] = restored[:, 1:3] + anchor[:, :2]
    return restored


def coordinate_mode_summary(
    *,
    columns: Mapping[str, Any],
    core_target_mode: str,
    coordinate_feature_mode: str,
) -> dict[str, Any]:
    detector_columns = list(columns.get("detector_features", []))
    pulse_columns = list(columns.get("pulse_features", []))
    detector_effective = filtered_columns(detector_columns, coordinate_feature_mode)
    pulse_effective = filtered_columns(pulse_columns, coordinate_feature_mode)
    return {
        "core_target_mode": normalize_core_target_mode(core_target_mode),
        "core_anchor_mode": normalize_core_target_mode(core_target_mode),
        "coordinate_feature_mode": normalize_coordinate_feature_mode(coordinate_feature_mode),
        "target_columns": list(target_columns_for_mode(core_target_mode)),
        "core_anchor_columns": list(CORE_ANCHOR_COLUMNS),
        "has_detector_absolute_scalar_xyz": any(_is_absolute_coordinate_column(name) for name in detector_effective),
        "has_pulse_absolute_scalar_xyz": any(_is_absolute_coordinate_column(name) for name in pulse_effective),
        "has_detector_relative_scalar_xyz": any(_is_relative_coordinate_column(name) for name in detector_effective),
        "has_pulse_relative_scalar_xyz": any(_is_relative_coordinate_column(name) for name in pulse_effective),
        "detector_feature_columns": detector_effective,
        "pulse_feature_columns": pulse_effective,
    }
