from __future__ import annotations

from typing import Any


LEGACY_FLAT50000_NODE_FEATURE_COLUMNS = [
    "x_km",
    "y_km",
    "z_km",
    "nearest_detector_distance_km",
    "mean3_detector_distance_km",
    "neighbor_count_1p5km",
    "local_detector_density_1p5km2",
    "dx_from_bary_km",
    "dy_from_bary_km",
    "dz_from_bary_km",
    "r_from_bary_km",
    "first_arrival_usec_rel",
    "trig_usec_rel",
    "log10_first_rho",
    "sqrt_first_rho",
    "log10_max_rho",
    "n_pulses",
    "pulse_time_span_usec",
    "n_wf_segments",
    "wf_length_usec",
    "log10_fadc_peak",
    "upper_ped",
    "lower_ped",
    "upper_ped_sigma",
    "lower_ped_sigma",
    "detector_pulse_order",
    "is_first_detector_pulse",
]
LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS = (
    "log10_total_rho",
    "sqrt_total_rho",
)
LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS = [
    "node_index",
    "arrival_usec_rel",
    "dt_from_first_usec",
    "log10_rho",
    "sqrt_rho",
    "pulse_order",
    "is_first_pulse",
]
LEGACY_FLAT50000_EDGE_FEATURE_COLUMNS = [
    "dx_km",
    "dy_km",
    "dz_km",
    "distance_km",
    "dt_usec",
    "abs_dt_usec",
    "dt_per_km",
    "log10_rho_ratio",
    "ising_weight",
    "ising_weight_raw",
    "ising_causal_excess_usec",
    "ising_spatial",
    "ising_causal",
]
LEGACY_FLAT50000_TARGET_COLUMNS = [
    "log10_energy_eV",
    "core_x_km",
    "core_y_km",
    "core_z_km",
    "dir_x",
    "dir_y",
    "dir_z",
]
LEGACY_FLAT50000_WAVEFORM_SCHEMA = "rise_aligned_raw_plus_accepted_gapped_v1"
LEGACY_FLAT50000_WAVEFORM_FEATURE_CHANNELS = [
    "upper_raw_window_vem",
    "lower_raw_window_vem",
    "upper_accepted_gapped_vem",
    "lower_accepted_gapped_vem",
]


def normalize_homogeneous_schema(schema: str | None) -> str:
    value = "current" if schema is None else str(schema).strip().lower().replace("-", "_")
    if value in {"", "current"}:
        return "current"
    if value in {"legacy_flat50000", "flat50000"}:
        return "legacy_flat50000"
    raise ValueError("homogeneous_schema must be 'current' or 'legacy_flat50000'")


def homogeneous_dataset_kwargs_for_schema(schema: str | None) -> dict[str, Any]:
    mode = normalize_homogeneous_schema(schema)
    if mode == "current":
        return {}
    return {
        "expected_node_feature_columns": LEGACY_FLAT50000_NODE_FEATURE_COLUMNS,
        "dropped_node_feature_columns": LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS,
        "expected_pulse_feature_columns": LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS,
        "dropped_pulse_feature_columns": (),
        "allowed_waveform_schemas": (LEGACY_FLAT50000_WAVEFORM_SCHEMA,),
        "expected_waveform_feature_channels": LEGACY_FLAT50000_WAVEFORM_FEATURE_CHANNELS,
    }


def legacy_flat50000_checkpoint_matches(model_config: dict[str, Any]) -> bool:
    node_dim = int(model_config.get("node_dim", -1))
    pulse_dim = int(model_config.get("pulse_dim", -1))
    target_dim = int(model_config.get("target_dim", -1))
    waveform_schema = str(model_config.get("waveform_schema", "")).strip()
    waveform_channels = int(model_config.get("waveform_channels", 0) or 0)
    return (
        node_dim == len(LEGACY_FLAT50000_NODE_FEATURE_COLUMNS)
        and pulse_dim == len(LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS) - 1
        and target_dim == 7
        and (not waveform_schema or waveform_schema == LEGACY_FLAT50000_WAVEFORM_SCHEMA)
        and waveform_channels == len(LEGACY_FLAT50000_WAVEFORM_FEATURE_CHANNELS)
    )


def homogeneous_dataset_kwargs_from_checkpoint(ckpt: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model_config = dict(ckpt.get("model_config", {}))
    if legacy_flat50000_checkpoint_matches(model_config):
        return "legacy_flat50000", homogeneous_dataset_kwargs_for_schema("legacy_flat50000")
    return "current", {}
