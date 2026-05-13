from __future__ import annotations

MEV_PER_VMIP = 2.0
AREA_OF_SD_M2 = 3.0
COIN_TIME_WINDOW_USEC = 0.24
FADC_BIN_WIDTH_USEC = 0.02
FADC_PRETRIGGER_BINS = 29

DEFAULT_EDGE_RADIUS_KM = 1.5
DEFAULT_EDGE_K = 6
DEFAULT_MIN_NODES = 3

NODE_FEATURE_COLUMNS = [
    "x_km",
    "y_km",
    "z_km",
    "dx_from_bary_km",
    "dy_from_bary_km",
    "dz_from_bary_km",
    "r_from_bary_km",
    "first_arrival_usec_rel",
    "trig_usec_rel",
    "log10_first_rho",
    "sqrt_first_rho",
    "log10_total_rho",
    "sqrt_total_rho",
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
]

EDGE_FEATURE_COLUMNS = [
    "dx_km",
    "dy_km",
    "dz_km",
    "distance_km",
    "dt_usec",
    "abs_dt_usec",
    "dt_per_km",
    "log10_rho_ratio",
]

TARGET_COLUMNS = [
    "log10_energy_eV",
    "core_x_km",
    "core_y_km",
    "core_z_km",
    "dir_x",
    "dir_y",
    "dir_z",
]

PULSE_FEATURE_COLUMNS = [
    "node_index",
    "arrival_usec_rel",
    "dt_from_first_usec",
    "log10_rho",
    "sqrt_rho",
    "pulse_order",
    "is_first_pulse",
]
