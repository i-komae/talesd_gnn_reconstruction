from __future__ import annotations

MEV_PER_VMIP = 2.0
AREA_OF_SD_M2 = 3.0
COIN_TIME_WINDOW_USEC = 0.24
FADC_BIN_WIDTH_USEC = 0.02
FADC_PRETRIGGER_BINS = 29

MAX_FADC_COUNT = 4095.0
MIN_VALID_MIP = 0.3
GEOM_MIN_POINTS = 4

ISING_EDGE_MAX_DISTANCE_KM = 1.5
ISING_EDGE_MAX_DT_USEC = 8.0
ISING_SPACE_SCALE_KM = 0.8
ISING_CAUSAL_TAU_USEC = 1.0
ISING_CAUSAL_GRACE_USEC = 0.3
ISING_SIGNAL_WEIGHT = 0.12
ISING_CAUSAL_PENALTY_PER_USEC = 0.0
ISING_EDGE_DEGREE_POWER = 0.70
LIGHT_SPEED_KM_PER_USEC = 0.299792458

NODE_FEATURE_COLUMNS = [
    "x_km",
    "y_km",
    "z_km",
    "nearest_detector_distance_km",
    "mean3_detector_distance_km",
    "neighbor_count_1p5km",
    "dx_from_signal_bary_km",
    "dy_from_signal_bary_km",
    "dz_from_signal_bary_km",
    "r_from_signal_bary_km",
    "pulse_arrival_usec_rel",
    "detector_trigger_usec_rel",
    "log10_pulse_rho",
    "sqrt_pulse_rho",
    "log10_detector_max_pulse_rho",
    "log10_detector_sum_pulse_rho",
    "sqrt_detector_sum_pulse_rho",
    "detector_accepted_pulse_count",
    "detector_accepted_pulse_time_span_usec",
    "detector_wf_segments",
    "detector_wf_length_usec",
    "log10_detector_fadc_peak",
    "detector_upper_ped",
    "detector_lower_ped",
    "detector_upper_ped_sigma",
    "detector_lower_ped_sigma",
    "accepted_pulse_order",
    "is_first_accepted_pulse",
]

DROPPED_NODE_FEATURE_COLUMNS = (
    "log10_total_rho",
    "sqrt_total_rho",
    "local_detector_density_1p5km2",
)

EDGE_FEATURE_COLUMNS = [
    "dx_km",
    "dy_km",
    "dz_km",
    "distance_km",
    "dt_usec",
    "abs_dt_usec",
    "dt_per_km",
    "dlog10_pulse_rho",
    "ising_weight",
    "ising_weight_raw",
    "ising_causal_excess_usec",
    "ising_spatial",
    "ising_causal",
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

PROTON_PARTTYPE = 14
IRON_PARTTYPE = 5626

PULSE_FEATURE_COLUMNS = [
    "node_index",
]

DROPPED_PULSE_FEATURE_COLUMNS = (
    "arrival_usec_rel",
    "dt_from_first_usec",
    "log10_rho",
    "sqrt_rho",
    "pulse_order",
    "is_first_pulse",
)

WAVEFORM_TRACE_BINS = 128
WAVEFORM_RISE_ANCHOR_BIN = 32
WAVEFORM_SCHEMA = "rise_aligned_raw_plus_accepted_mask_v1"
WAVEFORM_FEATURE_CHANNELS = [
    "upper_raw_window_vem",
    "lower_raw_window_vem",
    "upper_accepted_mask",
    "lower_accepted_mask",
]
