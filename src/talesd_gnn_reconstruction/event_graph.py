from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import (
    AREA_OF_SD_M2,
    EDGE_FEATURE_COLUMNS,
    FADC_BIN_WIDTH_USEC,
    GEOM_MIN_POINTS,
    IRON_PARTTYPE,
    ISING_CAUSAL_GRACE_USEC,
    ISING_CAUSAL_PENALTY_PER_USEC,
    ISING_CAUSAL_TAU_USEC,
    ISING_EDGE_DEGREE_POWER,
    ISING_EDGE_MAX_DISTANCE_KM,
    ISING_EDGE_MAX_DT_USEC,
    ISING_SIGNAL_WEIGHT,
    ISING_SPACE_SCALE_KM,
    LIGHT_SPEED_KM_PER_USEC,
    MAX_FADC_COUNT,
    MEV_PER_VMIP,
    MIN_VALID_MIP,
    NODE_FEATURE_COLUMNS,
    PULSE_FEATURE_COLUMNS,
    PROTON_PARTTYPE,
    TARGET_COLUMNS,
    WAVEFORM_FEATURE_CHANNELS,
    WAVEFORM_PRE_PULSE_BINS,
    WAVEFORM_TRACE_BINS,
)
from .dst_reader import BankRecord
from .layout import DetectorPosition
from .signal import find_coincident_pulses, sd_signal_search_10a


@dataclass(frozen=True)
class GraphEvent:
    event_id: str
    node_features: np.ndarray
    node_positions_km: np.ndarray
    node_lids: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    pulse_features: np.ndarray
    waveform_features: np.ndarray
    target: np.ndarray | None
    particle_label: float | None
    metadata: dict[str, Any]


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _position_from_sub(
    sub: dict[str, Any],
    detector_positions: dict[int, DetectorPosition] | None,
) -> tuple[float, float, float] | None:
    pos_x = _finite_float(sub.get("posX"))
    pos_y = _finite_float(sub.get("posY"))
    pos_z = _finite_float(sub.get("posZ"))
    if math.isfinite(pos_x) and math.isfinite(pos_y) and math.isfinite(pos_z):
        return pos_x / 1.0e3, pos_y / 1.0e3, pos_z / 1.0e3
    if detector_positions is None:
        return None
    detector = detector_positions.get(int(sub.get("lid", -1)))
    if detector is None:
        return None
    return detector.x_km, detector.y_km, detector.z_km


def _trigger_usec(base_usec: float, sub: dict[str, Any]) -> float | None:
    max_clock = int(sub.get("maxClock", 0))
    if max_clock <= 0:
        return None
    trg_usec = int(sub.get("clock", 0)) / max_clock * 1.0e6
    if base_usec < 64.0 and trg_usec > 999_936.0:
        trg_usec -= 1.0e6
    if base_usec > 999_936.0 and trg_usec < 64.0:
        trg_usec += 1.0e6
    return float(trg_usec)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _waveform_order_key(sub: dict[str, Any]) -> tuple[int, int]:
    return (_safe_int(sub.get("wfId"), 0), _safe_int(sub.get("clock"), 0))


def _unwrap_segment_clocks(rows: list[dict[str, Any]]) -> list[float]:
    ordered = sorted(rows, key=_waveform_order_key)
    unwrapped: list[float] = []
    previous: float | None = None
    for sub in ordered:
        clock = float(_safe_int(sub.get("clock"), 0))
        max_clock = max(float(_safe_int(sub.get("maxClock"), 0)), 1.0)
        if previous is not None:
            while clock < previous - 0.5 * max_clock:
                clock += max_clock
        unwrapped.append(clock)
        previous = clock
    return unwrapped


def _combine_waveform_segments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=_waveform_order_key)
    unwrapped = _unwrap_segment_clocks(ordered)
    base_clock = min(unwrapped)
    max_clock = max(float(_safe_int(ordered[0].get("maxClock"), 0)), 1.0)

    offsets = [
        int(round(((clock - base_clock) / max_clock * 1.0e6) / FADC_BIN_WIDTH_USEC))
        for clock in unwrapped
    ]
    upper_arrays = [np.asarray(sub.get("uwf", []), dtype=float) for sub in ordered]
    lower_arrays = [np.asarray(sub.get("lwf", []), dtype=float) for sub in ordered]
    wf_len = max(
        [offset + int(array.shape[0]) for offset, array in zip(offsets, upper_arrays)]
        + [offset + int(array.shape[0]) for offset, array in zip(offsets, lower_arrays)]
    )

    upper_ped = float(np.mean([_finite_float(sub.get("upedAvr"), 0.0) for sub in ordered]))
    lower_ped = float(np.mean([_finite_float(sub.get("lpedAvr"), 0.0) for sub in ordered]))
    upper_wf = np.full(wf_len, upper_ped, dtype=np.float32)
    lower_wf = np.full(wf_len, lower_ped, dtype=np.float32)
    for offset, upper, lower in zip(offsets, upper_arrays, lower_arrays):
        upper_wf[offset : offset + upper.shape[0]] = upper
        lower_wf[offset : offset + lower.shape[0]] = lower

    combined = dict(ordered[unwrapped.index(base_clock)])
    combined["clock"] = int(round(base_clock)) % max(_safe_int(combined.get("maxClock"), 0), 1)
    combined["uwf"] = upper_wf
    combined["lwf"] = lower_wf
    combined["upedAvr"] = upper_ped
    combined["lpedAvr"] = lower_ped
    combined["upedStdev"] = float(np.mean([_finite_float(sub.get("upedStdev"), 0.0) for sub in ordered]))
    combined["lpedStdev"] = float(np.mean([_finite_float(sub.get("lpedStdev"), 0.0) for sub in ordered]))
    combined["umipMev2cnt"] = float(np.mean([_finite_float(sub.get("umipMev2cnt"), 1.0) for sub in ordered]))
    combined["lmipMev2cnt"] = float(np.mean([_finite_float(sub.get("lmipMev2cnt"), 1.0) for sub in ordered]))
    combined["numSegments"] = len(ordered)
    combined["wfLengthBins"] = wf_len
    combined["wfIds"] = ",".join(str(_safe_int(sub.get("wfId"), 0)) for sub in ordered)
    return combined


def _combine_sub_waveforms(subs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lid: dict[int, list[dict[str, Any]]] = {}
    for sub in subs:
        by_lid.setdefault(_safe_int(sub.get("lid"), -1), []).append(sub)

    combined: list[dict[str, Any]] = []
    for rows in by_lid.values():
        rows = [row for row in rows if _safe_int(row.get("dontUse"), 0) == 0]
        if not rows:
            continue
        if len(rows) == 1:
            row = dict(rows[0])
            row["numSegments"] = 1
            row["wfLengthBins"] = max(len(row.get("uwf", [])), len(row.get("lwf", [])))
            row["wfIds"] = str(_safe_int(row.get("wfId"), 0))
            combined.append(row)
            continue
        combined.append(_combine_waveform_segments(rows))
    return combined


def _calibrated_vem_trace(wf: np.ndarray, ped: float, mev2cnt: float) -> np.ndarray:
    denom = max(float(mev2cnt) * MEV_PER_VMIP, 1.0e-6)
    trace = (np.asarray(wf, dtype=np.float32) - float(ped)) / denom
    return np.clip(trace, -20.0, 200.0).astype(np.float32, copy=False)


def _copy_window(values: np.ndarray, center_bin: int, length: int = WAVEFORM_TRACE_BINS) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    start = int(center_bin) - int(WAVEFORM_PRE_PULSE_BINS)
    end = start + int(length)
    src_start = max(start, 0)
    src_end = min(end, int(values.shape[0]))
    if src_end <= src_start:
        return out
    dst_start = src_start - start
    out[dst_start : dst_start + (src_end - src_start)] = values[src_start:src_end]
    return out


def _copy_compact_pulses(
    values: np.ndarray,
    accepted_pulses: list[Any],
    *,
    channel: str,
    length: int = WAVEFORM_TRACE_BINS,
) -> np.ndarray:
    pieces = []
    for pulse in accepted_pulses:
        if channel == "upper":
            start = int(pulse.upper_rise_bin)
            end = int(pulse.upper_fall_bin) + 1
        else:
            start = int(pulse.lower_rise_bin)
            end = int(pulse.lower_fall_bin) + 1
        start = max(start, 0)
        end = min(end, int(values.shape[0]))
        if end > start:
            pieces.append(values[start:end])
    out = np.zeros(length, dtype=np.float32)
    if not pieces:
        return out
    compact = np.concatenate(pieces).astype(np.float32, copy=False)
    n_copy = min(int(length), int(compact.shape[0]))
    if n_copy > 0:
        out[:n_copy] = compact[:n_copy]
    return out


def _waveform_features_for_pulse(
    *,
    upper_wf: np.ndarray,
    lower_wf: np.ndarray,
    upper_ped: float,
    lower_ped: float,
    upper_mev2cnt: float,
    lower_mev2cnt: float,
    pulse: Any,
    accepted_pulses: list[Any],
) -> np.ndarray:
    upper_vem = _calibrated_vem_trace(upper_wf, upper_ped, upper_mev2cnt)
    lower_vem = _calibrated_vem_trace(lower_wf, lower_ped, lower_mev2cnt)
    center_bin = min(int(pulse.upper_rise_bin), int(pulse.lower_rise_bin))
    return np.stack(
        [
            _copy_window(upper_vem, center_bin),
            _copy_window(lower_vem, center_bin),
            _copy_compact_pulses(upper_vem, accepted_pulses, channel="upper"),
            _copy_compact_pulses(lower_vem, accepted_pulses, channel="lower"),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _local_detector_context(positions: np.ndarray, detector_ids: list[int]) -> np.ndarray:
    unique: dict[int, np.ndarray] = {}
    for lid, position in zip(detector_ids, positions):
        unique.setdefault(int(lid), np.asarray(position, dtype=np.float32))
    ids = list(unique.keys())
    if len(ids) <= 1:
        return np.zeros((positions.shape[0], 4), dtype=np.float32)

    unique_positions = np.stack([unique[lid] for lid in ids], axis=0)
    id_to_row = {lid: row for row, lid in enumerate(ids)}
    context_by_id: dict[int, np.ndarray] = {}
    area = math.pi * ISING_EDGE_MAX_DISTANCE_KM * ISING_EDGE_MAX_DISTANCE_KM
    for lid, row in id_to_row.items():
        delta = unique_positions - unique_positions[row]
        distances = np.linalg.norm(delta[:, :2], axis=1)
        distances = distances[distances > 1.0e-6]
        if distances.size == 0:
            context_by_id[lid] = np.zeros(4, dtype=np.float32)
            continue
        nearest = float(np.min(distances))
        mean3 = float(np.mean(np.sort(distances)[: min(3, distances.size)]))
        neighbor_count = float(np.sum(distances <= ISING_EDGE_MAX_DISTANCE_KM))
        density = float(neighbor_count / area)
        context_by_id[lid] = np.asarray([nearest, mean3, neighbor_count, density], dtype=np.float32)
    return np.stack([context_by_id[int(lid)] for lid in detector_ids], axis=0).astype(np.float32, copy=False)


def _extract_hit(
    bank: dict[str, Any],
    sub: dict[str, Any],
    detector_positions: dict[int, DetectorPosition] | None,
) -> dict[str, Any] | None:
    if int(sub.get("dontUse", 0)) != 0:
        return None
    position = _position_from_sub(sub, detector_positions)
    if position is None:
        return None

    base_usec = float(bank.get("usec", 0))
    trg_usec = _trigger_usec(base_usec, sub)
    if trg_usec is None:
        return None

    upper_wf = np.asarray(sub.get("uwf", []), dtype=float)
    lower_wf = np.asarray(sub.get("lwf", []), dtype=float)
    if upper_wf.size == 0 or lower_wf.size == 0:
        return None

    upper_ped = _finite_float(sub.get("upedAvr"), 0.0)
    lower_ped = _finite_float(sub.get("lpedAvr"), 0.0)
    upper_sigma = max(_finite_float(sub.get("upedStdev"), 0.0), 1.0e-6)
    lower_sigma = max(_finite_float(sub.get("lpedStdev"), 0.0), 1.0e-6)
    upper_mev2cnt = max(_finite_float(sub.get("umipMev2cnt"), 1.0), 1.0e-6)
    lower_mev2cnt = max(_finite_float(sub.get("lmipMev2cnt"), 1.0), 1.0e-6)

    upper_result = sd_signal_search_10a(upper_wf, upper_ped, upper_sigma)
    lower_result = sd_signal_search_10a(lower_wf, lower_ped, lower_sigma)
    pulses = find_coincident_pulses(upper_result, lower_result, upper_mev2cnt, lower_mev2cnt)
    if not pulses:
        return None

    pulse_rows = []
    for order, pulse in enumerate(pulses):
        rho = pulse.energy_mev / (MEV_PER_VMIP * AREA_OF_SD_M2)
        pulse_rows.append(
            {
                "arrival_usec": float((trg_usec - base_usec) + pulse.time_usec),
                "rho": float(rho),
                "order": int(order),
                "upper_rise_bin": int(pulse.upper_rise_bin),
                "upper_fall_bin": int(pulse.upper_fall_bin),
                "lower_rise_bin": int(pulse.lower_rise_bin),
                "lower_fall_bin": int(pulse.lower_fall_bin),
                "waveform_features": _waveform_features_for_pulse(
                    upper_wf=upper_wf,
                    lower_wf=lower_wf,
                    upper_ped=upper_ped,
                    lower_ped=lower_ped,
                    upper_mev2cnt=upper_mev2cnt,
                    lower_mev2cnt=lower_mev2cnt,
                    pulse=pulse,
                    accepted_pulses=pulses,
                ),
            }
        )

    fadc_peak = float(max(np.max(upper_wf), np.max(lower_wf)))
    total_rho = float(pulse_rows[0]["rho"])
    max_rho = float(max(row["rho"] for row in pulse_rows))
    time_span = float(max(row["arrival_usec"] for row in pulse_rows) - min(row["arrival_usec"] for row in pulse_rows))

    return {
        "lid": int(sub.get("lid", -1)),
        "position": position,
        "trig_usec_rel": float(trg_usec - base_usec),
        "fadc_peak": fadc_peak,
        "upper_ped": upper_ped,
        "lower_ped": lower_ped,
        "upper_sigma": upper_sigma,
        "lower_sigma": lower_sigma,
        "num_pulses": len(pulses),
        "total_rho": total_rho,
        "max_rho": max_rho,
        "pulse_time_span_usec": time_span,
        "num_segments": int(sub.get("numSegments", 1)),
        "wf_length_usec": float(int(sub.get("wfLengthBins", upper_wf.size)) * FADC_BIN_WIDTH_USEC),
        "pulses": pulse_rows,
    }


def _merge_hits_by_lid(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for hit in hits:
        buckets.setdefault(int(hit["lid"]), []).append(hit)

    merged: list[dict[str, Any]] = []
    for lid, rows in buckets.items():
        if len(rows) == 1:
            merged.append(rows[0])
            continue

        pulses = [dict(pulse) for row in rows for pulse in row["pulses"]]
        pulses.sort(key=lambda pulse: float(pulse["arrival_usec"]))
        for order, pulse in enumerate(pulses):
            pulse["order"] = order

        total_rho = float(pulses[0]["rho"])
        max_rho = float(max(float(pulse["rho"]) for pulse in pulses))
        time_span = float(float(pulses[-1]["arrival_usec"]) - float(pulses[0]["arrival_usec"]))
        first_row = min(rows, key=lambda row: min(float(pulse["arrival_usec"]) for pulse in row["pulses"]))

        merged.append(
            {
                "lid": lid,
                "position": first_row["position"],
                "trig_usec_rel": float(np.mean([row["trig_usec_rel"] for row in rows])),
                "fadc_peak": float(max(row["fadc_peak"] for row in rows)),
                "upper_ped": float(np.mean([row["upper_ped"] for row in rows])),
                "lower_ped": float(np.mean([row["lower_ped"] for row in rows])),
                "upper_sigma": float(np.mean([row["upper_sigma"] for row in rows])),
                "lower_sigma": float(np.mean([row["lower_sigma"] for row in rows])),
                "num_pulses": len(pulses),
                "total_rho": total_rho,
                "max_rho": max_rho,
                "pulse_time_span_usec": time_span,
                "num_segments": int(sum(row.get("num_segments", 1) for row in rows)),
                "wf_length_usec": float(max(row.get("wf_length_usec", 0.0) for row in rows)),
                "pulses": pulses,
            }
        )
    return merged


def _target_from_sim(sim: dict[str, Any] | None) -> np.ndarray | None:
    if not sim:
        return None
    energy = _finite_float(sim.get("primaryEnergy"))
    cos_zenith = _finite_float(sim.get("primaryCosZenith"))
    azimuth_deg = _finite_float(sim.get("primaryAzimuth"))
    core_x_m = _finite_float(sim.get("primaryCorePosX"))
    core_y_m = _finite_float(sim.get("primaryCorePosY"))
    core_z_m = _finite_float(sim.get("primaryCorePosZ"), 0.0)
    values = [energy, cos_zenith, azimuth_deg, core_x_m, core_y_m, core_z_m]
    if not all(math.isfinite(v) for v in values) or energy <= 0.0:
        return None

    cos_zenith = min(max(cos_zenith, -1.0), 1.0)
    zenith = math.acos(cos_zenith)
    azimuth = math.radians(azimuth_deg)
    direction = np.array(
        [
            math.sin(zenith) * math.cos(azimuth),
            math.sin(zenith) * math.sin(azimuth),
            math.cos(zenith),
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            math.log10(energy),
            core_x_m / 1.0e3,
            core_y_m / 1.0e3,
            core_z_m / 1.0e3,
            direction[0],
            direction[1],
            direction[2],
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(target)):
        return None
    return target


def _particle_label_from_sim(sim: dict[str, Any] | None) -> tuple[int | None, float | None]:
    if not sim:
        return None, None
    parttype = int(sim.get("primaryParticleId", -1))
    if parttype == PROTON_PARTTYPE:
        return parttype, 0.0
    if parttype == IRON_PARTTYPE:
        return parttype, 1.0
    return parttype if parttype >= 0 else None, None


def _build_node_features(
    hits: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    pulse_nodes: list[dict[str, Any]] = []
    for hit in hits:
        pulses = sorted(hit["pulses"], key=lambda pulse: float(pulse["arrival_usec"]))
        if not pulses:
            continue
        first_arrival = float(pulses[0]["arrival_usec"])
        max_rho = float(max(float(pulse["rho"]) for pulse in pulses))
        time_span = float(float(pulses[-1]["arrival_usec"]) - first_arrival)
        fadc_peak = float(hit["fadc_peak"])
        for order, pulse in enumerate(pulses):
            rho = float(pulse["rho"])
            if rho < MIN_VALID_MIP or fadc_peak >= MAX_FADC_COUNT:
                continue
            pulse_nodes.append(
                {
                    "lid": int(hit["lid"]),
                    "position": hit["position"],
                    "arrival_usec": float(pulse["arrival_usec"]),
                    "trig_usec_rel": float(hit["trig_usec_rel"]),
                    "rho": rho,
                    "detector_max_rho": max_rho,
                    "detector_num_pulses": len(pulses),
                    "detector_pulse_time_span_usec": time_span,
                    "detector_pulse_order": order,
                    "is_first_detector_pulse": 1.0 if order == 0 else 0.0,
                    "fadc_peak": fadc_peak,
                    "upper_ped": float(hit["upper_ped"]),
                    "lower_ped": float(hit["lower_ped"]),
                    "upper_sigma": float(hit["upper_sigma"]),
                    "lower_sigma": float(hit["lower_sigma"]),
                    "num_segments": int(hit["num_segments"]),
                    "wf_length_usec": float(hit["wf_length_usec"]),
                    "waveform_features": np.asarray(
                        pulse.get(
                            "waveform_features",
                            np.zeros((len(WAVEFORM_FEATURE_CHANNELS), WAVEFORM_TRACE_BINS), dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ),
                }
            )

    if not pulse_nodes:
        return (
            np.zeros((0, len(NODE_FEATURE_COLUMNS)), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, len(PULSE_FEATURE_COLUMNS)), dtype=np.float32),
            np.zeros((0, len(WAVEFORM_FEATURE_CHANNELS), WAVEFORM_TRACE_BINS), dtype=np.float32),
            [],
            [],
        )

    positions = np.asarray([node["position"] for node in pulse_nodes], dtype=np.float32)
    detector_ids = [int(node["lid"]) for node in pulse_nodes]
    local_context = _local_detector_context(positions, detector_ids)
    rho = np.asarray([node["rho"] for node in pulse_nodes], dtype=np.float32)
    weights = np.maximum(rho, 0.0) + 1.0e-6
    bary = np.sum(positions * weights[:, None], axis=0) / np.sum(weights)
    delta = positions - bary[None, :]
    radius = np.linalg.norm(delta, axis=1)
    arrival = np.asarray([node["arrival_usec"] for node in pulse_nodes], dtype=np.float32)
    arrival_min = float(np.min(arrival))
    arrival_rel = arrival - arrival_min

    features = []
    pulse_features = []
    waveform_features = []
    lids = []
    for i, node in enumerate(pulse_nodes):
        rho_i = max(float(node["rho"]), 1.0e-6)
        max_rho_i = max(float(node["detector_max_rho"]), 1.0e-6)
        fadc_peak = max(float(node["fadc_peak"]), 1.0)
        lid = int(node["lid"])
        lids.append(lid)
        features.append(
            [
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                local_context[i, 0],
                local_context[i, 1],
                local_context[i, 2],
                local_context[i, 3],
                delta[i, 0],
                delta[i, 1],
                delta[i, 2],
                radius[i],
                arrival_rel[i],
                float(node["trig_usec_rel"]),
                math.log10(rho_i),
                math.sqrt(rho_i),
                math.log10(rho_i),
                math.sqrt(rho_i),
                math.log10(max_rho_i),
                float(node["detector_num_pulses"]),
                float(node["detector_pulse_time_span_usec"]),
                float(node["num_segments"]),
                float(node["wf_length_usec"]),
                math.log10(fadc_peak),
                float(node["upper_ped"]),
                float(node["lower_ped"]),
                float(node["upper_sigma"]),
                float(node["lower_sigma"]),
                float(node["detector_pulse_order"]),
                float(node["is_first_detector_pulse"]),
            ]
        )
        pulse_features.append(
            [
                float(i),
                float(arrival_rel[i]),
                float(arrival_rel[i]),
                math.log10(rho_i),
                math.sqrt(rho_i),
                float(node["detector_pulse_order"]),
                float(node["is_first_detector_pulse"]),
            ]
        )
        waveform = np.asarray(node["waveform_features"], dtype=np.float32)
        expected_shape = (len(WAVEFORM_FEATURE_CHANNELS), WAVEFORM_TRACE_BINS)
        if waveform.shape != expected_shape:
            fixed = np.zeros(expected_shape, dtype=np.float32)
            channels = min(fixed.shape[0], waveform.shape[0] if waveform.ndim >= 1 else 0)
            bins = min(fixed.shape[1], waveform.shape[1] if waveform.ndim >= 2 else 0)
            if channels > 0 and bins > 0:
                fixed[:channels, :bins] = waveform[:channels, :bins]
            waveform = fixed
        waveform_features.append(waveform)
    return (
        np.asarray(features, dtype=np.float32),
        positions,
        np.asarray(pulse_features, dtype=np.float32),
        np.asarray(waveform_features, dtype=np.float32),
        lids,
        detector_ids,
    )


def _build_edges(
    positions: np.ndarray,
    arrivals_usec: np.ndarray,
    log_rho: np.ndarray,
    rho: np.ndarray,
    detector_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = int(positions.shape[0])
    if n_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)

    undirected_edges: list[tuple[int, int, float, float, float, float, float]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if detector_ids[i] == detector_ids[j]:
                continue
            dxyz = positions[j] - positions[i]
            dist = float(np.linalg.norm(dxyz))
            if dist > ISING_EDGE_MAX_DISTANCE_KM:
                continue
            abs_dt = abs(float(arrivals_usec[j] - arrivals_usec[i]))
            if abs_dt > ISING_EDGE_MAX_DT_USEC:
                continue

            spatial = math.exp(-dist / ISING_SPACE_SCALE_KM)
            causal_excess = max(
                0.0,
                abs_dt - dist / LIGHT_SPEED_KM_PER_USEC - ISING_CAUSAL_GRACE_USEC,
            )
            causal = math.exp(-causal_excess / ISING_CAUSAL_TAU_USEC)
            signal = math.sqrt(math.log1p(max(float(rho[i]), 0.0)) * math.log1p(max(float(rho[j]), 0.0)))
            raw_weight = (
                spatial * causal * (1.0 + ISING_SIGNAL_WEIGHT * signal)
                - ISING_CAUSAL_PENALTY_PER_USEC * causal_excess
            )
            if math.isfinite(raw_weight) and abs(raw_weight) > 1.0e-12:
                undirected_edges.append((i, j, raw_weight, causal_excess, spatial, causal, dist))

    positive_degree = np.zeros(n_nodes, dtype=np.float64)
    for i, j, raw_weight, *_ in undirected_edges:
        if raw_weight > 0.0:
            positive_degree[i] += 1.0
            positive_degree[j] += 1.0

    edge_rows: list[list[float]] = []
    edge_index: list[tuple[int, int]] = []
    for i, j, raw_weight, causal_excess, spatial, causal, dist in undirected_edges:
        degree_i = max(float(positive_degree[i]), 1.0)
        degree_j = max(float(positive_degree[j]), 1.0)
        norm = (degree_i * degree_j) ** (0.5 * ISING_EDGE_DEGREE_POWER)
        weight = raw_weight / max(norm, 1.0e-12)
        for src, dst in ((i, j), (j, i)):
            dxyz = positions[dst] - positions[src]
            dt = float(arrivals_usec[dst] - arrivals_usec[src])
            edge_index.append((src, dst))
            edge_rows.append(
                [
                    float(dxyz[0]),
                    float(dxyz[1]),
                    float(dxyz[2]),
                    dist,
                    dt,
                    abs(dt),
                    dt / max(dist, 1.0e-6),
                    float(log_rho[dst] - log_rho[src]),
                    float(weight),
                    float(raw_weight),
                    float(causal_excess),
                    float(spatial),
                    float(causal),
                ]
            )

    if not edge_index:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)
    return np.asarray(edge_index, dtype=np.int64).T, np.asarray(edge_rows, dtype=np.float32)


def build_graph_event(
    record: BankRecord,
    detector_positions: dict[int, DetectorPosition] | None = None,
) -> GraphEvent | None:
    bank = record.bank
    combined_subs = _combine_sub_waveforms([sub for sub in bank.get("sub", []) if isinstance(sub, dict)])
    hits = [
        hit
        for sub in combined_subs
        for hit in [_extract_hit(bank, sub, detector_positions)]
        if hit is not None
    ]
    hits = _merge_hits_by_lid(hits)

    node_features, positions, pulse_features, waveform_features, node_lids, detector_ids = _build_node_features(hits)
    if node_features.shape[0] < GEOM_MIN_POINTS or len(set(detector_ids)) < GEOM_MIN_POINTS:
        return None

    arrival_rel = node_features[:, NODE_FEATURE_COLUMNS.index("first_arrival_usec_rel")]
    log_rho = node_features[:, NODE_FEATURE_COLUMNS.index("log10_first_rho")]
    rho = np.power(10.0, log_rho)
    edge_index, edge_features = _build_edges(positions, arrival_rel, log_rho, rho, detector_ids)
    target = _target_from_sim(bank.get("sim"))
    parttype, particle_label = _particle_label_from_sim(bank.get("sim"))

    date = int(bank.get("date", 0))
    time = int(bank.get("time", 0))
    usec = int(bank.get("usec", 0))
    event_id = f"{date:06d}_{time:06d}_{usec:06d}_{record.source_index:06d}"
    if bank.get("sim", {}).get("eventNum", -1) is not None and int(bank.get("sim", {}).get("eventNum", -1)) >= 0:
        event_id = f"MC{int(bank['sim']['eventNum'])}_{event_id}"

    metadata = {
        "event_id": event_id,
        "date": date,
        "time": time,
        "usec": usec,
        "source_path": record.source_path,
        "source_index": record.source_index,
        "source_kind": record.source_kind,
        "n_nodes": int(node_features.shape[0]),
        "n_sd": int(len(set(detector_ids))),
        "n_pulses": int(node_features.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "lids": ",".join(str(lid) for lid in node_lids),
        "unique_lids": ",".join(str(lid) for lid in sorted(set(detector_ids))),
        "graph_definition": "coincidence_analysis_ising_pulse_graph",
        "has_target": target is not None,
        "parttype": int(parttype) if parttype is not None else -1,
        "particle_label": float(particle_label) if particle_label is not None else math.nan,
        "particle_name": "iron" if particle_label == 1.0 else ("proton" if particle_label == 0.0 else "unknown"),
    }
    return GraphEvent(
        event_id=event_id,
        node_features=node_features,
        node_positions_km=positions,
        node_lids=np.asarray(node_lids, dtype=np.int64),
        edge_index=edge_index,
        edge_features=edge_features,
        pulse_features=pulse_features,
        waveform_features=waveform_features,
        target=target,
        particle_label=particle_label,
        metadata=metadata,
    )


def graph_columns() -> dict[str, list[str]]:
    return {
        "node_features": list(NODE_FEATURE_COLUMNS),
        "edge_features": list(EDGE_FEATURE_COLUMNS),
        "pulse_features": list(PULSE_FEATURE_COLUMNS),
        "waveform_features": list(WAVEFORM_FEATURE_CHANNELS),
        "target": list(TARGET_COLUMNS),
    }
