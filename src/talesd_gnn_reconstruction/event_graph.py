from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import (
    AREA_OF_SD_M2,
    DEFAULT_EDGE_K,
    DEFAULT_EDGE_RADIUS_KM,
    DEFAULT_MIN_NODES,
    EDGE_FEATURE_COLUMNS,
    FADC_BIN_WIDTH_USEC,
    MEV_PER_VMIP,
    NODE_FEATURE_COLUMNS,
    PULSE_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from .dst_reader import BankRecord
from .layout import DetectorPosition
from .signal import find_coincident_pulses, sd_signal_search_10a


@dataclass(frozen=True)
class GraphEvent:
    event_id: str
    node_features: np.ndarray
    node_positions_km: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    pulse_features: np.ndarray
    target: np.ndarray | None
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


def _build_node_features(
    hits: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    positions = np.asarray([hit["position"] for hit in hits], dtype=np.float32)
    total_rho = np.asarray([hit["total_rho"] for hit in hits], dtype=np.float32)
    weights = np.maximum(total_rho, 0.0) + 1.0e-6
    bary = np.sum(positions * weights[:, None], axis=0) / np.sum(weights)
    delta = positions - bary[None, :]
    radius = np.linalg.norm(delta, axis=1)
    first_pulses = [hit["pulses"][0] for hit in hits]
    arrival = np.asarray([pulse["arrival_usec"] for pulse in first_pulses], dtype=np.float32)
    arrival_min = float(np.min(arrival))
    arrival_rel = arrival - arrival_min

    features = []
    lids = []
    pulse_features = []
    for i, hit in enumerate(hits):
        first = hit["pulses"][0]
        first_rho = max(float(first["rho"]), 1.0e-6)
        total_rho_i = max(float(hit["total_rho"]), 1.0e-6)
        max_rho_i = max(float(hit["max_rho"]), 1.0e-6)
        fadc_peak = max(float(hit["fadc_peak"]), 1.0)
        lids.append(int(hit["lid"]))
        features.append(
            [
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                delta[i, 0],
                delta[i, 1],
                delta[i, 2],
                radius[i],
                arrival_rel[i],
                float(hit["trig_usec_rel"]),
                math.log10(first_rho),
                math.sqrt(first_rho),
                math.log10(total_rho_i),
                math.sqrt(total_rho_i),
                math.log10(max_rho_i),
                float(hit["num_pulses"]),
                float(hit["pulse_time_span_usec"]),
                float(hit["num_segments"]),
                float(hit["wf_length_usec"]),
                math.log10(fadc_peak),
                float(hit["upper_ped"]),
                float(hit["lower_ped"]),
                float(hit["upper_sigma"]),
                float(hit["lower_sigma"]),
            ]
        )
        first_arrival = float(first["arrival_usec"])
        for pulse in hit["pulses"]:
            rho_i = max(float(pulse["rho"]), 1.0e-6)
            pulse_order = int(pulse["order"])
            pulse_features.append(
                [
                    float(i),
                    float(pulse["arrival_usec"] - arrival_min),
                    float(pulse["arrival_usec"] - first_arrival),
                    math.log10(rho_i),
                    math.sqrt(rho_i),
                    float(pulse_order),
                    1.0 if pulse_order == 0 else 0.0,
                ]
            )
    return (
        np.asarray(features, dtype=np.float32),
        positions,
        np.asarray(pulse_features, dtype=np.float32).reshape(-1, len(PULSE_FEATURE_COLUMNS)),
        lids,
    )


def _build_edges(
    positions: np.ndarray,
    arrivals_usec: np.ndarray,
    log_rho: np.ndarray,
    radius_km: float = DEFAULT_EDGE_RADIUS_KM,
    k_nearest: int = DEFAULT_EDGE_K,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = int(positions.shape[0])
    if n_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)

    edges: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        diff = positions - positions[i]
        dist = np.linalg.norm(diff, axis=1)
        neighbor_order = np.argsort(dist)
        chosen = 0
        for j in neighbor_order:
            if i == int(j):
                continue
            if dist[j] <= radius_km or chosen < k_nearest:
                edges.add((i, int(j)))
                chosen += 1
            if chosen >= k_nearest and dist[j] > radius_km:
                break

    edge_rows: list[list[float]] = []
    edge_index: list[tuple[int, int]] = []
    for src, dst in sorted(edges):
        dxyz = positions[dst] - positions[src]
        dist = float(np.linalg.norm(dxyz))
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
            ]
        )

    return np.asarray(edge_index, dtype=np.int64).T, np.asarray(edge_rows, dtype=np.float32)


def build_graph_event(
    record: BankRecord,
    detector_positions: dict[int, DetectorPosition] | None = None,
    min_nodes: int = DEFAULT_MIN_NODES,
    edge_radius_km: float = DEFAULT_EDGE_RADIUS_KM,
    edge_k: int = DEFAULT_EDGE_K,
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
    if len(hits) < min_nodes:
        return None

    node_features, positions, pulse_features, node_lids = _build_node_features(hits)
    arrival_rel = node_features[:, NODE_FEATURE_COLUMNS.index("first_arrival_usec_rel")]
    log_rho = node_features[:, NODE_FEATURE_COLUMNS.index("log10_first_rho")]
    edge_index, edge_features = _build_edges(positions, arrival_rel, log_rho, edge_radius_km, edge_k)
    target = _target_from_sim(bank.get("sim"))

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
        "n_sd": int(node_features.shape[0]),
        "n_pulses": int(pulse_features.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "lids": ",".join(str(lid) for lid in node_lids),
        "unique_lids": ",".join(str(lid) for lid in node_lids),
        "has_target": target is not None,
    }
    return GraphEvent(
        event_id=event_id,
        node_features=node_features,
        node_positions_km=positions,
        edge_index=edge_index,
        edge_features=edge_features,
        pulse_features=pulse_features,
        target=target,
        metadata=metadata,
    )


def graph_columns() -> dict[str, list[str]]:
    return {
        "node_features": list(NODE_FEATURE_COLUMNS),
        "edge_features": list(EDGE_FEATURE_COLUMNS),
        "pulse_features": list(PULSE_FEATURE_COLUMNS),
        "target": list(TARGET_COLUMNS),
    }
