from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    COIN_TIME_WINDOW_USEC,
    FADC_BIN_WIDTH_USEC,
    FADC_PRETRIGGER_BINS,
)


@dataclass(frozen=True)
class LayerPulse:
    time_usec: float
    rise_bin: int
    fall_bin: int
    energy_mev: float


@dataclass(frozen=True)
class CoincidentPulse:
    time_usec: float
    onset_usec: float
    energy_mev: float
    upper_rise_bin: int
    upper_fall_bin: int
    lower_rise_bin: int
    lower_fall_bin: int


def bin_to_usec(bin_idx: int | float) -> float:
    return (float(bin_idx) - FADC_PRETRIGGER_BINS) * FADC_BIN_WIDTH_USEC


def sd_signal_search_10a(wf: np.ndarray, ped: float, ped_sigma: float) -> dict[str, list[float]]:
    """TA SDPreAnalysis-like pulse search used in the reference code."""

    trg_thr = 7.0
    win_size = 8
    n_sigma = 1.5

    wf_copy = np.asarray(wf, dtype=float).copy()
    wf_len = int(wf_copy.shape[0])
    sig_thr = n_sigma * float(ped_sigma)

    rise_bins: list[int] = []
    fall_bins: list[int] = []
    sums: list[float] = []

    i = 0
    while i <= wf_len - win_size:
        sig = float(np.sum(wf_copy[i : i + win_size]) - float(ped) * win_size)
        if sig >= trg_thr:
            i_start = i + win_size - 1
            sig_val = float(wf_copy[i_start] - ped)

            for j in range(i_start - 1, 0, -1):
                cnt = float(wf_copy[j] - ped)
                if cnt <= sig_thr:
                    break
                sig_val += cnt
                i_start = j

            for j in range(i_start, i + win_size):
                cnt = float(wf_copy[j] - ped)
                if cnt > sig_thr:
                    break
                sig_val -= cnt
                i_start = j + 1

            for j in range(i_start - 1, 0, -1):
                cnt = float(wf_copy[j] - ped)
                if cnt <= sig_thr:
                    break
                sig_val += cnt
                i_start = j

            i_start = max(i_start, 0)
            i_end = i + win_size - 1

            for j in range(i_end + 1, wf_len):
                cnt = float(wf_copy[j] - ped)
                if cnt <= sig_thr:
                    break
                sig_val += cnt
                i_end = j

            for j in range(i_end, i_start, -1):
                cnt = float(wf_copy[j] - ped)
                if cnt > sig_thr:
                    break
                sig_val -= cnt
                i_end = j - 1
            i_end += 1

            i_end = min(i_end, wf_len - 1)
            if i_end > i_start:
                rise_bins.append(int(i_start))
                fall_bins.append(int(i_end))
                sums.append(float(sig_val))

            i = i_end + 1 - win_size
            wf_copy[i_start:i_end] = ped
        i += 1

    return {"riseBin": rise_bins, "fallBin": fall_bins, "sumList": sums}


def _layer_pulses(result: dict[str, list[float]], mev2cnt: float) -> list[LayerPulse]:
    if mev2cnt <= 0:
        return []
    pulses: list[LayerPulse] = []
    for rise_bin, fall_bin, summed_counts in zip(result["riseBin"], result["fallBin"], result["sumList"]):
        pulses.append(
            LayerPulse(
                time_usec=bin_to_usec(rise_bin),
                rise_bin=int(rise_bin),
                fall_bin=int(fall_bin),
                energy_mev=max(float(summed_counts) / float(mev2cnt), 0.0),
            )
        )
    return pulses


def find_coincident_pulses(
    upper_result: dict[str, list[float]],
    lower_result: dict[str, list[float]],
    upper_mev2cnt: float,
    lower_mev2cnt: float,
    min_layer_mev: float = 0.3,
) -> list[CoincidentPulse]:
    upper = _layer_pulses(upper_result, upper_mev2cnt)
    lower = _layer_pulses(lower_result, lower_mev2cnt)
    coincident: list[CoincidentPulse] = []
    pairs: list[tuple[int, int, float, float]] = []

    for upper_index, up in enumerate(upper):
        if up.energy_mev <= min_layer_mev:
            continue
        for lower_index, low in enumerate(lower):
            if low.energy_mev <= min_layer_mev:
                continue
            if abs(up.time_usec - low.time_usec) > COIN_TIME_WINDOW_USEC:
                continue
            if up.fall_bin <= low.rise_bin or low.fall_bin <= up.rise_bin:
                continue

            coin_t = 0.5 * (up.time_usec + low.time_usec)
            onset_t = min(up.time_usec, low.time_usec)
            pairs.append((upper_index, lower_index, coin_t, onset_t))

    for upper_index, lower_index, coin_t, onset_t in pairs:
        use_upper = {
            charge_upper
            for charge_upper, _charge_lower, _charge_time, charge_onset in pairs
            if charge_onset >= onset_t
        }
        use_lower = {
            charge_lower
            for _charge_upper, charge_lower, _charge_time, charge_onset in pairs
            if charge_onset >= onset_t
        }
        upper_sum = sum(upper[index].energy_mev for index in use_upper)
        lower_sum = sum(lower[index].energy_mev for index in use_lower)
        up = upper[upper_index]
        low = lower[lower_index]
        coincident.append(
            CoincidentPulse(
                time_usec=coin_t,
                onset_usec=onset_t,
                energy_mev=0.5 * (upper_sum + lower_sum),
                upper_rise_bin=up.rise_bin,
                upper_fall_bin=up.fall_bin,
                lower_rise_bin=low.rise_bin,
                lower_fall_bin=low.fall_bin,
            )
        )

    coincident.sort(key=lambda pulse: pulse.time_usec)
    return coincident
