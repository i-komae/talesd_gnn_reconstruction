from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layout import DetectorPosition
from .mc_calibration import TaleMcCalibrationDB, get_cached_mc_calibration_db


class DstUnitExhaustionError(RuntimeError):
    pass


class MissingMcCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class BankRecord:
    bank: dict[str, Any]
    source_path: str
    source_index: int
    source_kind: str


def _get_mc_calibration_db(calib_dir: str | Path | None) -> TaleMcCalibrationDB | None:
    return get_cached_mc_calibration_db(calib_dir)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _nested(value: Any, *keys: int, default: float = 0.0) -> float:
    try:
        obj = value
        for key in keys:
            obj = obj[key]
        return float(obj)
    except Exception:
        return float(default)


def _positive(value: float, fallback: float = 1.0e-6) -> float:
    value = float(value)
    if value > 0.0 and math.isfinite(value):
        return value
    return fallback


def _rusdmc_to_sim(rusdmc: dict[str, Any] | None) -> dict[str, Any]:
    if not rusdmc:
        return {}
    theta = float(rusdmc.get("theta", 0.0))
    phi = float(rusdmc.get("phi", 0.0)) + math.pi
    azimuth = math.degrees(phi) % 360.0
    if azimuth <= 0.0:
        azimuth += 360.0
    return {
        "primaryEnergy": float(rusdmc.get("energy", 0.0)) * 1.0e18,
        "primaryCosZenith": math.cos(theta),
        "primaryAzimuth": azimuth,
        "primaryArrivalTimeFromPps": float(rusdmc.get("tc", 0.0)) * 20.0 / 1.0e9,
        "primaryCorePosX": _nested(rusdmc.get("corexyz"), 0, default=0.0) / 1.0e2,
        "primaryCorePosY": _nested(rusdmc.get("corexyz"), 1, default=0.0) / 1.0e2,
        "primaryCorePosZ": _nested(rusdmc.get("corexyz"), 2, default=0.0) / 1.0e2,
        "primaryParticleId": int(rusdmc.get("parttype", -1)),
        "eventNum": int(rusdmc.get("event_num", -1)),
    }


def _rusdraw_sub_to_talesd(
    rusdraw: dict[str, Any],
    sid: int,
    lid: int,
    detector: DetectorPosition,
    calibration_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mip = rusdraw.get("mip")
    pchped = rusdraw.get("pchped")
    lhpchped = rusdraw.get("lhpchped")
    rhpchped = rusdraw.get("rhpchped")
    fadc = rusdraw.get("fadc")
    calib = calibration_record or {}

    return {
        "site": int(lid),
        "lid": int(lid),
        "dontUse": int(calib.get("dontUse", 0)),
        "clock": int(_nested(rusdraw.get("clkcnt"), sid, default=0)),
        "maxClock": int(_nested(rusdraw.get("mclkcnt"), sid, default=50_000_000)),
        "uwf": _as_list(fadc[sid][1]) if fadc is not None else [],
        "lwf": _as_list(fadc[sid][0]) if fadc is not None else [],
        "upedAvr": float(calib.get("upedAvr", _nested(pchped, sid, 1, default=0.0) / 8.0)),
        "lpedAvr": float(calib.get("lpedAvr", _nested(pchped, sid, 0, default=0.0) / 8.0)),
        "upedStdev": _positive(
            float(calib.get("upedStdev", (_nested(rhpchped, sid, 1, default=0.0) - _nested(lhpchped, sid, 1, default=0.0)) / 8.0 / 2.35))
        ),
        "lpedStdev": _positive(
            float(calib.get("lpedStdev", (_nested(rhpchped, sid, 0, default=0.0) - _nested(lhpchped, sid, 0, default=0.0)) / 8.0 / 2.35))
        ),
        "umipMev2cnt": _positive(float(calib.get("umipMev2cnt", _nested(mip, sid, 1, default=0.0) / 2.4))),
        "lmipMev2cnt": _positive(float(calib.get("lmipMev2cnt", _nested(mip, sid, 0, default=0.0) / 2.4))),
        "umipMev2pe": float(calib.get("umipMev2pe", 0.0)),
        "lmipMev2pe": float(calib.get("lmipMev2pe", 0.0)),
        "posX": detector.x_km * 1.0e3,
        "posY": detector.y_km * 1.0e3,
        "posZ": detector.z_km * 1.0e3,
        "mip": _as_list(mip[sid]) if mip is not None else [],
        "pchped": _as_list(pchped[sid]) if pchped is not None else [],
        "lhpchped": _as_list(lhpchped[sid]) if lhpchped is not None else [],
        "rhpchped": _as_list(rhpchped[sid]) if rhpchped is not None else [],
    }


def _convert_rusdraw_event(
    event: dict[str, Any],
    detector_positions: dict[int, DetectorPosition],
    mc_calibration: TaleMcCalibrationDB | None = None,
) -> dict[str, Any] | None:
    rusdraw = event.get("rusdraw")
    if not rusdraw:
        return None
    date = int(rusdraw.get("yymmdd", 0))
    time_value = int(rusdraw.get("hhmmss", 0))
    calibration_records = mc_calibration.get_records(date, time_value) if mc_calibration is not None else None
    if mc_calibration is not None and calibration_records is None:
        raise MissingMcCalibrationError(
            f"TALE MC calibration source/time not found for event date/time {date:06d} {time_value:06d} "
            f"in {mc_calibration.calib_dir}"
        )

    sub: list[dict[str, Any]] = []
    for sid, lid_value in enumerate(rusdraw.get("xxyy", [])):
        lid = int(lid_value)
        detector = detector_positions.get(lid)
        if detector is None:
            continue
        calibration_record = calibration_records.get(lid) if calibration_records is not None else None
        if calibration_records is not None and calibration_record is None:
            continue
        try:
            sub.append(_rusdraw_sub_to_talesd(rusdraw, sid, lid, detector, calibration_record=calibration_record))
        except (KeyError, IndexError, TypeError, ValueError):
            continue

    if not sub:
        return None

    return {
        "eventCode": 0 if event.get("rusdmc") else 1,
        "date": date,
        "time": time_value,
        "usec": int(rusdraw.get("usec", 0)),
        "trgMode": 0,
        "sub": sub,
        "sim": _rusdmc_to_sim(event.get("rusdmc")),
    }


def _event_date(event: dict[str, Any]) -> int | None:
    for bank_name in ("rusdraw", "talesdcalibev", "talesdcalib"):
        bank = event.get(bank_name)
        if not bank:
            continue
        try:
            date = int(bank.get("yymmdd", bank.get("date", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if date > 0:
            return date
    return None


def _event_time(event: dict[str, Any]) -> int | None:
    for bank_name in ("rusdraw", "talesdcalibev", "talesdcalib"):
        bank = event.get(bank_name)
        if not bank:
            continue
        try:
            time_value = int(bank.get("hhmmss", bank.get("time", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if time_value > 0:
            return time_value
    return None


def _bank_filter(kind: str) -> list[str]:
    if kind == "mc":
        return ["rusdraw", "rusdmc"]
    if kind == "data":
        return ["talesdcalibev", "talesdcalib"]
    return ["talesdcalibev", "talesdcalib", "rusdraw", "rusdmc"]


def _is_dst_unit_exhaustion(exc: BaseException) -> bool:
    if isinstance(exc, DstUnitExhaustionError):
        return True
    message = str(exc)
    return "unit 1024" in message or "out of allowed range [0-1023]" in message


def _raise_dst_unit_exhaustion(exc: BaseException) -> None:
    raise DstUnitExhaustionError(
        "DST unit handles were exhausted in this process. "
        "Update and rebuild dstio so closed DST units are reused, or use --worker-max-files "
        "as a temporary workaround."
    ) from exc


def iter_dst_banks(
    paths: Sequence[str | Path],
    detector_positions: dict[int, DetectorPosition] | None = None,
    kind: str = "auto",
    max_events: int | None = None,
    require_trigger_mode0: bool = True,
    skip_errors: bool = False,
    source_indices: set[int] | None = None,
    open_retries: int = 1,
    open_retry_delay: float = 1.0,
    mc_calib_dir: str | Path | None = None,
    min_event_date: int | None = None,
    skip_missing_mc_calibration: bool = False,
) -> Iterator[BankRecord]:
    """Stream TALE-SD-like calibev banks from data or MC DST files."""

    import dstio

    if kind not in {"auto", "data", "mc"}:
        raise ValueError(f"unsupported input kind: {kind}")
    if kind == "mc" and mc_calib_dir is None:
        raise ValueError("MC rusdraw input requires --mc-calib-dir to load TALE-SD calibration")

    emitted = 0
    mc_calibration = _get_mc_calibration_db(mc_calib_dir)
    for path_obj in paths:
        path = Path(path_obj).expanduser()
        dst_handle = None
        try:
            last_exc: Exception | None = None
            for attempt in range(max(int(open_retries), 1)):
                try:
                    dst_handle = dstio.open(str(path), banks=_bank_filter(kind))
                    break
                except Exception as exc:
                    if _is_dst_unit_exhaustion(exc):
                        _raise_dst_unit_exhaustion(exc)
                    last_exc = exc
                    if attempt + 1 < max(int(open_retries), 1):
                        time.sleep(max(float(open_retry_delay), 0.0) * (attempt + 1))
            if dst_handle is None:
                if last_exc is not None:
                    raise last_exc
                raise OSError(f"failed to open DST: {path}")
            with dst_handle as dst:
                for source_index, event in enumerate(dst):
                    if source_indices is not None and source_index not in source_indices:
                        continue
                    if min_event_date is not None:
                        event_date = _event_date(event)
                        if event_date is None or event_date < int(min_event_date):
                            continue
                    if skip_missing_mc_calibration and mc_calibration is not None and kind in {"auto", "mc"}:
                        event_date = _event_date(event)
                        event_time = _event_time(event)
                        if (
                            event_date is None
                            or event_time is None
                            or not mc_calibration.has_calibration_time(event_date, event_time)
                        ):
                            continue
                    bank = event.get("talesdcalibev") or event.get("talesdcalib")
                    source_kind = "data"
                    if bank is None and (kind in {"auto", "mc"}):
                        if detector_positions is None:
                            raise ValueError("MC rusdraw input requires TALE-SD positions from talesdconst; pass --const-dst")
                        if mc_calibration is None:
                            raise ValueError("MC rusdraw input requires --mc-calib-dir to load TALE-SD calibration")
                        bank = _convert_rusdraw_event(event, detector_positions, mc_calibration=mc_calibration)
                        source_kind = "mc"
                    if bank is None:
                        continue
                    if require_trigger_mode0 and int(bank.get("trgMode", 0)) != 0:
                        continue
                    yield BankRecord(
                        bank=bank,
                        source_path=str(path),
                        source_index=source_index,
                        source_kind=source_kind,
                    )
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        return
        except Exception as exc:
            if _is_dst_unit_exhaustion(exc):
                _raise_dst_unit_exhaustion(exc)
            if isinstance(exc, MissingMcCalibrationError):
                raise
            if not skip_errors:
                raise
            print(f"warning: skipping unreadable DST {path}: {exc}")
