from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layout import DetectorPosition


class DstUnitExhaustionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BankRecord:
    bank: dict[str, Any]
    source_path: str
    source_index: int
    source_kind: str


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
) -> dict[str, Any]:
    mip = rusdraw.get("mip")
    pchped = rusdraw.get("pchped")
    lhpchped = rusdraw.get("lhpchped")
    rhpchped = rusdraw.get("rhpchped")
    fadc = rusdraw.get("fadc")

    return {
        "site": int(lid),
        "lid": int(lid),
        "dontUse": 0,
        "clock": int(_nested(rusdraw.get("clkcnt"), sid, default=0)),
        "maxClock": int(_nested(rusdraw.get("mclkcnt"), sid, default=50_000_000)),
        "uwf": _as_list(fadc[sid][1]) if fadc is not None else [],
        "lwf": _as_list(fadc[sid][0]) if fadc is not None else [],
        "upedAvr": _nested(pchped, sid, 1, default=0.0) / 8.0,
        "lpedAvr": _nested(pchped, sid, 0, default=0.0) / 8.0,
        "upedStdev": (_nested(rhpchped, sid, 1, default=0.0) - _nested(lhpchped, sid, 1, default=0.0)) / 8.0 / 2.35,
        "lpedStdev": (_nested(rhpchped, sid, 0, default=0.0) - _nested(lhpchped, sid, 0, default=0.0)) / 8.0 / 2.35,
        "umipMev2cnt": _positive(_nested(mip, sid, 1, default=0.0) / 2.4),
        "lmipMev2cnt": _positive(_nested(mip, sid, 0, default=0.0) / 2.4),
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
) -> dict[str, Any] | None:
    rusdraw = event.get("rusdraw")
    if not rusdraw:
        return None

    sub: list[dict[str, Any]] = []
    for sid, lid_value in enumerate(rusdraw.get("xxyy", [])):
        lid = int(lid_value)
        detector = detector_positions.get(lid)
        if detector is None:
            continue
        try:
            sub.append(_rusdraw_sub_to_talesd(rusdraw, sid, lid, detector))
        except Exception:
            continue

    if not sub:
        return None

    return {
        "eventCode": 0 if event.get("rusdmc") else 1,
        "date": int(rusdraw.get("yymmdd", 0)),
        "time": int(rusdraw.get("hhmmss", 0)),
        "usec": int(rusdraw.get("usec", 0)),
        "trgMode": 0,
        "sub": sub,
        "sim": _rusdmc_to_sim(event.get("rusdmc")),
    }


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
        "For large exports, keep file-level worker recycling enabled with --worker-max-files."
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
) -> Iterator[BankRecord]:
    """Stream TALE-SD-like calibev banks from data or MC DST files."""

    import dstio

    if kind not in {"auto", "data", "mc"}:
        raise ValueError(f"unsupported input kind: {kind}")

    emitted = 0
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
                    bank = event.get("talesdcalibev") or event.get("talesdcalib")
                    source_kind = "data"
                    if bank is None and (kind in {"auto", "mc"}):
                        if detector_positions is None:
                            raise ValueError("MC rusdraw input requires TALE-SD positions from talesdconst; pass --const-dst")
                        bank = _convert_rusdraw_event(event, detector_positions)
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
            if not skip_errors:
                raise
            print(f"warning: skipping unreadable DST {path}: {exc}")
