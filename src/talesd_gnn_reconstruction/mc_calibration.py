from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _seconds_from_hhmmss(value: int) -> int:
    hour = int(value) // 10000
    minute = (int(value) // 100) % 100
    second = int(value) % 100
    return hour * 3600 + minute * 60 + second


def _record_from_calib_sub(sub: dict[str, Any]) -> dict[str, Any] | None:
    lid_raw = sub.get("lid", sub.get("site"))
    if lid_raw is None:
        return None
    lid = int(lid_raw)
    return {
        "lid": lid,
        "site": int(sub.get("site", lid)),
        "dontUse": int(sub.get("dontUse", 0)),
        "umipMev2pe": float(sub.get("umipMev2pe", 0.0)),
        "lmipMev2pe": float(sub.get("lmipMev2pe", 0.0)),
        "umipMev2cnt": float(sub.get("umipMev2cnt", 0.0)),
        "lmipMev2cnt": float(sub.get("lmipMev2cnt", 0.0)),
        "upedAvr": float(sub.get("upedAvr", 0.0)),
        "lpedAvr": float(sub.get("lpedAvr", 0.0)),
        "upedStdev": float(sub.get("upedStdev", 0.0)),
        "lpedStdev": float(sub.get("lpedStdev", 0.0)),
    }


@dataclass
class TaleMcCalibrationDB:
    """TALE MC calibration lookup compatible with the Java SDCalibrator path."""

    calib_dir: Path

    def __post_init__(self) -> None:
        self.calib_dir = Path(self.calib_dir).expanduser()
        self._daily_cache: dict[int, list[tuple[int, dict[int, dict[str, Any]]]] | None] = {}
        self._source_cache: dict[int, bool] = {}

    def get_record(self, date: int, time: int, lid: int) -> dict[str, Any] | None:
        records = self._load_daily_records(date, time)
        if records is None:
            return None
        return records.get(int(lid))

    def has_calibration_source(self, date: int, time: int) -> bool:
        date = int(date)
        if date not in self._source_cache:
            self._source_cache[date] = self._daily_candidate(date) is not None
        return self._source_cache[date]

    def _daily_candidate(self, date: int) -> Path | None:
        stem = f"talesdcalib_pass2_{int(date):06d}.dst"
        for name in (stem, f"{stem}.gz", "talesdcalib_pass2_typical.dst", "talesdcalib_pass2_typical.dst.gz"):
            path = self.calib_dir / name
            if path.is_file():
                return path
        return None

    def _load_daily_records(self, date: int, time: int) -> dict[int, dict[str, Any]] | None:
        date = int(date)
        if date not in self._daily_cache:
            path = self._daily_candidate(date)
            if path is None:
                self._daily_cache[date] = None
                return None

            import dstio

            daily_records: list[tuple[int, dict[int, dict[str, Any]]]] = []
            with dstio.open(str(path), banks=["talesdcalib", "talesdcalibev"]) as dst:
                for event in dst:
                    bank = event.get("talesdcalib") or event.get("talesdcalibev")
                    if not bank:
                        continue
                    bank_time = int(bank.get("time", time))
                    records: dict[int, dict[str, Any]] = {}
                    for sub in bank.get("sub", []):
                        record = _record_from_calib_sub(sub)
                        if record is not None:
                            records[int(record["lid"])] = record
                    daily_records.append((_seconds_from_hhmmss(bank_time), records))
            self._daily_cache[date] = daily_records
        daily_records = self._daily_cache[date]
        if not daily_records:
            return None
        target_sec = _seconds_from_hhmmss(time)
        selected = daily_records[-1][1]
        for bank_sec, records in daily_records:
            if abs(bank_sec - target_sec) < 300:
                selected = records
                break
        return selected
