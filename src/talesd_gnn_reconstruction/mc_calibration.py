from __future__ import annotations

from collections import OrderedDict
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
    max_time_difference_seconds: int = 300
    max_record_cache_days: int = 8

    def __post_init__(self) -> None:
        self.calib_dir = Path(self.calib_dir).expanduser()
        self._daily_cache: OrderedDict[int, list[tuple[int, dict[int, dict[str, Any]]]] | None] = OrderedDict()
        self._daily_time_cache: dict[int, list[int] | None] = {}
        self._source_cache: dict[int, bool] = {}
        self._time_cache: dict[tuple[int, int], bool] = {}

    def get_records(self, date: int, time: int) -> dict[int, dict[str, Any]] | None:
        return self._select_daily_records(date, time)

    def get_record(self, date: int, time: int, lid: int) -> dict[str, Any] | None:
        records = self.get_records(date, time)
        if records is None:
            return None
        return records.get(int(lid))

    def has_calibration_source(self, date: int, time: int) -> bool:
        date = int(date)
        if date not in self._source_cache:
            self._source_cache[date] = self._daily_candidate(date) is not None
        return self._source_cache[date]

    def has_calibration_time(self, date: int, time: int) -> bool:
        key = (int(date), int(time))
        if key not in self._time_cache:
            self._time_cache[key] = self._has_daily_time(*key)
        return self._time_cache[key]

    def _daily_candidate(self, date: int) -> Path | None:
        stem = f"talesdcalib_pass2_{int(date):06d}.dst"
        for name in (stem, f"{stem}.gz", "talesdcalib_pass2_typical.dst", "talesdcalib_pass2_typical.dst.gz"):
            path = self.calib_dir / name
            if path.is_file():
                return path
        return None

    def _load_daily_records(self, date: int) -> list[tuple[int, dict[int, dict[str, Any]]]] | None:
        date = int(date)
        if date in self._daily_cache:
            self._daily_cache.move_to_end(date)
            return self._daily_cache[date]

        path = self._daily_candidate(date)
        if path is None:
            self._daily_cache[date] = None
            self._daily_time_cache[date] = None
            self._trim_record_cache()
            return None

        import dstio

        daily_records: list[tuple[int, dict[int, dict[str, Any]]]] = []
        daily_times: list[int] = []
        with dstio.open(str(path), banks=["talesdcalib", "talesdcalibev"]) as dst:
            for event in dst:
                bank = event.get("talesdcalib") or event.get("talesdcalibev")
                if not bank:
                    continue
                bank_time_raw = bank.get("time")
                if bank_time_raw is None:
                    continue
                bank_sec = _seconds_from_hhmmss(int(bank_time_raw))
                records: dict[int, dict[str, Any]] = {}
                for sub in bank.get("sub", []):
                    record = _record_from_calib_sub(sub)
                    if record is not None:
                        records[int(record["lid"])] = record
                daily_records.append((bank_sec, records))
                daily_times.append(bank_sec)
        self._daily_cache[date] = daily_records
        self._daily_time_cache[date] = daily_times
        self._trim_record_cache()
        return daily_records

    def _load_daily_times(self, date: int) -> list[int] | None:
        date = int(date)
        if date in self._daily_time_cache:
            return self._daily_time_cache[date]
        if date not in self._daily_cache:
            path = self._daily_candidate(date)
            if path is None:
                self._daily_time_cache[date] = None
                return None

            import dstio

            daily_times: list[int] = []
            with dstio.open(str(path), banks=["talesdcalib", "talesdcalibev"]) as dst:
                for event in dst:
                    bank = event.get("talesdcalib") or event.get("talesdcalibev")
                    if not bank:
                        continue
                    bank_time_raw = bank.get("time")
                    if bank_time_raw is None:
                        continue
                    daily_times.append(_seconds_from_hhmmss(int(bank_time_raw)))
            self._daily_time_cache[date] = daily_times
        else:
            daily_records = self._daily_cache[date]
            self._daily_time_cache[date] = None if daily_records is None else [bank_sec for bank_sec, _records in daily_records]
        return self._daily_time_cache[date]

    def _has_daily_time(self, date: int, time: int) -> bool:
        daily_times = self._load_daily_times(date)
        if not daily_times:
            return False
        target_sec = _seconds_from_hhmmss(time)
        return any(abs(bank_sec - target_sec) < int(self.max_time_difference_seconds) for bank_sec in daily_times)

    def _select_daily_records(self, date: int, time: int) -> dict[int, dict[str, Any]] | None:
        daily_records = self._load_daily_records(date)
        if not daily_records:
            return None
        target_sec = _seconds_from_hhmmss(time)
        for bank_sec, records in daily_records:
            if abs(bank_sec - target_sec) < int(self.max_time_difference_seconds):
                return records
        return None

    def _trim_record_cache(self) -> None:
        max_days = int(self.max_record_cache_days)
        if max_days <= 0:
            return
        while len(self._daily_cache) > max_days:
            self._daily_cache.popitem(last=False)


_MC_CALIBRATION_CACHE: dict[str, TaleMcCalibrationDB] = {}


def get_cached_mc_calibration_db(calib_dir: str | Path | None) -> TaleMcCalibrationDB | None:
    if calib_dir is None:
        return None
    key = str(Path(calib_dir).expanduser())
    db = _MC_CALIBRATION_CACHE.get(key)
    if db is None:
        db = TaleMcCalibrationDB(Path(key))
        _MC_CALIBRATION_CACHE[key] = db
    return db
