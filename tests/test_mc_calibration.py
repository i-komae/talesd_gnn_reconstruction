from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from talesd_gnn_reconstruction.mc_calibration import TaleMcCalibrationDB


class _FakeDst:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    def __enter__(self) -> "_FakeDst":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def __iter__(self):
        return iter(self._events)


class TaleMcCalibrationTest(unittest.TestCase):
    def test_reads_matching_talesdcalib_pass2_dst_event(self) -> None:
        date = 160101
        events = [
            {
                "talesdcalib": {
                    "time": 115500,
                    "sub": [{"lid": 5401, "umipMev2cnt": 1.0, "lmipMev2cnt": 2.0}],
                }
            },
            {
                "talesdcalib": {
                    "time": 120100,
                    "sub": [
                        {
                            "lid": 5401,
                            "dontUse": 0,
                            "umipMev2pe": 11.0,
                            "lmipMev2pe": 12.0,
                            "umipMev2cnt": 21.0,
                            "lmipMev2cnt": 22.0,
                            "upedAvr": 31.0,
                            "lpedAvr": 32.0,
                            "upedStdev": 2.1,
                            "lpedStdev": 2.2,
                        }
                    ],
                }
            },
        ]

        open_count = 0

        def fake_open(*_args, **_kwargs):
            nonlocal open_count
            open_count += 1
            return _FakeDst(events)

        fake_dstio = types.SimpleNamespace(open=fake_open)
        old_dstio = sys.modules.get("dstio")
        sys.modules["dstio"] = fake_dstio
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                (Path(tmpdir) / f"talesdcalib_pass2_{date:06d}.dst").touch()
                db = TaleMcCalibrationDB(Path(tmpdir))
                record = db.get_record(date, 120000, 5401)
                records = db.get_records(date, 120000)
                missing_time_record = db.get_record(date, 130000, 5401)
                has_matching_time = db.has_calibration_time(date, 120000)
                has_missing_time = db.has_calibration_time(date, 130000)
        finally:
            if old_dstio is None:
                sys.modules.pop("dstio", None)
            else:
                sys.modules["dstio"] = old_dstio

        self.assertIsNotNone(record)
        assert record is not None
        self.assertIsNotNone(records)
        assert records is not None
        self.assertIs(record, records[5401])
        self.assertEqual(record["lid"], 5401)
        self.assertEqual(record["umipMev2cnt"], 21.0)
        self.assertEqual(record["lmipMev2cnt"], 22.0)
        self.assertEqual(record["upedAvr"], 31.0)
        self.assertEqual(record["lpedAvr"], 32.0)
        self.assertIsNone(missing_time_record)
        self.assertTrue(has_matching_time)
        self.assertFalse(has_missing_time)
        self.assertEqual(open_count, 1)


if __name__ == "__main__":
    unittest.main()
