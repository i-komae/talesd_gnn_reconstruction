from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from talesd_gnn_reconstruction.cli import _merge_candidate_reservoirs, _validate_mc_calibration_dates
from talesd_gnn_reconstruction.dst_reader import _event_date


class ExportDateFilterTest(unittest.TestCase):
    def test_event_date_prefers_available_dst_bank_dates(self) -> None:
        self.assertEqual(_event_date({"rusdraw": {"yymmdd": 191004}}), 191004)
        self.assertEqual(_event_date({"talesdcalibev": {"date": 191005}}), 191005)
        self.assertIsNone(_event_date({"rusdraw": {"yymmdd": 0}}))

    def test_merge_candidate_reservoirs_tracks_selected_event_dates(self) -> None:
        selected_by_path, _seen_by_bin, selected_by_bin, selected_event_dates, _missing, _raw_events, _hit_events = (
            _merge_candidate_reservoirs(
                [
                    {
                        "path": "/tmp/a.dst.gz",
                        "raw_events": 2,
                        "hit_events": 2,
                        "seen_by_bin": {160: 2},
                        "reservoirs": {
                            160: [
                                (-0.2, "/tmp/a.dst.gz:0", 0, 16.0, 191003, 120000),
                                (-0.1, "/tmp/a.dst.gz:1", 1, 16.0, 191004, 235753),
                            ]
                        },
                    }
                ],
                per_bin_limit=2,
            )
        )

        self.assertEqual(selected_by_path, {"/tmp/a.dst.gz": {0, 1}})
        self.assertEqual(selected_by_bin, {160: 2})
        self.assertEqual(selected_event_dates, {191003: 1, 191004: 1})

    def test_validate_mc_calibration_dates_reports_missing_selected_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_dir = Path(tmpdir)
            (calib_dir / "talesdcalib_pass2_191003.dst").touch()

            with self.assertRaises(SystemExit) as raised:
                _validate_mc_calibration_dates(
                    calib_dir,
                    {191003: 2, 191004: 5},
                    context="unit-test",
                )

        self.assertIn("191004(5)", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
