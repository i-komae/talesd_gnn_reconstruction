from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from talesd_gnn_reconstruction.cli import (
    _interleaved_selected_entries,
    _merge_candidate_reservoirs,
    _selected_path_chunks,
    _validate_mc_calibration_dates,
)
from talesd_gnn_reconstruction.dst_reader import _event_date


class ExportDateFilterTest(unittest.TestCase):
    def test_event_date_prefers_available_dst_bank_dates(self) -> None:
        self.assertEqual(_event_date({"rusdraw": {"yymmdd": 191004}}), 191004)
        self.assertEqual(_event_date({"talesdcalibev": {"date": 191005}}), 191005)
        self.assertIsNone(_event_date({"rusdraw": {"yymmdd": 0}}))

    def test_merge_candidate_reservoirs_tracks_selected_event_dates(self) -> None:
        selected_by_path, selected_entries, _seen_by_bin, selected_by_bin, selected_event_dates, _missing, _raw_events, _hit_events = (
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
        self.assertEqual([entry[3] for entry in selected_entries], [0, 1])
        self.assertEqual(selected_by_bin, {160: 2})
        self.assertEqual(selected_event_dates, {191003: 1, 191004: 1})

    def test_interleaved_selected_entries_keeps_short_source_runs(self) -> None:
        entries = []
        for source_offset, particle in enumerate(("proton", "iron")):
            for index in range(8):
                entries.append(
                    (
                        (particle, 170),
                        f"{particle}:{index}",
                        f"/tmp/{particle}.dst.gz",
                        source_offset * 100 + index,
                        17.0,
                        191004,
                        120000,
                    )
                )

        ordered = _interleaved_selected_entries(entries, seed=123, locality_run_size=2)
        self.assertEqual(sorted(entry[1] for entry in ordered), sorted(entry[1] for entry in entries))
        max_run = 1
        current = 1
        for previous, current_entry in zip(ordered, ordered[1:]):
            if previous[2] == current_entry[2] and previous[0] == current_entry[0]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1
        self.assertLessEqual(max_run, 2)

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

    def test_selected_path_chunks_pack_by_selected_event_count(self) -> None:
        chunks = _selected_path_chunks(
            ["a.dst", "b.dst", "c.dst", "d.dst"],
            {
                "a.dst": {1, 2},
                "b.dst": {3, 4},
                "c.dst": set(),
                "d.dst": {5},
            },
            shard_size=3,
        )

        self.assertEqual(chunks, [["a.dst"], ["b.dst", "d.dst"]])


if __name__ == "__main__":
    unittest.main()
