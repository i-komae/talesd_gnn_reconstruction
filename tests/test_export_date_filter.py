from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from talesd_gnn_reconstruction.cli import (
    HeteroSourceFileManifest,
    HeteroSourceGroupManifest,
    HeteroSelectionCandidate,
    _allocate_cell_quotas,
    _allocate_hetero_source_group_quotas,
    _dat_tag_from_path,
    _energy_bin_code_from_dat_tag,
    _hetero_refill_bin_targets,
    _hetero_selection_summary,
    _interleaved_selected_entries,
    _merge_candidate_reservoirs,
    _select_balanced_hetero_candidates,
    _selected_path_chunks,
    _source_group_selection_payloads,
    _source_group_key_for_path,
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

    def test_balanced_hetero_selection_round_robins_source_groups(self) -> None:
        candidates = []
        for source_group in ("DAT000001", "DAT000002", "DAT000003"):
            for index in range(6):
                candidates.append(
                    HeteroSelectionCandidate(
                        bin_key=("proton", 170),
                        unique_id=f"{source_group}:{index}",
                        source_path=f"/mc/{source_group}_gea_trg_001.dst.gz",
                        source_group=source_group,
                        source_index=index,
                        log10_energy=17.0,
                        particle="proton",
                        zenith_deg=30.0,
                        azimuth_deg=30.0 * index,
                        core_x_km=float(index),
                        core_y_km=0.0,
                        date=191004,
                        time_value=100000 + index,
                        sort_key=0.01 * index,
                        balance_key=(str(index % 2), str(index % 3), str(index), "0", "0"),
                    )
                )

        selected = _select_balanced_hetero_candidates(candidates, per_bin=6, seed=123)
        counts: dict[str, int] = {}
        for candidate in selected:
            counts[candidate.source_group] = counts.get(candidate.source_group, 0) + 1
        self.assertEqual(set(counts), {"DAT000001", "DAT000002", "DAT000003"})
        self.assertTrue(all(count == 2 for count in counts.values()))

    def test_balanced_hetero_selection_interleaves_cells_within_source(self) -> None:
        candidates = []
        for index in range(8):
            cell = ("cell-a",) if index < 4 else ("cell-b",)
            candidates.append(
                HeteroSelectionCandidate(
                    bin_key=("iron", 180),
                    unique_id=f"DAT000010:{index}",
                    source_path="/mc/DAT000010_gea_trg_001.dst.gz",
                    source_group="DAT000010",
                    source_index=index,
                    log10_energy=18.0,
                    particle="iron",
                    zenith_deg=45.0,
                    azimuth_deg=45.0 * index,
                    core_x_km=0.1 * index,
                    core_y_km=-0.1 * index,
                    date=191004,
                    time_value=120000 + index,
                    sort_key=0.01 * index,
                    balance_key=cell,
                )
            )

        selected = _select_balanced_hetero_candidates(candidates, per_bin=4, seed=456)
        cells = {candidate.balance_key for candidate in selected}
        self.assertEqual(cells, {("cell-a",), ("cell-b",)})

    def test_hetero_filename_energy_code_comes_from_dat_tag(self) -> None:
        path = "/mc/proton/DAT123416_gea_trg_007.dst.gz"

        self.assertEqual(_dat_tag_from_path(path), "DAT123416")
        self.assertEqual(_energy_bin_code_from_dat_tag("DAT123416"), "16")
        self.assertTrue(_source_group_key_for_path(path).endswith("DAT123416"))

    def test_hetero_source_group_quotas_are_group_balanced(self) -> None:
        groups = {}
        for index in range(3):
            source_group = f"/mc/proton/DAT0001{index}16"
            file_manifest = HeteroSourceFileManifest(
                path=f"{source_group}_gea_trg_000.dst.gz",
                source_group=source_group,
                dat_tag=f"DAT0001{index}16",
                energy_bin_code="16",
                particle="proton",
                gea_trg_index=0,
                source_zenith_deg=20.0 + index,
                eligible_event_count=100,
                date_counts={"260606": 100},
                cell_counts={("0", "0", "0", "0"): 100},
            )
            groups[source_group] = HeteroSourceGroupManifest(
                source_group=source_group,
                dat_tag=file_manifest.dat_tag,
                energy_bin_code="16",
                particle="proton",
                source_zenith_deg=file_manifest.source_zenith_deg,
                eligible_event_count=100,
                files=(file_manifest,),
                date_counts={"260606": 100},
                cell_counts={("0", "0", "0", "0"): 100},
            )

        quotas, summary = _allocate_hetero_source_group_quotas(
            groups,
            per_bin=10,
            seed=123,
            stratify_particle=True,
        )

        self.assertEqual(sum(quotas.values()), 10)
        self.assertEqual(max(quotas.values()) - min(quotas.values()), 1)
        self.assertEqual(summary["by_bin"]["proton:16"]["source_groups"], 3)

    def test_hetero_source_group_refill_quotas_are_group_balanced(self) -> None:
        groups = {}
        for index in range(3):
            source_group = f"/mc/proton/DAT0001{index}16"
            groups[source_group] = HeteroSourceGroupManifest(
                source_group=source_group,
                dat_tag=f"DAT0001{index}16",
                energy_bin_code="16",
                particle="proton",
                source_zenith_deg=20.0,
                eligible_event_count=100,
                files=(),
                date_counts={"260606": 100},
                cell_counts={("0", "0", "0", "0"): 100},
            )

        quotas, summary = _allocate_hetero_source_group_quotas(
            groups,
            per_bin=50,
            seed=123,
            stratify_particle=True,
            bin_targets={"proton:16": 6},
        )

        self.assertEqual(sum(quotas.values()), 6)
        self.assertEqual(set(quotas.values()), {2})
        self.assertEqual(summary["by_bin"]["proton:16"]["assigned_events"], 6)

    def test_hetero_refill_targets_use_actual_write_efficiency(self) -> None:
        targets = _hetero_refill_bin_targets(
            {"proton:16": 100000},
            {"proton:16": 1490},
            {"proton:16": 100000},
            safety_factor=1.25,
            min_efficiency=0.01,
        )

        self.assertEqual(targets["proton:16"], 8264262)

    def test_hetero_refill_targets_use_min_efficiency_when_no_graphs_written(self) -> None:
        targets = _hetero_refill_bin_targets(
            {"proton:16": 100000},
            {"proton:16": 0},
            {"proton:16": 100000},
            safety_factor=1.0,
            min_efficiency=0.01,
        )

        self.assertEqual(targets["proton:16"], 10000000)

    def test_hetero_cell_quotas_follow_cell_counts(self) -> None:
        quotas = _allocate_cell_quotas(
            {("az0", "x0", "y0", "t0"): 80, ("az1", "x0", "y0", "t0"): 20},
            target=10,
            seed=123,
            source_group="DAT000116",
        )

        self.assertEqual(quotas[("az0", "x0", "y0", "t0")], 8)
        self.assertEqual(quotas[("az1", "x0", "y0", "t0")], 2)

    def test_hetero_source_group_selection_payload_filters_excluded_indices(self) -> None:
        group = HeteroSourceGroupManifest(
            source_group="/mc/proton/DAT000116",
            dat_tag="DAT000116",
            energy_bin_code="16",
            particle="proton",
            source_zenith_deg=20.0,
            eligible_event_count=10,
            files=(
                HeteroSourceFileManifest(
                    path="/mc/proton/DAT000116_gea_trg_000.dst.gz",
                    source_group="/mc/proton/DAT000116",
                    dat_tag="DAT000116",
                    energy_bin_code="16",
                    particle="proton",
                    gea_trg_index=0,
                    source_zenith_deg=20.0,
                    eligible_event_count=5,
                    date_counts={"260606": 5},
                    cell_counts={("0", "0", "0", "0"): 5},
                ),
                HeteroSourceFileManifest(
                    path="/mc/proton/DAT000116_gea_trg_001.dst.gz",
                    source_group="/mc/proton/DAT000116",
                    dat_tag="DAT000116",
                    energy_bin_code="16",
                    particle="proton",
                    gea_trg_index=1,
                    source_zenith_deg=20.0,
                    eligible_event_count=5,
                    date_counts={"260606": 5},
                    cell_counts={("0", "0", "0", "0"): 5},
                ),
            ),
            date_counts={"260606": 10},
            cell_counts={("0", "0", "0", "0"): 10},
        )

        payloads = _source_group_selection_payloads(
            {group.source_group: group},
            {group.source_group: 4},
            argparse.Namespace(seed=123),
            {
                "/mc/proton/DAT000116_gea_trg_001.dst.gz": {2, 3},
                "/mc/iron/DAT999918_gea_trg_000.dst.gz": {99},
            },
        )

        self.assertEqual(len(payloads), 1)
        _payload_group, quota, _args, excluded = payloads[0]
        self.assertEqual(quota, 4)
        self.assertEqual(excluded, {"/mc/proton/DAT000116_gea_trg_001.dst.gz": {2, 3}})

    def test_hetero_selection_summary_includes_time_and_dynamic_bins(self) -> None:
        candidates = [
            HeteroSelectionCandidate(
                bin_key=("proton", 175),
                unique_id="DAT000001:0",
                source_path="/mc/DAT000001_gea_trg_001.dst.gz",
                source_group="DAT000001",
                source_index=0,
                log10_energy=17.5,
                particle="proton",
                zenith_deg=12.0,
                azimuth_deg=45.0,
                core_x_km=0.4,
                core_y_km=-0.6,
                date=260606,
                time_value=123000,
                sort_key=0.1,
                balance_key=("1", "1", "0", "-1", "12"),
            )
        ]
        summary = _hetero_selection_summary(
            candidates,
            bin_width=0.1,
            zenith_bin_width_deg=10.0,
            azimuth_bin_width_deg=30.0,
            core_bin_width_km=0.5,
            time_bin_width_sec=3600,
        )
        self.assertEqual(summary["events"], 1)
        self.assertEqual(summary["by_time_bin"]["12"], 1)
        self.assertEqual(summary["by_zenith_bin"]["1"], 1)


if __name__ == "__main__":
    unittest.main()
