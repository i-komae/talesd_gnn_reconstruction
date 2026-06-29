from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from talesd_gnn_reconstruction import dataset as graph_dataset
from talesd_gnn_reconstruction.constants import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS
from talesd_gnn_reconstruction.dst_reader import BankRecord
from talesd_gnn_reconstruction.event_graph import GraphEvent, build_graph_event
from talesd_gnn_reconstruction.graph_io import create_graph_file, write_graph
from talesd_gnn_reconstruction.homogeneous_schema import (
    LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS,
    LEGACY_FLAT50000_EDGE_FEATURE_COLUMNS,
    LEGACY_FLAT50000_NODE_FEATURE_COLUMNS,
    LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS,
    LEGACY_FLAT50000_TARGET_COLUMNS,
    LEGACY_FLAT50000_WAVEFORM_FEATURE_CHANNELS,
    homogeneous_dataset_kwargs_for_schema,
    legacy_flat50000_checkpoint_matches,
    normalize_homogeneous_schema,
)
from scripts import summarize_split_distributions


class HomogeneousSchemaTest(unittest.TestCase):
    def test_normalize_homogeneous_schema_aliases(self) -> None:
        self.assertEqual(normalize_homogeneous_schema(None), "current")
        self.assertEqual(normalize_homogeneous_schema("current"), "current")
        self.assertEqual(normalize_homogeneous_schema("legacy-flat50000"), "legacy_flat50000")
        self.assertEqual(normalize_homogeneous_schema("flat50000"), "legacy_flat50000")

    def test_legacy_flat50000_dataset_kwargs(self) -> None:
        kwargs = homogeneous_dataset_kwargs_for_schema("legacy_flat50000")

        self.assertEqual(kwargs["expected_node_feature_columns"], LEGACY_FLAT50000_NODE_FEATURE_COLUMNS)
        self.assertEqual(
            tuple(kwargs["dropped_node_feature_columns"]),
            LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS,
        )
        self.assertEqual(kwargs["expected_pulse_feature_columns"], LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS)
        self.assertEqual(tuple(kwargs["dropped_pulse_feature_columns"]), ())

    def test_legacy_checkpoint_match_requires_old_dimensions(self) -> None:
        self.assertTrue(
            legacy_flat50000_checkpoint_matches(
                {
                    "node_dim": len(LEGACY_FLAT50000_NODE_FEATURE_COLUMNS),
                    "pulse_dim": len(LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS) - 1,
                    "target_dim": 7,
                    "waveform_schema": "rise_aligned_raw_plus_accepted_gapped_v1",
                    "waveform_channels": 4,
                }
            )
        )
        self.assertFalse(
            legacy_flat50000_checkpoint_matches(
                {
                    "node_dim": len(NODE_FEATURE_COLUMNS),
                    "pulse_dim": len(PULSE_FEATURE_COLUMNS) - 1,
                    "target_dim": 6,
                    "waveform_schema": "rise_aligned_raw_plus_accepted_mask_v1",
                    "waveform_channels": 4,
                }
            )
        )

    def test_current_h5_is_rejected_when_legacy_schema_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "current.h5"
            graph = GraphEvent(
                event_id="event0",
                node_features=np.zeros((4, len(NODE_FEATURE_COLUMNS)), dtype=np.float32),
                node_positions_km=np.zeros((4, 3), dtype=np.float32),
                node_lids=np.arange(4, dtype=np.int64),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_features=np.zeros((0, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32),
                pulse_features=np.zeros((4, len(PULSE_FEATURE_COLUMNS)), dtype=np.float32),
                waveform_features=np.zeros((4, 4, 128), dtype=np.float32),
                target=np.zeros(6, dtype=np.float32),
                particle_label=0.0,
                metadata={},
            )
            with create_graph_file(path) as handle:
                write_graph(handle, 0, graph)

            with self.assertRaisesRegex(ValueError, "stored node feature columns are incompatible"):
                graph_dataset.H5GraphDataset(
                    path,
                    require_target=True,
                    **homogeneous_dataset_kwargs_for_schema("legacy_flat50000"),
                )

    def test_legacy_builder_writes_flat50000_dimensions_and_attrs(self) -> None:
        subs = []
        for lid in range(4):
            subs.append(
                {
                    "lid": lid + 1,
                    "posX": float(lid * 1200.0),
                    "posY": float((lid % 2) * 600.0),
                    "posZ": float(lid * 10.0),
                    "clock": 1000 + lid * 10,
                    "maxClock": 1_000_000,
                    "uwf": np.full(128, 20.0 + lid, dtype=np.float32),
                    "lwf": np.full(128, 19.0 + lid, dtype=np.float32),
                    "upedAvr": 10.0,
                    "lpedAvr": 10.0,
                    "upedStdev": 1.0,
                    "lpedStdev": 1.0,
                    "umipMev2cnt": 1.0,
                    "lmipMev2cnt": 1.0,
                    "dontUse": 0,
                }
            )
        record = BankRecord(
            bank={
                "date": 220804,
                "time": 85639,
                "usec": 123456,
                "sub": subs,
                "sim": {
                    "primaryEnergy": 1.0e18,
                    "primaryCosZenith": 0.8,
                    "primaryAzimuth": 45.0,
                    "primaryCorePosX": 1000.0,
                    "primaryCorePosY": 2000.0,
                    "primaryCorePosZ": 30.0,
                    "primaryParticleId": 14,
                    "eventNum": 7,
                },
            },
            source_path="DAT000016_gea_trg_001.dst.gz",
            source_index=3,
            source_kind="mc",
        )
        pulse = SimpleNamespace(
            energy_mev=3.0,
            time_usec=0.4,
            upper_rise_bin=40,
            upper_fall_bin=45,
            lower_rise_bin=41,
            lower_fall_bin=46,
        )
        with mock.patch(
            "talesd_gnn_reconstruction.event_graph.sd_signal_search_10a",
            return_value=[],
        ), mock.patch(
            "talesd_gnn_reconstruction.event_graph.find_coincident_pulses",
            return_value=[pulse],
        ):
            graph = build_graph_event(record, homogeneous_schema="legacy_flat50000")

        self.assertIsNotNone(graph)
        assert graph is not None
        self.assertEqual(graph.node_features.shape, (4, len(LEGACY_FLAT50000_NODE_FEATURE_COLUMNS)))
        self.assertEqual(graph.pulse_features.shape, (4, len(LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS)))
        self.assertEqual(graph.edge_features.shape[1], len(LEGACY_FLAT50000_EDGE_FEATURE_COLUMNS))
        self.assertEqual(graph.waveform_features.shape, (4, len(LEGACY_FLAT50000_WAVEFORM_FEATURE_CHANNELS), 128))
        self.assertEqual(graph.target.shape, (len(LEGACY_FLAT50000_TARGET_COLUMNS),))
        self.assertAlmostEqual(float(graph.target[3]), 0.03)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.h5"
            with create_graph_file(path, config={"homogeneous_schema": "legacy_flat50000"}) as handle:
                write_graph(handle, 0, graph)
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["homogeneous_schema"], "legacy_flat50000")
                self.assertEqual(handle.attrs["waveform_schema"], "rise_aligned_raw_plus_accepted_gapped_v1")

            dataset = graph_dataset.H5GraphDataset(
                path,
                require_target=True,
                **homogeneous_dataset_kwargs_for_schema("legacy_flat50000"),
            )
            sample = dataset[0]
            self.assertEqual(sample["node_features"].shape[1], len(LEGACY_FLAT50000_NODE_FEATURE_COLUMNS))
            self.assertEqual(sample["pulse_features"].shape[1], len(LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS))
            self.assertEqual(sample["target"].shape[0], len(LEGACY_FLAT50000_TARGET_COLUMNS))

    def test_split_summary_auto_detects_legacy_flat50000_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            h5_path = tmp_path / "legacy.h5"
            output_path = tmp_path / "summary.json"
            with create_graph_file(h5_path, config={"homogeneous_schema": "legacy_flat50000"}) as handle:
                for index in range(30):
                    graph = GraphEvent(
                        event_id=f"event{index}",
                        node_features=np.zeros((4, len(LEGACY_FLAT50000_NODE_FEATURE_COLUMNS)), dtype=np.float32),
                        node_positions_km=np.zeros((4, 3), dtype=np.float32),
                        node_lids=np.arange(4, dtype=np.int64),
                        edge_index=np.zeros((2, 0), dtype=np.int64),
                        edge_features=np.zeros((0, len(LEGACY_FLAT50000_EDGE_FEATURE_COLUMNS)), dtype=np.float32),
                        pulse_features=np.zeros((4, len(LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS)), dtype=np.float32),
                        waveform_features=np.zeros(
                            (4, len(LEGACY_FLAT50000_WAVEFORM_FEATURE_CHANNELS), 128),
                            dtype=np.float32,
                        ),
                        target=np.array([18.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        particle_label=float(index % 2),
                        metadata={
                            "source_path": f"/mc/proton/DAT{index:04d}16_gea_trg_001.dst.gz",
                            "source_index": index,
                        },
                    )
                    write_graph(handle, index, graph)

            argv = [
                "summarize_split_distributions.py",
                str(h5_path),
                "-o",
                str(output_path),
                "--val-fraction",
                "0.2",
                "--test-fraction",
                "0.2",
                "--source-val-fraction",
                "0.2",
                "--source-test-fraction",
                "0.2",
                "--split-workers",
                "0",
                "--no-progress",
            ]
            with mock.patch.object(sys, "argv", argv):
                summarize_split_distributions.main()

            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
