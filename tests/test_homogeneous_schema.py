from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from talesd_gnn_reconstruction import dataset as graph_dataset
from talesd_gnn_reconstruction.constants import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS, PULSE_FEATURE_COLUMNS
from talesd_gnn_reconstruction.event_graph import GraphEvent
from talesd_gnn_reconstruction.graph_io import create_graph_file, write_graph
from talesd_gnn_reconstruction.homogeneous_schema import (
    LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS,
    LEGACY_FLAT50000_NODE_FEATURE_COLUMNS,
    LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS,
    homogeneous_dataset_kwargs_for_schema,
    legacy_flat50000_checkpoint_matches,
    normalize_homogeneous_schema,
)


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


if __name__ == "__main__":
    unittest.main()
