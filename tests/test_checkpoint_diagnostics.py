from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from talesd_gnn_reconstruction import dataset as graph_dataset


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_diagnostics_from_checkpoint.py"
    spec = importlib.util.spec_from_file_location("generate_diagnostics_from_checkpoint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSourceDataset:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def source_path(self, index: int) -> str:
        return self.paths[index]


class CheckpointDiagnosticsTest(unittest.TestCase):
    def test_dat_group_seen_unseen_test_split(self) -> None:
        module = _load_script_module()
        dataset = _FakeSourceDataset(
            [
                "/mc/proton/DAT000001_gea_trg_000.dst.gz",
                "/mc/proton/DAT000002_gea_trg_000.dst.gz",
                "/mc/proton/DAT000001_gea_trg_001.dst.gz",
                "/mc/proton/DAT000003_gea_trg_000.dst.gz",
            ]
        )

        seen, unseen, summary = module._test_seen_unseen_split(
            dataset,
            train_indices=[0, 1],
            test_indices=[2, 3],
            source_group_mode="dat",
        )

        self.assertEqual(seen, [2])
        self.assertEqual(unseen, [3])
        self.assertEqual(summary["test_seen_train_source_graphs"], 1)
        self.assertEqual(summary["test_unseen_train_source_graphs"], 1)
        self.assertEqual(summary["test_sources_seen_in_train"], 1)
        self.assertEqual(summary["test_sources_unseen_in_train"], 1)

    def test_empty_source_overlap_metrics_do_not_fail(self) -> None:
        module = _load_script_module()
        metrics = module._source_overlap_metrics_from_test_predictions(
            test_indices=[10, 11],
            seen_indices=[],
            unseen_indices=[10, 11],
            training_task="reconstruction",
            pred_test=np.zeros((2, 6), dtype=np.float32),
            target_test=np.zeros((2, 6), dtype=np.float32),
            mass_logit_test=None,
            mass_label_test=None,
            quality_test=None,
            predicted_error_test=None,
            mass_threshold=0.5,
            tuned_mass_threshold=0.5,
            energy_bin_width=0.1,
            min_bin_count=1,
        )

        self.assertEqual(metrics["test_seen_train_source"]["n_graphs"], 0)
        self.assertIsNone(metrics["test_seen_train_source"]["reconstruction"])
        self.assertEqual(metrics["test_unseen_train_source"]["n_graphs"], 2)
        self.assertIsNotNone(metrics["test_unseen_train_source"]["reconstruction"])

    def test_legacy_flat50000_dataset_kwargs_match_old_checkpoint_dims(self) -> None:
        module = _load_script_module()
        ckpt = {
            "model_config": {
                "node_dim": 27,
                "pulse_dim": 6,
                "target_dim": 7,
                "waveform_schema": "rise_aligned_raw_plus_accepted_gapped_v1",
                "waveform_channels": 4,
            }
        }

        kwargs = module._homogeneous_dataset_kwargs_from_checkpoint(ckpt)

        self.assertEqual(kwargs["expected_node_feature_columns"], module._LEGACY_FLAT50000_NODE_FEATURE_COLUMNS)
        self.assertEqual(
            tuple(kwargs["dropped_node_feature_columns"]),
            module._LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS,
        )
        self.assertEqual(kwargs["expected_pulse_feature_columns"], module._LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS)
        self.assertEqual(tuple(kwargs["allowed_waveform_schemas"]), ("rise_aligned_raw_plus_accepted_gapped_v1",))

    def test_current_checkpoint_dims_do_not_enable_legacy_flat50000_schema(self) -> None:
        module = _load_script_module()
        ckpt = {
            "model_config": {
                "node_dim": 28,
                "pulse_dim": 0,
                "target_dim": 6,
                "waveform_schema": "rise_aligned_raw_plus_accepted_mask_v1",
                "waveform_channels": 4,
            }
        }

        self.assertEqual(module._homogeneous_dataset_kwargs_from_checkpoint(ckpt), {})

    def test_legacy_flat50000_node_and_pulse_column_selection(self) -> None:
        module = _load_script_module()
        stored_node_columns = [
            "x_km",
            "y_km",
            "z_km",
            "nearest_detector_distance_km",
            "mean3_detector_distance_km",
            "neighbor_count_1p5km",
            "local_detector_density_1p5km2",
            "dx_from_bary_km",
            "dy_from_bary_km",
            "dz_from_bary_km",
            "r_from_bary_km",
            "first_arrival_usec_rel",
            "trig_usec_rel",
            "log10_first_rho",
            "sqrt_first_rho",
            "log10_total_rho",
            "sqrt_total_rho",
            "log10_max_rho",
            "n_pulses",
            "pulse_time_span_usec",
            "n_wf_segments",
            "wf_length_usec",
            "log10_fadc_peak",
            "upper_ped",
            "lower_ped",
            "upper_ped_sigma",
            "lower_ped_sigma",
            "detector_pulse_order",
            "is_first_detector_pulse",
        ]
        node_indices, effective_node_columns = graph_dataset._node_feature_selection_from_columns(
            {"node_features": stored_node_columns},
            expected_columns=module._LEGACY_FLAT50000_NODE_FEATURE_COLUMNS,
            dropped_columns=module._LEGACY_FLAT50000_DROPPED_NODE_FEATURE_COLUMNS,
        )
        pulse_indices, effective_pulse_columns = graph_dataset._pulse_feature_selection_from_columns(
            {"pulse_features": module._LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS},
            expected_columns=module._LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS,
            dropped_columns=(),
        )

        self.assertEqual(effective_node_columns, module._LEGACY_FLAT50000_NODE_FEATURE_COLUMNS)
        self.assertIsNotNone(node_indices)
        self.assertEqual(len(node_indices), 27)
        self.assertNotIn(15, node_indices.tolist())
        self.assertNotIn(16, node_indices.tolist())
        self.assertEqual(effective_pulse_columns, module._LEGACY_FLAT50000_PULSE_FEATURE_COLUMNS)
        self.assertIsNone(pulse_indices)


if __name__ == "__main__":
    unittest.main()
