from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
