from __future__ import annotations

import unittest

import numpy as np

from talesd_gnn_reconstruction.metrics import (
    balanced_accuracy_threshold,
    binary_classification_metrics,
    energy_particle_bias_metrics,
)


class BinaryClassificationMetricsTest(unittest.TestCase):
    def test_validation_threshold_handles_offset_logits(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
        probs = np.asarray([0.55, 0.57, 0.58, 0.61, 0.64, 0.67], dtype=float)
        logits = np.log(probs / (1.0 - probs))

        fixed = binary_classification_metrics(logits, labels)
        threshold = balanced_accuracy_threshold(logits, labels)
        tuned = binary_classification_metrics(logits, labels, threshold=threshold)

        self.assertEqual(fixed["pred_proton"], 0)
        self.assertGreater(tuned["pred_proton"], 0)
        self.assertGreater(tuned["balanced_accuracy"], fixed["balanced_accuracy"])

    def test_auc_uses_average_ranks_for_tied_scores(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=float)
        logits = np.zeros_like(labels)

        metrics = binary_classification_metrics(logits, labels)

        self.assertEqual(metrics["auc"], 0.5)

    def test_energy_particle_bias_uses_centered_energy_bins(self) -> None:
        target = np.asarray(
            [
                [18.86, 0, 0, 0, 0, 1],
                [18.88, 0, 0, 0, 0, 1],
                [18.89, 0, 0, 0, 0, 1],
                [18.91, 0, 0, 0, 0, 1],
                [18.92, 0, 0, 0, 0, 1],
                [18.94, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
        pred = target.copy()
        pred[labels < 0.5, 0] += 0.01
        pred[labels >= 0.5, 0] += 0.06

        metrics = energy_particle_bias_metrics(pred, target, labels, bin_width=0.1, min_bin_count=3)

        self.assertEqual(metrics["energy_particle_bias_n_bins"], 1)
        self.assertAlmostEqual(metrics["energy_particle_bias_abs_mean_log10"], 0.05)
        row = metrics["energy_particle_bias_bins"][0]
        self.assertAlmostEqual(row["log10_energy_low"], 18.85)
        self.assertAlmostEqual(row["log10_energy_high"], 18.95)


if __name__ == "__main__":
    unittest.main()
