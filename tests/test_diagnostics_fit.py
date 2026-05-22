from __future__ import annotations

import unittest

import numpy as np

from talesd_gnn_reconstruction.diagnostics import (
    QUALITY_CUT_KEEP_FRACTIONS,
    QUALITY_ENERGY_KEEP_FRACTIONS,
    QUALITY_THRESHOLD_KEEP_FRACTIONS,
    _fit_gaussian_hist,
    _valid_energy_fit_rows,
)


class DiagnosticsFitTest(unittest.TestCase):
    def test_gaussian_fit_uses_robust_window_for_large_outliers(self) -> None:
        rng = np.random.default_rng(12345)
        core = rng.normal(loc=-0.02, scale=0.18, size=5000)
        values = np.concatenate([core, np.asarray([1.0e6, -1.0e6, 5.0e5])])

        stats = _fit_gaussian_hist(values)

        self.assertTrue(stats["fit_ok"], stats.get("fit_error"))
        self.assertLess(stats["fit_n"], stats["n"])
        self.assertAlmostEqual(float(stats["mu"]), -0.02, delta=0.03)
        self.assertAlmostEqual(float(stats["sigma"]), 0.18, delta=0.03)
        self.assertLess(abs(float(stats["mu"])), 0.1)
        self.assertLess(float(stats["sigma"]), 0.3)

    def test_rejected_fit_rows_are_not_plotted_as_gaussian_resolution(self) -> None:
        rows = [
            {"n": 1000, "fit_ok": False, "mu": 100.0, "sigma": 1000.0},
            {"n": 1000, "fit_ok": True, "mu": 0.01, "sigma": 0.2},
        ]

        valid_rows = _valid_energy_fit_rows(rows, min_bin_count=1000)

        self.assertEqual(valid_rows, [rows[1]])

    def test_quality_cut_defaults_keep_requested_high_survival_cuts(self) -> None:
        for fractions in (QUALITY_THRESHOLD_KEEP_FRACTIONS, QUALITY_CUT_KEEP_FRACTIONS, QUALITY_ENERGY_KEEP_FRACTIONS):
            self.assertIn(0.95, fractions)
            self.assertIn(0.90, fractions)
            self.assertIn(0.80, fractions)
            self.assertNotIn(0.20, fractions)


if __name__ == "__main__":
    unittest.main()
