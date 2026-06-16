from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import talesd_gnn_reconstruction.diagnostics as diagnostics
from talesd_gnn_reconstruction.feature_analysis import _feature_group_importance_plot_data
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

    def test_learning_progress_writes_total_and_component_loss_curves(self) -> None:
        history = [
            {
                "epoch": 1,
                "train_loss": 3.0,
                "val_loss": 3.4,
                "train_reconstruction_loss": 2.5,
                "val_reconstruction_loss": 2.8,
                "train_mass_loss": 0.8,
                "val_mass_loss": 0.9,
                "train_quality_loss": 0.3,
                "val_quality_loss": 0.4,
            },
            {
                "epoch": 2,
                "train_loss": 2.0,
                "val_loss": 2.3,
                "train_reconstruction_loss": 1.7,
                "val_reconstruction_loss": 1.9,
                "train_mass_loss": 0.7,
                "val_mass_loss": 0.8,
                "train_quality_loss": 0.25,
                "val_quality_loss": 0.35,
            },
        ]
        old_require = diagnostics.require_matplotlib_latex
        diagnostics.require_matplotlib_latex = lambda: None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = diagnostics.save_learning_progress(Path(tmpdir) / "checkpoint.pt", history)
                self.assertIn("learning_curve_pdf", paths)
                self.assertIn("loss_component_curves_pdf", paths)
                self.assertTrue(Path(paths["learning_curve_pdf"]).exists())
                self.assertTrue(Path(paths["loss_component_curves_pdf"]).exists())
                self.assertNotEqual(paths["learning_curve_pdf"], paths["loss_component_curves_pdf"])
        finally:
            diagnostics.require_matplotlib_latex = old_require

    def test_feature_importance_plot_values_use_common_performance_loss_direction(self) -> None:
        result = {
            "split": "validation",
            "n_graphs": 10,
            "baseline": {
                "reconstruction": {"core_68_km": 0.24, "energy_particle_bias_abs_mean_log10": 0.04},
                "mass": {"balanced_accuracy": 0.80},
            },
            "groups": [
                {
                    "group": "detector",
                    "reconstruction": {"core_68_km": 0.36, "energy_particle_bias_abs_mean_log10": 0.08},
                    "reconstruction_delta": {"core_68_km": 0.12, "energy_particle_bias_abs_mean_log10": 0.04},
                    "mass": {"balanced_accuracy": 0.82},
                    "mass_delta": {"accuracy": -0.03, "balanced_accuracy": 0.02},
                }
            ],
        }

        plot_data = _feature_group_importance_plot_data(result)

        values_by_name = {spec["display_name"]: spec["values"][0] for spec in plot_data["plot_specs"]}
        labels_by_name = {spec["display_name"]: spec["label"] for spec in plot_data["plot_specs"]}
        relative_by_name = {spec["display_name"]: spec["values"][0] for spec in plot_data["relative_plot_specs"]}
        self.assertAlmostEqual(values_by_name["delta_core_68_km"], 0.12)
        self.assertAlmostEqual(values_by_name["delta_energy_particle_bias_abs_mean_log10"], 0.04)
        self.assertAlmostEqual(values_by_name["balanced_accuracy_drop"], -0.02)
        self.assertAlmostEqual(relative_by_name["relative_delta_core_68_km"], 0.5)
        self.assertAlmostEqual(relative_by_name["relative_delta_energy_particle_bias_abs_mean_log10"], 1.0)
        self.assertAlmostEqual(relative_by_name["relative_balanced_accuracy_drop"], 0.80 / 0.82 - 1.0)
        self.assertEqual(labels_by_name["balanced_accuracy_drop"], "mass accuracy loss")
        self.assertNotIn("accuracy_drop", values_by_name)
        self.assertIn("Positive values", plot_data["display_convention"])
        self.assertIn("Baseline performance is 0.0", plot_data["relative_display_convention"])


if __name__ == "__main__":
    unittest.main()
