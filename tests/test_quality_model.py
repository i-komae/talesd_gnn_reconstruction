from __future__ import annotations

import math
import unittest

import torch

from talesd_gnn_reconstruction.model import PhysicsTaleSdGNN, WaveformEncoder, build_model_from_config
from talesd_gnn_reconstruction.train import (
    _angular_loss_from_vectors,
    _energy_bin_bias_loss,
    _energy_particle_bias_loss,
    _gaussian_reconstruction_nll,
    _inverse_scaled_target_with_unit_direction,
    _mass_classification_loss,
)


class QualityModelTest(unittest.TestCase):
    def test_waveform_transformer_masks_invalid_rows_and_caps_tokens(self) -> None:
        torch.manual_seed(123)
        encoder = WaveformEncoder(
            waveform_channels=2,
            waveform_length=2048,
            embedding_dim=8,
            mode="transformer",
            dropout=0.0,
            transformer_heads=2,
            transformer_layers=1,
            transformer_max_tokens=128,
        )
        encoder.eval()
        token_counts: list[int] = []

        def _capture_tokens(_module, args):
            token_counts.append(int(args[0].shape[1]))

        hook = encoder.transformer.register_forward_pre_hook(_capture_tokens)
        try:
            waveform = torch.randn(3, 2, 2048)
            valid_mask = torch.tensor([1.0, 0.0, 1.0])
            with torch.no_grad():
                baseline = encoder(
                    waveform,
                    num_nodes=3,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    valid_mask=valid_mask,
                )
                changed_invalid = waveform.clone()
                changed_invalid[1] += 1000.0
                repeated = encoder(
                    changed_invalid,
                    num_nodes=3,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    valid_mask=valid_mask,
                )
                sliced = encoder(
                    waveform[[0, 2]],
                    num_nodes=2,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    valid_mask=torch.ones(2),
                )
        finally:
            hook.remove()

        self.assertEqual(token_counts, [128, 128, 128])
        self.assertTrue(torch.allclose(baseline[1], torch.zeros_like(baseline[1])))
        self.assertTrue(torch.allclose(baseline[[0, 2]], sliced, atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(baseline, repeated, atol=1.0e-6, rtol=1.0e-6))

    def test_waveform_valid_mask_accepts_autocast_dtype_output(self) -> None:
        encoder = WaveformEncoder(
            waveform_channels=2,
            waveform_length=16,
            embedding_dim=8,
            mode="cnn",
            dropout=0.0,
        )

        def _half_output(waveform: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            del dtype
            return torch.ones(waveform.shape[0], encoder.output_dim, dtype=torch.float16, device=device)

        encoder._encode_valid_waveforms = _half_output  # type: ignore[method-assign]
        waveform = torch.randn(3, 2, 16)
        valid_mask = torch.tensor([1.0, 0.0, 1.0])

        with torch.no_grad():
            output = encoder(
                waveform,
                num_nodes=3,
                device=torch.device("cpu"),
                dtype=torch.float32,
                valid_mask=valid_mask,
            )

        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.allclose(output[0], torch.ones_like(output[0])))
        self.assertTrue(torch.allclose(output[1], torch.zeros_like(output[1])))
        self.assertTrue(torch.allclose(output[2], torch.ones_like(output[2])))

    def test_physics_model_outputs_reconstruction_mass_and_quality(self) -> None:
        model = PhysicsTaleSdGNN(
            node_dim=5,
            edge_dim=7,
            pulse_dim=0,
            target_dim=6,
            classification_dim=1,
            quality_dim=1,
            hidden_dim=16,
            num_layers=1,
            readout_heads=2,
        )
        batch = {
            "x": torch.randn(3, 5),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "edge_attr": torch.randn(2, 7),
            "batch": torch.zeros(3, dtype=torch.long),
            "num_graphs": 1,
        }

        out = model(batch)

        self.assertEqual(tuple(out.shape), (1, 8))

    def test_physics_model_outputs_eventwise_error_head_when_enabled(self) -> None:
        model = PhysicsTaleSdGNN(
            node_dim=5,
            edge_dim=7,
            pulse_dim=0,
            target_dim=6,
            classification_dim=1,
            quality_dim=1,
            error_dim=3,
            hidden_dim=16,
            num_layers=1,
            readout_heads=2,
        )
        batch = {
            "x": torch.randn(3, 5),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "edge_attr": torch.randn(2, 7),
            "batch": torch.zeros(3, dtype=torch.long),
            "num_graphs": 1,
        }

        out = model(batch)

        self.assertEqual(tuple(out.shape), (1, 11))

    def test_physics_mass_only_model_omits_reconstruction_heads(self) -> None:
        model = PhysicsTaleSdGNN(
            node_dim=5,
            edge_dim=7,
            pulse_dim=0,
            target_dim=0,
            classification_dim=1,
            hidden_dim=16,
            num_layers=1,
            readout_heads=2,
            classification_arch="enhanced",
        )
        batch = {
            "x": torch.randn(3, 5),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "edge_attr": torch.randn(2, 7),
            "batch": torch.zeros(3, dtype=torch.long),
            "num_graphs": 1,
        }

        out = model(batch)

        self.assertEqual(tuple(out.shape), (1, 1))
        self.assertIsNone(model.energy_head)
        self.assertIsNone(model.core_head)
        self.assertIsNone(model.direction_head)

    def test_old_checkpoint_config_disables_new_time_encoder_by_default(self) -> None:
        model = build_model_from_config(
            {
                "architecture": "physics",
                "node_dim": 5,
                "edge_dim": 7,
                "pulse_dim": 0,
                "target_dim": 7,
                "classification_dim": 0,
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.05,
                "readout_heads": 2,
                "detector_lids": [],
                "detector_embedding_dim": 0,
                "waveform_channels": 0,
                "waveform_length": 0,
                "waveform_encoder": "none",
                "waveform_embedding_dim": 8,
                "waveform_transformer_heads": 1,
                "waveform_transformer_layers": 1,
            }
        )

        self.assertFalse(model.time_edge_encoder.enabled)

    def test_angular_loss_is_not_cosine_small_angle_suppressed(self) -> None:
        angle_rad = math.radians(1.0)
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        pred = torch.tensor(
            [[0.0, 0.0, 0.0, math.sin(angle_rad), 0.0, math.cos(angle_rad)]],
            dtype=torch.float32,
        )

        angular_loss = float(_angular_loss_from_vectors(pred, target, angular_loss_scale_deg=1.0))
        old_cosine_loss = 1.0 - math.cos(angle_rad)

        self.assertGreater(angular_loss, 100.0 * old_cosine_loss)

    def test_inverse_scaled_target_normalizes_direction_components(self) -> None:
        pred_scaled = torch.tensor([[18.0, 0.1, -0.2, 10.0, 0.0, 0.0]], dtype=torch.float32)
        mean = torch.zeros(6, dtype=torch.float32)
        std = torch.ones(6, dtype=torch.float32)

        pred = _inverse_scaled_target_with_unit_direction(pred_scaled, mean, std)

        self.assertAlmostEqual(float(torch.linalg.vector_norm(pred[:, 3:6], dim=1)[0]), 1.0, places=6)
        self.assertAlmostEqual(float(pred[0, 3]), 1.0, places=6)

    def test_energy_bin_bias_loss_penalizes_mean_loge_bias(self) -> None:
        target = torch.tensor(
            [
                [18.05, 0.0, 0.0, 0.0, 0.0, 1.0],
                [18.06, 0.0, 0.0, 0.0, 0.0, 1.0],
                [18.15, 0.0, 0.0, 0.0, 0.0, 1.0],
                [18.16, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        unbiased = target.clone()
        biased = target.clone()
        biased[:, 0] += torch.tensor([0.10, 0.10, -0.05, -0.05], dtype=torch.float32)

        unbiased_loss = _energy_bin_bias_loss(unbiased, target, bin_width=0.1, min_bin_count=2)
        biased_loss = _energy_bin_bias_loss(biased, target, bin_width=0.1, min_bin_count=2)

        self.assertEqual(float(unbiased_loss), 0.0)
        self.assertGreater(float(biased_loss), 0.0)

    def test_energy_particle_bias_loss_penalizes_proton_iron_difference(self) -> None:
        target = torch.tensor(
            [
                [18.05, 0.0, 0.0, 0.0, 0.0, 1.0],
                [18.06, 0.0, 0.0, 0.0, 0.0, 1.0],
                [18.07, 0.0, 0.0, 0.0, 0.0, 1.0],
                [18.08, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        same_bias = target.clone()
        same_bias[:, 0] += 0.05
        particle_biased = target.clone()
        particle_biased[:, 0] += torch.tensor([0.05, 0.05, -0.05, -0.05], dtype=torch.float32)

        same_loss = _energy_particle_bias_loss(
            same_bias,
            target,
            labels,
            bin_width=0.1,
            min_bin_count=2,
        )
        particle_loss = _energy_particle_bias_loss(
            particle_biased,
            target,
            labels,
            bin_width=0.1,
            min_bin_count=2,
        )

        self.assertEqual(float(same_loss), 0.0)
        self.assertGreater(float(particle_loss), 0.0)

    def test_mass_ranking_loss_pushes_classes_apart(self) -> None:
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        flat_logits = torch.zeros(4, dtype=torch.float32, requires_grad=True)

        flat_loss = _mass_classification_loss(
            flat_logits,
            labels,
            mode="bce",
            pos_weight=None,
            focal_gamma=2.0,
            ranking_weight=0.5,
            ranking_margin=1.0,
        )
        flat_loss.backward()

        self.assertGreater(float(flat_loss.detach()), 0.0)
        self.assertGreater(float(flat_logits.grad[0]), 0.0)
        self.assertLess(float(flat_logits.grad[2]), 0.0)

        separated_logits = torch.tensor([-2.0, -2.0, 2.0, 2.0], dtype=torch.float32)
        separated_loss = _mass_classification_loss(
            separated_logits,
            labels,
            mode="bce",
            pos_weight=None,
            focal_gamma=2.0,
            ranking_weight=0.5,
            ranking_margin=1.0,
        )

        self.assertLess(float(separated_loss), float(flat_loss.detach()))

    def test_gaussian_reconstruction_nll_uses_predicted_errors(self) -> None:
        target_mean = torch.zeros(6, dtype=torch.float32)
        target_std = torch.ones(6, dtype=torch.float32)
        target = torch.tensor([[16.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        perfect = target.clone().requires_grad_(True)
        shifted = torch.tensor([[16.1, 0.05, 0.0, 0.1, 0.0, 0.995]], dtype=torch.float32)
        error_raw = torch.zeros(1, 3, dtype=torch.float32, requires_grad=True)

        perfect_loss = _gaussian_reconstruction_nll(
            error_raw,
            perfect,
            target,
            target_mean=target_mean,
            target_std=target_std,
            energy_weight=1.0,
            core_weight=1.0,
            direction_weight=1.0,
            error_angular_scale_deg=1.0,
            error_core_scale_km=0.05,
            error_energy_scale=0.10,
            sigma_energy_floor=0.01,
            sigma_angle_floor_deg=0.05,
            sigma_core_floor_km=0.005,
        )
        shifted_loss = _gaussian_reconstruction_nll(
            error_raw,
            shifted,
            target,
            target_mean=target_mean,
            target_std=target_std,
            energy_weight=1.0,
            core_weight=1.0,
            direction_weight=1.0,
            error_angular_scale_deg=1.0,
            error_core_scale_km=0.05,
            error_energy_scale=0.10,
            sigma_energy_floor=0.01,
            sigma_angle_floor_deg=0.05,
            sigma_core_floor_km=0.005,
        )

        self.assertTrue(torch.isfinite(perfect_loss))
        self.assertTrue(torch.isfinite(shifted_loss))
        self.assertGreater(float(shifted_loss.detach()), float(perfect_loss.detach()))
        shifted_loss.backward()
        self.assertIsNotNone(error_raw.grad)


if __name__ == "__main__":
    unittest.main()
