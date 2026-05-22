from __future__ import annotations

import math
import unittest

import torch

from talesd_gnn_reconstruction.model import PhysicsTaleSdGNN, build_model_from_config
from talesd_gnn_reconstruction.train import _angular_loss_from_vectors, _mass_classification_loss


class QualityModelTest(unittest.TestCase):
    def test_physics_model_outputs_reconstruction_mass_and_quality(self) -> None:
        model = PhysicsTaleSdGNN(
            node_dim=5,
            edge_dim=7,
            pulse_dim=0,
            target_dim=7,
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

        self.assertEqual(tuple(out.shape), (1, 9))

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
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        pred = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, math.sin(angle_rad), 0.0, math.cos(angle_rad)]],
            dtype=torch.float32,
        )

        angular_loss = float(_angular_loss_from_vectors(pred, target, angular_loss_scale_deg=1.0))
        old_cosine_loss = 1.0 - math.cos(angle_rad)

        self.assertGreater(angular_loss, 100.0 * old_cosine_loss)

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


if __name__ == "__main__":
    unittest.main()
