from __future__ import annotations

import unittest

import torch

from talesd_gnn_reconstruction.hetero_model import HeteroAttentionMessageLayer, HeteroAttentiveReadout
from talesd_gnn_reconstruction.model import _scatter_softmax


def _active_group_sums(weights: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(num_graphs, weights.shape[1], dtype=torch.float32, device=weights.device)
    sums.index_add_(0, batch, weights.float())
    counts = torch.zeros(num_graphs, 1, dtype=torch.float32, device=weights.device)
    counts.index_add_(0, batch, torch.ones(batch.shape[0], 1, dtype=torch.float32, device=weights.device))
    return sums, counts.squeeze(-1) > 0


class AmpScatterTest(unittest.TestCase):
    def test_scatter_softmax_cpu_float32_groups_sum_to_one(self) -> None:
        scores = torch.tensor(
            [[0.0, 1.0], [0.5, -0.5], [1.0, 0.0], [-2.0, 3.0]],
            dtype=torch.float32,
        )
        batch = torch.tensor([0, 0, 1, 2], dtype=torch.long)

        weights = _scatter_softmax(scores, batch, num_graphs=4)
        sums, active = _active_group_sums(weights, batch, num_graphs=4)

        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(torch.allclose(sums[active], torch.ones_like(sums[active]), atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(sums[~active], torch.zeros_like(sums[~active]), atol=1.0e-6, rtol=1.0e-6))

    def test_scatter_softmax_empty_input_is_safe(self) -> None:
        scores = torch.zeros(0, 3, dtype=torch.float16)
        batch = torch.zeros(0, dtype=torch.long)

        weights = _scatter_softmax(scores, batch, num_graphs=0)

        self.assertEqual(weights.dtype, scores.dtype)
        self.assertEqual(tuple(weights.shape), (0, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scatter_softmax_fp16_cuda(self) -> None:
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(123)
        scores = torch.randn(100, 4, device=device, dtype=torch.float16, generator=generator)
        batch = torch.randint(0, 7, (100,), device=device, dtype=torch.long, generator=generator)

        weights = _scatter_softmax(scores, batch, num_graphs=7)
        sums, active = _active_group_sums(weights, batch, num_graphs=7)

        self.assertEqual(weights.dtype, torch.float16)
        self.assertTrue(torch.allclose(sums[active], torch.ones_like(sums[active]), atol=2.0e-3, rtol=2.0e-3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scatter_softmax_cuda_autocast_linear_output(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(123)
        layer = torch.nn.Linear(8, 4).to(device)
        inputs = torch.randn(96, 8, device=device)
        batch = torch.randint(0, 6, (96,), device=device, dtype=torch.long)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            scores = layer(inputs)
            weights = _scatter_softmax(scores, batch, num_graphs=6)

        sums, active = _active_group_sums(weights, batch, num_graphs=6)
        self.assertEqual(weights.dtype, scores.dtype)
        self.assertTrue(torch.allclose(sums[active], torch.ones_like(sums[active]), atol=2.0e-3, rtol=2.0e-3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_hetero_attentive_readout_cuda_autocast(self) -> None:
        device = torch.device("cuda")
        readout = HeteroAttentiveReadout(hidden_dim=192, heads=4).to(device).eval()
        state = torch.randn(64, 192, device=device)
        batch = torch.randint(0, 5, (64,), device=device, dtype=torch.long)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = readout(state, batch, num_graphs=5)

        self.assertEqual(tuple(output.shape), (5, 192 * 6))
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_hetero_attention_message_layer_cuda_autocast(self) -> None:
        device = torch.device("cuda")
        layer = HeteroAttentionMessageLayer(
            hidden_dim=192,
            edge_dims={"detector__observes__pulse": 0, "pulse__observed_by__detector": 0},
            dropout=0.0,
            attention_heads=4,
        ).to(device).eval()
        node_states = {
            "detector": torch.randn(12, 192, device=device),
            "pulse": torch.randn(18, 192, device=device),
        }
        edge_index_by_type = {
            "detector__observes__pulse": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
                dtype=torch.long,
                device=device,
            ),
            "pulse__observed_by__detector": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
                dtype=torch.long,
                device=device,
            ),
        }
        edge_features_by_type = {
            "detector__observes__pulse": torch.zeros(8, 0, device=device),
            "pulse__observed_by__detector": torch.zeros(8, 0, device=device),
        }

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = layer(node_states, edge_index_by_type, edge_features_by_type)

        self.assertEqual(tuple(output["detector"].shape), (12, 192))
        self.assertEqual(tuple(output["pulse"].shape), (18, 192))
        self.assertTrue(torch.isfinite(output["detector"]).all())
        self.assertTrue(torch.isfinite(output["pulse"]).all())


if __name__ == "__main__":
    unittest.main()
