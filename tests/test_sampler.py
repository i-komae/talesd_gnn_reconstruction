from __future__ import annotations

import unittest

from talesd_gnn_reconstruction.train import LocalityBatchSampler


class LocalityBatchSamplerTest(unittest.TestCase):
    def test_training_shuffle_breaks_contiguous_batches(self) -> None:
        sampler = LocalityBatchSampler(
            indices=list(range(40)),
            batch_size=5,
            shuffle_batches=True,
            seed=123,
        )

        batches = list(iter(sampler))
        flattened = [index for batch in batches for index in batch]

        self.assertEqual(len(batches), 8)
        self.assertEqual(sorted(flattened), list(range(40)))
        self.assertTrue(
            any(batch != list(range(min(batch), min(batch) + len(batch))) for batch in batches)
        )

    def test_validation_order_keeps_sorted_batches(self) -> None:
        sampler = LocalityBatchSampler(
            indices=[5, 2, 4, 1, 3, 0],
            batch_size=3,
            shuffle_batches=False,
            seed=123,
        )

        self.assertEqual(list(iter(sampler)), [[0, 1, 2], [3, 4, 5]])


if __name__ == "__main__":
    unittest.main()
