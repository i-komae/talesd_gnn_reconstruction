from __future__ import annotations

import os
import unittest
from unittest import mock

import torch.multiprocessing as torch_mp

from talesd_gnn_reconstruction.hetero_training import _configure_torch_sharing_strategy


class HeteroTrainingRuntimeTest(unittest.TestCase):
    def test_configures_file_system_sharing_for_worker_loaders(self) -> None:
        original = torch_mp.get_sharing_strategy()
        try:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TALESD_GNN_TORCH_SHARING_STRATEGY", None)
                os.environ.pop("TORCH_SHARING_STRATEGY", None)
                strategy = _configure_torch_sharing_strategy(2)
            self.assertEqual(strategy, "file_system")
            self.assertEqual(torch_mp.get_sharing_strategy(), "file_system")
        finally:
            torch_mp.set_sharing_strategy(original)

    def test_skips_sharing_strategy_for_single_process_loader(self) -> None:
        self.assertEqual(_configure_torch_sharing_strategy(0), "single-process")


if __name__ == "__main__":
    unittest.main()
