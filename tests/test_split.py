from __future__ import annotations

import random
import unittest

import numpy as np

from talesd_gnn_reconstruction.train import _assign_source_group, split_indices_by_stratified_source_path


class _FakeGraphDataset:
    def __init__(self, source_counts: dict[str, int]):
        self._source_paths: list[str] = []
        self._targets: list[np.ndarray] = []
        self._labels: list[float] = []
        for source_path, count in source_counts.items():
            label = 1.0 if "/iron/" in source_path else 0.0
            loge = 17.0 if "_17" in source_path else 16.0
            target = np.asarray([loge, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            for _ in range(count):
                self._source_paths.append(source_path)
                self._targets.append(target)
                self._labels.append(label)

    def __len__(self) -> int:
        return len(self._source_paths)

    def source_path(self, index: int) -> str:
        return self._source_paths[index]

    def target(self, index: int) -> np.ndarray:
        return self._targets[index]

    def particle_label(self, index: int) -> float:
        return self._labels[index]


class SourceSplitTest(unittest.TestCase):
    def test_assign_source_group_targets_graph_counts_not_source_counts(self) -> None:
        counts = {"large": 50, **{f"small_{index}": 10 for index in range(9)}}
        split_sources = {"train": [], "val": [], "test": []}

        _assign_source_group(
            split_sources,
            list(counts),
            val_fraction=0.1,
            test_fraction=0.2,
            rng=random.Random(1),
            source_event_counts=counts,
        )

        split_counts = {
            name: sum(counts[source_path] for source_path in sources)
            for name, sources in split_sources.items()
        }
        self.assertEqual(split_counts["test"], 30)
        self.assertEqual(split_counts["val"], 10)
        self.assertEqual(split_counts["train"], 100)

    def test_stratified_source_split_keeps_source_paths_disjoint(self) -> None:
        source_counts = {
            **{f"/mc/proton/bin_16/source_{index}_16.dst.gz": 12 for index in range(12)},
            **{f"/mc/iron/bin_17/source_{index}_17.dst.gz": 12 for index in range(12)},
        }
        dataset = _FakeGraphDataset(source_counts)

        split = split_indices_by_stratified_source_path(
            dataset,  # type: ignore[arg-type]
            val_fraction=0.1,
            test_fraction=0.2,
            seed=3,
            show_progress=False,
            min_group_sources=4,
            workers=0,
        )

        split_sources = {
            name: {dataset.source_path(index) for index in indices}
            for name, indices in split.items()
        }
        self.assertTrue(split_sources["train"].isdisjoint(split_sources["val"]))
        self.assertTrue(split_sources["train"].isdisjoint(split_sources["test"]))
        self.assertTrue(split_sources["val"].isdisjoint(split_sources["test"]))
        self.assertEqual(sum(len(indices) for indices in split.values()), len(dataset))
        self.assertGreater(len(split["train"]), len(split["test"]))
        self.assertGreater(len(split["train"]), len(split["val"]))


if __name__ == "__main__":
    unittest.main()
