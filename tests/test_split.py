from __future__ import annotations

import random
import unittest

import numpy as np

from talesd_gnn_reconstruction.train import (
    _assign_source_group,
    _source_stratification_keys,
    source_group_key,
    split_indices_by_source_path,
    split_indices_by_stratified_source_path,
)


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

    def test_assign_source_group_avoids_single_large_source_in_holdout(self) -> None:
        counts = {f"large_{index}": 1000 for index in range(3)}
        counts.update({f"small_{index}": 100 for index in range(27)})
        split_sources = {"train": [], "val": [], "test": []}

        _assign_source_group(
            split_sources,
            list(counts),
            val_fraction=0.1,
            test_fraction=0.2,
            rng=random.Random(5),
            source_event_counts=counts,
            source_val_fraction=0.10,
            source_test_fraction=0.20,
        )

        self.assertTrue(all(counts[source_path] <= 100 for source_path in split_sources["val"]))
        self.assertTrue(all(counts[source_path] <= 100 for source_path in split_sources["test"]))
        self.assertEqual(len(split_sources["test"]), 6)
        self.assertEqual(len(split_sources["val"]), 3)

    def test_assign_source_group_can_target_source_fractions_separately(self) -> None:
        counts = {f"source_{index}": 10 for index in range(100)}
        split_sources = {"train": [], "val": [], "test": []}

        _assign_source_group(
            split_sources,
            list(counts),
            val_fraction=0.05,
            test_fraction=0.10,
            rng=random.Random(7),
            source_event_counts=counts,
            source_val_fraction=0.10,
            source_test_fraction=0.20,
        )

        self.assertEqual(len(split_sources["train"]), 70)
        self.assertEqual(len(split_sources["val"]), 10)
        self.assertEqual(len(split_sources["test"]), 20)

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

    def test_corsika_split_dst_chunks_stay_in_same_split(self) -> None:
        source_counts = {}
        for dat_index in range(20):
            particle = "proton" if dat_index < 10 else "iron"
            loge = 16 if dat_index < 10 else 17
            for chunk_index in range(3):
                source_counts[
                    f"/mc/{particle}/bin_{loge}/DAT{dat_index:06d}_gea_trg_{chunk_index:03d}.dst.gz"
                ] = 4
        dataset = _FakeGraphDataset(source_counts)

        split = split_indices_by_stratified_source_path(
            dataset,  # type: ignore[arg-type]
            val_fraction=0.1,
            test_fraction=0.2,
            seed=11,
            show_progress=False,
            min_group_sources=4,
            workers=0,
        )

        group_to_split: dict[str, str] = {}
        for split_name, indices in split.items():
            for index in indices:
                group = source_group_key(dataset.source_path(index))
                previous = group_to_split.setdefault(group, split_name)
                self.assertEqual(previous, split_name)

    def test_source_stratification_energy_uses_dat_suffix(self) -> None:
        target = np.asarray([18.9, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        keys = _source_stratification_keys("/mc/proton/DAT123416", target, 0.0)

        self.assertEqual(keys["fine"][2], "16")
        self.assertEqual(keys["mid"][2], "16")

    def test_source_path_split_keeps_corsika_chunks_in_same_split(self) -> None:
        source_counts = {}
        for dat_index in range(18):
            for chunk_index in range(3):
                source_counts[f"/mc/proton/DAT{dat_index:06d}_gea_trg_{chunk_index:03d}.dst.gz"] = 3
        dataset = _FakeGraphDataset(source_counts)

        split = split_indices_by_source_path(
            dataset,  # type: ignore[arg-type]
            val_fraction=0.1,
            test_fraction=0.2,
            seed=13,
            show_progress=False,
        )

        group_to_split: dict[str, str] = {}
        for split_name, indices in split.items():
            for index in indices:
                group = source_group_key(dataset.source_path(index))
                previous = group_to_split.setdefault(group, split_name)
                self.assertEqual(previous, split_name)


if __name__ == "__main__":
    unittest.main()
