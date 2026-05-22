"""TALE-SD GNN reconstruction package."""

from __future__ import annotations

import random
from collections.abc import Iterator

__version__ = "0.1.0"


def enable_epoch_global_batch_shuffle() -> None:
    """Make training batches use a full-index shuffle every epoch."""

    from . import train

    sampler = train.LocalityBatchSampler
    if getattr(sampler, "_uses_epoch_global_shuffle", False):
        return

    def iter_epoch_batches(self) -> Iterator[list[int]]:
        indices = list(self.indices)
        if self.shuffle_batches:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        self.epoch += 1
        yield from batches

    sampler.__iter__ = iter_epoch_batches
    sampler._uses_epoch_global_shuffle = True


enable_epoch_global_batch_shuffle()
