import random
from typing import List, Iterable, Iterator, Sequence

import torch
from torch.utils.data.sampler import BatchSampler


class StratifiedBatchSampler(BatchSampler):
    """BatchSampler that composes each batch from multiple buckets.

    buckets: sequence of lists of indices (neg, small, mid, full)
    counts: list of integers giving how many samples to draw from each bucket per batch
    epoch_size: number of batches per epoch (optional). If None, iterate until
                the largest bucket is exhausted (with replacement allowed when bucket too small).
    replace: if True, sample with replacement when bucket lacks enough items.
    shuffle: whether to shuffle within-bucket sampling each epoch.
    """

    def __init__(
        self,
        buckets: Sequence[Sequence[int]],
        counts: Sequence[int],
        epoch_size: int | None = None,
        replace: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        assert len(buckets) == len(counts)
        self.buckets = [list(b) for b in buckets]
        self.counts = list(counts)
        self.epoch_size = epoch_size
        self.replace = replace
        self.shuffle = shuffle
        self.random = random.Random(seed)

        # total batch size
        self.batch_size = sum(self.counts)

    def __len__(self) -> int:
        if self.epoch_size is not None:
            return int(self.epoch_size)
        # default: use a large epoch size (user should set epoch_size)
        return 1000

    def __iter__(self) -> Iterator[List[int]]:
        # local copies to avoid mutating original lists
        buckets = [list(b) for b in self.buckets]

        # If shuffle, pre-shuffle buckets
        if self.shuffle:
            for b in buckets:
                self.random.shuffle(b)

        batch_count = 0
        while True:
            if self.epoch_size is not None and batch_count >= self.epoch_size:
                break

            batch: List[int] = []
            for i, b in enumerate(buckets):
                k = self.counts[i]
                if k == 0:
                    continue
                if len(b) >= k and not self.replace:
                    # draw without replacement
                    chosen = [b.pop() for _ in range(k)]
                else:
                    # sample with replacement or from remaining
                    chosen = [self.random.choice(b) if b else None for _ in range(k)]
                    # if any None (empty bucket), skip them
                    chosen = [c for c in chosen if c is not None]
                batch.extend(chosen)

            # If batch smaller than expected (due to empty buckets), we may pad by sampling across buckets
            if len(batch) < self.batch_size:
                # sample randomly from the union of non-empty buckets
                pool = [x for b in buckets for x in b] if any(buckets) else []
                while len(batch) < self.batch_size and pool:
                    batch.append(self.random.choice(pool))

            # shuffle within-batch before yielding
            self.random.shuffle(batch)

            yield batch
            batch_count += 1
