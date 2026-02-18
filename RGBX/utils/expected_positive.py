from typing import Iterable, List, Tuple
import math
import random


def _normalize(proportions: Iterable[float]) -> List[float]:
    p = list(proportions)
    s = sum(p)
    if s == 0:
        raise ValueError("Proportions sum to zero")
    return [x / s for x in p]


def expected_positive_fraction_continuous(
    proportions: Iterable[float],
    avg_positive_fractions: Iterable[float],
) -> float:
    """
    Continuous expectation of positive fraction in a sampled tile.

    Args:
        proportions: Sequence of bucket sampling probabilities (will be normalized).
        avg_positive_fractions: Sequence of average positive fractions for each bucket
            (values in [0, 1]; same length as `proportions`).

    Returns:
        Expected positive fraction (0..1) when sampling a single tile from the
        mixture given by `proportions`.
    """
    p = _normalize(proportions)
    f = list(avg_positive_fractions)
    if len(p) != len(f):
        raise ValueError("proportions and avg_positive_fractions must have same length")
    return sum(pi * fi for pi, fi in zip(p, f))


def expected_positive_fraction_discrete(
    proportions: Iterable[float],
    avg_positive_fractions: Iterable[float],
    batch_size: int,
) -> float:
    """
    Compute expected positive fraction for a single batch by allocating integer
    tile counts to buckets deterministically.

    The algorithm floors the ideal counts `p_i * batch_size` and distributes
    the remaining slots to buckets with largest fractional remainders (ties
    broken by index). This provides a deterministic integer allocation that
    sums to `batch_size`.

    Args:
        proportions: Bucket probabilities (will be normalized).
        avg_positive_fractions: Average positive fraction per bucket.
        batch_size: Number of tiles sampled into the batch.

    Returns:
        Expected positive fraction across the batch (0..1).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    p = _normalize(proportions)
    f = list(avg_positive_fractions)
    if len(p) != len(f):
        raise ValueError("proportions and avg_positive_fractions must have same length")

    # ideal fractional counts
    ideal = [pi * batch_size for pi in p]
    base = [math.floor(x) for x in ideal]
    remainder = [x - b for x, b in zip(ideal, base)]
    allocated = sum(base)
    remaining = batch_size - allocated

    # distribute remaining to largest remainders
    indices = sorted(range(len(remainder)), key=lambda i: (-remainder[i], i))
    for i in range(remaining):
        base[indices[i]] += 1

    # base now holds integer counts per bucket
    total_positive = sum(count * frac for count, frac in zip(base, f))
    return total_positive / batch_size


def simulate_expected_positive_fraction(
    proportions: Iterable[float],
    avg_positive_fractions: Iterable[float],
    batch_size: int,
    trials: int = 10000,
    rng_seed: int | None = None,
) -> Tuple[float, float]:
    """
    Monte Carlo simulation of expected positive fraction for binned sampling.

    Args:
        proportions: Bucket probabilities.
        avg_positive_fractions: Average positive fraction per bucket.
        batch_size: Tiles per batch.
        trials: Number of Monte Carlo trials.
        rng_seed: Optional random seed for reproducibility.

    Returns:
        (mean, std) of the empirical positive fraction across trials.
    """
    if rng_seed is not None:
        random.seed(rng_seed)
    p = _normalize(proportions)
    f = list(avg_positive_fractions)
    if len(p) != len(f):
        raise ValueError("proportions and avg_positive_fractions must have same length")

    means = []
    k = len(p)
    for _ in range(trials):
        # sample counts with multinomial-like draw using random.choices
        picks = random.choices(range(k), weights=p, k=batch_size)
        # average positive fraction across sampled tiles
        frac = sum(f[idx] for idx in picks) / batch_size
        means.append(frac)

    # compute mean and std
    mean = sum(means) / trials
    var = sum((x - mean) ** 2 for x in means) / (trials - 1) if trials > 1 else 0.0
    std = math.sqrt(var)
    return mean, std


__all__ = [
    "expected_positive_fraction_continuous",
    "expected_positive_fraction_discrete",
    "simulate_expected_positive_fraction",
]
