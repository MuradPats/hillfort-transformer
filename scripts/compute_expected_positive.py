"""CLI to compute expected positive percentage for stratified sampling.

Usage examples:
  python scripts/compute_expected_positive.py --proportions 0.5,0.25,0.15,0.1 \
      --fractions 0.0,0.02,0.07,0.2 --batch 8 --trials 2000
"""

from __future__ import annotations

import argparse
from typing import List

from RGBX.utils.expected_positive import (
    expected_positive_fraction_continuous,
    expected_positive_fraction_discrete,
    simulate_expected_positive_fraction,
)


def _parse_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proportions", required=True, help="Comma-separated bucket proportions"
    )
    parser.add_argument(
        "--fractions",
        required=True,
        help="Comma-separated avg positive fractions per bucket",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size (tiles)")
    parser.add_argument(
        "--trials", type=int, default=0, help="Monte Carlo trials (0 to skip)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for simulation"
    )
    args = parser.parse_args()

    p = _parse_list(args.proportions)
    f = _parse_list(args.fractions)

    cont = expected_positive_fraction_continuous(p, f)
    disc = expected_positive_fraction_discrete(p, f, args.batch)

    print(f"Continuous expected positive fraction: {cont:.6f} ({cont * 100:.4f}%)")
    print(
        f"Discrete (batch_size={args.batch}) expected fraction: {disc:.6f} ({disc * 100:.4f}%)"
    )

    if args.trials and args.trials > 0:
        mean, std = simulate_expected_positive_fraction(
            p, f, args.batch, trials=args.trials, rng_seed=args.seed
        )
        print(f"Simulated mean: {mean:.6f} ({mean * 100:.4f}%), std: {std:.6f}")


if __name__ == "__main__":
    main()
