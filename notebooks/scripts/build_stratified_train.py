#!/usr/bin/env python3
"""
Build stratified train.txt from a tiles CSV produced by `tile_dataset.py`.

Creates bucket files (neg/small/mid/full) and a `train.txt` (either concatenation
or a mixed file sampled to specified proportions).

CSV format expected: header containing `rgb_tile` and `positive_fraction`.
The script writes tile stems (no extension) to txt files so they can be used
by existing training pipelines that load by basename.

Usage examples:
  python notebooks/scripts/build_stratified_train.py --csv data/tiles/tiles.csv --out data/tiles
  python notebooks/scripts/build_stratified_train.py --csv data/tiles/tiles.csv --out data/tiles --mix-size 20000

Options (defaults chosen for sparse targets):
  neg     : positive_fraction == 0
  small   : 0 < f <= 0.001
  mid     : 0.001 < f <= 0.01
  full    : f > 0.01

You can override the thresholds with `--small-thresh` and `--mid-thresh`.
If `--mix-size N` is provided, the script will create `stratified_train.txt`
with N entries sampled to match proportions (see `--proportions`).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Tiles CSV produced by tile_dataset.py")
    p.add_argument(
        "--out", required=True, help="Output folder for bucket txts and train.txt"
    )
    p.add_argument(
        "--small-thresh",
        type=float,
        default=0.001,
        help="Upper bound for small-pos bucket (exclusive of zero)",
    )
    p.add_argument(
        "--mid-thresh", type=float, default=0.01, help="Upper bound for mid-pos bucket"
    )
    p.add_argument(
        "--mix-size",
        type=int,
        default=None,
        help="If provided, create stratified_train.txt with this many entries sampled to proportions",
    )
    p.add_argument(
        "--proportions",
        type=str,
        default="0.5,0.4,0.09,0.01",
        help="Comma-separated proportions for neg,small,mid,full (must sum to 1) when --mix-size used",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return p.parse_args()


def read_tiles(csv_path: Path) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if (
            "rgb_tile" not in reader.fieldnames
            or "positive_fraction" not in reader.fieldnames
        ):
            raise ValueError(
                "CSV must contain 'rgb_tile' and 'positive_fraction' columns"
            )
        for r in reader:
            try:
                frac = float(r["positive_fraction"])
            except Exception:
                frac = 0.0
            stem = Path(r["rgb_tile"]).stem
            rows.append((stem, frac))
    return rows


def bucket_tiles(rows: List[Tuple[str, float]], small_thresh: float, mid_thresh: float):
    neg, small, mid, full = [], [], [], []
    for stem, frac in rows:
        if frac <= 0.0:
            neg.append(stem)
        elif frac <= small_thresh:
            small.append(stem)
        elif frac <= mid_thresh:
            mid.append(stem)
        else:
            full.append(stem)
    return neg, small, mid, full


def write_list(path: Path, items: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it + "\n")


def sample_bucket(bucket: List[str], n: int, rng: random.Random) -> List[str]:
    if len(bucket) == 0:
        return []
    if n <= len(bucket):
        return rng.sample(bucket, n)
    # not enough items: sample with replacement
    return [rng.choice(bucket) for _ in range(n)]


def main():
    args = parse_args()
    print("Arguments that will be used:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tiles(csv_path)
    neg, small, mid, full = bucket_tiles(rows, args.small_thresh, args.mid_thresh)

    print(f"Tiles total: {len(rows)}")
    print(f"  neg: {len(neg)}")
    print(f"  small: {len(small)}")
    print(f"  mid: {len(mid)}")
    print(f"  full: {len(full)}")

    write_list(out_dir / "neg.txt", neg)
    write_list(out_dir / "small_pos.txt", small)
    write_list(out_dir / "mid_pos.txt", mid)
    write_list(out_dir / "full_pos.txt", full)

    # By default write train.txt as concatenation (neg, small, mid, full)
    concat = neg + small + mid + full
    write_list(out_dir / "train.txt", concat)
    print(f"Wrote train.txt ({len(concat)} entries) and bucket files to {out_dir}")

    # Optionally produce a mixed stratified train file sampled to proportions
    if args.mix_size is not None:
        rng = random.Random(args.seed)
        props = [float(x) for x in args.proportions.split(",")]
        if len(props) != 4 or abs(sum(props) - 1.0) > 1e-6:
            raise ValueError(
                "--proportions must be 4 comma-separated numbers that sum to 1"
            )
        n = args.mix_size
        targets = [int(round(p * n)) for p in props]
        # adjust rounding to match total n
        diff = n - sum(targets)
        i = 0
        while diff != 0:
            targets[i % 4] += 1 if diff > 0 else -1
            diff = n - sum(targets)
            i += 1

        sampled = []
        sampled += sample_bucket(neg, targets[0], rng)
        sampled += sample_bucket(small, targets[1], rng)
        sampled += sample_bucket(mid, targets[2], rng)
        sampled += sample_bucket(full, targets[3], rng)
        rng.shuffle(sampled)
        write_list(out_dir / "stratified_train.txt", sampled)
        print(f"Wrote stratified_train.txt ({len(sampled)} entries) to {out_dir}")


if __name__ == "__main__":
    main()
