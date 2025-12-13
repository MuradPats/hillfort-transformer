"""
check_mask.py — inspect a single-channel mask PNG for non-zero pixels.

Example:
    python check_mask.py datasets/HillfortMVP/Label/62093_dtm_1m.png
"""
from pathlib import Path
from typing import Tuple, Dict
from PIL import Image
import numpy as np
import sys

def inspect_mask(path: Path) -> Tuple[int, int, float, Dict[int,int]]:
    """Return (total_pixels, nonzero_pixels, percent_nonzero, value_counts)."""
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.uint8)
    total = int(arr.size)
    nonzero = int((arr != 0).sum())
    percent = (nonzero / total * 100) if total else 0.0
    unique, counts = np.unique(arr, return_counts=True)
    counts_map = {int(k): int(v) for k, v in zip(unique, counts)}
    return total, nonzero, percent, counts_map

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_mask.py path/to/mask.png"); sys.exit(1)
    p = Path(sys.argv[1])
    total, nonzero, percent, counts = inspect_mask(p)
    print(f"File: {p}")
    print(f"Total pixels: {total}")
    print(f"Non-zero pixels: {nonzero} ({percent:.6f}%)")
    print("Value counts:", counts)