#!/usr/bin/env python3
"""
Smoke test to compute pixel-level class weights from tile_stats.csv
and print the resulting normalized weights.

Run from project root:
python scripts/smoke_pixel_weights.py
"""
import sys
from pathlib import Path
# Ensure repo root is importable when running the script from `scripts/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import csv

try:
    # Prefer importing the config module from the package
    from RGBX import config as cfg_mod
    config = cfg_mod.config
except Exception:
    # Fallback: try importing via module path
    import importlib

    config = importlib.import_module("RGBX.config").config


def find_csv_path(cfg):
    candidates = [
        Path(getattr(cfg, "dataset_path", "")) / "tile_stats.csv",
        Path(getattr(cfg, "root_dir", "")) / "tile_stats.csv",
        Path("tile_stats.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_train_set(cfg):
    ts = Path(getattr(cfg, "train_source", ""))
    if ts.exists():
        with open(ts, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return None


def compute_weights(cfg):
    csv_path = find_csv_path(cfg)
    if csv_path is None:
        print("tile_stats.csv not found; checked dataset_path/root_dir/current dir")
        return 2

    train_set = load_train_set(cfg)

    pos_pixels = 0
    tile_count = 0

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rgb_name = row.get("rgb_tile") or row.get("image") or row.get("image_base")
            pos_cnt = row.get("positive_count") or row.get("positive_pixels") or row.get("positive")
            try:
                pos_val = int(float(pos_cnt)) if pos_cnt not in (None, "") else 0
            except Exception:
                pos_val = 0

            stem = None
            if isinstance(rgb_name, str):
                stem = Path(rgb_name).stem

            if train_set is not None:
                if stem is None or stem not in train_set:
                    continue

            pos_pixels += pos_val
            tile_count += 1

    if tile_count == 0:
        print("No tiles matched the train set / CSV was empty")
        return 3

    pixels_per_tile = int(getattr(cfg, "image_height", 512)) * int(
        getattr(cfg, "image_width", 512)
    )
    total_pixels = float(tile_count * pixels_per_tile)
    neg_pixels = max(total_pixels - float(pos_pixels), 1.0)
    pos_pixels_f = max(float(pos_pixels), 1.0)

    w_neg = total_pixels / neg_pixels
    w_pos = total_pixels / pos_pixels_f

    mean_w = (w_neg + w_pos) / 2.0
    max_w = float(getattr(cfg, "max_class_weight", 100.0))
    w_neg = min(w_neg / mean_w, max_w)
    w_pos = min(w_pos / mean_w, max_w)

    print(f"csv_path: {csv_path}")
    print(f"tiles considered: {tile_count}")
    print(f"pos_pixels: {int(pos_pixels)}, total_pixels: {int(total_pixels)}")
    print(f"computed weights: background={w_neg:.6f}, positive={w_pos:.6f}")

    return 0


def main():
    rc = compute_weights(config)
    return rc


if __name__ == "__main__":
    sys.exit(main())
