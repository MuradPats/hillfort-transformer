#!/usr/bin/env python3
"""
Tile the HillfortMVP dataset into n x n or fixed-size tiles and compute
positive-pixel fraction for each tile.

Saves tiles (optional) and a CSV with per-tile statistics.

Usage examples:
python scripts/tile_dataset.py --root datasets/HillfortMVP --out data/tiles --tile-size 512 --save-tiles
python scripts/tile_dataset.py --root datasets/HillfortMVP --out data/tiles --n-tiles 10 --stride 512
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pad_tile(img: Image.Image, tile_w: int, tile_h: int, fill=0) -> Image.Image:
    """Pad an image to (tile_w, tile_h) by pasting into a new image filled with `fill`."""
    if img.width == tile_w and img.height == tile_h:
        return img
    mode = img.mode
    new = Image.new(mode, (tile_w, tile_h), color=fill)
    new.paste(img, (0, 0))
    return new


def tile_image_pair(
    rgb_path: Path,
    label_path: Path,
    dtm_path: Path | None,
    out_rgb_dir: Path,
    out_label_dir: Path,
    out_dtm_dir: Path | None,
    tile_w: int,
    tile_h: int,
    stride: int,
    save_tiles: bool,
) -> list[tuple[str, int, int, str, str, str, float, int]]:
    """
    Tile a single RGB/label image pair and return per-tile stats.

    Returns list of tuples:
      (image_base, x, y, rgb_tile_name, label_tile_name, positive_fraction, positive_count)
    """
    img = Image.open(rgb_path).convert("RGB")
    lbl = Image.open(label_path).convert("L")

    w, h = img.width, img.height
    results = []

    # If image smaller than tile, handle once at end (pad)
    y_starts = list(range(0, max(1, h - tile_h + 1), stride))
    x_starts = list(range(0, max(1, w - tile_w + 1), stride))

    for y in y_starts:
        for x in x_starts:
            box = (x, y, x + tile_w, y + tile_h)
            rgb_tile = img.crop(box)
            lbl_tile = lbl.crop(box)
            # pad if needed (right/bottom edge)
            rgb_tile = pad_tile(rgb_tile, tile_w, tile_h, fill=(0, 0, 0))
            lbl_tile = pad_tile(lbl_tile, tile_w, tile_h, fill=0)

            # handle dtm if present
            dtm_name = ""
            if dtm_path is not None:
                dtm_img = Image.open(dtm_path).convert("F")
                dtm_tile = dtm_img.crop(box)
                dtm_tile = pad_tile(dtm_tile, tile_w, tile_h, fill=0)
                dtm_name = f"{rgb_path.stem}_x{x:05d}_y{y:05d}.tif"

            lbl_arr = np.array(lbl_tile, dtype=np.uint8)
            # treat any non-zero as positive (handles 1 and 255 label encodings)
            pos_mask = lbl_arr > 0
            pos_count = int(pos_mask.sum())
            pos_frac = float(pos_count) / (tile_w * tile_h)

            base = rgb_path.stem
            rgb_name = f"{base}_x{x:05d}_y{y:05d}.png"
            lbl_name = f"{base}_x{x:05d}_y{y:05d}.png"

            if save_tiles:
                rgb_tile.save(out_rgb_dir / rgb_name)
                lbl_tile.save(out_label_dir / lbl_name)
                if dtm_path is not None and out_dtm_dir is not None:
                    # save DTM as 32-bit TIFF float
                    dtm_tile.save(out_dtm_dir / dtm_name)

            results.append(
                (base, x, y, rgb_name, lbl_name, dtm_name, pos_frac, pos_count)
            )

    # Handle case where image smaller than tile or remaining right/bottom strip
    if (h < tile_h or w < tile_w) or (y_starts == [] or x_starts == []):
        x, y = 0, 0
        rgb_tile = pad_tile(img, tile_w, tile_h, fill=(0, 0, 0))
        lbl_tile = pad_tile(lbl, tile_w, tile_h, fill=0)
        dtm_name = ""
        if dtm_path is not None:
            dtm_img = Image.open(dtm_path).convert("F")
            dtm_tile = pad_tile(dtm_img, tile_w, tile_h, fill=0)
            dtm_name = f"{rgb_path.stem}_x{x:05d}_y{y:05d}.tif"

        lbl_arr = np.array(lbl_tile, dtype=np.uint8)
        pos_mask = lbl_arr > 0
        pos_count = int(pos_mask.sum())
        pos_frac = float(pos_count) / (tile_w * tile_h)
        base = rgb_path.stem
        rgb_name = f"{base}_x{x:05d}_y{y:05d}.png"
        lbl_name = rgb_name
        if save_tiles:
            rgb_tile.save(out_rgb_dir / rgb_name)
            lbl_tile.save(out_label_dir / lbl_name)
            if dtm_path is not None and out_dtm_dir is not None:
                dtm_tile.save(out_dtm_dir / dtm_name)
        results.append((base, x, y, rgb_name, lbl_name, dtm_name, pos_frac, pos_count))

    return results


def compute_tile_grid(
    img_w: int, img_h: int, n_tiles: int
) -> Tuple[int, int, int, int]:
    """Compute tile size and stride to split image into n_tiles x n_tiles grid.

    Returns (tile_w, tile_h, stride_x, stride_y).
    We'll use equal tile size in x and y and stride == tile size (no overlap) by default.
    """
    tile_w = math.ceil(img_w / n_tiles)
    tile_h = math.ceil(img_h / n_tiles)
    stride_x, stride_y = tile_w, tile_h
    return tile_w, tile_h, stride_x, stride_y


def find_pairs(root: Path) -> list[tuple[Path, Path, Path | None]]:
    """Find RGB/Label filename pairs under `root`. Assumes structure with subfolders `RGB` and `Label`.
    Matches by filename (stem). Also looks for optional DTM files in `DTM` subfolder.
    """
    rgb_dir = root / "RGB"
    label_dir = root / "Label"
    if not rgb_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected RGB and Label folders under {root}")

    rgb_files = {p.stem: p for p in rgb_dir.glob("*.png")}  # adjust ext if necessary
    label_files = {p.stem: p for p in label_dir.glob("*.png")}

    # optional DTM folder
    dtm_dir = root / "DTM"
    dtm_files: dict[str, Path] = {}
    if dtm_dir.exists():
        for p in dtm_dir.glob("*"):
            dtm_files[p.stem] = p

    pairs: list[tuple[Path, Path, Path | None]] = []
    for stem, rgb_path in rgb_files.items():
        lbl = label_files.get(stem)
        if lbl is None:
            continue
        dtm = dtm_files.get(stem)
        pairs.append((rgb_path, lbl, dtm))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to dataset")
    parser.add_argument(
        "--out", default="data/tiles", help="Output directory for tiles and CSV"
    )
    parser.add_argument(
        "--tile-size", type=int, default=None, help="Tile size in pixels (square)."
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=None,
        help="Split each image into n x n tiles (alternative to --tile-size)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between tiles. Defaults to tile size (no overlap)",
    )
    parser.add_argument(
        "--save-tiles", action="store_true", help="Save per-tile RGB and Label PNGs"
    )
    parser.add_argument(
        "--csv", default="tiles.csv", help="CSV filename to write per-tile stats"
    )
    args = parser.parse_args()

    # Print args for logging
    print("Arguments that will be used:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    root = Path(args.root)
    out = Path(args.out)
    ensure_dir(out)
    out_rgb = out / "RGB"
    out_lbl = out / "Label"
    ensure_dir(out_rgb)
    ensure_dir(out_lbl)

    pairs = find_pairs(root)
    if not pairs:
        print("No image pairs found under", root)
        return

    csv_path = out / args.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_base",
                "x",
                "y",
                "rgb_tile",
                "label_tile",
                "dtm_tile",
                "positive_fraction",
                "positive_count",
            ]
        )

        for rgb_path, lbl_path, dtm_path in tqdm(pairs, desc="Images", unit="image"):
            # print("Processing", rgb_path.name)
            sample_img = Image.open(rgb_path)
            w, h = sample_img.width, sample_img.height

            if args.n_tiles is not None:
                tile_w, tile_h, stride_x, stride_y = compute_tile_grid(
                    w, h, args.n_tiles
                )
                stride = min(stride_x, stride_y)
            else:
                if args.tile_size is None:
                    raise ValueError("Either --tile-size or --n-tiles must be provided")
                tile_w = tile_h = args.tile_size
                stride = args.stride if args.stride is not None else tile_w

            out_dtm = out / "DTM"
            ensure_dir(out_dtm)

            tile_results = tile_image_pair(
                rgb_path,
                lbl_path,
                dtm_path,
                out_rgb,
                out_lbl,
                out_dtm,
                tile_w,
                tile_h,
                stride,
                args.save_tiles,
            )

            for row in tile_results:
                writer.writerow(row)

    print("Done. CSV written to", csv_path)


if __name__ == "__main__":
    main()
