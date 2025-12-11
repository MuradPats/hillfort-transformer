"""
Batch convert DTM and satellite TIFFs to PNG dataset structure for CMX MVP.

Creates dataset folder structure:
  datasets/HillfortMVP/
    RGB/
    DTM/
    Label/   (empty)
    train.txt
    test.txt

Usage:
  python data/batch_convert.py --dtm_dir data/raw/dtm --sat_dir data/raw/satellite --out datasets/HillfortMVP

This script converts single-band DTM -> 8-bit PNG (percentile scaling) and
multiband satellite TIFF -> RGB PNG (per-band percentile scaling). It writes
matching basenames and a `train.txt` file listing the processed tiles.
"""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
from typing import Optional, List


def scale_to_uint8(arr: np.ndarray, method: str = 'percentile', pmin: float = 2, pmax: float = 98,
                   nodata: Optional[float] = None) -> np.ndarray:
    """Scale a numeric array to 8-bit (0-255).

    Parameters
    - arr: input numeric array (2D)
    - method: 'percentile' (clip between pmin/pmax percentiles) or 'minmax'
    - pmin, pmax: percentiles used when method == 'percentile'
    - nodata: optional nodata value to ignore during scaling

    Returns
    - uint8 numpy array with same shape as `arr`.
    """
    a = arr.astype(np.float32)
    # Build mask for nodata values
    if nodata is None:
        mask = np.isnan(a)
    else:
        mask = (a == nodata)
    a[mask] = np.nan

    # Determine scaling bounds
    if method == 'percentile':
        vmin = np.nanpercentile(a, pmin)
        vmax = np.nanpercentile(a, pmax)
    else:
        vmin = np.nanmin(a)
        vmax = np.nanmax(a)

    # Avoid division by zero
    if np.isclose(vmin, vmax):
        out = np.zeros_like(a, dtype=np.uint8)
    else:
        out = (255.0 * (a - vmin) / (vmax - vmin)).clip(0, 255)
        out = np.nan_to_num(out, nan=0).astype(np.uint8)
    return out


def process_dtm(src_path: str, dst_path: str, method: str = 'percentile', pmin: float = 2,
                pmax: float = 98) -> None:
    """Read a single-band DTM and save an 8-bit PNG.

    The function reads the first band from `src_path`, scales it to 0-255
    with `scale_to_uint8` and writes a single-channel PNG to `dst_path`.
    """
    with rasterio.open(src_path) as src:
        arr = src.read(1)
        prof = src.profile
        nodata = prof.get('nodata', None)

    out = scale_to_uint8(arr, method=method, pmin=pmin, pmax=pmax, nodata=nodata)
    img = Image.fromarray(out, mode='L')
    img.save(dst_path)


def process_satellite(src_path: str, dst_path: str, method: str = 'percentile', pmin: float = 2,
                      pmax: float = 98) -> None:
    """Read a satellite raster and write an RGB PNG.

    Attempts to read up to three bands (R,G,B). If only one band exists it is
    replicated to RGB. If more than three bands are present, the first three
    are used. Each band is scaled independently to 8-bit using
    `scale_to_uint8`.
    """
    with rasterio.open(src_path) as src:
        bands = src.count
        # Read first 3 bands if available, otherwise replicate single band
        if bands >= 3:
            b1 = src.read(1)
            b2 = src.read(2)
            b3 = src.read(3)
            arr = np.stack([b1, b2, b3], axis=0)
        elif bands == 1:
            b = src.read(1)
            arr = np.stack([b, b, b], axis=0)
        else:
            # Read all bands and take first three channels
            data = src.read()
            if data.shape[0] >= 3:
                arr = data[:3]
            else:
                # fallback: replicate mean across three channels
                mean = data.mean(axis=0)
                arr = np.stack([mean, mean, mean], axis=0)

        # If already uint8, skip scaling
        if src.dtypes[0] == 'uint8':
            rgb = np.stack([arr[0], arr[1], arr[2]], axis=2).astype(np.uint8)
            Image.fromarray(rgb, mode='RGB').save(dst_path)
            return

    # scale each band independently to uint8 and stack into HxWx3
    out_bands: List[np.ndarray] = []
    for i in range(arr.shape[0]):
        out = scale_to_uint8(arr[i], method=method, pmin=pmin, pmax=pmax, nodata=None)
        out_bands.append(out)

    rgb = np.stack(out_bands, axis=2)
    img = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    img.save(dst_path)


def find_tifs(folder: Path) -> List[Path]:
    """Return a sorted list of .tif files under `folder` (recursive)."""
    return sorted([p for p in folder.glob('**/*.tif') if p.is_file()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtm_dir', default='data/raw/dtm')
    parser.add_argument('--sat_dir', default='data/raw/satellite')
    parser.add_argument('--out', default='datasets/HillfortMVP')
    parser.add_argument('--method', default='percentile', choices=['minmax', 'percentile'])
    parser.add_argument('--pmin', type=float, default=2.0)
    parser.add_argument('--pmax', type=float, default=98.0)
    parser.add_argument('--write_test', action='store_true', help='Also write test.txt (duplicates train.txt by default)')
    args = parser.parse_args()

    dtm_dir = Path(args.dtm_dir)
    sat_dir = Path(args.sat_dir)
    out_dir = Path(args.out)

    rgb_out = out_dir / 'RGB'
    dtm_out = out_dir / 'DTM'
    label_out = out_dir / 'Label'
    rgb_out.mkdir(parents=True, exist_ok=True)
    dtm_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    dtm_files = find_tifs(dtm_dir) if dtm_dir.exists() else []
    sat_files = find_tifs(sat_dir) if sat_dir.exists() else []

    basenames = set()

    # process DTM files
    for p in tqdm(dtm_files, desc='DTM'): 
        name = p.stem
        out_path = dtm_out / f"{name}.png"
        try:
            process_dtm(str(p), str(out_path), method=args.method, pmin=args.pmin, pmax=args.pmax)
            basenames.add(name)
        except Exception as e:
            print(f"Failed to process DTM {p}: {e}")

    # process satellite files
    for p in tqdm(sat_files, desc='SAT'):
        name = p.stem
        out_path = rgb_out / f"{name}.png"
        try:
            process_satellite(str(p), str(out_path), method=args.method, pmin=args.pmin, pmax=args.pmax)
            basenames.add(name)
        except Exception as e:
            print(f"Failed to process SAT {p}: {e}")

    # create train/test files listing basenames that have both RGB and DTM
    common = []
    for name in sorted(basenames):
        rgb_f = rgb_out / f"{name}.png"
        dtm_f = dtm_out / f"{name}.png"
        if rgb_f.exists() and dtm_f.exists():
            common.append(name)

    train_file = out_dir / 'train.txt'
    test_file = out_dir / 'test.txt'
    with open(train_file, 'w') as f:
        for n in common:
            f.write(n + '\n')
    if args.write_test:
        with open(test_file, 'w') as f:
            for n in common:
                f.write(n + '\n')

    print(f"Processed {len(common)} tiles. Dataset saved to {out_dir}")


if __name__ == '__main__':
    main()
