"""Create binary ground-truth masks from polygon shapefile for ortho tiles.

Usage (defaults chosen for this repo layout):
    python scripts/rasterize_polygons_to_masks.py \
        --ortho-dir data/raw/orto --csv data/linnamagede_ruudunumbrid_v2.csv \
        --shapefile data/raw/inspire/PS_ProtectedSite_malestisedPolygon.shp \
        --out-dir data/gt_masks --mask-size 5000

The script will iterate all .tif files under `--ortho-dir`, find matching rows
in the CSV by tile stem (column `Ruudunumber(1:10000)` by default), collect the
INSPIRE ids from the CSV column `INSPIRE id`, filter the shapefile GeoDataFrame
by that id, rasterise geometries to the ortho image grid and save a 8-bit
PNG mask with values 0 (background) and 255 (positive).
"""
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
from PIL import Image


def parse_inspire_ids(val: str):
    """Split and clean an INSPIRE id cell which may contain separators.

    Returns a list of id strings (may be empty).
    """
    if pd.isna(val):
        return []
    # sometimes there may be multiple ids separated by commas or semicolons
    parts = [p.strip() for p in str(val).replace(';', ',').split(',')]
    return [p for p in parts if p]


def make_mask_for_tile(tif_path: Path, map_tiles_df: pd.DataFrame, gdf, cfg,
                       mask_size: int, out_path: Path):
    """Rasterise polygons for a single ortho tile and save PNG mask.

    - tif_path: path to ortho .tif
    - map_tiles_df: dataframe with tile -> INSPIRE id mapping
    - gdf: full GeoDataFrame of polygons
    - cfg: dict with csv/shapefile id column names
    - mask_size: desired output size (square pixels)
    - out_path: where to write the PNG mask
    """
    stem = tif_path.stem
    rows = map_tiles_df[map_tiles_df[cfg['tile_col']].astype(str).str.strip() == stem]

    if rows.empty:
        # no mapping rows -> empty mask
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        Image.fromarray(mask).save(out_path)
        return

    # collect INSPIRE ids (split if necessary)
    ids = []
    for val in rows[cfg['inspire_col']].dropna().unique():
        ids.extend(parse_inspire_ids(val))
    ids = list(dict.fromkeys(ids))

    if not ids:
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        Image.fromarray(mask).save(out_path)
        return

    # filter gdf by id column
    try:
        gdf_sub = gdf[gdf[cfg['gdf_id_col']].isin(ids)]
    except Exception:
        # fallback: try matching on stringified id
        gdf_sub = gdf[gdf[cfg['gdf_id_col']].astype(str).isin(ids)]

    if gdf_sub.empty:
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        Image.fromarray(mask).save(out_path)
        return

    with rasterio.open(tif_path) as src:
        src_crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width

        # reproject geometries to the raster CRS if needed
        if gdf_sub.crs != src_crs:
            try:
                gdf_sub = gdf_sub.to_crs(src_crs)
            except Exception:
                warnings.warn(f"Could not reproject geometries to {src_crs}; mask will be empty")
                mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
                Image.fromarray(mask).save(out_path)
                return

        shapes = [(mapping(geom), 1) for geom in gdf_sub.geometry if geom is not None]

        if not shapes:
            mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
            Image.fromarray(mask).save(out_path)
            return

        mask_arr = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

    # ensure output is the desired mask_size x mask_size
    if (height, width) != (mask_size, mask_size):
        img = Image.fromarray((mask_arr * 255).astype(np.uint8))
        img = img.resize((mask_size, mask_size), resample=Image.NEAREST)
    else:
        img = Image.fromarray((mask_arr * 255).astype(np.uint8))

    # ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ortho-dir", default="data/raw/orto", help="Directory with ortho TIFs")
    p.add_argument("--csv", default="data/linnamagede_ruudunumbrid_v2.csv", help="CSV mapping tiles to INSPIRE ids")
    p.add_argument("--shapefile", default="data/raw/inspire/PS_ProtectedSite_malestisedPolygon.shp", help="Polygon shapefile")
    p.add_argument("--out-dir", default="data/gt_masks", help="Output directory for PNG masks")
    p.add_argument("--mask-size", default=5000, type=int, help="Output mask size (square)")
    p.add_argument("--tile-col", default="Ruudunumber(1:10000)", help="Tile id column in CSV")
    p.add_argument("--inspire-col", default="INSPIRE id", help="INSPIRE id column in CSV")
    p.add_argument("--gdf-id-col", default="inspireid_", help="id column name in shapefile GeoDataFrame")
    args = p.parse_args()

    ortho_dir = Path(args.ortho_dir)
    csv_path = Path(args.csv)
    shapefile = Path(args.shapefile)
    out_dir = Path(args.out_dir)
    mask_size = int(args.mask_size)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not shapefile.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile}")
    if not ortho_dir.exists():
        raise FileNotFoundError(f"Ortho directory not found: {ortho_dir}")

    map_tiles_df = pd.read_csv(csv_path, dtype=str)

    # load shapefile once
    import geopandas as gpd

    gdf = gpd.read_file(shapefile)

    cfg = {
        'tile_col': args.tile_col,
        'inspire_col': args.inspire_col,
        'gdf_id_col': args.gdf_id_col,
    }

    tif_files = sorted(ortho_dir.glob("*.tif"))
    if not tif_files:
        tif_files = sorted(ortho_dir.rglob("*.tif"))

    if not tif_files:
        print(f"No .tif files found under {ortho_dir}")
        return

    for tif in tif_files:
        out_mask = out_dir / f"{tif.stem}.png"
        print(f"Processing {tif.name} -> {out_mask}")
        try:
            make_mask_for_tile(tif, map_tiles_df, gdf, cfg, mask_size, out_mask)
        except Exception as e:
            print(f"ERROR processing {tif}: {e}")


if __name__ == "__main__":
    main()
