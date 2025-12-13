"""
Rasterize polygons into binary mask PNGs aligned to reference rasters.

Usage examples:
  # Rasterize polygons to a single reference raster
  python data/rasterize_polygons.py --geom data/inspire/PS_ProtectedSite_malestisedPolygon.shp \
      --ref data/raw/dtm/62093_dtm_1m.tif --out datasets/HillfortMVP/Label

  # Rasterize polygons to all rasters in a directory (matching ext .tif)
  python data/rasterize_polygons.py --geom hillforts.geojson --ref_dir data/raw/dtm --out datasets/HillfortMVP/Label

The script will reproject polygon geometries to the raster CRS if necessary,
select polygons that intersect the raster extent and burn value `--value`
(default 1) into the output mask. Output masks are saved as single-channel
uint8 PNG files with the same basename as the reference raster.
"""
from pathlib import Path
from typing import Iterable, List, Optional
import argparse
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_geoms(geom_path: Path) -> gpd.GeoDataFrame:
    """Load polygon features into a GeoDataFrame.

    Supports shapefile, GeoJSON, GeoPackage, etc.
    """
    gdf = gpd.read_file(str(geom_path))
    return gdf


def rasterize_for_raster(ref_raster: Path, geoms_gdf: gpd.GeoDataFrame, out_path: Path, value: int = 1) -> None:
    """Rasterize polygons that intersect `ref_raster` into `out_path`.

    - Reprojects `geoms_gdf` to the raster CRS if needed.
    - Selects polygons intersecting raster bounds.
    - Burns `value` where polygons are present; background is 0.
    """
    with rasterio.open(str(ref_raster)) as src:
        transform = src.transform
        width = src.width
        height = src.height
        dst_crs = src.crs

    # Reproject polygons to raster CRS if required
    if geoms_gdf.crs != dst_crs:
        geoms = geoms_gdf.to_crs(dst_crs)
    else:
        geoms = geoms_gdf

    # Select geometries that intersect the raster bounds to limit work
    with rasterio.open(str(ref_raster)) as _src:
        bounds = _src.bounds

    from shapely.geometry import box
    bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    try:
        candidates = geoms[geoms.intersects(bbox_geom)]
    except Exception:
        candidates = geoms

    # If nothing intersects, create empty mask
    if candidates.empty:
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        shapes = ((mapping(g), int(value)) for g in candidates.geometry)
        mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

    Image.fromarray(mask, mode="L").save(str(out_path))


def find_rasters(folder: Path, pattern: str = "**/*.tif") -> List[Path]:
    return sorted([p for p in folder.glob(pattern) if p.is_file()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom", required=True, help="Path to polygon file (shp, geojson)")
    parser.add_argument("--ref", help="Reference raster file (single). If provided, only this raster is processed.")
    parser.add_argument("--ref_dir", help="Directory with reference rasters (processed recursively)")
    parser.add_argument("--out", required=True, help="Output directory for masks")
    parser.add_argument("--value", type=int, default=255, help="Value to burn for polygons in mask (default 255)")
    parser.add_argument("--map_numbers", help="CSV with map numbers and INSPIRE ids (e.g. data/linnamagede_ruudunumbrid.csv)")
    parser.add_argument("--map_tile_col", default="Ruudunumber(1:10000)", help="Column name for tile id in map CSV (default: Ruudunumber(1:10000))")
    parser.add_argument("--map_inspire_col", default="INSPIRE id", help="Column name for INSPIRE id in map CSV (default: 'INSPIRE id')")
    parser.add_argument("--shp_inspire_col", default="inspireid_", help="Column name in shapefile that contains INSPIRE ids (default: inspireid_)")
    args = parser.parse_args()

    geom_path = Path(args.geom)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_geoms(geom_path)
    if gdf.empty:
        print("No geometries found in", geom_path)
        return

    map_df = None
    if args.map_numbers:
        import pandas as pd
        map_csv = Path(args.map_numbers)
        if map_csv.exists():
            map_df = pd.read_csv(str(map_csv))
        else:
            print(f"Map numbers CSV not found: {map_csv}")

    raster_list: List[Path] = []
    if args.ref:
        raster_list = [Path(args.ref)]
    elif args.ref_dir:
        raster_list = find_rasters(Path(args.ref_dir))
    else:
        raise SystemExit("Either --ref or --ref_dir must be provided")

    for ref in tqdm(raster_list, desc="Rasterizing"):
        out_path = out_dir / f"{ref.stem}.png"
        try:
            # If a map CSV is provided, try to filter polygons by tile -> INSPIRE id mapping
            if map_df is not None:
                tile_id = ref.stem
                # Match tile id in CSV (cast to str for robust comparison)
                col_tile = args.map_tile_col
                col_inspire = args.map_inspire_col
                try:
                    matched = map_df[map_df[col_tile].astype(str) == str(tile_id)]
                    inspire_ids = matched[col_inspire].dropna().astype(str).tolist()
                except Exception:
                    inspire_ids = []

                if inspire_ids:
                    # filter gdf by the shapefile's INSPIRE id column
                    shp_col = args.shp_inspire_col
                    try:
                        subset = gdf[gdf[shp_col].astype(str).isin(inspire_ids)]
                    except Exception:
                        subset = gdf

                    if subset.empty:
                        # fallback to spatial intersection
                        rasterize_for_raster(ref, gdf, out_path, value=args.value)
                    else:
                        rasterize_for_raster(ref, subset, out_path, value=args.value)
                else:
                    # No mapping for this tile, fallback to spatial intersection of all polygons
                    print(f"No INSPIRE IDs found for tile {tile_id}, using spatial intersection.")
                    rasterize_for_raster(ref, gdf, out_path, value=args.value)
            else:
                print(f"No map CSV provided, using spatial intersection for {ref.name}.")
                rasterize_for_raster(ref, gdf, out_path, value=args.value)
        except Exception as e:
            print(f"Failed for {ref}: {e}")


if __name__ == "__main__":
    main()
