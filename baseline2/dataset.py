from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

def _read_png(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    return arr

def _read_tif(path: Path) -> np.ndarray:
    # Prefer rasterio for GeoTIFFs; fallback to PIL if needed.
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            arr = src.read(1)  # first band
        return arr
    except Exception:
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

def _normalize_dtm(dtm: np.ndarray) -> np.ndarray:
    # Robust normalization per-tile (avoid NaN)
    dtm = dtm.astype(np.float32)
    finite = np.isfinite(dtm)
    if not finite.any():
        return np.zeros_like(dtm, dtype=np.float32)
    v = dtm[finite]
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        return np.zeros_like(dtm, dtype=np.float32)
    out = (dtm - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)

def _binarize_label(lbl: np.ndarray) -> np.ndarray:
    # Your masks might be 0/1 or 0/255. Make it {0,1}.
    if lbl.ndim == 3:
        lbl = lbl[..., 0]
    if lbl.max() > 1:
        lbl = (lbl > 0).astype(np.uint8)
    return lbl.astype(np.uint8)

def _read_id_list(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids

@dataclass
class HillfortPaths:
    root: Path
    rgb_dir: str = "RGB"
    dtm_dir: str = "DTM"
    label_dir: str = "Label"

    def rgb_path(self, tile_id: str) -> Path:
        return self.root / self.rgb_dir / f"{tile_id}.png"

    def dtm_path(self, tile_id: str) -> Path:
        return self.root / self.dtm_dir / f"{tile_id}.tif"

    def label_path(self, tile_id: str) -> Path:
        return self.root / self.label_dir / f"{tile_id}.png"

class HillfortDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_list_file: str,
        use_dtm: bool = True,
        dtm_scale: float = 1.0,
    ):
        self.paths = HillfortPaths(dataset_root)
        self.tile_ids = _read_id_list(dataset_root / split_list_file)
        self.use_dtm = use_dtm
        self.dtm_scale = float(dtm_scale)

        # Filter out missing files early (helps avoid mid-epoch crashes)
        kept: List[str] = []
        for tid in self.tile_ids:
            rp = self.paths.rgb_path(tid)
            lp = self.paths.label_path(tid)
            dp = self.paths.dtm_path(tid)
            ok = rp.exists() and lp.exists()
            if self.use_dtm:
                ok = ok and dp.exists()
            if ok:
                kept.append(tid)
        self.tile_ids = kept

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tid = self.tile_ids[idx]

        rgb = _read_png(self.paths.rgb_path(tid))  # H,W,3
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB must be HxWx3, got {rgb.shape} for {tid}")

        rgb = rgb.astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)  # 3,H,W

        if self.use_dtm:
            dtm = _read_tif(self.paths.dtm_path(tid))  # H,W
            dtm = _normalize_dtm(dtm) * self.dtm_scale
            dtm_t = torch.from_numpy(dtm).unsqueeze(0)  # 1,H,W
            x = torch.cat([rgb_t, dtm_t], dim=0)        # 4,H,W
        else:
            x = rgb_t  # 3,H,W

        lbl = _read_png(self.paths.label_path(tid))     # H,W or H,W,?
        lbl = _binarize_label(lbl)
        y = torch.from_numpy(lbl.astype(np.int64))      # H,W (class indices)

        return x, y
