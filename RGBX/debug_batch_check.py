from dataloader.dataloader import get_train_loader
from dataloader.RGBXDataset import RGBXDataset
from engine.engine import Engine
from config import config
import torch
from pathlib import Path
import os
import faulthandler

faulthandler.enable()

config.num_workers = 0

torch.autograd.set_detect_anomaly(True)

engine = Engine()  # will read args; run with same args you use
# If a stratified buckets directory exists, enable it for the loader so we exercise the new sampler
default_buckets = Path("../datasets/HillfortMVP/")
if default_buckets.exists():
    config.stratified_buckets_dir = str(default_buckets)

train_loader, _ = get_train_loader(engine, RGBXDataset)
it = iter(train_loader)
batch = next(it)
imgs = batch["data"]
gts = batch["label"]
xs = batch["modal_x"]
import numpy as np

print(
    "imgs:",
    imgs.dtype,
    imgs.min().item(),
    imgs.max().item(),
    torch.isfinite(imgs).all().item(),
)
print(
    "modal_x:",
    xs.dtype,
    xs.min().item(),
    xs.max().item(),
    torch.isfinite(xs).all().item(),
)
# detailed histogram of ground-truth pixels
gts_np = gts.numpy()
unique, counts = np.unique(gts_np, return_counts=True)
hist = dict(zip(unique.tolist(), counts.tolist()))
print("gts unique:", unique)
print("gts histogram:", hist)
print("any gts==255:", (gts == 255).any().item())
print(
    "imgs finite:",
    torch.isfinite(imgs).all().item(),
    imgs.min().item(),
    imgs.max().item(),
)
print(
    "modal_x finite:", torch.isfinite(xs).all().item(), xs.min().item(), xs.max().item()
)
print("gts unique:", torch.unique(gts))

# Print filenames and which bucket they belong to (if buckets available)
fns = batch.get("fn", None)
if fns is not None:
    # collated batch: fns may be a list or tensor; convert to python list
    try:
        fn_list = list(fns)
    except Exception:
        fn_list = [fns]

    print("filenames in batch:")
    for fn in fn_list:
        print("  ", fn)

    # check bucket membership
    # Print filenames and which bucket they belong to (if buckets available)
    print("filenames in batch:")
    for fn in fn_list:
        print("  ", fn)

        # try to locate the label file on disk using common dataset locations
        # common layout: datasets/HillfortMVP/Label/<label_filename>
        label_candidates = []
        try:
            dataset_root = Path("datasets") / "HillfortMVP"
            # possible subfolders
            for sub in ["Label", "label", "labels", "Labels"]:
                p = dataset_root / sub / (key + ".png")
                label_candidates.append(p)
                p2 = dataset_root / sub / (key + ".tif")
                label_candidates.append(p2)
        except Exception:
            label_candidates = []

        found = None
        for p in label_candidates:
            if p.exists():
                found = p
                break
        print("    label file found:", found)
        if found is not None:
            # print a small histogram of the raw file contents for cross-check
            try:
                from PIL import Image

                im = Image.open(found)
                arr = np.array(im)
                u, c = np.unique(arr, return_counts=True)
                print("    on-disk label unique:", u, "counts:", c)
            except Exception as e:
                print("    could not read label file:", e)
    bdir = (
        Path(config.stratified_buckets_dir)
        if getattr(config, "stratified_buckets_dir", None)
        else None
    )
    if bdir and bdir.exists():
        buckets = {}
        for name in ["neg.txt", "small_pos.txt", "mid_pos.txt", "full_pos.txt"]:
            p = bdir / name
            if p.exists():
                with open(p, "r") as f:
                    for line in f:
                        buckets[line.strip()] = name

        print("bucket membership:")
        for fn in fn_list:
            # normalize to a stem for reliable lookup (handles paths and extensions)
            try:
                key = Path(str(fn)).stem
            except Exception:
                key = os.path.splitext(str(fn))[0]
            print(
                "  ", fn, "->", buckets.get(key, "(unknown)"), "(lookup key:", key, ")"
            )

# Compute batch positive fraction and whether downweight would trigger
# show both raw (counts include ignore value 255) and adjusted (exclude 255)
raw_pos_frac = float((gts > 0).float().mean().item())
valid_mask = (gts != 255)
valid_count = int(valid_mask.sum().item())
if valid_count == 0:
    adj_pos_frac = 0.0
else:
    adj_pos_frac = float(((gts > 0) & valid_mask).float().sum().item() / valid_count)

print(f"batch positive fraction (raw incl.255): {raw_pos_frac:.6f}")
print(f"batch positive fraction (adjusted excl.255): {adj_pos_frac:.6f} (valid pixels: {valid_count})")
th = getattr(config, "full_pos_frac_threshold", None)
dw = getattr(config, "full_pos_downweight", None)
if th is not None and dw is not None:
    print(f"downweight threshold: {th}, downweight factor: {dw}")
    # decide using adjusted fraction (exclude ignore value)
    if adj_pos_frac >= th:
        print("This batch WOULD be downweighted by factor", dw)
    else:
        print("This batch would NOT be downweighted")
