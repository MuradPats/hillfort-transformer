from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

from baseline2.dataset import HillfortDataset
from baseline2.model import UNetSmall

@torch.no_grad()
def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets/HillfortDataSet")
    parser.add_argument("--split-list", default="test.txt")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--use-dtm", action="store_true", default=True)
    parser.add_argument("--no-dtm", action="store_false", dest="use_dtm")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = HillfortDataset(Path(args.dataset_root), args.split_list, use_dtm=args.use_dtm)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    in_ch = 4 if args.use_dtm else 3
    model = UNetSmall(in_channels=in_ch, num_classes=args.num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ious = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0)  # H,W
            iou = compute_iou(pred.cpu(), y.squeeze(0).cpu(), num_classes=args.num_classes)
            ious.append(iou)

            if args.limit and i >= args.limit:
                break

    miou = float(np.mean(ious)) if ious else 0.0
    print(f"Eval on {min(len(ds), args.limit) if args.limit else len(ds)} samples | mIoU={miou:.4f}")

if __name__ == "__main__":
    main()
