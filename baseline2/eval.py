from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

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
    parser.add_argument("--save-path", type=str, default=None, help="Directory prefix to save predictions (creates <save-path> and <save-path>_color)")
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

    # Prepare save dirs if requested
    save_path = None
    save_color_path = None
    if args.save_path:
        save_path = Path(args.save_path)
        save_color_path = Path(f"{args.save_path}_color")
        save_path.mkdir(parents=True, exist_ok=True)
        save_color_path.mkdir(parents=True, exist_ok=True)

    ious = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0)  # H,W
            iou = compute_iou(pred.cpu(), y.squeeze(0).cpu(), num_classes=args.num_classes)
            ious.append(iou)

            # Save predicted masks if requested. Use dataset ordering (loader is not shuffled).
            if save_path is not None:
                try:
                    tid = ds.tile_ids[i - 1]
                except Exception:
                    tid = f"sample_{i}"

                pred_np = pred.detach().cpu().numpy().astype(np.uint8)

                # binary image (0/1)
                # Save values as 0 or 1 (uint8) per request
                bin_img = pred_np.astype(np.uint8)
                bin_p = save_path / f"{tid}.png"
                Image.fromarray(bin_img).save(bin_p)

                # colored binary: background black, positive class green
                h, w = pred_np.shape
                # colored: background black, positive = red
                color = np.zeros((h, w, 3), dtype=np.uint8)
                color[pred_np == 1] = np.array([255, 0, 0], dtype=np.uint8)
                color_p = save_color_path / f"{tid}.png"
                Image.fromarray(color).save(color_p)

            if args.limit and i >= args.limit:
                break

    miou = float(np.mean(ious)) if ious else 0.0
    print(f"Eval on {min(len(ds), args.limit) if args.limit else len(ds)} samples | mIoU={miou:.4f}")

if __name__ == "__main__":
    main()
