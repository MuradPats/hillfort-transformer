from __future__ import annotations

import argparse
from pathlib import Path
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline2.config import Config
from baseline2.dataset import HillfortDataset
from baseline2.model import UNetSmall

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    # pred/target are HxW tensors (class indices)
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
    parser.add_argument("--train-list", default="train.txt")
    parser.add_argument("--val-list", default="test.txt")
    parser.add_argument("--use-dtm", action="store_true", default=True)
    parser.add_argument("--no-dtm", action="store_false", dest="use_dtm")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="runs/baseline2")
    args = parser.parse_args()

    cfg = Config(
        dataset_root=Path(args.dataset_root),
        train_list=args.train_list,
        val_list=args.val_list,
        use_dtm=args.use_dtm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=(args.steps_per_epoch if args.steps_per_epoch > 0 else None),
        lr=args.lr,
        num_workers=args.num_workers,
    )

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = HillfortDataset(cfg.dataset_root, cfg.train_list, use_dtm=cfg.use_dtm, dtm_scale=cfg.dtm_scale)
    val_ds   = HillfortDataset(cfg.dataset_root, cfg.val_list,   use_dtm=cfg.use_dtm, dtm_scale=cfg.dtm_scale)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, cfg.num_workers // 2), pin_memory=torch.cuda.is_available()
    )

    in_ch = 4 if cfg.use_dtm else 3
    model = UNetSmall(in_channels=in_ch, num_classes=cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_dir = Path(cfg.dataset_root.parent) / cfg.dataset_root.name  # just to avoid accidental weirdness
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            global_step += 1

            if cfg.steps_per_epoch is not None and step >= cfg.steps_per_epoch:
                break

        avg_loss = running_loss / max(1, step)
        dt = time.time() - t0

        # quick val IoU
        model.eval()
        val_ious = []
        with torch.no_grad():
            for i, (xv, yv) in enumerate(val_loader, start=1):
                xv = xv.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)

                logits = model(xv)
                pred = torch.argmax(logits, dim=1).squeeze(0)  # H,W
                iou = compute_iou(pred.cpu(), yv.squeeze(0).cpu(), num_classes=cfg.num_classes)
                val_ious.append(iou)

                if i >= 25:  # keep val quick by default
                    break

        mean_iou = float(np.mean(val_ious)) if val_ious else 0.0
        print(f"Epoch {epoch}/{cfg.epochs} | loss {avg_loss:.4f} | val mIoU {mean_iou:.4f} | {dt:.1f}s")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": cfg.__dict__,
        }
        torch.save(ckpt, save_dir / f"ckpt_epoch_{epoch}.pt")

    print(f"Done. Checkpoints in: {save_dir}")

if __name__ == "__main__":
    main()
