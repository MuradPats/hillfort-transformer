"""Compute per-tile confusion matrix and metrics between GT and predicted masks.

Produces a CSV with columns: tile, gt_path, pred_path, TP, FP, FN, TN,
total_pixels, pos_gt, pos_pred, pos_fraction, IoU, precision, recall, f1

Usage:
    python scripts/compute_confusion_metrics.py \
        --gt-dir data/gt_masks --pred-dir runs/modelv4_results/eval \
        --out-csv runs/modelv4_results/eval_metrics.csv
"""

from pathlib import Path
import argparse
import csv

import numpy as np
from PIL import Image


def load_mask(path: Path):
    im = Image.open(path).convert("L")
    arr = np.asarray(im)
    return arr > 0


def resize_to(arr: np.ndarray, target_shape):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_metrics(gt, pred):
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    total = gt.size
    tn = int(total - tp - fp - fn)

    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        safe_div(2 * precision * recall, precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "total": total,
        "pos_gt": int(gt.sum()),
        "pos_pred": int(pred.sum()),
        "pos_fraction": float(gt.sum()) / float(total) if total else 0.0,
        "IoU": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)

    gt_files = {p.stem: p for p in gt_dir.glob("*.png")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.png")}

    keys = sorted(set(gt_files.keys()) | set(pred_files.keys()))
    if not keys:
        print("No PNG masks found in the provided directories.")
        return

    rows = []
    sums = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total": 0}

    for k in keys:
        gt_path = gt_files.get(k)
        pred_path = pred_files.get(k)

        if gt_path is None:
            # treat missing GT as empty mask
            gt = np.zeros((5000, 5000), dtype=np.uint8)
        else:
            gt = load_mask(gt_path)

        if pred_path is None:
            pred = np.zeros_like(gt)
        else:
            pred = load_mask(pred_path)

        # resize pred to gt if shapes mismatch
        if pred.shape != gt.shape:
            pred = resize_to(pred, gt.shape)

        m = compute_metrics(gt, pred)
        sums["TP"] += m["TP"]
        sums["FP"] += m["FP"]
        sums["FN"] += m["FN"]
        sums["TN"] += m["TN"]
        sums["total"] += m["total"]

        rows.append(
            {
                "tile": k,
                "gt_path": str(gt_path) if gt_path else "",
                "pred_path": str(pred_path) if pred_path else "",
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TN": m["TN"],
                "total_pixels": m["total"],
                "pos_gt": m["pos_gt"],
                "pos_pred": m["pos_pred"],
                "pos_fraction": m["pos_fraction"],
                "IoU": m["IoU"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )

    # aggregate global IoU and metrics
    global_iou = safe_div(sums["TP"], sums["TP"] + sums["FP"] + sums["FN"])
    global_precision = safe_div(sums["TP"], sums["TP"] + sums["FP"])
    global_recall = safe_div(sums["TP"], sums["TP"] + sums["FN"])
    global_f1 = (
        safe_div(2 * global_precision * global_recall, global_precision + global_recall)
        if (global_precision + global_recall)
        else 0.0
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile",
        "gt_path",
        "pred_path",
        "TP",
        "FP",
        "FN",
        "TN",
        "total_pixels",
        "pos_gt",
        "pos_pred",
        "pos_fraction",
        "IoU",
        "precision",
        "recall",
        "f1",
    ]

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

        # write aggregate as final row
        writer.writerow(
            {
                "tile": "AGGREGATE",
                "gt_path": "",
                "pred_path": "",
                "TP": sums["TP"],
                "FP": sums["FP"],
                "FN": sums["FN"],
                "TN": sums["TN"],
                "total_pixels": sums["total"],
                "pos_gt": "",
                "pos_pred": "",
                "pos_fraction": "",
                "IoU": global_iou,
                "precision": global_precision,
                "recall": global_recall,
                "f1": global_f1,
            }
        )

    print(f"Wrote metrics for {len(rows)} tiles to {out_csv}")
    print("Global metrics:")
    print(
        f"  IoU={global_iou:.6f}  precision={global_precision:.6f}  recall={global_recall:.6f}  f1={global_f1:.6f}"
    )


if __name__ == "__main__":
    main()
