"""
parse_slurm_logs.py

Scan SLURM output files in the `slurm_outputs` directory and extract per-epoch
metrics for two log formats used in this repo:

- RGBX training logs (prefix: "slurm-transformers_train_rgbx") contain per-iteration
  lines like:
    Epoch 40/40 Iter 1244/1251: lr=2.2990e-08 loss=0.0149 total_loss=0.0296: [...]
  We collect the final `total_loss` seen for each epoch.

- Baseline training logs (prefix: "slurm-baseline2_train") contain per-epoch
  summary lines like:
    Epoch 1/10 | loss 0.1727 | val mIoU 1.0000 | 498.9s
  We collect `loss`, `val_mIoU` and epoch time (seconds).

The script writes one CSV per matched SLURM file into
`slurm_outputs/results/` with columns appropriate for each format.

Usage (from repo root):
    python scripts/parse_slurm_logs.py --slurm-dir hillfort-transformer/slurm_outputs

This script uses only Python stdlib so it should run in minimal environments.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

RGBX_PREFIX = "slurm-transformers_train_rgbx"
BASELINE_PREFIX = "slurm-baseline2_train"

# Regex patterns
RGBX_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)/(\d+)\s+Iter\s+(?P<iter>\d+)/(\d+):.*?total_loss=(?P<total_loss>[0-9eE+\-\.]+)"
)
BASELINE_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)/(\d+)\s+\|\s+loss\s+(?P<loss>[0-9eE+\-\.]+)\s+\|\s+val mIoU\s+(?P<val_mIoU>[0-9eE+\-\.]+)\s+\|\s*(?P<time_sec>[0-9eE+\-\.]+)s"
)

# Evaluation summary and per-class lines
EVAL_SUMMARY_RE = re.compile(
    r"mean_IoU\s*(?P<mean_IoU>[0-9.]+)%\s*freq_IoU\s*(?P<freq_IoU>[0-9.]+)%\s*mean_pixel_acc\s*(?P<mean_pixel_acc>[0-9.]+)%\s*pixel_acc\s*(?P<pixel_acc>[0-9.]+)%"
)
EVAL_CLASS_RE = re.compile(r"^\s*(?P<index>\d+)\s+(?P<class>\S+)\t(?P<pct>[0-9.]+)%")


def parse_rgbx_lines(lines: List[str]) -> List[Dict[str, object]]:
    """Parse RGBX per-iteration lines and return one row per epoch.

    For each epoch we keep the last-seen `total_loss` (the final iteration).
    """
    epoch_latest: Dict[int, Dict[str, object]] = {}
    for ln in lines:
        m = RGBX_RE.search(ln)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        iter_no = int(m.group("iter"))
        total_loss = float(m.group("total_loss"))
        # store/update latest seen for the epoch
        epoch_latest[epoch] = {
            "epoch": epoch,
            "iter": iter_no,
            "total_loss": total_loss,
        }
    # return rows sorted by epoch
    return [epoch_latest[k] for k in sorted(epoch_latest.keys())]


def parse_baseline_lines(lines: List[str]) -> List[Dict[str, object]]:
    """Parse baseline per-epoch summary lines and return rows."""
    rows: List[Dict[str, object]] = []
    for ln in lines:
        m = BASELINE_RE.search(ln)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        loss = float(m.group("loss"))
        val_miou = float(m.group("val_mIoU"))
        time_sec = float(m.group("time_sec"))
        rows.append(
            {"epoch": epoch, "loss": loss, "val_mIoU": val_miou, "time_sec": time_sec}
        )
    return rows


def parse_eval_lines(lines: List[str]):
    """Parse evaluation output returning (summary_dict, classes_list).

    summary_dict contains mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc as floats.
    classes_list is a list of dicts with keys: index, class, pct.
    """
    summary = {}
    classes = []
    for ln in lines:
        m = EVAL_CLASS_RE.search(ln)
        if m:
            classes.append({
                "index": int(m.group("index")),
                "class": m.group("class"),
                "pct": float(m.group("pct")),
            })
            continue
        m2 = EVAL_SUMMARY_RE.search(ln)
        if m2:
            summary = {
                "mean_IoU": float(m2.group("mean_IoU")),
                "freq_IoU": float(m2.group("freq_IoU")),
                "mean_pixel_acc": float(m2.group("mean_pixel_acc")),
                "pixel_acc": float(m2.group("pixel_acc")),
            }
    return summary, classes


def process_file(path: Path, out_dir: Path) -> List[Path]:
    """Process a single SLURM file. Detects format by filename prefix and
    writes one or more CSVs to `out_dir`.

    Returns list of written CSV paths (may be empty).
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    basename = path.name
    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if basename.startswith(RGBX_PREFIX):
        rows = parse_rgbx_lines(text)
        if rows:
            out_path = out_dir / (basename + ".csv")
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "iter", "total_loss"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            written.append(out_path)

    elif basename.startswith(BASELINE_PREFIX):
        rows = parse_baseline_lines(text)
        if rows:
            out_path = out_dir / (basename + ".csv")
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "loss", "val_mIoU", "time_sec"]
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            written.append(out_path)

    # evaluation logs (filename containing "_eval") -> write summary and classes CSVs
    elif "_eval" in basename:
        summary, classes = parse_eval_lines(text)
        if summary:
            out_summary = out_dir / (basename + ".csv")
            with out_summary.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["mean_IoU", "freq_IoU", "mean_pixel_acc", "pixel_acc"])
                writer.writeheader()
                writer.writerow(summary)
            written.append(out_summary)
        if classes:
            out_classes = out_dir / (basename + "_classes.csv")
            with out_classes.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "class", "pct"])
                writer.writeheader()
                for c in classes:
                    writer.writerow(c)
            written.append(out_classes)

    return written


def find_and_process(slurm_dir: Path, out_dir: Path) -> List[Path]:
    """Find SLURM files in `slurm_dir` matching the known prefixes and process them.

    Returns list of created CSV paths.
    """
    created: List[Path] = []
    if not slurm_dir.exists():
        raise FileNotFoundError(f"SLURM directory does not exist: {slurm_dir}")

    for p in sorted(slurm_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith((RGBX_PREFIX, BASELINE_PREFIX)) or "_eval" in p.name:
            out_paths = process_file(p, out_dir)
            for out in out_paths:
                created.append(out)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SLURM training logs into CSV per-epoch metrics."
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=Path("hillfort-transformer") / "slurm_outputs",
        help="Path to the directory containing SLURM output files (default: hillfort-transformer/slurm_outputs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write results. Default is `<slurm-dir>/results`.",
    )
    args = parser.parse_args()

    slurm_dir = args.slurm_dir
    out_dir = args.out_dir or (slurm_dir / "results")

    created = find_and_process(slurm_dir, out_dir)
    if created:
        print("Wrote:")
        for c in created:
            print(" -", c)
    else:
        print("No matching SLURM files found or no metrics extracted.")


if __name__ == "__main__":
    main()
