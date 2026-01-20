"""Dry-run helpers for `download_maps.py`.

These functions inspect rows from `linnamagede_ruudunumbrid_v2.csv` and call the
`get_*_file_url` helpers to print which filenames and URLs would be fetched.

This is intentionally non-destructive: no files are downloaded. Use this to
verify parsing of ruudunumbers and to sanity-check the geoportal lookup
functions.

Usage (from repository root):

python data/raw/test_download_maps.py --rows 8

Set environment variable `DRY_RUN_NO_NET=1` to avoid any network requests; in
that mode the script only parses ruudunumbers.
"""

from __future__ import annotations

import csv
import os
import sys
import argparse
import logging
from typing import List, Tuple, Optional

import download_maps as dm

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_ruudunumbers(cell: str) -> List[str]:
    """Parse a CSV cell like "463636, 462636" into a list of ruudunumbers.

    Trims whitespace and ignores empty entries.
    """
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def dry_run_row_urls(
    ruudunumber_2000: str, ruudunumber_10000: str
) -> List[Tuple[str, Optional[str]]]:
    """Return a list of (dataset_key, url) tuples discovered for the given row.

    The functions may return (None, None) if the geoportal lookup did not find
    a file for that ruudunumber; callers should handle exceptions.
    """
    results = []

    # laz uses 1:2000 (ruudunumber_2000)
    for r in parse_ruudunumbers(ruudunumber_2000):
        try:
            fname, url = dm.get_tava_file_url(r) or (None, None)
            results.append((f"laz:{r}", url or fname))
        except Exception as e:  # network/parse errors
            results.append((f"laz:{r}", f"ERROR: {e}"))

    # dtm, reljeef, orto use 1:10000 (ruudunumber_10000)
    for r in parse_ruudunumbers(ruudunumber_10000):
        try:
            fname, url = dm.get_dtm_file_url(r) or (None, None)
            results.append((f"dtm:{r}", url or fname))
        except Exception as e:
            results.append((f"dtm:{r}", f"ERROR: {e}"))

        try:
            fname, url = dm.get_reljeef_file_url(r) or (None, None)
            results.append((f"reljeef:{r}", url or fname))
        except Exception as e:
            results.append((f"reljeef:{r}", f"ERROR: {e}"))

        try:
            fname, url = dm.get_orto_file_url(r) or (None, None)
            results.append((f"orto:{r}", url or fname))
        except Exception as e:
            results.append((f"orto:{r}", f"ERROR: {e}"))

    return results


def dry_run_first_n(csv_path: str, n: int = 10, no_net: bool = False) -> None:
    """Inspect the first `n` rows of the CSV and print candidate filenames/URLs.

    If `no_net` is True the function will only parse ruudunumbers and skip
    calling the geoportal lookup helpers that perform network requests.
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV not found: {csv_path}")
        return

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)

        printed = 0
        for row_number, row in enumerate(reader, start=2):
            if printed >= n:
                break
            if not row or len(row) < 3:
                continue

            linnamagi_name = row[0].strip()
            r2000 = (row[1] or "").strip()
            r10000 = (row[2] or "").strip()

            if r2000 == "" and r10000 == "":
                continue

            logging.info(f"\nRow {row_number}: {linnamagi_name}")
            logging.info(f"  ruudunumber_2000: {r2000}")
            logging.info(f"  ruudunumber_10000: {r10000}")

            if no_net:
                logging.info("  DRY MODE (no network): parsed ruudunumbers:")
                logging.info(f"    2000 -> {parse_ruudunumbers(r2000)}")
                logging.info(f"    10000 -> {parse_ruudunumbers(r10000)}")
            else:
                results = dry_run_row_urls(r2000, r10000)
                for key, url in results:
                    logging.info(f"    {key}: {url}")

            printed += 1


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run downloader for Hillfort map tiles."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV")
    parser.add_argument(
        "--rows", type=int, default=10, help="How many CSV rows to inspect"
    )
    parser.add_argument(
        "--no-net",
        action="store_true",
        help="Parse ruudunumbers but do not perform network lookups",
    )

    args = parser.parse_args(argv)

    dry_run_first_n(args.csv, n=args.rows, no_net=args.no_net)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
