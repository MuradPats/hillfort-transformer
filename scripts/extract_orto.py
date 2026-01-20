#!/usr/bin/env python3
"""Extract all .zip files found in the orto folder into the same folder.

This script uses only the Python standard library so it will work on systems
without the `unzip` binary.

Usage:
  python scripts/extract_orto.py                     # use default repo-relative orto
  python scripts/extract_orto.py --path /abs/path/to/orto --overwrite

Options:
  --path PATH       Path to the folder containing .zip files (default: repo/data/orto)
  --overwrite       Overwrite existing files when extracting
  --dry-run         Show which files would be extracted without writing

The script detects the repository root relative to its own location and
resolves the default `data/orto` path accordingly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import zipfile


logger = logging.getLogger("extract_orto")


def extract_zip(zip_path: Path, dest_dir: Path, overwrite: bool = False) -> int:
    """Extract a single zipfile to dest_dir.

    Returns the number of files extracted.
    """
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = dest_dir / member.filename
            # Protect extraction to dest_dir
            try:
                target.relative_to(dest_dir)
            except Exception:
                logger.warning(
                    "Skipping suspicious member %s in %s", member.filename, zip_path
                )
                continue

            if target.exists() and not overwrite:
                logger.debug("Skipping existing: %s", target)
                continue

            # Ensure parent exists
            target.parent.mkdir(parents=True, exist_ok=True)
            # Extract the single member
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract .zip files in the orto folder using Python."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to orto folder (default: repo/data/orto)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to extract but don't write them",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )

    args = parser.parse_args(argv)

    # Configure logging
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Resolve default repo-relative path (script in scripts/)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    default_orto = repo_root / "data" / "orto"

    orto_dir = Path(args.path).resolve() if args.path else default_orto

    if not orto_dir.exists() or not orto_dir.is_dir():
        logger.error("Orto directory not found: %s", orto_dir)
        return 2

    zip_files = sorted(orto_dir.glob("*.zip"))
    if not zip_files:
        logger.info("No .zip files found in %s", orto_dir)
        return 0

    total_extracted = 0
    for z in zip_files:
        logger.info("Found zip: %s", z.name)
        if args.dry_run:
            logger.info("Would extract: %s -> %s", z, orto_dir)
            continue
        try:
            n = extract_zip(z, orto_dir, overwrite=args.overwrite)
            logger.info("Extracted %d files from %s", n, z.name)
            total_extracted += n
        except zipfile.BadZipFile:
            logger.error("Bad zip file: %s", z)
        except Exception as e:
            logger.exception("Error extracting %s: %s", z, e)

    logger.info("Total files extracted: %d", total_extracted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
