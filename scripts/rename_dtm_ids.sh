#!/bin/bash
set -euo pipefail

# Rename DTM files like 44984_dtm_1m.tif -> 44984.tif
# Dry-run by default; pass --apply or -y to perform changes.

SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DTM_DIR="$REPO_ROOT/data/dtm"

DRY_RUN=1
if [ "${1:-}" = "--apply" ] || [ "${1:-}" = "-y" ]; then
  DRY_RUN=0
fi

echo "DTM directory: $DTM_DIR"
if [ ! -d "$DTM_DIR" ]; then
  echo "Directory not found: $DTM_DIR" >&2
  exit 1
fi

shopt -s nullglob
count=0
for f in "$DTM_DIR"/*.tif; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  id="${base%%_*}"
  target="$DTM_DIR/${id}.tif"

  # Skip if already in desired form
  if [ "$f" = "$target" ]; then
    continue
  fi

  # Avoid clobbering existing files
  if [ -e "$target" ]; then
    echo "Skipping (target exists): $base -> $(basename "$target")"
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN: $base -> $(basename "$target")"
  else
    mv -- "$f" "$target"
    echo "Renamed: $base -> $(basename "$target")"
  fi
  count=$((count+1))
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete. To apply changes, run: $0 --apply"
else
  echo "Renamed $count files." 
fi
