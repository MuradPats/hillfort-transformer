#!/bin/bash
# SBATCH script to extract ortofoto .zip files
#SBATCH -J transformers_extract_orto_job
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:10:00
#SBATCH --mem=4G

# Uncomment and set your email if you want notifications
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@here.com

module load python/3.12.3

SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORTO_DIR="$REPO_ROOT/data/orto"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

# Activate virtualenv

echo "Changing working directory to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Failed to cd to project root $PROJECT_ROOT"; exit 1; }
echo "Activating Python virtual environment"

if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
else
  echo "Virtual environment not found in $PROJECT_ROOT/venv. Please set up the venv first."
  exit 1
fi

# Check that orto directory exists
if [ ! -d "$ORTO_DIR" ]; then
  echo "Orto directory not found: $ORTO_DIR" >&2
  exit 1
fi

echo "Invoking Python extractor for .zip files in: $ORTO_DIR"
# Check that the extractor script exists
PY_EXTRACT="$REPO_ROOT/scripts/extract_orto.py"
if [ ! -f "$PY_EXTRACT" ]; then
  echo "Python extractor not found: $PY_EXTRACT" >&2
  exit 1
fi

# Run the Python extractor (uses stdlib zipfile). Use --verbose for progress.
python "$PY_EXTRACT" --path "$ORTO_DIR" --verbose --overwrite || {
  echo "Python extractor failed" >&2
  exit 1
}

echo "Extraction complete."
