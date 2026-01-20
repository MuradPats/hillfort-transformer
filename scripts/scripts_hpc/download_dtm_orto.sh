#!/bin/bash
# SBATCH script to download only 1:10000 DTM and ortofoto (no GPU)
#SBATCH -J transformers_download_dtm_orto_job
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH --mem=4G

# Uncomment and set your email if you want notifications
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@here.com

module load python/3.12.3

# Resolve script directory and cd to repository root so relative paths are stable
SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_DIR/.." && pwd)"

echo "Changing working directory to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Failed to cd to project root $PROJECT_ROOT"; exit 1; }
echo "Activating Python virtual environment"

# Activate virtualenv: check repo root venv then parent-level venv (common when repo is inside 'transformers/')
if [ -f "$REPO_DIR/venv/bin/activate" ]; then
  source "$REPO_DIR/venv/bin/activate"
elif [ -f "$REPO_DIR/../venv/bin/activate" ]; then
  source "$REPO_DIR/../venv/bin/activate"
else
  echo "Virtual environment not found in $REPO_DIR/venv or ../venv. Please set up the venv first."
  exit 1
fi

echo "Changing working directory to repository root: $REPO_DIR"
cd "$REPO_DIR" || { echo "Failed to cd to repo root $REPO_DIR"; exit 1; }
echo "Starting map download (DTM and ortofoto only)..."

# Run the downloader CLI using absolute repo-root paths (disables LAZ and reljeef)
python "$REPO_DIR/data/raw/download_maps.py" "$REPO_DIR/data/linnamagede_ruudunumbrid_v2.csv" \
  --laz None --dtm "$REPO_DIR/data/dtm/" --reljeef None --orto "$REPO_DIR/data/orto/" --sleep 1
