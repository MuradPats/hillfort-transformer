#!/bin/bash
# SBATCH script to run the polygon rasterisation script (create GT masks)
#SBATCH -J transformers_rasterize_polygons_masks
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 02:00:00
#SBATCH --mem=2G

##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=sander.saska@ut.ee

set -euo pipefail

module load python/3.12.3

# Resolve locations (adjust these to your environment if needed)
SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

cd "$REPO_ROOT" || { echo "Failed to cd to repo root $REPO_ROOT"; exit 1; }

echo "Activating virtualenv (expected at $PROJECT_ROOT/venv)"
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Virtual environment not found in $PROJECT_ROOT/venv. Please set up the venv first." >&2
    exit 1
fi

# Paths for the rasterisation script
ORTHO_DIR="$REPO_ROOT/data/orto"
CSV_PATH="$REPO_ROOT/data/linnamagede_ruudunumbrid_v2.csv"
SHAPEFILE="$REPO_ROOT/data/raw/inspire/PS_ProtectedSite_malestisedPolygon.shp"
OUT_DIR="$REPO_ROOT/data/gt_masks"
MASK_SIZE=5000

echo "Running rasterisation: ortho=$ORTHO_DIR out=$OUT_DIR mask_size=$MASK_SIZE"

# Run the python script via srun (keeps job resource accounting)
python "$SCRIPT_DIR/rasterize_polygons_to_masks.py" \
    --ortho-dir "$ORTHO_DIR" \
    --csv "$CSV_PATH" \
    --shapefile "$SHAPEFILE" \
    --out-dir "$OUT_DIR" \
    --mask-size "$MASK_SIZE"

echo "Rasterisation job finished. Masks saved to $OUT_DIR"
