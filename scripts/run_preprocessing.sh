#!/bin/bash
# SBATCH script to run the full preprocessing pipeline (4 scripts)
#SBATCH -J transformers_data_preprocessing_job
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 02:00:00
#SBATCH --mem=3G

# Uncomment and set your email if you want notifications
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=sander.saska@ut.ee

module load python/3.12.3

# Flags to control which steps to run (1=run, 0=skip)
RUN_BATCH=0
RUN_RASTERIZE=0
RUN_TILE=0
RUN_STRATIFY=1

# Resolve locations
SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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

# Paths to preprocessing scripts (notebooks/scripts)
SCRIPT_BASE="$REPO_ROOT/notebooks/scripts"

if [ "$RUN_BATCH" -eq 1 ]; then
  echo "Running: batch_convert.py"
  python "$SCRIPT_BASE/batch_convert.py" \
    --dtm_dir "$REPO_ROOT/data/dtm" \
    --sat_dir "$REPO_ROOT/data/orto" \
    --out "$REPO_ROOT/datasets/HillfortDataSet" \
    --method minmax --write_test
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "batch_convert.py failed (exit $rc)" >&2
    exit $rc
  fi
  echo "batch_convert.py completed successfully."
fi

if [ "$RUN_RASTERIZE" -eq 1 ]; then
  echo "Running: rasterize_polygons.py"
  python "$SCRIPT_BASE/rasterize_polygons.py" \
    --geom "$REPO_ROOT/data/raw/inspire/PS_ProtectedSite_malestisedPolygon.shp" \
    --ref_dir "$REPO_ROOT/data/dtm" \
    --map_numbers "$REPO_ROOT/data/linnamagede_ruudunumbrid_v2.csv" \
    --out "$REPO_ROOT/datasets/HillfortDataSet/Label"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "rasterize_polygons.py failed (exit $rc)" >&2
    exit $rc
  fi
  echo "rasterize_polygons.py completed successfully."
fi

if [ "$RUN_TILE" -eq 1 ]; then
  echo "Running: tile_dataset.py"
  python "$SCRIPT_BASE/tile_dataset.py" \
    --root "$REPO_ROOT/datasets/HillfortDataSet" \
    --out "$REPO_ROOT/datasets/HillfortDataSet" \
    --tile-size 512 --save-tiles --csv "$REPO_ROOT/datasets/HillfortDataSet/tile_stats.csv"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "tile_dataset.py failed (exit $rc)" >&2
    exit $rc
  fi
  echo "tile_dataset.py completed successfully."
fi

if [ "$RUN_STRATIFY" -eq 1 ]; then
  echo "Running: build_stratified_train.py"
  python "$SCRIPT_BASE/build_stratified_train.py" \
    --csv "$REPO_ROOT/datasets/HillfortDataSet/tile_stats.csv" \
    --out "$REPO_ROOT/datasets/HillfortDataSet/" \
    --small-thresh 0.01647 \
    --mid-thresh 0.05634 \
    --proportions "0.25,0.25,0.25,0.25" \
    --mix-size 10000 \
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "build_stratified_train.py failed (exit $rc)" >&2
    exit $rc
  fi
  echo "build_stratified_train.py completed successfully."
fi
echo "Preprocessing pipeline completed successfully."