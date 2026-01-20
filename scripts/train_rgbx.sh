#!/bin/bash
# SBATCH script to run RGBX training on 1 GPU
#SBATCH -J transformers_train_rgbx
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

# Uncomment and set your email if you want notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sander.saska@ut.ee

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Resolve paths
SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
TRAIN_PY="$REPO_ROOT/RGBX/train.py"

if [ ! -f "$TRAIN_PY" ]; then
  echo "Train script not found: $TRAIN_PY" >&2
  exit 1
fi

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

# Change to repo root for stable relative paths
cd "$REPO_ROOT" || { echo "Failed to cd to repo root $REPO_ROOT"; exit 1; }

# Default training args if none provided: use GPU device 0
if [ "$#" -eq 0 ]; then
  ARGS=("-d" "0")
else
  ARGS=("$@")
fi

echo "Starting training: python $TRAIN_PY ${ARGS[*]}"
python "$TRAIN_PY" "${ARGS[@]}"

echo "Training finished."