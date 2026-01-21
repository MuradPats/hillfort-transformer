#!/bin/bash
# SBATCH script to run baseline evaluation on 1 GPU
#SBATCH -J transformers_eval_baseline
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 04:00:00
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

# Uncomment and set your email if you want notifications
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=sander.saska@ut.ee

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Resolve paths
SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EVAL_PY="$REPO_ROOT/baseline/eval.py"

if [ ! -f "$EVAL_PY" ]; then
  echo "Eval script not found: $EVAL_PY" >&2
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

# Ensure repo-root imports work (RGBX, baseline, shims)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Default evaluation args if none provided:
# - device 0
# - save predictions under repo_root/eval_baseline (adjust if your eval.py uses different flag names)
if [ "$#" -eq 0 ]; then
  ARGS=("-d" "0" "-p" "$REPO_ROOT/eval_baseline")
else
  ARGS=("$@")
fi

echo "Starting baseline evaluation: python $EVAL_PY ${ARGS[*]}"
python "$EVAL_PY" "${ARGS[@]}"

echo "Baseline evaluation finished."
