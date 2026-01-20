#!/bin/bash
# SBATCH script to test PyTorch and GPU availability
#SBATCH -J transformers_test_torch_gpu
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:10:00
#SBATCH --mem=2G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

# Uncomment and set your email if you want notifications
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your_email@here.com

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PY_TEST="$REPO_ROOT/scripts/test_torch_gpu.py"

if [ ! -f "$PY_TEST" ]; then
  echo "Test script not found: $PY_TEST" >&2
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

# Show GPU status
echo "----- nvidia-smi -----"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not available"
fi

echo "----- Python torch check -----"
python "$PY_TEST" --verbose

echo "Test complete."