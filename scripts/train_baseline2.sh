#!/bin/bash
#SBATCH -J baseline2_train
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 08:00:00
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

SCRIPT_DIR="/gpfs/helios/home/sandersa/transformers/hillfort-transformer/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/venv/bin/activate"

cd "$REPO_ROOT"

export DEBUG_VRAM=1

python -u -m baseline2.train \
  --dataset-root "$REPO_ROOT/datasets/HillfortDataSet" \
  --batch-size 2 \
  --epochs 10 \
  --num-workers 0 \
  --save-dir "$REPO_ROOT/runs/baseline2"
