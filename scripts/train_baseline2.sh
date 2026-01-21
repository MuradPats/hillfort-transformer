#!/bin/bash
#SBATCH -J baseline2_train
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 04:00:00
#SBATCH --mem=8G
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

python -m baseline2.train \
  --dataset-root "$REPO_ROOT/datasets/HillfortDataSet" \
  --batch-size 8 \
  --epochs 1 \
  --steps-per-epoch 5 \
  --save-dir "$REPO_ROOT/runs/baseline2"
