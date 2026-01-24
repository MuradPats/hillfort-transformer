#!/bin/bash
#SBATCH -J baseline2_eval
#SBATCH --output=/gpfs/helios/home/sandersa/transformers/slurm_outputs/slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH --mem=5G
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

python "$REPO_ROOT/RGBX/eval_baseline2.py" \
  --ckpt "$REPO_ROOT/runs/baseline2/ckpt_epoch_10.pt" \
  -d 0 \
  --save_path "$REPO_ROOT/runs/baseline2/eval"