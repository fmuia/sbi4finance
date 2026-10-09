#!/bin/bash
# Train the univariate MMAR approximator on one Oscar GPU.
#
#   sbatch cluster/train.sbatch                               # defaults below
#   sbatch --account=A --partition=P cluster/train.sbatch     # sbatch flags override #SBATCH lines
#   sbatch cluster/train.sbatch --n-train 20000 --epochs 10   # args go to the training script
#
# Account and partition: lab-wide defaults from cluster/discover_cluster.py; check
# cluster/oscar.local.json for the lanes you hold. Wall time is a quoted string on
# purpose: unquoted 04:00:00 would be read as 14400 minutes by YAML-based tooling.
#SBATCH --job-name=sbi4finance-train
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time="04:00:00"
#SBATCH --output=%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"   # pixi's installer puts it here; batch shells skip the rc file
cd "$SLURM_SUBMIT_DIR"                # submit from the repo root
nvidia-smi -L
pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
pixi run -e gpu python scripts/train_univariate.py "$@"
