#!/bin/bash
# Train a calibration network on one Oscar GPU: the liquid network (S2) or the thin-chain network (S2-T).
#
#   sbatch [--account=A --partition=P] cluster/train_calibration.sbatch --bank BANK --out RUN [options]
#
# Arguments go to studies/train_calibration.py (plans/CLUSTER-RUNS.md gives the exact
# ones). Submit from lab/ of the run checkout, the commit the bank was built from. Loading the full bank's train split
# needs about 13 GB, hence 96 GB. A CPU node also works: the Heston network (50,000 surfaces, 500 epochs) trained
# in 32 minutes on 4 cores.
#SBATCH --job-name=sbi4finance-train-calibration
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time="12:00:00"
#SBATCH --output=%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"   # pixi's installer puts it here; batch shells skip the rc file
cd "${SLURM_SUBMIT_DIR:?Submit from lab/ of the run checkout}"
export KERAS_BACKEND=jax
nvidia-smi -L
pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
pixi run -e gpu python studies/train_calibration.py "$@"
