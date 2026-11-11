#!/bin/bash
#SBATCH --job-name=rough-main-jc128
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --array=0-127%128
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rough-bank-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/rough_actual_layout_bank.py" worker --plan "$ROUGH_BANK_PLAN" --admission "$ROUGH_BANK_ADMISSION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT/shard-$(printf '%06d' "$SLURM_ARRAY_TASK_ID")"
