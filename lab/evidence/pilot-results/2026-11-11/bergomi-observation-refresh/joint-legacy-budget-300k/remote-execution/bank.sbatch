#!/bin/bash
#SBATCH --job-name=rough-main-jc-full
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab:$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rough-bank-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/rough_actual_layout_overnight_bank.py" allocation --plan "$ROUGH_BANK_PLAN" --admission "$ROUGH_BANK_ADMISSION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT"
