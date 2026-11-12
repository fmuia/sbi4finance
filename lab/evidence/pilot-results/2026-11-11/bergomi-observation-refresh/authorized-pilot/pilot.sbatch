#!/bin/bash
#SBATCH --job-name=rough-actual-A1
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --array=0-47%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
set -euo pipefail
: "${ROUGH_REPO_ROOT:?stage frozen complete source checkout}"
: "${ROUGH_PYTHON:?set verified execution-host Python}"
: "${ROUGH_WORK_ORDER:?set unchanged portable work-order.json}"
: "${ROUGH_AUTHORIZATION:?set explicit execution authorization.json}"
: "${ROUGH_PAYLOAD_ROOT:?set NEW output root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rough-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/run_rough_actual_layout_authorized.py" run --work-order "$ROUGH_WORK_ORDER" --authorization "$ROUGH_AUTHORIZATION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT/component-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
