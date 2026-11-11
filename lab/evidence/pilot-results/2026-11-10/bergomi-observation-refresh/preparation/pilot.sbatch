#!/bin/bash
#SBATCH --job-name=rough-actual-R1
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --array=0-47%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
set -euo pipefail
: "${ROUGH_REPO_ROOT:?stage full lab and sibling tutorial checkout first}"
: "${ROUGH_PYTHON:?set verified execution-host Python}"
: "${ROUGH_WORK_ORDER:?set frozen work-order.json}"
: "${ROUGH_CONDITION:?bind completed Heston result first}"
: "${ROUGH_HESTON_EVALUATION:?stage exact hash-bound Heston evaluation JSON}"
: "${ROUGH_PAYLOAD_ROOT:?set NEW output root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/run_rough_actual_layout_pilot.py" --work-order "$ROUGH_WORK_ORDER" --condition "$ROUGH_CONDITION" --heston-evaluation "$ROUGH_HESTON_EVALUATION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT/component-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
