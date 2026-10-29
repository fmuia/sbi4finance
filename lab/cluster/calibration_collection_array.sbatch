#!/usr/bin/env bash
# Submit an inclusive index range from the exact job count printed by `plan`.
# Optional offset splits a large map across the cluster's MaxArraySize limit.
#SBATCH --job-name=sbi4finance-collection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
if (( $# != 2 && $# != 3 )); then
    echo "Usage: $0 /absolute/python job-map.json [offset]" >&2
    exit 2
fi
collection_python=$1
collection_jobs=$2
collection_offset=${3:-0}
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen source checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$collection_python" scripts/calibration_bank_collection.py shard \
    --job-map "$collection_jobs" \
    --task-index "$(( collection_offset + ${SLURM_ARRAY_TASK_ID:?Array task required} ))"
