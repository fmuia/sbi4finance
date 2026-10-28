#!/bin/bash
# Supply account/partition/QOS and --array=0-N%concurrency externally.
# Usage: sbatch [flags] cluster/calibration_bank_array.sbatch /python jobs.json /new/shards [OFFSET]
# jobs.json comes from `scripts/calibration_bank.py job-map`; each array task writes one shard, job
# OFFSET + SLURM_ARRAY_TASK_ID (OFFSET defaults to 0; use it when MaxArraySize is below the job count).
# A rerun or requeue of a task keeps a finished shard and redoes a partial one.
# Assemble afterwards with `scripts/calibration_bank.py assemble` in a job with at least 16 GB.
#SBATCH --job-name=sbi4finance-calibration-chunks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time="02:00:00"
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
if (( $# != 3 && $# != 4 )); then
    echo "Usage: $0 /absolute/python job-map.json /scratch/shards [offset]" >&2
    exit 2
fi
calibration_python=$1
calibration_job_map=$2
calibration_output_root=$3
calibration_offset=${4:-0}
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$calibration_python" scripts/calibration_bank.py shard \
    --job-map "$calibration_job_map" --task-index "$(( calibration_offset + ${SLURM_ARRAY_TASK_ID:?Array task required} ))" \
    --output-root "$calibration_output_root"
