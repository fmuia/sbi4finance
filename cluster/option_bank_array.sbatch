#!/bin/bash
# Supply account/partition/QOS and --array=0-N%concurrency externally.
# Usage: sbatch [flags] cluster/option_bank_array.sbatch /python jobs.json /new/shards
# Each task uses one fixed range from jobs.json; all outputs go to supplied scratch.
#SBATCH --job-name=sbi4finance-option-chunks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time="00:45:00"
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
if (( $# != 3 )); then
    echo "Usage: $0 /absolute/python job-map.json /scratch/shards" >&2
    exit 2
fi
option_python=$1
option_job_map=$2
option_output_root=$3
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$option_python" scripts/generate_option_bank_shard.py \
    --job-map "$option_job_map" --task-index "${SLURM_ARRAY_TASK_ID:?Array task required}" \
    --output-root "$option_output_root"
