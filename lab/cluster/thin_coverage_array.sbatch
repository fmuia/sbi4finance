#!/bin/bash
# Study S10 as an array: task i runs test rows [FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK).
#
#   sbatch --array=0-9 [--account=A --partition=P] cluster/thin_coverage_array.sbatch BANK S2T_RUN OUT FIRST CHUNK
#
# Each task appends finished chains to OUT/rows-START-STOP.jsonl, so a requeued task continues where it stopped.
# With CHUNK 100 and 16 cores a task takes about an hour (about 7 CPU-minutes a chain; CLUSTER-RUNS.md).
# Afterwards: python studies/thin_coverage.py summarize --out OUT
#SBATCH --job-name=sbi4finance-s10
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time="04:00:00"
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
if (( $# != 5 )); then
    echo "Usage: $0 BANK S2T_RUN OUT FIRST CHUNK" >&2
    exit 2
fi
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:?Submit from lab/ of the run checkout}"
export PYTHONPATH="$PWD/src" KERAS_BACKEND=jax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
start=$(( $4 + ${SLURM_ARRAY_TASK_ID:?Array task required} * $5 ))
pixi run python studies/thin_coverage.py run --bank "$1" --checkpoint "$2" --out "$3" \
    --start "$start" --stop "$(( start + $5 ))" --workers "${SLURM_CPUS_PER_TASK:-16}"
