#!/bin/bash
# Submit from the immutable checkout. Supply the absolute Pixi executable.
# sbatch [resource overrides] cluster/core_gpu.sbatch /absolute/pixi COMMAND [ARGS...]
# Example command: python scripts/check_core_cluster.py --require-gpu --out SHARED_NEW_DIR
#SBATCH --job-name=sbi4finance-core-gpu
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /absolute/pixi COMMAND [ARGS...]" >&2
    exit 2
fi
PIXI_EXECUTABLE=$1
shift
if [[ "$PIXI_EXECUTABLE" != /* || ! -x "$PIXI_EXECUTABLE" ]]; then
    echo "An executable absolute Pixi path is required" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Run inside a Slurm GPU allocation}"
cd "${SLURM_SUBMIT_DIR:?Submit from the immutable checkout}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" KERAS_BACKEND=jax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi -L
"$PIXI_EXECUTABLE" run --locked -e core-gpu core-check --require-gpu
exec "$PIXI_EXECUTABLE" run --locked -e core-gpu "$@"
