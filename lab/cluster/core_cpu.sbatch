#!/bin/bash
# Submit from the immutable checkout. Supply the absolute Pixi executable.
# sbatch [resource overrides] cluster/core_cpu.sbatch /absolute/pixi core COMMAND [ARGS...]
# Select core-reference for PyMC jobs. No archived/default environment is used.
# Choose the account and partition with the caller's sbatch flags; neither is fixed here.
#SBATCH --job-name=sbi4finance-core-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 /absolute/pixi core|core-reference COMMAND [ARGS...]" >&2
    exit 2
fi
PIXI_EXECUTABLE=$1
CORE_ENVIRONMENT=$2
shift 2
if [[ "$PIXI_EXECUTABLE" != /* || ! -x "$PIXI_EXECUTABLE" ]]; then
    echo "An executable absolute Pixi path is required" >&2
    exit 2
fi
case "$CORE_ENVIRONMENT" in
    core|core-reference) ;;
    *) echo "Only core or core-reference is supported by this CPU launcher" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
cd "${SLURM_SUBMIT_DIR:?Submit from the immutable checkout}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" KERAS_BACKEND=jax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PIXI_EXECUTABLE" run --locked -e "$CORE_ENVIRONMENT" "$@"
