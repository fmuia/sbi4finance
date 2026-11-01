#!/bin/bash
# Use an existing verified locked runtime with an immutable source checkout.
# sbatch [resource overrides] cluster/path_job.sbatch /absolute/python cpu|gpu SCRIPT [ARGS...]
# GPU jobs require --partition=gpu-he --gres=gpu:1 and the core-gpu Python.
#SBATCH --job-name=sbi4finance-path
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 /absolute/python cpu|gpu SCRIPT [ARGS...]" >&2
    exit 2
fi
PATH_PYTHON=$1
PATH_DEVICE=$2
shift 2
if [[ "$PATH_PYTHON" != /* || ! -x "$PATH_PYTHON" ]]; then
    echo "An executable absolute runtime Python path is required" >&2
    exit 2
fi
case "$PATH_DEVICE" in
    cpu|gpu) ;;
    *) echo "Device must be cpu or gpu" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
cd "${SBI4FINANCE_SOURCE_LAB:-${SLURM_SUBMIT_DIR:?Submit from the immutable lab checkout}}"
test -d src/sbi4finance/path_inference
test -d tests
export PYTHONPATH="$PWD/src" KERAS_BACKEND=jax PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/path-numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/path-mpl-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
if [[ "$PATH_DEVICE" == gpu ]]; then
    nvidia-smi -L
    "$PATH_PYTHON" scripts/check_full_surface_environment.py --require-gpu
else
    export JAX_PLATFORMS=cpu
fi
"$PATH_PYTHON" -c 'import json; from importlib.metadata import version; print(json.dumps({"runtime_versions": {k: version(k) for k in ("bayesflow", "keras", "jax", "numpy", "scipy", "numba")}})); assert version("bayesflow") == "2.1.0"'
exec "$PATH_PYTHON" "$@"
