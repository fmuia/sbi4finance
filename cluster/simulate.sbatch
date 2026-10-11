#!/bin/bash
# CPU-only GBM bank smoke. Submit from the source checkout; choose your actual
# account/partition/QOS and output log with sbatch flags after discovery.
# Usage: sbatch [flags] cluster/simulate.sbatch /absolute/python /new/bank [config]
#SBATCH --job-name=sbi4finance-bank
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time="00:05:00"
#SBATCH --output=%x-%j.out

set -euo pipefail
if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 /absolute/python /new/bank [config]" >&2
    exit 2
fi
smoke_python=$1
smoke_bank=$2
smoke_config=${3:-configs/tasks/gbm_bank.json}
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$smoke_python" scripts/generate_gbm_bank.py --config "$smoke_config" --out "$smoke_bank"
"$smoke_python" scripts/check_gbm_bank.py --bank "$smoke_bank" > "$smoke_bank/verification.json"
cat "$smoke_bank/verification.json"
