#!/bin/bash
# Same option-bank CLI as the laptop. Supply account/partition/QOS externally.
# Usage: sbatch [flags] cluster/option_bank.sbatch /absolute/python /new/bank config
#SBATCH --job-name=sbi4finance-options
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time="00:10:00"
#SBATCH --output=%x-%j.out
set -euo pipefail
if (( $# != 3 )); then
    echo "Usage: $0 /absolute/python /new/bank config" >&2
    exit 2
fi
option_python=$1
option_bank=$2
option_config=$3
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$option_python" scripts/generate_option_bank.py --config "$option_config" --out "$option_bank"
