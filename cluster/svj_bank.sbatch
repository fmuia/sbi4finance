#!/bin/bash
# CPU bank smoke using exactly the laptop CLI; select account/partition/QOS externally.
# Usage: sbatch [flags] cluster/svj_bank.sbatch /absolute/python /new/bank [config]
#SBATCH --job-name=sbi4finance-svj-bank
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
svj_python=$1
svj_bank=$2
svj_config=${3:-configs/tasks/svj_daily_smoke.json}
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$svj_python" scripts/generate_svj_bank.py --config "$svj_config" --out "$svj_bank"
"$svj_python" scripts/check_svj_bank.py --bank "$svj_bank" > "$svj_bank/verification.json"
cat "$svj_bank/verification.json"
