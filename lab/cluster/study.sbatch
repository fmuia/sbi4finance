#!/bin/bash
# Run one study command on one CPU node (the audit S5, SBC S2 and S2-T, S1-G, S11, assembly, ...).
#
#   sbatch [--account=A --partition=P --cpus-per-task=N --time=T] cluster/study.sbatch python SCRIPT ARGS...
#
# The command runs in the pixi default environment, from the run checkout. Give the study --workers equal to
# --cpus-per-task. tutorial_final/CLUSTER-RUNS.md gives each command with its cores and time.
#SBATCH --job-name=sbi4finance-study
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time="12:00:00"
#SBATCH --output=%x-%j.out

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:?Submit from the run checkout}"
export PYTHONPATH="$PWD/src" KERAS_BACKEND=jax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pixi run "$@"
