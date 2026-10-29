# Known-curve rough Bergomi bank: complete

The frozen bank contains **20,000 training, 5,000 validation and 7,000 reserved test surfaces**, in 1,280 parts of 25 rows. Each case has a supplied seven-level forward-variance curve, which varies across the bank; inference targets are H, eta and rho. Its task fingerprint matches Francesco’s original completed pilot. All original worker indices 0–1279 completed successfully, with no retries, failed tasks, seed replacements or omitted ranges.

The data remain on Oscar at `/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-20k`. The uncompressed [snapshot manifest](snapshot-manifest.json.gz) has SHA-256 `7cf0314dc6371fd045ad6b314dd6dd605db2f3343dbccf4e35b4df9ddf9397eb`. Generation used root seed `271300`, frozen producer `616352035be4b9d271151090aa5d2d9539da8b1c`, and the original `80/504/504` numerical grids.

Total allocation was **95.2475 CPU-hours**, including preflight and integrity checks, against the 150-hour ceiling. Generation itself used 95.2186 CPU-hours. No GPU was used by this data-generation unit. [Raw accounting](accounting-all-jobs.psv.gz) retains all job and step records; [top-level accounting](accounting-top-level.psv) supports the total without counting batch/extern steps twice.

The [final integrity check](final-check.json) verified all 32,000 prior draws, supplied curves and quote-noise streams exactly, preserved the initial 100 payload and manifest hashes, and checked the frozen snapshot’s coverage, hashes and numerical metadata. One case, `train:3999`, reached the final 2,097,152-path rung and had maximum empirical MCSE / quote sigma `0.3020122`, above the nominal 0.25 target and within the declared 0.5 cap. It is retained without redraw. There were no hard-cap exceptions; validation and test had no nominal exceptions.

This is the matched-task bank for developing and comparing the estimator at Francesco’s original numerical resolution. Empirical MCSE checks do not establish discretization accuracy. Bounded posterior and product comparisons under numerical refinement remain necessary before final tutorial claims. Test use here was limited to generation and numerical/ID/RNG integrity; this unit performed no neural training or inference evaluation.

The archive contains metadata and logs, not bank payloads:

- [summary.json](summary.json) and [execution.json](execution.json) record the final outcome and submission ledger.
- [worker-logs.json.gz](worker-logs.json.gz) preserves all 1,280 worker logs, including per-log byte counts and hashes; [the final Slurm log](slurm-8030755.out) records snapshot completion.
- [controller-script.py](controller-script.py), [submission-operator.py](submission-operator.py) and [final-integrity-script.py](final-integrity-script.py) preserve the one-off orchestration and check scripts. The journal and `split-progress.json` are timestamped historical observations; `controller-state.json` and the final integrity report establish final completion.
- [compressed-artifacts.json](compressed-artifacts.json) records original and stored hashes for lossless gzip archives. [downloaded-files.sha256](downloaded-files.sha256) verifies the retained files.

The preceding [preflight](../known-preflight/), [initial four parts](../known-initial/), [initial integrity check](../known-initial-check/) and [first 192-part wave](../known-wave-4-195/) retain the staged validation evidence. This known-curve bank has its own budget and ledger, separate from the joint-calibration bank.
