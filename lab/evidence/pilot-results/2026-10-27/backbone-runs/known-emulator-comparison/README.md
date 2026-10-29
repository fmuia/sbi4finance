# Known-curve posterior versus approximate-emulator PyMC

All three comparisons completed successfully, but **cases 1 and 2 fail the proposed parameter-agreement checks**, with the largest discrepancy in case 2. These conditional differences remain alongside the [20k fit’s](../known-fit/README.md) passing global rank screen. They need diagnosis before tutorial claims about conditional agreement.

The cases were the fixed development observations `validation:0`, `validation:1` and `validation:2`, with their supplied curves held fixed. Each comparison used 10,000 neural draws, seeds `271500+i`, and the completed [PyMC emulator references](../known-emulator-references/README.md). All source references met their parameter and rho × eta chain-diagnostic screens. The proposed checks require absolute mean error at most 0.3 reference SD and a neural/reference 90% interval-width ratio between 0.75 and 1.25. They describe agreement with this approximate target; they are not scientific acceptance rules.

| Case | Target | Absolute mean error / reference SD | 90% width ratio | Proposed checks |
| --- | --- | ---: | ---: | --- |
| 0 | H | 0.207 | 1.003 | Met |
| 0 | eta | 0.190 | 1.004 | Met |
| 0 | rho | 0.186 | 1.003 | Met |
| 1 | H | 0.020 | 1.356 | Width failed |
| 1 | eta | 0.289 | 1.141 | Met |
| 1 | rho | 0.445 | 1.375 | Mean and width failed |
| 2 | H | 0.898 | 1.584 | Mean and width failed |
| 2 | eta | 2.052 | 1.676 | Mean and width failed |
| 2 | rho | 1.079 | 1.403 | Mean and width failed |

For the derived rho × eta target, the corresponding `(mean error / reference SD, width ratio)` values were `(0.072, 0.962)`, `(0.601, 1.392)` and `(0.736, 1.514)`. These are descriptive comparisons without a registered derived-target acceptance threshold. The full [case 0](case-0/comparison.json), [case 1](case-1/comparison.json) and [case 2](case-2/comparison.json) reports preserve the reference diagnostics, uncertainty summaries and every failed check. Reference quantile MCSE remains unassessed.

The PyMC target is a deterministic forward-pricer emulator plus independent Gaussian quote noise. The amortized estimator trains on stochastic Monte Carlo bank prices plus fresh quote noise. Emulator bias, pricing-grid error and bank pricing randomness/covariance are not propagated by this reference. Consequently, the discrepancies cannot be assigned solely to the amortized estimator; agreement with this emulator would also leave independent model-price accuracy and like-for-like simulator-posterior agreement unestablished. No new fits, financial pricing, MCMC or reserved-test evaluations occurred in this comparison unit.

The [plan](plan.json) froze source `56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9`, the estimator checkpoint, pricer, observations, curves and reference inputs before submission. [Preflight](preflight.json) and the terminal [source check](source-freeze.json) verified their hashes, the clean source revision and unchanged locked environment. The estimator checkpoint SHA is `9d6cbd534a49bd0862316c9e38a02ed4e9da818986951d1b020162fa0ef9573e`.

Job `8030954` used one ten-minute L40S allocation with four CPUs, 24 GiB memory, 180-second per-case caps and a 570-second driver deadline. It completed `0:0` after **84 seconds**, costing **336 allocated CPU-seconds and 84 allocated GPU-seconds**. Together with the preceding fit, the shared pool used **1.21333 CPU-hours and 0.30333 GPU-hours**, within its 3-CPU-hour / 0.75-GPU-hour ceiling. The CPU references’ 788 allocated CPU-seconds remain in their separate pool. [Accounting](accounting.psv), [execution.json](execution.json) and [summary.json](summary.json) retain these totals without counting batch/extern records twice. No retries or additional allocations were submitted.

The neural draw payloads remain on Oscar at `/oscar/scratch/afengler/sbi4finance-rough-known-reference-20261027/runs/comparisons/case-{0,1,2}`. [artifact-inventory.json](artifact-inventory.json) records their sizes and SHA-256 hashes. This archive contains the small reports, logs and one-off scripts; [downloaded-artifacts.json](downloaded-artifacts.json) and [downloaded-files.sha256](downloaded-files.sha256) verify the retained evidence.
