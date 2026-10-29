# Known-curve rough Bergomi: first 20k fit

Job `8030774` completed successfully, fitting a posterior and forward pricer on the [20k known-curve bank](../known-bank-final/README.md). The curve is supplied for each inference case and varies across training. The posterior infers H, eta and rho from complete option quotes and their supplied curve; rho × eta is an additional diagnostic target. These are development results at the original `80/504/504` pricing grids, with no reserved-test evaluation or tutorial acceptance claim.

The posterior uses BayesFlow 2.1.0 flow matching, a vector encoder and four hidden layers of width 256. Posterior sampling uses BayesFlow’s native adaptive Tsit5 solver; this solver is separate from the financial simulator’s pricing grids. Epoch 292 of 300 was selected by validation loss. Same-device, same-seed checkpoint reload was exact. The forward pricer selected epoch 294 of 300, also with four hidden layers of width 256.

On the declared 2,000 development-validation cases, with 1,000 posterior draws per case, all four rank KS statistics were below the simultaneous DKW band **0.0356201**, at family alpha 0.05. The [evaluation report](validation/evaluation.json) retains all results and pointwise Wilson intervals.

| Target | Rank KS | 50% coverage | 80% coverage | 90% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| H | 0.0232581 | 49.50% | 79.95% | 89.85% | 94.70% |
| eta | 0.0189171 | 52.45% | 80.25% | 90.10% | 94.85% |
| rho | 0.0247174 | 47.65% | 79.40% | 90.20% | 94.85% |
| rho × eta | 0.0221336 | 49.55% | 79.45% | 89.75% | 94.35% |

The 50% eta and rho coverage intervals exclude their nominal level under the report’s pointwise Wilson calculation. These intervals are unadjusted for the multiple targets and levels; the deviations remain visible for review. The validation split also selected the training epoch, so this screen is not an independent acceptance test. Marginal calibration does not establish conditional posterior agreement or product accuracy.

The [forward-pricer report](pricer/checkpoint/pricer.json) separates the selected **Keras validation MSE, 0.0152087072** in squared quote-SD units, from the saved **NumPy pricer’s numerical-label RMSE, 0.123268124** quote SD. Its absolute quote-error p99 was 0.418128 and maximum 3.570962. Surface L2 error p99 was 2.310544 and maximum 10.861258, so small average per-quote errors alone do not establish posterior accuracy.

**Keras-to-NumPy export parity is not established:** the maximum validation prediction difference was **0.0498446350 quote SD**. Reload of the saved NumPy pricer itself was exact. The exact-erf activation removes the old activation mismatch, but does not guarantee identical GPU and NumPy algebra. GPU arithmetic or precision is a possible contribution to this remaining difference, not a confirmed explanation.

The posterior’s simulator includes Monte Carlo pricing-label randomness and quote noise. A deterministic emulator learns an approximation to numerical prices; a Gaussian likelihood with only the declared quote covariance omits the pricing-label randomness and its cross-contract covariance. Label-comparison errors also mix emulator error and label randomness. Consequently, an emulator-based MCMC calculation would target an approximation; a like-for-like claim requires matching or justifying that target, as well as checking numerical refinement and posterior regions. This unit ran no new financial pricing or MCMC.

The allocation used **1,008 seconds with four CPUs and one L40S: 1.12 allocated CPU-hours and 0.28 allocated GPU-hours**. The 834.18-second validation stage stayed within its prospective 1,050-second cap; the whole run stayed within its 40-minute allocation and separate 3-CPU-hour / 0.75-GPU-hour ceiling. [Accounting](accounting.psv) retains job and step records without counting batch/extern costs twice.

Source `845c458b849c472271527cccf815ea6714f02206`, driver SHA `e19aa4f1f8011ebe9249113dc2058b0cee0e2aa9a530f39307a375045ff96945` and bank manifest SHA `7cf0314dc6371fd045ad6b314dd6dd605db2f3343dbccf4e35b4df9ddf9397eb` bind the run. [source-freeze.json](source-freeze.json) verifies unchanged source, environment, driver and input bank. The actual models and draws remain at `/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027/runs/known-20k-native`; [artifact-inventory.json](artifact-inventory.json) records their SHA-256 hashes and sizes without transferring payloads. [execution.json](execution.json), [run.json](run.json), [summary.json](summary.json), the compressed logs and [downloaded-files.sha256](downloaded-files.sha256) preserve the execution evidence.
