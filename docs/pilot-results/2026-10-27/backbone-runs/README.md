# Tutorial backbone runs — 27 October 2026

Start with the [human-readable Heston comparison](../../../heston-backbone-comparison-2026-10-27.md) and [current work order](../../../tutorial-backbone-work-order.md). These runs deliver reusable banks and candidate estimators for Heston, known-curve rough Bergomi, then joint calibration. They do not yet establish accepted teaching checkpoints.

## Completed and running work

| Work | Evidence | State and resource scope |
| --- | --- | --- |
| Heston 50k/100k data | [Earlier bank checks](../tutorial-backbone/README.md) | Complete; preserved prefix and holdouts. **1.0114 allocated CPU-hours**, no GPU, separate from fitting. |
| New 50k fit | [Run](heston-50k/run.json), [validation](heston-50k/validation/evaluation.json) | Complete. 2k development cases pass 3/4 rank screens; initial variance fails. |
| Archived paired control | [Final control](archived-control/attempt-8029251/evaluation.json) | Complete on the same 2k observations: 4/4 rank screens. Original BF2.0.14/readout retained; 500 committed CPU cases plus 1,500 GPU cases. [Cancelled CPU attempt](archived-control/attempt-8028804/), [failed GPU setup](archived-control/attempt-8028941/) preserved. |
| New 100k fit | [Original run](heston-100k/run.json), [conditional reports](heston-100k/comparisons/) | Training/reload and three conditional checks complete; 1/3 screens pass. Original 2k evaluation hit its 900-second cap; its original failure is retained. |
| 100k evaluation and diagnosis | [Completed continuation](heston-100k/validation-continuation/evaluation.json), [readout audit](heston-100k/readout-diagnosis/README.md) | 2k panel completed: 3/4 rank screens pass; initial-variance KS improves to .05084 but misses .03562. Preprocessing passes; tighter readout has negligible effect. Keep the native default. |
| Known-curve 32k bank | [Full integrity check](known-bank-final/final-check.json), [final summary](known-bank-final/summary.json) | Complete: **20k / 5k / 7k**, all 1,280 shards without retries. **95.2475 allocated CPU-hours**, no GPU, below 150-hour ceiling. |
| Known-curve first fits | [Fit/coverage report](known-fit/README.md), [execution](known-fit/execution.json) | Complete: posterior, pricer and 2k development panel; 4/4 rank screens pass, with other coverage deviations retained. **1.12 CPU-hours / .28 GPU-hours**. |
| Known-curve PyMC | [Three references](known-emulator-references/README.md) | Complete; all parameter/rho×eta convergence checks pass. **788 CPU-seconds**, no GPU, under separate 6-hour ceiling. Approximate fixed-emulator targets only. |
| Known-curve comparisons | [Conditional report](known-emulator-comparison/README.md) | Complete; case 0 meets proposed rules, cases 1/2 fail. **336 CPU-seconds / 84 GPU-seconds**; shared fit/comparison pool remains below 3 CPU-hours / .75 GPU-hours. |
| Joint-bank recovery | [Completed recovery](joint-stop/recovery-378-380/README.md) | Complete: 397 checked parts / **9,925 training rows**, existing 394 parts unchanged. **1.1353 CPU-hours**, below 9.5-hour recovery bound and original 1,500-hour campaign. Remaining bank not generated. |

The completed Heston fit/control/continuation/diagnostic pool totals **17,660 allocated CPU-seconds / 3,860 GPU-seconds**, below **6 CPU-hours / 1.25 GPU-hours**. This includes the cancelled archived CPU attempt, failed archived GPU setup and failed original 100k evaluation. Known-curve fitting plus comparison totals **4,368 CPU-seconds / 1,092 GPU-seconds**; its separate PyMC unit adds **788 CPU-seconds**, no GPU. Joint cumulative allocation is **283.4164 CPU-hours**, including the old timeouts and successful recovery. Generation and fitting pools are distinct. Count top-level elapsed × allocated CPUs once; do not add batch/extern records again.

## Numerical and statistical scope

The known-curve bank reproduces the pilot task and its **80/504/504** grids. **One training row, `train:3999`,** has reported MCSE/quote-SD **0.3020122**, above nominal 0.25 but within the predeclared final-rung 0.5 cap. All three blocks used 2,097,152 paths. Retain its original ID and diagnostics; no redraw or replacement occurred. Validation/test rows all meet 0.25. Reported MCSE is empirical and does not certify grid bias or continuous-model prices.

The original Heston archive still outperforms the new models on the matched rank screen. The 100k fit improves initial-variance ranks relative to 50k but does not close the archive/conditional gap. Differences in data, update budgets and stochastic training prevent a data-only causal claim. The completed readout diagnosis supports leaving the native sampler unchanged. Keep point-estimate failures and reference uncertainty visible; three cases do not replace the original broader benchmark.

Known curves are held fixed within each three-target rough inference. Joint six-target inference excludes auxiliary generating curves. The implemented PyMC mean-emulator likelihood omits training-label Monte Carlo randomness/covariance. Its three conditional comparisons reveal substantial disagreements despite the passing global rank screen; the largest eta mean difference is **2.05 emulator-reference SD**. Neither agreement nor convergence alone proves accurate model-price posteriors. Independent posterior-region, refinement and product checks remain required.

Only development validation outcomes are used. Test-bank integrity checks inspect IDs, RNG reconstruction and numerical precision; they do not evaluate model predictions or select estimators. No participant notebooks were executed.

## Provenance and artifacts

Heston generation stays on its compatible frozen producer. Known-curve generation uses **6163520**, root **271300**; the checked snapshot manifest is `7cf0314dc6371fd045ad6b314dd6dd605db2f3343dbccf4e35b4df9ddf9397eb`. Its [preflight](known-preflight/preflight.json), [initial 100 rows](known-initial/summary.json), [initial gate](known-initial-check/initial-check.json), controller ledger and final check retain stream/source/runtime identities. All prior draws, curves and noisy quotes were reconstructed exactly; all initial 100 payload/manifest hashes were preserved. The retained stream audit reports no collisions within this bank or against the historical/Heston/joint inventories.

Known-curve fitting/reference sampling use frozen **845c458**, independently of the producer; the comparison uses **56dcb13**. Exact source bundles, deployment checks, commands, limits and identities are recorded with each unit. Reports and raw logs preserve failed attempts. Bulky metadata is losslessly compressed with raw-byte checksums. Model/bank/draw payloads remain on Oscar; small copies have checksum inventories.

The [work order](../../../tutorial-backbone-work-order.md) gives the remaining steps: one focused Heston density-model experiment on the existing bank; independent pricing at known-curve cases 0/2 and doubled grids; the rest of the joint bank; then frozen acceptance and notebook caches. Sparse-chain work stays downstream.
