# Independent pricing checks for known-curve rough Bergomi

This unit diagnoses the existing neural-versus-emulator discrepancy at the fixed development observations `validation:0` and `validation:2`. It does not fit or correct a posterior, retrain a network, expand a bank, inspect reserved test outcomes, or establish tutorial acceptance.

## Completed results

All **48 pricing batches completed**, with unchanged inputs. Errors below are in units of the declared quote-noise standard deviation. RMS and maximum entries are **point estimates across the 63 quotes**; per-quote uncertainty is retained in the [full summary](summary.json), its [compressed copy](summary.json.gz) and [worst-quote intervals](worst-quote-intervals.json).

| Case / parameter point | Emulator RMS, bank / doubled grid | Emulator maximum absolute error, bank / doubled grid | Doubled-minus-bank grid RMS / maximum |
| --- | ---: | ---: | ---: |
| 0 / generating parameter | .0817 / .0894 | .2174 / .2452 | .0334 / .0962 |
| 0 / neural posterior mean | .0763 / .0864 | .2108 / .2392 | .0346 / .0991 |
| 0 / PyMC emulator posterior mean | .0772 / .0866 | .2119 / .2392 | .0341 / .0976 |
| 2 / generating parameter | .2536 / .2383 | .7424 / .6455 | .0968 / .2746 |
| 2 / neural posterior mean | .1658 / .2111 | .4239 / .5573 | .0977 / .2760 |
| 2 / PyMC emulator posterior mean | .1946 / .2098 | .5727 / .5072 | .0953 / .2669 |

The coarse-grid emulator error at case 2's generating parameter is **−.7424 sigma**, with pointwise t7 interval **[−.8110, −.6738]**, on the three-month call with strike **1.2214** in spot-one units (contract 44). Its quote sigma is .00029797. The largest grid difference there is **−.2746 sigma**, interval **[−.2926, −.2566]**, on the five-trading-day call with strike 1.05796 (contract 8). These worst quotes are selected descriptively after pricing; their intervals are pointwise, not simultaneous confidence bounds for a maximum.

The maximum empirical standard error of an eight-batch controlled mean is about **.059–.062 sigma in case 0** and **.120–.154 sigma in case 2**, depending on grid and point; other contracts have much smaller errors. Raw-minus-controlled means have maximum absolute magnitudes .20–.22 and .33–.43 sigma respectively, but **all their pointwise t7 intervals include zero**. This finds no resolved raw/control discrepancy at this precision; it does not prove unbiased controls or reliable prior-wide uncertainty.

Both **emulator error and a pricing-grid effect remain visible**. More rows at the same grid cannot remove either error by themselves. Conversely, these checks do not support assigning the entire neural/reference discrepancy to the emulator. As a descriptive fixed-point calculation, case 2's independent coarse prices favor the emulator-reference mean over the neural mean by about 1.55 Gaussian log-likelihood units, alongside the emulator's 1.71; the doubled-grid calculation favors it by about 2.46. This comparison concerns only two points, omits the simulator-posterior covariance distinction, and is **not** a corrected posterior or a validated reference likelihood.

Case 2 is close to the eta prior boundary: generating eta **3.49450**, neural posterior mean **3.46660**, emulator-PyMC mean **3.48710**, prior upper bound **3.5**. The earlier 2.05-reference-SD discrepancy therefore involves a narrow boundary posterior, rather than a large absolute eta shift. The 20k bank remains a reusable finite-grid asset; choosing the next larger bank's numerical scope and improving/checking both estimators are separate next decisions.

## Frozen design and uncertainty

The [prospective plan](known-rough-pricing-plan-20261028.json) fixes six parameter points before pricing: the generating parameter, the saved neural posterior mean and the saved PyMC emulator-posterior mean for each observation. The seven observed curve levels remain fixed within a case. Each point receives **eight independent batches of 65,536 total antithetic paths**, or **32,768 independent antithetic pairs per batch**.

Both the bank grids **80/504/504** and doubled grids **160/1008/1008** are evaluated. This is a fixed-path price diagnostic, distinct from the bank's adaptive path-selection law. The hybrid Gaussian channels are coupled by normalized block sums, preserving each grid's iid-normal marginal law. It is **not** an exact common-Wiener bridge for the singular Volterra integrals. The same stream is intentionally shared across the three point roles within each case and replicate, supporting paired point contrasts; cases, replicates and maturity blocks have independent streams.

The helper reproduces the bank's antithetics and fitted control variates. It also retains bounded raw antithetic conditional-put estimates, call parity, within-batch cross-grid/raw-controlled covariance and paired grid differences. A tiny [synthetic check](synthetic-validation.json) reproduced the existing `rough_block` finest-grid prices and bank covariance **bit-for-bit**. The four numerical source files are [byte-identical to the bank producer](source-parity.json).

Here, independent pricing means fresh streams independent of the bank labels and emulator fit. It reuses the existing hybrid simulator; it is not an independent simulator implementation or an analytic rough-model price oracle.

The [stream audit](stream-audit.json) found no collisions between 48 new independent block streams and 1,392,800 reconstructed outer-purpose streams plus 3,742,200 possible block/rung pricing streams from the recorded current/historical/planned configurations. Planned configuration coverage does not imply those historical payloads were generated or recovered.

Uncertainty uses **Student t intervals with seven degrees of freedom** across the independent batch estimates. Intervals are pointwise, not simultaneous acceptance bands. Within-batch covariance conditions on same-sample fitted control coefficients; independent-batch dispersion separately checks its empirical precision. The rank-seven 63-contract replicate covariance is never inverted. Raw/control agreement and a doubled-grid comparison cannot certify continuum prices throughout the prior.

Optional Gaussian fixed-point log-likelihood contrasts are descriptive only. Their Monte Carlo trace-bias adjustment is approximate for fitted controls; this unit has **not** performed the protocol's required Heston/noisy-toy controls for a validated rough-likelihood correction. These numbers do not correct PyMC references or constitute posterior acceptance. The primary diagnostics are independent prices, emulator error and grid changes in quote-noise units. [Summary helper](summarize_known_pricing_validation_20261028.py) and [algebra/t-interval checks](summary-validation.json) preserve the calculation.

The one-shot [submission journal](execution.json) records the metadata allocation, retained first pricing batch and remaining fixed array. The unit's ceiling is **24 allocated CPU-hours, no GPU**, with at most 16 one-CPU workers, 16 GiB and 20 minutes per pricing batch, no requeue and no automatic retry. The prospective worst-case reservation includes every worker's full time limit, 240 seconds per job for cleanup and a separate final integrity reserve.

Metadata job **8032278**, first retained pricing batch **8032395** and remaining array **8032433_1 through 8032433_47** all completed with exit code 0:0: **49 allocations**. Total actual allocation was **5,790 CPU-seconds = 1.60833 CPU-hours, zero GPU**. Count only each top-level allocation/array element once. Individual pricing drivers took 29.6–49.7 seconds, but some Slurm allocations lasted up to 291 seconds; that scheduler/teardown overhead remains included in actual costs. [Accounting](accounting.psv), [collection check](collection.json), [compressed raw logs](worker-logs.json.gz) and [compressed full worker reports](worker-reports.json.gz) retain the evidence.

Full per-batch path covariance, fitted moments and partial block payloads remain on Oscar; [remote-artifacts.json](remote-artifacts.json) records their sizes and SHA-256 hashes. The small [batch-summaries.npz](batch-summaries.npz) contains the independent replicate means and covariance diagonals used by the summary, not training or reserved-test data.

The [driver](known_rough_pricing_check_20261028.py), [stream-audit script](audit_known_pricing_streams_20261028.py) and [submission operator](submit_known_pricing_validation_20261028.py) are one-off evidence helpers; they do not add runtime infrastructure to the tutorial.
