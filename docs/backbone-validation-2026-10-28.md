# Heston candidate and known-curve rough pricing checks — 28 October 2026

**Both checks completed.** The frozen Heston candidate passes all four rank screens on 2,000 untouched observations and all twelve analytic development-product quantile checks. Independent rough pricing detects emulator error and grid dependence, while leaving the full neural/reference mismatch unresolved. These checks reused existing banks, estimators and PyMC chains; no notebooks, training, new MCMC or bank expansion ran. Combined allocation was **2.6917 CPU-hours / .2708 GPU-hours**, within the two separate envelopes below.

## Heston: one frozen candidate

Use the already-saved final epoch 500 from the 100k run, SHA-256 `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`. It was identified after development failures and is a post-hoc candidate. Its original selected epoch-402 model and failures remain unchanged. The [replication audit](heston-replication-audit.md) explains this decision.

Freeze the candidate, bank, exact 2,000 reserved test IDs, seeds, native BayesFlow sampler, diagnostics and rules before accessing their outcomes. First inspect metadata and earlier journals to establish that this new-root test pool has not influenced selection. Keep all outcomes; do not replace cases, adjust criteria or tune the model after seeing them. This is one candidate's held-out calibration check, not the larger training-seed confirmation study.

Alongside calibration, reuse the three development cases' neural draws and four-chain PyMC references to assess the one-year variance-swap rate and the six-to-twelve-month forward-variance rate. Report reference convergence and chain-aware product-quantile uncertainty. The proposed tolerance is `max(10% of the reference 90% width, 0.0001 annualized variance units)` for each product's q10/q90. A pass requires the error plus its uncertainty to remain inside tolerance; inadequate precision is inconclusive. These analytic products test variance-curve uncertainty; they do not test leverage, vol-of-vol or path-dependent pricing.

One allocation is limited to four CPUs, one L40S GPU, 24 GiB and 25 minutes, with no automatic retry. The separate ceilings are **2 allocated CPU-hours / 0.5 GPU-hours**, including full queued/running exposure and a 240-second cleanup reserve. No additional models or training seeds may be evaluated on these test outcomes as part of this unit. Job **8032275** was submitted once, after declaration SHA-256 `6db210193e45b7d6af634ed77a5de17e5fbf7102c22298d18c13a3e1a141f525` was frozen; it was initially queued for GPU resources.

### Completed Heston results

| Target | Rank KS on untouched test cases | Central-95% coverage |
| --- | ---: | ---: |
| Initial variance | .02639 | 95.20% |
| Theta | .01781 | 94.40% |
| Vol-of-vol | .01745 | 94.55% |
| Rho | .01841 | 94.80% |

All four KS values are inside the predeclared simultaneous **.03562** band. Some coverage deviations remain: initial variance and theta have 50% coverage of **52.25% and 52.30%**, respectively, with pointwise Wilson intervals just excluding nominal. This is not perfect calibration or a joint-calibration guarantee.

All **twelve product quantile checks**—three development cases, two analytic products and q10/q90—are within their declared tolerance including Monte Carlo uncertainty. This is meaningful evidence about inferred variance curves, with the product scope and development-case status retained. It does not validate a path-product pricer or replace the planned training-seed confirmation and broader conditional checks.

Job **8032275** completed `0:0` in **975 allocated seconds**, using **3,900 CPU-seconds / 975 GPU-seconds**. All 31 frozen input/source hashes were rechecked. No candidate or criterion was changed after the test results. [Declaration, test results and product evidence](pilot-results/2026-10-28/heston-candidate-acceptance/README.md).

## Known-curve rough: independently check prices

Keep cases **0 and 2**, respectively passing and failing the existing approximate-emulator conditional comparison. At each case, freeze three parameter points: the generating vector, neural posterior mean and emulator-PyMC posterior mean. Keep the same observed quotes and supplied curve.

At all six points compare the bank's **80/504/504** grid with **160/1008/1008**. Use eight independent batches of 65,536 antithetic paths, paired across grids and parameter roles through the existing Gaussian-channel coupling. This coupling preserves the required marginal innovation laws; it is not claimed to be a shared singular-Wiener bridge. Retain both bounded raw conditional prices and the bank's controlled prices, their covariance, and the grid differences. Independent batches assess the reliability of fitted-control error estimates.

Compare these estimates with the exact emulator used by PyMC. Report uncertainty and quote-noise units, rather than assigning all differences to the amortized estimator. With eight batches, use the appropriate finite-replicate uncertainty calculation. The refined grid is a comparator, not exact model-price truth. These six points diagnose local mechanisms; they cannot certify a posterior region or the whole prior.

The first planned batch is also the timing and memory check and remains one of the 48 planned calls. Check it before submitting the rest. Each call has one CPU, at most 16 GiB and a 20-minute limit, with at most 16 concurrent workers. The **24 allocated CPU-hour** ceiling includes setup, all full outstanding wall limits, cleanup and final checks; no GPU. Failures retain their original points and seeds. Stop on an ambiguous submission or exhausted allowance; no automatic retries or extra batches.

### Completed rough pricing results

All **48/48 batches completed without retries**. The fine-grid estimator reproduced the existing bank routine bit-for-bit in the synthetic parity check; the numerical source files match the frozen bank producer. The six-point comparison found both emulator error and grid dependence:

| Case / point | Emulator RMS error, bank grid / quote SD | Emulator RMS error, refined grid / quote SD | Largest paired grid shift / quote SD |
| --- | ---: | ---: | ---: |
| 0 / generating parameters | .0817 | .0894 | .0962 |
| 0 / neural mean | .0763 | .0864 | .0991 |
| 0 / PyMC-emulator mean | .0772 | .0866 | .0976 |
| 2 / generating parameters | .2536 | .2383 | .2746 |
| 2 / neural mean | .1658 | .2111 | .2760 |
| 2 / PyMC-emulator mean | .1946 | .2098 | .2669 |

These norms are point estimates. The retained report includes per-quote, pointwise t7 intervals and covariance/replicate diagnostics. In case 2 at its generating parameters, one bank-grid emulator error is **−.7424 quote SD**, interval **[−.8110, −.6738]**; the corresponding refined-grid error is **−.6455**, interval **[−.7263, −.5647]**. The largest case-2 grid shift, at the neural mean, is **−.2760 quote SD**, interval **[−.2946, −.2574]**. These selected worst-quote intervals are post-hoc and pointwise, not simultaneous accuracy bands.

Raw-minus-controlled price differences do not resolve a control-variate bias in this experiment; their intervals remain broad enough that this is not a precision certificate. For case 2, the descriptive fixed-point Gaussian quote-fit comparison favors the reference mean over the neural mean at both grids; case 0's coarse-grid contrast is unresolved. These point comparisons do not characterize either posterior, validate an emulator correction or assign the full conditional mismatch to the network. The required controls for a formal rough likelihood correction have not been performed.

Case 2 is a boundary challenge: true eta is **3.4945**, the emulator-PyMC mean **3.4871**, and the neural mean **3.4666**, with prior upper bound **3.5**. More training data may improve local approximation, while more rows alone cannot remove the measured emulator and grid errors. The existing 20k bank remains useful for its declared finite-simulator task; it is not a continuous-model price-accuracy certificate. [Full pricing evidence](pilot-results/2026-10-28/known-pricing-validation/README.md).

The rough unit used **5,790 allocated CPU-seconds (1.6083 hours)**, no GPU: one metadata audit, the first pricing batch and 47 remaining pricing workers. Scheduler/teardown overhead is included, rather than quoting just the 34–55-second worker computations. All 49 allocations completed successfully; all 19 frozen input/source hashes and 48 pricing outputs were rechecked.

## Why the rough bank is still 20k

The checked known-curve bank contains **20k training / 5k validation / 7k reserved test**, up from Francesco's 1,600-training-surface pilot. This was an initial bounded baseline, not a claim of data sufficiency. At the start of this unit, Oscar's queue was empty and neither known-curve nor joint-curve data generation was active.

Known-curve generation used about **10.71 allocated CPU-seconds per surface** across the original 32k surfaces. Holding the task and numerical policy fixed gives these planning estimates:

| Training target | Additional training surfaces | Estimated additional CPU-hours |
| --- | ---: | ---: |
| 50k | 30k | 89.3 |
| 100k | 80k | 238.0 |

These are measured-rate extrapolations before new setup and headroom. At 200 continuously available cores the ideal additional times are about 27 and 71 minutes; queueing and slow shards add time. The first bank's separate 150-hour ceiling has about 54.75 hours unused, so a 50k extension needs its own bounded allocation.

The [incremental collection](incremental-calibration-banks.md) already supports growth. Under the frozen producer `6163520` and runtime, plan a larger training range with unchanged root **271300**, append only missing train IDs, audit derived seed collisions, retain holdout bytes and freeze another immutable snapshot. Do not regenerate existing surfaces. If the pricing check instead calls for a different grid, prior or observation law, use a distinct collection rather than mixing incompatible prices.

The joint-curve collection is separate: **9,925 training surfaces** are preserved, with the remaining first 20k training target and 12k holdouts pending. It uses **1280/2016/2016** and a different curve law, and has consumed **283.4164 allocated CPU-hours** within its original 1,500-hour ceiling. Its generation remains paused while these foundational checks take priority.

## Next work

1. Prepare the Heston candidate's cached teaching examples and regeneration commands, retaining the passed one-candidate gate and its claim limits. Resolve the path-product definition and validate that pricer before claiming path-product acceptance.
2. Make **50k training surfaces** the next known-curve data milestone, with a nested 20k comparison and a separately frozen generation allocation. Keep the finite-simulator scope explicit. If a different numerical target is selected, create a distinct collection and preserve the current bank. No such expansion was submitted in this unit.
3. Refit and independently check both the amortized estimator and forward emulator under the chosen target. Predeclare the checkpoint rule and optimization budget. Keep the price-emulator approximation and bank pricing covariance visible when comparing with PyMC; improving the bank size alone does not make its current emulator reference exact.
4. Resume the remaining joint collection under its existing ceiling after the foundational lesson's numerical/training decisions are fixed. Sparse chains and broad architecture screens remain downstream.

## Product definitions to preserve

The historical `products.forward_start_call` computes the return payoff `max(S_end / S_start - K, 0)`. The current experiment protocol instead specifies `max(S_end - K*S_start, 0)` in initial-spot units. These are distinct contracts under stochastic volatility. This unit's analytic Heston products avoid that ambiguity. Before a later path-product acceptance check, implement or select the declared contract explicitly and validate its pricer; preserve historical ratio-payoff results and labels.

## Posterior geometry for the tutorial

An inspection of the existing saved reference chains already supports **skewed, bounded posteriors**. Heston case 1's vol-of-vol skew is about **−.91**; applying the standard prior-box logit leaves skew about **+1.28**. A moment-matched marginal Gaussian differs from its empirical CDF by about **.09** in either coordinate system. Heston also has strongly correlated vol-of-vol/rho draws in case 0, with modest curvature; correlation alone is not evidence against a multivariate Gaussian.

Known-curve rough case 2 has eta skew about **−1.31**; a physical-coordinate Gaussian with the same moments puts about **9.8%** probability beyond the upper prior bound. Eta's **0.5–3.5** interval is a chosen prior/training domain, not a theoretical upper limit in rough Bergomi. The strong skew is principally boundary-associated. Its logit-coordinate skew is only about **+.39**, so a simple transformed Gaussian may capture much of that marginal shape. Its current neural approximation still has the recorded conditional discrepancies, and its PyMC target is an approximate emulator. Do not present either as an accepted exact-model geometry result.

These inspected cases do **not** establish multiple modes or a dramatic banana-shaped posterior. The planned nearly-flat joint-curve case is a principled place to investigate richer dependence: when short and long variance levels coincide, changing decay does not change the initial curve. Weak identification can therefore arise from the model itself. That is a motivation for the next experiment, not an observed joint-posterior result yet.

The notebooks should show at least one reference-supported non-Gaussian example, compare physical and transformed Gaussian approximations, and overlay the amortized output so neural errors are visible. Preserve the same prior, observation and numerical target. Do not select an attractive neural shape without a checked reference. No new inference or simulations are needed to inspect the current chains.

The forwardable [Heston figure and captions](pilot-results/2026-10-28/heston-posterior-geometry/README.md) and [rough geometry figures and captions](pilot-results/2026-10-28/rough-posterior-geometry/README.md) retain source/input identities and coordinate caveats. For Heston case 1's vol-of-vol, the current neural marginal CDF distance from PyMC is **.0316**, compared with **.0890/.0866** for the fitted physical/logit Gaussians. This is a descriptive case comparison, not a new architecture benchmark; a larger case-2 initial-variance discrepancy remains recorded. Strong interior curvature or multimodality remains to be demonstrated.

Both linked evidence archives retain prospective declarations, source/input identities, all outcomes, raw logs and allocation accounting. The [central plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) and [work order](tutorial-backbone-work-order.md) remain the overall entry points.
