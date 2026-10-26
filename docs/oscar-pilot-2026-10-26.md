# Full-surface development on Oscar — 26 October 2026

**The CPU/GPU workflow works, and the first Heston development bank and estimators exist. Scientific acceptance is still open.** The rough pilot found material time-grid effects in difficult regions of the candidate prior. The monotone exponential initial curve is now the agreed simplified teaching family, with its relationship to the published hump family and its limitations documented. Large rough-bank generation remains on hold while numerical settings and the parameter prior are checked.

Sparse-chain support remains a separate downstream unit. No notebooks were executed or changed for these runs.

**Next-step clarification, 27 October:** the [review plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) extends Francesco's existing approximate-price workflow with bounded posterior/product refinement checks. His runbook's descriptive grid study does not define a separate inference philosophy for the whole course. Further execution is on hold for review; the numerical records below are unchanged.

## What ran in parallel

| Work | Result |
| --- | --- |
| CPU and single-GPU readiness | Passed on Oscar, including real incremental generation, separate-process locking, retry reuse, prefix preservation, interrupted training recovery and exact same-seed reload. |
| Rough grid panel | Eight declared cases, three grids and two independent repetitions: all prices returned; three difficult cases exceeded the Monte Carlo precision caps. Base-to-refined price shifts are material in some regions. |
| Curve-family challenge | Flat/monotone controls and one hump case, followed by three exact-observation hump refits. Relaxing the decay bound improves the found fit but does not remove all discrepancies. |
| Heston data | A fresh 20k/5k/5k training/validation/test collection was generated and an immutable 20k snapshot published. Root seed is fixed at 271028. |
| Heston training | Spline/vector baselines on the same 20k surfaces with 60 and 300 epochs. The longer schedule improves calibration diagnostics; neither checkpoint is accepted as a tutorial reference yet. |
| PyMC references | Native four-core reruns completed in 6.2–8.5 minutes per case, but none meets all R-hat/ESS requirements. Fourier point checks show very small changes; conditional neural-accuracy verdicts remain inconclusive. |

The [rough numerical report](rough-numerical-pilot.md) and [curve-family screen](curve-family-screen.md) give the scientific details. The checked [JSON evidence](pilot-results/2026-10-26/) includes declarations, outcomes, source/runtime hashes and seeds. Inconclusive runs are retained rather than replaced with favourable seeds.

![Constructed hump truth and monotone point fits; no posterior uncertainty bands.](pilot-results/2026-10-26/curve-shape-comparison.png)

## Readiness and provenance

CPU readiness took 79.9 seconds. GPU readiness took 90.7 seconds on an NVIDIA RTX A6000. Both included generation, recovery and evaluation; these are execution timings, not scientific throughput estimates. The lock check used separate processes on one node, so it does not certify every cross-node filesystem behavior. [CPU evidence](pilot-results/2026-10-26/readiness-cpu.json), [GPU evidence](pilot-results/2026-10-26/readiness-gpu.json).

The run root is `/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4`. Independent immutable checkouts preserve each phase:

- `9d903db`: readiness, initial pilots and the Heston bank/training workflow.
- `16f8c7e`: precision components, exact-observation curve follow-ups and the reference precision checker.
- `8566b29`: native parallel PyMC chains and partial-draw retention.
- `efdb41b`: selected short-block refinement and the simplified-curve scope decision.
- `e7b3e2e`: data-only conditioned PyMC follow-up and analytic variance-functionals.
- `17344a1` / `180e358`: conditional rough-output and Heston maturity-reuse benchmarks. These reuse the already installed byte-identical locked runtime through explicit source paths; the immutable checkout identity remains separate.

The project was transferred as verified Git bundles because the cluster checkout did not have GitHub credentials. The original experiments and archived environments were not overwritten. A short GPU readiness job moved from the condo queue to the permitted debug allocation when the condo GPU quota was occupied.

## Heston development findings

The bank contains 20,000 training surfaces and fixed 5,000-surface validation/test pools. Generation uses 500-row shards and a fresh root seed, 271028, distinct from the smoke/reference pilots. Continue expansion with the frozen generator/runtime and increasing training IDs; do not regenerate the first tranche or move holdouts. [Configuration](pilot-results/2026-10-26/heston-development/config.json), [immutable snapshot](pilot-results/2026-10-26/heston-development/snapshot-20k-manifest.json).

Both development evaluations use the first 500 validation surfaces and 1,000 posterior draws per surface. Final test data were not evaluated. The second training run changes optimization effort/schedule on the same bank, not the amount of unique data.

| Diagnostic | 60 epochs | 300 epochs |
| --- | ---: | ---: |
| Initial-variance SBC KS distance | 0.1144 | 0.0580 |
| Simultaneous DKW threshold across four parameters | 0.0712 | 0.0712 |
| SBC parameter checks passed | 3/4 | 4/4 |
| Initial-variance nominal 90% interval coverage | 99.2% | 95.4% |
| Theta / vol-of-vol / rho nominal 90% coverage | 93.2 / 94.0 / 95.8% | 91.8 / 93.8 / 94.4% |

The longer fit improves the first failure, but intervals remain conservative. Passing the rank screen does not establish conditional agreement with MCMC, adequate products or final acceptance. The same development pool influenced this diagnosis and must not be relabelled untouched acceptance evidence. [60-epoch evaluation](pilot-results/2026-10-26/heston-development/evaluation-60epochs.json), [300-epoch evaluation](pilot-results/2026-10-26/heston-development/evaluation-300epochs.json).

![Development coverage before and after longer optimization; neither is final acceptance.](pilot-results/2026-10-26/heston-coverage.png)

The 300-epoch GPU-trained model also loaded and produced finite draws on the local CPU environment. That is a portability check, not an additional inference-accuracy result. Model binaries remain development artifacts on Oscar rather than replacing tracked teaching checkpoints.

## Conditional references are still inconclusive

The first three four-chain runs used sequential chains and exceeded a 15-minute wall allowance. Oscar did not permit extending those running jobs. Native PyMC parallel chains and atomic partial trace files were then added and checked on known Gaussian targets and Heston. The same observations and seeds were rerun; these are replacements for incomplete computations, not independent extra reference evidence. Partial files retain tuning flags but cannot resume sampler/adaptation/RNG state.

| Development surface | Parallel elapsed time | Largest parameter R-hat | Smallest bulk / tail ESS | Meets all requirements? |
| --- | ---: | ---: | ---: | --- |
| validation:0 | 395 s | 1.040 | 158 / 129 | No |
| validation:1 | 512 s | 1.399 | 8.4 / 9.5 | No |
| validation:2 | 369 s | 1.038 | 113 / 70 | No |

The requirements remain R-hat at most 1.01 and bulk/tail ESS at least 400. These failures do not establish a neural-method failure or success. The comparison tool preserves an `inconclusive_reference` outcome and does not sample the neural estimator or emit an accuracy verdict against an unchecked target. [Reference evidence](pilot-results/2026-10-26/heston-reference-parallel/), [guarded comparison outcomes](pilot-results/2026-10-26/heston-reference-comparison/).

Case 1 has visible retained-draw drift: two chains' first-half mean vol-of-vol values around 1.17/1.07 move to about 1.46/1.46 in the second half. The other cases also approach prior boundaries. The next bounded reference experiment should use dispersed, data-only classical fits for initialization and regularized proposal conditioning in PyMC's transformed coordinates, retaining the same likelihood/prior/Jacobians. Keep the built-in sampler and test the hard case first; do not remove the drifting half post hoc, relax criteria or change prior bounds to manufacture convergence.

Doubling Fourier nodes and cutoffs at 16 deterministic chain/draw points per surface produced maximum price changes of 0.0000030, 0.0000072 and 0.0000030 quote-noise standard deviations. The largest absolute log-likelihood change was 0.000025. All selected points priced and the pricing sources matched the recorded reference. This supports pointwise pricing stability, not a posterior-refinement proof or convergence. The immediate reference problem is mixing/initialization/proposal conditioning. [Pointwise precision evidence](pilot-results/2026-10-26/heston-reference-precision/).

### Conditioned-proposal follow-up

The difficult validation:1 case was rerun with the same observation, prior, likelihood, root seed and 2,000-tune/2,000-draw budget. Four bounded data-only transformed-MAP fits supplied dispersed starts and local proposal geometry. Native PyMC DEMetropolisZ uses an explicit multivariate-normal proposal in logit coordinates; Uniform transforms/Jacobians and full prior support remain unchanged. All fits, costs, covariance regularization and actual starts are retained in [reference metadata](pilot-results/2026-10-26/heston-reference-conditioned/case-1/reference.json).

The run completed in 446 seconds. The largest parameter R-hat improved from 1.399 to 1.024, but vol-of-vol still misses R-hat ≤1.01 and bulk ESS ≥400: its bulk/tail ESS are 377/418. The other three parameters pass. Both analytic curve products pass their own MCMC checks, but this does not certify the joint posterior, product quantile accuracy or neural estimator. The [guarded neural comparison](pilot-results/2026-10-26/heston-conditioned-comparison/case-1/comparison.json) remains `inconclusive_reference`, without neural sampling or a verdict. [Full diagnostic record](pilot-results/2026-10-26/heston-reference-conditioned/case-1/diagnostics.json).

The matching 16-point Fourier check has maximum price change 0.0000081 quote SD and maximum absolute log-likelihood change 0.000030. Pricing sources match. This again supports pointwise numerical stability only; it cannot rescue nonconverged chains. No further Heston sampling round was automatically submitted after this bounded result. [Precision evidence](pilot-results/2026-10-26/heston-reference-conditioned-precision/case-1/precision.json).

## Rough follow-ups did not close the numerical gate

Higher-path, independent 4×/8× components were run concurrently. For high variance, all four calls met the 0.25 Monte Carlo standard-error target, but the one-week far call still moved by −0.851 quote-noise standard deviations, with estimated difference SE 0.040. For the falling-curve control the largest shift was +0.283, SE 0.099; a more precisely estimated wing shift remains. These are pointwise comparisons, not global posterior/product acceptance. Checked reductions were computed locally from the recorded Oscar components; this required no new pricing or replacement seeds.

The low-H/high-eta case completed both 4× calls at 1,048,576 paths each, meeting the Monte Carlo target. Both 8× calls reached the 20-minute wall limit without final prices. That comparison is incomplete; it supplies no 8× accuracy verdict and was not automatically retried. [Checked reductions](pilot-results/2026-10-26/rough-fine-reduced/), [full numerical interpretation](rough-numerical-pilot.md).

Across the recorded development unit, 97 Slurm tasks completed and five timed out: the three initial sequential references and the two largest rough components. Reported CPU accounting totals about 4.68 hours; this is neither GPU-hours nor a confirmed large-campaign throughput estimate. Final bank generation succeeded without redrawing failed surface IDs. [Accounting](pilot-results/2026-10-26/accounting-summary.json).

## Targeted short-maturity follow-up

After choosing the monotone curve family, twelve additional Oscar components isolated the first maturity block rather than refining the whole one-year simulation. Cases 2 and 6 used 131,072 paths, two independent repetitions, and short-block grids 4×/8×/16×; the other blocks stayed at 4×. All components completed in 82–127 seconds. The prices below concern the same five-day call, contract index 8.

| Case | 4× → 8× change / quote SD | 8× → 16× change / quote SD | MCSE of the 8× → 16× change |
| --- | ---: | ---: | ---: |
| Falling curve, case 2 | −0.0677 | −0.0313 | 0.0108 |
| High variance, case 6 | −0.9666 | −0.5001 | 0.0693 |

Selective refinement is affordable, but the high-variance short-call movement remains clearly resolved. This does not establish the error relative to the continuous model or make 16× a truth. Case 6 also misses the panel-wide MC target at this moderate path count. Indices 18–62 are independent MC controls on unchanged grids, not evidence of convergence at longer maturities. Total recorded pricing cost is 1,054 CPU-seconds (0.293 CPU-hours), with peak RSS about 1 GiB. [Raw components](pilot-results/2026-10-26/rough-short-block/), [checked reductions and extraction](pilot-results/2026-10-26/rough-short-block-reduced/).

The [numerical-options discussion](rough-numerical-pilot.md#numerical-options-after-choosing-the-monotone-curve) separates additional paths, grid allocation, application scope, clear numerical conventions and algorithm/emulator work. The next step is the shared workflow's bounded refinement check, not a separate finite-simulator track. No large rough bank is approved by this diagnostic.

The twelve components plus the conditioned Heston reference and its precision check all completed. Their separate [continuation accounting](pilot-results/2026-10-26/continuation-accounting-summary.json) records 0.69 CPU-hours; it does not overwrite the earlier accounting or estimate future bank throughput.

## Pricing performance pass

The existing rough implementation already uses a vectorized FFT hybrid driver, antithetic conditional pricing, control variates and adaptive per-block path counts. Profiling found avoidable work within that foundation. The conditional estimator now omits unused asset-return/log-spot outputs while retaining the exact integral operations and all three RNG channels. A conservative finiteness check falls back to the original full calculations on extreme inputs. The public full-path simulator still returns its existing outputs by default.

Heston's characteristic function depends on maturity, not strike. Reusing exact repeated maturity values avoids evaluating it separately at every strike; quadrature and refinement decisions are unchanged. No approximate maturity rounding or parameter cache is used.

Both changes were benchmarked on Oscar with one CPU and unchanged locked runtime versions:

| Pricing workload | Before | After | Measured improvement |
| --- | ---: | ---: | --- |
| Rough long block, 504 steps / 2,048 antithetic pairs | 0.301 CPU-s | 0.246 CPU-s | 18.4% less CPU; process peak RSS 354 → 222 MB |
| Rough long block, 2,016 steps / 2,048 pairs | 1.197 CPU-s | 1.013 CPU-s | 15.3% less CPU; process peak RSS 1,065 → 606 MB |
| Heston full 63-contract task, warm base/refined pricing | 36.1 ms wall | 11.7 ms wall | 3.08× faster; cold call 146 → 63.7 ms |

Rough timings are medians of four order-balanced repetitions in separate processes, including an untimed warmup per process; RSS includes imports and warmup. Heston uses 80 alternating warm repetitions and one cold call per implementation. Returned prices, standard errors, covariance and applicable refinement outputs match bit for bit in the recorded comparisons. These are specific pricing benchmarks, not average adaptive-bank generation rates, universal speedups or numerical-accuracy improvements. [Rough benchmark](pilot-results/2026-10-26/pricing-optimization/rough-oscar/benchmark.json), [Heston benchmark](pilot-results/2026-10-26/pricing-optimization/heston-oscar.json), [source/runtime/job definitions](pilot-results/2026-10-26/pricing-optimization/jobs.json).

The integrated focused suite passes 139 tests, including default API preservation, extreme failure behavior, unchanged RNG/rung decisions, exact pricing outputs, all Heston quadrature rungs and broadcast contract layouts. Historical bank producers remain frozen: do not bypass collection source checks to adopt an optimized generator or regenerate existing IDs. The new rough bank can use the optimized implementation when its task/estimand is frozen.

## Decisions before scaling rough generation

1. The curve-family scope is agreed: use the monotone exponential as a simplified subfamily of the published five-coefficient specification. Keep the hump challenge as a visible limitation. The parameter prior still needs a numerical check; the hump study does not establish market-shape frequencies or general market adequacy.
2. Review the proposed working/refinement grids and bounded numerical-cost panel. More Monte Carlo paths alone do not remove time-discretization bias; assess materiality for the declared posterior/product conclusions.
3. Complete the bounded Heston reference follow-up using the already implemented data-only initialization/proposal conditioning before expanding the reference library. Check conditional agreement, numerical sensitivity and product uncertainty before declaring the neural estimator accurate. Preserve failed, timed-out and inconclusive cases.
4. Expand only selected learning curves and configurations. The data-growth tools are ready; million-surface banks are capacity, not an immediate requirement.

The [central plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) still governs the full tutorial campaign. These results justify targeted scientific follow-ups and Heston development; they do not complete the campaign or approve a large rough bank.
