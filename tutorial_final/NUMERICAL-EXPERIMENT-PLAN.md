# Numerical experiment plan for Heston and rough Bergomi

**Execution update — 27 October 2026. The reviewed A/B unit is complete; it added no large rough bank or GPU training.** The implementation and completed runs are in [PR 48](https://github.com/fmuia/sbi4finance/pull/48); the course baseline is [Francesco's PR 44](https://github.com/fmuia/sbi4finance/pull/44), inspected at `cebfb1f`.

The aim is a tutorial that shows **what the posterior learned, how we checked it, when amortization saves work, and how joint uncertainty affects a product**. We extend Francesco's existing simulation and inference workflow. For rough Bergomi, both his workflow and the new one use numerical time grids and Monte Carlo price estimates; Heston uses Fourier quadrature. We retain the existing precision controls and add a bounded check of whether numerical refinement changes the tutorial's posterior and product conclusions. The supported claims will match that evidence.

The reviewed A/B unit was deliberately small: **finish a Heston reference check and measure the new rough task's numerical/cost trade-offs in parallel**. It ended with results and a conditional first-bank cost scenario, within its **26-allocated-CPU-hour ceiling, with no GPU training**. The subsequent **7-CPU-hour plus 1-GPU-hour** follow-up is also [complete](../docs/followup-results-2026-10-27.md), using 1.24 allocated Oscar CPU-hours and 0.231 GPU-hours plus small local analysis. Its results support the empirical rough development policy; the new Heston fits still fail accuracy checks. The next proposals are a [costed 32k development bank](../docs/rough-development-bank-work-order.md) and a cached Heston solver check. Neither has launched.

**Result:** all three Heston references pass the declared mixing checks, while the BayesFlow 2.1 development checkpoint fails conditional agreement. The rough pilot returned 43 of 44 calls; one refined prior-point call timed out. Read the [completed unit, costs and next decision](../docs/reviewed-unit-2026-10-27.md) before stage C. The A/B specification below is retained to show what was reviewed and run; it does not authorize another round.

## Scope already agreed

- Two core examples: complete-surface Heston and joint rough Bergomi, using the declared 63-contract design. Participants follow observation → posterior → diagnostics → product uncertainty.
- Heston infers four parameters with mean reversion fixed. Rough Bergomi infers `H, eta, rho, short_variance, long_variance, decay`; its generating curve is not supplied as an inference input.
- The rough curve is `long_variance + (short_variance - long_variance) * exp(-decay * t)`. Positive levels and decay allow rising, falling and flat curves, but no hump. It is the simplified subfamily of [Baschetti, Bormetti and Rossi, Section 4.3](https://arxiv.org/html/2306.11061v2#S4.SS3) obtained by removing the hump term. It shares Heston's mean-curve shape, not Heston's stochastic dynamics. The [completed hump challenge](../docs/curve-family-screen.md) remains a documented limitation.
- Use BayesFlow 2.1.0 for new fits; preserve archived 2.0.14 estimators in their original environment. Prefer PyMC where natural, following [AGENTS.md](../AGENTS.md) and [the MCMC policy](../docs/mcmc-policy.md). No wholesale sampler migration is required.
- Sparse quotes, masks, arbitrary contract layouts and thin-chain market examples remain a [separate downstream unit](SPARSE-QUOTE-CASE-PLAN.md). Core inputs must be complete and finite.
- Numerical difficulty alone does not justify excluding a financial regime. Keep the broad rough candidate prior for the pilot; any later restriction needs an application-based explanation and explicit review.

The candidate rough prior remains independent uniforms: H 0.02–0.35, eta 0.5–3.5, rho −0.99 to −0.2, each curve variance level 0.01–0.16, and decay 0.25–8 per year. These are working bounds under review, not an empirical distribution of market parameters.

## What exists and what we reuse

| Existing work | How it enters this plan |
| --- | --- |
| Earlier rough bank, H/eta checkpoint, reference and held-out tests on main | Reuse its numerical foundation, checks and teaching material; retain its provisional status and original restricted task. |
| Francesco's 2,200-surface rough pilot, with 1,600 training surfaces | A completed conditional-on-known-curve pilot, not a missing attempt. Reuse its methods and results with their limitations. Its model/raw bank are not tracked; recover them if available before recreating that old task. |
| Committed 50k-trained Heston estimator | Keep it as the established BayesFlow 2.0.14 comparison baseline. Its scientific task matches our current Heston task. Raw bank availability still needs checking. |
| New 20k/5k/5k Heston collection and BayesFlow 2.1 development fits | Reuse this data and the selected 300-epoch development checkpoint. Preserve fixed holdouts; grow training only if diagnostics justify it. These fits have not replaced the course baseline's evidence. |
| New joint rough generator, incremental banks, training/reload and optimized pricing | Implemented and checked. The new six-target task still needs compatible bank data and trained estimators. Old supplied-curve prices cannot be relabelled or projected into it. |

The [reuse inventory](../docs/full-surface-calibration.md#relationship-to-the-existing-rough-examples) gives artifact details. The [Oscar report](../docs/oscar-pilot-2026-10-26.md) retains failures, timeouts and inconclusive results. Pricing benchmarks show about 3.08× faster warm Heston calls and 15–18% less CPU on measured rough blocks, with identical outputs. These are not bank-wide throughput estimates. Infrastructure is sufficient; add machinery only for a concrete unmet need.

## Reviewed A/B unit: two parallel workstreams

All cases below are **development evidence**, not untouched final acceptance data. Record exact observations, IDs, seeds, sources, runtime and settings before execution. Budgets are ceilings, not promises that every check will finish. Retain failed/incomplete outcomes; no replacement seeds or automatic budget doubling.

### A. Close the bounded Heston reference follow-up

**Question:** can the new PyMC route provide a trustworthy conditional comparison for the existing BayesFlow 2.1 development estimator?

1. Keep the same three development observations and deterministic Fourier likelihood. The difficult case's latest vol-of-vol R-hat is 1.024, so it remains inconclusive despite improvement. Fix the observed prices, prior, noise and initialization rule; new sampler settings go into new output directories.
2. Start with that difficult case, using the optimized pricer, the recorded data-only initialization and PyMC's built-in sampler. Proposed single follow-up: four chains, 4,000 warmup and 8,000 retained draws per chain; four CPUs, at most 30 minutes. If it passes, apply the same recipe to the other two cases. Maximum allocation across three runs: **6 CPU-hours**.
3. Keep R-hat ≤1.01 and bulk/tail ESS ≥400 for parameters and important products. Also check trace behavior, numerical sensitivity and uncertainty in reference summaries. No neural starts or post-hoc trimming. A remaining failure stops this reference unit; it does not trigger repeated longer chains.
4. For valid references, compare the new neural posterior and its existing analytic variance summaries within the same allocation. Cost forward-start-call simulation separately with the later product stage. If neural accuracy fails, diagnose optimization/data/capacity before increasing the bank. Recover compatible old assets where possible; retain the archived Heston estimator as a control.

**Deliverable:** one short Heston comparison report, including failures, and a concrete list of what remains before notebook packaging. These three cases are development checks, not the final reference library.

### B. Measure rough numerical quality and realistic campaign cost

**Question:** which working numerical settings are useful and affordable for the six-target task, and which cases need further posterior/product checks?

1. Reuse the existing eight declared stress-case definitions and recorded results. Add **16 fixed prior-drawn parameter points**, pricing each on both candidate policies: **32 adaptive price-vector calls**. Reserve root seed **271100** for this pilot, with separate recorded parameter/grid/repetition purposes and an audit of actual pricing streams. Do not select or redraw points after seeing timing or errors.
2. Retain the current total-path ladder: **32,768 / 65,536 / 131,072 / 262,144 / 524,288 / 1,048,576 / 2,097,152**, selected separately by block. The MC standard-error target stays 0.25 quote SD; any final-rung 0.5 exception is separately counted and reviewed. Fixed-path stress timings alone are not representative bank costs.
3. Add independent precision comparisons only at three existing anchors: central (case 0), low-H/high-eta (case 4), and high variance (case 6). Each gets two repetitions at each policy: **12 fixed-path calls**, using respectively **131,072 / 1,048,576 / 524,288 total paths per call**, not antithetic pairs. The historical eight-case panel is not regenerated wholesale.
4. Freeze the following **44-call** manifest before submission. Use one CPU and 8 GiB per pricing component. Cap the whole pilot, including reduction/setup overhead, at **20 CPU-hours**. Report failures/timeouts and censored costs; do not estimate an unconditional bank average from completed cases alone.

| Proposed policy | Through 10 trading days | Through 63 days | Through 252 days |
| --- | ---: | ---: | ---: |
| Working-grid candidate | 1,280 steps | 2,016 steps | 2,016 steps |
| Refinement comparator | 2,560 steps | 4,032 steps | 4,032 steps |

These are `(16,4,4)` and `(32,8,8)` refinements of the original `80/504/504` blocks, chosen from the observed short-end problem. Neither is already accepted as accurate. The comparator doubles every working block, so it checks more than short-maturity quotes.

| Component group | Calls | Wall limit per one-CPU call | Maximum allocation |
| --- | ---: | ---: | ---: |
| Prior-point adaptive pricing | 32 | 20 minutes | 10 h 40 min |
| Fixed-path precision, excluding refined case 4 | 10 | 30 minutes | 5 h |
| Refined low-H/high-eta anchor | 2 | 90 minutes | 3 h |

The component allocation totals **18 h 40 min**, leaving 1 h 20 min within the 20-hour ceiling for setup/reduction. If the limits prevent a conclusion, keep it inconclusive; do not replace a timed-out point or promise a complete cost distribution from this small panel.

Run up to 32 independent pricing components concurrently as Oscar permits, with one numerical-library thread per CPU. Heston's first case can run alongside them; its other two cases follow only after that check passes. CPU-hours are allocation ceilings, not elapsed-time forecasts or confirmed cluster availability.

**Deliverable:** a working-grid recommendation, a refinement comparator, identified difficult cases, and a costed first-bank work order. Include generation, fixed holdouts, training, references and independent checks. Use CPU-hours and measured queue/allocation assumptions, not an unsupported dollar quote or an idealized core-count promise.

### Decision at the end of A/B

Record the intended observation/noise law, rough prior, working grid, refinement grid and actual generation budget. A working grid is a development choice, not a certificate of continuous-model accuracy. Material price shifts flag cases for the posterior/product check below; they do not automatically justify narrowing the prior or abandoning the workflow. If the evidence or cost is insufficient, bring that specific issue back for review before substantial generation.

## Following stages, in dependency order

These stages describe the campaign after the initial results. **Large-bank spending and GPU allocations must be costed from A/B before submission.** They are not included in the 26-CPU-hour initial-unit ceiling.

| Stage | Concrete work | Deliverable and stop rule |
| --- | --- | --- |
| C. Freeze and generate the first joint bank | Proposed first snapshot: **20k training / 5k validation / 7k reserved test** surfaces: **32k price surfaces in total**. Use one frozen task, root seed and generator identity. | Checked immutable snapshot and actual cost. Stop on coverage, seed, source or precision failures; do not replace failed surface IDs. The reserved test pool remains unopened. |
| D. Establish one rough baseline | Fit **one vector-input posterior** and **one forward price emulator** on the same training data. Select/reload using validation only. The emulator maps all six physical parameters to prices; it is not the posterior network. | A working inference → check → product example and independently measured emulator errors. No broad family/encoder sweep yet. Price-emulator agreement alone is not ground truth. |
| E. Check numerical sensitivity of the conclusions | Run the six-case development panel below, starting with two feasibility cases. Use deterministic PyMC emulator references where natural, plus independent numerical pricing checks. | Casewise posterior/product refinement report. Stop or revise when references, overlap, emulator accuracy or numerical precision are inadequate; do not turn an inconclusive comparison into a pass. |
| F. Grow only where useful | If development evidence calls for more unique data, grow training **20k → 50k → 100k**, preserving the prefix and holdouts. Give each fit an adequate, declared optimization budget. | Learning curves that distinguish data, training and reference limitations. Plateauing with failed checks is a diagnosis, not evidence of sufficiency. |
| G. Confirm and package | Freeze configuration/selection rules, confirm training-seed variability, and evaluate untouched acceptance cases. Bind checked results/checkpoints to the notebooks and rehearse cached/live paths. | One reviewable tutorial lesson per model, with precisely scoped claims and retained failures. No notebook execution is part of the present plan-editing task. |

Stage C prices the fixed holdouts up front because the existing immutable-snapshot workflow requires complete declared splits. The work order must include that cost; do not quote only the 20k training cost. Holdout prices are reserved for later evaluation and are not opened by training or model selection. Revisiting the initial split allocation is a review decision before freezing the collection, not a reason to add new snapshot infrastructure now.

## The bounded posterior and product refinement check

Use six predeclared joint-rough development observations: central, falling curve, nearly flat curve, low-H/high-eta, high variance and steep rise (existing case definitions 0/2/3/4/6/7). Begin with **central and high variance** to establish feasibility; keep the other four fixed rather than selecting easier replacements. At most **six observations × two numerical resolutions = twelve reference fits** are planned. Cost the independent pricing labels and reference runs before submission.

Compare the **same observed quote vector**, prior and quote-noise law under the working and refined pricing approximations. Do not regenerate observations when changing the reference grid. Account consistently for pricing randomness/covariance and emulator approximation. A fixed-seed likelihood, a simulator-marginal likelihood and an ideal-price approximation are distinct reference targets; label the one actually used. A standard MCMC call with a fresh random log likelihood is not a valid substitute.

Use one validated baseline forward emulator. If needed, add a small, separately checked local refinement correction rather than a full second bank or architecture campaign. Check it on independent higher-precision labels covering both posterior regions; stop if it is inaccurate or the refined posterior leaves that coverage. The [protocol](EXPERIMENT-PROTOCOL.md#bounded-posterior-and-product-refinement-panel) defines this route and its limits.

Proposed checks for review:

- Parameter posterior means move by at most **0.1 refined-reference SD**; 90% interval widths change by at most **10%**. Reference uncertainty must be small enough to resolve those changes; ESS 400 is a minimum, not a guarantee of that precision.
- Main teaching product: an **at-the-money forward-start call resetting at three months and expiring at six months**, within the quoted horizon. Also check the **one-year continuously monitored variance-swap rate** as an analytic curve-uncertainty control. It does not check leverage or vol-of-vol by itself.
- Proposed absolute tolerance floors: **0.0001 initial-spot units** for the call (one basis point of spot) and **0.0001 annualized variance units** for the variance-swap rate. These are review proposals, not accepted economic materiality thresholds.
- With reference 90% product width `W`, define `T = max(0.10 * W, absolute floor)`. Product q10/q90 refinement changes must be at most `T/4`, with pricing/reference uncertainty resolved. Assess product-pricer refinement separately from changes in the parameter posterior.

An unresolved check is **inconclusive**. A resolved large change calls for a targeted numerical/model/claim revision. Passing this small panel supports statements about those cases; it does not establish accuracy over every prior point or empirical market adequacy. Final neural accuracy and calibration still require their own held-out checks.

## Data growth and later architecture comparisons

Append only missing split-qualified IDs; keep root seeds and holdouts fixed. Publish immutable snapshots and record the snapshot hash for each fit. Preserve numerical-error/covariance records. Existing producers remain frozen; do not bypass source checks to extend an old collection with changed code. A changed task/grid/prior needs a distinct collection, not relabelled old prices. Dataset growth starts a new training run; interruption recovery remains bound to the original snapshot/settings. The [incremental-bank guide](../docs/incremental-calibration-banks.md) is the implementation reference.

The wider comparison remains bounded and follows checked baselines:

| Screen | Fits |
| --- | ---: |
| Heston vector input: affine coupling, spline coupling, FlowMatching, MeanFlow | Four short fits, reusing a compatible baseline |
| DeepSet and SetTransformer with one fixed posterior family | Two additional fits, with complete quote sets |
| Transfer two promising configurations to the joint rough task | Two fits |
| Confirm one selected configuration per model across three seeds | Six total confirmation fits; reuse matching runs where possible |

This is at most eight screen fits plus six confirmations, with data sizes and GPU limits chosen from measured development results. MeanFlow sampling-step and FlowMatching solver checks use existing checkpoints where possible. Million-surface capacity, extra architectures, direct rough Monte Carlo MCMC and additional products remain optional; they do not gate the core tutorial.

Final validation includes prior-predictive calibration, conditional-reference agreement, product uncertainty, inference cost and fresh participant execution. Preserve **2,000 untouched acceptance surfaces per model** for the declared final SBC/coverage check and a separate final reference collection; inspected pilot/test results remain development evidence. Use matched observations/noise/prior for comparisons, report raw and corrected neural results separately, and retain original sampler provenance. The existing [claims registry](CLAIMS.md) and [course status](../docs/course/status.md) change only when the corresponding evidence supports a teaching claim.

## Next review

Review the [completed follow-up](../docs/followup-results-2026-10-27.md) and [first-bank spending proposal](../docs/rough-development-bank-work-order.md) before substantial generation. The new configuration reserves the same broad prior and working task for development; no narrower regime has been adopted. The six-case posterior/product panel remains necessary for teaching claims, rather than a requirement to certify every price before creating development data.

Technical details live in [EXPERIMENT-PROTOCOL.md](EXPERIMENT-PROTOCOL.md); completed results in [the Oscar report](../docs/oscar-pilot-2026-10-26.md); runnable instructor commands in [the implementation guide](../docs/full-surface-calibration.md). The older [cluster runbook](CLUSTER-RUNS.md) remains historical/reusable guidance, not an instruction to run its entire older programme.
