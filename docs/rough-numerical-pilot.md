# Rough Bergomi numerical pilot

**Initial finding:** the current base grid cannot be accepted as uniformly accurate over the candidate six-parameter prior. Several declared cases show repeatable price movements of multiple quote-noise standard deviations when the grid is refined. The [27 October plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) uses these findings to define a bounded cost/precision pilot and posterior/product refinement check within the existing workflow. Large new-task generation remains on hold for review. These results do not establish that 4× is accurate, choose a replacement grid, or justify changing the prior automatically.

These are development results for complete synthetic option surfaces, not posterior validation, market evidence or accepted tutorial checkpoints. The [core implementation guide](full-surface-calibration.md) and [scientific protocol](../tutorial_final/EXPERIMENT-PROTOCOL.md#numerical-precision) describe the remaining gates.

This is an extension of existing work, not the repository's first numerical check. The [earlier course runbook, R2](../tutorial_final/CLUSTER-RUNS.md), describes its bank's discretized model and its grid study without a pass/fail threshold. That 12-surface study reports a resolved five-day wing movement around one quote SD. The [existing examples and their validation](full-surface-calibration.md#relationship-to-the-existing-rough-examples) remain useful with their original scope. The proposed addition ties numerical refinement to the tutorial's posterior and product conclusions; it is not a separate inference philosophy or a certificate over the full prior.

## What was run

The Oscar panel used the eight predeclared cases in `numerical_pilot.DEFAULT_CASES`, covering typical, rising/falling and nearly flat curves, low H/high eta, high H/low eta, high variance and a steep rising curve. All cases use the same complete 63-contract design and the candidate joint rough task. Cases were not selected from their observed errors.

- Three grids: the base block step counts are 80/504/504, followed by 2× and 4× refinements of each block. The analytic initial variance curve is rebuilt on each grid; control expectations use the corresponding discrete left-cell integral.
- Two independent repetitions per grid, each with 32,768 paths. Different grids, repetitions and blocks use distinct recorded pricing streams. These are independent comparisons, not a coupled refinement obtained by reusing an integer seed for differently shaped normal arrays.
- Fixed path counts: no precision-based redraw, replacement seed or discarded failure. Every run was retained, including those exceeding the MC target.
- Prices use antithetic conditional estimators and fitted control variates. Reported Monte Carlo standard errors concern that finite estimator; they do not include time-discretization bias.

Each case's declaration records the full task, panel, settings, actual pricing-seed audit, source hashes and runtime versions before pricing. The tracked results are under [pilot-results/2026-10-26/rough-grid](pilot-results/2026-10-26/rough-grid). Preserve `declaration.json` together with `report.json`; the report refers to the declaration by hash.

## Initial results

The table compares pooled 4× prices with pooled base prices. Positive means the refined price is higher. Each quote has its own declared Gaussian noise SD; the values below divide both the price change and its MC standard error by that quote's SD. The contract index is zero-based in the stored task design. Strikes are in forward units and maturities are trading days, with 252 days per year.

| Case | Contract with largest absolute base → 4× change | Signed change / quote SD | MCSE of change / quote SD | All six calls meet MC target 0.25? |
| --- | --- | ---: | ---: | --- |
| [0: typical flat](pilot-results/2026-10-26/rough-grid/case-0/report.json) | Index 0: 5-day put, strike 0.945214 | +0.378 | 0.146 | Yes |
| [1: rising curve](pilot-results/2026-10-26/rough-grid/case-1/report.json) | Index 45: 126-day put, strike 0.753638 | +0.224 | 0.160 | Yes |
| [2: falling curve](pilot-results/2026-10-26/rough-grid/case-2/report.json) | Index 0: 5-day put, strike 0.945214 | +0.682 | 0.205 | Yes |
| [3: nearly flat, fast decay](pilot-results/2026-10-26/rough-grid/case-3/report.json) | Index 54: 252-day put, strike 0.670320 | +0.232 | 0.120 | Yes |
| [4: low H, high eta](pilot-results/2026-10-26/rough-grid/case-4/report.json) | Index 49: 126-day ATM put, strike 1 | +4.704 | 0.467 | No: all six exceed 0.5 too |
| [5: high H, low eta](pilot-results/2026-10-26/rough-grid/case-5/report.json) | Index 8: 5-day call, strike 1.057961 | −0.026 | 0.005 | Yes |
| [6: high variance](pilot-results/2026-10-26/rough-grid/case-6/report.json) | Index 8: 5-day call, strike 1.057961 | −3.267 | 0.217 | No: all six exceed 0.5 too |
| [7: low short level, steep rise](pilot-results/2026-10-26/rough-grid/case-7/report.json) | Index 53: 126-day call, strike 1.326896 | −2.256 | 0.421 | No: all six exceed 0.5 too |

For cases 4, 6 and 7, every grid/repetition call has at least one quote exceeding both the 0.25 target and the existing 0.5 exception cap; this does not mean every quote exceeds them.

The largest absolute change need not be the most precisely estimated one. For example, falling-curve case 2's largest change is **+0.682** at the index-0 put. Its index-8 call instead moves **−0.330** quote SD with MCSE **0.0256**. These are distinct contracts and opposite directions.

The strongest effects repeat across the independent comparisons. For case 4's index-49 put, the two base-to-4× changes are +4.410 and +4.998 quote SD. For case 6's index-8 call they are −3.164 and −3.369. For case 7's index-53 call they are −1.939 and −2.574, with more MC uncertainty. Substantial changes also remain between 2× and 4×, so 4× is not a reference truth.

The 48 calls cost **586 CPU-seconds (0.163 CPU-hours)** in total, with process peak RSS about **0.94 GiB**. These fixed-path refinement calls are not a representative throughput benchmark for adaptive bank generation, whose difficult surfaces can require far more paths.

## How to interpret uncertainty

For each grid, the two independent price estimates are averaged. The pooled standard error is the square root of the sum of the squared per-repetition errors, divided by two. Independent-grid errors then add in quadrature for a price difference. The reports contain adjacent-grid comparisons; the table above derives base-to-4× changes from the same retained price/error arrays.

The reported 95% intervals are **approximate pointwise MC intervals**, not simultaneous bounds across 63 correlated quotes, eight cases and several grid comparisons. Selecting the largest observed shift introduces selection effects. Statistically detectable tiny wing shifts are not automatically economically material, and passing an MC standard-error target does not establish negligible grid bias. Conversely, an over-target MCSE does not erase a large repeatable grid movement; it means precision and bias must be resolved separately.

Within-grid differences between the two repetitions are below 3 estimated SE in cases 0–6; case 7 reaches about 3.56 SE on the base grid. Two repetitions are insufficient to validate tail behavior or MC error calibration. Do not treat them as a comprehensive reliability test.

## Higher-precision follow-up

**Two comparisons completed; the low-H/high-eta comparison is incomplete. No new grid or prior is accepted.** The follow-up isolated 4× versus 8× for cases 4, 6 and 2, with two independent repetitions and predeclared fixed paths of 1,048,576, 524,288 and 131,072 respectively. Each `(case, grid, repetition)` ran as a separate component. The reducer checks declarations, sources/runtime, settings, exact coverage and pricing seeds before combining results; failed or missing components cannot silently disappear.

| Case | Paths per grid/repetition | Contract with largest absolute 4× → 8× change | Signed change / quote SD | MCSE at that contract / quote SD | Largest pointwise MC95 halfwidth across all quotes |
| --- | ---: | --- | ---: | ---: | ---: |
| [6: high variance](pilot-results/2026-10-26/rough-fine-reduced/case-6/report.json) | 524,288 | Index 8: 5-day call, strike 1.057961 | −0.8515 | 0.0398 | 0.3641 quote SD |
| [2: falling curve](pilot-results/2026-10-26/rough-fine-reduced/case-2/report.json) | 131,072 | Index 18: 21-day put, strike 0.890947 | +0.2831 | 0.0989 | 0.1995 quote SD |

All four grid/repetition calls in each completed case meet the 0.25 MCSE target. That does not make the time-grid effect negligible. In high-variance case 6, the index-8 call's approximate pointwise interval is −0.929 to −0.774 quote SD: its price still moves substantially from 4× to 8×. The 4× grid therefore cannot be accepted as uniformly stable across the candidate prior. The largest interval halfwidth in the table concerns a different contract from the largest observed shift; these uncertainty summaries must not be interchanged.

Falling-curve case 2 has a smaller, less precisely resolved maximum shift, with an approximate pointwise interval of +0.089 to +0.477 quote SD at index 18. Its more precisely estimated index-8 call moves −0.0831 quote SD with MCSE 0.0111. These are exploratory contract-level observations, with the same multiple-comparison and posterior/product-accuracy caveats as the initial panel.

For low-H/high-eta case 4, both [4× repetition 0](pilot-results/2026-10-26/rough-fine/case-4/grid-4-rep-0/report.json) and [4× repetition 1](pilot-results/2026-10-26/rough-fine/case-4/grid-4-rep-1/report.json) completed at 1,048,576 paths. Their maximum MCSE/quote-SD ratios are 0.119 and 0.155. **Both 8× jobs reached the 20-minute Slurm time limit without a final price report.** The comparison is incomplete: the timeout is neither an 8× precision pass nor a numerical price failure. No 4× → 8× shift can be inferred for this case, and no automatic retry is scheduled.

The completed case-6 and case-2 components used about 2,317 and 995 CPU-seconds respectively; successful refined components reached about 1.46 GiB process peak RSS. Scheduler accounting also retains the unfinished low-H runs rather than treating them as zero-cost successes.

This targeted follow-up resolves some of the initial MC uncertainty but does not establish a converged pricing scheme. Further refinement, path increases, numerical-target changes or prior restrictions need an explicit scientific decision. A converged pricing panel must still be followed by posterior and product-refinement checks before the tutorial's inference claims are accepted.

## Numerical options after choosing the monotone curve

The core curve family is now the positive monotone exponential. The remaining choice is how accurately and over what domain to represent its rough variance dynamics. The curve's simplicity does not remove the rough time-grid problem.

| Choice | Benefit | Cost or limitation |
| --- | --- | --- |
| Increase paths at a fixed grid | Reduces Monte Carlo noise and makes grid comparisons easier to interpret. Under ordinary MC scaling, four times the paths roughly halves the standard error. | Does not remove grid bias. High-variance case 6 already meets the MC target at both 4× and 8×, yet its short call moves 0.85 quote SD. Scaling and error estimates still need empirical checks in difficult regions. |
| Refine every block | Directly investigates the time-discretization approximation while retaining the model/prior. | Expensive; the full 8× low-H runs hit their time limits. Neither 8× nor an assumed convergence rate can be treated as truth. Refining all blocks can spend effort away from the observed discrepancy. |
| Allocate finer steps to the affected block | Can resolve a short-maturity problem without multiplying the whole one-year simulation cost. | Establishes only the selected block's stability. Remaining maturities and downstream products still need their own checks. |
| State the numerical approximation and limit the claims to its checked accuracy | Makes the observation law, reference target and evidence reproducible. | Numerical conventions and pricing randomness must match across comparisons. This does not remove the need for targeted refinement checks or establish market accuracy. |
| Restrict the teaching prior or horizons | Can make a limited, well-checked example affordable. | Changes the financial question and scope of amortization. Restrictions need a substantive explanation and explicit validation; shrinking the prior merely to discard failed cases is not a numerical fix. Shortening the final horizon does not resolve a bias already present in five-day options. |
| Change the simulation algorithm or use an emulator | A better simulator may improve the accuracy/cost trade-off; an emulator can make repeated reference inference much faster. | Requires separate implementation and validation. An emulator trained on biased labels learns that numerical target; it does not repair its grid bias. |

The current MC target, 0.25 quote SD, is a development compromise, not a claim that numerical error is negligible. For perspective, an independent centered error with that SD adds 6.25% of the quote variance; at 0.5 it adds 25%. These per-quote variance calculations do not account for bias or the shared-path covariance across quotes. Raising the numerical tolerance or the observation noise changes what is being checked and must not silently substitute for resolving a detected price shift.

The simulator already uses a one-cell hybrid approximation of the singular Volterra kernel and normalizes the variance by the approximation's own Gaussian variance. That preserves its mean variance curve; it does not make the asset-integral approximation exact. The [hybrid-scheme paper](https://doi.org/10.1007/s00780-017-0335-5) studies driver approximation error and rough-option pricing, with discretization of the asset dynamics still present. Its results do not supply a universal option-price error bound for our candidate prior. The small-grid covariance reference in the code is capped at 128 steps, so it is a diagnostic rather than a drop-in replacement for all fine grids.

The conditional pricer also constructs full asset-path outputs that its conditional payoff does not use, including an independent asset channel that has already been integrated out. A focused implementation optimization might reduce time or memory after equivalence checks. It would be a performance improvement, not evidence that a coarser grid is accurate; a changed generator would carry a new source identity.

### Spend and the meaning of a realistic teaching domain

**Do not equate numerically difficult cases with unrealistic markets.** Case 6 has a flat variance curve of 0.15, equivalent to about 38.7% forward-variance volatility; this is not its at-the-money option implied volatility. The [published curve training domain](https://arxiv.org/html/2306.11061v2#S4.SS3), Table 2, includes long variance levels up to 0.16. Its Table 5 includes rough Bergomi `H` from 0.025 to 0.50 and `eta` from 0.50 to 4.00. These are training/calibration domains, not empirical probabilities. [Rømer's empirical SPX study](https://doi.org/10.1080/14697688.2022.2081592), Section 4.1, reports an average fitted rough Bergomi H around 0.09; removing low H merely for computational convenience could remove important market behavior.

A narrower domain can be a realistic **subset**, such as a declared low/moderate-volatility equity-index regime, while deliberately excluding stressed markets. Low H or extreme correlation need not themselves indicate financial stress. Exact cutoffs and a restricted domain have not been adopted. The present independent-uniform endpoint-variance prior is a design choice, not an empirical market distribution: on `[0.01, 0.16]`, the median endpoint volatility is about 29.2%, and about 71.6% of draws have at least one endpoint above 30% volatility. A financially motivated regime/prior could change how much training effort goes to high-volatility curves, but its support still needs numerical checks. Empirical fits with more flexible curves do not directly validate the new monotone family's joint prior.

Measured fixed-path costs show the scale of the numerical trade-off:

| Declared price call | Mean CPU-seconds at full 4× | Mean CPU-seconds at full 8× | Interpretation |
| --- | ---: | ---: | --- |
| Falling case 2, 131,072 paths | 69 | 428 | MC target met; grid stability assessed separately. |
| High-variance case 6, 524,288 paths | 274 | 884 | MC target met, but material short-call grid movement remains. |
| Low-H/high-eta case 4, 1,048,576 paths | 563 | No completed result | Both 8× components reached 20-minute limits. |

At 32,768 paths, the eight-case median costs were 4.46 / 8.02 / 19.63 CPU-seconds for full 1× / 2× / 4× grids. The inexpensive calls do not establish adequate prices: three difficult cases miss the MC caps and the coarse grids show material bias. Node/working-set effects matter too; observed full 8×/4× cost ratios are not a universal scaling law. Selective short-block refinement cost only 17.6 CPU-minutes for its whole twelve-component panel, but it did not close the numerical gate.

For planning only, this arithmetic illustrates a **100,000-training-surface** bank:

| Assumed average generation cost | CPU-hours | Ideal elapsed time with 200 continuously busy cores |
| --- | ---: | ---: |
| 30 CPU-seconds/surface | 833 | 4.2 hours |
| 120 CPU-seconds/surface | 3,333 | 16.7 hours |
| 600 CPU-seconds/surface | 16,667 | 83.3 hours, about 3.5 days |

These are sensitivity scenarios, not measured moderate-versus-broad-domain forecasts or an approved allocation. Add holdouts, independent validation/audits, queues and overheads. Pilot calls use the same fixed path count in every block; actual bank generation selects paths by block and also pays for earlier ladder rungs. Corner cases are deliberately overrepresented here. A representative throughput pilot at the chosen grid/prior is necessary before quoting the bank's mean cost or a savings factor from narrowing it. No dollar rate has been established.

The subsequent [performance pass](oscar-pilot-2026-10-26.md#pricing-performance-pass) reduced CPU work by 15–18% on its measured Oscar rough blocks with identical estimates. That is useful savings, not evidence of a bank-wide savings factor or a remedy for grid bias. The older fixed-call timings above retain their original source/runtime identities.

The domain decision remains open for scientific review. Continue the independent Heston work; do not silently narrow rough support, relax observation noise or start a large rough bank while that decision is pending.

## Targeted diagnostic: the first maturity block

This panel isolated the confirmed short-maturity issue before choosing a global grid. It used cases 2 and 6, 131,072 fixed paths, and two independent repetitions per setting:

| Setting | First block: through 10 trading days | Other two blocks |
| --- | ---: | ---: |
| Short-block 4× | 320 steps | 2,016 / 2,016 steps |
| Short-block 8× | 640 steps | 2,016 / 2,016 steps |
| Short-block 16× | 1,280 steps | 2,016 / 2,016 steps |

All 63 quotes remain in each output. **Only indices 0–17, the five- and ten-day options, diagnose the changed grid.** Indices 18–62 are MC controls on unchanged grids; their independent random price differences cannot establish convergence of their maturity blocks. In particular, a maximum taken over all 63 differences could select noise from those controls and misdescribe this experiment.

This is a new declared block-allocation experiment with a separate seed namespace. All blocks were repriced with independent streams at each setting, so the independent-error comparison remains valid; there is no hidden sharing of noisy prices. Twelve independent components allowed parallel execution. The `rough-short-block` array mapping selects case 2 then case 6, with 4×/8×/16× and repetitions 0/1 for each. Other historical array mappings and seeds remain unchanged.

Measured full-4× calls at 131,072 paths cost about 70 CPU-seconds each. On that basis, roughly 15–25 CPU-minutes across the twelve selected-block components was the planning estimate, not a guaranteed rate; only the smaller first block grows. The per-component request was one CPU, 8 GiB and a ten-minute limit. This panel did not retry the timed-out low-H case or claim a global MC precision pass at the moderate path count.

### Observed short-block outcome

All twelve components completed, and both six-component reductions passed their declaration, source/runtime, coverage and seed checks. The retained [raw components](pilot-results/2026-10-26/rough-short-block) and [checked reductions and analysis summary](pilot-results/2026-10-26/rough-short-block-reduced) preserve the complete observations. The table follows the previously identified **index-8 five-day call, strike 1.057961**, rather than selecting an unrelated full-surface maximum. Each signed change and its parenthesized MCSE is in that quote's noise-SD units.

| Case | Short-block 4× → 8× | Short-block 8× → 16× | Short-block 4× → 16× |
| --- | ---: | ---: | ---: |
| [2: falling curve](pilot-results/2026-10-26/rough-short-block-reduced/case-2/report.json) | −0.0677 (0.0111) | −0.0313 (0.0108) | −0.0990 (0.0110) |
| [6: high variance](pilot-results/2026-10-26/rough-short-block-reduced/case-6/report.json) | −0.9666 (0.0800) | −0.5001 (0.0693) | −1.4667 (0.0765) |

High-variance case 6 still has clear short-price grid dependence. Its index-8 call's 8× → 16× approximate pointwise interval is **−0.636 to −0.364 quote SD**, and its 4× → 16× interval is **−1.617 to −1.317**. The decreasing size of successive shifts is useful evidence, but does not identify a convergence rate or establish 16× as truth. The 8× grid is not accepted as uniformly accurate over the candidate prior.

Falling-curve case 2 shows a smaller residual movement at the same call. Its 8× → 16× interval is approximately **−0.0525 to −0.0102 quote SD**; the 4× → 16× interval is **−0.1206 to −0.0774**. Practical acceptance still depends on posterior and product sensitivity, not statistical detection alone.

The largest affected-quote change can be less informative than this precise endpoint. For case 6, the affected 8× → 16× maximum is **+0.784 at the index-0 five-day put**, with MCSE **0.365** and a wide pointwise interval of +0.069 to +1.499. It must not be substituted for the index-8 call result or read as accurately resolved positive bias. Likewise, unchanged-grid control quotes in case 6 have a maximum absolute 4× → 16× difference of about 0.613 quote SD from MC noise; those quotes were not refined in this experiment.

All six case-2 calls meet the 0.25 MCSE target, with maximum affected-quote ratios of 0.099–0.102 per call. For case 6, all six calls exceed the target, with maxima 0.363–0.374, although they remain below the existing 0.5 exception cap. Pooling two repetitions yields maximum affected-quote MCSEs of about 0.070–0.072 for case 2 and 0.258–0.263 for case 6. The particular index-8 call is more precise: its pooled errors are about 0.0075–0.0080 and 0.046–0.061 respectively. This explains how a clear wing-price movement can coexist with unresolved MC precision elsewhere on the same surface.

The twelve components used **1,054 CPU-seconds (17.6 CPU-minutes, or 0.293 CPU-hours)** in total and about **1.00 GiB** peak process RSS. The selective design completed within its planned compute envelope without repeating the timed-out low-H jobs. These results isolate the short block; they do not validate the fixed later blocks, the full prior, posterior inference or product valuation.

The next reviewed unit is the bounded numerical/cost pilot in the [central plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md#next-unit-after-review-two-parallel-workstreams), followed by targeted posterior/product refinement checks. No new grid/prior or large rough bank follows automatically from this report. Price, posterior and product accuracy remain separate checks within the shared workflow.
