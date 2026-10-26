# Rough Bergomi numerical pilot

**Initial finding:** the current base grid cannot be accepted as uniformly accurate over the candidate six-parameter prior. Several declared cases show repeatable price movements of multiple quote-noise standard deviations when the grid is refined. This blocks substantial rough-bank generation. It does not establish that the 4× grid is accurate, choose a replacement grid, or justify changing the prior automatically.

These are development results for complete synthetic option surfaces, not posterior validation, market evidence or accepted tutorial checkpoints. The [core implementation guide](full-surface-calibration.md) and [scientific protocol](../tutorial_final/EXPERIMENT-PROTOCOL.md#numerical-precision) describe the remaining gates.

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
