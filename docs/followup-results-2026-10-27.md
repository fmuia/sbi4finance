# Pricing reliability and Heston inference follow-up

**The authorized unit is complete. All 52 rough calls and both Heston training/evaluation runs finished within budget.** The rough results support an empirical precision policy for a development bank, with the known limitations retained. Neither new Heston fit resolves the inference checks. The subsequent [native-default check and bank start](native-default-and-bank-start.md) are recorded separately; the results below retain their original sampling settings.

Oscar used **1.24 allocated CPU-hours and 0.231 GPU-hours**, against ceilings of 7 CPU-hours and 1 GPU-hour. The archived comparison additionally used **68.1 process CPU-seconds locally**; small saved-data reductions also ran locally. No new reference MCMC, large bank, acceptance-pool evaluation or notebook execution occurred.

This executes the [reviewed work order](reviewed-unit-2026-10-27.md#next-bounded-work-order-for-review). Results here remain development evidence. They extend the existing numerical workflow, rather than define a separate inference philosophy.

## Rough: enough to make a development-bank decision

The three existing anchors—central, low-H/high-eta and high variance—each received 16 independent batches of 32,768 total paths per block, on the working grid `1280/2016/2016`. Controlled and raw conditional-put estimates share the same paths. These are fixed-path diagnostics, not 52 adaptive calls required to meet the 0.25 error target.

All 52 calls completed, including four timing calls. The 156 new block streams are distinct from one another and from all 724 initialized states in the preceding reviewed declaration. No failed point, stream or seed was replaced.

### Reported errors and observed variation

The ratio below compares the empirical standard error of the pooled mean across independent batches with the pooled error estimated from regression residuals:

| Anchor | Median ratio across 63 quotes | 90th percentile | Largest ratio |
| --- | ---: | ---: | ---: |
| Central, case 0 | 0.974 | 1.245 | 1.391 |
| Low-H/high-eta, case 4 | 0.957 | 1.177 | 1.270 |
| High variance, case 6 | 0.917 | 1.214 | 1.281 |

There is no broad underestimation apparent in this panel. Sixteen batches and many correlated quotes do not validate nominal coverage or rare tails. Pooling averages separately fitted batches; it does not refit OLS on a larger combined path sample. At the preidentified five-day call and 126-day ATM put, these pooled controlled means also agree with the previous independent larger-batch working estimates within one nominal combined SE. Both estimates can still share bias.

The bounded raw-put comparison is much noisier. At the high-variance five-day call, controlled minus raw is −1.613 quote SD, with a heuristic interval [−3.163, −0.063]. That deserves disclosure, but is not proof of bias: the interval is approximate, the quotes are correlated, and the raw comparison has insufficient precision to resolve small differences. For scale, interval half-widths at this quote are 0.389 / 0.967 / 1.550 quote SD across the three anchors.

**The 0.25 quote-SD target is a reported MCSE policy, not a newly imposed requirement to certify absolute bias below 0.25 at every quote.** Turning this small diagnostic into that certificate would require a much larger study. The [martingale-control moment qualification](reviewed-unit-2026-10-27.md#what-the-reported-monte-carlo-errors-establish) remains: this work does not prove finite variance of the fitted estimator or validate every residual-based interval.

### A further check using only saved statistics

After inspecting the first diagnostic, we froze an **exploratory** calculation: pair adjacent recorded batches, swap their fitted coefficients, and compare the eight independent pair averages with the original estimates. No new paths, model evaluations or coefficient fits were used. The original prices reconstructed to within 5.2e−15 spot units; the method, pairing, tolerance and input hashes were frozen before calculation.

| Preidentified quote | Same-sample minus cross-fitted mean / quote SD | Heuristic interval |
| --- | ---: | --- |
| High-variance five-day call | −0.00173 | [−0.00633, 0.00288] |
| Low-H/high-eta five-day call | +0.114 | [−0.028, 0.255] |
| High-variance 126-day ATM put | +0.0219 | [−0.0084, 0.0522] |

The earlier high-variance raw-price discrepancy is not reproduced as an effect of fitting coefficients on the same sample. The low-H result remains uncertain; it does not establish a material fitting effect above the 0.25 scale. This exploratory calculation does not settle control centering, rare tails, grid bias or continuous-model prices. All 63 results are retained, rather than only favourable quotes.

**Recommendation:** retain the existing estimator and reported 0.25 adaptive precision rule for a scoped development bank. No blanket SE inflation, replacement control or additional raw-MC campaign is justified by this panel. Preserve covariance, final-rung exceptions and failures. Keep independent pricing checks in the later posterior/product panel, particularly in the known difficult cases; do not describe this as prior-wide accuracy certification.

[Checked diagnostic analysis](pilot-results/2026-10-27/followup/rough/analysis.json), [checked reduction](pilot-results/2026-10-27/followup/rough/reduction/report.json), [frozen exploratory method](pilot-results/2026-10-27/followup/rough/cross-batch-method.json), [cross-batch results](pilot-results/2026-10-27/followup/rough/cross-batch-analysis.json).

### Timing: refinement is affordable, but this is not bank throughput

At the previously slow **random prior point6**, a single allocation ran working/refined/refined/working, with 32,768 paths per block and no diagnostic capture or extra warmup:

- Working CPU times: 19.92 and 19.21 seconds.
- Refined CPU times: 46.04 and 45.98 seconds.
- Ratio of means: **2.35×**.

This point differs from **stress anchor6**. The first working call may include cold effects; no call was omitted. Earlier extreme timing differences involved different nodes and unusually high system CPU. This matched result confirms the importance of runtime conditions without identifying their cause or replacing the measured adaptive-bank cost scenario.

## Heston: more epochs did not resolve conditional accuracy

Both new fits use the same 20k training snapshot, fresh quote noise, bounded parameter adapter, seed 0 and minimum-validation-loss selection. The spline retains its 128/128 architecture; FlowMatching uses four 256-unit layers. Both passed exact seeded save/reload checks. The reported checkpoint is the selected one, not the final epoch.

The 500 validation observations belong to collection `1a7906f…`. The three saved MCMC observations belong to the original readiness collection `e5c12a5f…`. Their short labels overlap, but their collection identities and price vectors differ. Every conditional comparison uses the exact same three saved observations, priors, noise and passing references.

| Estimator | Selected epoch | Proposed conditional checks passed | Main finding |
| --- | ---: | ---: | --- |
| Existing 20k spline, 300-epoch schedule | 269 | 0/3 | Previous development baseline. |
| New 20k spline, 1,000-epoch schedule | 472 | 0/3 | Extra optimization did not solve the errors. |
| New 20k FlowMatching, 500-epoch schedule | 403 | 1/3 | Better in several coordinates, but large initial-variance errors remain. |
| Archived 50k FlowMatching, BayesFlow 2.0.14 | 487 | 2/3 | Useful existing control; not a matched data-size/framework experiment. |

For the new spline, initial-variance mean errors are 1.485 / 0.265 / 0.545 reference SD; its corresponding 90% width ratios are 1.808 / 1.642 / 1.194. Training loss improved after the selected epoch while validation loss worsened, consistent with overfitting this training bank. Another blind epoch extension is not the next action.

For new FlowMatching, initial-variance mean errors are 0.024 / 1.400 / 1.492 SD, with width ratios 1.069 / 1.067 / 1.715. Case1 also has a theta width ratio of 1.267. Analytic variance-product discrepancies remain; these fits do not establish product acceptance.

The same 500-case development validation panel reinforces that conclusion:

| Fit | Parameters passing SBC rank screen | Initial-variance 90% coverage | Theta 90% coverage |
| --- | ---: | ---: | ---: |
| Existing 300-epoch spline | 4/4 | 95.4% | 91.8% |
| New 1,000-epoch spline | 3/4 | 92.8% | 90.8% |
| New 500-epoch FlowMatching | 2/4 | 84.8% | 84.8% |

The simultaneous DKW screen bound is 0.07124. Spline initial-variance KS is 0.0972; FlowMatching initial-variance/theta KS are 0.0845/0.0760. Passing a prior-predictive rank screen alone does not establish conditional agreement, as the previous spline already illustrates. Validation losses from spline likelihood and flow-matching regression objectives are not comparable.

### The archived control and the next Heston diagnosis

The 50k course checkpoint was restored locally in the existing BayesFlow 2.0.14 environment, retaining its serialized preprocessing, alpha−0.5 and historical 100-step nonuniform Euler sampling rule. Cases 0/1 pass the proposed parameter screen; case 2 fails initial-variance mean agreement at 0.432 reference SD, with width ratio 1.158. No fitting, pricing, MCMC or holdout evaluation was performed. JAX used multiple CPU threads despite requested one-thread settings; 68.1 CPU-seconds across all threads and 9.55 wall-seconds are recorded, not a one-core throughput claim.

At the time of these fits, the BayesFlow 2.1 factory explicitly used **100 uniform Euler steps**, overriding BayesFlow 2.1's native adaptive Tsit5 default. New workflows now use the native default; these recorded results and checkpoints retain Euler. The archive uses a grid dense near its data endpoint. Those choices can contribute to sampling error, alongside the learned model/data; this unit does not identify the cause of the discrepancies.

**Subsequent decision and result:** the user chose BayesFlow's native default, superseding the proposed 400-Euler/native sweep. A native-only check on the same three cached observations and latent seeds is [complete](native-default-and-bank-start.md), within the separate 1-CPU-hour / 0.25-GPU-hour allowance. It widens some intervals but leaves the conditional pass count at 1/3. No retraining, pricing, MCMC or native 500-case SBC evaluation was performed. The 500-case metrics above describe the original Euler readout.

[New Heston runs and comparisons](pilot-results/2026-10-27/followup/heston/), [archived control declaration](pilot-results/2026-10-27/followup/archived-heston/declaration.json), [archived results](pilot-results/2026-10-27/followup/archived-heston/run.json).

## Subsequent bank decision

The user subsequently authorized the [32k bank work order](rough-development-bank-work-order.md): existing working grid, broad prior, 20k training / 5k validation / 7k reserved test surfaces, and a separate **1,500-allocated-CPU-hour generation ceiling**. The collection and job map are frozen, the seed audit passes, and staged generation has started. See the [execution report](native-default-and-bank-start.md) for status. This does not add an unbounded bias-certification gate.

That ceiling is headroom over the previous 1,178-hour empirical planning scenario, not a statistical completion guarantee. The 2.35× fixed-path result does not replace that estimate. Training, emulator labels and the later six-case posterior/product refinement panel remain separately budgeted. Generate only a first development snapshot, then one posterior and one forward emulator; wider architecture screens and sparse-chain work remain downstream.

## Cost, provenance and retained evidence

| Work | Oscar allocated CPU-hours | GPU-hours |
| --- | ---: | ---: |
| Rough declaration and four groups | 0.317 | 0 |
| Heston spline, 511 seconds | 0.568 | 0.142 |
| Heston FlowMatching, 322 seconds | 0.358 | 0.089 |
| **Oscar total** | **1.242** | **0.231** |

Allocated CPU is elapsed time times requested CPUs; measured process CPU is a separate quantity. `.batch` and `.extern` CPU is not counted twice. The local archived comparison and small saved-data analyses are additional local CPU work within the remaining allowance.

Main `574b940` was merged before source freeze. Heston runs use `51fec86`; rough pricing uses `615a72f`. Original pending A6000 requests were updated in place to permitted L40S resources; job IDs, recipes, seeds, counts and limits remained fixed. Actual GPU devices and raw accounting are retained; CPU frequency counters are not labelled GPU clocks.

The [execution ledger](pilot-results/2026-10-27/followup/execution.json) and [evidence guide](pilot-results/2026-10-27/followup/README.md) identify sources, streams, records and replay commands. Large model/validation artifacts remain on Oscar with hashes and sizes. Four large rough reports are losslessly compressed with original-byte hashes; no outcomes were discarded. Eighteen new guard/diagnostic tests passed, alongside existing reference and conditional-output checks; saved-summary identities and gzip reproduction were independently verified. Neither new estimator is an accepted teaching checkpoint, and the original course artifacts remain intact.
