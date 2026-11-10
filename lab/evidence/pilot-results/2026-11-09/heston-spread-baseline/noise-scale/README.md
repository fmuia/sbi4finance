# One Gaussian scale: fixed sensitivity and joint inference

**Completed development study, 10 November 2026; no accepted neural checkpoint.** This study examines the noise-scale step of the Heston plan on the **existing 54 actual SPXW contracts**. It follows the user-authorized scale investigation in [PR #85](https://github.com/fmuia/sbi4finance/pull/85). The [offline HTML report](report.html) shows the results, recovery/SBC plots and executed network/rate code. The frozen [study protocol](protocol.json), original [reference protocol](reference-protocol.json) and [training protocol](training-protocol.json) were persisted before their respective runs. The explicit [cache recovery addendum](reference-recovery-v2.json) preserves the original targets and budgets.

The subsequent user-authorized [larger-bank training comparison](capacity-study/README.md) is complete under separate protocols and output identities. Both corrected continuations use the same six-target posterior, 100k physical surfaces, batches of 128/512 and 500 additional complete bank passes. Loss improves but c remains close to its prior, with final central-90% widths 1.349/1.345 against prior 1.35. The [new report](capacity-study/report.html) retains the numerical repair, all eight evaluations and an exploratory input-conditioning diagnostic. The results and stop below describe this original study and remain unchanged.

The five physical Heston priors, contract geometry, carry, ACT/365F clock and observed half-spread profile remain unchanged. No grid labels, spread-profile variation, additional tau/vega component, flexible-input training or Bergomi work belongs to this study. The earlier fixed-c neural failures and capped boundary reference remain under their original identities.

## Results and reflection

All seven new reference targets pass their declared convergence and sampled pricing checks. Fixed-scale sensitivity behaves as expected on the same observed prices:

| Synthetic case | Physical 90% widths at c=.5 / c=1 | Widths at c=2 / c=1 | Joint reference c central-90% interval | 60k neural c width / joint reference width |
| --- | --- | --- | --- | --- |
| Broad | .477–.494 | 1.93–1.99 | [.938,1.315] | 3.48 |
| Market-like | .495–.514 | 1.97–2.10 | [.970,1.364] | 3.33 |

The fixed-scale ranges span the five physical parameters. They are descriptive changes between different assumed targets, not accuracy scores. The joint references establish useful scale information in both synthetic cases. The current neural approximation misses much of it: at 60k its median c interval width is 1.336 versus the Uniform prior's 1.35, and its posterior-median recovery RMSE is .4452 versus .4371 for the constant prior-median predictor.

All six marginal population SBC screens pass at 60k. Nevertheless, the matched conditional panels fail at every predeclared endpoint (10k,50k,60k). At 60k, broad-case c has .94 reference-SD mean error and 3.48× width; market-like c has 1.03 SD error and 3.33× width. These are definite failures under the predeclared Monte Carlo-aware criteria. Strong physical-parameter recovery and reasonable pooled ECDFs coexist with these conditional errors. Neural ignorance does not establish that c is unidentifiable.

The separate real-market reference concentrates at the illustration prior's upper bound: c mean 1.99904, SD .00097, central-90% interval [1.99710,1.99995], with all sampled mass above 1.95. The network also fails that matched target. Boundary pressure is a warning about residual-scale support, model/data assumptions and fit under this truncated working model; it does not identify pure quote noise or justify retrospectively widening the prior.

The single six-target fit completed 60k finite updates in 480.48 seconds, with exact seeded checkpoint reload at all three endpoints. All 20k physical training rows were seen; reserved arrays remained unopened. The [neural exports](neural-evaluations/060000/evaluation.json) preserve the original population results and conditional samples. The [final comparisons](evaluations/060000/evaluation.json) consume those same samples after reference completion; no additional neural sampling or SBC recomputation occurs.

Across the seven references, maximum R-hat is 1.00893 and minimum bulk/tail ESS is 421.36. The largest sampled price-refinement change is approximately 1.03e-5 of the minimum supported measurement SD, and maximum absolute log-likelihood change is 6.81e-5. These sampled checks support numerical accuracy where tested, not a prior-wide certificate. Charged reference CPU, including failed zero-draw startup work and the symbolic cache preflight, is 2,947.86 of 6,300 seconds; the original wall deadline was preserved. [Recovery records](recovery-v2-campaign.json) retain all costs and failures; [verification](verification.json) records software and compute checks.

**Stop for reflection here.** The scale study is complete; neither the old fixed-c estimator nor the new joint estimator is accepted. A further neural improvement study must keep its target and comparisons explicit. Revised-grid labels, flexible inputs and Bergomi remain unstarted.

## Two inference questions on the same observations

Use one conditional Gaussian error law:

`Y_i = P_i(phi) + c * h_i * Z_i`, with independent standard-Normal `Z_i` conditional on c.

Here `h_i = (ask_i - bid_i)/2` is a frozen price half-spread and c is one positive dimensionless multiplier shared by all 54 quotes in a surface. It is not one absolute price SD applied equally to all quotes. Equating h to a Gaussian SD at c=1 remains a working convention.

| Question | What is supplied or inferred? | What is held fixed? |
| --- | --- | --- |
| Known-scale sensitivity | Assume c=.5,1,2 separately; infer the same five physical parameters | Exactly the same observed price vector y, h, geometry, physical prior and numerical gate |
| Unknown-scale inference | Infer the five physical parameters and c jointly, with independent physical-coordinate `c ~ Uniform(.5,2)` | The same y, h and geometry; the generating c is hidden |

Changing the assumed fixed c does **not** regenerate the observed prices. The two synthetic reference observations were generated at c=1; treating those same observations under c=.5 or2 is deliberate likelihood sensitivity, not a fresh correctly specified calibration experiment at those scales. With independent Uniform physical priors, changing a global fixed c preserves the physical MAP mathematically, because it multiplies the same weighted squared residual objective by a positive constant. Posterior means and interval widths can nevertheless change with nonlinear geometry and prior boundaries.

Joint inference is different from averaging the three fixed-scale posterior plots. It satisfies

`p(phi | y) = integral p(phi | y,c) * p(c | y) dc`,

where the scale weights are learned from the data under the declared prior. They are not equal fixed weights. The normalized likelihood includes `-54*log(c) - R(phi)/(2*c*c)`, where `R = sum(((y-P(phi))/h)^2)`, plus c-independent constants. Omitting the normalization changes the target.

The existing five-target c=1 network is not applied at other assumed scales or relabelled as a six-target network. Its failures do not disappear if a different joint target proves easier. Five- and six-target training losses are also not directly comparable accuracy scores.

## Matched reference panel and fixed-physical oracle

| Saved observation | Fixed c=.5 | Fixed c=1 | Fixed c=2 | Joint unknown c |
| --- | --- | --- | --- | --- |
| Broad synthetic, generating c=1 | New reference | Reuse ready reference | New reference | New six-target reference |
| Market-like synthetic, generating c=1 | New reference | Reuse ready reference | New reference | New six-target reference |
| Actual market | Deferred | Reuse ready application reference | Deferred | New application reference |

This is **seven new targets**, not seven new observations. Exact observation, design, task, source and selected-diagnostic/payload hashes bind every comparison. The existing boundary reference exhausted its cap and remains inconclusive; this matrix does not retry it or replace its historical outcome.

The reference protocol also records `p(c | phi_truth,y)` for the two saved synthetic observations, using their saved refined clean prices. At fixed physical parameters its density is proportional to `c^(-54)*exp(-R/(2*c*c))` on [.5,2], and its physical-coordinate mode is `sqrt(R/54)` clipped to that interval. This is a useful normalization and finite-sample check. It is **not** the six-dimensional posterior, its marginal `p(c|y)`, or evidence that physical parameters are known at inference. Its center need not equal the generating c in each noise realization.

No paired generating-scale .6/1/1.8 reference observations are added in this bounded matrix. The two conditional synthetic cases therefore do not establish conditional neural accuracy away from generating c=1. Broader scale calibration is assessed separately through the declared prior-predictive development population.

Each new reference uses four sequential native PyMC chains on one CPU, starting with 2k tuning/2k retained draws per chain. Only the one already declared 6k/12k escalation is available when initial chain diagnostics fail; numerical failure does not authorize escalation. The inner reference budget is 900 CPU/wall seconds and the outer process cap is 950 wall seconds including overhead. The controller permits at most two concurrent reference processes, 3,900 seconds for the whole reference campaign and 6,300 summed **inner** CPU seconds. A cap or unresolved reference leaves the comparison inconclusive; there is no automatic retry or budget extension.

## One new six-target estimator

The output is exactly `initial_variance, kappa, theta, vol_of_vol, rho, c`: five physical parameters plus one scale. Reuse the immutable 20k physical-row clean-price bank and current spline family, six coupling blocks, 128-by-128 conditioners, batch 128 and effective dropout zero. Different six-dimensional weights from the old five-dimensional model are unavoidable; no cross-dimension identity claim is made.

For each training exposure, draw one c independently from Uniform(.5,2), then apply the same Gaussian quote-error shape to `clean_price + c*h*Z`. Keep the c RNG namespace separate from the preserved physical-row/quote-error schedule. The input contains only the 54 ordered `Y/h` values. The generating c, `c*h`, clean prices and physical truth are not inference conditions. The new c target is not an old tau column reused under another name.

The single fit has predeclared 10k/50k/60k endpoints. Its schedule matches the current controlled recipe: the original 10k cosine schedule, then 5e-5 through step 49,999, 5e-6 for steps 50,000–54,999 and 5e-7 for steps 55,000–59,999. Retain Adam state across rate changes. The fit/reload wall bound is 900 seconds, with no automatic extension; the source-bound training protocol supplies the executable identity and limits. This document does not authorize another seed, capacity change or fit after a failure.

The population evaluation uses the first 512 declared validation physical rows, fresh independent c values from its frozen namespace 3, and the paired Gaussian-error seed. It evaluates 512 posterior draws per surface. Rank/coverage checks concern the full declared six-parameter prior-predictive population. There is no uniform-rank assertion conditional on a fixed physical truth or on truth-c bins. All reserved test members remain unopened.

## Acceptance scope and interpretation

- Pricing retains the strictest measurement gate `.02 * (.5*h)`, independently of assumed/inferred c. Larger inferred scatter cannot relax numerical accuracy. Reference readiness needs parameter/product convergence and sampled pointwise refinement checks; it is not a prior-wide numerical certificate.
- Conditional comparisons retain chain/draw structure and Monte Carlo-aware means within .3 reference SD and central-90% width ratios [.75,1.25]. The declared family has 144 checks: two metrics, eight quantities (six parameters and two analytic variance products), three cases and three endpoints. Missing, failed or capped references are inconclusive.
- Report all endpoints, physical/c recovery, prior-predictive ranks/coverage, paired parameter dependence and the two variance products. Passing population calibration alone cannot establish conditional accuracy or accept a checkpoint. The two synthetic conditional cases are development evidence, not untouched acceptance data.
- The market case has no generating physical truth or scale. It can test neural agreement with the specified joint reference, while market-model adequacy remains separate. Report c marginals, prior comparison, mass below .55/above 1.95 and physical–c correlations descriptively.
- A posterior pressing against c=2 can indicate inadequate residual-scale support, mean-model error or data/prior conditions under this truncated working model. It is not an estimate of pure measurement noise, proof of neural error, or permission to widen the prior retrospectively. Persistent strike/maturity residual structure remains relevant even with an inferred global scale.
- Stop with an explicit development outcome and reflection. This study does not promote tutorial assets, certify the old c=1 estimator, accept a revised grid or authorize Bergomi production.

## Grid reminder: unchanged in this study

| Stage | Locations and coordinates | Scope |
| --- | --- | --- |
| Original published Heston | 63 contracts: seven nominal maturities `[5,10,21,42,63,126,252]/252`, nine symmetric targets `[-2,-1.5,-1,-.5,0,.5,1,1.5,2]` at fixed 20% | Historical normalized complete-grid task, with kappa fixed |
| Current pilot | 54 actual contracts: six expiries, selected once near ATM-scaled `[-4,-3,-2,-1.5,-1,-.5,0,.5,1]`, then frozen | Already put-heavy; 21.03–324.99 calendar days with exact ACT/365F. These are the contracts used here |
| Audited E candidate | 63 actual nearest contracts at seven expiry buckets; fixed 16% targets `[-5,-3.5,-2.25,-1.25,-.5,0,.5,1,2]` | Data audit only: 62/63 targets supported, no one-year PM expiry. Explicit final-call +1.5 declaration supports the same physical contracts but does not meet the original endpoint/+2 criteria |

The candidate extends short-expiry and wing coverage relative to the current 54; it is not the first asymmetric design. Reference-volatility coordinates differ: original ±2 at 20% equals ±2.5 at 16%, so E does not extend the original positive wing at a matched maturity. See the [raw coverage audit](../grid-coverage/README.md). No actual contract, half-spread, carry estimate or maturity changes during this noise-scale study; new-grid labels and inference remain a separate later stage.
