# Calibration from sparse option quotes: Heston verification first

**Status, 9 November 2026: Heston pilot explicitly reprioritized by the user.** Verify the revised European-option observation/design setup with Heston and stop for reflection before any rough-Bergomi extension. The pilot has a distinct task identity and does not promote its results or checkpoints into the lessons. General sparse adapters, variable-length sets and production release changes remain outside this pilot. Historical requirements and evidence below remain relevant where consistent with this bounded specification.

**Heston verification completed; reflection gate reached.** The [run record](../evidence/pilot-results/2026-11-09/heston-observations/README.md) contains the frozen SPXW designs, 11 converged reference cases, numerical and carry audits, and four neural development endpoints. The forward-variance information effect is visible but prior-sensitive (short/full interval-width ratio 6.49 with vol-of-vol capped at 1.5; 3.11 with cap 3). Structured market residuals remain. Neural conditional verification fails at both 2,000 and 10,000 updates, despite much better prior-average coverage at the latter budget. Pricing/software checks passed; the revised market/neural task is **not accepted for release**. Stop here before choosing final discrepancy/prior/carry assumptions or running Bergomi.

## Continuation: full-panel network training study

Following the reflection and training discussion, the user requested continuation. The next bounded step improves the full 54-quote Heston network. The short panel, wider architecture search, production acceptance, lesson publication and all Bergomi work remain deferred.

Keep the existing observation law and fixed quote/vega/carry design as an **engineering benchmark**; its structured market residuals remain unresolved. Use the already audited vol-of-vol support `[.1,3]` and extended Fourier ladder, then check whether increasing the full-panel kappa ceiling from 10 to 20 materially changes the reference posterior before freezing the training target. This is not a claim that the pilot prior or independent discrepancy is scientifically final.

Generate one new source-bound bank with 100,000 clean surfaces, a separate 5,000-row validation pool and a reserved 2,000-row final pool. The 20,000-row training condition is the first 20,000 rows of the same training pool. Generate prices under the strictest supported spread multiplier .5. Reuse clean prices while independently sampling tau from its declared prior and fresh Gaussian observation errors at each batch. Tau changes the observation distribution, not the Heston prices.

Hold the spline architecture (two 128-unit subnet layers), batch size 128, seed 9011 and optimizer recipe fixed across four predeclared checkpoints:

| Training rows | Optimizer updates | Comparison |
| --- | --- | --- |
| 20,000 | 10,000 | Re-established baseline under the new support and augmentation |
| 20,000 | 50,000 | More optimization on the same physical parameter locations |
| 100,000 | 10,000 | More physical parameter locations at the same update budget |
| 100,000 | 50,000 | Both increased together |

Use two fresh 50,000-update training trajectories, one per data size, with checkpoints at 10,000 and 50,000 updates. Both use the same 50,000-update learning-rate schedule, making the optimizer comparison an identical-prefix continuation within each data size. Log fixed independent-noise train/validation probes every 1,000 updates, with explicit metric scope. No retrospective checkpoint or best-seed selection is allowed.

Use 512 new validation-pool observations for development calibration and compare every endpoint with the existing matched market reference plus six predeclared synthetic reference observations. Declare 12 synthetic cases before viewing any network outcomes: six broad-prior cases, four market-like cases, and two low-discrepancy/boundary cases. Initially run indices `[0,2,4,6,8,10]`; retain failures and do not replace hard cases. Simulated Gaussian observations may be negative; the reference accepts arbitrary finite observations around frozen valid contract/scale metadata rather than truncating or regenerating them.

Report parameter and variance-product mean/interval errors, chain-aware Monte Carlo uncertainty, prior support, fresh calibration ranks and coverage. Use the existing conditional margins (mean within .3 reference SD, 90% width ratio .75–1.25) as development checks; they do not establish acceptance on their own. The reserved 2,000-case pool stays unopened until a recipe and any seed confirmation are frozen. Do not infer correctness from average coverage or loss alone.

Bound this continuation to two fits/four checkpoints, their declared development checks, one kappa-prior sensitivity reference and six synthetic references. At most two training fits run concurrently; initial training compute is capped at two hours of wall time including both fits and their checks. Preserve incomplete/failing runs. Report the optimization/data comparison before adding network capacity, further seeds or the short-panel recipe.

## Authorized Heston pilot

Use one frozen public Cboe delayed SPX snapshot, selecting only European PM-settled **SPXW** roots. Source bytes, retrieval metadata and hashes are recorded in [the run evidence](../evidence/pilot-results/2026-11-09/heston-observations/source.json). This is intraday delayed data, with no bid/ask update timestamps. Snapshot UTC and trade-time New York assignments are explicit vendor-field assumptions; the default valuation time subtracts an assumed 15-minute delay. Exact ACT/365F elapsed time ends at 16:00 New York. Exclude same-day expiries and known half-day expiries.

The fixed expiry panel is 30 November and 31 December 2026; 29 January, 31 March, 30 June and 30 September 2027. At each expiry reserve every third paired strike within 5% of snapshot spot as a carry anchor. Estimate forward and discount using spread-weighted put-call parity. Keep these anchors disjoint from calibration and scoring quotes. Both information arms condition on the same anchors at **all six expiries**; carry is point-estimated, with uncertainty sensitivity required. Therefore the experiment removes later **smile prices conditional on carry**, not all later option information.

Compare two equal-size nine-target designs, scaling standardized log-moneyness by each expiry's observed ATM IV:

- Symmetric candidate: `[-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]`.
- Put-heavy candidate: `[-4, -3, -2, -1.5, -1, -.5, 0, .5, 1]`.

Select distinct actual OTM quotes with positive bid, noncrossed ask and half-spread/midpoint at most 25%. Retain target and realized coordinates and flag distances over 0.35. Require all put-heavy targets supported; retain the symmetric candidate's visibly unsupported targets as nearest available actual quotes, without interpolating or claiming its full intended coverage. This comparison isolates the put shift **after** ATM scaling; it is not a direct test of the historical fixed-20%-vol grid. Later ATM-based selection and frozen market-vega scales are additional common conditioning information. Predictive assessment is conditional on them.

The full arm uses 54 actual smile quotes; the short arm uses the same first 27 observations and spreads from the first three expiries. No placeholders enter the likelihood. Use independent Gaussian measurement SD `c * price_half_spread`, explicitly a convention, with `c=1` as baseline and `c=.5,2` sensitivity. Compare spread-only with `sqrt((c*h)^2 + (tau*observed_Black_vega)^2)` discrepancy, with `tau ~ Uniform(0,.02)` in decimal IV units and a wider-prior sensitivity. Freeze vegas identically in simulation and reference; this is a conditional weighting model, not an independently estimated noise law. Inspect residual structure; a scalar independent discrepancy cannot establish market adequacy.

Infer five Heston parameters jointly, including kappa (the historical lesson fixed it). Independent physical uniform bounds are v0/theta `[.01,.16]`, kappa `[.1,10]`, vol-of-vol `[.1,1.5]` and rho `[-.95,-.1]`. Report boundary behavior and prior sensitivity. Numerical error must pass a refinement gate relative to measurement SD alone; inferred discrepancy must never relax it. Audit both Fourier cutoffs and node counts over a fixed prior/corner panel, an alternative integrator and, where feasible, an independently implemented pricer.

The primary affected quantity is the analytic Heston fair strike of a forward variance swap between the third and final selected expiries (29 January to 30 September 2027). The control is expected average variance from valuation to the first expiry. Both are conditional model fair values in variance units, not market reserve recommendations. Later vanilla quote prediction supplies an observable outcome check.

Use the existing native PyMC deterministic-reference workflow, preserving physical priors, chain/draw structure, exact versions and source/design hashes. Reference claims require rank-normalized split R-hat <=1.01 and bulk/tail ESS >=400. Run a bounded fixed-layout BayesFlow development study with paired clean prices, fresh noise and explicit spread inputs; assess synthetic calibration and conditional reference agreement separately from market fit. A short development run may fail this comparison; preserve the failure and do not relabel it as validation.

**Reflection gate:** record data/contract quality, pricing precision, posterior convergence, full/short and design comparisons, discrepancy/prior/carry sensitivity, neural-reference agreement and residual patterns. Explicitly identify what passed and what remains unresolved. No rough pricing, bank generation, training or lesson promotion follows automatically.

The core uses all 63 declared contract prices with valid noise/design metadata and explicitly rejects incomplete or unsupported inputs. Keep complete clean-price banks and their provenance so compatible data can be reused later. Existing thin-chain code, EWG observations and seven-level results remain historical evidence; they do not implement sparse Heston or the new six-target rough task.

## Specify the problem before implementing it

The follow-up should first agree a short specification covering these decisions:

| Decision | What must be explicit |
| --- | --- |
| Learning question | Which missing information affects which parameter, curve quantity or downstream product? Quote count alone does not define thinness. |
| Supported observations | One fixed subset, a declared family of masks, or variable-length raw quotes? Declare strikes, maturities, option types, valid noise metadata and behavior for unsupported inputs. |
| Observation provenance | Actual quotes versus interpolated grid values; coverage flags, forward/discount/exercise conventions and dependence introduced by preprocessing. Never count extrapolated values as newly observed evidence. |
| Missingness and noise | Fixed missingness by design versus liquidity/selection-dependent missingness; which mechanism is simulated or conditioned on, and what reliability/selection claims are justified. Vary missingness and noise separately. |
| Training and input contract | Separate fixed-layout checkpoints versus one estimator trained across masks/designs. Declare coordinate/noise encoding, missing/padded representation and the training support; an encoder choice alone is not a support guarantee. |
| Matched comparisons | The same available quotes, surviving quote errors, prior, numerical model and inferred targets for neural inference, MCMC and audits. A forward emulator may predict a full surface, but the likelihood uses only the observed subset and its relevant covariance. |
| Acceptance and cost | Conditional reference agreement, calibration, invariance checks where relevant, prior/curve sensitivity and downstream product error; separate development from fresh acceptance evidence and declare a bounded compute budget. |

Absent quotes must contribute no artificial observation or likelihood term. If zero or another placeholder is used internally, a validated mask must exclude its value; it is not a zero-price observation. Any interpolation remains a declared preprocessing approximation rather than a missing-data solution. If missingness depends on prices or spreads, random masking alone does not demonstrate robustness to that selection process.

## Candidate first experiment, after specification

A small controlled full-versus-sparse comparison on the supported grid is the simplest starting point. Begin with Heston, where refined Fourier-likelihood MCMC can check the neural answer. One predeclared subset missing later maturities is a candidate; narrow wings or dispersed missing quotes are alternatives, not mandatory extra layouts. A fixed-layout adapter/checkpoint may be sufficient if that is the agreed support contract. Do not assume the current rough-only thin trainer already provides Heston support.

Use paired synthetic base surfaces, preserving the generating parameters and quote errors for surviving contracts. Refine complete clean prices under the original numerical checks, then select prices, noise and covariance consistently. Subsetting must not bypass pricing-precision checks. Reusing compatible prices avoids new generation, but does not make a full-surface checkpoint support sparse inputs.

Extend to rough Bergomi only after that control is useful. Match the full task's joint prior, numerical model and unknowns; with the current baseline these are H, eta, rho, short variance, long variance and decay. Both full and sparse comparisons infer the curve. Do not supply a curve reconstructed from additional, withheld quotes. If the core promotes another curve family, the sparse specification must follow that actual task rather than silently keep six targets. Old seven-level curves/prices cannot be projected or relabelled as the new task without repricing.

Judge what the data identify through posterior contraction, conditional reference agreement, calibration and product uncertainty. Weakly quoted curve regions may be constrained mainly by the compact curve family and prior. Wider intervals alone do not prove better inference; narrow intervals can reflect restrictive shape/extrapolation assumptions. Select one affected downstream product and one well-supported control before inspecting final results.

For a mask-conditioned or variable-length approach, train over the declared support and verify that changing masked values or adding padding leaves inference unchanged. Set encoders must also preserve permutation invariance of complete quote records. Arbitrary new strikes/maturities need corresponding simulation and validation designs. These are follow-up requirements only if that representation is chosen, not work to add to the core now.

## Market illustration and historical evidence

Choose a market case only after the controlled design and claim are fixed. EWG is a historical candidate, not the task definition or a required final ticker. Select for relevant coverage, reproducible dated quotes, coherent preprocessing/noise conventions, feasible reference/audit checks and a clear product story, before inspecting method rankings. A synthetic experiment is sufficient if no suitable market case is available.

Preserve EWG's original observations, seven-level outcomes, source labels, 27-point proposal layout and distinct 61-quote audit. A new curve family, observation design or ticker requires reviewing the checkpoint, audit, product horizons and continuation assumptions, not just swapping a layout file. Existing flat/widened-last-increment studies do not transfer unchanged to the compact curve prior.

A real market chain has no known parameter truth. EWG-posterior-conditioned experiments establish local sensitivity, not broad market performance. Population reserve-coverage and wrong-extrapolation studies require their own declared truths, continuation laws, products and acceptance criteria; do not inherit them as mandatory work. Keep successful prior-generated cases and misspecified alternatives together when making robustness claims.

**A quick check, 28 and 29 October, on request.** The six-target task was run on EWG's fixed 27-point layout, on one machine, as evidence for this specification, not as its implementation (`lab/checks/why_bayes/`, Section 5; course Lecture 16, Section 16.4). The bank (24,000 surfaces, built by PR #48's bank builder) and the network stay off the repository, with their SHA-256 recorded. On 100 prior-generated chains, least squares without a prior over-reserved, $H$ fixed at 0.10 was biased, the flat continuation was biased beyond the last quote, and a Laplace approximation with the posterior's prior matched the network's posterior, which matched importance sampling on ten audited chains. It answers none of the decisions above: the layout is fixed, the truths come from the prior, and no misspecified alternative was run.

The historical downstream deliverable was an agreed support contract and bounded experiment, followed by implementation and evidence. The authorized Heston pilot above now implements that bounded first step. General sparse and rough-model jobs remain deferred. See the [central experiment plan](NUMERICAL-EXPERIMENT-PLAN.md) and [core protocol](EXPERIMENT-PROTOCOL.md).
