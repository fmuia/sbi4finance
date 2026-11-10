# Calibration from sparse option quotes: Heston verification first

**Status, 9 November 2026: Heston pilot explicitly reprioritized by the user.** Verify the revised European-option observation/design setup with Heston and stop for reflection before any rough-Bergomi extension. The pilot has a distinct task identity and does not promote its results or checkpoints into the lessons. General sparse adapters, variable-length sets and production release changes remain outside this pilot. Historical requirements and evidence below remain relevant where consistent with this bounded specification.

## Proposed combined follow-up after PR #81 review

This is the proposed next Heston execution specification, not a report of new runs. It combines the user's known-Gaussian/direct-sigma discussion with the coverage ideas in [PR #81](https://github.com/fmuia/sbi4finance/pull/81), reviewed at **df2cfc442de07459a8b971f070e3a6f25b0b2020**. The companion noise PDF explains the original, current tau and proposed single-scale models. Existing tasks, banks, checkpoints and evidence remain under their original identities. No PR merge, new pricing bank, training or Bergomi work follows automatically from this review.

### What to adopt from the PR

- Select contracts for useful coverage rather than increase density alone: deep puts, ATM, near calls and calls near +2 standardized log-moneyness; include short as well as later expiries.
- Keep a small fixed layout and actual European SPXW observations for the first implementation. A set encoder and an interpolated multi-day reader are unnecessary for this unit.
- Use one shared observation/design specification for simulation, deterministic calibration, reference inference and neural inference. Keep numerical pricing accuracy separate from observation uncertainty.
- Preserve the distinctions between synthetic calibration, conditional neural/reference agreement, market-model adequacy and downstream product uncertainty.

The actual pilot's 54-quote design has these coverage properties, measured in the PR's fixed-16% coordinate `z = log(K/F)/(0.16*sqrt(T))`:

| Feature | Existing 54 | Implication |
| --- | --- | --- |
| Expiries | 30 Nov 2026 to 30 Sep 2027; 21.03-324.99 calendar days | No one-week expiry; only one within a month; final expiry about 10.7 months |
| Deepest puts per expiry | z approximately -3.69 to -4.29 | Passes the proposed below-minus-3 coverage criterion |
| Nearest ATM | absolute z at most 0.011 | ATM coverage is adequate |
| Furthest call per expiry | z approximately +0.93 to +1.07 | No +2 call at any expiry |

Thus retaining the exact existing contracts requires relaxing the PR's criteria. Retain them as a useful control, not as an already adequate final grid. Fifty-four is not inherently inadequate: a redesigned 6-by-9 layout is possible. The PR's 7-by-9 layout E is a simple candidate, not a required point count or a proven optimum.

### Grid candidate and fair comparison

Start the coverage audit from layout E: approximately 1 and 2 weeks, 1, 2, 3, 6 and 12 months, with targets `[-5,-3.5,-2.25,-1.25,-0.5,0,+0.5,+1,+2]` in fixed-16% standardized log-moneyness. Select distinct actual quoted contracts near those targets. Record realized strike/expiry distances, spreads, roots and unsupported targets; revise the declared design rather than fill gaps with interpolated pseudo-observations. If approximately one-year SPXW coverage is unavailable, disclose the actual endpoint or revise the snapshot/design; do not silently substitute AM settlement conventions.

Freeze the physical strikes, actual ACT/365F maturities, carry anchors and quote provenance before production training. Revised expiries need additional carry anchors: define one common conditioning record over the union of expiries for both grids, preserving the old anchor estimates on unchanged expiries and keeping new anchors disjoint from fitted/scored quotes. A fixed reference volatility is only a coordinate convention. Selecting contracts once using observed ATM information and then freezing them does not require adaptive strikes for every simulated parameter draw.

The PR's E63 evidence is preliminary: `layout_choice.py` calls a grid constructed with fitted per-maturity ATM levels (14.30%-17.25%), whereas the written specification uses fixed 16%. The resulting log-moneyness offsets differ in scale by about -10.6% to +7.8%; those are not percentage changes in the physical strikes. Its Fisher comparisons also use a fixed smooth-error covariance at one fitted SPX point; they do not establish the ranking under our proposed single-scale likelihood. Its reported E layout can worsen local rho widths by 9% or 31% in its two scenarios. Recheck the literal candidate under the intended likelihood on a small representative panel. Do not claim that comparing 54 versus 63 isolates placement alone: expiry coverage and point count change too.

### One Gaussian family, two inference questions

Use `Y_i = P_i(phi) + sigma * h_i * Z_i`, with known frozen price half-spreads, independent standard-Normal quote errors and the same five-parameter physical prior in every condition. This returns to the original known-Gaussian model class for the core example, replacing its prescribed noise profile with a spread-anchored convention.

- **Known scale:** sample sigma in [0.5,2] as a supplied condition; infer only the five physical parameters. The main assumed scale is 1, with sensitivity at 0.5, 1 and 2 on the **same observed prices**.
- **Unknown scale:** use the explicit illustration prior `sigma ~ Uniform(0.5,2)`, independent of the physical parameters, and infer the six-dimensional joint posterior. Here sigma is a dimensionless SD in spread units. Hide its generating value and any feature containing it from network conditions; supply only the known spread profile and design.
- Use matched synthetic truths and price observations within each grid to compare these questions. The unknown-scale marginal uses posterior weights over sigma, not an equal average of the fixed-scale sensitivity plots. Fresh datasets for synthetic calibration are a separate experiment.
- Keep half-spread-as-Gaussian-SD and the scale prior visibly labelled as conventions. A large inferred scale or boundary concentration can reflect model/data/prior mismatch, not a measured pure observation-noise process. The [noise proposal PDF renderer](../scripts/render_heston_simplification_report.py) specifies the conditional-sigma likelihood and oracle.

The bounded development comparison has four conditions:

| Contract design | Known Gaussian scale | One inferred Gaussian scale |
| --- | --- | --- |
| Existing 54 contracts | Control using preserved clean labels | Isolate scalar scale inference on that control |
| Revised fixed actual-quote layout | Test the coverage upgrade | Combined candidate |

Use paired physical parameter rows and the same layer-width/batch/optimizer recipe where compatible; reuse quote errors on identical contracts when pairing designs. Give each target/design its own identity. Stage this screen: establish both Gaussian arms on the reusable 54-contract bank first, then repeat on a revised design after its coverage and pricing gates pass. Declare update/checkpoint and wall-time limits before launching each pair; a failure prompts diagnosis rather than automatic longer training or a wider search. This is a development screen, not a claim that all changes are individually isolated.

### Execution sequence and gates

| Phase | Work | Required evidence before advancing |
| --- | --- | --- |
| 1. Lock the specification and failure capture | Audit actual quote coverage; declare one revised candidate and both Gaussian targets. Record individual losses, gradient/parameter finiteness and state at the first non-finite event. | Reproducible design/target records; no sigma leakage; finite toy likelihood/conditional-sigma checks. |
| 2. Check the grid and pricing | Evaluate the literal candidate under known spread-based scales on a small prior/market-like panel. Preserve the audited Heston ladder and measurement-only precision gate. | Realized coverage, manageable conditioning, successful price refinement and a bounded measured cost. |
| 3. Prepare matched clean labels | Reuse the existing bank for unchanged 54-contract tasks. Reprice once for changed contracts, using paired physical rows, and create a new bank identity. | Unchanged producer/old payloads; a precision-checked new bank if needed. No relabelling of old labels as new strikes or maturities. |
| 4. Establish references and run the four-cell screen | New matched five/six-target references and bounded training runs with fixed checkpoints and logged probes. Do not infer success from decreasing loss alone. | Rank-normalized R-hat <=1.01 and bulk/tail ESS >=400; conditional means/widths and analytic variance products compared with Monte Carlo uncertainty. Use existing repository margins, including 90% width ratio [.75,1.25], not PR point-only replacements. |
| 5. Information and market checks | On the useful recipe, compare full/restricted quote information and hold-out predictions. Keep the current near-term average-variance control and affected forward-average-variance product first. | Separate evidence for approximation fidelity, simulator calibration and residual/model adequacy; no mandatory wide intervals. |
| 6. Freeze and reflect | Freeze the recipe before fresh acceptance data/seed confirmation and any lesson promotion. Keep the reserved final pool unopened during development. | Explicit acceptance or a recorded failure. Stop for reflection before Bergomi or broader product work. |

The old 54-contract bank cannot be reused unchanged after supplementing/replacing contracts or changing the clock. The PR incorrectly describes pilot maturities as whole business days: they are fractional ACT/365F. Its proposed clock, carry estimation and alternate Fourier ladder are additional changes, not covered by the old bank's provenance. Keep those conventions fixed in this execution unit.

### PR ideas to retain as later studies

Smooth Gaussian-process discrepancy is a reasonable model-adequacy hypothesis, not the next baseline or an established diagnosis of the neural failure. The PR also changes tau from Uniform(0,2 volatility points) to LogUniform(0.2,5), profiles correlation lengths on the fitted data and adds multi-day interpolation. That is substantially more than changing grid locations. A correlation study needs matched amplitude priors, explicit hyperparameter conditioning/sensitivity and checks on raw quote residuals; interpolated fitted residuals alone do not identify the covariance law.

The proposed V4b attribution requires success on the network's own simulator, which our completed checks already fail: all six cases fail collective agreement, and tau alone fails five with one inconclusive at 75k. Smooth market residuals do not explain that finding, or establish the cause of the 88k non-finite event. Resolve matched synthetic fidelity first.

Retain posterior product propagation as the eventual purpose, but defer the forward-start/barrier/volatility-swap suite, book aggregation, four-day reader and prudent-value terminology. Conditional product quantiles are not automatically regulatory reserves. Finite-strip variance replication is a useful market-informed benchmark with tail/convention uncertainty, not an independent observed variance-swap quote. For future Bergomi work, a fixed-seed importance sampler is an approximate numerical reference, and marginal price-SE tolerances do not control every contrast direction under smooth covariance. Those require a separate specification and numerical audit.

## Completed Heston pilot and prior continuation

**Heston verification completed; reflection gate reached.** The [run record](../evidence/pilot-results/2026-11-09/heston-observations/README.md) contains the frozen SPXW designs, 11 converged reference cases, numerical and carry audits, and four neural development endpoints. The forward-variance information effect is visible but prior-sensitive (short/full interval-width ratio 6.49 with vol-of-vol capped at 1.5; 3.11 with cap 3). Structured market residuals remain. Neural conditional verification fails at both 2,000 and 10,000 updates, despite much better prior-average coverage at the latter budget. Pricing/software checks passed; the revised market/neural task is **not accepted for release**. Stop here before choosing final discrepancy/prior/carry assumptions or running Bergomi.

## Continuation: full-panel network training study

Following the reflection and training discussion, the user requested continuation. The next bounded step improves the full 54-quote Heston network. The short panel, wider architecture search, production acceptance, lesson publication and all Bergomi work remain deferred.

**Initial continuation completed; longer optimization requested.** Both 50,000-update trajectories and all four predeclared checkpoints are saved. The larger bank's small physical-NLL advantage is unresolved on 512 paired observations, and all four checkpoints fail the MC-aware market and six synthetic conditional comparisons. Discrepancy-scale inference remains the clearest weakness; some other intervals are too narrow. All six synthetic references converge, numerical audits pass, and the reserved 2,000-case pool remains unopened. The loss is still decreasing at 50k without a growing train/validation gap, so the evidence does not yet identify capacity as the limitation. See the [updated run record](../evidence/pilot-results/2026-11-09/heston-observations/README.md#full-panel-training-continuation).

**Longer optimization tested; reflection gate reached again.** We resumed only the 100k-surface model from its saved 50k state toward 100k total optimizer updates, preserving model/standardization state, Adam moments, absolute shuffle/noise numbering, dropout streams and the original 50k cosine schedule (thereafter its 5e-5 floor). Independent resumed-versus-uninterrupted smoke runs match exactly. The saved 75k checkpoint improves physical validation NLL by .09324 nats per observation (paired SE .01985), but market tau mean error worsens from -6.63 to -7.27 reference SD, several physical intervals narrow further, and collective conditional agreement still fails on all seven completed reference observations. The run then stops at the 88k monitoring check for a non-finite loss; last finite probes are at 87k, and no 100k checkpoint exists. The failure guard does not preserve the offending values or state, so its cause is unresolved. All 618 lab tests pass; no reserved cases were opened. See the [continuation results](../evidence/pilot-results/2026-11-09/heston-observations/README.md#additional-optimization-from-50k).

The evidence supports the user's request to test longer training before attributing the problem to capacity. It also shows that improving the average objective does not guarantee better conditional inference. Preserve the failed attempt; the next implementation step would add precise non-finite loss/state capture for a bounded diagnostic replay. No plateau or capacity limitation has been established. Stop for reflection before choosing that intervention, changing optimizer/architecture/conditioning, or starting short-panel/Bergomi work.

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
