# Heston observation baseline: single spread scale, then quote coverage

**Status, 10 November 2026: implementation active; neural and grid acceptance gates remain closed.** Establish a verified Heston baseline before adapting the Bergomi notebook. First resolve the observation model on the existing frozen 54-contract European SPXW snapshot. Then test one revised fixed quote layout. Separately develop varying locations, summaries and masking in the flexible-input Heston plan. Planning and independent data/numerical audits may run in parallel; later scientific targets do not inherit acceptance from earlier software checks.

This plan records the decisions reached after reviewing [PR #81](https://github.com/fmuia/sbi4finance/pull/81), at `df2cfc442de07459a8b971f070e3a6f25b0b2020`. Adopt its coverage principle as a candidate to test. Its Gaussian-process discrepancy, changed tau prior, four-day reader, exotic-product suite and prudent-value discussion are separate later decisions. Opening this plan does not merge PR #81 or publish a checkpoint.

**Implementation pointer: [PR #85](https://github.com/fmuia/sbi4finance/pull/85).** The single-component `c*h` likelihood with fixed c=1 is implemented and checked. The initial neural recipe, matched zero-dropout control and exact-state continuation to 50k updates remain unsuccessful on matched conditional-reference checks despite improvements. Three c=1 references—broad synthetic, market-like synthetic and actual market—are ready; the boundary case reached its time cap and remains inconclusive. The bounded lower-learning-rate comparison from 50k to 55k/60k completed in about 100 seconds: loss improved and the 60k population rank checks passed 5/5, but both endpoint conditional panels still fail. That five-target development round stopped for reflection without launching another fixed-c fit.

The user subsequently prioritized examining the single noise scale on the unchanged 54 quotes, five physical priors and observed half-spread profile. This is a bounded exception to the planned fixed-c-first sequencing in A1, not a relaxation of its acceptance gates. Fixed `.5/1/2` sensitivity references on the same observations are ready for both synthetic cases, with physical interval widths approximately halving/doubling relative to c=1. One joint six-target fit on the existing 20k physical rows completed 60k updates in 480.48 seconds. Its final six marginal population SBC checks pass, but the median central-90% width for c is 1.336 versus the prior's 1.35 and truth recovery remains weak. The broad joint reference's central-90% c interval `[.938,1.315]`, against the network's near-prior marginal, establishes a conditional approximation failure. All seven new references are ready, but both synthetic and the separate market conditional panels fail at every predeclared neural endpoint. The completed study stops for reflection; PR #85 and its noise-scale report retain the detailed results. Neither fixed nor joint inference is accepted.

The parallel frozen-data grid audit finds 62/63 literal E quote targets supported and no one-year PM expiry. Its explicit last-call +1.5 revision changes the target declaration, not the selected physical contracts, and still misses the original endpoint/+2 goals. No new-grid labels or fit have been produced; the grid-production gate is closed. Flexible-input B1/B2/B3 training has not started, and Bergomi production remains dependent on an accepted fixed-layout Heston baseline. Detailed results and failure records belong to PR #85 rather than this planning document.

## Purpose and success criteria

The teaching question is what observed European option prices identify under a declared error model, and how uncertainty affects an unobserved quantity. A trustworthy conditional posterior is the first deliverable. Wide intervals, decreasing training loss and a close least-squares fit are not acceptance criteria on their own.

Keep separate judgments for software correctness, numerical pricing accuracy, neural/reference agreement, synthetic calibration and actual market-model adequacy. Heston can fail to fit a real chain even when inference is correct. Report that result rather than inflate a noise scale until the market looks well explained.

## A1. Freeze the observation law and establish fixed c

For each quote let `h_i = (ask_i - bid_i)/2`, in the same price units as the midpoint and model price. Use one error component:

`Y_i = P_i(phi) + c * h_i * Z_i`, with independent standard-Normal `Z_i`.

- Core target: five physical Heston parameters, with **c fixed and known at 1**. Keep the actual 54 locations, observed half-spread profile, carry, clock and physical prior fixed for this comparison.
- Physical prior: independent Uniform initial variance/theta `[.01,.16]`, kappa `[.1,10]`, vol-of-vol `[.1,3]`, rho `[-.95,-.1]`. Preserve boundary checks; changing the prior creates a separate target.
- Fixed-scale sensitivity: apply c values `.5,1,2` to the **same observed prices**, initially through matched reference inference. A network trained only at c=1 does not automatically support another c. A known-c-conditioned network, if needed, requires training over its declared c range and a separate identity.
- Paired later target: infer the same global c jointly with phi, using an explicit illustration prior `c ~ Uniform(.5,2)`. The network must not receive its generating c, `log(c*h)`, or another feature containing it. Known h and geometry remain conditions. Inferred-c synthetic calibration must draw c from this actual prior.
- The fixed observed spread profile is a deliberate restriction. Global rescaling does not establish robustness to arbitrary relative changes in spreads across strikes/expiries. Spread-profile variation is deferred support work, not an extra inferred variance component.

The normalized likelihood includes `-n*log(c) - R(phi)/(2*c*c)`, where `R = sum(((y-P(phi))/h)^2)`, plus the c-independent constants. At fixed phi, the Uniform-c conditional mode is `sqrt(R/n)` clipped to the prior interval. Use this as an oracle, not a joint-posterior formula. Fixed-c sensitivity and posterior marginalization over c answer different questions.

### Contrast with previous models and software conventions

| Model | Gaussian price SD | Interpretation |
| --- | --- | --- |
| Original published Heston | Fixed synthetic IV profile, converted with Black vega at 20% | Known independent Gaussian price noise; four physical targets with kappa fixed |
| Completed tau pilot | `sqrt((c*h_i)^2 + (tau*market_vega_i)^2)` | Known measurement multiplier plus inferred IV-equivalent extra scatter |
| Adopted baseline | `c*h_i` | One spread-shaped residual profile, first known then inferred globally |

The new core returns to the original known-Gaussian model class but replaces its prescribed SD rule. Inferring c adds one amplitude under that same law; it does not separate measurement noise from model discrepancy. An inferred c may absorb model mismatch and prior/data issues. Boundary concentration is not automatically neural failure; check the reference posterior first. Half-spread-as-Gaussian-SD and the illustration prior are conventions, not empirical industry estimates.

Document the primary-source software comparison: QuantLib offers price/relative-price/IV errors and user weights; Strata accepts supplied calibration errors; QuantFlow exposes alternative Heston residuals and weights; LSEG and Zeliade document other surface fit/weighting choices. These establish configurable calibration objectives, not a unique Bayesian noise law. Relative weights do not determine absolute posterior uncertainty. Preserve exact cited versions/access dates and source URLs in the report; do not claim another package validates this likelihood or our intervals. The source-verified companion PDF renderer is in the prior Heston study branch.

Primary sources checked for the discussion report on 9 November 2026: [QuantLib residuals](https://github.com/lballabio/QuantLib/blob/master/ql/models/calibrationhelper.cpp), [QuantLib weighted objective](https://github.com/lballabio/QuantLib/blob/master/ql/models/model.cpp), [Strata supplied errors](https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/option/RawOptionData.html), [Strata SABR price-space use](https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/capfloor/SabrIborCapletFloorletVolatilityCalibrator.html), [QuantFlow Heston](https://quantflow.quantmind.com/api/options/calibration/#quantflow.options.calibration.heston.HestonCalibration), [LSEG SVI settings](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti), and [Zeliade eSSVI](https://www.zeliade.com/wp-content/uploads/whitepapers/zwp-008-RobustNoArbSSVI.pdf). Strata's cited application is rates SABR and LSEG/Zeliade concern surface fitting; these are examples of different calibration choices, not interchangeable Heston likelihoods.

## A2. Coverage revision after the noise control

The pilot selects actual quotes once using observed ATM-scaled targets, then freezes them. It has six expiries from 30 November 2026 to 30 September 2027, 21.03-324.99 calendar days under ACT/365F. In a common fixed-16% coordinate `z=log(K/F)/(.16*sqrt(T))`, its deepest puts are approximately -3.69 to -4.29, nearest ATM has `abs(z)<=.011`, and furthest calls only +.93 to +1.07. It lacks a one-week expiry and +2 calls.

Start from the PR's layout E candidate: approximately 1 and 2 weeks and 1,2,3,6,12 months, with nine targets `[-5,-3.5,-2.25,-1.25,-.5,0,.5,1,2]` at fixed 16%. These are **desired quote locations**, not selected contracts or a guarantee of 63 usable quotes. Select distinct actual European PM-settled SPXW quotes after quality filters. Record realized distances, spreads, unsupported targets and provenance. Revise unsupported targets explicitly; do not interpolate artificial observations or silently substitute AM contracts.

For historical comparisons, original +/-2 at 20% equals +/-2.5 at 16%. E extends original put reach but has only 80% of the original positive log-moneyness reach at a matched maturity. Added +2 calls extend the current 54 design, not the original call endpoint. Relabelling coordinates changes no information; changing physical contracts does.

PR evidence is local and conditional: the layout-comparison script uses fitted per-expiry ATM scales, whereas its written candidate uses fixed 16%, and the Fisher comparisons assume smooth error. It does not establish an optimum under `c*h`; E also worsens rho widths in its two reported scenarios. Evaluate the literal actual-quote candidate under the intended likelihood on a small representative panel before production labels.

Freeze actual strikes, exact ACT/365F settlement times and quote types. Keep common carry conditioning over the union of expiries, preserving existing anchor estimates where unchanged; new anchors stay disjoint from fitted/scored quotes. Keep product horizons identical across comparisons. The comparison of 54 and about 63 tests a coverage package, not strike placement alone.

## Execution units, dependencies and bounded compute

| Unit | Work | Gate |
| --- | --- | --- |
| H1 | Implement one-scale likelihood, task/input identities and conditional-c oracle; capture first nonfinite training loss/gradient/state | Meaningful tests of density normalization, prior/transform handling, no c leakage, valid negative Gaussian observations, immutable old producers |
| H2 | Reuse verified unchanged 54 clean prices; prepare fixed synthetic cases, numerical checks and fixed-c reference | Confirm design/physical-prior/payload hashes and precision at the new scale; no reserved test access |
| H3 | Train fixed c=1, compare synthetic conditional references; then add the inferred-c pair | Fixed checkpoints, finite state, MC-aware agreement and declared failures preserved |
| H4 | Audit/select revised actual quotes and price only changed contracts on paired physical draws | Adequate support, stable pricing, measured cost; new bank/design identity |
| H5 | Repeat accepted likelihood recipe on the revised layout; assess full/restricted information and predictions | Separate approximation, calibration, information and market-fit findings |
| H6 | Freeze recipe, fresh acceptance/seed confirmation, reflect and authorize artifact promotion separately | Baseline evidence supports Bergomi port; flexible-input experiments remain separately versioned |

Initial engineering probes are limited to a representative panel of at most 16 physical points and three noise scales. Initial training has one c=1 fit with predeclared 2k/10k checkpoints, at most 60 minutes including development checks. The subsequent inferred-c fit has the same initial budget after its fixed-c control is useful. Do not repeat the old 88k attempt automatically. A failed conditional screen prompts a recorded diagnostic decision before longer training or architecture changes.

For each initial target, predeclare three synthetic development cases (broad-prior, market-like and boundary/low-noise) before viewing network results, plus the actual market application. Start reference smoke runs separately from acceptance; use four chains, initially 2k tuning plus 2k retained draws, with one declared escalation to 6k tuning/12k retained draws if needed. Initial reference spending is capped at 45 CPU-minutes per case; hitting a cap records an incomplete reference, not convergence. Actual measured timing may require an explicit budget revision before launching the panel. Keep fixed-c sensitivity separate from extra neural fits.

New-grid generation is gated by the representative-panel result and measured cost; declare row count, worker count and wall-time bound before launch. Reuse unchanged physical rows/quote errors where valid. Do not manufacture new geometry by relabelling prices. Large payloads remain outside Git, bound by hashes. A bank may be reused for a new observation law without claiming its old network targets the new posterior.

## Numerical, inference and information gates

- Retain the audited extended Heston quadrature ladder, including `(16384,9600)`. Gate refinement against measurement uncertainty alone: default maximum change `.02 * (.5*h) = .01*h`, independently of an inferred c/tau. Verify posterior-point likelihood sensitivity and an alternative integrator; the same formula with another quadrature is not an independent model implementation.
- Reference acceptance: rank-normalized R-hat <=1.01 and bulk/tail ESS >=400 for parameters and reported products. Record chain/draw structure, actual sampler and initialization, source/environment/task identities, failures and timing.
- Neural development comparisons use existing MC-aware margins: means within .3 reference SD and central-90% width ratios [.75,1.25], plus parameter dependence and analytic variance-product checks. Average loss or prior-average coverage alone cannot pass this gate.
- Synthetic SBC/coverage and fresh seed confirmation use the exact declared target. The previously reserved 2,000-case final pool remains unopened during development; freeze selection before fresh acceptance data.
- For full/restricted observations, remove only declared smile prices while retaining identical surviving prices/errors and common explicitly stated carry conditions. No hidden prices may leak through curve estimates, preprocessing or normalization. Use one fixed-horizon forward-average-variance quantity, the near-term control and withheld vanilla predictions.
- Existing tau checkpoints failed own-simulator conditional checks; smoother market residuals do not establish the cause. Preserve that evidence and the unresolved first nonfinite event rather than rewriting it as validation.

## Parallel work and publication

Now: H1-H3 engineering, the H4 raw coverage audit, flexible-input specification and Bergomi readiness inspection can proceed independently. H4 production labels wait for accepted noise/precision decisions. Bergomi production pricing/training waits for Heston H6; its numerical acceptance is independent. The later Heston campaign follows geometry -> summary -> masks, starting with fixed c and repeating its successful final recipe with inferred c.

Planning PRs carry specifications only. Implementation PRs remain compact and preserve historical producers/identities. Update this plan and compact evidence in place; no new handover document per run. Notebook changes retain live-training and pretrained routes and describe fixed-layout scope honestly. Tutorial assets enter only through native acceptance tools; no checkpoint or lesson is promoted because a plan was merged.

Dependencies: [flexible-input Heston plan](HESTON-FLEXIBLE-INPUTS-PLAN.md), [Bergomi observation refresh](BERGOMI-OBSERVATION-REFRESH-PLAN.md), [existing sparse support requirements](SPARSE-QUOTE-CASE-PLAN.md), and [MCMC policy](mcmc-policy.md). The two sibling plans are supplied by companion planning PRs; their files become available together after merging those plans.

The completed pilot and failed continuation are preserved at [study commit 78cf9de](https://github.com/fmuia/sbi4finance/tree/78cf9de/lab/evidence/pilot-results/2026-11-09/heston-observations). This is historical development evidence, not an accepted notebook model. Implementation should port only required support from that branch, rather than include its unrelated rough-notebook work or reinterpret its tau checkpoints.
