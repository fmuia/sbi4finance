# Numerical backbone for the Heston and rough Bergomi tutorials

**Delivery objective, revised 27 October 2026:** provide the reusable data, trained estimators and checked examples that the live tutorial actually loads. A working generator, a small development fit or a larger row count is an intermediate result. Each lesson needs a usable estimator and evidence that its posterior and product conclusions are trustworthy.

The implementation is in [PR 48](https://github.com/fmuia/sbi4finance/pull/48). It extends Francesco's course and existing numerical studies, whose [recorded results](../../course/status.md) remain the baseline to reproduce and improve. The [implementation guide](full-surface-calibration.md) gives the shared workflow; the [backbone work order](tutorial-backbone-work-order.md) gives the immediate actions and spending boundaries.

**Active delivery, 28 October:** known-curve **100k** and joint-curve **20k** banks are checked and complete; their intermediate 50k/20k estimators completed validation. Joint100k generation continues. The [parallel work order](backbone-delivery-work-order.md) starts the independent known100k fit, controlled architecture/summary pilot, independent pricing, current conditional PyMC comparisons and notebook integration now. The [rendered report](../evidence/reports/inference-performance-2026-10-28.pdf) records completed-model evidence and limitations. Both rough routes retain the **100k unique training-surface baseline target**. Increasing targets does not implicitly increase frozen phase budgets.

The [28 October validation unit](backbone-validation-2026-10-28.md) completed both parallel checks: Heston passed its frozen held-out calibration and analytic product gates; known-curve rough pricing detected emulator and grid errors. It records measured costs and the next 50k rough training-data milestone. No training or bank generation ran in this unit.

## The three delivered routes

| Lesson | What is inferred | What is supplied | Required backbone |
| --- | --- | --- | --- |
| Heston | Initial variance, theta, vol-of-vol and rho; kappa fixed. | Complete 63-quote observation and declared design/noise. | Reusable **100k training** bank with nested **50k/100k** snapshots and existing 5k validation/5k test pools. First reproduce the 50k baseline, then measure the benefit of 100k. |
| Rough Bergomi: known curve | H, eta and rho. | Complete quotes and seven known curve levels, held fixed within each inference problem. Curves vary across training cases. | Reproduce the existing supplied-curve task, then deliver an adequately trained estimator and bank. Initial bank planning size is **20k training / 5k validation / 7k reserved test**, subject to its own measured generation budget. |
| Rough Bergomi: joint curve | H, eta, rho and three curve coefficients. | Complete quotes and design/noise. The generating curve is not an inference input. | The existing **20k / 5k / 7k** joint collection, one posterior, one forward price emulator and the bounded refinement/reference panel. |

The first rough lesson fixes the curve **for the inference problem**, not globally across the bank. This is the user's agreed interpretation and matches Francesco's original three-target task. The next lesson releases curve uncertainty into the posterior. The older two-target/nine-put flat-curve example remains a separate historical example, not a matched substitute.

## What counts as finished

For each route, deliver these artifacts together:

1. An immutable bank snapshot with unique IDs, fixed holdouts, clean numerical prices, the quote-noise law and numerical-error records.
2. A selected, reloadable BayesFlow 2.1 checkpoint, with the exact bank identity, training recipe and sampling settings.
3. A comparison report covering calibration, conditional references, product uncertainty and actual compute cost. Retain failures and state the numerical scope.
4. Small cached observations, posterior draws and plots that the notebook can load, with an instructor regeneration command. A participant must not need cluster generation or retraining.

Use the existing bank, checkpoint and report formats. Wire the checked artifacts into the Heston and calibration lessons in [the course notebooks](../../tutorial/notebooks/), with a known-curve rough step before the joint extension. Rehearse cached and live inference paths after the artifacts pass; notebook execution has not been requested in the current corrective unit.

## What a fair Heston comparison must control

Francesco already trained on **50,000 unique surfaces**. Our 20k collection was a fresh development prefix; it was never a sufficient replacement target. The extension to nested **50k/100k training snapshots is now complete**, preserving the existing prices, IDs, root seed and holdouts.

The original and new course-style recipes already largely agree: four 256-unit layers, 500 epochs, batch 256, seed 0, fresh quote noise, AdamW with the recorded warm-up/cosine schedule and alpha -0.5. However, 500 epochs on 20k gives about **39,500 optimizer updates**, compared with **98,000 on 50k**. The smaller run reduced both data coverage and optimization. It does not establish whether 50k or 100k is sufficient.

The earlier new 20k FlowMatching fit passed one of three conditional development cases; the archived model passed two. These are useful diagnoses. The matching requirement also includes Francesco's broader evidence: **2,000-case SBC**, recorded **95% coverage of 93.5–95.2%**, and conditional mean/90%-width agreement on **40/50 observations**. The historical 50-case comparison principally uses importance reweighting, with four MCMC checks; it is not 50 converged PyMC chains. [Earlier evidence](followup-results-2026-10-27.md), [50k/100k comparison](heston-backbone-comparison-2026-10-27.md).

Compare original and new estimators on the same recoverable observations, prior, quote law and metrics. If an old panel cannot be recovered, freeze a shared new comparison panel and evaluate both there. Keep coverage levels consistent and report paired differences and Monte Carlo uncertainty. Meeting the recent three-case screen alone does not deliver the replacement estimator.

**Current execution, 29 October:** Heston and both rough routes now have checked **100k training banks and completed baseline fits**. The final joint100k fit completed its 500 posterior epochs, 300 pricer epochs and 2,000-case validation; all seven marginal rank checks pass. Its pricer label RMSE falls from .10118 to .08076 quote SD, and all four saved independent benchmark points improve on both grids. The known100k full-300 pricer also improves its independent benchmark. These are development comparisons with one training seed and different optimizer exposure. Known-curve fixed-case PyMC agreement remains **2/3**, and joint conditional/product acceptance is unfinished. The reserved rough test outcomes remain untouched, and no notebook has been executed. [Current work and resource bounds](backbone-delivery-work-order.md) and [joint100k comparison](../evidence/pilot-results/2026-10-29/joint100k-completed/README.md) supersede earlier queue states.

## Immediate work, in parallel

| Workstream | Next concrete action | Completion condition |
| --- | --- | --- |
| Heston data and estimator | Package the passed candidate's teaching examples and regeneration commands. Resolve the path-product contract and validate its pricer; retain remaining seed/conditional claim limits. | Checked bank, estimator and cached lesson examples with explicit calibration/reference/product scope. |
| Known-curve rough | Review the completed 100k posterior and improved full-300 pricer. Resolve the remaining conditional discrepancy with an appropriate PyMC target, then package reviewed teaching examples. | A three-target estimator using the known curve, independent reference and product checks, and cached lesson examples. |
| Joint rough | Review the completed 100k posterior/pricer and four-point independent rescore. Establish conditional PyMC and product checks before promoting the new teaching artifacts. | Checked snapshot, selected posterior/emulator and completed numerical/posterior/product checks. Bank generation alone is not the deliverable. |

The live Heston collection can grow without regeneration. The completed extension from 20k to 100k training rows cost **1.0114 allocated CPU-hours**, including integrity/freezes, below its separate 6-hour limit. The checked 100k snapshot is available for reuse; another extension is not the immediate Heston priority.

The joint bank serves the third lesson. Its analytic curve law differs from the known-curve route's seven-interval step-curve law; their prices cannot be relabelled across tasks. Both banks are now complete, so additional generation should answer a specific numerical or learning question. The initial joint phase and its 100k extension retain their separate historical cost records and ceilings.

## Fair comparison for the first rough lesson

The original pilot has **1,600 training surfaces out of 2,200**, on 63 contracts with seven supplied curve levels. Its 250-case SBC passed H and eta but failed rho and rho×eta. Those failures are problems to improve, not release criteria to copy. Include the derived rho×eta diagnostic in the new evaluator.

Use the pilot's declared parameter/curve prior, noise and numerical target for the reproduction control. The historical pilot grids are **80/504/504**; the current joint working grids are **1280/2016/2016**. A finer-grid known-curve model should agree with references for its own numerical target, with a paired refinement report. It should not be forced to reproduce coarse-grid bias. The original study already included adaptive paths, covariance/precision checks and grid evidence; reuse that work.

Before the main known-curve bank, freeze whether it retains the pilot curve prior or adopts the broader planned curve prior needed by the chosen market example. Preserve the overlap reproduction comparison; a changed curve prior needs a distinct collection identity and a measured cost. Do not claim the pilot supports a market curve outside its prior. This is one known-curve lesson with explicit numerical/domain versions, not another tutorial route.

## Checks that support the teaching claims

- Use BayesFlow **2.1.0** for new fits and its native adaptive Tsit5 default for FlowMatching. Preserve historical checkpoint sampling settings and archived 2.0.14 environments.
- Use validation for model selection; retain **2,000 untouched acceptance surfaces per model/task** for final calibration, plus separate conditional reference cases. Previously inspected cases remain development evidence.
- Compare raw amortized outputs with independent references and products. Report corrected/audited results separately. Prefer existing deterministic **PyMC** workflows where natural; never pass a fresh random Monte Carlo log likelihood to ordinary MCMC.
- Fit one baseline per route before the small architecture/summary comparison. Reuse data across affine/spline/FlowMatching/MeanFlow and vector/set encoders; report unique surfaces and optimizer updates. A learning curve informs the delivered estimator rather than replacing delivery.
- Keep reported Monte Carlo precision distinct from grid bias. A small reported MCSE is not an absolute-price-accuracy certificate, and matched emulator/neural predictions are not independent truth.
- Include a reference-supported example of non-Gaussian posterior geometry in the teaching material. The [existing-chain inspection](backbone-validation-2026-10-28.md#posterior-geometry-for-the-tutorial) already shows boundary skew; compare physical and transformed Gaussian baselines and the amortized output. Investigate the predeclared nearly-flat joint-curve case for weak identification and curved dependence; do not claim modes or geometry that have not been demonstrated by checked samples.

## Joint curve and bounded numerical refinement

The joint curve is `long_variance + (short_variance - long_variance) * exp(-decay * t)`, the simplified subfamily of [Baschetti, Bormetti and Rossi, Section 4.3](https://arxiv.org/html/2306.11061v2#S4.SS3) with the hump term removed. It allows rising, falling and flat curves, but no hump; retain the [completed hump challenge](curve-family-screen.md). The existing broad candidate prior remains unchanged: H 0.02–0.35, eta 0.5–3.5, rho -0.99 to -0.2, each variance level 0.01–0.16 and decay 0.25–8/year. Numerical difficulty alone does not justify narrowing it.

Retain the six predeclared development cases (central, falling, nearly flat, low-H/high-eta, high variance and steep rise), starting with **central and high variance**. Compare the same observed quotes/prior/noise under working and doubled grids; at most twelve reference fits across the six cases and two resolutions. Cost independent labels and references before submission. Reuse one validated forward emulator and, if needed, a small checked local refinement correction rather than a full second bank.

Main product: the ATM forward-start call resetting at three months and expiring at six months; analytic control: the one-year continuously monitored variance-swap rate. The proposed refinement tolerances, reference precision requirements and limitations remain in [EXPERIMENT-PROTOCOL.md](EXPERIMENT-PROTOCOL.md#bounded-posterior-and-product-refinement-panel). An unresolved comparison remains inconclusive; do not turn it into a pass or an endless prior-wide certification campaign.

For the known-curve rough lesson, that variance-swap mean rate is fixed by the supplied curve; varying H, eta and rho does not create uncertainty in it. Use it as a deterministic control and use the forward-start call to demonstrate posterior product uncertainty. Joint curve inference then adds uncertainty to the variance-rate control itself.

## Boundaries and history

Sparse quotes, masks, arbitrary chains and thin-market examples remain a [separate downstream unit](SPARSE-QUOTE-CASE-PLAN.md). Complete the usable complete-surface backbones first. Wider data growth and architecture comparisons are bounded by measured learning/accuracy needs; each new allocation has an explicit limit.

Keep producer/source/runtime identities and append only missing split-qualified IDs. Preserve successful shards, fixed holdouts, failures, covariance records and all spending. A changed task/grid/prior needs a distinct collection; training on a new snapshot starts a new fit. [Incremental-bank instructions](incremental-calibration-banks.md).

Completed work remains in the [26 October report](oscar-pilot-2026-10-26.md), [reviewed A/B results](reviewed-unit-2026-10-27.md), [reliability/Heston follow-up](followup-results-2026-10-27.md) and [native-solver/bank report](native-default-and-bank-start.md). The older [cluster runbook](CLUSTER-RUNS.md) is historical/reusable guidance, not an instruction to launch its entire programme. Update [claims](../../tutorial/CLAIMS.md) and course acceptance only when the corresponding backbone evidence supports them.
