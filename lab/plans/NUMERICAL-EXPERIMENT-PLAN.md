# Numerical backbone for the Heston and rough Bergomi tutorials

**Delivery objective, revised 29 October 2026:** provide the reusable data, trained estimators and checked examples that the live tutorial actually loads. A working generator, a small development fit or a larger row count is an intermediate result. Each lesson needs a usable estimator and evidence that its posterior and product conclusions are trustworthy.

The numerical implementation was delivered through [PR 48](https://github.com/fmuia/sbi4finance/pull/48) and its follow-ups. It extends Francesco's course and existing numerical studies, whose [recorded results](../../course/status.md) remain the baseline to reproduce and improve. The [implementation guide](full-surface-calibration.md) gives the shared workflow; the [backbone work order](tutorial-backbone-work-order.md) gives the immediate actions and spending boundaries.

**Current state, 29 October:** Heston, known-curve rough Bergomi and joint-curve rough Bergomi each have a checked **100k training bank and completed baseline fit**. Their native archives, models and small cases are available through the [training-data guide](training-data-guide.md), on [GitHub](https://github.com/fmuia/sbi4finance/releases/tag/tutorial-backbones-2026-10-29) and [Drive](https://drive.google.com/drive/folders/1Zez6mSu8Y_B7IIjdRYWJlDxdAl49fTLq). The participant package uses BayesFlow 2.1, ships six cached cases and supports optional matching native checkpoints. The [report](../evidence/reports/inference-performance-2026-10-28.pdf) and [course status](../../course/status.md#the-joint-runs) distinguish completed J1 development checks from pending J2 conditional references and J3 product/long-end comparisons. Bank generation is complete; lesson curation is the immediate priority. Historical budgets and controllers do not authorize another campaign.

## The three delivered routes

| Lesson | What is inferred | What is supplied | Required backbone |
| --- | --- | --- | --- |
| Heston | Initial variance, theta, vol-of-vol and rho; kappa fixed. | Complete 63-quote observation and declared design/noise. | Delivered **100k / 5k validation / 5k test** snapshot and final-500 candidate; the earlier 50k comparison remains recorded evidence. |
| Rough Bergomi: known curve | H, eta and rho. | Complete quotes and seven known curve levels, held fixed within each inference problem. Curves vary across training cases. | Delivered **100k / 5k validation / 7k reserved test** snapshot, final-500 posterior and separate pricer candidates. Retain the known-curve conditional-reference limitations. |
| Rough Bergomi: joint curve | H, eta, rho and three curve coefficients. | Complete quotes and design/noise. The generating curve is not an inference input. | Delivered **100k / 5k validation / 7k reserved test** snapshot, final-500 posterior and forward pricer. J1 is complete; J2 and J3 remain pending. |

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

**Current execution, 29 October:** Heston and both rough routes now have checked **100k training banks and completed baseline fits**. The final joint100k fit completed its 500 posterior epochs, 300 pricer epochs and 2,000-case validation; all seven marginal rank checks pass. Its pricer label RMSE falls from .10118 to .08076 quote SD, and all four saved independent benchmark points improve on both grids. The known100k full-300 pricer also improves its independent benchmark. These are development comparisons with one training seed and different optimizer exposure. Known-curve fixed-case agreement against the reused **50k-pricer PyMC references remains 2/3**; the full-300 pricer has no new conditional PyMC target. Joint conditional/product acceptance is unfinished. The reserved rough test outcomes remain untouched. Cached assets and runtime boundaries have been tested; interactive lesson rehearsal remains to do. [Current work and resource bounds](backbone-delivery-work-order.md) and [joint100k comparison](../evidence/pilot-results/2026-10-29/joint100k-completed/README.md) supersede earlier queue states.

## Immediate work, in parallel

| Workstream | Next concrete action | Completion condition |
| --- | --- | --- |
| Heston lesson | Review the supplied cached example and full inference/checks/output narrative in slot 4. Retain the passed calibration/analytic-product evidence and explicit limits; specify any unresolved path-product check before new computation. | A coherent cached lesson and reviewed live options, with matching evidence for each claim. |
| Known-curve rough lesson | Curate the supplied known-curve case as the first rough step after Heston. Keep its seven supplied levels distinct from inferred targets and explain the deterministic variance-rate control. Scope the remaining conditional discrepancy before additional reference work. | A clear first rough lesson with its current reference and product limitations visible. |
| Joint rough lesson and J2/J3 | Curate the supplied six-target examples and J1 diagnostics. Specify matched conditional PyMC comparisons (J2) and the payoff, precision and extrapolation protocol for products (J3) before commissioning those checks. | A usable joint lesson; stronger conditional/product claims only after their separate checks pass. |

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

## Current next work and repository trimming

**Scope update, 1 November 2026:** merged PR #64 includes the three core
application notebooks, including trajectory inference. The current
[collaborator handover plan](../../tutorial/curriculum/TUTORIAL-REVIEW.md#g-collaborator-handover-before-the-next-feedback-round)
supersedes older returns exclusions and the initial lesson-integration status
below. Keep active trajectory code and its dependency closure during trimming.


**Source ownership update, 2 November 2026:** the lesson runtime is now owned
under `tutorial/src/sbi4f_tutorial/{heston_options,rough_options,trajectories,shared}/`.
Lab workflows reuse these implementations; there is no source-copy publication
step. See the [current source map and handover](../../tutorial/README.md#collaborator-handover)
and [refactor plan](../../tutorial/curriculum/TUTORIAL-REVIEW.md#h-tutorial-owned-runtime-refactor).
Keep both sibling directories from one revision on compute. Read/train on
preserved banks normally; append or resume frozen generation only with its
producer-matching historical source/environment. New generation records the
changed source identity. This organization change requires no new scientific fit.

**First content-review action:** reconcile [the slot 4 curriculum](../../tutorial/curriculum/04-slot4-heston-in-bayesflow.md), [the slot 5 curriculum](../../tutorial/curriculum/05-slot5-rough-bergomi-to-a-reserve.md) and their notebook prose with the complete-surface Heston → known-curve rough → joint-curve rough sequence. The existing narrative still contains historical SPY/known-curve framing, and the slot 5 plan names a thin-chain case as required. Retain historical SPY provenance, but keep thin-chain material optional rather than a prerequisite for the core lesson. Cached integration and completed numerical inputs do not mean the lessons are fully curated.

Use the supplied data and saved cases to finish that sequence. J1's marginal calibration, contraction and observation-dependence evidence does not complete J2's conditional references or J3's financial-product comparisons. Keep dataset size, optimizer-update count, task/curve prior, observation noise and reference method explicit when comparing results. New numerical work should resolve a named lesson claim; another bank or infrastructure layer is not the default next step.

The [repository README](../../README.md) is the collaborator front door. Maintain this plan, the curriculum overview and course status in place. Trim in small reviewed follow-ups:

1. **Keep active content and its dependency closure.** Retain the core notebooks and course, their runtime, publication tools, tests, task configurations and code used by supported load/train/evaluate/generate/reference/product commands. Retain historical accepted checkpoints and exact reference evidence while a lesson or claim still uses them.
2. **Check dependencies before cutting.** Search imports, CLI/pixi tasks, tests, the runtime-boundary/parity checks, asset manifests, claims, source-hash bindings and links. Deferred sparse/legacy helpers may still supply utilities to active routes; first separate or replace that dependency with checks.
3. **Retire dated orchestration and bulk records.** Once no supported route needs them, remove one-off launch/recovery controllers, duplicated frozen operator copies and bulk dated logs/per-case outputs from the working tree. Preserve a pinned source commit/tag, immutable external archives and checksums, plus the compact record needed to support every retained claim. Update references in the same change. Git retains retired source; failures and scientific provenance are not relabelled or erased. Do not repurpose dated controllers bound to old checkouts as current launchers.
4. **Use the current three-application scope.** The Heston and rough-Bergomi trajectory application is active core work; the earlier D21 returns exclusion is superseded for that application. Separate legacy SVJ/multifractal take-home material remains outside this refinement scope, but utility dependencies must be checked before removal. Preserve unique earlier ideas through immutable [PR 37 source](https://github.com/fmuia/sbi4finance/tree/1f9972ecb2bee087248af1f2abef7bcdf0c7a125) and [PR 38 source](https://github.com/fmuia/sbi4finance/tree/d00ccb121be1e93a1100b6681ca1e17ec0c3b573). PR 40's promotion plan is independent; no PR state changes here.
5. **Validate the smaller route.** Run the affected lab tests/release checks and tutorial independence/parity/notebook checks. Stop when further removal would obscure a retained scientific claim, break a supported command or require an abstraction solely to justify deletion.

### Bounded retirement batch · 1 November 2026

Two completed, Oscar-specific continuation controllers are retired from
`lab/cluster/`. They are not the canonical generation/training/evaluation
interface. A tracked-text search of names and SHA-256 identities, plus the
then-current Python imports, CLI/pixi tasks, tests, publication/copy registry and
Markdown references, found only historical consumers and the explanatory
documentation. No worker, notebook, current test or supported command invokes
either controller.

| File | Decision and dependency finding | Retained exact source |
| --- | --- | --- |
| `lab/cluster/start_known100k_20261028.py` (6,564 bytes) | Retired. Its two source-hash references are frozen October progress receipts; no current caller was found. | [Pinned main source](https://github.com/fmuia/sbi4finance/blob/08f4ebde1b47893057c61706a1a94ae017806c47/lab/cluster/start_known100k_20261028.py); archive member `sbi4finance-independent-work-source-20261028/start_known100k_20261028.py`. |
| `lab/cluster/start_joint100k_20261029.py` (11,341 bytes) | Retired. References are frozen handoff/fit receipts and the historical `joint100k-fit-handoff/check_handoff.py`; that original check belongs to the pinned historical checkout, not current CI. | [Pinned main source](https://github.com/fmuia/sbi4finance/blob/08f4ebde1b47893057c61706a1a94ae017806c47/lab/cluster/start_joint100k_20261029.py); archive member `sbi4finance-joint100k-handoff-source-20261029/start_joint100k_20261029.py`. |
| `lab/cluster/submit_full_surface_backbones_20261028.py` | Retained. `tests/test_full_surface_post_bank_operator.py` imports its scheduling guards; other retained historical controllers also refer to it. Retirement requires separating those consumers first. | Existing source and pinned history remain unchanged. |
| The three October 27 evidence copies named `submission-operator.py` | Retained. Inspection found three different byte sequences (6,295 / 10,106 / 4,923 bytes), not duplicate files. Their README, archive script and checksum references remain intact. | `evidence/pilot-results/2026-10-27/{backbone-runs/known-bank-final,native-bank-start,tutorial-backbone}/submission-operator.py`. |

Before retirement, each controller matched both the pinned Git blob and the
already preserved payload-manifest entry exactly:

```text
a6293857c61a9f4178190deb4f2bc2bffde9826518a7c35141047f2d47484d67  start_known100k_20261028.py
525ad97a6a2db32e3f24f9f2e0cc53a9510a2e209c322251ecf17973108311ce  start_joint100k_20261029.py
```

The external copy is the single part
`oscar-history-20261101/historical-options-operators-and-controls/historical-options-operators-and-controls.tar.zst.part-00000`
in the existing [workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k).
Its archive SHA-256 is
`ba0e88812ffb2f9a548921b1d2e109ecbcf59370cb7059778703ab1d5621b740`;
the payload manifest SHA-256 is
`883be3380cda99a102f0d2445dc610f8cd736a9a7a9707daf252b6a13717e9fe`.
The [existing preservation summary](../evidence/pilot-results/2026-11-01/oscar-preservation/summary.json)
binds the full historical closeout and its manifest, all 16,682 locally checked
logical payloads in this group, and provider-confirmed upload. This retirement
rechecks those local proof bindings and the two source entries; it does not
claim another archive read, cold-file rehydration or independent cloud download.

Historical evidence, including `check_handoff.py`, `sha256sums.txt`, receipts
and failed attempts, is unchanged. To reproduce that old no-submission handoff
check, use its full pinned `08f4ebde1b47893057c61706a1a94ae017806c47`
checkout. The source-recovery check was run in a temporary copy of that layout
with the exact pinned helper, checker and plan; scheduler calls remained mocked.
At that retirement baseline, all four cases in
`tests/test_full_surface_post_bank_operator.py` passed; the then-current
`studies/copy_to_tutorial.py --check`, `studies/manifest.py --check` and
`studies/claims.py --check` passed. The current source-ownership refactor replaces
copy checking with `studies/runtime_boundary.py`; the historical record is
retained against its original source revision. The next candidate operator copies are retained
until their distinct behavior and provenance have a separate retirement proof.

This bounded batch changes no numerical worker, task, model, scientific result
or immutable evidence hash. No new cluster run or collaborator-machine
rehearsal is implied.
