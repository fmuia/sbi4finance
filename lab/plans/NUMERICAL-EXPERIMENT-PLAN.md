# Numerical backbone for the Heston and rough Bergomi tutorials

**Delivery objective, revised 29 October 2026:** provide the reusable data, trained estimators and checked examples that the live tutorial actually loads. A working generator, a small development fit or a larger row count is an intermediate result. Each lesson needs a usable estimator and evidence that its posterior and product conclusions are trustworthy.

**Performance follow-up, 7 November:** the [Heston PR and commit plan](#heston-performance-pr-and-commit-plan--7-november-2026)
below separates pricing optimizations, a bounded training-budget study and the
publication of measured recipes. No implementation or compute jobs start in
this planning change.

The numerical implementation was delivered through [PR 48](https://github.com/fmuia/sbi4finance/pull/48) and its follow-ups. It extends Francesco's course and existing numerical studies, whose [recorded results](../../course/status.md) remain the baseline to reproduce and improve. The [implementation guide](full-surface-calibration.md) gives the shared workflow; the [backbone work order](tutorial-backbone-work-order.md) gives the immediate actions and spending boundaries.

**Current state, 29 October:** Heston, known-curve rough Bergomi and joint-curve rough Bergomi each have a checked **100k training bank and completed baseline fit**. Their native archives, models and small cases are available through the [training-data guide](training-data-guide.md), on [GitHub](https://github.com/fmuia/sbi4finance/releases/tag/tutorial-backbones-2026-10-29) and [Drive](https://drive.google.com/drive/folders/1Zez6mSu8Y_B7IIjdRYWJlDxdAl49fTLq). The participant package uses BayesFlow 2.1, ships six cached cases and supports optional matching native checkpoints. The [report](../evidence/reports/inference-performance-2026-10-28.pdf) and [course status](../../course/status.md#the-joint-runs) distinguish completed J1 development checks from pending J2 conditional references and J3 product/long-end comparisons. Bank generation is complete; lesson curation is the immediate priority. Historical budgets and controllers do not authorize another campaign.

## Heston performance: PR and commit plan · 7 November 2026

**Objective:** reduce the cost of generating Heston option data and find a
shorter training recipe that meets the tutorial's stated posterior checks.
Keep the model, prior, quote noise, numerical target and scientific claims
explicit. This is a performance follow-up for the four-parameter, 63-quote
Heston option task; it does not launch a new bank or extend the rough/trajectory
campaigns.

The [initial audit](../evidence/pilot-results/2026-11-07/heston-performance-audit/summary.json)
is exploratory evidence, not a release benchmark. On the current notebook,
15 online epochs mean only **30 updates / 1,920 training presentations**. The
local timed fit took **9.36 s**, including 7.26 s in pricing callbacks; imports
and workflow construction are outside that timer. The full 100k fit used
**195,500 updates / 50 million presentations** and took **482 s including reload**
on an L40S GPU. Those are different workloads and machines.

The current checked scalar pricer averaged **4.84 ms/surface** on 64 fixed prior
rows. A fixed-grid prototype averaged **1.51 ms** with the same refinement and
price-bound checks, with maximum price difference below 6e-16. The published
bank's generation timings belong to an older frozen producer. The prototype
needs the regression and reproducibility work below before production use.

### Implementation progress · 8 November 2026

P1 is implemented: the lab lock metadata is repaired without external dependency
changes; the portable benchmark retains all 276 predeclared rows; training and
evaluation now report actual work and separately scoped clocks. The
[baseline record](../evidence/pilot-results/2026-11-08/heston-performance-baseline/summary.json)
contains the measured panel, source identities, cold/warm summaries and
reproduction command. Warm median pricing is **3.92 ms** for the checked bank
route and **2.39 ms** for the notebook route on this CPU. This measures pricing,
not complete bank generation. All cold/warm prices and refinement differences
match the frozen reference exactly; there are no failed panel rows. The frozen
reference median is stable at 8.53 ms for the checked route.

The additive training report distinguishes completed versus recovered work,
partial final batches, selected-checkpoint updates, nested timing phases, host
peak RSS and scheduler allocations. Existing epoch recovery/defaults remain in
place. The baseline lab suite passes **534 tests, with three optional skips**;
manifest, claims and runtime-boundary checks pass. P2 and P3 proceed as separate
branches. No study matrix has been admitted and no model has been replaced.

### Sequence and parallel work

| PR | Purpose | Depends on | Can proceed alongside |
| --- | --- | --- | --- |
| Planning PR, this change | Record the audit and agree this sequence | Current main `3d6febb` | — |
| P1 | Reproducible benchmarks and separate timing fields | Planning review | — |
| P2 | Faster checked scalar pricing | P1 baseline | P3 and the bank-based training study |
| P3 | Frozen training-budget protocol and bounded runner | P1 timing contract | P2 |
| P4 | Run the study and report accuracy versus cost | P3 protocol review and readiness | P2; it reads the existing bank |
| P5 | Publish the measured recipe and refresh tutorial cost statements | P2 and P4 results | Content review of the other lessons |
| P6, optional | Bounded batch pricing | P2 benchmark demonstrates a remaining need | Does not block P3–P5 |

Each row is a separate reviewable PR. The implementation starts from current
main, not another collaborator's unmerged lesson branch. P2 and P3 should avoid
editing each other's files. Changes to the Heston notebook land in P5; resolve
against the then-current notebook without reverting Stefan's simplifications.

The planning PR has two commits: record the compact exploratory audit, then add
this plan. It changes no executing code and submits no jobs.

### P1 · Establish a reproducible baseline

**Commit 1 — reconcile local package lock metadata.** At baseline `3d6febb`,
`pixi install --locked -e core` in the lab fails because its lock metadata no
longer matches the sibling tutorial package. `pixi lock --dry-run --json`
reports no environment/package changes. Refresh the local-package metadata and
verify both locked environments install. Keep all external package versions and
builds unchanged; any required upgrade needs its own reviewed scope. Do not
work around the mismatch by changing the documented install to `--frozen`.

**Commit 2 — portable pricing benchmark.** Add one benchmark entry point under
`lab/scripts/`, reusing the checked pricing APIs and the existing
`tests/test_hkde_maturity_reuse.py` reference. Accept input/output paths and
CPU thread settings; remove local paths from the exploratory scripts. Freeze
256 prior rows, all 16 prior-box corners, targeted difficult short-maturity
cases, the ordered 63-contract grid, seeds and source/task identities. Include
one-rung notebook pricing and the full bank refinement policy as separate
routes. Record every failure; do not skip or redraw a difficult row.

**Commit 3 — training timing and work counters.** Extend the existing training
worker/report with backward-compatible additive fields: actual optimizer
updates, batch sizes, unique rows, example presentations, architecture and
optimizer/schedule. Measure imports, data loading, first build/compile, pricing,
training, validation, checkpoint serialization/reload and posterior sampling.
Record wall time, process CPU time and allocated resource time separately.
For accelerator phases, synchronize at measurement boundaries; callback times
may overlap device work and must not be subtracted as if all phases were serial.
Keep the existing interrupted-run and optimizer-state recovery contract.

**Commit 4 — baseline report and cost labels.** Record hardware/device, package
and source versions, cold/warm cache state, sample counts, failure counts,
median/p95/max and total panel time. Use alternating implementation order and
repeats for microbenchmarks. Label old generation/training records as historical
rather than overwriting them. Define the fields P2 and P4 must report.

**Files:** local-package lock metadata, `lab/scripts/`, the existing full-surface trainer/evaluator, focused
lab tests and compact evidence under `lab/evidence/pilot-results/`. Runtime
mathematics and notebook defaults stay outside this PR.

**Gate:** the benchmark reruns from a complete checkout using caller-owned paths;
prices, seeds and failures agree; current training defaults and recovery still
work. No unit test passes or fails because of wall-clock speed. A benchmark
failure is a recorded result, not an excuse to shorten the panel.

### P2 · Optimize scalar pricing while retaining the checks

**Commit 1 — cache fixed-grid Fourier plans.** Cache the immutable contract-grid
quantities used repeatedly: strike phases, weights, maturity index maps and
quadrature controls. Use exact, content-based keys that retain contract order;
never round maturities or key only by a mutable task object's identity. Bound
cache memory and prevent callers from mutating cached arrays. Keep the general
pricing API and use its existing path when fast-path conditions are not met.

**Commit 2 — share work across calls and puts.** Evaluate characteristic
functions once per distinct maturity for the combined surface. Derive puts by
put–call parity, retaining the existing price bounds, precision and failure
behavior. Place the implementation in the tutorial-owned
`shared/pricing.py` / `heston_options/model.py`; lab callers reuse it directly.

**Commit 3 — reuse refinement rungs and publish measurements.** When an accuracy
comparison fails, carry its successfully computed refined prices into the next
comparison rather than repricing that rung. Failed/bounds-violating evaluations
must not become accepted cached results. Return the accepted refined prices and
the same diagnostic discrepancy as before. Run the P1 benchmark after each
optimization so its contribution can be measured or reverted independently.

**Regression panel:** compare against the frozen reference at every rung,
including cold/warm caches, changed and mutated grids, mixed/unsorted contracts,
scalar/broadcast inputs, expiry, zero vol-of-vol, zero/nonzero supported jumps,
price-bound failures and refinement exhaustion. The supported general Heston–Kou
API retains its tests even though this optimization targets the Heston lesson.

**Gate:** identical acceptance/failure outcomes and target/task/noise controls;
per-price differences at most **1e-12 forward units** and **1e-6 quote-noise SD**
on the declared valid panel. Report differences in prices and refinement
records; investigate any acceptance-boundary discrepancy rather than widening
the tolerance after seeing it. Publish cold/warm total and tail timings plus
peak memory. A useful target is at least 2× on the fixed-grid benchmark; the
observed 3.2× prototype is not a promised end-to-end bank speedup.

This changes producer source identity. Preserve existing bank/checkpoint bytes
and keep old generation resumable only from its original producer. Run the
shared-runtime boundary, pricing/collection tests and required release checks.

### P3 · Freeze the training-budget experiment before fitting

**Commit 1 — protocol and data-role audit.** Add a bounded experiment
configuration beside the existing Heston configs and update this plan with its
hash before launching. Reuse the immutable 100k bank and saved full-model
baseline. Audit all previously inspected IDs, reference cases and validation
usage. The 5k validation pool already influenced epoch selection; a new slice of
it is not an untouched final test. Reserve genuinely unused test IDs before
screening. If none can be established, propose a small independent evaluation
panel separately; do not silently reuse inspected observations.

Hold the task, prior, quote noise, target transform, production adapter,
FlowMatching family, vector conditions and native sampler fixed. Compare
**2×128 and 4×256** subnet widths, **batch 256**, and optimizer-update budgets
**30, 100, 300, 1,000, 3,000, 10,000 and 30,000**. The initial screen uses seed 0:
**14 fresh fits at most**. This isolates network size and training exposure;
it is not a claim that the notebook's different adapter/default optimizer is
the same training recipe. Benchmark that exact notebook recipe separately.
Short runs visit fewer unique rows of the same source bank. Record the actual
visited IDs outside neural conditions; do not report the entire bank as seen.

**Commit 2 — explicit update cap and learning-rate schedule.** Add a small
fresh-run-only budget mode to the existing worker using the public fit/callback
interfaces. Stop at the exact requested update, including a partial epoch, and
record actual exposure and the endpoint checkpoint. Use the same existing
AdamW/warm-up/cosine rule parameterized by each run's own total update budget;
record its discrete values for very short budgets. Preserve the current default
500-epoch behavior and epoch-recovery format. Do not claim that an early
checkpoint from a 500-epoch schedule is an independently designed short fit.
Test the cap, stop/checkpoint ordering, schedule, determinism and rejection of
incompatible resume attempts. Assert actual optimizer iterations and endpoint
weights even when several batches could execute inside one compiled block;
checking the requested limit alone does not establish an exact stop.

**Commit 3 — posterior-based screening and report.** Use one fixed, already
inspected development panel of **512 observations and 128 draws each**, the
same seeds and the same full-model control. Reuse the
[three cached conditional comparisons](heston-backbone-comparison-2026-10-27.md#the-three-cached-conditional-comparisons)
with **4,096 draws per case**; retain their MCMC limitations and
Monte Carlo uncertainty. Use **32 fixed development observations × 32 paired
parameter draws** for the bounded posterior-predictive check.
Conditional mean and interval comparisons account for neural sampling error
and reference mean/quantile uncertainty, retaining the reference chain/draw
structure and autocorrelation. They must not treat all MCMC draws as independent.
Report rank/coverage diagnostics, parameter recovery, physical interval widths,
conditional mean/width agreement and a bounded posterior-predictive check.
Validation loss is descriptive, not the model-selection gate: the old epoch-402
minimum-loss candidate failed checks that final epoch 500 passed.
Before admitting the fit matrix, evaluate the saved full-model control using
these same sampling settings, conditional rules and uncertainty calculation.
If even that control is inconclusive or fails the intended reference gate,
resolve the protocol/reference-precision question before commissioning the
matrix; fitting more models does not repair an imprecise reference.

**Commit 4 — resource admission and independent execution.** Give each fit its
own directory and durable state. A failed lane does not cancel unrelated lanes;
there are no automatic retries or automatic larger budgets. Keep the existing
source/task/bank hashes, same-checkout guard and projected bank reads. New
budget fits use no online simulation and never decode the unused covariance
payloads. Expose ordinary CPU/GPU CLI commands; Slurm is optional deployment.
Each job uses an immutable checkout. Merging P2 must not update source beneath
an active P4 run; the independent training lane does not wait for that merge.

**Gate before P4:** freeze panel IDs, candidate configs, seeds, metrics, numerical
thresholds, checkpoint policy and resource ceilings. Review a tiny exact-cap
training/reload smoke and its measured memory/time. No selected model or
notebook weight changes in P3.

### P4 · Execute and compare time to posterior quality

**Commit 1 — the bounded development screen.** Run the 14 predeclared seed-0
fits, with independent resource receipts. Use endpoint checkpoints and compare
to the saved full-model baseline; do not refit that baseline. Record setup,
training, simulation-free optimizer work, evaluation, sampling and reload costs
separately. Record failed/incomplete fits and all diagnostics, not just winners.

**Commit 2 — freeze and confirm at most two finalists.** A development finalist
must have finite in-support samples without clipping, pass the existing
simultaneous marginal rank screen on the declared panel, have 90% and 95%
coverage within three percentage points of the full-model control, and meet the
existing conditional rules on the three cached cases: absolute mean difference
at most 0.3 reference SD and 90% width ratio in [0.75, 1.25]. Keep borderline
comparisons inconclusive when reference/sample uncertainty crosses a threshold.
Among qualifying candidates, choose by measured total cost; do not pick a winner
from loss alone. Freeze up to two complete recipes, then run seeds 1 and 2 for
each. Including the reused seed-0 fits, this is **18 new fits at most**, not an
open architecture search. Report every seed; do not keep only the best one.
Seed 0 is the predeclared publication candidate; seeds 1 and 2 test robustness.

**Commit 3 — one final confirmation and compact report.** After candidate
selection freezes, compare the candidates and saved full-model control on
**2,000 audited unused observations, 512 draws per observation**. Evaluate all
three seeds for each finalist. Apply the existing simultaneous rank criterion;
report 50/80/90/95% coverage, recovery and paired uncertainty. For each coverage
level, compare **absolute coverage error against nominal**: candidate error
minus control error. Its upper confidence bound must be at most **two percentage
points**. The upper bound on each parameter's normalized mean-squared-error ratio
to the control must be at most **1.10**. Use predeclared simultaneous one-sided
95% paired-bootstrap bounds over observation IDs, covering the declared
finalists, seeds, parameters and levels. Each seed must meet the gates; there is
no best-seed selection. These proposed margins and their executable calculation
must be reviewed in P3, before the screen. An inconclusive interval is not a
pass. Normalize physical-coordinate squared errors by that parameter's declared
prior variance; resample observations jointly across models and targets to
preserve pairing. Apply the recovery gate per parameter, not to an average that
could hide an initial-variance regression. Conditional references and a fixed
posterior-predictive panel remain distinct checks, not proof of the full copula.

This is one predeclared final comparison of frozen candidates; no feedback from
it goes back into model/schedule tuning. A candidate that misses a gate remains
a fast demonstration. If none qualifies within 30k updates, retain the current
pretrained estimator and report the failed cost target; a new budget needs a
separate proposal. Initial scope adds no MCMC chains or path-product study.

**Proposed compute ceilings, to confirm from the P3 smoke:** no more than
**1.25 GPU-hours and six allocated CPU-hours total**, including failed attempts,
readiness, evaluation, reload and cleanup. Use at most four host cores per GPU
run; plan a 24 GiB allocation with GPU preallocation disabled and measure actual
peak use. Reserve the forecast final-confirmation cost before admitting the
screen. These are new proposed limits, not reuse of historical authorization.
If the measured forecast does not fit, reduce the predeclared matrix before
launch or stop for a revised plan; do not trim quality checks after results.
No new 100k bank, alternative-cluster requirement or overnight dependency chain
is needed.

**Gate:** a reproducible cost-versus-quality report with continuous discrepancies,
uncertainty and all failures; the original task/bank and independent final-panel
roles remain intact. Smaller/faster is a result only at the stated quality.

### P5 · Publish the recipe and update the teaching cost statements

**Commit 1 — explicit lesson actions and tested recipe.** Keep the direct,
small BayesFlow example understandable. Separate its measured fit time from
the evaluated pretrained estimator and from optional training on the downloaded
bank. Publish a short recommended recipe/model only if P4 supports it; otherwise
retain the current pretrained model and label the small fit as a demonstration.
Do not remove calibration or reference limitations to make the timing attractive.

**Commit 2 — cost labels, metadata and handover.** Update the Heston notebook,
existing curriculum/cost notes and training-data guide with clearly labelled
historical versus current figures. Report dataset size, unique examples,
updates, architecture, device, included phases and cold/warm state beside a
number. Use measured end-to-end generation before claiming bank-generation
speedups. Publish new assets only through the established acceptance/manifest
workflow; keep historical hashes, failure records and the unchanged full model
available. No changes to the opening Beamer slides or public website are needed
unless their actual cost claims need correction.

**Commit 3 — rehearse and close out.** Verify installed locked environments,
all tutorial tests and notebook exports, explicit training/inference buttons,
matching native reload and saved/live example identities. Record a cold and
warm presenter-machine run; headless export alone does not exercise a training
button. The collaborator workshop rehearsal remains a separate agreed step.

**Gate:** a collaborator can run the documented recipe and interpret the cost
and accuracy it establishes. The tutorial keeps a saved-example fallback and
opens without training. A production change includes the required lab tests,
release checks, tutorial checks and reviewed artifact metadata.

### P6 · Optional bounded batch pricing

Consider this only if the P2 end-to-end profile still shows row dispatch or
pricing as a material bottleneck. It is not required for the first improvement.

1. Add a bounded parameter-batch API with a caller-selected chunk limit; retain
   scalar callers and the general pricing fallback.
2. Carry per-row refinement, failure, order and split/seed identity through the
   batch. Verify batch versus scalar prices, diagnostics and failures on the
   same panel, including mixed easy/hard rows. Never drop/redraw a failed row.
3. Benchmark the complete bank writer and explicit online example, including
   memory, serialization and startup. A GPU/JAX or Numba kernel requires its own
   numerical-precision comparison; it is not assumed faster merely by using an
   accelerator.

**Gate:** a measured benefit beyond P2 with bounded memory and unchanged checks.
A covariance/storage-format change is a separate future task and should follow
measurement of actual I/O/compression cost, not the logical zero-array size.


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
