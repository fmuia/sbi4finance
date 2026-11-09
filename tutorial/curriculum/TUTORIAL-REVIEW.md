# Application notebook refinement: execution plan

> **9 November 2026.** The option applications' error model, observation design and
> downstream products are replanned for the 14 November tutorial in
> [`lab/plans/OPTION-APPLICATION-PLAN.md`](../../lab/plans/OPTION-APPLICATION-PLAN.md).
> Where it conflicts with the option sections below, that plan takes precedence.

Updated 1 November 2026 against merged baseline `7751e4c` (PRs #54 and #63).
The workshop is now two hours. Refine the content and presentation first; defer
minute allocation and pruning until the application sequence can be rehearsed.

This plan covers the existing Heston-option, rough-Bergomi-option and trajectory
application notebooks. Introduction, Amortized Bayes, Architectures and the
separate backbone-review notebook are outside this workstream. Trajectory
inference is a full application with the same delivery and evidence standards
as option inference. Its present development status is a starting point to
improve, not a reason to demote the application.

The six subplans below are ordered by their dependencies. Several can run in
parallel. Their status distinguishes available software, preserved outputs and
scientific acceptance; none implies the others.

## A. Close Oscar work and preserve its outputs

**Starting evidence.** The three harmonized trajectory banks currently exist on
Oscar: each has 1,000,000 training, 4,000 validation and 4,000 test histories, with
1,260 daily returns. Together their uncompressed NPZ payloads occupy 31.124 GB.
Live verification found all 23,631 shards, matching sizes and manifest hashes;
sampled payload hashes passed. Earlier full verification is recorded separately.
The published workshop archives include all six scale-up fits and diagnostics,
but explicitly exclude these full banks. Older fixed-drift banks, repaired fits
and some raw logs also require reconciliation.

1. Reconcile project job accounting with every relevant run root and local
   transfer receipt. Classify stopped attempts by their actual outcome and
   superseding run; do not resume failed historical experiments blindly.
2. Preserve unique banks, native models, references (including failed/partial
   chains), evaluations, logs, source and runtime identities. Exclude installed
   environments and redundant code/cache copies from preservation payloads.
3. Package immutable native files in bounded, resumable parts. Verify source
   manifests and payloads, transferred archive hashes and extracted contents.
4. Copy the missing deliverables to the existing shared Drive folder, retaining
   the distinction between local checksum verification, provider-reported upload
   completion and an independent cloud download.
5. Record an inventory mapping each produced output to its durable copy or an
   explicit redundancy/exclusion reason. Retain remote originals.

**Deliverables:** durable full-bank downloads, historical-output archives,
checksums, extraction instructions and one compact closeout record linked from
the existing path plan and training guide. Large payloads remain outside Git.

**Completion:** no unexplained active jobs or unique results left only on
scratch; every advertised transferred payload has a matching verification
receipt. Scientific questions still pending remain named as such.

## B. Establish a consistent application and evidence contract

For each selected example, display observations, unknown parameters, supplied
quantities, prior family/support, simulator/numerical law, observation design,
checkpoint identity and reference scope. Use the actual task metadata rather
than retyping a competing definition in explanatory prose.

Keep these distinctions visible:

- Option calibration concerns a pricing model; trajectory tasks concern the
  specified physical return process, including drift.
- Known-curve option inference conditions on seven supplied curve levels. Joint
  option inference targets three exponential-curve coefficients alongside the
  volatility parameters. This changes the curve family as well as uncertainty.
- Current synthetic cases, historical SPY studies, pilot fits and the separate
  long-end Gaussian/Laplace experiment keep their own identities and claims.
- A reference using an approximate neural pricer is not an exact-pricer target.
- Cached, live and newly trained outputs are labelled by their actual origin.

**Deliverable:** consistent task/source panels and a selected-case map, reusing
the existing notebooks and asset catalogue. No new asset platform is needed.

**Completion:** a reader can identify the target of every posterior or product
plot without relying on an opening disclaimer or presenter explanation.

Start with the existing case inventory, keeping illustration and evaluation
separate:

| Application | Existing primary case | Teaching role |
| --- | --- | --- |
| Heston options | `heston-final500` | Follow one current synthetic observation and its numerical reference. |
| Known-curve rough options | `rough-known100k-validation0` | Identify supplied curve inputs and posterior targets. |
| Joint-curve rough options | `rough-joint100k-validation0` | Explain inferred curve uncertainty; selected cases 1983/1941/1962 extend geometry inspection. |
| Trajectories | Predeclared validation case 0 per model; cases 1–3 remain available | Reveal history and inspect uncertainty, with each model's own synthetic path and task. |

The shared trajectory grid does not make the cached observations identical.
Likewise, switching current option development cases changes the observation.
Any future same-observation cross-model comparison needs an explicit shared-data
and measure contract; compare current interfaces without implying that experiment
has already been performed.

## C. Refine the option notebooks in place

**Retain:** Heston's simulator → prior predictive → bank/adapter/network →
training → diagnostics → numerical reference → application sequence; rough
Bergomi's parameter controls, forward simulation, computational-cost comparison,
classical baseline, product controls and assumption-sensitivity interactions.
These descend from the earlier `tutorial_final` notebooks and provide the
teaching structure to improve.

1. Remove stale full-bank-pending statements and identify historical results
   beside the sections that use them.
2. Carry an explicitly selected current synthetic case through its supported
   parameter, reference and derived-quantity views. Do not splice a new model's
   posterior into an old model's diagnostics or product results.
3. Keep historical SPY model checks available independently of the disabled
   archived neural loader, where their inputs are already accepted and packaged.
4. Preserve the known-curve/joint progression, explaining fixed inputs,
   uncertainty and weakly identified decay through actual examples.
5. Add prediction and interpretation prompts to existing controls. Keep a
   clear distinction between exploring a forward simulator and conditioning on
   the selected inference case.
6. Use analytic variance-related quantities for a supported uncertainty example.
   Retain historical reserve/long-end demonstrations with their method labels.
   New rough product conclusions depend on the existing J2/J3 validation work,
   including the exact forward-start contract definition.

**Deliverable:** refined `slot4_heston_bayesflow.py` and
`slot5_calibration.py`, with their existing useful interactions preserved.

**Completion:** current/historical continuity is explicit, cached viewing is
complete, optional actions remain explicit, and no statement exceeds its source.

## D. Develop the trajectory notebook to application parity

**Retain:** the shared Heston/known-curve/joint-curve selector, daily observation
grid, drift inference, every integer history length from 1 to 1,260, posterior
evolution, cached fallback and optional exact-length CPU inference.

**Reuse presentation patterns:** earlier Heston trajectory notebooks provide
prior/neural/reference marginals, paired joint geometry, per-case diagnostics
and exercises. Earlier Bergomi notebooks provide simulation/seed controls and
observed/latent displays. `sbi_fin_pocs` supplies a complete batch training and
diagnostic example. Their numerical claims and weights do not transfer to the
new tasks.

1. Add model/parameter/prior panels and a concrete observation-versus-hidden-state
   explanation. Synthetic truth may be displayed for interpretation, never fed
   into inference conditions.
2. Add prior/posterior marginal and joint views for the selected current case,
   retaining paired draw rows. Integrate the existing recovery, coverage and
   risk reports with their actual failures visible.
3. Walk through bank reading, random prefixes/masks, fixed return scaling,
   encoder, conditional density, offline training and native checkpoint loading.
   Use actual APIs and saved results. Optional small fits must be bounded and
   must not masquerade as reproduction of the published fit.
4. Add explicit simulation/prior-predictive exploration using the frozen model
   law. A displayed latent path is a teaching aid, not an estimator input.
5. Add matched reference overlays only at supported cases and lengths. A
   one-day comparison must not imply long-history conditional accuracy.
6. Develop posterior-predictive replicated-history checks. Continuation forecasts
   and future VaR/ES require joint latent-state/history conditioning, including
   rough memory; a parameter posterior alone is insufficient for that extension.

**Completion:** participants can follow generation → training → checks →
inference → interpretation for all three tasks. Numerical readiness is governed
by subplan E; a working interface never implies an acceptance certificate.

## E. Resolve trajectory numerical and inference limitations

**Current evidence:** all three selected candidates passed execution and risk
screens but failed the declared final marginal-calibration screen. Heston's
discrete simulator can also create deterministic daily returns and an atomic
drift posterior that its continuous neural density cannot represent exactly.
The 4,000 reserved tests per model have already been examined.

1. Diagnose before refitting. Use existing development reports to locate errors
   by model, history length, parameter and prior region. Separate simulator-law
   effects, finite-history information and estimator error.
2. For Heston, specify and verify a numerical target compatible with the intended
   posterior representation. Compare candidate laws on a bounded coupled panel;
   preserve the original atom-bearing task and its results. Never silently floor
   observations, discard difficult histories or relabel an old bank.
3. For the rough candidates, use a bounded experiment addressing the diagnosed
   mechanism (for example prefix exposure, optimization or conditional density
   capacity). Reuse the existing bank when the task is unchanged. Compare at
   matched training exposure and retain the common observation design.
4. Define task, candidate, budgets and validation rules before new production
   runs. Check full-shape memory and native reload locally and on Oscar. Use
   offline banks and bounded shard streaming; online simulation is outside the
   agreed training design.
5. Extend independent conditional-reference feasibility at selected short and
   longer lengths. Preserve actual sampler provenance, numerical target,
   convergence diagnostics and unsuccessful attempts. No framework name or
   optimizer completion substitutes for a validated reference.
6. Freeze selected candidates before opening fresh acceptance cohorts. Apply
   declared coverage/recovery/risk checks across the target history range, with
   conditional/joint claims limited to the independent comparisons performed.

**Deliverables:** bounded numerical proposal, new immutable task/bank only when
required, reloadable selected fits and source-bound validation reports. No broad
architecture sweep or automatic larger-bank campaign is implied.

**Completion:** selected examples meet the predeclared checks or retain an
explicit development label and narrowed teaching claim. Equal application
status means applying the same evidence discipline to both applications, not
promoting an unresolved result to meet a presentation deadline.

## F. Integrate and rehearse the application sequence

Use the same visible sequence in the application notebooks: task, simulation,
bank/training, checks, inference and interpretation. Add a few assessable
predict → change → inspect → explain exercises using existing controls. Keep
parameter names, colors, interval labels and source labels consistent.

Start with these application activities; they require cached assets and existing
controls, with fresh CPU inference optional:

| Application activity | Participant action | Evidence of understanding |
| --- | --- | --- |
| Heston: define the data | Inspect the current quote/noise plot and contract table before the posterior. | Distinguish the 63 observations, assumed noise, fixed mean reversion and four unknowns; identify the synthetic case rather than SPY. |
| Heston: check and propagate | Compare neural/PyMC marginals and pairs, then the two analytic variance-rate summaries. | Explain why reference agreement tests the approximation under this task, and why a derived quantity depends on paired parameter draws. It does not establish model adequacy on another observation. |
| Rough: change what is supplied | Identify the known curve inputs, then select a joint-curve case and inspect dependence and variance rates. | Explain the known-curve zero rate spread as imposed by conditioning. Recognize that the selected observations and curve families also change, so this is not a controlled numerical ablation. |
| Trajectories: reveal information | Move from one day to 252 and 1,260 days; compare parameter intervals and drift learning. | Describe learning about fixed parameters, distinguish a single path's changing intervals from average contraction, and retain the candidate's calibration limitations. |
| Trajectories: inspect a non-landmark input | Select 137 days and open the input preview; optionally run fresh CPU inference. | Identify exactly 137 observed mask entries, the labelled 126-day cached posterior, and why only the explicit inference action conditions on the newly selected complete prefix. Never multiply overlapping prefix posteriors. |

Across these activities, ask what additional *training* should improve: the
approximation to the declared posterior, rather than automatically narrowing
its uncertainty. Additional *observations* change the posterior target. This
distinction can be taught directly from the application, without expanding the
separate introductory/architecture lessons in this workstream.

Run required area tests, notebook execution/export, Marimo checks, copy parity,
manifest/claims generation and release checks after relevant changes. Explicitly
exercise optional live/model actions separately from headless cached export.
Rehearse with supplied assets on a CPU and without Oscar access. Then decide
what to prune for two hours; no minute-by-minute allocation is required now.

## Parallel execution and current tranche

| Unit | Can start independently | Depends on |
| --- | --- | --- |
| A: inventory, archival and transfer | Yes; active | Existing immutable outputs |
| B: task/evidence labels | Yes; active | Current metadata and predecessor review |
| C: option notebook refinements | Yes; active | B for shared wording; stronger product claims require J2/J3 |
| D1: path task/plots/diagnostic views | Yes | Current caches and reports |
| D2: new simulation/reference views | Partly | Verified model law and matched reference assets |
| E1: numerical diagnosis/proposal | Yes | Existing development evidence |
| E2: generation/training/acceptance | After specification | Only its own task, bank and fit prerequisites |
| F: integration and rehearsal | Incrementally | Completed, independently checked slices |

Use one draft PR for this plan and its first implementation tranche. Preserve
isolated failures: generation depends only on that model's readiness, training
on its own bank, evaluation on its own fit. Packaging and notebook edits do not
depend on successful new scientific training. Report what has landed in the PR,
what is actively transferring and what remains a developed numerical subplan.

The first tranche is operational preservation, option continuity and evidence
labels, path inspection views, and the bounded numerical diagnosis. Training
submission follows its tested specification rather than the completion of an
unrelated notebook or transfer.

### First implementation status

- **A:** preservation is complete: all three full trajectory banks and unique
  historical outputs are off Oscar, verified locally and provider-confirmed
  uploaded to the same Drive folder. The [compact record](../../lab/evidence/pilot-results/2026-11-01/oscar-preservation/README.md)
  binds the inventories and receipts. Original Gmail copy/readback proofs are
  retained; the last five bank parts were copied and hash-verified through the
  verified Brown-account view, followed by fresh provider metadata checks over
  the same Google parent/workshop identities. Old cold payloads were not re-read.
  The cause of the Gmail-path denial remains unknown; sharing permissions were
  unchanged. No unpreserved unique output or nonterminal allocation was
  identified in the audited records. The latest queue retry was blocked by
  expired SSH authentication; the prior successful check remains separately
  bound, with no fresh live-empty-queue claim. No jobs were submitted after
  the pilot. Failed scientific attempts and remote originals remain preserved;
  scientific acceptance is unchanged.
- **B/C:** the option lessons now name their current and historical evidence
  locally, correct stale pending-bank text, retain their controls, add
  interpretation prompts and restore the independent stored Heston
  mean-reversion check. Current cases now connect their exact quotes/noise to
  posterior and analytic mean-variance functionals; the longer historical
  simulation/training studies remain separately labelled. Main application
  parameter and derived-quantity tables use central 90% intervals.
- **D1:** the trajectory lesson now displays actual task/prior metadata,
  prior/posterior densities, paired joint geometry and hash-bound final
  diagnostics, plus an actual observed/padded-input preview using the native
  conditioning function. Simulator/prior-predictive and fuller training
  walkthroughs remain developed follow-ups.
- **E:** the bounded diagnosis protocol and the
  [first source-bound development diagnosis](../../lab/evidence/path-inference/2026-11-01/development-diagnosis/README.md)
  are complete. The latter uses 768 already-inspected development histories,
  binds 27 inputs and supports no production refit yet. An explicit separate
  development-cohort evaluator binds both bank identities and preserves the
  original reserved-test guards; the versioned fresh-acceptance protocol is
  still pending. The isolated
  [Heston numerical admission pilot](../../lab/evidence/path-inference/2026-11-01/numerical-pilot/README.md)
  completed on Oscar and its full source/results are preserved. Projection
  removed zero daily integrated variance in that panel, without establishing
  weak accuracy or selecting a production law. No production refit or new
  acceptance result is claimed.
- **F:** first-tranche checks pass: 514 lab tests (three unchanged optional PyMC
  skips), 29 tutorial tests and all seven cached exports (220 cells, zero errors),
  plus Marimo, manifest/claims and copy parity. Offline checks cover all six
  current option cases and nine path model/length combinations. Independent
  reviews found no code blockers. The [live isolated-Chrome check record](../../lab/evidence/pilot-results/2026-11-01/application-browser-qa/summary.json)
  also covers the
  current option cases, trajectory model/case/joint controls, the actual
  137-day input with its labelled 126-day cache, and 760-pixel layouts. No
  browser runtime errors were found. A display-math delimiter defect found in
  that review was fixed and checked visually. This implementation tranche is
  ready for review. Rehearsal and two-hour pruning follow content refinement;
  the existing versioned workshop kit has not been repackaged with
  these new notebook interfaces.

## G. Collaborator handover before the next feedback round

**Baseline:** merged PR #64, `main` at `08f4ebde1b47893057c61706a1a94ae017806c47`.
The local main checkout was pulled after the merge. The application source tree
matches reviewed commit `517887c`; this plan starts from that delivered work.
The immediate objective is that another collaborator can run, understand and
change the three application lessons, and execute the same numerical workers
on their own supported compute. Notebook-content feedback resumes after this
handover pass. No new scientific campaign is part of the handover.

**Current source ownership:** the follow-up in [section H](#h-tutorial-owned-runtime-refactor)
supersedes the original lab-to-tutorial copy workflow. G0–G5 were delivered in
merged PR #65 (`c1fc88f95a5c3ae52f8d7597d6a17fc15350c271`); G6 remains deferred.
The edit map below reflects the tutorial-owned runtime.

### Audit findings and design decisions

- **Cached review already needs no Oscar.** All three lessons use shipped assets;
  a full bank or native model is unnecessary for the default review. Optional
  model inference and research-scale training need separate downloads.
- **The numerical core is already scheduler-free.** Bank generation, posterior
  training, evaluation and native loading accept filesystem paths. Oscar binds
  deployment/accounting wrappers, dated controllers and campaign configurations.
  Reuse the workers; introduce a thin compute profile only where it is consumed.
- **Documentation conflicts are a real cleanup risk.** Several entry pages still
  exclude returns, call trajectories proposed work, or direct users to Oscar for
  already published models. Correct their authority/scope before retiring code.
- **Navigation and retention are separate.** At this baseline, 1,309 of 2,057
  tracked files live under `lab/evidence/`. The feedback entry point should show
  a small editing surface; reducing that surface does not require deleting the
  numerical history. Historical dependencies still matter: `long_end.py`
  eagerly loads SPY/quick-check assets, and `gbm`/`svj` names include active utilities.
- **Keep one current handover plan.** This section is the execution plan; root,
  tutorial and lab READMEs supply the short entry points. Update those and the
  existing training guide instead of adding more overlapping handover documents.

### G0 · Establish the current front door and review baseline

**Work.** Name the three application notebooks in the root README; identify
introduction/architecture/backbone review and historical studies as supporting
material. Correct stale returns exclusions in active instructions, and mark
older operational guides as historical. Give the current plan priority over
old slot timings and D21. Keep the completed numerical limitations visible.

**Deliverables.** One short root editing map, aligned author instructions and
links to the merged review baseline, existing checks and this plan. This
planning change performs the scope/navigation corrections; it does not claim
that the clean-machine or external-compute rehearsals below have happened.

**Done when.** A new reader can identify all three lessons from the first README
screen and finds no active instruction telling them to remove the trajectory
application. Comments can name an exact source version.

### G1 · Publish the minimal edit and dependency map

**Current editing rule.** The runtime is authoritative under
`tutorial/src/sbi4f_tutorial/`. Application-specific helpers live beside one
another in `heston_options/`, `rough_options/` and `trajectories/`; reusable
numerical components live in `shared/`. Lab workflows import these same
definitions. Older flat names remain compatibility imports, and the former
function-copy workflow is retired. The source-ownership follow-up is recorded
in [section H](#h-tutorial-owned-runtime-refactor).

**Required map.**

| Intervention | First editing location | Source boundary |
| --- | --- | --- |
| Prose, equations, visible snippets, layout and controls | Three application files under `tutorial/notebooks/` | Notebook Python/Markdown; verify displayed code against the executing API. |
| Heston illustration, Fourier smile or posterior prediction | `tutorial/src/sbi4f_tutorial/heston_options/lesson.py` | Authoritative lesson helpers; Fourier model in `model.py`. |
| Rough forward-simulation button/figure | `tutorial/src/sbi4f_tutorial/rough_options/lesson.py` | Authoritative lesson helpers; numerical model/simulator in the same package. |
| Option case selection/tables/corners | `tutorial/src/sbi4f_tutorial/shared/backbones.py` | Shared option replay/presentation. |
| Option adapter/conditional network/native sampling | `tutorial/src/sbi4f_tutorial/shared/full_surface.py` | Shared numerical implementation reused by lab training/loading. |
| Trajectory task definitions and small simulation recipe | `tutorial/src/sbi4f_tutorial/trajectories/simulation.py` | `teaching_task`, `validate_task`, `task_fingerprint`, `generate`; lesson uses the NumPy backend. |
| Trajectory scaling, prefix masks and native inference | `tutorial/src/sbi4f_tutorial/trajectories/inference.py` | Shared numerical implementation; lab retains production training/orchestration. |
| Trajectory cached loading/presentation | `tutorial/src/sbi4f_tutorial/trajectories/lesson.py` | Participant catalogue, replay selection, figures and explicit local-model UI. |
| Contract grid/noise, option tasks and plot style | `tutorial/src/sbi4f_tutorial/shared/{surface,tasks,plots}.py` | Shared runtime components. |
| Workflow illustrations | Existing `lab/evidence/illustrations/*/render.py` | Regenerate and publish with the asset tools. |
| Shipped observations/draws/results | Source study/export plus `accept`, `manifest`, `claims` | Generated artifacts; never hand-edit payloads or hash tables. |

**Done when.** A collaborator can locate the executing implementation behind
one figure/control in each lesson, edit it in `tutorial/`, and validate the
change without updating another numerical copy or reaching Oscar. A displayed
snippet is explanatory code; editing it alone does not alter an executing helper.

#### Follow a computation from the lesson

All paths in the runtime column are relative to `tutorial/src/sbi4f_tutorial/`.

| Visible behavior | Authoritative executing code | Inputs / lab boundary |
| --- | --- | --- |
| Heston path illustration, Fourier quote prices and predictive quotes | `heston_options/lesson.py`: `heston_illustrative_paths`, `heston_smile`, `heston_posterior_predictive`; Fourier model in `heston_options/model.py` | Current saved posterior/reference under `assets/backbones/heston-final500/`; the lab imports the same runtime definitions. |
| Option case selector, posterior table and corner | `shared/backbones.py`: `load_example`, `parameter_rows`, `corner`; cache checks in `shared/full_surface_teaching.py` | Rough cases under `assets/backbones/rough-known*/` or `rough-joint*/`. |
| Explicit rough forward-simulation button | `rough_options/lesson.py`: `price_rough_smile` → `shared/pricing.py` → `rough_options/{simulator,model}.py` | Reads `assets/tasks/calibration_rough_pilot.json`; it does not replace the selected cached posterior. |
| Trajectory teaching data generation | `trajectories/simulation.py`: `teaching_task`, `generate`; Heston physical path kernel in `heston_options/simulator.py` | NumPy backend for small tutorial examples; optional accelerated backend for production. Disk-backed banks and verification remain in the lab. |
| Trajectory case/history views | `trajectories/lesson.py`: `load_example`, `cached_posterior`, `plot_returns`, `plot_evolution` | `assets/trajectories/{heston,rough-known,rough-joint}/`; aggregate diagnostics under `assets/results/trajectories/closeout-20261031/`. |
| Actual trajectory prefix/mask and optional inference | `trajectories/inference.py`: `conditioning_data`, `build_workflow`, `sample`; explicit local loading via `trajectories/lesson.py`: `load_live` | Optional model bytes under `assets/local-checkpoints/path-*/`; lab training reuses the numerical definitions. |
| Shared plot style | `shared/plots.py`: `style` | Historical lab imports remain compatibility entry points. |

Historical dependencies remain real. In particular, `rough_options/long_end.py`
reads SPY and quick-check assets eagerly; collapsing a notebook section does not
make those assets removable. Heston's optional small fit executes shared
`full_surface` APIs; the rough training scaffold remains displayed code until
explicitly run by an author.

#### A small intervention and its checks

- **Prose, math or layout:** edit the relevant notebook's Markdown/cell, then
  run `check`, `test` and `test-notebooks` from `tutorial/`. A prose-only change
  should leave `assets/` and scientific task/model identities unchanged.
- **Application presentation helper:** edit the owning application module or
  `shared/backbones.py`. Run tutorial checks and the lab runtime-boundary check.
- **Shared numerical behavior:** edit the authoritative application/shared
  definition, run its relevant numerical tests, the lab test/release checks and
  tutorial checks. Do not copy a second implementation into `lab/`. A changed
  model/prior/noise/numerical target needs its own specified run; never patch
  frozen metadata to accept old weights.
- **Published result/figure:** edit its producer, regenerate and publish through
  the lab tools, then refresh/check the asset manifest and claims. Keep prior
  scientific evidence and reference scope attached to the original estimator.

### G2 · Document and check scheduler-free compute recipes

**Work.** Turn the existing lab entry points into a clearly named supported
command set. Use explicit `pixi run --locked -e core` for current BF2.1 work;
lab's default environment retains BF2.0 historical studies. `pip install lab`
alone does not install the scientific dependencies. Keep the experiment
configuration (task/prior/noise/seeds/holdouts) separate from data/output paths
and machine resources.

| Operation | Existing scheduler-free route |
| --- | --- |
| Option generation/collections | `scripts/calibration_bank.py` local execution; `scripts/calibration_bank_collection.py` |
| Option posterior/pricer training and evaluation | `train_full_surface.py`, `train_full_surface_pricer.py`, `evaluate_full_surface.py` |
| Trajectory generation/verification | `scripts/path_bank.py` generate, benchmark and check |
| Trajectory training/evaluation | `train_path_posterior.py`, `evaluate_path_posterior.py`; use bounded shard streaming for million-history banks |
| Software readiness | Start with the `core-check` task / `check_full_surface_environment.py`; bounded local smokes use `check_core_cluster.py --allow-local` or `check_path_cluster.py --quick --teaching`, with explicit output paths |
| References | Task-compatible reference CLIs and their actual sampler/environment; older path reference targets do not validate the new inferred-drift tasks |

**Deliverables.** Existing lab README/training guide gains a local CPU route:
locked environment → supplied data verification → tiny generation/fit/reload/
sampling/evaluation → a downloaded-bank training recipe. Examples use caller
paths and fresh output directories. Explain which runs can resume and which
require a new fit; warm-starting weights is not exact state continuation.
No Slurm commands, SSH session or owner-specific account is needed for this path.

**Checks.** On a supported non-Oscar CPU environment, exercise one Heston-option
workflow and all three tiny path tasks, including native reload, arbitrary
prefixes/masks and validation-only evaluation. Bound size, time and host memory;
use distinct smoke outputs and preserve all reserved-test guards. A tiny fit
checks software execution, not posterior accuracy. Verify that commands do not
need `sbatch`, `squeue`, shared-login state or Oscar filesystem paths.

**Supported scope.** Test Linux x86-64 CPU and/or Apple Silicon as the first
concrete target; record the machine actually tested. NVIDIA Linux GPU uses
`core-gpu`, and references use `core-reference`. Windows lab execution, Apple
Intel, object-store paths and other accelerators are additional support work,
not claims of the current handover. POSIX locks/hard links and ordinary paths
remain part of collection-generation requirements.

**Done when.** A collaborator can reproduce the bounded worker chain with their
own paths on a supported machine and inspect a reload/evaluation record, without
changing scientific code or any owner's credentials.

### G3 · Make artifact access and optional inference unambiguous

**Work.** Separate three access levels: cached lesson feedback, optional native
model inference, and full-bank research. Confirm collaborators' own repository/
Drive access once; no workflow should depend on the owner's SSH login or desktop
Drive mount. Use existing immutable downloads and hashes; do not regenerate
successful data to repair a transfer.

**Deliverables.** Consolidate exact archive names, destination layouts and
verification commands in `training-data-guide.md`, linked from the participant
README. In particular document `sbi4finance-pretrained-and-cases-20261029.zip`
and its `pretrained/{heston100k-final500,rough-known100k-final500,rough-joint100k-final500}`
directories. Explain extraction beneath `assets/local-checkpoints/`, unchanged
native metadata/model files, and corresponding UI paths. Retain the already
specified path-model archive and defaults. Full option snapshots must retain
their referenced collection tree; trajectory banks are external sharded files
and must not be eagerly loaded as million-history tensors.

**Checks.** Resolve each advertised archive and example model path from its
recorded inventory; verify identity using the native loader. Run cached lessons
with native models absent; test missing-model recovery separately. Confirm that
opening a notebook starts no download, large Monte Carlo job or training.

**Done when.** A reviewer knows which download is necessary for their chosen
activity, where it goes and how to verify it, using their own access.

### G4 · Keep scheduler deployment thin and optional

**Work.** Reuse the same workers under a local shell or an allocated Slurm job.
Document runtime, project/data/output roots, account, partition/QOS, time/memory
and GPU settings. Existing flags take precedence over adding a configuration
framework. If a reusable profile is useful, supply one small consumed example
with no credentials or user-specific paths.

**Deliverables.** One bounded single-job Slurm recipe with a reviewable command
plan/preflight. Distinguish portable worker commands from Slurm orchestration:
`run_bank_campaign.py` submits jobs despite residing under `scripts/`, and
`submit_path_scaleup.py` is a fixed multi-job campaign rather than a generic
training command. Keep dated Oscar controllers/configs as frozen provenance;
do not replace their paths and present them as fresh deployment scripts.

**Checks.** Test command construction without submission, custom paths/resource
flags and preserved duplicate-submission, source, memory and spending guards.
After the actual target is chosen, perform a bounded allocated GPU readiness/
small-fit check before any full GPU campaign. Local runs report available
local metrics; they must not invent Slurm accounting. Each model's generation,
fit and evaluation depend only on their necessary inputs, not unrelated models.

**Done when.** Deployment changes machine settings and wrapper invocation,
while the reviewed scientific worker/configuration remains the same. This plan
submits no job and authorizes no research-scale campaign by itself.

### G5 · Retire files in small dependency-checked batches

**First pass.** Fix navigation, superseded instructions and broken operational
links. Keep historical numerical records clearly scoped. Exclude generated
evidence from the core editing map instead of introducing another directory
layout or duplicating the tutorial into a new package.

**Second pass.** Classify candidate dated controllers, duplicate operator copies
and bulk historical outputs. For each candidate record its consumers, retained
source commit/archive, evidence bindings and replacement link. Check imports,
CLI/pixi tasks, tests, runtime-boundary checks, manifests, claims and Markdown
links before deleting or moving it. Retire source through Git history/pinned
references and the existing verified external archives; do not rewrite old
scientific hashes or remove failures to make the tree look cleaner.

The first bounded retirement batch covered
`lab/cluster/start_known100k_20261028.py` and
`lab/cluster/start_joint100k_20261029.py`, followed by duplicated evidence-bound
operator copies. Their absence from current worker imports alone is not enough:
check source/evidence bindings before removing them from the working tree.

**Protected by default.** Current notebooks and their full dependency closure;
shipped cached/reference assets; native format/loaders; canonical generation,
training and evaluation; accepted evidence and unresolved failures; publication
and verification tools. Collapsed notebook sections still execute their imports.
Do not remove `gbm`/`svj` or path code simply because a directory name looks old.

**Done when.** Each removal batch has a positive dependency/retention argument,
all affected checks pass and a clean checkout still supports the advertised
workflows. Stop rather than introduce new numerical abstractions solely to
justify removing a few files. No mass rename, source move or evidence deletion
is performed by the planning change.

### G6 · Rehearse the handover and collect actionable feedback

**Feedback route.** From a fresh checkout of the named baseline, install the
locked tutorial environment, run its checks and open Heston → rough options →
trajectories. Inspect quote/noise and posterior-predictive views; switch known/
joint curves; reveal 137 days and explain the 126-day cache; locate failed
calibration and the Heston limitation. Make one small notebook-only edit and
verify it using the documented checks. HTML exports are static previews; the
reactive review uses marimo. No Pair/Studio add-on is needed for this minimum.

**Feedback format.** Record source version, platform, notebook and section
heading, selected model/friendly case and relevant controls/history length,
expected/observed behavior, and a suggested improvement. Include error text or
a screenshot when useful. Distinguish a teaching/presentation comment from a
runtime bug or a proposed numerical change. GitHub review comments or the
existing guided-review notes host the feedback; no automatic collaborator
messages/invitations are sent by this workstream.

The feedback form is ready to use when rehearsal resumes:

```text
Source version / platform:
Notebook and section heading:
Mode (cached / optional live / author recipe):
Model, friendly case, history length and relevant controls:
Expected behavior or learning point:
Observed behavior / error text:
Suggested change (teaching / UI / numerical):
```

**Rehearsal status:** deferred at the user's request. Author-side software,
artifact and worker checks below do not count as a collaborator rehearsal.

**Release gates.**

- **Feedback-ready:** aligned scope, three-file entry map, own-machine cached
  walkthrough and a small edit validated without Oscar or large downloads.
- **Compute-ready:** scheduler-free recipes, usable artifact instructions and
  a bounded supported non-Oscar CPU worker rehearsal. Any untested machine or
  GPU deployment remains named as untested.
- **Cleanup-ready:** dependency/retention evidence exists for each proposed
  deletion; the same clean-checkout walkthrough passes afterward.

Publish one named review baseline and its small check record once these gates
are met. Begin the next content-feedback round from that baseline. Numerical
improvements remain separately specified; the handover cannot turn failed
trajectory calibration into acceptance.

### Order, parallel work and current status

| Slice | Can run in parallel | Prerequisites |
| --- | --- | --- |
| G0 scope/front door | First, small documentation change | Merged main verified |
| G1 core edit/dependency map | With G2/G3 | G0 scope |
| G2 scheduler-free recipes/smokes | With G1/G3 | Existing workers, declared supported target |
| G3 artifact/model path clarity | With G1/G2 | Existing download inventories and native loaders |
| G4 optional Slurm profile | With low-risk G5 cleanup | G2 worker contract; actual target for allocated test |
| G5 source retirement | After dependency mapping, in small batches | G1 and retained provenance; G6 repeated after each batch |
| G6 collaborator rehearsal | Incremental; final gate after relevant cleanup | G0–G3 minimum, target-specific deployment only if advertised |

### Implementation status · 1 November 2026

The implementation is recorded in the [compact software/artifact check record](../../lab/evidence/pilot-results/2026-11-01/collaborator-handover/summary.json).
These author-side checks do not stand in for the deferred collaborator rehearsal.

| Slice | Implemented and checked | Scope still explicit |
| --- | --- | --- |
| G0 | Three-file front door, current trajectory scope and historical-guide designation. | The selected source version still belongs in each feedback report. |
| G1 | Named launch tasks, concrete source walks and the original source-copy boundary/check matrix. | Section H supersedes source-copy ownership; deep historical branches retain their actual dependencies. |
| G2 | Caller-path CPU recipes and all 16 documented generation/check/train/reload/validation CLI commands passed on Apple Silicon, with SSH and Slurm state absent. Heston recovery and all three path mask/reload readiness checks passed. | Short software tasks and tiny fits check execution, not posterior quality. The existing verified locked runtime was reused; no fresh installation is claimed. |
| G3 | Two existing model archives verified/extracted; all six native task/model identities match the publication records and tutorial caches. Exact archive, extraction and UI paths are documented. | Each collaborator needs their own repository/Drive permission. Model draws and full-bank downloads were not repeated for this check. |
| G4 | Reusable CPU/GPU wrappers have no owner account/partition defaults. Caller flags, allocation guards, GPU preflight and bounded print-only single-job recipe are documented/tested. | No Slurm job was submitted and no new cluster/GPU deployment is claimed tested. A target allocation still needs its own readiness check before a full campaign. |
| G5 | Two completed controllers retired with exact pinned Git/archive recovery and a dependency/retention ledger. Distinct operator versions and current tested dependencies retained. | No immutable evidence or scientific result was deleted; further retirement needs a separate dependency proof. |
| G6 | Feedback form and rehearsal checklist are prepared. | **Deferred at the user's request; no collaborator rehearsal performed.** |

The two retired controllers and their retained sources are listed in the
[numerical-plan retirement record](../../lab/plans/NUMERICAL-EXPERIMENT-PLAN.md#bounded-retirement-batch--1-november-2026).
The user requested the source-ownership refinement below before the next
feedback round. Collaborator rehearsal remains deferred and can begin when
requested from the resulting review baseline. No additional scientific bank or
training campaign is part of this organization work.


## H. Tutorial-owned runtime refactor

**Baseline:** merged PR #65, `main` at
`c1fc88f95a5c3ae52f8d7597d6a17fc15350c271`. The user requested that code needed
to understand, run and refine each tutorial be owned where collaborators find
the lesson, with updated documentation and handover guidance.

### H0 · Give each application a clear source home

Keep the three notebook filenames and named launch tasks stable. Group Heston
helpers in `heston_options/`, rough-option helpers in `rough_options/`, and
trajectory helpers in `trajectories/` under `tutorial/src/sbi4f_tutorial/`.
Separate trajectory replay/UI code (`lesson.py`) from native inference
(`inference.py`). Own physical-measure task definitions and small trajectory
simulation in `trajectories/simulation.py`, with a NumPy teaching backend and
optional accelerated production backend; Heston physical paths use
`heston_options/simulator.py`. Shared numerical, cache and plotting utilities live once in
`shared/`. Existing flat module names are compatibility imports only.

**Acceptance:** the root editing map, tutorial author map and G1 computation
walks all point to the same executing implementation. The tutorial remains
self-contained, with cached opening independent of neural initialization.

### H1 · Make the runtime authoritative and reuse it from the lab

Edit shared implementations directly in their tutorial package. Lab workers
import/re-export the same definitions while keeping lab-only generation,
training orchestration, references, exporters and studies. Retire source-copy
mutation: `runtime-check` checks the dependency/ownership boundary;
`copy-to-tutorial` only explains the migration and cannot overwrite source.
Historical `bs_hierarchy` and `long_end` quick-check programs retain their
original study input/global contracts and numerical parity tests; current lesson
edits belong in the tutorial runtime, not those preserved study programs.

Keep base runtime requirements separate from the participant neural/notebook
stack. `pixi install --locked` in `tutorial/` selects the participant stack;
manual participant installation uses `pip install -e ".[participant]"`.
Lab installs the sibling base runtime with its own recorded current/historical
BayesFlow environment. Deploy the complete checkout, with `lab/` and
`tutorial/` from one revision visible to workers.
The lab source-checkout import guard rejects missing or differently located
editable runtimes before numerical imports, with the expected/actual paths and
locked reinstall guidance. Installed distributions use their declared package
dependency; the source-layout guard is specific to deployed/editable checkouts.

**Acceptance:** one authoritative implementation per reused definition; no
tutorial imports from its lab consumer; current and historical
lab environments keep their intended neural versions; release/runtime checks
reject ownership drift.

### H2 · Preserve data, model and producer provenance

Move source without changing the mathematical model, prior, noise law, task
contracts, conditioning inputs, native sampler or saved asset/model bytes.
Generation provenance includes the actual reused numerical sources. A source
move changes the producer identity even if numerical outputs are equivalent.

Existing downloaded banks remain readable and usable for a new fit. Extending
or resuming their generation requires their producer-matching historical
checkout/environment. Do not rewrite a frozen manifest to fit the new source
layout. Newly generated banks record the new source identity. This refactor
launches no scientific campaign and creates no new accuracy claim.

**Acceptance:** cached asset hashes and native task/model identities unchanged;
seeded numerical outputs and load/sample behavior equivalent; new producer
records bind the executing runtime; frozen producer-mismatch guards remain.

### H3 · Update handover and validate the route

The short [collaborator handover](../README.md#collaborator-handover) is the
shareable entry point. Root/tutorial/lab READMEs and repository instructions
must agree on ownership, installation, editing and checks. Publication of data
and figures remains separate from source edits.

Required checks cover tutorial independence/tests, every notebook's headless
execution and marimo checks, lab tests/release/runtime boundaries, native model
loading and bounded generation/training/evaluation workers. Reuse the existing
supported environment and record which platforms/routes were actually checked.
Do not describe a fresh install, GPU/other-cluster deployment or collaborator
rehearsal as tested unless it was performed.

**Status:** implemented and checked on Apple Silicon CPU in newly installed
locked worktree environments, reusing the existing package cache. The
[compact validation record](../../lab/evidence/pilot-results/2026-11-02/tutorial-runtime-refactor/summary.json)
records 527 lab tests in core (three optional PyMC skips), 527 in the historical
environment (21 environment-specific/optional skips), 36 tutorial tests and all
seven notebook exports (220 cells, zero errors). All 183 registered shared lab
symbols reuse their tutorial definitions. Manifest, claims, marimo and runtime
boundary checks pass; no shipped asset bytes changed.

An independent pre/post comparison reproduces 62 arrays exactly. All six
published native estimators load with unchanged task/model identities and
produce exactly matching seeded draws. The participant-only simulation →
one-epoch training → inference recipe works for all three trajectory models,
without lab or Numba imports (eight 29-day histories per model; about 15 seconds
and 703 MiB peak process memory). This is a bounded software check, not a new
estimator or accuracy result. Four native BF2.1 tests now explicitly require
that environment and remain executed in core.

No GPU/other-cluster deployment or fresh collaborator rehearsal was performed.
The collaborator rehearsal remains deferred at the user's request. Numerical
limitations and failed trajectory calibration screens retain their original
scope throughout this change.

## Guided notebook review · 1 November 2026

The Heston review establishes the presentation contract for the remaining
application lessons: define the model and data locally; pair a short intuitive
statement with the relevant mathematics; show syntax-highlighted core workflow
code by default; reserve implementation accounting for optional source panels.
The prerequisite is probability/Bayes and Python, rather than prior expertise
in option calibration. Timing and pruning remain deferred.

### Heston feedback implemented

The [source-bound check record](../../lab/evidence/pilot-results/2026-11-01/heston-lesson-review/summary.json)
records the final code, exports and browser review for this revision.

- Explain the pricing-measure Heston equations, all four inferred parameters,
  fixed mean reversion and variance versus volatility before showing inference.
  Show coupled price/volatility paths and the update code in the main flow.
- Illustrate the construction of a training pair: prior draw, Fourier expected
  option prices, prescribed quote error and observed price vector. Distinguish
  path randomness, quote noise, numerical pricing error and bid–ask spreads.
- Explain both neural routes visually and mathematically: a forward price
  surrogate plus a new sampler for each observation, and an amortized posterior
  conditional on observations. Original diagrams use the inspected Overleaf
  deck's palette, with editable sources and attribution retained.
- Show the simulator, adapter/network, offline fit and posterior-draw APIs in
  highlighted code by default. Plotting and input validation remain helpers;
  visible snippets disclose that division rather than claiming the financial
  pricer itself is only a few lines.
- Add a synthetic posterior-predictive illustration from saved joint neural and
  reference draws. Distinguish model-price uncertainty from replicated quotes
  with fresh Gaussian errors; label the bounded empirical bands and in-sample
  scope. No learned model or posterior cache is replaced.
- Explain recovery, coverage, SBC and reference comparisons with both slogans
  and formal definitions. Describe historical contraction in its actual logit
  coordinates rather than as a physical interval-width reduction.
- Replace standardized historical comparison panels with four selectable,
  descriptively named cases in original parameter units, alongside illustrative
  Heston paths and saved observed quotes. All four cases/methods/truths remain.
- Audit the displaced truth in case 2001. [The source-bound diagnosis](../../lab/evidence/pilot-results/2026-11-01/heston-case-2001/summary.json)
  reproduces its declared simulation and finds a likelihood shift consistent
  with its quote errors. Retain its importance-weight and reference limitations;
  a single case neither establishes nor refutes aggregate calibration.
- Explain strike/forward and premium/forward with a currency example and the
  put/call switch at the forward. Hide technical case IDs and hashes behind a
  reproducibility panel while keeping scientific limitations visible.

### Rough options and trajectories: shared presentation contract

The [source-bound application check record](../../lab/evidence/pilot-results/2026-11-01/application-lesson-review/summary.json)
binds the final notebook/runtime sources, tests and cached-control browser review.

The same contract now applies to the two remaining core applications. The
opening establishes the model and observed data before posterior plots, and
short highlighted code stays visible throughout the generative and Bayesian
workflow. New original diagrams use the inspected Overleaf palette and retain
source, editable SVG and attribution. Parameter tables and corner axes use
readable names/symbols; technical identities remain optional source details.

**Rough options.** Explain the Volterra-memory variance model, its parameters
and the forward-variance curve. Distinguish many latent paths averaged into
numerical prices, finite-path pricing error and separately prescribed Gaussian
quote errors. Supplied seven-level and jointly inferred exponential curves
remain different tasks and curve families. Friendly selectors retain every
case. Reference, calibration, importance-audit and product uncertainty each
receive an intuitive explanation and their mathematical scope. Current neural
examples do not inherit historical validation or become a matched comparison
when the selected data change. Explicit forward simulation is not relabelled
as posterior prediction.

**Trajectories.** Explain observed daily log returns, hidden variance/shocks,
physical drift and each model's equation locally. Show the actual generation,
offline training, random-prefix/mask and GRU/conditional-flow APIs, distinguishing
a small author example from million-history streaming production. The revealed
history and mask are visible; detailed tensor/accounting fields are optional.
All integer lengths from 1 to 1,260 remain supported. The 137-day input versus
126-day cached posterior is explicit, and prefixes use the original prior and
full revealed data. Generating labels and hidden states remain excluded. Model
switches change the synthetic observation; the display does not pretend to be
a matched cross-model experiment.

Calibration explanations distinguish concentration, recovery, coverage and
prior-averaged ranks. All failed trajectory marginal screens and the Heston
atomic-drift limitation remain visible in plain language. New independent
replicate histories and continuation forecasts are distinct; the latter need
latent-state and rough-memory conditioning beyond a parameter posterior.
No new posterior-predictive trajectory cache or conditional reference is
fabricated. These interface and teaching changes replace no scientific task,
bank, checkpoint or inference result.

## Historical review: 26 October 2026

The following assessment records the earlier option-focused proposal. Its slot
scope and extension decisions are superseded by the execution plan above.

26 October 2026. Review of the course and curriculum proposed in PR #44 at
`cebfb1f`. This assessment assumes that Heston and rough Bergomi remain the two
core models.

## Verdict

**Keep the broad learning path.** Black–Scholes gives a small problem whose
posterior can be checked directly; Heston makes the complete amortized workflow
checkable against a numerical likelihood; rough Bergomi preserves the financial
question while making pricing and inference substantially more expensive. This
is a relevant application of simulation-based inference (SBI).

The main weakness is the argument for *why amortization is useful*, rather than
the model choice. Bayesian uncertainty, an expensive pricing model and a fast
neural posterior do not by themselves establish that amortized inference is the
best tool for a particular workload. The tutorial will be stronger if it teaches
that decision openly, including situations where another method is sufficient.

The [curriculum overview](00-overview.md) provides a coherent sequence.
The sixteen-lecture [course](../../course/README.md) should remain supporting
material, with only the concepts needed for the live task introduced on stage.

## Follow-up self-review: keep the tutorial small

The plan had accumulated a tutorial release, a broad architecture benchmark, sparse-chain support and direct rough-inference research. Separate them. The selected core teaching path is complete-grid Heston with four inferred parameters, followed by complete-grid rough Bergomi with six jointly inferred parameters and a Heston-shaped initial mean-variance curve. Retain the bounded four-family/encoder screen and confirm only the two selected configurations across three seeds. Complete data, training, reload, inference, numerical/reference checks, product uncertainty and participant packaging before starting sparse-chain work. No quote-subset adapter, mask-conditioned training or arbitrary-chain support is a prerequisite. One bounded check against the published hump-capable curve family precedes substantial rough generation.

Sparse-chain inference is a separate downstream unit of work that needs its own input/layout, missingness, training and validation specification before implementation. Preserve the existing [case plan](../../lab/plans/SPARSE-QUOTE-CASE-PLAN.md) as its starting point, rather than promising a sparse lesson in this release. Supplied-curve rough fits, runner-up campaigns, direct rough MCMC and broad reserve-coverage claims remain separate extensions.

The incremental-bank and recovery safeguards address real duplicate-ID, interrupted-write, memory and changed-dataset failure modes. Keep the tested primitives rather than rewriting them to reduce line count, and stop expanding the infrastructure unless a pilot exposes a concrete need. Collection/recovery machinery belongs in instructor tools; participants should use supplied data/checkpoints and follow the inference workflow. The infrastructure's passing mocked tests are not scientific validation. Prioritize actual pricing refinement, independent holdouts, calibrated/conditionally checked posteriors, matched baselines and an honest product/cost figure over more framework work.

The [central plan](../../lab/plans/NUMERICAL-EXPERIMENT-PLAN.md#current-next-work-and-repository-trimming) and [protocol](../../lab/plans/EXPERIMENT-PROTOCOL.md#training-comparisons) reflect the reduced programme. No new tracking system or additional planning document is needed.

## What already works well

- **One financial question links the examples.** Inferring a calibration and
  propagating it into an unquoted product is a concrete reason to care about a
  joint posterior. It gives the audience something more useful than parameter
  recovery alone.
- **Heston is an effective control experiment.** The numerical likelihood lets
  us investigate the neural approximation separately from the financial model.
  The recorded conditional failures, alongside successful marginal calibration,
  are valuable teaching material.
- **Rough Bergomi supplies a real computational problem.** Repeated Monte Carlo
  pricing makes the trade-off between simulation, training and repeated inference
  tangible. Training data, computation and accuracy all have visible costs.
- **The evidence ledger is unusually useful.** The
  [study status](../../course/status.md) and [claims ledger](../CLAIMS.md) distinguish
  completed studies, pilot results and pending evidence. Retain this discipline
  when new architectures and BayesFlow versions are added.

## Three arguments that must stay separate

**Why use a posterior?** Weak identification, nonlinear transformations and
uncertain inputs can make one fitted calibration an incomplete description of a
product's value. However, a wider Bayesian interval is not automatically a better
one. Its width depends on the prior and observation model. On well-identified
liquid surfaces, agreement with a classical local interval is a useful result.

**Why use SBI?** Learning an inverse map from simulations is attractive when
likelihood-based inference is expensive or awkward. Here, the option observation
likelihood is specified; its mean requires numerical pricing. Say explicitly that
the difficulty is repeated expensive pricing. This distinguishes the application
from a simulator for which no evaluable observation likelihood is available.

**Why amortize?** The strongest case is a recurring workload: many smiles, dates,
scenarios or posterior queries within the declared task. Making thousands of
validation posteriors affordable is useful, but partly circular as the sole
justification: many of those evaluations are needed to validate the amortizer we
introduced. Put an operational workload and an honest break-even calculation
beside it.

[Slot 2](02-slot2-amortized-bayes.md) should compare total costs against
both a good classical proposal with importance sampling and a validated neural
pricer with MCMC. The cost must include bank generation, training, warm-up and any
audit applied to each query. The
[course plan, Section 2.4](../../course/v2-plan.md#24-why-simulation-based-inference)
already acknowledges these competitors; make that comparison central on stage.

If either competitor matches the accuracy and total cost of the intended
workload, the tutorial remains a useful demonstration of SBI. The conclusion
becomes “a practical alternative with measured trade-offs,” rather than “SBI is
necessary.” A campaign that can reach either conclusion is more convincing.

## Make the information available to each method visible

The core plan has complete-grid Heston with four targets and complete-grid rough
Bergomi with six targets. Each receives all 63 declared contract prices and its
noise information; missing or invalid entries are not silently filled to pass
this input contract. The rough model has H, eta, rho and three coefficients of
an initial Heston-shaped variance curve, with no supplied curve. Heston already determines its
mean curve from its parameters; using that functional family in rough Bergomi
does not make the stochastic variance processes equivalent. Heston currently
fixes kappa at 2 while rough curve decay is inferred, so match those assumptions
in any controlled cross-model diagnostic and avoid attributing all differences
solely to roughness.

The existing course's supplied-curve and seven-level examples remain historical
or optional comparisons. Holding a market-reconstructed curve fixed gives
conditional intervals that omit curve-estimation uncertainty. In the new joint
task an unquoted segment is constrained globally by the coefficient prior and
exponential shape. Explain that source of information, including extrapolation;
it is not direct evidence learned from unavailable quotes. Near-flat curves can
leave decay weakly identified even when curve bands and product values are stable.

A plug-in curve reconstructed from the same quotes is a conditional modelling
approximation. Do not describe it as additional independent data. Show how
plausible curve perturbations change the result before interpreting the interval
as a full description of calibration uncertainty.

Use matched priors, observed information, noise and numerical accuracy when
comparing methods. Include a small prior/noise sensitivity check in the core
full-grid examples. In the separate sparse-chain unit, the alternative laws for the unquoted curve level in
[Lecture 16, Sections 16.4–16.5](../../course/16-a-thin-chain-ewg.md)
are important. Keep the favourable prior-generated case and the deliberately
misspecified cases together; neither alone settles comparative performance.

The option-noise model deserves an early, short explanation. A complete synthetic
design is not a guarantee that a raw market chain supplies every contract.
Any retained market example must label how its complete grid was constructed.
Independent noise
on interpolated grid values is a modelling choice, not a property of the source
quotes. The likelihood should not gain artificial precision merely because a
smoother was read at many points. The actual-quote EWG audit and noise-scale
sensitivity help address this concern, but cannot establish the model's general
market adequacy from one date.

## Tighten claims that could undermine an otherwise good lesson

- **Reference accuracy:** describe Fourier/MCMC results as checked numerical
  references. Label surrogate-MCMC and finite Monte Carlo targets precisely.
  The current rough-MCMC cost extrapolation is not a measured completed chain.
- **Model criticism:** violating Heston's Feller condition does not invalidate
  the model. [Lecture 10, Section 10.2](../../course/10-heston.md)
  correctly explains this. Carry that distinction into the live handoff and use
  product or predictive discrepancies as the substantive evidence. Fixing mean
  reversion at two also restricts the Heston comparison. Show the existing
  mean-reversion profile or an appropriate sensitivity control before attributing
  the mismatch to the whole Heston family or arguing that roughness is necessary.
- **Uncertainty:** the proposed product quantile is a conditional, within-model
  uncertainty measure. The course's
  [three ranges](../../course/06-marking-a-product-the-smile-does-not-quote.md#67-three-ranges-and-baselines-that-fix-what-the-quotes-do-not)
  explain why it does not cover every source of valuation uncertainty. Avoid
  implying that one posterior quantile automatically establishes a complete
  prudent-valuation or regulatory-compliance procedure.
- **Validation:** SBC, conditional reference agreement, posterior predictive
  checks and importance-weight diagnostics answer different questions. Preserve
  their separate outcomes, including failures; a successful correction must not
  conceal a poor raw amortizer.

## Make the live path smaller than the research programme

The curriculum's 115 minutes of talks can fit a half-day, but the remaining time
needs an explicit allocation. One workable proposal is 60 minutes of exercises,
20 minutes of breaks and 15 minutes of questions: 210 minutes in total.

| Part | Main learner action | Keep outside the essential live path |
|---|---|---|
| Slots 1–2 | Predict how changing one quote changes the toy posterior; compare the exact and learned answers | A survey of every classical SBI method |
| Slot 3 | Distinguish the observation encoder from the posterior estimator | Detailed derivations and every architecture in the benchmark |
| Slot 4 | Trace input → adapter → posterior → reference check → product | Full training and every historical architecture experiment |
| Slot 5 | Predict which product or curve component is weakly identified; inspect the result and its diagnostics | Several products, several market cases and every audit variant at once |
| Slot 6 | Change one prior or noise assumption and explain the effect | A second full workflow or several additional trained detectors |

Use one primary downstream product to carry the complete-grid inference story;
a second product can be a control. Teach identification through posterior
dependence and prior/noise sensitivity while keeping every declared quote
available. A suitable market case can illustrate the workflow if its preprocessing
and observation assumptions are explicit; it is not required to introduce sparse
support. EWG remains historical evidence for the downstream sparse-chain unit,
whose specification starts from the [case plan](../../lab/plans/SPARSE-QUOTE-CASE-PLAN.md).
Keep architecture comparisons in a compact results
panel and the take-home material. Each participant should finish with one
sentence naming the data, unknowns, inference approximation, validation result
and downstream quantity.

## What would make the tutorial strong enough to release

The learning path is strong enough when a participant can reproduce the Heston
example, interpret the rough example's evidence, and explain when amortization
is worth its cost. Require reproducible checkpoints, scoped reference comparisons,
numerical-error checks, explicit assumptions, a timed rehearsal and a cached
fallback for each expensive demonstration.

Sparse-chain outcomes do not gate this release. If the full rough checkpoint misses
its checks, label the advanced material as an unfinished research case and narrow
the advertised hands-on scope. That does not require replacing the broad
learning path; it requires changing the claims made about it.
