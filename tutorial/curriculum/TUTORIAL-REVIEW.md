# Application notebook refinement: execution plan

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

- **A:** all three trajectory banks are downloaded; every archive part and all
  23,631 native shards pass integrity checks. All historical logical payloads
  and source snapshots are verified off Oscar. No unique output remains only
  on scratch in the audited scope. The [local verification record](../../lab/evidence/pilot-results/2026-11-01/oscar-preservation/LOCAL-VERIFICATION.md)
  binds these results. Drive completion is pending because macOS denies
  destination file and identity-metadata access; strict resume monitors retain
  the original checks. No scientific job is active or being retried as a
  preservation task.
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
- **F:** first-tranche checks pass: 505 lab tests (three unchanged optional PyMC
  skips), 29 tutorial tests and all seven cached exports (192 cells, zero errors),
  plus Marimo, manifest/claims and copy parity. Offline checks cover all six
  current option cases and nine path model/length combinations. Independent
  reviews found no code blockers. Rehearsal and two-hour pruning follow content
  refinement; the existing versioned workshop kit has not been repackaged with
  these new notebook interfaces.

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

The [central plan](../../lab/plans/NUMERICAL-EXPERIMENT-PLAN.md#tutorial-scope-after-self-review) and [protocol](../../lab/plans/EXPERIMENT-PROTOCOL.md#training-comparisons) reflect the reduced programme. No new tracking system or additional planning document is needed.

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
