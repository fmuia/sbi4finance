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
6. Provide posterior-predictive replicated-history checks (implemented below). Continuation forecasts
   and future VaR/ES require joint latent-state/history conditioning, including
   rough memory; a parameter posterior alone is insufficient for that extension.
7. Offer an explicit recovery/rank check of the workflow actually trained or
   loaded. Generate independent validation histories once per task and reuse
   them across checked prefixes; keep this separate from the final-cohort tables.

**Completion:** participants can follow generation → training → inference →
checks → interpretation for all three tasks. Numerical readiness is governed
by subplan E; a working interface never implies an acceptance certificate.

### Trajectory revision plan · 7 November 2026

Carry over the direct BayesFlow workflow and concise prose from Stefan’s Heston
revision. Keep the sequence model and offline-prefix training appropriate to
ordered daily returns rather than copying Heston’s architecture or training budget.

1. Start with one rough-known history and the inference question. Bring the
   observed data forward; keep priors, numerical settings and other model
   variants available without making them the opening reading task.
2. Connect simulation, adapter, causal GRU summary, conditional flow, random-prefix
   training and sampling in one visible walkthrough. Move million-history
   bank/shard commands into reproduction details. A bounded small fit must
   use fresh independent pairs and display only its own session results.
3. Use saved history landmarks in cached mode. Offer arbitrary integer lengths
   beside explicit live inference. Fresh draws must update the same posterior
   table, marginals and joint plot, with task/history/length changes invalidating
   stale results; missing models leave a clearly labelled cached fallback.
4. Keep one mask illustration and the distinction between a genuine zero return
   and padding. Move tensor previews and detailed scaling explanations aside.
5. Keep recovery versus coverage, one clear calibration-failure verdict and a
   useful diagnostic figure in the main flow. Preserve detailed failure tables,
   Heston’s drift atom, the lack of a matched conditional reference and the
   distinction between independent replication and continuation forecasts.
6. Update the handover and this plan. Check saved/live modes, all models and
   history choices, missing-checkpoint behavior, plots and narrow layouts.

**Acceptance:** the plotted posterior identifies the returns it actually uses;
all posterior views share the same paired draws; cached viewing needs no backend
or downloads; optional fits remain teaching exercises, not scientific acceptance.
Run tutorial tests, marimo checks and all notebook exports, plus affected lab
checks and release checks where runtime interfaces change. Models, banks and
recorded numerical results remain unchanged.

**Status:** implemented. The lesson introduces the model before its observed
synthetic history, offers actual
saved lengths by default and places exact-prefix live inference beside the
posterior. Tables, marginals and the evolution endpoint share the
selected paired draws. Selection changes clear fresh results; missing models
leave the labelled saved fallback. A separate five-year exercise generates 128
fresh histories and trains for eight optimizer updates. Production and detailed
input/diagnostic material are optional; all scientific limitations remain.

The early forward simulator now exposes parameter controls for all three models.
It uses eight five-year paths with fixed innovations, showing price paths, daily
returns and a return-based realized-volatility summary. It shares the canonical
path implementation with training-pair generation. Extracting that shared path
block preserves all 54 pre-edit check arrays exactly across three models, three
splits and two offsets. No new bank identity or latent inference channel is added.
The generating source bytes have changed; extending a preserved bank still needs
its matching producer checkout and environment.

The architecture now separates the causal GRU encoder from the spline coupling
flow in a source-rendered diagram. The visible primary route constructs the
16-unit/16-summary/two-block tutorial model and performs eight updates. The
optional pretrained route loads the separate 64-unit/32-summary/six-block model.
Each route labels its own outputs, and the code shown matches the executed settings.
Component panels now explain P versus Q, Gaussian innovations, the actual
left-endpoint daily calculations and the supplied/inferred P variance-scale
curves. One observed history is one data set, not a Monte Carlo price average.
A separate input concept diagram traces return/time tokens, masks, the GRU
and flow conditions, with parameter labels used only in the training loss.
The simulator uses stated fixed y-axis presets for all parameter settings.
Manual scale changes alter only the view; outside counts and complete data
ranges make view clipping explicit without dropping any path or return.

### Presentation scope · 14 November 2026

Remove the standalone parameter-dependence explorer (former 4.2), independent
network checks and recovery-case controls (former 5.2), and the drawdown
exercise (former 6.1) from the participant notebook. Remove their UI cells,
unused controls and notebook-only cell tests, while retaining shared numerical
helpers and their tests. The separate MMAR take-home keeps its drawdown story.

The core flow keeps posterior marginals and interpretation, evidence playback,
posterior-predictive checks and saved final-cohort evidence. Playback becomes
4.2, saved evidence becomes 5.2, and the retained continuation discussion becomes
6.1. The Use banner describes forecasting requirements rather than promising
a scenario exercise. Preserve the explicit failed-calibration verdicts.

The dated records below describe earlier revisions; live cohort diagnostics
and the drawdown exercise are superseded by this presentation scope.

### Align the lesson with the SPX workflow · 12 November 2026

**Presentation alignment · 14 November:** match the current Heston/SPX page at
`d6c3c2c`: one linked workflow map, numbered six-stage banners with short leads,
completed/current/upcoming stage links, and numbered subsection headings with
anchors. Use its typography, spacing, rules, tables and foldable-detail styles
from a tutorial-local stylesheet. The data introduction has the same unnumbered
"Before the workflow" banner. Code, controls and model behavior are preserved;
the iframe players retain their own presentation. Setup code stays available
in the editor while being folded in the reading view.

Use the useful structure from the SPX Heston notebook without transferring its
data, estimator or accuracy claims. Keep the observed-history opening, then
mark each workflow stage and map its BayesFlow calls. A compact design table
explains why ordered returns use a masked causal GRU, a spline coupling flow,
fixed scaling and prior-CDF parameter transforms. Quote whitening and the
SPX flow-matching recipe are not changes to this task.

Show one named, executing tutorial-training function with `mo.show_code`.
It generates 128 training histories and a separate validation set, performs
two epochs/eight updates, plots native BayesFlow training/validation loss and
infers parameters for an independent example. Continue using the canonical
preprocessing and network helpers. The supplied estimator remains a separate
source with its own task, weights and failed final calibration screen.

Move checks after the posterior, dependence and evidence-evolution views.
Add questions beside results and a short table separating what each check
answers. The 200-history diagnostic panel gains empirical coverage with
binomial intervals and a selected
case view of paired posterior draws. A selected validation case is not a
matched reference posterior, and repeated inspection does not make it a new
acceptance cohort.

Add two explicit model checks through `trajectories/model_checks.py`:

- **Prior predictive:** simulate 200 independent P histories from the declared
  prior and compare return summaries with the selected observation.
- **Posterior predictive:** keep 32 full parameter rows paired, simulate
  independent histories from the model's origin and compare the same summaries.
  Use the displayed supplied-model posterior, saved or fresh, at its actual
  prefix length, including a labelled saved fallback. The tutorial fit remains
  a separate source.

The observed history does not supply future shocks or hidden states. These are
replicated histories, not filtered paths or continuations. They do not provide
conditional return, VaR or expected-shortfall forecasts, and neither a plausible
replicate nor a passed rank check establishes posterior accuracy. Preserve all
existing native-model failures and the Heston drift point-mass limitation.
The synthetic inference examples retain the existing weights and reference
scope. A separate empirical motivation panel is described below.

The final interpretation exercise now propagates complete paired parameter
draws into independently replicated drawdown. Compare it with process
simulations at the posterior-mean parameter vector. Both groups retain process
randomness and use the actual displayed posterior's conditioning length.
These are fresh-origin scenarios, not conditional forecasts.

### Clarify physical features, learned summaries and finite checks

Carry over the helpful teaching patterns from
[Heston/SPX PR #89](https://github.com/fmuia/sbi4finance/pull/89) at `224b057`
and the published learned-summary experiment in
[PR #91](https://github.com/fmuia/sbi4finance/pull/91) at `1dcd9e3`.
Keep the trajectory P law, existing weights and numerical task unchanged.

Four responsive cards explain physical price drift versus expected log growth,
volatility bursts, leverage, and variance level/memory for the selected model.
The forward controls use predict → move one setting → observe prompts, with
fixed innovations and the existing explicit axis presets.

Explain the bottleneck beside the architecture: the GRU emits 16 learned
summary values in the tutorial fit or 32 in the supplied fit. Two known timing
features condition the flow separately. This is a compression choice, not a
parameter count or a sufficient-statistic claim. Training loss uses random
prefixes; validation loss uses full five-year histories. Their gap therefore also
changes the conditioning-length distribution and cannot by itself diagnose
overfitting or information loss.

A readout of the actual displayed posterior compares central-90% widths with
the prior's central-90% widths, reports Pearson dependence from paired draws,
and shows synthetic-truth positions. It follows the saved/fresh source and
actual prefix. These are descriptions of concentration and geometry, not an
accuracy score or a new numerical reference.

Expand the rank guide with one worked calculation of the ECDF difference and
four toy shapes: calibrated, biased, too narrow and too wide. Explain three
different uncertainty statements:

- A central 90% posterior interval concerns one parameter for one history.
- A 95% binomial interval describes uncertainty in empirical coverage across
  independent histories; it does not include posterior Monte Carlo error.
- The native 95% ECDF band describes reference variation for each parameter's
  rank curve. It is not a simultaneous guarantee over all parameters.

Keep the native band calculation unchanged. With 200 posterior draws, strict
fractional ranks occupy 201 possible positions. Ties and Heston's drift atom
retain their separate limitations. Marginal rank agreement cannot establish
conditional or joint posterior fidelity. The supplied final calibration
failures remain authoritative.

No Q-option-bank timing or terminal-pricer speedup from PR #88/#90 is inherited
by this P-history lesson. No new simulation, training or inference performance
claim follows from these presentation changes.

**Latest inheritance verification · 12 November:** 84 tutorial tests, 527 core
lab tests (three optional PyMC skips), all seven exports (66 trajectory cells,
zero errors) and release/ownership checks pass. Actual posterior readouts
match all three models at 1, 252 and 1260 days; the helper also covers the
137-day request with its saved 126-day posterior. Browser checks cover the
cards, summary explanations, worked/toy rank views and actual 200-case ECDF
table. Explanations reuse existing results. No runtime or math errors, and
the cards and rank views fit at 760 pixels. The native band method, numerical
law, model weights and frozen calibration decisions remain unchanged.

**Initial SPX-alignment verification · 12 November:** 72 tutorial tests, 527 core lab tests (three
optional PyMC skips), all seven exports (63 trajectory cells, zero errors),
marimo and release/ownership checks pass. Prior/PPC smoke checks cover all three
models, two supplied cases and requested lengths 1, 137 and 1260, retaining
all histories and paired rows. Browser review covers actual training/validation
loss, prior and predictive checks, saved-126 versus fresh-137 conditioning,
recovery brushing, retained pair views, coverage, threshold controls and a
760-pixel layout. No runtime or math errors. The comparison source is
[Heston/SPX PR #89](https://github.com/fmuia/sbi4finance/pull/89) at `a8a9fdf`.

**Earlier diagnostic verification · 10 November:** 56 tutorial tests, 527 core lab tests (three
optional PyMC skips), all seven exports (48 trajectory cells, zero errors) and
release/ownership checks pass. Actual diagnostic sampling covers every native
model at 252 and 137 days. Chrome checks the tutorial and loaded rough-known
sources, session reuse across prefixes, missing networks, task/source changes
and 760-pixel layout. Canonical generation of 200 histories took 0.016–0.074
seconds on the local CPU, excluding imports. No new input assets are needed.
Existing weights, posterior evidence and final calibration decisions remain
unchanged.

Earlier component verification passed: **46 tutorial tests**, all seven notebook exports
(the trajectory lesson has 41 cells and zero errors), 12 lab runtime/parity
tests, the 183-symbol ownership check, marimo/release checks and documentation
links. The preceding full lab suite passed 527 tests with three optional skips;
this component/axis pass changes no canonical simulator or inference law.

Chrome checks open the component math for every model, exercise all three
manual scale presets after parameter changes, inspect outside counts and ranges,
and verify the input diagram. Fresh inference at 137 days stays source-bound
when only the view scale changes. The 760-pixel layout has no horizontal
overflow, and there are no page or math-rendering errors. The earlier workflow
review also covers history-change invalidation, missing-model fallback, the
small training action and all 16 simulator-slider range endpoints. Existing
empty-cell warnings in slots 2 and 3 are unchanged. Handover guidance is updated;
scientific limitations and the deferred collaborator rehearsal retain their status.

### Connect temporal behavior and financial scenarios

Borrow the useful teaching patterns from amortized-fractals without replacing
the trajectory model or its inference architecture. A fixed, hash-bound VOO
window (256 daily simple returns from 3 January 2020 to 7 January 2021) enters
through the lab acceptance workflow. The opening uses its recorded log-return
conversion to compare observed tails with a Gaussian reference. This empirical
motivation is separate from the synthetic histories used for inference.

The simulator gains an alternate matched three-setting view with fixed axes
and a shared realization-refresh action. Preserve the selected curve in the
joint model. Explain parameter effects across several paths; do not equate
Bergomi H with the signed-memory exponent in the separate MMAR extension.

Magnitude and Ordering tabs reuse the same prior or posterior replicas. The
ordering view compares signed- and absolute-return autocorrelation, and an
explicit shuffled-prefix intervention demonstrates which information the
magnitude summaries ignore. Below 21 days or with constant values, dependence
is unavailable rather than zero. The default 32 posterior replicas appear as
individual curves; a larger 256-history action supports pointwise descriptive
bands. These are not simultaneous model-adequacy or calibration guarantees.

Replace the abstract parameter-threshold exercise with point-parameter versus
posterior-mixture drawdown ECDFs and a reused threshold readout. Simulate the
unchanged full task before taking the actual posterior prefix. Pair process
inputs between groups, use independent seeds between chunks, and select whole
posterior rows with replacement. Include initial wealth in a stable log-space
drawdown calculation. A threshold or display change reuses outcomes; a changed
task, posterior, prefix or budget invalidates them. All numerical failures stay
explicit, without dropping difficult histories.

The new empirical asset and simulator/checking illustrations do not change
weights, training banks, numerical laws or final calibration decisions.
Documented failures remain applicable to derived scenario probabilities.

**Verification · 13 November:** 97 tutorial tests, 529 core lab tests (three
optional PyMC skips), all seven exports (76 trajectory cells, zero errors),
marimo and release/ownership checks pass. Browser review covers the empirical
opening, all three matched model views, ordering/shuffle controls, 32/256
posterior replicas, 128/512 drawdown scenarios and threshold reuse. Saved-126
fallback and fresh-137 predictive checks retain their actual conditioning
lengths. The new displays fit a 760-pixel layout without runtime or math errors.

### Make evidence revelation interactive and explain the inference target

Separate parameter inference, latent-state filtering and independent scenarios.
The present loss targets generating parameters, not terminal variance or rough
memory. BayesFlow can support a different target, but the current GRU summary
is not a calibrated state posterior. Explain continuous-time Heston's Markov
variance state and the extra signed auxiliary state in its numerical scheme;
rough Bergomi additionally needs relevant shock memory or a validated reduced
representation. Direct past-to-future amortization is another possible task.

Contrast the inspected amortized-fractals source at `069d5dc`: its estimators
also target parameters. Its rolling risk and allocation applications evaluate
decisions prospectively, using fresh cascades/phases for scenarios. They do
not filter the observed cascade state, and a finite random-phase window does
not itself specify inter-window evolution. Preserve this distinction without
dismissing restart-based risk prediction or the practical backtests.

An early section motivates parameter uncertainty, repeated matched-history
analysis and scenario propagation. A compact PyMC comparison explains the
conditional Gaussian likelihood for the discretized rough task, latent-path
integration and the roughly 10,000 driving coordinates in a direct five-year
augmentation. NUTS needs a deterministic differentiable target and validated
gradients. No matched timing or superiority result is claimed.

Replace the static evidence-evolution chart with a local play/pause/advance/
reset/scrub player. Start at the prior in the full selected-horizon view, with
Fast playback and trading-month axes; offer early-time zoom and day/week axes.
Link the revealed returns to actual posterior
distributions and central intervals, with fixed axes and explicit frame/source
labels. Saved frames remain bound to task, model and history; exact native
queries use the original prior and mask all future returns. Query preparation
is explicit and bounded to 61 frames per action; playing and changing units
perform no inference. The cursor is local and does not silently change the
other posterior and checking panels. Never interpolate posterior samples or
present playback as hidden-state propagation.

The known-curve rough-Bergomi replay bundles 53 extra exact-prefix queries for
each of the same four preserved validation histories: 65 actual frames after
combining them with the original 12. The first 21 days are daily, followed by
5-day, 21-day and 63-day spacing. Original paired samples remain byte-identical.
The 834 KB extra payload is hash-bound to task, checkpoint, replay and observed
history bytes, accepted through the lab tools and labelled as development
playback, not new calibration evidence. Other model replays retain their
original landmarks. Runtime and asset tests cover masked future returns,
context invalidation, sampling reuse, exact-frame fallback and atomic failure.

**Evidence-player verification · 13 November:** 116 tutorial tests, 529 core
lab tests (three optional PyMC skips), all seven exports (83 trajectory cells,
zero errors), marimo and release/ownership checks pass. Browser review verifies
the prior start, play/pause/reset/advance/scrub, exact saved-frame changes,
history/model changes, all 65 known-curve frames and the other models' 12
landmarks, daily native queries through day 21 and query reuse. The 760-pixel
view has no horizontal overflow. The iframe expands to fit its content, keeping
the footer visible. The requested user preview remains on port 8792; temporary
QA servers are stopped separately.

### Refine the model story, paired histories and observation-to-inference route · 14 November 2026

Make the statistical target explicit wherever it matters. The conditional flow
learns **one joint distribution over a fixed generating parameter vector**.
Separate density panels are marginals of the same paired samples; they do not
come from separately trained estimators. This is different from a joint
posterior over the time-varying hidden state and parameters. Revealing more
returns changes the evidence about that vector, rather than changing the
generating parameters or recursively using an earlier posterior as a new prior.

The known-curve rough example conditions on a simulation design with
ξ(t)=0.04. That deterministic ensemble mean is neither observed realized
variance nor a time-varying quantity inferred by the GRU. The existing
rough-joint extension relaxes the supplied curve by jointly inferring the
static coefficients S, L and k of ξ(t)=L+(S−L)exp(−kt), alongside H, η, ρ and
drift. It still does not estimate a latent variance history. Retain the
distinction from pricing-measure forward-variance curves and from state
filtering/smoothing tasks.

The empirical opening now defaults to **1,726 VOO daily returns from
3 January 2020 through 13 November 2026**, with its adjusted-close source,
corporate-action convention and exact `log1p` return conversion recorded in
the accepted snapshot. The original 256-day crisis window remains selectable.
Use the same trace and absolute-return tail-frequency layout for observed VOO
and an explicitly specified rough-Bergomi model. Each Gaussian reference uses
the corresponding history's mean and standard deviation. One illustrative
Bergomi history supplies the trace and tail curve; the multi-history band is
removed to focus on the departure from a matched Gaussian. The example is not
evidence of a calibrated fit to VOO.
The full dated market display exceeds the neural estimator's 1,260-day input
support. It remains a motivation panel, with no out-of-support market query.

Replace the static observed-history view with a local reveal player. Show the
unchanged daily returns and their **actual paired hidden variance at observed
day boundaries**, plus marginal-frequency panels on the right. The latent
asset reconstructs the same original task, generating parameters and
innovation stream for all four preserved cases of each model. It checks
observed-history and truth identities before displaying a latent path; it does
not substitute a newly simulated path. Heston replay is exact, while rough
replay agrees within the recorded numerical-library tolerance. Heston's signed
numerical state is retained separately from its effective nonnegative variance.
The orange path is available because these data are synthetic; it is not a
neural estimate and never enters inference conditions. The right-hand panels
summarize revealed values without replacing the ordered inference input.
The player also reports raw minima, exact-zero counts and early/late medians,
and offers a log-variance view. True zeros remain undefined in log coordinates;
they are counted, not replaced by a small positive number. Rough mean/median
reference curves use the same generating parameters and hybrid-grid variance.

The later audit corrects an incomplete interpretation of the quiet examples:
the ageing law matters, but the four preserved known-curve paths are also
unrepresentative of its prior frequency. With a frozen criterion of average
daily-boundary V in days 1009–1260 below 0.0004, all four saved cases qualify,
versus 69/512 independent prior paths (13.5%; Wilson 95% interval 10.8–16.7%).
At the same parameters as case 3, only 2/512 new realizations qualify. The
compact accepted diagnostic retains every prior value, every original saved
marker and all four conditional ensembles. It exposes model ageing and chance
variation without choosing prettier replacement histories. The numerical
audit found healthy Gaussian inputs and the intended grid compensator, with
no rough-variance zeros, underflow, clipping, floors or redrawn failures.

Explain η beside the controls as log-variance dispersion under a fixed-mean
compensator. Increasing η raises the spread across realizations while lowering
the marginal median; it need not raise an individual realized variance path.
At a fixed driver realization, the derivative of log variance is
Y−η Var_grid(Y). Keep the original numerical law and parameter support, and
distinguish the true hidden variance from rolling volatility estimated from
returns. The existing zero-origin/nonstationarity limitation stays visible.

Preserve the original workflow image, input diagram and actual training code.
Use a guided walkthrough of the first real tutorial training pair to answer
one question: what changes when the observed prefix gets shorter? Show signed
returns with a zero line, excluded future positions, mask counts and actual
canonical tokens and timing conditions. The observed prefix changes the
conditions, while the generating vector and its canonical transformed target
stay fixed. Separate Training and Inference views: the former scores the known
transformed label; the latter illustrates conditional sampling and inverse
target transforms without that label. Balanced label/objective blocks make
the training-only route explicit. Named stages end at a takeaway rather than
looping indefinitely. Learned summaries, density values and gradient updates
are schematic; the separate training action executes the optimizer.
The encoder comparison identifies the causal
GRU as a recurrent observation encoder, rather than an autoregressive
posterior decoder. It also names the separate density-family, unit, transform
and masking changes from the option lessons instead of implying that adding
the GRU is the only code difference.

The preceding guided-animation review passed 144 tutorial tests and all seven notebook
exports (88 trajectory cells, zero errors), plus marimo and release/ownership
checks. Focused tests bind the physical and transformed joint label to the
canonical training API at every displayed prefix. Browser checks cover
prefixes 1/21/63/252, Training and Inference routes, pause/reset/replay,
stop-at-end behavior and wide/760-pixel layouts without overflow or hidden
footers. The animation evaluates preprocessing only; it creates no learned
summary, loss, gradient update or posterior sample.

Default the synthetic calculations to **1,260 days**. The history player uses
21-trading-day reveal steps and trading-month axes. Posterior playback uses
the full selected horizon, Fast speed and trading-month axes, with generating
truth markers in every parameter panel; early zoom remains available. Local
reveal controls do not silently change other posterior queries. Model
extensions, prefix changes, native-frame preparation, detailed input/mask
tables and the optional 200-history network check are collapsed secondary
tools. Keep the final failed-calibration evidence visible in the main route.
The forward simulator, live-training/validation/example histories, input
diagram, architecture, training walkthrough and diagnostic defaults now share
this five-year maximum. The small fit remains 128 training histories, 64
validation histories, two epochs and eight optimizer updates. A measured local
execution completed in 13.4 seconds including imports. A displayed 252-day
prefix is explicitly one trading year; 252 in time steps and annualization
remains the trading-days-per-year convention. Group the governing rough
equations and Heston comparison before showing the observed-history section.

Redesign the prior/posterior-predictive displays around a direct data/model
comparison, retaining the existing simulation results and parameter targets.
The prior action explicitly draws one parameter vector and one history for
each of 200 simulations. Its default observation is the same-length prefix
of the accepted VOO snapshot; the supplied synthetic history remains an
alternative. Market returns are used only for these descriptive comparisons,
never passed through the neural estimator. The posterior check continues to
use the displayed synthetic posterior and its actual conditioning length,
including a saved fallback.

The default comparison aligns return paths, cumulative log wealth, pooled
return histograms and absolute-tail frequencies. A data-fitted IID Gaussian
reference appears in the same views, so matching a mean and scale is
distinguished from reproducing extreme moves or temporal dependence. At small
replication counts, show individual paths; larger panels use labelled
pointwise simulation bands. The optional Summary view retains the original
magnitude statistics. Timing compares signed and absolute-return ACF with a
finite-sample Gaussian band. A feature table reports large-move frequency and
lag-one amplitude dependence, with explicit unavailable short/constant cases.
Full denominators and outside-view counts prevent hiding difficult simulations.
Neither a visually plausible replicate nor a difference from this Gaussian
reference is a calibration result or a guarantee that the volatility model
fits. Actions and simulation-budget controls remain collapsed secondary tools.

**Verification · 14 November, after horizon/variance review:** 151 tutorial tests, 531 core lab tests (three
optional PyMC skips), all seven exports (88 trajectory cells, zero errors),
marimo and release/ownership checks pass. Browser checks cover both market
windows, all three paired synthetic-history players, replay and fixed axes,
the prefix/mask training animation, encoder comparison, full-horizon posterior
playback and case-bound truth markers, collapsed controls, exact native queries,
the real/synthetic prior comparison and posterior replicas. Wide and 760-pixel
views have no horizontal overflow, hidden widget footers, math or runtime
errors. The final review also checks model → observed history → simulator order,
full-horizon training and diagnostic defaults, every rough-known and Heston
history in linear/log views, mean/median references, exact-zero gaps and raw
value preservation. The accepted gallery audit is bound to the same task and
original cases. Standalone predictive checks also cover all three models, one-day
and constant histories, and the optional larger replication budget. The original
model weights, parameter priors, numerical law and calibration failures are
preserved. Temporary QA port 8794 is stopped; the user review preview is separate.

**Windows asset loading · 14 November:** reproduced the collaborator's
`trajectory catalogue/bundle hash differs` failure through Git checkout with
`core.autocrlf=true`. Git converted JSON metadata to CRLF before the raw hash
check; numerical arrays and the recorded task were unchanged. Trajectory replay,
latent-history, dense-evidence, variance-audit, market and closeout checks now
reuse the option loader's JSON-only CRLF/LF-equivalent digest. NPZ, PNG and model
payloads keep exact byte hashes, and changed JSON content remains rejected.
Accepted-asset Git attributes prevent future checkout conversion; legacy CRLF
checkouts also load without regenerating data or rewriting their pins.

Verification adds three participant regressions for unchanged data/identities
and genuine corruption, plus an isolated lab Git-checkout test. All **154
tutorial tests**, the **531-test lab suite plus the new checkout regression**,
seven exports, marimo and release/ownership checks pass. A separate complete
trajectory export against CRLF metadata also runs all 88 cells without errors.
The participant tests remain self-contained; Git policy is checked in the lab.

## E. Resolve trajectory numerical and inference limitations

The zero-origin rough physical-return law is nonstationary even with a flat
mean-variance curve: its Gaussian driver variance grows with age, so the median
variance can fall while rare large paths preserve the mean. This materially
affects the supplied known-curve histories. The notebook now explains that
behavior rather than presenting it as generic stationary market dynamics.
An eventual stationary or mean-reverting rough physical law would require a
new simulator specification, banks and estimators. Picking different display
histories or changing the supplied curve would not resolve that modeling issue.
The existing H prior also conditions inference on the rough regime; it cannot
establish preference over an excluded Brownian boundary. The driver remains
Gaussian for every H.

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
| Trajectory case/history and evidence views | `trajectories/lesson.py`: `load_example`, `saved_lengths`, `posterior_view`; `latent_assets.py`/`history_widget.py` for paired synthetic truth; `evidence.py`/`evidence_assets.py`/`evidence_widget.py` for actual posterior frames | Original caches under `assets/trajectories/{heston,rough-known,rough-joint}/`, paired paths under `latent-history-v1/` and dense known-curve frames under `evidence-rough-known-v1/`; final diagnostics under `assets/results/trajectories/closeout-20261031/`. Posterior marginals retain the same paired rows; displayed latent truth never becomes an inference condition. |
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
a small tutorial training route from the optional pretrained and million-history streaming routes. The revealed
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
