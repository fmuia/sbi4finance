# Application notebooks

We start the workshop with **Beamer slides** on the introduction, amortized
Bayes and network architectures. These notebooks cover the applications:

| Application | Notebook | Main code |
| --- | --- | --- |
| Heston options | [slot4_heston_bayesflow.py](notebooks/slot4_heston_bayesflow.py) | [heston_options/](src/sbi4f_tutorial/heston_options/) |
| Rough Bergomi options | [slot5_calibration.py](notebooks/slot5_calibration.py) | [rough_options/](src/sbi4f_tutorial/rough_options/) |
| Parameter estimation from returns | [trajectory_inference.py](notebooks/trajectory_inference.py) | [trajectories/](src/sbi4f_tutorial/trajectories/) |

An optional [MMAR take-home](notebooks/mmar_takehome.py) carries the same workflow
into a different return model and a sensitivity experiment; see its
[curriculum](curriculum/07-takehome-mmar.md).

Each follows the same workflow: **model → simulate → train → infer → check**.
The slot 1–3 notebooks remain available as supporting demos; they are not the
planned format for the opening lessons.

## Collaborator handover

For now, please focus on the three application notebooks above. We want to
improve the content and presentation; we can trim for timing later.

1. Follow the [setup](#install), then [open a notebook](#run-or-review-an-existing-lesson).
   The saved examples work without Oscar or extra downloads.
2. Review the explanations, figures and visible code. Is it clear what we
   observe, what we estimate and how the Bayesian workflow fits together?
3. Edit the notebook for text, layout and controls. Edit its application
   package for the calculations; common code lives in
   [shared/](src/sbi4f_tutorial/shared/). Lab scripts use this same code.
4. Leave a review comment or open a PR. Please name the notebook and section,
   explain what you would change, and why.

For bugs, include the model, example, settings and error message. For return
histories, include saved/live mode and the history length shown beside the posterior. See the [editing guide](#author-or-change-a-lesson) for checks.

## Install

With [Pixi](https://pixi.sh), from the repo root:

```bash
cd tutorial
pixi install --locked
pixi run --locked test
```

`--locked` uses the versions in `pixi.lock` and stops if the dependency settings
and lock file disagree. It does not prevent code edits.

The notebooks use Python 3.12, BayesFlow 2.1 and marimo. A CPU is sufficient.
The lock covers Linux, Apple Silicon macOS and Windows; Windows has not been
runtime-tested here, and Intel macOS is not included.

With pip on Python 3.12:

```bash
cd tutorial
pip install -e ".[participant]"
export KERAS_BACKEND=jax
```

Pixi includes the `participant` extra automatically. The lab installs the base
package and selects its own BayesFlow version.

## Run or review an existing lesson

From `tutorial/`:

```bash
pixi run --locked heston-options
pixi run --locked rough-options
pixi run --locked path-inference
```

Run one command at a time. Each opens the marimo editor in your browser;
press Ctrl+C in the terminal before opening the next notebook.

The saved observations and posterior draws are included. Training and fresh
model sampling only start when requested. Heston also computes small path and
pricing illustrations on opening. The trajectory notebook also runs a small
NumPy forward illustration when its parameter controls change.

For a read-only app:

```bash
pixi run --locked marimo run notebooks/slot4_heston_bayesflow.py
```

Exported HTML is a static preview; use marimo for interactive controls.
`slot1`, `slot2` and `slot3` open the supporting demos. `backbones` opens the
separate model-review notebook. The older `slot4`, `slot5` and `trajectories`
commands still work.

## Choose files for the task

| What you want to do | What you need |
| --- | --- |
| Review or edit a notebook | This checkout; no extra download |
| Draw fresh option posteriors | [Option model bundle and setup](../lab/plans/training-data-guide.md#option-models-heston-and-knownjoint-curve-rough-bergomi) |
| Draw fresh trajectory posteriors | [Trajectory model bundle and setup](../lab/plans/training-data-guide.md#trajectory-models-daily-return-histories) |
| Retrain an estimator | The matching [option bank](../lab/plans/training-data-guide.md#full-option-training-banks) or [trajectory bank](../lab/plans/training-data-guide.md#trajectory-banks-daily-histories-with-inferred-drift) |

The linked guide gives download locations, checksums and extraction commands.
Use your own GitHub or Drive account with access to the files; Drive Desktop
and Oscar are not required. Put downloaded models in `assets/local-checkpoints/`
as instructed. The loaders check that the model matches the selected example.

**Windows checkouts:** accepted assets are protected from Git line-ending
conversion. Participant JSON checks also accept LF/CRLF differences in older
checkouts; array and model files still require exact hashes, and changed JSON
content remains an error. If an older checkout reports `trajectory
catalogue/bundle hash differs`, pull the updated branch and restart marimo to
load the corrected helper code. No data regeneration or model download is
needed for this line-ending issue. Other checksum failures can indicate a
mixed or damaged asset set; do not replace the recorded hashes to bypass them.

## Trajectory inference

The page follows the Heston/SPX lesson's presentation: one linked workflow map,
six numbered section banners, numbered subsection rules, and the same type,
spacing and foldable-detail styling. Its local stylesheet is
`notebooks/trajectory_inference.css`; the presentation helpers live in the
notebook's opening hidden cell. Numerical code and the iframe players retain
their existing behavior.

The three models use daily log returns and infer **one joint distribution over
fixed generating parameters**, including drift. The separate parameter plots
are marginals of those same paired draws. They are not independently fitted
parameter estimates or a posterior over the hidden volatility history.
The supplied estimators accept integer history lengths from **1 to 1,260
trading days** (five years).

The supplied-curve rough task fixes the deterministic mean curve ξ(t)=0.04 as
part of its simulation design; realized variance remains random. Relax that
assumption by selecting the rough joint-curve example: it infers the three
static coefficients S, L and k in ξ(t)=L+(S−L)exp(−kt), together with H, η, ρ
and drift. Learning these coefficients is still parameter inference, rather
than filtering time-varying variance. These are physical-return model curves,
not pricing-measure curves estimated from option quotes.

A stage bar follows the workflow, and a short API map shows where BayesFlow
enters. A design table explains the masked GRU, spline flow and preprocessing.
The training code shown is the function called by the button.
The opening defaults to a dated VOO snapshot with **1,726 daily returns from
3 January 2020 through 13 November 2026**; the earlier 256-day crisis window
remains selectable. Identical trace and tail-frequency views compare VOO with
an explicitly specified rough-Bergomi simulation and a Gaussian matched to
each history's mean and standard deviation. One Bergomi history supplies both
its trace and empirical tail curve, making the excess of large moves over the
matched Gaussian easy to inspect. The model parameters were fixed for
illustration, not calibrated to VOO. The longer market display does not extend
the neural estimator's 1,260-day support or feed market returns into it.
Both windows load offline accepted data, with no vendor request on opening.
Four model cards connect physical drift versus log growth, volatility bursts,
leverage and variance level/memory to the parameters. Predict a change, move
one simulator control, then compare what happened with your prediction.

Start with the rough model and its supplied variance curve. The observed
synthetic history defaults to all **1,260 days**. Its local reveal player shows
the original returns and their **actual paired hidden variance at day
boundaries**, with marginal-frequency panels alongside. The hidden path comes
from a verified replay of the same parameters and innovations; it is simulation
truth, never a network input or an estimated state. Play, advance, reset and
scrub reuse that one history; the defaults reveal 21 days per step on trading-month
axes. Fixed axes and outside counts keep comparisons honest.

The model introduction groups the price, Volterra-driver, variance and
correlation equations, with Heston's mean-reverting variance equation beside
them. The observed synthetic history follows that introduction.
The separate forward simulator changes parameters while keeping innovations
fixed. It shows eight five-year price paths, daily returns and return-based
realized volatility. This noisy rolling statistic differs from the hidden
variance revealed for the fitted synthetic history.
The observed-history explanation flags an important design limitation:
zero-origin rough variance is nonstationary even with a constant mean curve.
Its median can decline while rare large paths keep its mean fixed. The four
saved known-curve histories are also an unusually quiet gallery. A frozen
512-history prior audit separates that finite-gallery effect from the model's
age-dependent distribution. The original examples remain unchanged. The
history player adds paired mean/median reference curves, a log-variance view
and exact-zero/minimum/first-versus-last-window readouts. Logarithms leave true
zeros undefined instead of replacing them with a positive floor.
The H-prior explanation distinguishes roughness from Gaussianity:
the driver is Gaussian for every H, while H=1/2 is its Brownian boundary. The
chosen Uniform[0.03,0.4] training prior excludes that boundary and cannot test it.
Increasing η increases log-variance dispersion, while its compensator preserves
the ensemble mean and lowers the marginal median. A fixed realized variance
path can therefore fall when η rises. The controls explain this effect instead
of promising a monotone increase in every path's level.
An alternate three-setting view compares H or η for rough Bergomi and σ or κ
for Heston with matched random inputs. **New realizations** refreshes every
column together; switching display scales does not change the draws.
Its y axes stay fixed as parameters change. Choose **Standard**, **Wide** or
**Very wide** yourself; each panel reports points outside the view. The full
series and its minimum/maximum stay available in the axis-details table.
Heston and the inferred-curve model remain available as extensions.

There are two explicit routes:

- **Train here:** generate 128 new five-year training histories and a separate
  validation set. Fit a 16-unit GRU and two-block spline coupling flow for eight
  updates, inspect training/validation loss, then infer parameters for an
  independent history. Results stay in the session and have no calibration claim.
- **Use the optional pretrained model:** inspect saved draws or load the matching
  five-year model. Its larger encoder and flow have their own data and checks.

Open the component panels for the physical P law versus pricing Q law,
Gaussian innovations, daily numerical steps and supplied versus inferred
variance curves. One history is one observed data set; simulation paths are
not averaged into that observation. The input diagram traces returns into
scaled time/mask tokens, the GRU and flow conditions, with generating labels
kept on a separate training route. The architecture diagram and visible code
show each route's actual settings. Switching the supplied model
also changes its synthetic history.
The original workflow graphic and input diagram remain. The guided walkthrough
asks what changes when less of the same training history is observed: the
revealed length, mask and duration change, while past return values and the
generating parameter vector stay fixed. Training and inference views separate
the known label and density-loss route from conditional sampling without labels. Concrete token,
mask, time-context and transformed-label values use the canonical preprocessing;
network summaries, density evaluation and optimization remain schematic.
Playback ends at an explicit takeaway rather than looping through highlights.
The actual training action below performs fitting. A comparison diagram distinguishes the
recurrent observation encoder from the option lessons' direct condition
vectors, while also naming the density-family and preprocessing changes.
The GRU compresses the returns into 16 tutorial or 32 pretrained summary values;
two known timing features condition the flow separately. Summary size is a
design choice, not a guarantee that all information survives. Training uses
random prefixes while validation loss uses full five-year histories, so their loss
gap also reflects different conditioning lengths.

**Saved mode** offers only lengths with saved posterior snapshots, so the
returns and posterior use the same history. **Live mode** accepts any integer
length. With the matching model downloaded, the CPU action updates the same
posterior plots using exactly that prefix. Without it, the labelled saved
fallback remains available. Changing the model, history or length clears a
previous live result.
The **Watch the evidence grow** player has separate local play/pause, advance,
reset and scrub controls. It defaults to the full selected horizon, **Fast**
playback and trading-month axes; optional early-time zoom and day/week units
make the first changes legible. Generating-value markers stay visible in each
parameter marginal for this synthetic example and never enter inference.
It advances through actual posterior frames rather than interpolating samples.
The four supplied-curve rough-Bergomi examples include 65 actual saved frames
each, daily through day 21 and then more widely spaced. Their original 12
paired-sample snapshots are retained unchanged. Other models keep their
original saved landmarks; the player states how many frames are available.
Preparing additional native frames is an explicit bounded action under the
original prior, with unrevealed returns masked. Playback itself runs no inference
and does not change the other notebook views. Revealing evidence is parameter
inference, not hidden-volatility propagation.
Beside the posterior, a short readout compares its central-90% widths with the
prior's, reports paired-draw Pearson correlations and locates the synthetic
truth. These describe concentration and dependence; they do not measure accuracy.

The two model sources are labelled beside their outputs. The tutorial fit does
not reproduce or validate the supplied five-year fits. Model extensions,
posterior-prefix changes, additional native frames, production-bank instructions
and detailed input/diagnostic tables are collapsed secondary tools.

**Compare data with replicated histories.** The prior-predictive action draws
one parameter vector and one independent history for each of 200 simulations.
Its default comparison uses the same-length prefix of the accepted VOO snapshot;
the supplied synthetic observation is an alternative. This is a descriptive
model/prior check, with no neural inference on market data.

The posterior-predictive action selects 32 complete parameter rows from the
displayed supplied-model posterior, saved or fresh, and simulates independent
histories from the model's origin. It compares the actual posterior conditioning
length, including any labelled saved fallback, with that synthetic observation.
The default display aligns return paths, cumulative log wealth, pooled return
frequencies and absolute-return tails. A Gaussian fitted to the comparison
history provides an IID reference in the same views. Matching scale alone does
not establish matching tails or timing; neither model is guaranteed to fit.
Summary and Timing views add the original magnitude checks and signed/absolute
return autocorrelation, with a finite-sample Gaussian comparison. A feature
table highlights large moves and amplitude dependence. Short or constant
histories have explicitly unavailable estimates. All simulation rows remain in
the data; fixed views report outside counts. Individual curves and pointwise
simulation bands have different meanings and do not establish calibration.
The 32/256 replication budgets and actions are optional controls.
These are independent replications, not continuations of the observed path.
Parameter inference alone does not provide the hidden-state or rough-memory
conditioning needed for a continuation forecast.
An early practical-use section distinguishes describing dynamics, repeatedly
analyzing matched histories and propagating parameter uncertainty into scenarios.
The PyMC comparison explains latent-path integration and gradient requirements;
it makes no timing or superiority claim. The final explanation shows why this
parameter-only training target does not produce a state posterior. It contrasts
the fresh-cascade risk and allocation applications in amortized-fractals with
state-conditioned continuation, while preserving their prospective evaluation
and practical usefulness.
The final exercise compares replicated-history drawdown at the posterior-mean
parameter vector with drawdown from complete posterior draws. Both retain
process randomness; threshold changes reuse the simulated outcomes. These
scenarios start at the model's origin and do not condition on today's latent
variance or rough-shock memory. The supplied posterior's calibration failures
also apply to calculations derived from it.

**Check the network you used.** Expand the optional independent-history tools,
choose **Network to check**, set **Days per
diagnostic history**, then press **Check this network on 200 independent histories**.
This length is separate from the single-history controls above. First use
generates 200 validation histories for that task and keeps them in session
memory. Later checks reuse them and draw 200 fresh samples per history for
recovery, rank-ECDF and coverage plots. Select a diagnostic history to inspect
its paired posterior draws and generating values.
Before a check completes, the recovery-selection panels stay empty. Closing the
tools or changing the source, task, length or workflow clears the displayed
result and requires a fresh click; failed checks expose no partial result.
The rank guide includes a worked ECDF difference and four illustrative shapes.
It distinguishes posterior intervals, uncertainty in empirical coverage, and
the native 95% rank-ECDF band for each parameter. With 200 draws, ranks have
201 possible positions; Heston's drift atom needs separate interpretation.
Generation took under 0.1 seconds per 200 histories in our local CPU check, so
there is no separate diagnostic dataset to download.
Every history uses the same chosen prefix; later returns and time tokens are
masked. Generating parameters are labels for the checks, never model inputs.
The tutorial and supplied rough-known fits now share the five-year task
design, while retaining separate weights and training sets. Other supplied
models keep their own matching task definitions.

These independent, fixed-seed histories are reusable validation data, not an untouched final
test. Neither the inputs nor the new posterior draws are written to the repo.
The figures describe the selected workflow at that history length; they add a
check of the model you just ran alongside the saved final-cohort results below.
They do not replace those results or validate the tiny tutorial fit.
Edit `trajectories/diagnostics.py` for this check; the shared network and masking
code remains in `trajectories/inference.py`. Prior/posterior-predictive numerical
checks live in `trajectories/model_checks.py`; their data/model comparison views
live in `trajectories/predictive_visuals.py`.

**Checks are not yet satisfactory:** all three estimators failed the final
calibration check. The Heston simulator can also produce zero-variance days
that reveal drift exactly, which a continuous neural density cannot fully
represent. These are development examples. Keep those limitations alongside
the results. See the [report](assets/results/trajectories/closeout-20261031/closeout-summary.json),
[drift figure](assets/results/trajectories/closeout-20261031/drift-learning.png)
and [coverage figure](assets/results/trajectories/closeout-20261031/marginal-coverage.png).

## MMAR take-home

From `tutorial/`, run:

```bash
pixi run --locked mmar-takehome
```

**From turbulent returns to uncertain drawdown scenarios** is a complete
one-hour take-home inspired by
[amortized-fractals](https://github.com/stefanradev93/amortized-fractals).
It explains a volatility cascade and Student-t innovations, shows the BayesFlow
construction and checks, infers four parameters for one fixed VOO window, then
propagates paired draws into replicated wealth and drawdown. The lesson ends
with links to upstream notebooks for the expanded story.

The saved-results route is complete and works offline. It uses eight
32-day volatility blocks. The unknowns remain
daily location, baseline scale, cascade contrast and innovation-tail thickness;
the number of blocks is a known input to the estimator. These are fixed
256-day **simple-return** observations, not the trajectory lesson's variable
prefixes of log returns. The FusionTransformer and spline flow have their own
fitted preprocessing and do not reuse the trajectory model's weights.

**Fresh inference is optional.** Download the
[pinned source sensitivity checkpoint](https://github.com/stefanradev93/amortized-fractals/raw/069d5dc5943dbe85505849f2fa8bfb563617ff97/checkpoints/sensitivity/model.keras)
to `assets/local-checkpoints/mmar-sensitivity/model.keras`, then select the
supplied-model route and load it explicitly. The loader restricts the path to
the asset cache and checks its SHA-256 against the saved lesson report before
restoring the serialized preprocessing. The original model bytes stay outside
Git. Missing or mismatched files leave the saved route usable.

**Live training is optional.** Its button trains a separate, smaller session
model across the supported resolutions, with independent validation and its
own recovery/calibration checks. The small fit does not inherit the supplied
estimator's evidence.

The supplied model was checked on 200 independent histories with 512 posterior
draws at each of three evaluated resolutions (4, 8 and 128 blocks). None of the
twelve rank tests was rejected by the predeclared Holm-adjusted screen;
empirical central-90% coverage ranged
from 88.5% to 93.0%. These finite checks support the scoped example, not exact
posterior accuracy or arbitrary real-market use. The preserved historical
training bank/seed was not independently reconstructed. See the
[bound report](assets/takehomes/mmar-v2/report.json) for the checkpoint,
simulation-parity, data and evaluation identities.

Scenario paths start with fresh cascade phases. They are model-generated
replications rather than continuations conditioned on today's hidden volatility.
Threshold controls reuse outcomes. No live vendor data, training or checkpoint
sampling starts on opening. After the drawdown section, the notebook links to
the upstream univariate and resolution-sensitivity notebooks and the expanded
rolling-risk, joint-market and signed-memory story.

## Give feedback on an application

Useful starting points:

- **Heston:** are the model, quote noise, posterior comparison and predictive
  checks clear?
- **Rough Bergomi:** is the difference between a supplied and an estimated
  variance curve clear? Are the remaining checks easy to spot?
- **Trajectories:** can you follow how longer histories change the posterior?
  Is the distinction between saved and freshly computed results clear?

Please report the notebook, section and proposed improvement. For an execution
problem, also include your platform and the exact error. A collaborator
rehearsal is still planned separately.

## Author or change a lesson

| Change | Edit here |
| --- | --- |
| Text, equations, layout and controls | The notebook |
| Heston prices, path illustrations and predictive checks | [heston_options/](src/sbi4f_tutorial/heston_options/) |
| Rough Bergomi simulation and lesson figures | [rough_options/](src/sbi4f_tutorial/rough_options/) |
| Trajectory simulations, masks, inference and figures | [trajectories/](src/sbi4f_tutorial/trajectories/) |
| Option inputs, networks and loading | [shared/full_surface.py](src/sbi4f_tutorial/shared/full_surface.py) |
| Common grids, noise assumptions and plotting | [shared/](src/sbi4f_tutorial/shared/) |

Edit the implementation directly. Older flat modules are compatibility imports;
there is no code-copy step. Changing a displayed code snippet alone does not
change the calculation behind it. The [source map](curriculum/TUTORIAL-REVIEW.md#follow-a-computation-from-the-lesson)
gives more detail.

Before opening a PR, run from `tutorial/`:

```bash
pixi run --locked test
pixi run --locked check
pixi run --locked test-notebooks
```

For shared numerical changes, also run the affected lab tests and
`pixi run --locked -e core release-check` from `lab/`.

Use the [lab tools](../lab/README.md#from-lab-code-to-a-lesson) to update saved
results and figures. Do not edit generated assets or their manifest by hand.
Keep `tutorial/` and `lab/` together when deploying lab work elsewhere. Existing
banks can be read and used for training; extending an old run requires its
matching code and environment.

## What is in this folder

| Folder | Contents |
| --- | --- |
| `notebooks/` | Lessons and supporting demos |
| `src/sbi4f_tutorial/` | Application packages and shared code |
| `assets/` | Saved examples, results and figures |
| `tests/` | Code, data and notebook checks |
| `curriculum/` | Teaching plans and review notes |

## Where the numbers come from

The [application plan](curriculum/TUTORIAL-REVIEW.md) and
[trajectory plan](../lab/plans/PATH-INFERENCE-PLAN.md) record the current checks
and limitations. Rough Bergomi still needs further reference-posterior and
product checks. Older SPY comparisons and calibration claims have their own
scope in [CLAIMS.md](CLAIMS.md). Passing software tests does not establish
posterior accuracy.

## If something does not work

- **Keras backend error:** set `KERAS_BACKEND=jax` before starting marimo.
  Pixi does this for you.
- **Slow first model call:** the first call includes backend setup and compilation.
- **A cell says “pending”:** that result is not included yet. Check its section
  notes; this does not mean the current examples or training banks are missing.
