# Application notebooks

We start the workshop with **Beamer slides** on the introduction, amortized
Bayes and network architectures. These notebooks cover the applications:

| Application | Notebook | Main code |
| --- | --- | --- |
| Heston options | [slot4_heston_bayesflow.py](notebooks/slot4_heston_bayesflow.py) | [heston_options/](src/sbi4f_tutorial/heston_options/) |
| Rough Bergomi options | [slot5_calibration.py](notebooks/slot5_calibration.py) | [rough_options/](src/sbi4f_tutorial/rough_options/) |
| Parameter estimation from returns | [trajectory_inference.py](notebooks/trajectory_inference.py) | [trajectories/](src/sbi4f_tutorial/trajectories/) |

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
histories, include the requested history length and the length used by the
saved posterior. See the [editing guide](#author-or-change-a-lesson) for checks.

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
pricing illustrations on opening.

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

## Rough options

Start with the supplied-curve example: infer H, eta and rho from option quotes
and seven known forward-variance levels. The joint-curve extension estimates
three curve coefficients as well. These examples use different observations
and curve families; they are not a matched comparison of curve uncertainty.

Start by changing H, eta, rho and the variance scale in the component simulator.
Its driver, variance, return and price panels use fixed axes. Random realizations
and values outside the view are explained. Collapsible notes cover each
equation, the Q measure and the forward-variance curve. A second simulator
shows how path payoffs become estimated prices and independently noisy quotes.
Compare Heston and rough Bergomi through short-maturity skew: an instant shape
guide sits beside the model, and an optional pricing action compares actual
smiles on the same contract grid. A dated SPY example motivates the pattern;
the illustrative parameters are not market calibrations.
The architecture diagram then separates the forward pricer from the conditional
posterior network. Both have visible, commented training code. A short table and
inline comments mark what stays the same as Heston and what changes: curve
conditioning and offline reuse of Monte Carlo prices.

Choose how to run the lesson:

- **Saved results:** no download or training needed.
- **Tutorial version:** train a small pricer or posterior network on fresh markets.
- **Pretrained version:** optionally load the matching posterior weights and draw
  for the worked example. See the [model guide](../lab/plans/training-data-guide.md).
- **Network check:** choose the trained tutorial network or loaded pretrained
  network, then run recovery and rank-ECDF plots on **200 markets × 200 draws**.
  The 200 input markets are bundled; posterior draws are generated when requested.

The small fits demonstrate the APIs; their outputs stay separate from the saved
accuracy checks. Monte Carlo pricing error is reported separately from quote noise.
The network check reuses a fixed validation panel, not an untouched final test.
It needs no new option pricing. Generating the panel took about **11½ minutes**
on one local CPU worker; checking the pretrained network took about **34 seconds**
for sampling, plus plotting and model loading. The input cache is about **305 KB**.
An [optional regeneration recipe](../lab/plans/training-data-guide.md#fixed-rough-option-diagnostic-inputs)
keeps this expensive step outside the notebook.

Edit the path explorer and architecture figure in `rough_options/explorer.py`,
the Heston comparison in `rough_options/comparison.py`,
the supervised pricer in `rough_options/surrogate.py`, and the notebook for its
controls and visible workflows. Lab pricer training reuses the same implementation.
The fixed validation panel and native BayesFlow checks use `rough_options/diagnostics.py`.

The **Additional studies** selector offers three short learning extensions:

- Check whether a new market's supplied curve fits the simulation design.
- Propagate paired parameter draws into a variance-swap strike; compare with
  deliberately shuffled marginals.
- Change an explicit assumption beyond the latest quoted maturity.

Each has a prediction prompt, a control or calculation, and an interpretation.
They use current task/checkpoint evidence and freshly evaluated analytic checks.
SPY is a fixed real-market example for the input-support check; its supplied
curve is outside the current known-curve training prior, so no supported SPY
posterior is claimed. Old pilot outputs and timing projections are absent from
the student narrative; their source records remain in the repository.

An input figure explains the 70 supplied-curve features and the 63-feature joint
task. The pricer check shows predicted prices against numerical labels at one
expiry, then signed errors in quote-SD units.

The joint extension includes its six-input pricer counterpart. An optional
matching model load adds a least-squares point calibration beside the saved
posterior propagation: compare one variance-swap strike with a distribution
of fair strikes on the **same underlying**. This is not a transfer to another
underlying or a forecast of realized variance. Exact-model and joint MCMC
references remain unfinished.

Edit the data story in `rough_options/data_story.py`, input figures in
`rough_options/lesson.py`, and uncertainty/point-fit calculations in
`rough_options/studies.py`. The point fit needs the published joint-pricer
folder; posterior propagation works without it.

## Trajectory inference

The three models use daily returns and estimate drift. The joint rough model
also estimates its variance curve. The estimator accepts any integer history
length from **1 to 1,260 trading days** (five years).

The notebook opens with saved posterior snapshots. Between snapshots, it shows
the nearest **past** posterior and labels the length it uses. For example,
requesting 137 days shows 137 returns but a saved posterior based on 126 days.
It does not interpolate posteriors or use future returns.

With the matching model downloaded, the CPU inference button uses exactly the
selected history length. The result remains an approximate posterior.

**Checks are not yet satisfactory:** all three estimators failed the final
calibration check. The Heston simulator can also produce zero-variance days
that reveal drift exactly, which a continuous neural density cannot fully
represent. These are development examples. Keep those limitations alongside
the results. See the [report](assets/results/trajectories/closeout-20261031/closeout-summary.json),
[drift figure](assets/results/trajectories/closeout-20261031/drift-learning.png)
and [coverage figure](assets/results/trajectories/closeout-20261031/marginal-coverage.png).

## Give feedback on an application

Useful starting points:

- **Heston:** are the model, quote noise, posterior comparison and predictive
  checks clear?
- **Rough Bergomi:** is the supplied/inferred curve distinction clear? Does
  each Additional study have a useful question, action and takeaway?
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
