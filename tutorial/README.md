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

## Trajectory inference

The three models use daily returns and estimate drift. The joint rough model
also estimates its variance curve. The estimator accepts any integer history
length from **1 to 1,260 trading days** (five years).

Start with the rough model and its supplied variance curve. Inspect returns,
then use the early simulator to change parameters while keeping innovations
fixed. It shows eight one-year price paths, daily returns and realized volatility.
That forward example stays separate from the selected fitted history.
Its y axes stay fixed as parameters change. Choose **Standard**, **Wide** or
**Very wide** yourself; each panel reports points outside the view. The full
series and its minimum/maximum stay available in the axis-details table.
Heston and the inferred-curve model remain available as extensions.

There are two explicit routes:

- **Train here:** generate 128 new one-year histories, fit a 16-unit GRU and
  two-block spline coupling flow for eight updates, then infer parameters for
  an independent history. Results stay in the session and have no calibration claim.
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

**Saved mode** offers only lengths with saved posterior snapshots, so the
returns and posterior use the same history. **Live mode** accepts any integer
length. With the matching model downloaded, the CPU action updates the same
posterior plots using exactly that prefix. Without it, the labelled saved
fallback remains available. Changing the model, history or length clears a
previous live result.

The two model sources are labelled beside their outputs. The tutorial fit does
not reproduce or validate the supplied five-year fits. Production-bank
instructions and detailed input/diagnostic tables are optional.
Parameter inference alone does not provide a conditional continuation forecast.

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
