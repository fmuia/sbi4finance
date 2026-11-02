# sbi4finance

Materials for our ICAIF '26 tutorial on simulation-based inference in finance,
using [BayesFlow](https://github.com/bayesflow-org/bayesflow).

**We start with Beamer slides** for the introduction, amortized Bayes and network
architectures. **We then work through three application notebooks:** Heston
options, rough Bergomi options and parameter estimation from return histories.
The introductory notebooks remain available as supporting demos.

## Start editing here

For this feedback round, please focus on these three notebooks. We want to
refine the explanations, figures and code before worrying about exact timing.

| Application | Notebook | Code |
| --- | --- | --- |
| Heston options | [Open notebook](tutorial/notebooks/slot4_heston_bayesflow.py) | [heston_options/](tutorial/src/sbi4f_tutorial/heston_options/) |
| Rough Bergomi options | [Open notebook](tutorial/notebooks/slot5_calibration.py) | [rough_options/](tutorial/src/sbi4f_tutorial/rough_options/) |
| Return histories | [Open notebook](tutorial/notebooks/trajectory_inference.py) | [trajectories/](tutorial/src/sbi4f_tutorial/trajectories/) |

Shared code is in [shared/](tutorial/src/sbi4f_tutorial/shared/). The lab uses
this same code. Edit it here; there is no copying step.

## Run a notebook

Install [Pixi](https://pixi.sh), then run:

```bash
cd tutorial
pixi install --locked
pixi run --locked heston-options
```

Use `rough-options` or `path-inference` to open the other applications. Run one
at a time; press Ctrl+C before starting the next. Each opens in your browser.

The saved examples are included. **You do not need Oscar, a training bank or
additional model downloads to review the notebooks.**

Start with the [collaborator handover](tutorial/README.md#collaborator-handover)
for review and editing guidance. The [tutorial README](tutorial/README.md)
also covers setup, optional downloads and checks.

## Where things belong

| Folder | Contents |
| --- | --- |
| [tutorial/](tutorial/) | Notebooks, their code, tests and saved examples |
| [lab/](lab/) | Data generation, larger training runs, reference calculations and results |
| [course/](course/) | Background notes and figures |

## Run numerical work on your compute

The generation, training and evaluation scripts can run without Oscar or Slurm.
Follow the [lab instructions](lab/README.md#canonical-workflow) and
[training-data guide](lab/plans/training-data-guide.md).

Keep `tutorial/` and `lab/` together when copying the repo to another machine.
For current lab work, use `pixi run --locked -e core ...` from `lab/`.

## Current state

All three notebooks have saved examples, trained models and training banks.
Some numerical checks are still unfinished:

- Rough Bergomi needs further reference-posterior and product checks.
- All three trajectory estimators failed the final calibration check. The
  Heston trajectory simulator also has a known limitation when estimating drift.

Please keep these limitations visible when reviewing or editing results.
Details are in the [application plan](tutorial/curriculum/TUTORIAL-REVIEW.md)
and [trajectory plan](lab/plans/PATH-INFERENCE-PLAN.md).
