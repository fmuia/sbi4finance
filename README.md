# sbi4finance

Materials for our ICAIF '26 tutorial on simulation-based inference in finance,
using [BayesFlow](https://github.com/bayesflow-org/bayesflow).

**We start with slides** for the introduction, amortized Bayes and network
architectures. **We then work through the notebooks presented at the tutorial:**

| Application | Notebook | Code |
| --- | --- | --- |
| Heston and rough Bergomi on SPX options | [Open notebook](lab/notebooks/heston_spx_calibration.py) | [lab/](lab/) and [shared/](tutorial/src/sbi4f_tutorial/shared/) |
| Parameter estimation from returns | [Open notebook](tutorial/notebooks/trajectory_inference.py) | [trajectories/](tutorial/src/sbi4f_tutorial/trajectories/) |

The SPX notebook runs from the [lab environment](lab/README.md#review-the-heston-spx-notebook).
It includes a live training example; the larger pretrained model is an optional download.

Shared code is in [shared/](tutorial/src/sbi4f_tutorial/shared/). The lab uses
this same code. Edit it here; there is no copying step.

## Run a notebook

Install [Pixi](https://pixi.sh), then run:

```bash
cd tutorial
pixi install --locked
pixi run --locked path-inference
```

It opens in your browser; press Ctrl+C to stop it. The SPX notebook has its own
[lab setup](lab/README.md#review-the-heston-spx-notebook).

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
| [site/](site/) | The public tutorial webpage, published with GitHub Pages ([how to edit](site/README.md)) |

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
