# sbi4finance

Code and materials for the ICAIF '26 tutorial **Amortized Bayesian Calibration of Volatility
Models** (Milan, 15 November 2026), built with [BayesFlow](https://github.com/bayesflow-org/bayesflow).
The [tutorial website](https://fmuia.github.io/sbi4finance/) has the programme, the presenters,
slides and static copies of the notebooks.

The tutorial shows how amortized simulation-based inference turns model calibration into a
Bayesian posterior that is fast enough to compute every day and cheap enough to check. A network is
trained once on simulated data. It then returns the posterior of the model's parameters for new
data in under a second, where MCMC takes minutes to hours.

## What was presented

The tutorial opened with slides on Bayesian calibration, amortized inference and network
architectures, and then worked through two notebooks.

| Notebook | What it does | Static copy |
| --- | --- | --- |
| [Heston and rough Bergomi on SPX options](lab/notebooks/heston_spx_calibration.py) | Calibrates Heston to 63 SPX option quotes. The notebook checks the posterior against PyMC, prices a product the market does not quote, and diagnoses what Heston misses. It then repeats the inference with rough Bergomi. | [page](https://fmuia.github.io/sbi4finance/notebooks/heston_spx_calibration.html) |
| [Parameter estimation from return histories](tutorial/notebooks/trajectory_inference.py) | Infers the parameters of rough Bergomi (and of Heston) from daily returns, as the history grows. It simulates, trains live or loads a pretrained network, and checks calibration. | [page](https://fmuia.github.io/sbi4finance/notebooks/trajectory_inference.html) |

[**docs/spx-option-calibration.md**](docs/spx-option-calibration.md) explains the SPX notebook in
about 15 minutes, for readers who want the argument before the code.

## Run the notebooks

**Requirements.** [Pixi](https://pixi.sh), which installs Python 3.12, BayesFlow 2.1, marimo and
the other dependencies from the lock files. A laptop CPU is enough. The return-history notebook
runs on macOS (Apple Silicon), Linux (x86-64) and Windows (x86-64); the SPX notebook needs macOS
(Apple Silicon) or Linux (x86-64), for example through WSL on Windows. The static copies above
work in any browser.

Each notebook opens in the browser; stop it with Ctrl+C before starting the other. The saved
examples are included, so you do not need a cluster, a training bank or extra downloads.

Return histories, from `tutorial/`:

```bash
cd tutorial
pixi install --locked
pixi run --locked path-inference
```

Heston and rough Bergomi on SPX, from `lab/`, in the lab's `core-reference` environment (the
scientific stack plus PyMC, which computes the exact reference posteriors):

```bash
cd lab
pixi install --locked -e core-reference
pixi run --locked -e core-reference marimo edit notebooks/heston_spx_calibration.py
```

The SPX notebook's optional pretrained Heston estimator is a separate 465 MB archive on the
project's Google Drive, available to the project team only. Without it the notebook still shows
every saved result and can train a small network live; only fresh sampling from the pretrained
estimator is unavailable. See the [lab README](lab/README.md#review-the-heston-spx-notebook). The
[tutorial README](tutorial/README.md) covers the return-history notebook's options and checks.

## What the results show

- **Heston on SPX.** The network returns a posterior in under 0.3 s, where PyMC takes 17 to
  32 minutes. On simulated data it is well calibrated for five of the six parameters and too
  cautious for the sixth. On the SPX quotes, however, Heston itself fails: it misses them by
  1.1 volatility points (RMS), far beyond the assumed quote noise. Refitting it to different
  subsets of the quotes moves a product's price 18 times more than the posterior range.
- **Rough Bergomi on SPX** lowers the misfit by about a quarter. Its amortized posterior still
  underestimates the noise level that MCMC finds, so it is a development result, not an accepted
  calibration.
- **Return histories.** All three trajectory estimators failed their final calibration check. The
  Heston trajectory simulator also has a known limitation when estimating drift. See the
  [trajectory plan](lab/plans/PATH-INFERENCE-PLAN.md).

## Repository layout

| Folder | Contents |
| --- | --- |
| [docs/](docs/) | The written companion to the SPX notebook |
| [tutorial/](tutorial/) | The return-history notebook, the shared runtime both notebooks use, tests and saved examples |
| [lab/](lab/) | The SPX notebook, data generation, training runs, reference calculations and their evidence |
| [site/](site/) | The public tutorial website, published with GitHub Pages ([how to edit](site/README.md)) |

`tutorial/` is self-contained: it never imports the lab. The lab installs the tutorial runtime
as a local dependency, so keep the two folders together. To change a notebook, see
[Review or edit a notebook](tutorial/README.md#review-or-edit-a-notebook); the conventions for
both folders, for people and coding agents alike, are in [AGENTS.md](AGENTS.md).

Further notebooks prepared for the tutorial but not presented, including introductory demos and
an MMAR take-home, are kept in
[pull request 101](https://github.com/fmuia/sbi4finance/pull/101).

## Reproduce the numerical work

The generation, training and evaluation scripts run on a laptop or on a cluster. Follow the
[lab instructions](lab/README.md#canonical-workflow) and the
[training-data guide](lab/plans/training-data-guide.md). Large training banks and models are kept
outside Git, identified by SHA-256 hashes, as GitHub release assets or on the project's Google
Drive (which needs project access).

## Presenters

- Francesco Muia, Muia AI & Analytics Solutions
- Alexander Fengler, Brown University
- Stefan T. Radev, Rensselaer Polytechnic Institute
- Jerry M. Huang, Rensselaer Polytechnic Institute

The [tutorial website](https://fmuia.github.io/sbi4finance/#presenters) has their profiles and a
contact address.
