# sbi4finance

Tutorial material on amortized / simulation-based Bayesian inference for finance, built on [BayesFlow](https://github.com/bayesflow-org/bayesflow).

## Tentative schedule

| # | Topic | Time | Lead |
|---|-------|------|------|
| 1 | Problem introduction: Bayes for volatility models | 15 min | Francesco |
| 2 | What is amortized Bayes? | 15 min | Alex |
| 3 | Relevant neural network architectures | 10 min | Stefan |
| 4 | Basic worked-out example in BayesFlow | 20 min | Stefan |
| 5 | Advanced: something interesting for finance | 40 min | Francesco & Alex |
| 6 | Advanced: changing priors and dealing with misspecified models | 15 min | Stefan |

Total: ~2 hours.

## Setup

One environment, managed by [pixi](https://pixi.sh) from `pyproject.toml`:

```bash
pixi install            # BayesFlow 2.0.14 on the jax backend, marimo, python 3.12
pixi run selfcheck      # simulator sanity check
pixi run check          # marimo lint for notebooks/
pixi run edit           # marimo editor (no auth token, so agents can pair with it)
```

`pixi shell` drops you into the environment. `KERAS_BACKEND=jax` is set on activation.

## Notebooks

Notebooks are [marimo](https://marimo.io) files: plain Python, reactive, git-friendly.

| Notebook | Schedule item | What it shows |
|---|---|---|
| `notebooks/01_mmar_univariate.py` | 4 | Univariate multifractal model of asset returns (MMAR): simulator, prior predictive check, adapter and networks, pretrained checkpoint or CPU toy fit, recovery and calibration, posterior on SPMO and VOO |
| `notebooks/svj_var_pipeline.py` | 5 (proposal) | SVJ advanced example: simulator, particle filter, pretrained amortized posterior, misspecification test and audit on a real window, predictive VaR / ES with the uncertainty of the risk number |

Export for people without marimo: `pixi run marimo export ipynb notebooks/01_mmar_univariate.py -o /tmp/01.ipynb --include-outputs`.

### Pairing with an agent

`.claude/skills/marimo-pair/` is the [marimo-pair](https://github.com/marimo-team/marimo-pair) skill (Apache-2.0, vendored at commit 6cecaff). With `pixi run edit` running and the notebook open in the browser, an agent runs code in the live kernel and commits cell changes through marimo rather than editing the `.py` file. See the skill for the rules; the short version is that during a live session the kernel, not the file, is the source of truth.

## SVJ advanced example: the end-to-end pipeline

A proposal for item 5: calibrate a stochastic-volatility model with jumps (SVJ, Bates 1996 in the daily form of Eraker, Johannes and Polson 2003) from the last 1,000 daily returns of one name, and carry the posterior into VaR and ES. `src/sbi4finance/svj/` holds the whole daily process; [`docs/svj-pipeline.md`](docs/svj-pipeline.md) describes the design, how each piece was checked, and the results.

```bash
pixi run test-svj                  # unit tests: simulator, particle vs grid filter, MCMC diagnostics, risk, checks, audit
pixi run svj-daily --ticker VOO    # one window, end to end: posterior -> misspecification -> audit -> VaR / ES
pixi run svj-train                 # reference network, 2.56 million simulated series (about 1.5 h on 4 CPU cores)
pixi run svj-validate              # SBC, coverage, recovery, contraction; builds the misspecification reference
pixi run svj-agreement             # network and audit against particle MCMC on four series spanning the prior
pixi run svj-backtest              # VaR and ES backtests with a known truth
pixi run svj-rolling               # rolling posteriors and misspecification p-values on VOO and SPMO
```

## Cluster (Oscar)

`cluster/` holds what is lab-wide; `.claude/skills/cluster-setup/` walks a user through the per-person part:

```bash
pixi run python cluster/discover_cluster.py --ssh-host oscar   # which lanes do I hold? -> cluster/oscar.local.json
sbatch cluster/train.sbatch                                   # retrain the univariate checkpoint on one GPU
```

## Layout

```
notebooks/            marimo notebooks, one per code-bearing schedule item
src/sbi4finance/      package: mmar/univariate.py is vendored from amortized-fractals; svj/ is the SVJ pipeline
scripts/              train_univariate.py reproduces checkpoints/univariate
data/                 pinned return CSVs; notebooks never download
checkpoints/          pretrained approximators (univariate: 6 MB; svj: 6 MB plus its misspecification reference)
tests/                unittest suites (pixi run test-svj)
docs/                 SVJ pipeline notes; docs/results/ holds the validation and evaluation outputs
cluster/              Oscar discovery script and sbatch template
.claude/skills/       marimo-pair (vendored) and cluster-setup
```

Simulator, data cache and checkpoint come from Stefan Radev's [amortized-fractals](https://github.com/stefanradev93/amortized-fractals).
