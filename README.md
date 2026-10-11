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

## Simulator and inference plan

The [implementation plan and mathematical guide](docs/simulator-plan/README.md) describes the simulator library, pre-simulated training banks, and returns-inference and option-pricing workflows. It includes nine milestone subplans, beginner-oriented mathematical notes, and validation requirements. Shared-foundation implementation has started; the tutorial's central application remains open.

Its [delivery and evidence boundary](docs/simulator-plan/README.md#synthetic-delivery-and-market-evidence) requires a complete offline synthetic teaching workflow. Actual-market application and empirical evaluation are optional additions with separate data and evidence requirements; acquisition does not block the synthetic core.

## Setup

The first simulator-ladder implementation is the [GBM reference example](docs/gbm-reference.md): exact simulation, an analytic drift posterior, conditional future losses and separate European price checks. Run `pixi run gbm-demo` and `pixi run test-gbm`. It uses generated data and has no neural-training or market-data prerequisite.

The next increment adds a [saved synthetic bank and BayesFlow training/reload example](docs/gbm-bank-training.md). Run `pixi run gbm-bank --out /tmp/gbm-bank`, then `pixi run gbm-train --bank /tmp/gbm-bank --out /tmp/gbm-trained`. Both output directories must be new. A separate [one-CPU Oscar smoke run](docs/gbm-cluster-smoke.md) has verified actual cluster generation, storage and selected-example reproduction.

[Heston and Gaussian-price-jump kernels](docs/heston-svj-kernels.md) add hidden variance, leverage, signed-state continuation and serial Numba stepping. Run `pixi run test-heston`, `pixi run heston-validate`, and `pixi run heston-benchmark` for the synthetic numerical/timing pilots.

The [joint Heston inference pilot](docs/heston-joint-inference.md) adds a synthetic bank, a same-model particle reference, BayesFlow training/reload and conditional futures. Its saved checkpoint **failed conditional test checks** and remains provisional; the working pipeline and full reports are available for the next inference iteration. No market data are needed.

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

Each simulator milestone includes a didactic review notebook explaining the
new functionality, the mathematics, a small runnable example and the actual
results with their limits. These review notebooks are separate from the final
two-hour participant curriculum. Closely related increments share named sections.

| Review notebook | Milestones / increments | Run |
|---|---|---|
| [GBM, banks and cluster](notebooks/01_gbm_foundations.py) | M0–M2 / 1–3: exact inference, array roles, reproducible banks and the recorded cluster result | `pixi run review-foundations` |
| [Heston and Gaussian jumps](notebooks/04_heston_svj_kernels.py) | M3.1–M3.2 / 4: diffusion/jump simulation, continuation, numerical checks and Numba timings | `pixi run review-kernels` |
| [Joint Heston inference](notebooks/05_heston_joint_inference.py) | M3.3 / 5: observed histories, reference/neural joint posteriors, conditional futures and the failed frozen checks | `pixi run review-heston` |

Default execution uses synthetic examples and committed evidence, with no market
downloads, cluster submission or full training run. Recorded results and fresh
illustrations are labeled separately. The joint-inference notebook keeps the
original provisional checkpoint visible during further development.

See the [review guide and verification record](docs/milestone-review-notebooks.md)
for export commands, evidence boundaries and the current inference results.

| Notebook | Schedule item | What it shows |
|---|---|---|
| `notebooks/01_mmar_univariate.py` | 4 | Univariate multifractal model of asset returns (MMAR): simulator, prior predictive check, adapter and networks, pretrained checkpoint or CPU toy fit, recovery and calibration, posterior on SPMO and VOO |

Export for people without marimo: `pixi run marimo export ipynb notebooks/01_mmar_univariate.py -o /tmp/01.ipynb --include-outputs`.

### Pairing with an agent

`.claude/skills/marimo-pair/` is the [marimo-pair](https://github.com/marimo-team/marimo-pair) skill (Apache-2.0, vendored at commit 6cecaff). With `pixi run edit` running and the notebook open in the browser, an agent runs code in the live kernel and commits cell changes through marimo rather than editing the `.py` file. See the skill for the rules; the short version is that during a live session the kernel, not the file, is the source of truth.

## Cluster (Oscar)

`cluster/` holds what is lab-wide; `.claude/skills/cluster-setup/` walks a user through the per-person part:

For the simulator ladder, [the CPU smoke guide](docs/gbm-cluster-smoke.md) uses `cluster/simulate.sbatch`, current allocation discovery, and an explicit Python interpreter/output path. The existing script below is for MMAR GPU training.

```bash
pixi run python cluster/discover_cluster.py --ssh-host oscar   # which lanes do I hold? -> cluster/oscar.local.json
sbatch cluster/train.sbatch                                   # retrain the univariate checkpoint on one GPU
```

## Layout

```
notebooks/            marimo notebooks, one per code-bearing schedule item
src/sbi4finance/      package: mmar/univariate.py is vendored from amortized-fractals
scripts/              train_univariate.py reproduces checkpoints/univariate
data/                 pinned return CSVs; notebooks never download
checkpoints/          pretrained approximators (univariate: 6 MB)
cluster/              Oscar discovery script and sbatch template
.claude/skills/       marimo-pair (vendored) and cluster-setup
```

Simulator, data cache and checkpoint come from Stefan Radev's [amortized-fractals](https://github.com/stefanradev93/amortized-fractals).
