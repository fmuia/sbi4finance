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

[HKDE paths and European Fourier pricing](docs/hkde-reference.md) now supply
[fixed synthetic option-panel banks](docs/option-banks.md). Run `pixi run option-bank
--config configs/tasks/hkde_option_bank.json --out /tmp/new-hkde-bank` and inspect
`pixi run review-hkde`. These are reusable numerical building blocks; Q posterior
inference is available as a [provisional baseline](docs/hkde-joint-inference.md).
The separate [fixed-contract learned price map](docs/hkde-price-surrogate.md)
passes its regression checks and preserves that baseline's inference discrepancy.
Run `pixi run review-hkde-surrogate` for the math, actual accuracy and costs.
A [bounded likelihood-correction follow-up](docs/hkde-corrected-inference.md)
uses the frozen flow as a proposal and separately tests the resulting weighted
posterior. It preserves the raw-flow failure and the first correction's
boundary-reliability stop; its reports distinguish proposal quality, posterior
accuracy, finite-pool SBC and actual numerical likelihood cost.
Validation passes, but the fresh correction test fails one SBC rank gate;
the corrected pipeline remains provisional. See the saved diagnostics before
choosing a participant workflow.
The [M8 options candidate](docs/m8-options-candidate.md) combines the frozen
neural proposal, numerical-likelihood correction and that accepted target-price
map in a paced lesson. Open it with
`PYTHONPATH=src pixi run review-hkde-corrected`. It currently runs as a **cached
diagnostic review**: validation passed, but the new test failed its theta SBC
gate (0.10325 > 0.09251), so live inference stays disabled and M8 acceptance
remains pending. Conditional and proposal-reliability checks passed; they do not
override the failed gate. The frozen run stopped without tuning or a rerun.
The independent-grid PIT also fails on these panels, so the rank result does
not by itself isolate bias from the correction algorithm.
The original neural failure and open returns-versus-options choice remain visible.
The [analysis follow-up note](docs/simulator-analysis-next-steps.md)
preserves the Heston findings while simulator and bank coverage expands.

[One-factor Bergomi](docs/bergomi-simulators.md) adds an explicit forward-variance
curve, exact OU factor stepping, complete continuation and bounded European
Monte Carlo pricing. Generate its numerical pilot with `pixi run option-bank
--config configs/tasks/bergomi_option_bank.json --out /tmp/new-bergomi-bank`.
Sampling uncertainty and time-grid bias remain distinct from quote noise.

[Rough Bergomi](docs/rough-bergomi.md) adds a joint Gaussian reference and FFT
hybrid Volterra driver. Its separate preset is
`configs/tasks/rough_bergomi_option_bank.json`, consumed by the same bank command.
`pixi run review-bergomi` covers both models, their bank results and numerical
limits. A current variance alone is not a continuation state for rough memory;
historical-return forecasting and neural Q posterior training remain separate work.

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
| [HKDE and option panels](notebooks/06_hkde_and_pricing.py) | M4 / 6: jump law, Q prices, noisy observations, bank roles and reference checks | `pixi run review-hkde` |
| [HKDE learned prices](notebooks/07_hkde_price_surrogate.py) | M5.2 / 7: supervised regression, matched price propagation, independent accuracy and cost checks | `pixi run review-hkde-surrogate` |
| [Bergomi models](notebooks/08_bergomi_models.py) | M6–M7 / 8–9: factor and rough drivers, memory, MC pricing, numerical and bank evidence | `pixi run review-bergomi` |
| [Corrected HKDE options candidate](notebooks/10_hkde_corrected_workflow.py) | M8 / 10: cached correction diagnostics; new-test theta SBC failed, live inference disabled, participant acceptance pending | `PYTHONPATH=src pixi run review-hkde-corrected` |

Default execution uses synthetic examples and committed evidence, with no market
downloads, cluster submission or full training run. Recorded results and fresh
illustrations are labeled separately. The joint-inference notebook keeps the
original provisional checkpoint visible during further development.
Notebook 10 is designed as a 30-minute participant candidate built from the
earlier reviews; it does not select or replace the full tutorial curriculum.
The failed frozen test keeps its live 2,048-proposal correction disabled. Opening
it reviews the cached evidence and mathematical workflow without new inference,
market fixtures or raw training banks. A successful review export would verify
the presentation, not accept the unfinished participant workflow.

The Heston review explains the [time-context correction](docs/heston-time-context.md),
with before/after recurrent-gate diagnostics and the improved but still-failing
validation results. The reserved final test remains unevaluated.
The [diagnostic review](docs/heston-diagnostics.md) adds direct graph links,
explains SBC versus conditional checks, demonstrates prior-to-posterior updating,
and records the frozen-weight sampling/reference precision follow-up.
The subsequent [signed-state transform experiment](docs/heston-state-transform.md)
also failed validation. Its result is retained in the review notebook; the
predeclared stop rule leaves Heston inference provisional and moves the next
implementation increment to HKDE law and reference pricing.

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
