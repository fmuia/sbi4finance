# sbi4finance

Tutorial material on amortized / simulation-based Bayesian inference for finance, built on [BayesFlow](https://github.com/bayesflow-org/bayesflow).

## Selected participant route: Q/options

On 12 October 2026 the user selected **GBM foundations followed by a provisional
corrected-HKDE diagnostic lesson** for [issue #26](https://github.com/fmuia/sbi4finance/issues/26).
The financial question is: which parameter pairs explain nine synthetic option
premiums, and what model prices do they imply for two excluded contracts?
The historical failed calibration gate stays visible. This is selected-case
teaching, not scientific acceptance, physical-measure risk forecasting or a
claim about market performance; formal HKDE I4 remains deferred.

The [combined integration review](docs/stack-integration-review.md) records exact
source heads, conflict choices and verification for issue #27.
The [13 October main-merge review](docs/main-merge-review.md) adds the completed
Heston decision and final verification. Francesco's [SVJ application](docs/svj-integration-review.md) is incorporated as an optional P/returns companion.
SVJ and HKDE share an [executable inference and training workflow](docs/shared-inference-workflow.md),
with task-specific networks, references and downstream quantities.
The [participant route card](docs/tutorial-participant-route.md) specifies the
sections, exercises, inputs, assets and modes. The schedule below totals **120
minutes including exercises**; lead assignments carry forward the original
proposal and remain for the team to confirm.

| Minutes | Topic / participant activity | Proposed lead |
|---|---|---|
| 0–10 | Financial question, P/Q distinction and data roles | Francesco |
| 10–30 | Exact GBM inference and separate Q expectation pricing | Stefan |
| 30–45 | Amortized inference; spline coupling flow and its dense subnets; cached training/reload | Alex & Stefan |
| 45–60 | Local synthetic panel → raw joint draws → excluded price | Stefan |
| 60–75 | Likelihood correction and the full proposal-mixture denominator | Alex |
| 75–95 | Joint posterior → two price distributions; one opt-in 2,048-proposal demo or labelled saved replay | Francesco & Alex |
| 95–113 | Conditional checks, SBC and the shared numerical-reference alarm | Alex |
| 113–120 | Changing priors/task assumptions and the exit exercise | All |

**Delivery status:** curriculum selected. The pinned #27 combined revision
includes #22's local panel, cached default and explicit live opt-in, plus #29's
scientific-decision explanation. The [integration review](docs/stack-integration-review.md)
records combined tests and offline execution. #26's [participant delivery record](docs/tutorial-delivery-verification.md)
now confirms fresh locked setup and all five isolated exports on the same source
pin, with 19 figure appearances reviewed. Static mirror assets/links pass checks;
browser layout, interaction and offline runtime remain unverified. Scientific
acceptance remains false and formal I4 remains deferred.

The optional Heston, rough Bergomi and original MMAR reviews remain available;
they are outside this two-hour schedule and use their own task/checkpoint.

## Simulator and inference plan

For current code, PR dependencies, scientific limits and next review steps,
start with the [collaborator handover](docs/simulator-handover.md).
The [current code architecture](docs/code-architecture.md) maps the modules,
data flow, saved artifacts and extension points for both financial workflows.

The [implementation plan and mathematical guide](docs/simulator-plan/README.md) describes the simulator library, pre-simulated training banks, and returns-inference and option-pricing workflows. It includes nine milestone subplans, beginner-oriented mathematical notes, and validation requirements. The selected participant application is the provisional Q/options route above; implementation and scientific evidence remain separately scoped.

Its [delivery and evidence boundary](docs/simulator-plan/README.md#synthetic-delivery-and-market-evidence) requires a complete offline synthetic teaching workflow. Actual-market application and empirical evaluation are optional additions with separate data and evidence requirements; acquisition does not block the synthetic core.

## Setup

The first simulator-ladder implementation is the [GBM reference example](docs/gbm-reference.md): exact simulation, an analytic drift posterior, conditional future losses and separate European price checks. Run `pixi run gbm-demo` and `pixi run test-gbm`. It uses generated data and has no neural-training or market-data prerequisite.

The next increment adds a [saved synthetic bank and BayesFlow training/reload example](docs/gbm-bank-training.md). Run `pixi run gbm-bank --out /tmp/gbm-bank`, then `pixi run gbm-train --bank /tmp/gbm-bank --out /tmp/gbm-trained`. Both output directories must be new. A separate [one-CPU Oscar smoke run](docs/gbm-cluster-smoke.md) has verified actual cluster generation, storage and selected-example reproduction.

[Heston and Gaussian-price-jump kernels](docs/heston-svj-kernels.md) add hidden variance, leverage, signed-state continuation and serial Numba stepping. Run `pixi run test-heston`, `pixi run heston-validate`, and `pixi run heston-benchmark` for the synthetic numerical/timing pilots.

The [joint Heston inference pilot](docs/heston-joint-inference.md) adds a synthetic bank, a same-model particle reference, BayesFlow training/reload and conditional futures. Its saved checkpoint **failed conditional test checks** and remains provisional. The [completed Heston decision](docs/heston-validation-decision.md) retains its optional diagnostic lesson and defers further experiments for the selected options course. No market data are needed.

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
gate (0.10325 > 0.09251), so formal M8 acceptance remains pending. Conditional
and proposal-reliability checks passed; they do not
override the failed gate. The frozen run stopped without tuning or a rerun.
The independent-grid PIT also fails on these panels, so the rank result does
not by itself isolate bias from the correction algorithm.
The [calibration-gate audit](docs/hkde-calibration-gate-audit.md) checks this
distinction: ideal reference sampling would trigger the same theta gate in
98.85% of diagnostic runs on this fixed cohort. Direct CDF checks support
bounded approximation accuracy; the historical failure and participant gate
remain unchanged. Open `PYTHONPATH=src pixi run review-hkde-calibration` for
the didactic explanation, conditional controls and remaining uncertainty.
For the [opt-in provisional live demonstration](docs/hkde-provisional-demo.md),
run `pixi run demo-hkde-corrected`. It executes the same local panel with 2,048
proposals and reports its numerical checks separately from formal acceptance.
The original neural failure remains visible. The selected Q/options teaching scope is recorded in the [route card](docs/tutorial-participant-route.md); it does not upgrade scientific acceptance.
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
historical-return forecasting remains separate work. The
[restricted rough posterior pilot](docs/rough-joint-inference.md) adds a frozen
512-step Q inference task, cluster chunks, an independent importance reference,
and a separate review of conditional, SBC and mean-price evidence.
The [bounded reference extension](docs/rough-reference-expansion.md) preserves
the original 19/20-case precision stop and expands only its independent
reference bank to 8,192 rows. All 20 reference cases and all seven neural
validation gates now pass, including conditional/joint checks, marginal SBC
and target-price means. The later [held-out study](docs/rough-heldout-test.md)
passes its reference and six of seven neural gates, but eta's posterior SD is
31.0% too small on one case (limit 25%). Notebook09 explains this conditional
failure despite passing SBC and target means. The candidate remains provisional;
all 256 test cases are now used. No retraining or bank expansion followed.

The default environment is managed by [pixi](https://pixi.sh) from `pyproject.toml`:

```bash
pixi install --locked   # BayesFlow 2.0.14 on the jax backend, marimo, python 3.12
pixi run selfcheck      # simulator sanity check
pixi run check          # marimo lint for notebooks/
pixi run edit           # marimo editor (no auth token, so agents can pair with it)
```

`pixi shell` drops you into the environment. `KERAS_BACKEND=jax` is set on activation.

## Notebooks

Notebooks are [marimo](https://marimo.io) files: plain Python, reactive, git-friendly.

Each simulator milestone includes a didactic review notebook explaining the
new functionality, the mathematics, a small runnable example and the actual
results with their limits. The [selected two-hour curriculum](docs/tutorial-participant-route.md) uses named
sections of these reviews; the full development-review collection is optional. Closely related increments share named sections.

| Review notebook | Milestones / increments | Run |
|---|---|---|
| [HKDE calibration-gate audit](notebooks/11_hkde_calibration_audit.py) | M8 follow-up: threshold calibration, conditional reference-rank controls and finite-pool CDF errors; no acceptance override | `PYTHONPATH=src pixi run review-hkde-calibration` |
| [Corrected HKDE options candidate](notebooks/10_hkde_corrected_workflow.py) | M8 / 10: cached default and opt-in provisional live joint correction and two target-price distributions; historical theta SBC failure retained, participant acceptance pending | `PYTHONPATH=src pixi run review-hkde-corrected` |
| [GBM, banks and cluster](notebooks/01_gbm_foundations.py) | M0–M2 / 1–3: exact inference, array roles, reproducible banks and the recorded cluster result | `pixi run review-foundations` |
| [Heston and Gaussian jumps](notebooks/04_heston_svj_kernels.py) | M3.1–M3.2 / 4: diffusion/jump simulation, continuation, numerical checks and Numba timings | `pixi run review-kernels` |
| [Joint Heston inference](notebooks/05_heston_joint_inference.py) | M3.3 / 5: observed histories, reference/neural joint posteriors, conditional futures and the failed frozen checks | `pixi run review-heston` |
| [HKDE and option panels](notebooks/06_hkde_and_pricing.py) | M4 / 6: jump law, Q prices, noisy observations, bank roles and reference checks | `pixi run review-hkde` |
| [HKDE learned prices](notebooks/07_hkde_price_surrogate.py) | M5.2 / 7: supervised regression, matched price propagation, independent accuracy and cost checks | `pixi run review-hkde-surrogate` |
| [Bergomi models](notebooks/08_bergomi_models.py) | M6–M7 / 8–9: factor and rough drivers, memory, MC pricing, numerical and bank evidence | `pixi run review-bergomi` |
| [Rough posterior pilot](notebooks/09_rough_inference.py) | M7.3b / 9: numerical observation law, independent reference, prior control, joint inference, SBC and excluded-price means | `pixi run review-rough-inference` |

Default execution uses synthetic examples and committed evidence, with no market
downloads, cluster submission or full training run. Recorded results and fresh
illustrations are labeled separately. The joint-inference notebook keeps the
original provisional checkpoint visible during further development.
Notebook 10's original 30-minute pacing is a standalone review suggestion.
For the selected curriculum, follow the schedule and exercises in the route
card above; combined execution is verified in the linked integration records,
with the remaining browser limits stated separately.
Opening it reviews cached evidence; `pixi run demo-hkde-corrected` runs
the bounded 2,048-proposal correction and target-price calculation
offline while keeping the historical failure visible. The fixed example is a
development case, not new calibration evidence. No market fixtures or raw
training banks are required. Inspect the [joint posterior](docs/results/hkde-live-demo/live-joint.png),
[target-price distributions](docs/results/hkde-live-demo/live-target-prices.png)
and [live execution record](docs/results/hkde-live-demo/verification.json).
Technical execution and single-case numerical checks remain separate from M8
scientific acceptance. The earlier cached-only export (17 cells, zero errors,
8.29 seconds) is preserved as a historical record, not a live-inference timing.

The Heston review explains the [time-context correction](docs/heston-time-context.md),
with before/after recurrent-gate diagnostics and the improved but still-failing
validation results. The reserved final test remains unevaluated.
The [diagnostic review](docs/heston-diagnostics.md) adds direct graph links,
explains SBC versus conditional checks, demonstrates prior-to-posterior updating,
and records the frozen-weight sampling/reference precision follow-up.
The subsequent [signed-state transform experiment](docs/heston-state-transform.md)
also failed validation. Its result is retained in the review notebook; the
predeclared stop rule leaves Heston inference provisional. The completed
decision defers new Heston experiments while retaining this optional lesson.

See the [review guide and verification record](docs/milestone-review-notebooks.md)
for export commands, evidence boundaries and the current inference results.

| Notebook | Schedule item | What it shows |
|---|---|---|
| `notebooks/01_mmar_univariate.py` | Optional companion | Univariate multifractal model of asset returns (MMAR): simulator, prior predictive check, adapter and networks, pretrained checkpoint or CPU toy fit, recovery and calibration, posterior on SPMO and VOO |
| `notebooks/tutorial_1_why_bayesian.py` | Slot 1: intro | One beta, then five hundred: the Bayesian market model, the winner's curse among estimated betas, a prior learned from the cross-section, Bloomberg's rule against Bayes' rule, the hedge, what each interval covers, populations where the prior is wrong, Blume's test on real stocks, and why the closed form disappears for volatility models |
| `notebooks/svj_var_pipeline.py` | Optional P companion | Daily SVJ joint inference, particle audit and conditional log-loss forecasts |

Export for people without marimo: `pixi run marimo export ipynb notebooks/01_mmar_univariate.py -o /tmp/01.ipynb --include-outputs`.

### Pairing with an agent

`.claude/skills/marimo-pair/` is the [marimo-pair](https://github.com/marimo-team/marimo-pair) skill (Apache-2.0, vendored at commit 6cecaff). With `pixi run edit` running and the notebook open in the browser, an agent runs code in the live kernel and commits cell changes through marimo rather than editing the `.py` file. See the skill for the rules; the short version is that during a live session the kernel, not the file, is the source of truth.

For a lesson view with visual annotations and an MCP endpoint, use the optional
[marimo collaboration setup](docs/marimo-collaboration.md):

```bash
pixi install --locked -e collaboration
pixi run -e collaboration studio-build
pixi run -e collaboration studio-svj
```

Open the printed localhost URL and keep the browser tab open. Studio presents
the existing SVJ notebook's live outputs; Lens lets you point to a result or
follow a guided walkthrough, and Pair controls the same computation. The guide
covers session selection, the `mcp-check` task and other tutorial notebooks.

## Shared inference workflow and optional SVJ application

The [shared workflow guide](docs/shared-inference-workflow.md) follows the same
sequence for SVJ and HKDE: task and simulation checks, separate training and
validation, selected-checkpoint reload, compatible observation preparation,
joint inference, diagnostics and correction/fallback, then downstream analysis.
Both applications execute the same inference runner and training lifecycle.
Architectures, summary networks, numerical references and risk/pricing targets
remain specific to the financial question.

Francesco's daily SVJ application infers eight parameter coordinates and terminal
volatility from 1,000 daily returns. Its particle audit, PMMH and conditional
forecasting extend the P route; its finite model remains distinct from the
smaller Heston diagnostic example. The optional lesson defaults to a complete
cached synthetic case, with no market CSV or network access. Reported VaR/ES is
**percent log-loss**, and local diagnostic status accompanies forecasts.

```bash
pixi run review-svj
pixi run demo-svj                 # bounded synthetic live inference, no PMMH
pixi run svj-bank --config configs/tasks/svj_daily_smoke.json --out /tmp/svj-bank
pixi run svj-bank-check --bank /tmp/svj-bank
pixi run svj-train --bank /tmp/svj-bank --epochs 1 --batch-size 8 --out /tmp/svj-trained
```

Training and validation require new explicit output directories. The contributed
network, training record, diagnostic reference and numerical results remain
preserved; [historical results](docs/svj-results-summary.md) retain mixed SBC and
conditional-reference evidence. A tiny train/reload or a successful illustration
is not scientific acceptance. The [integration record](docs/svj-integration-review.md)
separates software, cluster, teaching and scientific evidence.

## Cluster (Oscar)

`cluster/` holds what is lab-wide; `.claude/skills/cluster-setup/` walks a user through the per-person part:

For the simulator ladder, [the CPU smoke guide](docs/gbm-cluster-smoke.md) uses `cluster/simulate.sbatch`, current allocation discovery, and an explicit Python interpreter/output path. The existing script below is for MMAR GPU training.

```bash
pixi run python cluster/discover_cluster.py --ssh-host oscar   # which lanes do I hold? -> cluster/oscar.local.json
sbatch cluster/train.sbatch                                   # retrain the univariate checkpoint on one GPU
```

## Layout

```
notebooks/            marimo milestone reviews and participant candidates
src/sbi4finance/      model kernels, banks, inference, references and forecasts
configs/              task designs and frozen experiment protocols
scripts/              generation, training, evaluation and plotting commands
tests/                numerical, data-boundary and workflow checks
data/                 pinned return CSVs for the earlier MMAR example
checkpoints/          saved approximators with task metadata and status notes
docs/                 plans, architecture, handover and scientific reports
docs/results/         committed summaries, selected arrays and diagnostic plots
cluster/              Oscar discovery and CPU-bank / MMAR-training wrappers
.claude/skills/       marimo-pair (vendored) and cluster-setup
```

The original MMAR simulator, data cache and checkpoint come from Stefan Radev's [amortized-fractals](https://github.com/stefanradev93/amortized-fractals). The simulator ladder adds the model and inference modules described in the [architecture guide](docs/code-architecture.md).

## Local observation fixtures

[Local data preparation](docs/local-data.md) supplies CSV/JSON IO, daily return windows and [fixed HKDE premium panels](docs/hkde-local-panels.md), with explicit observation/context roles and separate synthetic truth. The CSV/JSON and preparation helpers need no downloads or extra dependencies. Run `pixi run data-check` for unit tests; `pixi run hkde-data-check` checks the real HKDE checkpoint handoff when simulator PR #15 is available. That checkpoint remains provisional; passing IO checks does not establish scientific acceptance.

`pixi run hkde-bank-data-check` exercises the generated-bank → local-panel →
checkpoint path using the simulator's existing writer, reader and adapter.
It needs the simulator PR #15 implementation (or `--simulator-root` pointing to
that checkout); no provider account or real-market data are used.

The [cached HKDE data walkthrough](notebooks/hkde_data_walkthrough.py) is a
small offline example of a synthetic panel, paired parameter draws and a
held-out option's model-price uncertainty. Run `pixi run hkde-data-walkthrough`;
see [reproduction and evidence limits](docs/hkde-offline-teaching.md). Its
checkpoint is **provisional**, and the failed price gate is shown explicitly.

The [teaching handoff](docs/hkde-teaching-integration.md) connects that same local
panel to the corrected lesson and a cached raw/weighted/reference comparison.
Run `pixi run hkde-teaching-integration-check` to verify both lessons together
offline. Integration milestones I1–I3 and the separate I4a provisional live
demonstration are complete; formal I4 acceptance remains pending. Add `--live`
to the check command to exercise and record the actual demonstration.

## Inference tasks and data planning

Start with the [data implementation roadmap and milestone subplans](docs/data-plan/README.md). It sequences common contracts, local loading, separate returns/options preparation, checkpoint compatibility and the complete synthetic teaching workflow, with market application and empirical evaluation as optional branches. The [scenario specification](docs/inference-scenarios-and-data.md) explains the statistical objects, the [option format reference](docs/data-loading-plan.md) defines panels, and the [source comparison](docs/data-source-landscape.md) evaluates acquisition choices. The plans preserve their original scope; the implemented local interfaces and selected provisional Q/HKDE route are composed in this checkout.

The [data dependency and evidence levels](docs/inference-scenarios-and-data.md#three-delivery-and-evidence-levels) separate a complete synthetic teaching core (no market data required), an optional real-market application, and empirical evaluation using held-out observations. Market-data acquisition does not block simulator development, synthetic training or the required notebook.
