# Local panel to corrected HKDE lesson

This implements **I1–I3** of the [integration plan](hkde-teaching-integration-plan.md).
The same prepared synthetic panel now connects the data introduction to the
corrected-inference lesson. **I4 remains deferred:** the corrected validation
passed, but its independent test failed theta calibration. Successful notebook
execution does not change that scientific decision.

The subsequent [I4a provisional live demo](hkde-provisional-demo.md) adds an
explicit opt-in execution path based on the shared-reference audit. The I1–I3
record below describes the earlier cached handoff; formal I4 remains deferred.

## Run and inspect

From the combined checkout, with its pinned Pixi environment installed:

```bash
pixi run hkde-data-walkthrough
pixi run review-hkde-corrected
```

These open the [data introduction](../notebooks/hkde_data_walkthrough.py) and
[corrected lesson](../notebooks/10_hkde_corrected_workflow.py), respectively.
The introduction remains a small standalone cached example. The corrected lesson
adds the weighted comparison and existing diagnostics, and skips live inference
until both source-bound scientific reports pass.

For one reproducible combined check:

```bash
pixi run hkde-teaching-integration-check
# Optional new output paths; neither is overwritten:
pixi run hkde-teaching-integration-check \
  --out /tmp/hkde-integration.json --artifacts /tmp/hkde-integration-plots
```

The check copies only the required source/assets into a temporary directory,
blocks sockets and BayesFlow imports, then executes both notebooks. Market
fixtures, raw banks, other checkpoints and the original working directory are
absent. The HKDE checkpoint bytes remain present for provenance hashing but are
not loaded into a neural model. No scientific test is rerun.

## What is connected

1. The existing CSV/JSON loader reads synthetic **`validation:257`**. The lesson
   checks its full task against the actual saved checkpoint task, then extracts
   the nine ordered premiums as `[9,1]`. Prices must equal the recorded case.
   Truth and auxiliary fields are neither required nor used for conditioning.
2. A small table compares the same observation and held-out European put
   `(K=95,T=0.75)`: **4,096 raw draws**, **8,192 cached correction proposals**,
   and the numerical posterior reference. Corrected quantiles reuse the
   scientific weighted-summary function on matched price rows and weights.
   The larger precision cache is not a live 2,048-proposal participant run.
3. The existing live correction receives that local price array when its
   unchanged scientific gate permits. Currently it returns no live result.

All caches, checkpoint weights, pricing/inference functions, priors, numerical
settings and thresholds are unchanged. Editing the observation cannot reuse
the displayed posterior: the lesson rejects it until compatible inference and
reference results are supplied. No new loader abstraction or cache format was
introduced.

This supports the synthetic calibration lesson without any real observations.
Fitting a market snapshot would require compatible premiums and context plus
justified quote-error assumptions. Assessing market performance would also
need withheld quotes/dates and an empirical protocol. Neither claim follows
from this engineering check.

## Verification and dependency base

The [recorded combined check](results/hkde-teaching-integration.json) pins the
source and artifact hashes. It executes **8 introductory and 19 advanced cell
outputs**, renders **2 + 4 PNG plots**, and reproduces the weighted comparison.
It records `scientific_acceptance: false` and `live_executed: false` separately
from `passed: true`. All six plots were inspected. This uses an existing pinned
environment; it is not a fresh dependency-installation test or a live timing.

The combined checkout also passes **168 tests**, including four new boundary
tests for the actual lesson cell, and strict marimo checks for both notebooks.
The test suite includes existing tiny training/reload smoke tests in temporary
directories; it does not retrain or modify the published checkpoint. The older
intro-only and simulator-only verification reports remain historical records.

The implementation targets `integration/hkde-data-teaching-base`, commit
`0ee09dd9a6b5a5333960cb800e28d1f6dbe7b13d`. This isolated local merge combines:

| Published dependency | Pinned commit |
|---|---|
| [Simulator lesson PR #19](https://github.com/fmuia/sbi4finance/pull/19) | `160bccecea4e806e6efce1e422eabf6347b9fb1b` |
| [Integration plan PR #20](https://github.com/fmuia/sbi4finance/pull/20), including data PR #18 | `b9b37877ca92521b287eb3bf2bd92e2ea7093215` |

Only the shared README and Pixi task list needed conflict reconciliation. The
PR diff against this base contains the handoff, not the upstream implementations.
No existing PR or main branch was merged, and other worktrees were left alone.

The remaining step is **I4**: after the simulator/inference workstream supplies
scientifically accepted artifacts, run and record the actual offline
2,048-proposal participant calculation through this local entry point. Do not
override the gate or treat the cached precision result as that execution.
