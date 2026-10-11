# M3 Heston and a precisely specified Gaussian-jump SVJ

**Status:** M3.1–M3.2 implemented in 4/10 with a NumPy reference, serial Numba stepping, and [bounded numerical/timing evidence](../../heston-svj-kernels.md). The tested discrete law and continuation pass; continuous-time fidelity has diagnostic support only for the reported cases/grids. Broad-domain numerical acceptance, M3.3 and collaborator parity remain pending. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md) for kernels; the accepted local bank/loader pipeline in [M2.3](02-banks-and-cluster.md) for the returns inference pilot. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

Introduce hidden variance, leverage, and price jumps while preserving agreement between simulation, inference references, and forecasting. Distinguish our declared continuous-time implementation from Francesco's existing experiment until equivalence is demonstrated.

## Tasks

1. Implement a native Heston reference with explicit initial variance, parameter supports, correlated drivers, and a documented positivity treatment. Separate observation intervals from integration substeps. Select production numerics only after comparison with a stronger reference and refinement studies, including accessible-zero-variance regimes.
2. Add a supported Gaussian log-price jump extension. Specify full Poisson counts, jump-size law, independence assumptions, and drift compensation. Verify multiple-jump intervals; do not silently replace the compound Poisson process with a Bernoulli approximation.
3. Define a returns task targeting selected static parameters and the complete current continuation state. For a scheme retaining a potentially negative auxiliary variance, that auxiliary state cannot be replaced by its nonnegative reported variance. Forecasts use paired joint parameter–state draws. Preserve the same transition and initialization in filters, simulators, and forecasts.
4. When Francesco's bundle arrives, create an adapter around its exact model, preprocessing, parameterization, discretization, and checkpoint. Reproduce a saved seeded example, checkpoint output, and reference comparison before substituting kernels or retraining. Record any mapping and numerical discrepancy explicitly.
5. Include a serial Numba stepping kernel in M3.1–M3.2 alongside a readable, batched NumPy reference. Validate the declared law and numerics, compare the backends using matched prescribed random inputs, and measure cold compilation, warm stepping, random-input generation and complete simulation separately. Compare against vectorized NumPy across paths; do not use only slow scalar Python loops as the baseline. Compilation is part of this increment, with no promised speedup. Parallel execution remains conditional on measured need and stable example identities. Keep collaborator parity separate from numerical refinement.
6. Train a small joint posterior using generated histories, initialization/context, and targets through the accepted bank/adapter path, continuing forecasts from its sampled state. Compare independent synthetic held-out histories and futures against a validated same-model reference; assess joint dependence and predictive quantities before a large training run. These S checks are not historical backtests. A reduced target with declared fixed parameters is acceptable if the full task is not yet validated.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M3.1 Heston and continuation | Batched NumPy reference, serial Numba stepping, explicit discrete scheme, complete continuation state, and difficult-region fixtures | Pass moment, covariance, refinement, backend-agreement and matched-continuation checks; report separate compilation/steady-state measurements. Accept numerical fidelity only on the tested domain/grid. This shared-kernel gate unlocks M4 without requiring returns inference or Gaussian jumps. |
| M3.2 Gaussian price jumps | A separately identified compound-Poisson extension using the same compiled stepping interface | Verify counts, jump moments, compensation and zero-jump limits against the accepted reference. Recheck continuation and matched-input backend agreement after jumps are added. |
| M3.3 Joint returns inference | Versioned task, same-model inference reference, pilot bank/checkpoint and conditional forecast report using M2.3 | Specify and validate the reference procedure before training at scale. Accept joint parameter/state and predictive comparisons, or retain the kernels while recording inference as pending. This gate unlocks the returns route to M8. |
| M3.F Optional collaborator reproduction | Adapter and compatibility report for the exact supplied equations, preprocessing, checkpoint and saved cases | Reproduce the saved case before substitutions. Missing artifacts leave this gate pending and prevent a parity claim; they do not block native kernels, M3.3 or M4. |

Implement M3.1–M3.3 in order; M3.F can proceed independently when artifacts arrive. The first kernels live in `src/sbi4finance/heston.py` and their pilot configuration in `configs/tasks/heston_svj_pilot.json`; no simulator registry or adapter framework is needed. Add modules only as concrete outputs require them. The collaborator adapter does not silently replace the native model or its reference.

## Deliverables

Heston and Gaussian-price-jump capabilities, model-specific continuation, task configurations, difficult-region fixtures, a pilot posterior/forecast report, and a collaborator compatibility report. The report may state that the bundle is pending; common model work does not depend on its arrival.

## Acceptance and tests

Check variance moments, leverage/cross-covariances, jump count and size moments, compensation, zero-jump and deterministic-variance limits, and timestep sensitivity. Test continuation against uninterrupted simulations using matched prescribed random inputs. Evaluate backend agreement statistically and against declared numerical tolerances. Where references exist, assess parameter–state posterior dependence and conditional discrepancies separately from marginal SBC.

## Decision gate and deferred scope

Accept native Heston and the collaborator adapter independently. S requires no real return window. An optional M window normally conditions frozen weights without observed parameter/state truth; E additionally requires held-out later outcomes and a chronological evaluation design. Do not claim Francesco parity without artifacts, or impose the Feller condition merely to conceal a solver limitation. Defer joint price/variance jumps, automatic model selection, general-purpose filtering infrastructure, and strong risk-benefit claims until supported by matched experiments.
