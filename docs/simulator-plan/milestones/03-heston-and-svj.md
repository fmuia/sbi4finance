# M3 Heston and a precisely specified Gaussian-jump SVJ

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md) for kernels; the accepted local bank/loader pipeline in [M2.3](02-banks-and-cluster.md) for the returns inference pilot. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

Introduce hidden variance, leverage, and price jumps while preserving agreement between simulation, inference references, and forecasting. Distinguish our declared continuous-time implementation from Francesco's existing experiment until equivalence is demonstrated.

## Tasks

1. Implement a native Heston reference with explicit initial variance, parameter supports, correlated drivers, and a documented positivity treatment. Separate observation intervals from integration substeps. Select production numerics only after comparison with a stronger reference and refinement studies, including accessible-zero-variance regimes.
2. Add a supported Gaussian log-price jump extension. Specify full Poisson counts, jump-size law, independence assumptions, and drift compensation. Verify multiple-jump intervals; do not silently replace the compound Poisson process with a Bernoulli approximation.
3. Define a returns task targeting selected static parameters and the complete current continuation state. For a scheme retaining a potentially negative auxiliary variance, that auxiliary state cannot be replaced by its nonnegative reported variance. Forecasts use paired joint parameter–state draws. Preserve the same transition and initialization in filters, simulators, and forecasts.
4. When Francesco's bundle arrives, create an adapter around its exact model, preprocessing, parameterization, discretization, and checkpoint. Reproduce a saved seeded example, checkpoint output, and reference comparison before substituting kernels or retraining. Record any mapping and numerical discrepancy explicitly.
5. Add measured Numba acceleration for suitable loops only after the NumPy reference passes. Keep the collaborator-parity path versioned separately from any numerical refinement.
6. Train a small joint posterior using generated histories, initialization/context, and targets through the accepted bank/adapter path, continuing forecasts from its sampled state. Compare independent synthetic held-out histories and futures against a validated same-model reference; assess joint dependence and predictive quantities before a large training run. These S checks are not historical backtests. A reduced target with declared fixed parameters is acceptable if the full task is not yet validated.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M3.1 Heston and continuation | Heston reference, declared production scheme, complete continuation state, and difficult-region fixtures | Pass moment, covariance, refinement and matched-continuation checks. This shared-kernel gate unlocks M4 without requiring returns inference or Gaussian jumps. |
| M3.2 Gaussian price jumps | A separately identified compound-Poisson extension and fixtures; measured acceleration where useful | Verify counts, jump moments, compensation and zero-jump limits against the accepted reference. Recheck continuation after jumps are added. |
| M3.3 Joint returns inference | Versioned task, same-model inference reference, pilot bank/checkpoint and conditional forecast report using M2.3 | Specify and validate the reference procedure before training at scale. Accept joint parameter/state and predictive comparisons, or retain the kernels while recording inference as pending. This gate unlocks the returns route to M8. |
| M3.F Optional collaborator reproduction | Adapter and compatibility report for the exact supplied equations, preprocessing, checkpoint and saved cases | Reproduce the saved case before substitutions. Missing artifacts leave this gate pending and prevent a parity claim; they do not block native kernels, M3.3 or M4. |

Implement M3.1–M3.3 in order; M3.F can proceed independently when artifacts arrive. Likely destinations are the model kernels in `src/sbi4finance/simulators/`, task definitions in `tasks/`, and forecast/inference adapters in the existing inference boundary. Create only the modules these outputs require. The collaborator adapter does not silently replace the native model or its reference.

## Didactic review deliverable

Increment 4 backfill is pending in the proposed `notebooks/04_heston_svj_kernels.py`: explain hidden variance, leverage, compound-Poisson jumps and the numerical continuation state, with a seeded path/continuation demonstration and measured refinement/benchmark evidence. Exercises should distinguish signed auxiliary state from displayed variance and predict a zero-intensity limit.

Increment 5's immediate deliverable is `notebooks/05_heston_joint_inference.py`, following the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone). Explain the actual reduced task and joint posterior mathematically; identify fixed context, observations, targets and units. Run bounded checkpoint conditioning and matched-state continuation, and show neural/reference marginals, dependence, future losses and aggregate gates from linked artifacts. Include the provisional checkpoint's failed conditional tests and development history: this is a reviewable inference experiment, with M3.3 scientific acceptance still pending. Exercises should explain why marginal SBC cannot settle a conditional discrepancy and why independent marginal parameter/state draws change a forecast. If M3.F is added, give its exact-artifact parity evidence a separately labelled section.

Verify each notebook by offline headless execution and HTML export, checking the displayed gate status against the reports. Use cached reference inference; default execution must not train or launch PMCMC/grid studies. A passing notebook execution does not clear failed posterior gates or substitute for the kernel backfill.

## Deliverables

Heston and Gaussian-price-jump capabilities, model-specific continuation, task configurations, difficult-region fixtures, a pilot posterior/forecast report, verified kernel/inference review notebooks, and a collaborator compatibility report. The report may state that the bundle is pending; common model work does not depend on its arrival.

## Acceptance and tests

Check variance moments, leverage/cross-covariances, jump count and size moments, compensation, zero-jump and deterministic-variance limits, and timestep sensitivity. Test continuation against uninterrupted simulations using matched prescribed random inputs. Evaluate backend agreement statistically and against declared numerical tolerances. Where references exist, assess parameter–state posterior dependence and conditional discrepancies separately from marginal SBC.

## Decision gate and deferred scope

Accept native Heston and the collaborator adapter independently. S requires no real return window. An optional M window normally conditions frozen weights without observed parameter/state truth; E additionally requires held-out later outcomes and a chronological evaluation design. Do not claim Francesco parity without artifacts, or impose the Feller condition merely to conceal a solver limitation. Defer joint price/variance jumps, automatic model selection, general-purpose filtering infrastructure, and strong risk-benefit claims until supported by matched experiments.
