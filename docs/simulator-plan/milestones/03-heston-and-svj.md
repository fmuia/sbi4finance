# M3 Heston and a precisely specified Gaussian-jump SVJ

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md) for kernels; the accepted local bank/loader pipeline in [M2.3](02-banks-and-cluster.md) for the returns inference pilot. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

**Priority update:** M3.1–M3.2 kernels are available and M3.3's pipeline is
implemented, but conditional posterior acceptance remains pending. Further
network experiments are deferred during the independent simulator/bank coverage
push. Preserve root43022's untouched test and the [recorded analysis next steps](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/simulator-analysis-next-steps.md).

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

Increment 4 is delivered and verified in [PR #11](https://github.com/fmuia/sbi4finance/pull/11) as `notebooks/04_heston_svj_kernels.py`. It explains hidden variance, leverage, compound-Poisson jumps and complete numerical continuation state, with a seeded path/continuation calculation, measured numerical/timing evidence and review exercises.

The baseline version of increment 5's `notebooks/05_heston_joint_inference.py` was also delivered and verified in that PR. It follows the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone): reduced-task mathematics, observation/context/target units, bounded checkpoint conditioning, joint neural/reference marginals, matched-state future losses, source/artifact links and exercises. It shows both the original frozen-test failure and the [spline follow-up in PR #11](https://github.com/fmuia/sbi4finance/pull/11)'s failure on 16 validation reference histories. The follow-up had mean RMSE 0.2457 reference SD, maximum mean error 0.7921 SD, maximum relative posterior-SD error 0.461 and predictive-interval endpoint error 0.2998 SD, exceeding unchanged gates of 0.20, 0.40, 0.30 and 0.15 respectively. Root 43022's final-test split was not evaluated. **M3.3 scientific acceptance and its M8 handoff remain pending.**

Both baseline notebooks passed strict marimo checks, headless HTML export and visual inspection, including isolated execution without a `data/` directory and with external sockets blocked. The kernel run took 1.622 s; the final joint-inference offline run passed without a new wall-time measurement. Its default may compute one bounded selected-case reference (48 grid nodes, 4,096 particles) alongside cached aggregate evidence. Full refinement studies, large reference calculations, training and cluster jobs remain explicit commands. No reference-cache framework is required. These notebook checks do not clear failed posterior gates; M3.F would still require a separate exact-artifact parity section.

The subsequent [explicit-time follow-up](https://github.com/fmuia/sbi4finance/pull/11) changed only the time encoding in a matched 80-epoch spline experiment. Mean RMSE improved from 0.2457 to 0.1732 reference SD, but maximum mean error 0.4554 SD and predictive-interval endpoint error 0.2099 SD still exceed the 0.40 and 0.15 gates. Both failures occur on validation case 0; no other case exceeds an individual-error limit, and both aggregate rank checks pass. Root 43022's final test remains unevaluated. The implementation suite now passes 40 tests; notebook 05's new explanation passes strict checks, headless export and visual inspection, including a 13.09-second isolated run without market fixtures or external connections. **M3.3 acceptance remains pending.** This restricted Heston experiment makes no full-SVJ or market claim and does not block M4.

The [frozen diagnostic review](https://github.com/fmuia/sbi4finance/blob/feat/heston-joint-inference/docs/heston-diagnostics.md) adds prior/posterior, marginal SBC and conditional-error figures. The network visibly updates from each history; the prior-copy baseline has mean RMSE 3.265 reference SD versus 0.173 for the candidate. Higher-draw repeats and two finer references preserve the calm-history state-mean discrepancy (0.412–0.435 SD against the second refinement), while predictive endpoint errors straddle their engineering tolerance. These are descriptive development checks, not new SBC or a formal significance test at the tolerance boundary. The original failed gate and untouched final-test status remain unchanged. Review notebook 05 explains the distinction; M3.3 stays pending without blocking M4.

The final bounded [signed-asinh experiment](https://github.com/fmuia/sbi4finance/blob/feat/heston-joint-inference/docs/heston-state-transform.md) changed only the state-target coordinates and retained the same bank, architecture, seed and 80-epoch budget. It failed first-stage validation (maximum mean error 0.5268 SD, relative SD error 0.3212, predictive endpoint error 0.2403), despite passing marginal rank checks. The original-state v2 candidate remains the better provisional comparison. The fixed stop rule was applied: no higher-draw stage, new candidate or reserved-test evaluation. All 43 software tests pass. Notebook 05 explains the coordinate change and result; M3.3 posterior acceptance remains pending and the next implementation increment is M4.1.

## Deliverables

Heston and Gaussian-price-jump capabilities, model-specific continuation, task configurations, difficult-region fixtures, a pilot posterior/forecast report, verified kernel/inference review notebooks, and a collaborator compatibility report. The report may state that the bundle is pending; common model work does not depend on its arrival.

## Acceptance and tests

Check variance moments, leverage/cross-covariances, jump count and size moments, compensation, zero-jump and deterministic-variance limits, and timestep sensitivity. Test continuation against uninterrupted simulations using matched prescribed random inputs. Evaluate backend agreement statistically and against declared numerical tolerances. Where references exist, assess parameter–state posterior dependence and conditional discrepancies separately from marginal SBC.

## Decision gate and deferred scope

Accept native Heston and the collaborator adapter independently. S requires no real return window. An optional M window normally conditions frozen weights without observed parameter/state truth; E additionally requires held-out later outcomes and a chronological evaluation design. Do not claim Francesco parity without artifacts, or impose the Feller condition merely to conceal a solver limitation. Defer joint price/variance jumps, automatic model selection, general-purpose filtering infrastructure, and strong risk-benefit claims until supported by matched experiments.
