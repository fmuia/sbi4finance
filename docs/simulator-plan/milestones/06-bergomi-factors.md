# M6 One-factor Bergomi, with a two-factor extension

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md) for kernels; the local bank/loader pipeline in [M2.3](02-banks-and-cluster.md) for the pilot bank. This branch is independent of HKDE and M5. Read [models and numerics](../notes/02-models-and-numerics.md).

## Purpose

**Implementation update:** [PR #13](https://github.com/fmuia/sbi4finance/pull/13)
delivers M6.1–M6.3 infrastructure: curve contract, exact OU/asset covariance,
complete continuation, bounded conditional-put prices, shared bank and offline
notebook08. Reference/limit tests and bounded numerical screens pass. The actual
bank-prior grid screen has maximum difference 0.0309 currency with combined SE
0.0190; it does not prove bias below 0.025. Settle the final label fidelity
before scaling; no posterior, optional two-factor model or P forecast is accepted.
The numerical building blocks can support M7 while these limits remain visible.

**Coupled-grid follow-up:** the [frozen paired study](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/bergomi-coupled-refinement.md) resolves all one-factor comparisons over the five declared cases. The largest 128-to-1,024-step difference plus four paired SEs is 0.00827 currency, below 0.025; the largest 512-to-1,024 envelope is 0.00143. These are finite-grid checks, not a uniform continuum-bias certificate. The actual cluster route is delivered; a one-factor posterior and optional two-factor model remain separate.

Introduce lognormal forward variance with a finite-dimensional continuation state before rough-memory models. One factor is the required core; two factors are an optional scientific extension, not a prerequisite for rough Bergomi.

## Tasks

1. Specify an exponential-kernel one-factor Bergomi model, including the factor normalization, mean-reversion convention, asset correlation, and initial forward-variance curve. Classify the curve as known context or a low-dimensional inferred object; do not conceal curve assumptions inside the simulator. S may use generated admissible curves and pricing context; no observed market curve is required.
2. Implement a Gaussian/OU reference with exact factor-transition distributions where available. Couple factor and asset innovations using the appropriate joint covariance. Exact factor stepping does not make discretized asset integration or option prices exact.
3. Implement variance normalization and Q asset simulation with declared integration settings. Define conditional continuation using all factor states, the curve specification, and elapsed time; do not restart an unconditional curve at the forecast origin.
4. Add a fixed European option-panel task using the accepted observation/bank interfaces. Use variance reduction and numerical refinement before low-level acceleration. Justify the estimator's moments and error assessment; finite prices alone do not justify conventional MC error bars. Preserve sampling error separately from quote error.
5. If the reviewed use case needs separate fast and slow factors, extend to a two-factor model with a valid full correlation matrix, explicit loading conventions, and complete state output. Treat this as a scientific model, distinct from fitting numerical factors to a rough kernel.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M6.1 Model and curve contract | Reviewed factor normalization, covariance/support conventions, synthetic curve configuration and continuation-state specification | Resolve known versus inferred curve inputs and all elapsed-time conventions before implementing kernels. |
| M6.2 One-factor kernels and prices | Gaussian/OU reference, asset integration, complete continuation and MC European pricing fixtures | Pass covariance, normalization, continuation, martingale-domain and price-refinement gates with justified MC uncertainty. |
| M6.3 Option task and bank | Shared fixed-panel observation helper, task configuration and a verified small bank using M2.3 | Check context/contract ordering, quote versus numerical error, split groups and loading. Accepted M6.1–M6.3 unlock M7. |
| M6.4 Optional two-factor extension | Separate model configuration, valid joint correlation construction, kernels and acceptance report | Repeat the relevant one-factor checks and verify limiting cases. This step does not block M7. |

Implement the required one-factor steps in order. Use `simulators/`, `pricing/`, `observations/` and `tasks/` under `src/sbi4finance/` as they acquire implementations. M6.3 can implement the panel helper against M0's schema if M4.2 has not run, or reuse it if already accepted; neither the helper nor its task contract requires HKDE. A one-factor inference teaching release would additionally need an accepted posterior/reference workflow; the bank alone does not provide that inference gate.

## Didactic review deliverable

The one-factor sections are delivered in `notebooks/08_bergomi_models.py`: forward-variance curve, normalized factor model, complete continuation state and bounded European pricing. They follow the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone), with generated context, units, source links and measured evidence. The [coupled-grid follow-up](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/bergomi-coupled-refinement.md) adds a live pairing example and cached refinement results. Exercises explain why exact OU transitions do not make asset integration exact and why restarting the original curve loses conditioning. Add two-factor material only if M6.4 is implemented.

Accept the sections after offline headless execution/export displays the small calculation and uncertainty-aware references without market curves or expensive default MC. Kernel/bank evidence alone must not be presented as a validated posterior.

## Deliverables

One-factor kernels, continuation support, curve/task specifications, Gaussian reference fixtures, a small option bank with price-error diagnostics and verified M6 review sections. The optional two-factor extension receives a separate configuration and acceptance report.

## Acceptance and tests

Check factor transition moments and cross-covariances, positive-semidefinite correlation constraints, forward-variance expectation normalization, zero-volatility-of-variance limits, and curve-dependent integrated-variance checks. Compare conditional continuation with uninterrupted paths under matched random inputs. Test price convergence and martingale expectations only within domains where the specified asset process should be a true martingale. Evaluate MC error across the entire panel, including shared-path dependence.

## Decision gate and deferred scope

Accept one factor before adding two using synthetic fixtures independently of M/E market assets. Select numerics and any Numba kernels from measured error/cost evidence. Two-factor delays do not block M7. Q continuation/pricing checks do not establish P return-risk forecasts or future option-chain dynamics. Defer historical-return applications until P dynamics and initialization are specified, and defer rough-kernel factor approximations to a separately versioned numerical study.
