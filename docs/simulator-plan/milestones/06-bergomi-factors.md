# M6 One-factor Bergomi, with a two-factor extension

**Status:** proposed; implementation starts after review. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md); use [M2](02-banks-and-cluster.md) for banks. This branch is independent of HKDE and M5. Read [models and numerics](../notes/02-models-and-numerics.md).

## Purpose

Introduce lognormal forward variance with a finite-dimensional continuation state before rough-memory models. One factor is the required core; two factors are an optional scientific extension, not a prerequisite for rough Bergomi.

## Tasks

1. Specify an exponential-kernel one-factor Bergomi model, including the factor normalization, mean-reversion convention, asset correlation, and initial forward-variance curve. Classify the curve as known context or a low-dimensional inferred object; do not conceal curve assumptions inside the simulator.
2. Implement a Gaussian/OU reference with exact factor-transition distributions where available. Couple factor and asset innovations using the appropriate joint covariance. Exact factor stepping does not make discretized asset integration or option prices exact.
3. Implement variance normalization and Q asset simulation with declared integration settings. Define conditional continuation using all factor states, the curve specification, and elapsed time; do not restart an unconditional curve at the forecast origin.
4. Add a fixed European option-panel task using the accepted observation/bank interfaces. Use variance reduction and numerical refinement before low-level acceleration. Justify the estimator's moments and error assessment; finite prices alone do not justify conventional MC error bars. Preserve sampling error separately from quote error.
5. If the reviewed use case needs separate fast and slow factors, extend to a two-factor model with a valid full correlation matrix, explicit loading conventions, and complete state output. Treat this as a scientific model, distinct from fitting numerical factors to a rough kernel.

## Deliverables

One-factor kernels, continuation support, curve/task specifications, Gaussian reference fixtures, and a small option bank with price-error diagnostics. The optional two-factor extension receives a separate configuration and acceptance report.

## Acceptance and tests

Check factor transition moments and cross-covariances, positive-semidefinite correlation constraints, forward-variance expectation normalization, zero-volatility-of-variance limits, and curve-dependent integrated-variance checks. Compare conditional continuation with uninterrupted paths under matched random inputs. Test price convergence and martingale expectations only within domains where the specified asset process should be a true martingale. Evaluate MC error across the entire panel, including shared-path dependence.

## Decision gate and deferred scope

Accept one factor before adding two. Select numerics and any Numba kernels from measured error/cost evidence. Two-factor delays do not block M7. Defer historical-return applications until P dynamics and initialization are specified, and defer rough-kernel factor approximations to a separately versioned numerical study.
