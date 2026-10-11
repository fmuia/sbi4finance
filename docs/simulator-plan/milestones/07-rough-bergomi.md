# M7 Rough Bergomi, starting with option calibration

**Status:** proposed; implementation starts after review. **Depends on:** accepted one-factor core in [M6](06-bergomi-factors.md) and [M2](02-banks-and-cluster.md). Two-factor Bergomi, HKDE, and M5 are not prerequisites. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

Add a power-law-memory forward-variance model with an independently checked driver and controlled numerical pricing. The initial inference task is Q calibration from option panels, which avoids pretending that current variance alone solves conditional forecasting from historical returns.

## Tasks

1. Specify the rough kernel, its normalization, roughness parameter, leverage construction, forward-variance curve, and supports. Identify precisely which Gaussian field is being simulated; do not interchange fractional Brownian and Volterra conventions by name.
2. Build a small-grid covariance-based Gaussian reference including dependence with the asset driver. Use it to check a production hybrid/convolution scheme over representative and difficult parameter settings.
3. Validate normalization and implement MC European pricing with explicit grids and budgets. Verify the estimator admits its reported error assessment: a finite price need not have finite payoff variance. Bounded put-payoff estimation plus put-call parity is a candidate within a verified martingale domain. Keep sampling error, contract dependence and discretization bias explicit.
4. Generate a small option-panel bank with Q parameters and any declared curve coefficients as targets. Synthetic admissible curves, contracts, rates, and quote observations suffice for S; a real surface is optional M input. Reuse bank/adapter contracts and train a pilot posterior for a restricted task. Assess time-grid/kernel refinement, conditional cases and price-sensitive posterior functionals before increasing volume; unresolved reference accuracy limits acceptance for teaching.
5. If pursuing a finite-factor approximation, version its kernel, weights, factor count, and error targets as numerical settings for the specified rough model. Compare it against the reference; do not equate it with an independently estimated two-factor Bergomi model.

## Deliverables

Gaussian-driver fixtures, a checked production simulator/pricer, one Q calibration configuration, a pilot bank/posterior, and an accuracy/cost report covering roughness and leverage regimes. Separate simulator acceptance from posterior acceptance; the latter includes a reproducible validation procedure and its reference limitations.

## Acceptance and tests

Check Gaussian covariances, asset-driver dependence, variance expectations, deterministic-volatility limits, pricing identities and refinement stability. Require a justified estimator/error model before reporting MC precision; see the [moment discussion](../notes/03-inference-and-pricing.md). Set tolerances before full-bank production. Inspect difficult prior regions and conditional inference cases, not only average error or marginal SBC. Compare numerical results against scientific budgets rather than seeded trajectory equality alone.

## Decision gate and deferred scope

Accept a production bank only after the driver and pricing gates pass, independently of market acquisition. If posterior reference accuracy is unresolved, accept numerical infrastructure separately and defer an inference teaching claim. Q calibration is neither a P forecast nor a future-chain forecast. A P forecast requires declared P dynamics and inferred or conditionally reconstructed memory; true simulated memory labels do not solve that inference problem. Defer such forecasts, joint P/Q mapping, rough Heston and arbitrary path-dependent products until separately validated.
