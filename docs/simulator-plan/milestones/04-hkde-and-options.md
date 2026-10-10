# M4 HKDE and risk-neutral option observations

**Status:** proposed; implementation starts after review. **Depends on:** [M3](03-heston-and-svj.md) for the shared variance/jump boundary and [M2](02-banks-and-cluster.md) for banks. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

Add the double-exponential jump sibling of Gaussian-jump Heston and establish an options-calibration task. This tests the architecture's ability to generate pricing-based observations without conflating Q calibration with historical-return inference.

## Tasks

1. Implement the double-exponential log-jump law with explicit upward probability, upward/downward rates, intensity, units, and compensation. Share only validated Heston components. Register a descriptive model identifier rather than treating HKDE as an unspecified higher-complexity SVJ.
2. Enforce the relevant exponential-moment support: with nonzero upward mass, the upward rate must exceed one for the usual risk-neutral compensator. Document degenerate limits rather than letting invalid moments enter a pricer.
3. Integrate a validated Fourier/PROJ pricing adapter with declared grid/tolerance settings and an independent reference. Keep path simulation for cross-checks and later payoffs; European training quotes should use the suitable validated pricing route.
4. Specify a fixed initial panel of contracts, forwards/discount factors, curve conventions, quote units, and missingness semantics. Generate one complete panel per inference example. Define the quote-error law explicitly, including its parameterization if noise is inferred.
5. Generate a small Q-parameter bank with clean prices, observation outputs, and pricing diagnostics. Distinguish intentional quote noise, numerical integration error, MC uncertainty, and discretization bias. Separate latent labels from observed conditioning.

## Deliverables

HKDE jump and pricing adapters, a reviewed option-panel configuration, small bank/reference fixtures, and a pricing-accuracy report. Preserve the existing HKDE data-needs document as the source for later market-data integration.

## Acceptance and tests

Check jump moments, compensation, zero-intensity limits, put–call parity, price bounds, parameter-domain validity, and pricing-grid refinement. Compare representative and difficult prior-region prices with independent references using tolerances defined before the comparison. For any MC check, include sampling uncertainty and correlated errors across contracts. Confirm the bank retains contract ordering/context and that quote-noise augmentation cannot cross splits.

## Decision gate and deferred scope

Accept the panel task only when numerical error is controlled relative to its stated quote model. An earlier P posterior cannot substitute for this Q calibration. Defer American/exotic implementation, joint P/Q estimation, historical option purchases, and generalized variable-contract encoders unless the next reviewed use case needs them.
