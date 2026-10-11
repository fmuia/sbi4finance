# M1 GBM with analytic inference and pricing references

**Status:** proposed; implementation starts after review. **Depends on:** [M0](00-contracts.md). Read [foundations](../notes/01-foundations.md), [models and numerics](../notes/02-models-and-numerics.md), and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

Build the smallest complete scientific example whose simulation, likelihood, posterior, and European prices can be checked independently. All histories, futures, rates, contracts, and other required context can be generated for S. GBM is an infrastructure acceptance case, not an assertion that it describes market returns adequately.

## Tasks

1. Implement an inspectable float64 NumPy reference using exact log-price increments on the declared observation grid. Expose physical parameter values, initial state, and explicit random streams. Keep prior sampling in the task layer.
2. Define the first returns task with known constant volatility and an unknown proportional price drift with a Gaussian prior, matching the exact posterior derived in the foundations note. State the mapping between proportional drift and log-return drift. Start with a fixed observation design.
3. Produce analytic posterior and posterior-predictive references for complete return windows. Specify the quantile/loss convention before comparing risk outputs. A later joint drift/variance extension needs its own reviewed prior and analytic derivation; it is not required to accept this baseline.
4. Add risk-neutral European pricing and deterministic integrated-variance checks using the same declared units, calendar, rates, and dividend conventions. Keep the P returns experiment and Q pricing experiment separately identified.
5. Create a small collection of seeded examples covering ordinary settings, short windows, small variances, and permitted boundaries. Use these examples to exercise the proposed observation and target schemas.

## Ordered implementation subplan

All steps below are planned, not implemented. Work in order; each step uses the preceding step's output. Destinations are guidance for the existing package layout, not fixed API signatures.

| Step | Concrete output | Verification/handoff |
|---|---|---|
| M1.1 — Exact simulator and returns experiment | Exact float64 GBM log-increment simulation in `simulators/`, return construction in `observations/`, and the known-volatility/unknown-drift task in `tasks/` with a versioned configuration. | Verify increment moments, independence, shapes, explicit RNG use and the proportional-drift/log-drift mapping on fully generated context. |
| M1.2 — Analytic P and Q references | Analytic likelihood, Gaussian posterior and predictive utilities in `inference/`; European price and integrated-variance references in `pricing/`. Keep the P return task and Q pricing task separately identified. | Check formulas independently, reproduce posterior updates, and compare simulation moments/prices against analytic quantities using declared statistical tolerances and loss conventions. |
| M1.3 — Reproducible fixtures and baseline | Compact synthetic fixtures with metadata, a runnable local demonstration in `scripts/`, targeted validation checks and a NumPy baseline benchmark. | Reproduce the full simulation-to-prediction demonstration without market files, a neural checkpoint or cluster access. Hand accepted fixtures and reference results to M2 and later model branches; record whether acceleration is justified. |

## Deliverables

GBM reference kernels, a returns-task configuration, an analytic Bayesian reference utility, European price references, and compact reproducible synthetic fixtures with origin and truth-label metadata. Provide a small script demonstrating simulation → observations → analytic posterior → conditional prediction; it must not depend on a neural checkpoint, market file, provider, or cluster access.

## Acceptance and tests

Check exact increment means/covariances and independence, analytic likelihood values, posterior updates, and parameter-transform identities. Compare simulated prices and moments with analytic expectations using uncertainty-aware tolerances. Test array shapes including singleton axes and explicit invalid-input errors. Validate missingness/time handling for any supported case; do not imply unsupported irregular designs already work.

## Decision gate and deferred scope

Accept the reference only after independent formula checks and fixture reproduction. It then unlocks M2 and the model branches. Numba is optional here: first record a baseline benchmark, and accelerate only if worthwhile. Defer jump models, latent-state filtering, learned pricing, broad market-data preparation, and claims about real-world risk performance.
