# M7 Rough Bergomi, starting with option calibration

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** accepted one-factor steps [M6.1–M6.3](06-bergomi-factors.md) and the local bank/loader pipeline in [M2.3](02-banks-and-cluster.md). Optional M6.4 two-factor Bergomi, HKDE, and M5 are not prerequisites. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

**Implementation update:** [PR #14](https://github.com/fmuia/sbi4finance/pull/14)
delivers M7.1–M7.2 numerical infrastructure, M7.3a bank and offline rough review
sections. M7.3b posterior work is explicitly deferred. The exact-bank-prior
128-to-512-step screen passes with maximum difference 0.0427 currency and combined
SE 0.0248; this does not prove bias below 0.025. Numerical pilots are distinct
from accepted inference; rough-memory P forecasting remains deferred.

**Coupled-grid follow-up:** the [frozen paired study](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/bergomi-coupled-refinement.md) leaves the 128-step comparison unresolved at H=0.1 and eta=1.2: maximum 128-to-1,024 envelope 0.03839 exceeds 0.025. All 512-to-1,024 comparisons resolve within the budget (maximum 0.02011). This supports 512 steps as a candidate for a new explicitly numerical pilot; it does not prove continuum accuracy or change existing bank identities. The bounded numerical study is complete. M7.3b now needs a declared observation/reference strategy and a modest independent training bank, followed by conditional/joint/price-sensitive checks and its didactic sections.

Add a power-law-memory forward-variance model with an independently checked driver and controlled numerical pricing. The initial inference task is Q calibration from option panels, which avoids pretending that current variance alone solves conditional forecasting from historical returns.

## Tasks

1. Specify the rough kernel, its normalization, roughness parameter, leverage construction, forward-variance curve, and supports. Identify precisely which Gaussian field is being simulated; do not interchange fractional Brownian and Volterra conventions by name.
2. Build a small-grid covariance-based Gaussian reference including dependence with the asset driver. Use it to check a production hybrid/convolution scheme over representative and difficult parameter settings.
3. Validate normalization and implement MC European pricing with explicit grids and budgets. Verify the estimator admits its reported error assessment: a finite price need not have finite payoff variance. Bounded put-payoff estimation plus put-call parity is a candidate within a verified martingale domain. Keep sampling error, contract dependence and discretization bias explicit.
4. Generate a small option-panel bank with Q parameters and any declared curve coefficients as targets. Synthetic admissible curves, contracts, rates, and quote observations suffice for S; a real surface is optional M input. Reuse bank/adapter contracts and train a pilot posterior for a restricted task. Assess time-grid/kernel refinement, conditional cases and price-sensitive posterior functionals before increasing volume; unresolved reference accuracy limits acceptance for teaching.
5. If pursuing a finite-factor approximation, version its kernel, weights, factor count, and error targets as numerical settings for the specified rough model. Compare it against the reference; do not equate it with an independently estimated two-factor Bergomi model.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M7.1 Gaussian driver reference | Explicit kernel/field convention, supported parameter domain and small-grid joint covariance fixtures | Check Gaussian covariance and asset-driver dependence independently before production optimization. |
| M7.2 Reusable simulation and pricing | Hybrid/convolution generator, variance normalization, European pricer and numerical error report | Compare with M7.1, refine grids and justify the price estimator's moments/error bars before accepting a reusable training bank or scaling it. Small explicitly provisional pilots can inform these choices. |
| M7.3a Restricted bank | Q task, grouped pilot bank and verified load/regeneration | Reuse shared panel and bank interfaces; record numerical fidelity limits before scaling. Delivered in increment 9. |
| M7.3b Restricted posterior | Pilot checkpoint and conditional/reference validation report | Deferred during simulator coverage work. Match the declared observation law and assess joint/price-sensitive outputs. Accepted inference is required for M8. |
| M7.4 Optional numerical factor approximation | Versioned approximate kernel/weights and accuracy/cost comparison | Compare with the accepted rough reference at declared error targets. Do not present this approximation as the independently specified M6.4 model. |

Complete M7.1–M7.3b in order. M7.4 is optional and may follow the accepted numerical reference without delaying the reference-based posterior pilot. Likely destinations are `simulators/` and `pricing/` for numerical work, and `tasks/` and `inference/` for the restricted calibration experiment. These reuse the shared observation interface without creating a hidden dependency on M4 or M5.

## Didactic review deliverable

The rough-driver and numerical-pricing sections are delivered in `notebooks/08_bergomi_models.py`, following the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone). They explain the Volterra kernel, normalization and memory requirement with a small Gaussian-reference example and cached numerical evidence. The [coupled-grid follow-up](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/bergomi-coupled-refinement.md) adds a live pairing example and measured comparisons. Restricted-posterior sections remain pending with M7.3b; numerical review does not supply conditional-inference evidence. Exercises distinguish a finite-factor approximation from a two-factor model and explain why current variance alone cannot initialize a P rough-memory forecast.

Accept the sections after offline headless execution/export reproduces the bounded reference demonstration and renders the separate numerical/posterior gates. Large covariance studies, pricing banks and training remain opt-in; a failed inference gate stays visible even if simulation and notebook execution pass.

## Deliverables

Gaussian-driver fixtures, a checked production simulator/pricer, one Q calibration configuration, a pilot bank/posterior, an accuracy/cost report covering roughness and leverage regimes, and verified M7 review sections. Separate simulator acceptance from posterior acceptance; the latter includes a reproducible validation procedure and its reference limitations.

## Acceptance and tests

Check Gaussian covariances, asset-driver dependence, variance expectations, deterministic-volatility limits, pricing identities and refinement stability. Require a justified estimator/error model before reporting MC precision; see the [moment discussion](../notes/03-inference-and-pricing.md). Set tolerances before full-bank production. Inspect difficult prior regions and conditional inference cases, not only average error or marginal SBC. Compare numerical results against scientific budgets rather than seeded trajectory equality alone.

## Decision gate and deferred scope

Accept a production bank only after the driver and pricing gates pass, independently of market acquisition. If posterior reference accuracy is unresolved, accept numerical infrastructure separately and defer an inference teaching claim. Q calibration is neither a P forecast nor a future-chain forecast. A P forecast requires declared P dynamics and inferred or conditionally reconstructed memory; true simulated memory labels do not solve that inference problem. Defer such forecasts, joint P/Q mapping, rough Heston and arbitrary path-dependent products until separately validated.
