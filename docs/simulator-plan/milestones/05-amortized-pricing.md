# M5 Parameter posteriors and amortized pricing

**Status:** proposed; implementation starts after review. **Depends on:** accepted [M4.3](04-hkde-and-options.md) option observations and the local bank/loader pipeline in [M2.3](02-banks-and-cluster.md). Read [inference and pricing](../notes/03-inference-and-pricing.md) and [software and data](../notes/04-software-and-data.md).

## Purpose

Establish the relationship between Bayesian calibration and fast downstream prices. Build parameter-posterior inference first, then a separately assessed supervised pricing surrogate; study direct inference of posterior prices only if a concrete use case justifies it.

## Tasks

1. Train a posterior approximator for the reviewed synthetic option-panel task, including generated curves/context. Record prior, observed context, parameter transformations, initialization targets, bank identity, architecture, and training-only normalization. Optional M applies the normally frozen checkpoint to an unlabelled market panel; it is distinct from network training.
2. Validate against the explicit quote likelihood and trusted reference posterior calculations on fixed cases. Push matched parameter/state draws through the validated pricer to obtain distributions of downstream prices. These are the baseline uncertainty outputs.
3. Define a supervised price-surrogate task with inputs consisting of Q parameters, contract terms, and required market/state context. Labels are validated pricing expectations, not individual realized payoffs or presumed real-market fair values. Set the supported domain and loss/error scales deliberately; tiny prices and tail contracts require more than a single average relative-error score.
4. Train and test the surrogate using grouped splits that keep related contracts/parameter draws from leaking across partitions. Assess posterior price propagation with both exact and surrogate pricers. Keep the calibration observation generator on the validated reference route unless a separate approximation study permits replacement.
5. Optionally design a direct conditional posterior-price model for a fixed target payoff and context. Generate each noisy conditioning panel and reference target model price from the same parameter draw. Its labels and validation must reproduce the relevant push-forward distribution, rather than confusing a conditional price with a future realized payoff.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M5.1 Parameter posterior and reference prices | Validated calibration checkpoint, same-likelihood posterior reference cases and downstream price distributions from the reference pricer | Accept conditional/joint posterior checks and price propagation. This complete options baseline can enter M8 without waiting for a learned pricer. |
| M5.2 Supervised price surrogate | Separate regression bank/checkpoint, supported-domain specification, error report and measured cost comparison | Compare price errors and propagated quantiles against M5.1. Accept or reject the surrogate independently; its failure does not invalidate the accepted reference-price path. |
| M5.3 Optional direct price posterior | Fixed-target conditional price-posterior experiment and comparison with M5.1 | Verify matched panel/target generation and push-forward agreement. Keep this experiment independent of acceptance of either earlier route. |

Complete M5.1 first. M5.2 and, if selected, M5.3 then use that accepted baseline; M5.3 need not wait for surrogate acceptance. Likely destinations are `inference/` and `adapters/` for calibration, `surrogates/` for regression, and `tasks/` for their distinct datasets. Checkpoint cards must identify which route and supported domain they implement.

## Deliverables

A validated parameter-posterior workflow, reference posterior price distributions, a supervised pricing prototype and domain/error report, and a short decision on whether direct posterior-price inference adds value. Keep model-posterior and surrogate checkpoints distinct.

## Acceptance and tests

Evaluate conditional posterior agreement and risk/price functionals alongside SBC. Compare surrogate bias, tail errors, bounds/parity where applicable, and resulting posterior-price quantiles with reference calculations. Measure total inference and pricing cost, including preprocessing, amortization cost, and hardware. A faster evaluation is acceptable only at an explicitly reviewed accuracy level.

## Decision gate and deferred scope

First accept the parameter-posterior path; then accept or reject the surrogate independently. Direct posterior-price inference has its own gate and does not block either path. All three routes support S without market files; E requires genuinely excluded same-time quotes, including exclusion from smoothing and input construction. Q price uncertainty is not a forecast of future chain movements or P losses. American/exotic S outputs need validated target pricers, not observed target quotes; their implementation remains deferred. Also defer unverified exactness claims and arbitrary-payoff generalization.
