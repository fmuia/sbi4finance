# M5 Parameter posteriors and amortized pricing

**Status:** proposed; implementation starts after review. **Depends on:** [M4](04-hkde-and-options.md) and [M2](02-banks-and-cluster.md). Read [inference and pricing](../notes/03-inference-and-pricing.md) and [software and data](../notes/04-software-and-data.md).

## Purpose

Establish the relationship between Bayesian calibration and fast downstream prices. Build parameter-posterior inference first, then a separately assessed supervised pricing surrogate; study direct inference of posterior prices only if a concrete use case justifies it.

## Tasks

1. Train a posterior approximator for the reviewed option-panel task. Record prior, observed context, parameter transformations, initialization targets, bank identity, architecture, and training-only normalization.
2. Validate against the explicit quote likelihood and trusted reference posterior calculations on fixed cases. Push matched parameter/state draws through the validated pricer to obtain distributions of downstream prices. These are the baseline uncertainty outputs.
3. Define a supervised price-surrogate task with inputs consisting of Q parameters, contract terms, and required market/state context, and labels from accepted reference prices. Set the supported domain and loss/error scales deliberately; tiny prices and tail contracts require more than a single average relative-error score.
4. Train and test the surrogate using grouped splits that keep related contracts/parameter draws from leaking across partitions. Assess posterior price propagation with both exact and surrogate pricers. Keep the calibration observation generator on the validated reference route unless a separate approximation study permits replacement.
5. Optionally design a direct conditional posterior-price model for a fixed target payoff and context. Its labels and validation must reproduce the relevant push-forward distribution, rather than confusing a deterministic conditional price with a future realized payoff.

## Deliverables

A validated parameter-posterior workflow, reference posterior price distributions, a supervised pricing prototype and domain/error report, and a short decision on whether direct posterior-price inference adds value. Keep model-posterior and surrogate checkpoints distinct.

## Acceptance and tests

Evaluate conditional posterior agreement and risk/price functionals alongside SBC. Compare surrogate bias, tail errors, bounds/parity where applicable, and resulting posterior-price quantiles with reference calculations. Measure total inference and pricing cost, including preprocessing, amortization cost, and hardware. A faster evaluation is acceptable only at an explicitly reviewed accuracy level.

## Decision gate and deferred scope

First accept the parameter-posterior path; then accept or reject the surrogate independently. Direct posterior-price inference requires its own benefit and validation gate and does not block either path. Defer claims of exactness from unverified importance correction, arbitrary-payoff generalization, and exotic products lacking independently validated labels.
