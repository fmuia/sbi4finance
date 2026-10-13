# M4 HKDE and risk-neutral option observations

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** accepted [M3.1](03-heston-and-svj.md) Heston kernels and the local bank/loader pipeline in [M2.3](02-banks-and-cluster.md). Gaussian-jump returns inference and collaborator reproduction are not prerequisites. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

**Implementation update:** [PR #12](https://github.com/fmuia/sbi4finance/pull/12)
delivers M4.1–M4.2: HKDE law, Fourier pricing, shared fixed-panel bank and offline
notebook06. The [reference mathematics/results](../../hkde-reference.md),
[bank workflow](../../option-banks.md) and [measured bank](../../results/hkde-option-bank.json)
record bounded M4.3 checks in the declared European pilot/fixture domain. Actual option-bank cluster execution is now measured: Oscar job
**7297096** generated the 24-panel HKDE bank in **4 s** scheduler elapsed with
one CPU and 4 GB, and passed same-Linux selected-ID reproduction. The shared
[cluster report](../../option-cluster-smoke.md)
records this run, verified transferred-file hashes and the passed Mac numerical
comparison alongside the Bergomi jobs. It preserves the initial HKDE portable
tolerance failure and its adjustment separately from scientific pricing gates.
Broader numerical domains and American/exotic targets remain outside this
acceptance. Q posterior evidence is tracked separately in M5. Its delivered
baseline uses the existing `theta`/`jump_intensity` task and its deterministic Fourier
labels for [M5.1](05-amortized-pricing.md#next-increment-restricted-hkde-baseline);
it does not wait for Bergomi continuum-bias acceptance.
Flat modules implement the intended responsibilities without a package framework.

Add the double-exponential jump sibling of Gaussian-jump Heston and establish an options-calibration task. This tests the architecture's ability to generate pricing-based observations without conflating Q calibration with historical-return inference.

## Tasks

1. Implement the double-exponential log-jump law with explicit upward probability, upward/downward rates, intensity, units, and compensation. Share only validated Heston components. Register a descriptive model identifier rather than treating HKDE as an unspecified higher-complexity SVJ.
2. Enforce the relevant exponential-moment support: with nonzero upward mass, the upward rate must exceed one for the usual risk-neutral compensator. Document degenerate limits rather than letting invalid moments enter a pricer.
3. Integrate a validated Fourier/PROJ pricing adapter with declared grid/tolerance settings and an independent reference. Keep path simulation for cross-checks and later payoffs; European training quotes should use the suitable validated pricing route.
4. Specify a fixed initial panel of contracts, forwards/discount factors, curve conventions, quote units, and missingness semantics; all may be hypothetical generated context for S. Generate one complete panel per inference example without requiring a market snapshot. Define the quote-error law explicitly, including its parameterization if noise is inferred.
5. Generate a small Q-parameter bank with clean prices, observation outputs, and pricing diagnostics. Distinguish intentional quote noise, numerical integration error, MC uncertainty, and discretization bias. Separate latent labels from observed conditioning.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M4.1 HKDE law and reference pricing | Double-exponential jump implementation, supported-domain checks and Fourier/PROJ adapter with independent comparison fixtures | Verify moments, compensation, limits, price identities and representative reference comparisons before accepting a reusable bank. Small provisional panels may help develop these checks. |
| M4.2 Option task and pilot bank | Shared fixed-panel observation helper, explicit context/noise configuration, grouped pilot bank and diagnostics | Check the M0 observation contract, panel ordering, truth separation and M2.3 loading. Generate prices only within the supported numerical domain from M4.1. |
| M4.3 Numerical acceptance | Accuracy/refinement report covering difficult prior regions, quote-noise scales and correlated MC cross-checks | Accept the panel generator and its error budget before large-bank production or M5 training. Record unsupported regions explicitly. |

Likely destinations are `simulators/` for the jump variant, `pricing/` for reference adapters, `observations/` for panel construction and `tasks/` for the calibration configuration, all under `src/sbi4finance/`. The panel helper follows M0's contract: M4.2 implements it if absent and otherwise reuses the accepted helper from M6.3. It must not depend on an HKDE-specific kernel, so Bergomi can implement and use the same interface independently.

## Didactic review deliverable

Planned with M4: begin `notebooks/06_hkde_and_pricing.py` with the Q double-exponential jump law, compensator/support and discounted payoff expectation. Follow the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone). Construct a small synthetic contract panel using the implemented pricer; distinguish its clean expectation, intentional quote noise and numerical error, with units and hypothetical curve context. Link the task, bank and independent accuracy evidence, including failed regions. Exercises should explain why one realized payoff is not a price label and why these Q parameters do not forecast P returns.

Accept the M4 sections after offline headless execution/export shows the seeded panel and declared accuracy budget. Use a bounded pricing example and cached costly comparisons; no market acquisition or full-bank generation runs by default.

## Deliverables

HKDE jump and pricing adapters, a reviewed synthetic option-panel configuration, small bank/reference fixtures, a pricing-accuracy report and verified M4 review sections. Market conditioning and empirical excluded-quote checks remain optional M/E additions; observed quotes do not supply parameter truth. Preserve the existing HKDE data-needs document as the source for later market-data integration.

## Acceptance and tests

Check jump moments, compensation, zero-intensity limits, put–call parity, price bounds, parameter-domain validity, and pricing-grid refinement. Compare representative and difficult prior-region prices with independent references using tolerances defined before the comparison. For any MC check, include sampling uncertainty and correlated errors across contracts. Confirm the bank retains contract ordering/context and that quote-noise augmentation cannot cross splits.

## Decision gate and deferred scope

Accept the panel task only when numerical error is controlled relative to its stated quote model. An earlier P posterior cannot substitute for this Q calibration, and a Q fit does not supply a P risk forecast or dynamics for tomorrow's option chain. Synthetic calibration can also feed American/exotic model pricing without observed target quotes if a validated target pricer exists. Their implementation remains deferred, alongside joint P/Q estimation, historical option purchases, and generalized variable-contract encoders, unless the next reviewed use case needs them.
