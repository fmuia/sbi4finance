# First one-scale Heston implementation probe

**Status: initial engineering checks passed; no posterior acceptance.** Implements the first increment of planning PR [#82](https://github.com/fmuia/sbi4finance/pull/82). The companion plans are [#83](https://github.com/fmuia/sbi4finance/pull/83) for flexible inputs and [#84](https://github.com/fmuia/sbi4finance/pull/84) for the gated Bergomi refresh.

## Fixed-c development continuation

The first c=1 network completed 10,000 finite updates in 163.04 seconds, using the declared 20,000 physical-row prefix. Its 2k and 10k checkpoints reload with identical explicitly seeded posterior draws. All 20,000 rows were seen; 1,273,952 presentations include the short final batches of each pass. The reserved test arrays remain unopened. Compact records are under [training-fixed-c1](training-fixed-c1/); full checkpoints remain outside Git under their original payload hashes.

The first completed conditional comparisons **fail**. On the broad synthetic case, the 10k physical-parameter central-90% intervals are about 4.8-10.0 times reference widths; several means remain outside the declared tolerance. Prior-predictive 90% development coverage is approximately 97.1%-99.8%, indicating over-wide intervals. The actual-market query also fails reference agreement; market-model adequacy is a separate issue. The provisional evaluation files retain incomplete references as inconclusive. Lower validation loss does not accept this checkpoint or permit grid/Bergomi production.

The broad and actual-market references pass parameter/product convergence and pointwise pricing checks. The market-like and boundary references were launched under the original cap and remain separate recorded runs until completed. The [local likelihood curvature diagnostic](diagnostics/likelihood-curvature.json) agrees with broad-case MCMC SDs to roughly 2%, with stable finite differences, and verifies the normalized Gaussian scale exactly; it is a local sanity check rather than a posterior replacement.

A [same-state dropout diagnostic](diagnostics/dropout-stateless-update-010000.json) measures a 1.935-nat objective penalty from the inherited dropout .05, while the normalization-stage change is only about .00024 nats. Its exact producer is archived alongside it; reproduce against the frozen trainer source identified in that record (Git commit d7af9c1), not a later changed consumer. No model/optimizer/standardizer/RNG state changed during this diagnostic. The result supports, but does not prove the explanation for, a regularization effect.

The [declared control protocol](dropout-control-protocol.json) allows one matched 2k/10k fit with effective dropout zero. It replays the original initialization before changing only rates, checks unchanged state arrays and original initial probes, and retains the same data, noise, row order and schedule. No noise inflation, prior change, inferred-c fit or geometry change is part of that control. Its results must be checked before another training intervention.

Parallel [raw grid coverage evidence](grid-coverage/README.md) finds the literal E candidate has 62/63 supported quote targets and no one-year PM expiry. An explicit final-call +1.5 declaration supports all selected locations but selects the identical physical contracts and still misses the original +2/one-year goals. This is data-only preparation, not a priced or accepted grid revision.

## What changed

The new `HestonSpreadPilot` uses one independent Gaussian price SD `c * half_spread`. The initial target fixes c=1 and infers five physical parameters. A separate target appends c with the declared illustration prior Uniform(.5,2); c is not a conditioning feature for that target. This increment implements likelihoods, deterministic seeded simulation, immutable task validation and a normalized one-dimensional conditional-c oracle. It does not yet implement or accept a neural training run.

Checked Fourier pricing is reused from the earlier `HestonPilot`, ported unchanged from historical study commit `78cf9de`. The executing financial model remains in the participant runtime. The new spread-only initializer needs no observed IV or vega and accepts arbitrary finite Gaussian observations, including negative values. Numerical precision remains gated against `.5 * half_spread`, independently of inferred c.

## Numerical and density probe

[Run and source bindings](first-spread-probe/run.json), [fixed observations](first-spread-probe/cases.json) and [results](first-spread-probe/summary.json) bind the exact historical 54-contract design, clean-bank manifest/payload, prior, quadrature controls and new implementation hashes. Only validation rows 0,1,2 were accessed. No training or reserved test arrays were opened. The entire compressed payload was hashed as bytes for integrity; hashing does not load its reserved arrays.

| Check | Observed result |
| --- | --- |
| Preserved versus current checked clean prices | Bitwise-equal values on all 54 contracts in each of three validation rows |
| Doubled quadrature price shift | Maximum approximately 2.45e-6 of the strictest measurement SD `.5*h`; gate .02 |
| Refined synthetic log-likelihood shift | Maximum absolute change approximately 2.63e-6 |
| Fixed-c likelihood versus joint likelihood evaluated at c=1 | Identical in all three cases |
| Conditional-c density normalization on a fine grid | Maximum absolute integral error approximately 1.74e-11 |
| Reserved test access | None |

Conditional modes at the known physical truths are approximately .759,1.061,1.160 despite generating c=1; these are finite-sample conditional checks, not joint-posterior estimates or evidence of neural scale recovery. The same generated prices are preserved for future fixed-c sensitivity and joint-target comparisons.

This is a sampled label-reuse check. It does not establish a new grid, all-prior numerical stability, MCMC convergence, SBC, market fit, neural accuracy or accepted artifact delivery. The earlier tau-network failures remain historical evidence under their original identities.

## Software validation and next increment

The existing main lab suite plus the ported pricing and new likelihood tests exercised 599 tests. One environment-induced runtime-guard failure was traced to an overly strong local source-path overlay, corrected without changing product code or the guard. All 64 tests in the affected import/CLI modules and the added pricing/likelihood/probe modules then passed, including the guard's deliberate wrong-checkout rejection. The probe adds six tests covering source/bank mismatches, preserved observations, reserved-array isolation, precision failures and nonfinite failure records. Manifest, claims and runtime-boundary release checks pass; tutorial assets are unchanged.

Next: bind the predeclared synthetic/reference cases, add first-nonfinite training-state capture, and run the bounded c=1 training/reference control. A new c=1 checkpoint is not presumed to support other scales. Joint-c training follows its fixed-c control; new-grid labels and Bergomi production remain behind their respective gates. No expensive training or MCMC job was launched by this increment.
