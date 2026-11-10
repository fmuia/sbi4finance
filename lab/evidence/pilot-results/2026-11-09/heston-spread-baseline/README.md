# First one-scale Heston implementation probe

**Status: initial engineering checks passed; no posterior acceptance.** Implements the first increment of planning PR [#82](https://github.com/fmuia/sbi4finance/pull/82). The companion plans are [#83](https://github.com/fmuia/sbi4finance/pull/83) for flexible inputs and [#84](https://github.com/fmuia/sbi4finance/pull/84) for the gated Bergomi refresh.

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
