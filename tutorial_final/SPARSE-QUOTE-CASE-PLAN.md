# Calibration from sparse option quotes: deferred follow-up

**Status: separate downstream unit, specification first.** Complete and package the full-surface Heston and six-target joint rough Bergomi tutorial before starting this work, unless explicitly reprioritized. Sparse adapters, checkpoints, mask training, market-case selection and sparse coverage studies are not core release requirements. This document records questions and a candidate starting experiment, not an implemented missing-data support contract.

The core uses all 63 declared contract prices with valid noise/design metadata and explicitly rejects incomplete or unsupported inputs. Keep complete clean-price banks and their provenance so compatible data can be reused later. Existing thin-chain code, EWG observations and seven-level results remain historical evidence; they do not implement sparse Heston or the new six-target rough task.

## Specify the problem before implementing it

The follow-up should first agree a short specification covering these decisions:

| Decision | What must be explicit |
| --- | --- |
| Learning question | Which missing information affects which parameter, curve quantity or downstream product? Quote count alone does not define thinness. |
| Supported observations | One fixed subset, a declared family of masks, or variable-length raw quotes? Declare strikes, maturities, option types, valid noise metadata and behavior for unsupported inputs. |
| Observation provenance | Actual quotes versus interpolated grid values; coverage flags, forward/discount/exercise conventions and dependence introduced by preprocessing. Never count extrapolated values as newly observed evidence. |
| Missingness and noise | Fixed missingness by design versus liquidity/selection-dependent missingness; which mechanism is simulated or conditioned on, and what reliability/selection claims are justified. Vary missingness and noise separately. |
| Training and input contract | Separate fixed-layout checkpoints versus one estimator trained across masks/designs. Declare coordinate/noise encoding, missing/padded representation and the training support; an encoder choice alone is not a support guarantee. |
| Matched comparisons | The same available quotes, surviving quote errors, prior, numerical model and inferred targets for neural inference, MCMC and audits. A forward emulator may predict a full surface, but the likelihood uses only the observed subset and its relevant covariance. |
| Acceptance and cost | Conditional reference agreement, calibration, invariance checks where relevant, prior/curve sensitivity and downstream product error; separate development from fresh acceptance evidence and declare a bounded compute budget. |

Absent quotes must contribute no artificial observation or likelihood term. If zero or another placeholder is used internally, a validated mask must exclude its value; it is not a zero-price observation. Any interpolation remains a declared preprocessing approximation rather than a missing-data solution. If missingness depends on prices or spreads, random masking alone does not demonstrate robustness to that selection process.

## Candidate first experiment, after specification

A small controlled full-versus-sparse comparison on the supported grid is the simplest starting point. Begin with Heston, where refined Fourier-likelihood MCMC can check the neural answer. One predeclared subset missing later maturities is a candidate; narrow wings or dispersed missing quotes are alternatives, not mandatory extra layouts. A fixed-layout adapter/checkpoint may be sufficient if that is the agreed support contract. Do not assume the current rough-only thin trainer already provides Heston support.

Use paired synthetic base surfaces, preserving the generating parameters and quote errors for surviving contracts. Refine complete clean prices under the original numerical checks, then select prices, noise and covariance consistently. Subsetting must not bypass pricing-precision checks. Reusing compatible prices avoids new generation, but does not make a full-surface checkpoint support sparse inputs.

Extend to rough Bergomi only after that control is useful. Match the full task's joint prior, numerical model and unknowns; with the current baseline these are H, eta, rho, short variance, long variance and decay. Both full and sparse comparisons infer the curve. Do not supply a curve reconstructed from additional, withheld quotes. If the core promotes another curve family, the sparse specification must follow that actual task rather than silently keep six targets. Old seven-level curves/prices cannot be projected or relabelled as the new task without repricing.

Judge what the data identify through posterior contraction, conditional reference agreement, calibration and product uncertainty. Weakly quoted curve regions may be constrained mainly by the compact curve family and prior. Wider intervals alone do not prove better inference; narrow intervals can reflect restrictive shape/extrapolation assumptions. Select one affected downstream product and one well-supported control before inspecting final results.

For a mask-conditioned or variable-length approach, train over the declared support and verify that changing masked values or adding padding leaves inference unchanged. Set encoders must also preserve permutation invariance of complete quote records. Arbitrary new strikes/maturities need corresponding simulation and validation designs. These are follow-up requirements only if that representation is chosen, not work to add to the core now.

## Market illustration and historical evidence

Choose a market case only after the controlled design and claim are fixed. EWG is a historical candidate, not the task definition or a required final ticker. Select for relevant coverage, reproducible dated quotes, coherent preprocessing/noise conventions, feasible reference/audit checks and a clear product story, before inspecting method rankings. A synthetic experiment is sufficient if no suitable market case is available.

Preserve EWG's original observations, seven-level outcomes, source labels, 27-point proposal layout and distinct 61-quote audit. A new curve family, observation design or ticker requires reviewing the checkpoint, audit, product horizons and continuation assumptions, not just swapping a layout file. Existing flat/widened-last-increment studies do not transfer unchanged to the compact curve prior.

A real market chain has no known parameter truth. EWG-posterior-conditioned experiments establish local sensitivity, not broad market performance. Population reserve-coverage and wrong-extrapolation studies require their own declared truths, continuation laws, products and acceptance criteria; do not inherit them as mandatory work. Keep successful prior-generated cases and misspecified alternatives together when making robustness claims.

The deliverable of the first downstream step is an agreed support contract and bounded experiment, followed by its implementation and evidence. No sparse jobs are requested by this plan. See the [central experiment plan](NUMERICAL-EXPERIMENT-PLAN.md) and [core protocol](EXPERIMENT-PROTOCOL.md).
