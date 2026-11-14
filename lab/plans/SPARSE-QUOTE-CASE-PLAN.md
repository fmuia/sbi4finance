# Calibration from sparse option quotes: Heston follow-up

**Status, 9 November 2026: explicitly reprioritized as a separate Heston track.** The adopted sequence and execution gates are in [Heston flexible inputs](HESTON-FLEXIBLE-INPUTS-PLAN.md), dependent on the separately prepared [Heston noise and grid baseline](HESTON-NOISE-GRID-PLAN.md). Merge those planning dependencies before execution. First accept that complete-input baseline; then vary actual contract geometry with coordinate-aware direct inputs, compare a learned summary on the same data, and finally introduce declared structured/random masks. Establish the fixed Gaussian-scale task first and repeat the successful final recipe with one inferred scale. Spread-profile variation remains separate. No jobs or missing-data support are implemented merely by adopting this plan.

This follow-up can proceed independently of the Bergomi notebook port after the common baseline gate; completion of every rough lesson is no longer its prerequisite. It adds no automatic Bergomi flexible-input work. The notes below preserve supporting requirements and historical evidence, not a second competing work order; the linked Heston plan controls the current sequence, budgets and acceptance criteria.

Historical core checkpoints use all 63 declared contract prices with valid noise/design metadata and explicitly reject incomplete or unsupported inputs. The revised complete-input baseline will have its own accepted design identity; it does not silently change those checkpoints. Keep complete clean-price banks and their provenance so compatible data can be reused later. Existing thin-chain code, EWG observations and seven-level results remain historical evidence; they do not implement the new Heston geometry/summary/mask contract or a new rough task.

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

## Supporting controlled experiment

A controlled full-versus-sparse comparison is the information-removal study in the adopted Heston plan, after its complete varying-geometry and summary checks. Refined Fourier-likelihood MCMC can check the neural answer. A predeclared subset missing later maturities is the primary scientific example; wing removal and random thinning are limited support checks with frozen severities, not an unrestricted mask family. Do not assume the current rough-only thin trainer already provides Heston support.

Use paired synthetic base surfaces, preserving the generating parameters and quote errors for surviving contracts. Refine complete clean prices under the original numerical checks, then select prices, noise and covariance consistently. Subsetting must not bypass pricing-precision checks. Reusing compatible prices avoids new generation, but does not make a full-surface checkpoint support sparse inputs.

A future sparse rough Bergomi extension requires a separate decision after that control is useful; it is outside the adopted Heston work. Match the then-accepted full task's joint prior, numerical model and unknowns; the historical six-target baseline uses H, eta, rho, short variance, long variance and decay. Both full and sparse comparisons must infer any quantities declared unknown. Do not supply a curve reconstructed from additional, withheld quotes. If the core promotes another curve family, the sparse specification must follow that actual task rather than silently keep six targets. Old seven-level curves/prices cannot be projected or relabelled as the new task without repricing.

Judge what the data identify through posterior contraction, conditional reference agreement, calibration and product uncertainty. Weakly quoted curve regions may be constrained mainly by the compact curve family and prior. Wider intervals alone do not prove better inference; narrow intervals can reflect restrictive shape/extrapolation assumptions. Select one affected downstream product and one well-supported control before inspecting final results.

For a mask-conditioned or variable-length approach, train over the declared support and verify that changing masked values or adding padding leaves inference unchanged. Set encoders must also preserve permutation invariance of complete quote records. Arbitrary new strikes/maturities need corresponding simulation and validation designs. These are follow-up requirements only if that representation is chosen, not work to add to the core now.

## Market illustration and historical evidence

Choose a market case only after the controlled design and claim are fixed. EWG is a historical candidate, not the task definition or a required final ticker. Select for relevant coverage, reproducible dated quotes, coherent preprocessing/noise conventions, feasible reference/audit checks and a clear product story, before inspecting method rankings. A synthetic experiment is sufficient if no suitable market case is available.

Preserve EWG's original observations, seven-level outcomes, source labels, 27-point proposal layout and distinct 61-quote audit. A new curve family, observation design or ticker requires reviewing the checkpoint, audit, product horizons and continuation assumptions, not just swapping a layout file. Existing flat/widened-last-increment studies do not transfer unchanged to the compact curve prior.

A real market chain has no known parameter truth. EWG-posterior-conditioned experiments establish local sensitivity, not broad market performance. Population reserve-coverage and wrong-extrapolation studies require their own declared truths, continuation laws, products and acceptance criteria; do not inherit them as mandatory work. Keep successful prior-generated cases and misspecified alternatives together when making robustness claims.

**A quick check, 28 and 29 October, on request.** The six-target task was run on EWG's fixed 27-point layout, on one machine, as evidence for this specification, not as its implementation (`lab/checks/why_bayes/`, Section 5; course Lecture 16, Section 16.4). The bank (24,000 surfaces, built by PR #48's bank builder) and the network stay off the repository, with their SHA-256 recorded. On 100 prior-generated chains, least squares without a prior over-reserved, $H$ fixed at 0.10 was biased, the flat continuation was biased beyond the last quote, and a Laplace approximation with the posterior's prior matched the network's posterior, which matched importance sampling on ten audited chains. It answers none of the decisions above: the layout is fixed, the truths come from the prior, and no misspecified alternative was run.

The current deliverable is the bounded Heston support contract, implementation and evidence in the [adopted follow-up](HESTON-FLEXIBLE-INPUTS-PLAN.md). Planning approval alone starts no sparse jobs. See the [central experiment plan](NUMERICAL-EXPERIMENT-PLAN.md) and [core protocol](EXPERIMENT-PROTOCOL.md) for historical core scope and release boundaries.
