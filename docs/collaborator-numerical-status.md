# Tutorial numerical work: collaborator overview

Updated 28 October 2026.

**13:55 UTC update:** the known-curve **100k bank is checked and complete**. Its intermediate 50k models and validation are also complete. Two joint holdout shards timed out; exactly those ranges are now running on different nodes, preserving all completed work. See the [recovery and live records](run-recovery-2026-10-28.md) for the remaining integrity checks and downstream dependencies. Original failed receipts are retained.

We are building the data and trained models that the live finance tutorial will load. The aim is to infer plausible volatility-model parameters from option prices, then show how parameter uncertainty affects other prices. BayesFlow supplies the amortized neural inference; PyMC supplies reference calculations. The tutorial should run from prepared examples without asking participants to generate a large dataset or train a network.

The learning path has three stages. One training surface contains 63 simulated option prices.

| Lesson | Current position |
| --- | --- |
| Heston | A checked 100,000-surface training bank and a promising trained estimator. The frozen candidate passed its calibration checks on 2,000 previously unused cases, plus comparisons for simple variance products. It is ready for notebook packaging, with the limits below. |
| Rough Bergomi with a known variance curve | **Checked 100k training bank**, plus 5k validation and 7k test surfaces. The [intermediate 50k models and 2,000-case validation](pilot-results/2026-10-28/known50k-completed/README.md) completed; all four rank screens pass, with 93.85% coverage for nominal 95% product intervals. Full-100k fitting and conditional reference/independent-price checks remain ahead. |
| Rough Bergomi with an inferred variance curve | All 20k training surfaces are generated. Two missing holdout shards (50 surfaces) are running in a reviewed continuation, preserving all 1,278 successful parts. Integrity checks must release the initial 20k / 5k / 7k snapshot. The 100k append is armed but has not started; trained estimators and validation remain unfinished. |

The delivery target is **at least 100,000 unique training surfaces for each rough route**, matching Heston; learning curves may justify more. The known-curve bank has reached that target. GPU job **8046588** completed its intermediate 50k baseline; full-100k fitting follows the learning review. Existing prices, seeds and holdouts are reused. The known-curve append consumed 152.902 CPU-hours within its 200-hour phase cap; the joint append retains its 3,000-hour cap. Joint continuation and follow-on readiness use explicit completion receipts; both failed histories are preserved. Consult the recovery record for current job IDs rather than the historical overnight ones.

## Interesting posterior shapes

The main showcase is now [complete corner plots from interior examples](interior-corner-plots.md), rather than selected projections or boundary cases. Both plots are available: four Heston parameters and three known-curve rough parameters, with every pair shown. Both references' central 99% intervals are comfortably inside their priors. The new Heston reference passed convergence/numerical checks and agrees well with the amortized output. These interior posteriors are mostly regular; we do not manufacture exotic geometry.

The older examples below demonstrate mainly prior-boundary skew, not strong interior curvature or multiple modes. In particular, rough eta's upper bound of 3.5 is our chosen prior limit, not a theoretical model limit. They are retained as supplementary diagnostics. The nearly-flat joint-curve example is a natural future test of weak identification.

- [Heston: skewed marginals, Gaussian comparisons and a mildly curved ridge](pilot-results/2026-10-28/heston-posterior-geometry/heston-posterior-geometry.png).
- [Rough Bergomi: the joint posterior near the eta boundary](pilot-results/2026-10-28/rough-posterior-geometry/rough-case2-reference-corner.png).
- [Rough Bergomi: coordinate transforms and the current neural/reference mismatch](pilot-results/2026-10-28/rough-posterior-geometry/rough-eta-physical-and-logit.png).

The rough figures use an approximate pricing-emulator reference. They illustrate its geometry, not an accepted exact-model posterior.

## What still needs doing

1. Connect the checked Heston candidate, cached examples and regeneration commands to the marimo notebooks; check the participant workflow. Path-dependent product pricing and broader seed/conditional robustness remain separate checks.
2. Finish the running rough banks and inspect their automatically queued intermediate fits. Improve/check both neural inference and the forward pricing emulator; preserve reference-target limitations before claiming agreement with MCMC. Final 100k fits follow the learning review.
3. Complete and validate the joint-curve rough lesson.
4. Follow with the planned architecture/summary-network comparisons. Sparse option chains remain a separate later unit.

## What is on GitHub

The shared implementation, compact numerical evidence, these figures and the updated plans are pushed in **[draft PR 48](https://github.com/fmuia/sbi4finance/pull/48)**. These changes are **not merged into main**. Large banks, model payloads and full-panel draw archives remain on Oscar; small teaching caches are committed. GitHub records source/data identities and production commands. Notebook integration of the new backbones is not yet delivered.

The [artifact handover](full-surface-artifact-handover.md) now gives runnable commands for a small offline Heston replay cache and optional live loading of the exported final-500 model. The initial cache is a previously inspected integration example, distinct from the new interior showcase. It requires neither the full bank nor a loaded model; current notebooks still use their historical artifacts until explicitly connected.

For details, see the [latest numerical results](backbone-validation-2026-10-28.md), [current work order](tutorial-backbone-work-order.md) and [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md). Passing the Heston checks is a useful milestone, not complete scientific acceptance of every route or the finished tutorial.
