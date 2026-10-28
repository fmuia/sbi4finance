# Tutorial numerical work: collaborator overview

Updated 28 October 2026.

We are building the data and trained models that the live finance tutorial will load. The aim is to infer plausible volatility-model parameters from option prices, then show how parameter uncertainty affects other prices. BayesFlow supplies the amortized neural inference; PyMC supplies reference calculations. The tutorial should run from prepared examples without asking participants to generate a large dataset or train a network.

The learning path has three stages. One training surface contains 63 simulated option prices.

| Lesson | Current position |
| --- | --- |
| Heston | A checked 100,000-surface training bank and a promising trained estimator. The frozen candidate passed its calibration checks on 2,000 previously unused cases, plus comparisons for simple variance products. It is ready for notebook packaging, with the limits below. |
| Rough Bergomi with a known variance curve | A complete 20,000-surface training bank and first trained posterior/pricing networks. Conditional reference comparisons still disagree in some cases. Independent checks found errors in the pricing network and sensitivity to the simulation grid; their full contribution to the posterior discrepancy remains unresolved. |
| Rough Bergomi with an inferred variance curve | 9,925 training surfaces are preserved toward the initial 20,000 target. The remaining training data and holdouts, trained estimators and validation are unfinished. |

The known-curve bank can grow without repeating existing simulations. The next training milestone is 50,000 surfaces, estimated at about 89 additional CPU-hours before headroom. Its append-only generation phase and completion of the original joint bank are now being prepared, with identity/seed/preservation checks before submission. They have separate bounded allocations; job IDs will be recorded once submission is confirmed. The 20,000-surface bank is an initial baseline, not a demonstrated final data requirement.

## Interesting posterior shapes

The main showcase is now [complete corner plots from interior examples](interior-corner-plots.md), rather than selected projections or boundary cases. The complete three-parameter known-curve rough plot is available; its central 99% intervals are comfortably inside the prior bounds. A new four-parameter Heston reference at a predeclared interior truth is in progress as job 8032768; its complete plot is pending the reference checks. We will show the actual geometry even if it is nearly Gaussian.

The older examples below demonstrate mainly prior-boundary skew, not strong interior curvature or multiple modes. In particular, rough eta's upper bound of 3.5 is our chosen prior limit, not a theoretical model limit. They are retained as supplementary diagnostics. The nearly-flat joint-curve example is a natural future test of weak identification.

- [Heston: skewed marginals, Gaussian comparisons and a mildly curved ridge](pilot-results/2026-10-28/heston-posterior-geometry/heston-posterior-geometry.png).
- [Rough Bergomi: the joint posterior near the eta boundary](pilot-results/2026-10-28/rough-posterior-geometry/rough-case2-reference-corner.png).
- [Rough Bergomi: coordinate transforms and the current neural/reference mismatch](pilot-results/2026-10-28/rough-posterior-geometry/rough-eta-physical-and-logit.png).

The rough figures use an approximate pricing-emulator reference. They illustrate its geometry, not an accepted exact-model posterior.

## What still needs doing

1. Connect the checked Heston candidate, cached examples and regeneration commands to the marimo notebooks; check the participant workflow. Path-dependent product pricing and broader seed/conditional robustness remain separate checks.
2. Freeze the next known-curve rough data/training allocation and numerical scope, improve both neural inference and the forward pricing emulator, and check reference comparability before claiming agreement with MCMC.
3. Complete and validate the joint-curve rough lesson.
4. Follow with the planned architecture/summary-network comparisons. Sparse option chains remain a separate later unit.

## What is on GitHub

The shared implementation, compact numerical evidence, these figures and the updated plans are pushed in **[draft PR 48](https://github.com/fmuia/sbi4finance/pull/48)**. These changes are **not merged into main**. The large simulation banks, model payloads and full posterior draws remain on Oscar; GitHub records their identities and how they were produced. Notebook integration of the new backbones is not yet delivered.

The [artifact handover](full-surface-artifact-handover.md) now gives runnable commands for a small offline Heston replay cache and optional live loading of the exported final-500 model. The initial cache is a previously inspected integration example, distinct from the new interior showcase. It requires neither the full bank nor a loaded model; current notebooks still use their historical artifacts until explicitly connected.

For details, see the [latest numerical results](backbone-validation-2026-10-28.md), [current work order](tutorial-backbone-work-order.md) and [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md). Passing the Heston checks is a useful milestone, not complete scientific acceptance of every route or the finished tutorial.
