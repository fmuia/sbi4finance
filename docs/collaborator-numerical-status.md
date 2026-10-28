# Tutorial numerical work: collaborator overview

Updated 28 October 2026.

**Parallel work, 28 October:** known-curve **100k** and joint-curve **20k** banks are checked; both intermediate models completed validation. Joint100k generation continues with approximately 200 workers (36,700 training surfaces at 16:57 UTC; still active at 17:43). The known100k fit, independent pricing and current PyMC checks have completed; the architecture pilot is running. The larger fit improves some conditional posterior comparisons. Its original early-stopped pricer regressed; a completed full-300 diagnostic now improves validation error below both earlier fits, with independent rescoring still outstanding. See the [current work order](backbone-delivery-work-order.md), [rendered report](reports/inference-performance-2026-10-28.pdf), [post-report diagnostic](reports/inference-performance-2026-10-28-addendum.md) and [notebook review guide](tutorial-backbone-review.md). Original results remain preserved; no automatic promotion is made.

We are building the data and trained models that the live finance tutorial will load. The aim is to infer plausible volatility-model parameters from option prices, then show how parameter uncertainty affects other prices. BayesFlow supplies the amortized neural inference; PyMC supplies reference calculations. The tutorial should run from prepared examples without asking participants to generate a large dataset or train a network.

The learning path has three stages. One training surface contains 63 simulated option prices.

| Lesson | Current position |
| --- | --- |
| Heston | A checked 100,000-surface training bank and a promising trained estimator. The frozen candidate passed its calibration checks on 2,000 previously unused cases, plus comparisons for simple variance products. It is ready for notebook packaging, with the limits below. |
| Rough Bergomi with a known variance curve | **Checked 100k training bank**, plus 5k validation and 7k test surfaces. The 50k baseline passes all four validation rank screens, with 93.85% coverage for nominal 95% product intervals. Its emulator improves all six retained independent-price benchmark errors. New fixed-case PyMC comparisons pass 2/3 parameter screens, with case2 still failing and approximate-target/draw-count qualifications retained. **Full-100k fitting completed:** rank screens pass, conditional case2 remains discrepant. The subsequent pricer-only diagnostic improves validation RMSE to .07801 quote SD, pending independent checks. |
| Rough Bergomi with an inferred variance curve | **Checked 20k / 5k / 7k bank**. The [20k posterior, pricer and 2,000-case validation](pilot-results/2026-10-28/joint20k-completed/README.md) completed; all seven simultaneous rank screens pass, with coverage deviations and pricer tail errors still requiring review. All 32 independent paired-grid pricing batches completed; boundary-case quote error remains. The 100k append is running, with at least 36,700 training surfaces complete at 16:57 UTC. |

The delivery target is **at least 100,000 unique training surfaces for each rough route**, matching Heston; learning curves may justify more. Both intermediate baselines and the known100k fit are complete; joint data continue to accumulate. Existing prices, seeds and holdouts are reused. The known append consumed 152.902 CPU-hours within its 200-hour phase cap. The initial joint phase, including all failures and recovery, consumed 873.895 CPU-hours within 1,500; its active append retains the separate 3,000-hour cap. New work has explicit bounded allocations in the current work order.

## Interesting posterior shapes

The main showcase is now [complete corner plots from interior examples](interior-corner-plots.md), rather than selected projections or boundary cases. Both plots are available: four Heston parameters and three known-curve rough parameters, with every pair shown. Both references' central 99% intervals are comfortably inside their priors. The new Heston reference passed convergence/numerical checks and agrees well with the amortized output. These interior posteriors are mostly regular; we do not manufacture exotic geometry.

The older examples below demonstrate mainly prior-boundary skew, not strong interior curvature or multiple modes. In particular, rough eta's upper bound of 3.5 is our chosen prior limit, not a theoretical model limit. They are retained as supplementary diagnostics. The nearly-flat joint-curve example is a natural future test of weak identification.

- [Heston: skewed marginals, Gaussian comparisons and a mildly curved ridge](pilot-results/2026-10-28/heston-posterior-geometry/heston-posterior-geometry.png).
- [Rough Bergomi: the joint posterior near the eta boundary](pilot-results/2026-10-28/rough-posterior-geometry/rough-case2-reference-corner.png).
- [Rough Bergomi: coordinate transforms and the current neural/reference mismatch](pilot-results/2026-10-28/rough-posterior-geometry/rough-eta-physical-and-logit.png).

The rough figures use an approximate pricing-emulator reference. They illustrate its geometry, not an accepted exact-model posterior.

## What still needs doing

1. Review the new cached/live backbone notebook and its pinned Heston/rough examples. Its software/static checks pass; no notebook has been executed. Path-dependent product pricing and broader scientific robustness remain separate checks.
2. Finish joint100k generation and independently check the new full-300 pricer. Use the completed price/PyMC checks to diagnose conditional errors. Preserve reference-target limitations before claiming model-posterior agreement.
3. Inspect the six-variant architecture/summary pilot, then choose measured follow-ups. Known-curve set encoders now keep the seven curve levels as separate observed global context.
4. Complete product-precision validation and full-joint100k fitting when its bank finishes. Sparse option chains remain a separate later unit.

## What is on GitHub

The numerical backbone is in **[draft PR48](https://github.com/fmuia/sbi4finance/pull/48)**. The stacked **[draft PR50](https://github.com/fmuia/sbi4finance/pull/50)** adds this updated plan, the rendered report, known-curve set inputs and the actual cached/live review notebook. These changes are **not merged into main**. Large banks, model payloads and full-panel draws remain on Oscar; small teaching caches and compact evidence are committed. PR49's separate layout proposal remains open; the guide documents migration without mixing its historical artifacts into these models.

The [artifact handover](full-surface-artifact-handover.md) and [review guide](tutorial-backbone-review.md) give runnable commands for pinned offline Heston and rough replays and optional live loading. The new review notebook is linked from the historical slots; it does not silently replace SPY observations or inherit historical acceptance flags.

For details, see the [latest numerical results](backbone-validation-2026-10-28.md), [current work order](tutorial-backbone-work-order.md) and [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md). Passing the Heston checks is a useful milestone, not complete scientific acceptance of every route or the finished tutorial.
