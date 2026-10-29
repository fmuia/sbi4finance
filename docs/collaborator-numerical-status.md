# Tutorial numerical work: collaborator overview

Updated 29 October 2026.

**Collaborator data access:** all three complete native banks are copied and hash-verified locally (29,766 files; 2.35 GB). Versioned download archives total 1.96 GB, plus a separate 38 MB pretrained-and-cases ZIP. The [training-data guide](training-data-guide.md) provides download/checksum/extraction steps, array meanings and real loader/trainer commands. Experiments can reuse these prices without Oscar access or fresh simulation. Reserved-test policy and the scientific limitations below remain unchanged.

**Current position, 29 October:** all three routes now have **100,000-surface training banks and completed baseline fits**. The final joint-curve bank passed verification at 02:04 Berlin time; its automatic fit and validation finished at 02:36. Nothing is running on Oscar. Both joint20k and joint100k pass all seven declared marginal rank checks; the larger fit improves pricing errors, with the qualifications below. All six architecture pilot models have also finished. Remaining work is conditional reference/product validation and lesson packaging. See the [current work order](backbone-delivery-work-order.md), [completed joint comparison](pilot-results/2026-10-29/joint100k-completed/README.md), [report and addendum](reports/inference-performance-2026-10-28-addendum.md) and [notebook review guide](tutorial-backbone-review.md). New checkpoints have not automatically replaced reviewed caches.

We are building the data and trained models that the live finance tutorial will load. The aim is to infer plausible volatility-model parameters from option prices, then show how parameter uncertainty affects other prices. BayesFlow supplies the amortized neural inference; PyMC supplies reference calculations. The tutorial should run from prepared examples without asking participants to generate a large dataset or train a network.

The learning path has three stages. One training surface contains 63 simulated option prices.

| Lesson | Current position |
| --- | --- |
| Heston | A checked 100,000-surface training bank and a promising trained estimator. The frozen candidate passed its calibration checks on 2,000 previously unused cases, plus comparisons for simple variance products. It is ready for notebook packaging, with the limits below. |
| Rough Bergomi with a known variance curve | **Checked 100k training bank and completed fit**, plus 5k validation and 7k test surfaces. All four marginal rank checks pass. The full-300 pricer has validation RMSE .07801 quote SD and improves all six saved independent benchmark points over the earlier fits. The 100k posterior passes 2/3 fixed-case agreement screens against the existing **50k-pricer PyMC reference**; the difficult case remains discrepant. The new pricer has not yet been used for new PyMC references. |
| Rough Bergomi with an inferred variance curve | **Checked 100k / 5k / 7k bank and completed fit.** All seven marginal rank checks pass on 2,000 validation cases. Compared with20k, pricer label RMSE improves by20.2%, and measured RMS errors improve at all four saved independent benchmark points on both grids. Conditional PyMC agreement and product checks remain unfinished. [Completed comparison](pilot-results/2026-10-29/joint100k-completed/README.md). |

The initial delivery target of **100,000 unique training surfaces for each rough route**, matching Heston, is achieved. Both intermediate baselines and both full-data rough fits are complete. Further generation should follow a specific learning or numerical question. Existing prices, seeds and holdouts are reused. The known append consumed 152.902 CPU-hours within its 200-hour phase cap. The initial joint phase, including all failures and recovery, consumed 873.895 CPU-hours within 1,500; its completed append used 1,700.489 CPU-hours within the separate 3,000-hour cap, including the preserved timeout and recovery. New work has explicit bounded allocations in the current work order.

## Interesting posterior shapes

The reference-checked examples are [complete corner plots from interior cases](interior-corner-plots.md). Both plots are available: four Heston parameters and three known-curve rough parameters, with every pair shown. Both references' central 99% intervals are comfortably inside their priors. The new Heston reference passed convergence/numerical checks and agrees well with the amortized output. These interior posteriors are mostly regular; we do not manufacture exotic geometry.

The older examples below demonstrate mainly prior-boundary skew, not strong interior curvature or multiple modes. In particular, rough eta's upper bound of 3.5 is our chosen prior limit, not a theoretical model limit. They are retained as supplementary diagnostics. The new joint audit below quantifies the weak identification associated with nearly flat curves.

The new [joint100k sanity report](joint-rough-posterior-sanity.md) adds complete six-parameter corners from two explicitly selected interior cases. One shows a curved long-variance/decay ridge; the other shows strong eta/rho dependence. These are learned-posterior candidates, with conditional PyMC validation still pending. Across all 2,000 validation cases, physical interval widths, parameter recovery and a control pairing posteriors with other cases' truths strongly reject a prior-only sampler. Decay remains broad for nearly flat curves, while the variance curve itself can be tightly learned. No caches or acceptance flags were promoted.

- [Heston: skewed marginals, Gaussian comparisons and a mildly curved ridge](pilot-results/2026-10-28/heston-posterior-geometry/heston-posterior-geometry.png).
- [Rough Bergomi: the joint posterior near the eta boundary](pilot-results/2026-10-28/rough-posterior-geometry/rough-case2-reference-corner.png).
- [Rough Bergomi: coordinate transforms and the current neural/reference mismatch](pilot-results/2026-10-28/rough-posterior-geometry/rough-eta-physical-and-logit.png).

The rough figures use an approximate pricing-emulator reference. They illustrate its geometry, not an accepted exact-model posterior.

## What still needs doing

1. Review the new cached/live backbone notebook and its pinned Heston/rough examples. Its software/static checks pass; no notebook has been executed. Path-dependent product pricing and broader scientific robustness remain separate checks.
2. Review the completed 100k fits and independent pricing rescores. Diagnose the remaining known-curve conditional discrepancy and establish appropriate joint-curve PyMC comparisons; preserve the approximate reference-target limitations.
3. Inspect the six-variant architecture/summary pilot, then choose measured follow-ups. Known-curve set encoders now keep the seven curve levels as separate observed global context.
4. Complete the remaining product checks and package the reviewed models into the lessons. Sparse option chains remain a separate later unit.

## What is on GitHub

The numerical backbone is in **[PR48](https://github.com/fmuia/sbi4finance/pull/48)**. The stacked **[PR50](https://github.com/fmuia/sbi4finance/pull/50)** adds this updated plan, the rendered report, known-curve set inputs, the cached/live review notebook and collaborator data access. These linked PRs record the dependent integration stack and its final merge status. Large banks/model payloads are distributed as versioned release assets, with verified local copies independent of Oscar; small teaching caches and compact evidence are committed. PR49's separate layout proposal remains open; the guide documents migration without mixing its historical artifacts into these models.

The [artifact handover](full-surface-artifact-handover.md) and [review guide](tutorial-backbone-review.md) give runnable commands for pinned offline Heston and rough replays and optional live loading. The new review notebook is linked from the historical slots; it does not silently replace SPY observations or inherit historical acceptance flags.

For details, see the [latest numerical results](backbone-validation-2026-10-28.md), [current work order](tutorial-backbone-work-order.md) and [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md). Passing the Heston checks is a useful milestone, not complete scientific acceptance of every route or the finished tutorial.
