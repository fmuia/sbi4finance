# Finish the tutorial backbone in parallel

Updated 28 October 2026. The joint 100k bank blocks its full-100k fit; it does not block work on other banks, estimators, references or notebooks. The goal remains a usable tutorial substrate with scientific limitations visible, rather than a production experiment platform.

Start with the [rendered performance report](reports/inference-performance-2026-10-28.pdf), [collaborator overview](collaborator-numerical-status.md) and [notebook review guide](tutorial-backbone-review.md). The report covers Heston100k, known-curve rough50k and joint-curve rough20k, plus the completed known100k learning step. The 50k replay/reference is retained; a larger bank does not automatically promote a new fit.

## Work already started

| Track | Concrete output | Current position |
| --- | --- | --- |
| Joint data | Append unique training IDs 20,000-99,999, retaining all previous data and fixed holdouts. | Approximately 200 CPU workers; 36,700 training surfaces complete at 16:57 UTC. Existing 3,000 CPU-hour append cap unchanged. |
| Known-curve 100k fit | Same 500-epoch final-selection posterior, forward pricer and 2,000-case validation as the 50k baseline. | **8059257 completed**; all four rank screens pass. Fixed-reference posterior comparisons improve in case2 but still pass only 2/3 cases. The early-stopped pricer is worse than 50k, so it is not promoted; a bounded full-300-epoch diagnostic is being submitted separately. |
| Architecture/summary pilot | Six controlled variants on the same 20k Heston prefix: affine, spline, FlowMatching and MeanFlow with vector inputs; FlowMatching with DeepSet and SetTransformer. | **8059371 submitted, awaiting GPU resources**. Equal 200 epochs, seed and data; 200-case validation is an exploratory screen, not acceptance or a final winner selection. |
| Independent prices and gradients | Rescore known50k/100k on the same six-point panel; test joint20k at two fixed validation cases, truth and saved neural mean, with eight paired coarse/fine batches. | All 32 joint pricing batches and summary **8059983 completed**. The joint boundary challenge has a clear local quote-error issue; interior small RMS is at Monte Carlo noise level and not an accuracy certificate. Known50k improves on 20k at all six points; known100k regresses on those points. |
| Current conditional PyMC references | Reuse the original fixed known-curve cases, observed quotes/curves and PyMC DEMetropolisZ recipe with the current 50k pricer and saved neural draws. | **8059665 completed**; all three reference diagnostics pass and parameter-agreement screens pass 2/3 cases. The target remains the approximate conditional-mean emulator, not an exact stochastic-simulator posterior. |
| Notebook/artifact integration | Load pinned offline Heston/known50k/joint20k examples, inspect complete corner plots, and opt into live inference with a matching checkpoint. | Implemented in the new [backbone review notebook](../tutorial_final/notebooks/backbone_review.py). Contract tests and static checks pass. No notebook was executed. |

The set-input implementation also supports known curves correctly: quote tokens enter the learned summary, and seven observed curve levels remain separate global context. The vector contract is unchanged. The joint curve remains hidden from inference. Software tests do not establish posterior performance for the new summaries.

## What the next results should decide

1. Retain the matched known50k/100k comparison and diagnose the pricing-network regression. The 100k pricer stopped after 116 epochs (selected 86), while 50k completed 300 (selected 298); insufficient schedule completion is a hypothesis, not a proven cause. Test full 300 epochs in a new directory with the same data, widths and seed; preserve all old fits. Neither model's case2 posterior meets the predeclared fixed-reference H/eta rules.
2. Use independent prices to distinguish surrogate error, Monte Carlo variability and grid effects. Keep full batch covariance and paired differences. A refined grid is a comparator, not exact truth; low average pricer RMSE does not rule out damaging conditional errors.
3. Use the small architecture pilot to identify worthwhile follow-ups and measured training/sampling costs. Training losses from different flow families are not comparable scores. This is one-factor screening, not a full family-by-summary-by-seed study.
4. Validate the explicitly named spot-notional forward-start call and analytic variance controls before presenting rough product-price distributions. The legacy return-ratio payoff and a flat curve extension beyond the observed horizon are not substitutes. The product-precision recipe is prepared; reconcile its separate scope/allocation before submission.
5. Freeze the joint100k snapshot when generation finishes, then run its full-data fit. The joint20k model already supports diagnostics, reference preparation and notebook integration.

Reference convergence, neural/reference agreement, generator precision and downstream pricing answer different questions. Passing one does not imply the others. Keep PyMC where natural, preserve prior/transform Jacobians and chain axes, and label approximate-emulator references accurately.

## Bounded allocations

- **Known100k fit:** completed within the original 9 CPU-hour / 2.25 GPU-hour pool. Total actual use across both intermediate fits and known100k is 15,100 CPU-seconds / 3,775 GPU-seconds. A separate 10-minute pricer-only diagnostic plus cleanup reserves 3,360 / 840 seconds, keeping cumulative exposure at 18,460 / 4,615, below the original 32,400 / 8,100 ceilings.
- **Architecture pilot:** a separate single-job allowance: one GPU and four CPUs for at most one hour, plus a conservative four-minute cleanup reservation (1.067 GPU-hours / 4.267 CPU-hours). No automatic retries or full sweep are authorized by this pilot.
- **Price validation:** completed within the original 24 CPU-hour pool. Including prior work and the preserved failed two-second metadata check, cumulative actual cost is 12,404 seconds (3.446 CPU-hours). All 34 revised allocations completed 0:0. No automatic precision expansion follows from unresolved estimates.
- **Known50k references:** reuses the original six CPU-hour pool. Prior use was 788 seconds; the new job used 756, bringing actual cumulative use to 1,544 seconds, below 21,600.

Bank and numerical producers remain frozen. Large banks, models and draw panels remain on Oscar; small replays, report sources and compact evidence are committed. Sparse option chains stay a separate later unit.

## Repository and review boundaries

This unit starts from current PR48 in a new worktree. Main is unchanged; [PR49](https://github.com/fmuia/sbi4finance/pull/49)'s proposed course/tutorial/lab layout remains open. The review guide records migration points without moving or relabelling historical SPY artifacts.

Historical plans, failures and costs remain in [the original work order](tutorial-backbone-work-order.md), [recovery history](run-recovery-2026-10-28.md) and linked evidence. Their dated launch states are not the current queue. The [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) remains the scientific specification.
