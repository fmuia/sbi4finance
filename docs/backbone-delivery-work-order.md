# Finish the first tutorial backbone

Updated 28 October 2026. The objective is a usable bank, estimator, reference evidence and small replay bundle for each lesson. Larger architecture studies are not a prerequisite for this first delivery. The [collaborator summary](collaborator-numerical-status.md) and [artifact handover](full-surface-artifact-handover.md) are the short review entry points.

**Overnight setup is verified and armed:** current generation continues, both 100k append campaigns wait on checked predecessors, and intermediate training is gated to start automatically. See the [overnight handover](overnight-handover-2026-10-28.md) for exact process IDs, new phase limits, dependencies and failure behavior.

The delivery baseline is **100k unique training surfaces for each rough route**, matching Heston, subject to measured learning curves. The 50k known-curve and 20k joint snapshots below are intermediate milestones. They are not the final data-sufficiency targets. Larger tranches must retain the same identities and holdouts, and receive an explicit allocation rather than silently increasing the current caps.

## Work proceeding in parallel

| Work | Concrete output | Current execution |
| --- | --- | --- |
| Complete interior corner plots | Four inferred Heston parameters and three known-curve rough parameters, every pair shown; checked references and explicit boundary distances. | [Both plots](interior-corner-plots.md) completed. Heston **8032768** passed reference/precision/interior checks in .5289 CPU-hours, no GPU. |
| Grow known-curve rough data | Immutable **50k training / 5k validation / 7k test** snapshot; reuse the existing 20k prefix and all holdouts. | Array **8032877** began with 64 running workers; subsequent refill arrays append further missing ranges. Preflight **8032794** passed. |
| Finish the original joint bank | Immutable **20k training / 5k validation / 7k test** snapshot; retain all 9,925 existing training rows. | Array **8032878** began with 120 running workers after preflight **8032795** passed. The persistent controller manages both ledgers independently. |
| Package usable artifacts | Small offline Heston cache, correctly bound final-500 export and instructor regeneration/live commands. | Code, guide and initial integration fixture are pushed. Actual ordinary-loader/direct-candidate CPU sampling produced identical seeded draws; no raw bank is needed. |
| Prepare the next fits | One posterior and one forward emulator for each completed rough snapshot, using BayesFlow 2.1 and validation only. | Recipes and an explicit final-checkpoint policy are being prepared. No new training jobs have been submitted by this phase yet. |

The bank controller is a persistent, bounded process on Oscar; it does not depend on an open assistant tool call. Current PID **2709727** and `execution-v2.json` are under `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028`. Its launch, prospective plan, pre-pricing wrapper correction and job attempts are recorded in [the generation evidence](pilot-results/2026-10-28/bank-delivery/README.md). Missing scheduler records retain full exposure; an ambiguous submission, failed component or incompatible artifact stops the operator rather than generating a replacement seed. Current status is recorded in that ledger; old submission receipts do not imply active generation.

## Generation policy and spending

The known-curve append keeps the original **80/504/504 finite-simulator task**, source `6163520`, root **271300** and collection identity. It adds only `train:20000` through `train:49999`, in 1,200 shards of 25 surfaces. The new phase limit is **120 allocated CPU-hours**, including setup, final checks and full outstanding worker/cleanup exposure. Workers request one CPU, 8 GiB and 40 minutes; at most 64 are nonterminal. Reserve two CPU-hours for setup/final checks. This is a data-sufficiency step for the declared numerical task, not a claim that more coarse-grid rows remove the measured pricing bias.

The joint continuation keeps the existing **1280/2016/2016** task, source `c773ceb`, root **271200** and original map. Only original indices 397–1279 are missing: 883 shards covering 22,075 remaining training/holdout surfaces. Its existing cumulative ceiling remains **1,500 allocated CPU-hours**, including the **283.4164 hours** already consumed. Workers retain one CPU, 8 GiB and three hours; at most 120 are nonterminal. Both streams together have at most **184 bank workers**, leaving room for reference/other work. Library threads remain one per worker.

Before submitting workers, verify actual generator-module and runtime identities, preserved payload bytes, fixed holdouts, missing ranges and appended pricing streams. Before publishing a snapshot, verify exact coverage, numerical metadata, identities and seed collisions again. Successful surfaces are retained even if the phase stops. A changed grid, prior or observation law requires a distinct collection.

## Completion path after the banks

1. Freeze each checked snapshot and train its first posterior and forward emulator with the [declared post-bank phase](post-bank-baseline-phase.md) and explicit checkpoint rule. Use validation for development; reserve test outcomes until the candidate is frozen. Retain original minimum-loss and final-epoch provenance separately. These are intermediate learning-curve baselines on the way to the 100k targets.
2. Reuse the fixed reference cases and independent price-error benchmark. Improve/check the emulator before describing its PyMC posterior as accurate model inference. Keep Monte Carlo label covariance and numerical scope explicit; do not erase failed boundary cases merely because the main showcase is interior.
3. Complete the corresponding product checks. The standard spot-notional forward-start payoff and historical return-ratio payoff now have distinct implementation paths. Independent asset-payoff tests verify the new conditioning formula; production grid/path precision still needs its own bounded check.
4. Export correctly bound checkpoints and small interior example caches, complete full corner plots, and provide regeneration commands. Connect these artifacts to the marimo sequence only as an explicit integration step. No notebooks have been executed in this work.
5. Take stock of the three routes together: record demonstrated strengths, remaining limitations and the smallest useful next architecture/summary-network comparison. Sparse option chains remain a later unit.

The interior examples need not have exotic geometry. Show the checked posterior honestly; use the predeclared nearly-flat joint-curve case to investigate weak identification rather than moving prior bounds to manufacture skew or selecting visually attractive neural outputs without a reference.
