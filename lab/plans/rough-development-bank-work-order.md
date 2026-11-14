# First joint rough development bank — active work order

**Observation refresh planning, 10 November 2026:** the [Bergomi observation-refresh plan](BERGOMI-OBSERVATION-REFRESH-PLAN.md) retains known-c Heston and physical-grid acceptance as prerequisites. Four raw-input joint-c recipes miss information seen in 7/8 ready synthetic references; the isolated whitening pilot improves c recovery but fails all six SBC screens and remains unaccepted. Only read-only bank/contract/numerical-covariance and preprocessing-design readiness can proceed now; no refreshed rough prices, banks, transform fits or training are authorized. After Heston acceptance, a bounded exact-time rough numerical panel must audit both raw-likelihood and transformed-input error before a reuse/compute decision. Historical allocations below are preserved, and PR #74's rough lesson remains a separate integration track.

**Authorized and started on 27 October 2026, within the 1,500 allocated CPU-hour ceiling.** The [execution report](native-default-and-bank-start.md) records the frozen collection, seed audit, initial shards and current status. The [completed reliability unit](followup-results-2026-10-27.md) supports this development-bank decision. It does not certify absolute pricing bias or remove the known grid dependence. This work order adds no new infrastructure or architecture campaign.

## What we generate

Use [the prepared configuration](../configs/calibration_rough_joint_development_20k.json):

- **20,000 training + 5,000 validation + 7,000 reserved test = 32,000 price surfaces.** The test pool is generated and checked for integrity, but not evaluated or used for model selection.
- The same complete 63-contract design, six joint targets, broad prior and monotone exponential initial curve. No supplied generating curve, missing quotes or narrower prior.
- Working block steps **1280 / 2016 / 2016**, through 10 / 63 / 252 trading days. Retain **2560 / 4032 / 4032** as the later refinement comparator. A finer comparison is not exact truth.
- The existing control-variate estimator, conditional pricing, antithetics and adaptive path ladder, with diagnostic capture disabled. Keep the empirical 0.25 reported-MCSE/quote-SD target; separately count the existing final-rung allowance up to0.5 and retain hard failures. Do not change quote noise or redraw failed parameters.
- Root seed **271200** is frozen for this distinct collection. The pre-pricing audit found no collisions across all declared candidate streams and retained historical inventories; the candidate stream digests were replayed on Linux before pricing. Root-number differences alone are not an audit.

The configuration contains the exact frozen Linux working task from the diagnostic declaration. **Do not use the small pilot config** for bank generation: it still has 8/2/2 rows and the older 80/504/504 grids. Collection `817de3664968de0c605ae4c73e2c99c26465344294af2fc5412f39fffa4baa50` and the 1,280-entry job map are frozen. No complete snapshot has been published yet.

## Compute and submission limits

**Authorized generation ceiling: 1,500 allocated CPU-hours, with no GPU allocation.** This includes setup, generation, integrity/snapshot checks and any approved recovery within the unchanged collection. It is a spending cap, not a promise that the whole bank will finish.

The previous 16-point working cohort averaged 132.5 allocated seconds/surface, including its slow point. Scaling that mean gives **1,178 CPU-hours** for 32k prices; 1,500 provides about 27% headroom. Both numbers remain sensitive to the small sample and heterogeneous runtime. The new fixed-path 2.35× grid ratio is not a new adaptive-throughput estimate.

Use the existing incremental collection/shard commands with **25 surfaces per shard**: 1,280 planned shards, one CPU and 8 GiB each. The three-hour per-shard limit allows for uneven costs; a mean-throughput shard would take about 55 minutes. Start with a small set of real training shards to confirm the path. Keep successful shards as part of the bank, then expand to **at most 200 concurrent workers** as the account permits.

Submit in waves. Before each wave, require **completed allocated CPU spending + full time-limit exposure of every queued/running/new job + scheduler-cleanup allowance + final integrity reserve ≤ 1,500 hours**. Keep at most 200 nonterminal workers across all arrays, rather than applying that limit separately to each wave. The operator reserves 240 seconds per nonterminal task for cleanup beyond the time limit. Do not submit all 1,280 three-hour jobs at once: their combined potential exposure exceeds the ceiling. This can be managed with the existing job map and accounting; no new scheduler service is needed. Queue time is not allocated CPU spending, and 200 available cores are not guaranteed.

At 200 continuously available workers, 1,178 hours corresponds to an ideal work/cores time of **5.9 hours**. Shard imbalance, queueing, startup and failures add time. Stop rather than automatically increase the cap if completion is not affordable within it.

## Preservation and stop rules

Freeze the collection's task, source/runtime hashes, root, split IDs and producer before the first shard. Confirm the candidate source and byte-locked runtime on Oscar; the opt-in diagnostic addition has tested bit-identical default outputs. A changed algorithm, grid, prior or observation law needs a distinct collection, not relabelled prices.

Append only missing IDs. Preserve successful shards and fixed holdouts; publish the snapshot only after exact coverage, shapes, identities and seed checks pass. A recoverable interruption may repeat the same unfinished IDs/streams with its cost recorded. A numerical/support/precision failure keeps its original ID visible and stops expansion for review; no easier replacement seed. Do not silently bypass frozen-producer checks or regard a partial collection as complete.

The 0.25 rule remains empirical. Record covariance, path counts and exceptions; do not describe the bank as a prior-wide certificate of continuous-model accuracy. Retain the known low-H/high-eta and high-variance cases for the later numerical/posterior/product checks.

## What follows

When the 32k snapshot is checked, fit **one joint posterior and one forward price emulator**, with validation selection and reserved tests unopened. Cost those fits, independent labels and the predeclared six-case refinement panel separately before submission. Start its feasibility work with the already agreed central and high-variance cases; do not replace difficult cases after seeing results.

The independent native-only cached Heston check is complete, with results in the execution report. The earlier 400-Euler comparison was superseded by the user's choice to use the BayesFlow default. Wider data growth, architecture screens, direct rough Monte Carlo MCMC and sparse-chain work remain downstream. The 1,500-hour ceiling covers this first bank, not the whole tutorial campaign.
