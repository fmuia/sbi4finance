# Rough-bank runs: morning recovery

Updated **28 October 2026, 10:13 UTC**. These are intermediate results, not accepted final rough-Bergomi tutorial models.

The overnight run made useful data, but it did not advance through all the promised dependencies. The known-curve bank finished at 50k training surfaces. Joint generation stopped after 21 workers timed out; two orchestration bugs then kept the 100k append and model-training stages waiting. Both bugs have been fixed with regressions that reproduce the observed failures. Completed prices are preserved.

| Work | Verified position | What follows |
| --- | --- | --- |
| Known curve: bank | Checked **50,000 training / 5,000 validation / 7,000 test**. At 10:13 UTC, 201 additional shards had completed: **at least 55,025 training surfaces**. | Append only training IDs 50,000–99,999. Campaign PID **2533833**, first array **8046496**, up to 64 workers with automatic refill. Phase cap: 200 CPU-hours. |
| Known curve: intermediate models | GPU job **8046588** completed its 500-epoch posterior fit in 309 seconds and pricing-emulator fit in 79 seconds; now running validation. | Validate on 2,000 validation surfaces. The original pooled 9 CPU-hour / 2.25 GPU-hour allowance covers both intermediate routes. No reserved test results are inspected. |
| Joint curve: initial bank | **19,475 training / 4,800 validation / 0 test**, in 971 preserved immutable parts. Recovery array **8046699** is now running. | Recover exactly 309 original shard IDs, with up to 120 workers: 21 timed-out shards and 288 never submitted. Finish the checked 20k / 5k / 7k snapshot before releasing its model training. |
| Joint curve: 100k append | Replacement gate **PID 2629343 on login010** is armed, currently waiting for known-100k generation. | After known-100k completion and checked joint recovery, append training IDs 20,000–99,999 under the existing 3,000 CPU-hour cap, up to 200 workers. Completion is bound to the explicit recovery receipt; the old failed journal is not rewritten as successful. |

## Why it stalled

1. **Joint workers:** all 21 timeouts were on node2339, dominated by operating-system CPU time. Workers on other nodes completed. Recovery retains the same simulator, sample identities, seeds, three-hour wall time and 1-CPU/8-GiB request, while excluding nodes2338, 2339, 2336 and 2344, which showed severe slowdowns. There was no observed out-of-memory or Python numerical failure in those timed-out workers.
2. **100k waiter:** its scheduler query used a UTC date as Slurm's local calendar-date filter. A completed metadata job submitted shortly after midnight UTC was invisible to that query. The fix queries one day earlier while retaining exact job IDs.
3. **Training waiter:** the finalizer publishes an ordinary `status="running"` receipt during its integrity scan. The waiter treated that transient state as a permanent failure. It now waits for successful completion and still blocks failed or incompatible receipts.

The original launch checks established that the processes were detached and advancing. They did not exercise these two later transitions, so they were insufficient evidence that the entire overnight chain would proceed. The new regressions cover the actual date boundary and running-to-completed transition. All **18 campaign, extension-waiter and training-operator tests pass** in the pinned local core environment, including the recovery receipt and external resource-dependency checks.

## Data, budgets and provenance

The known-curve 50k snapshot manifest SHA-256 is `412f3cf437895b4764fa981bedbc1f8ad0f532d6f5b18e1b1fb957297cb627c4`.

Joint recovery begins with **662.385 allocated CPU-hours already consumed**, including the failed allocations, within the original cumulative **1,500 CPU-hour** cap. Its first wave of up to 120 workers fits within that cap even if every requested wall time is consumed. No cap is increased. Recovery retries the 21 reviewed failed identities once; it does not replace their seeds or regenerate successful shards. The remaining 288 identities had never been submitted.

The model-training watcher is PID **2529399 on login009**. Its durable ledger is `/oscar/scratch/afengler/sbi4finance-post-bank-baselines-recovery-v2-20261028/execution.json`. The known-100k campaign has its own ledger under `/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons/known-100k/`. The joint-100k gate ledger is `/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons/joint-recovery-gate-20261028/waiter/execution.json`; the joint recovery ledger is `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/morning-recovery-20261028/execution.json`. These detached jobs do not require an interactive SSH session to stay open. A fresh authenticated connection is needed to inspect them.

Original failed journals remain available beside separately versioned recovery records. Scientific claims remain unchanged: the joint bank targets the recorded finite simulator, known-curve emulator/MCMC disagreement still needs review, and final 100k fitting follows review of the intermediate learning results. No notebook was run during recovery.

[10:13 UTC live status receipt](pilot-results/2026-10-28/run-recovery-live-status.json) · [Accounting and known-100k launch evidence](pilot-results/2026-10-28/bank-followons-accounting-recovery/known100k-launch.json) · [Joint accounting inspection](pilot-results/2026-10-28/bank-delivery/morning-recovery/inspection.json) · [Joint recovery plan](pilot-results/2026-10-28/bank-delivery/morning-recovery/recovery-plan.json) · [Armed joint-100k gate and heartbeat](pilot-results/2026-10-28/joint100k-recovery-gate/README.md) · [Training recovery and regression evidence](pilot-results/2026-10-28/training-readiness-recovery/README.md) · [Collaborator overview](collaborator-numerical-status.md).
