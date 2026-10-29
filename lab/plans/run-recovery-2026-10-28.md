# Rough-bank runs: completed recovery and current generation

**Historical recovery snapshot:** this document retains the 15:09 UTC completion/handoff state. Independent work has since started; use the [current parallel work order](backbone-delivery-work-order.md), [collaborator overview](collaborator-numerical-status.md) and [rendered report](../evidence/reports/inference-performance-2026-10-28.pdf) for the new runs and findings.

Status checked **28 October 2026, 15:09 UTC**. Known-curve 100k and joint-curve 20k banks are checked and complete. Both intermediate model-training routes have completed. Joint generation toward 100k is running; final rough-Bergomi model acceptance remains unfinished.

The overnight run made useful data, but it did not advance through all the promised dependencies. The known-curve bank finished at 50k training surfaces. Joint generation stopped after 21 workers timed out; two orchestration bugs then kept the 100k append and model-training stages waiting. Both bugs have been fixed with regressions that reproduce the observed failures. Completed prices are preserved.

| Work | Verified position | What follows |
| --- | --- | --- |
| Known curve: bank | **Checked 100,000 training / 5,000 validation / 7,000 test**, completed at **12:39:54 UTC**. All 2,000 appended shards succeeded. | The 50k-to-100k append charged **152.902 CPU-hours**, within its 200-hour cap. Inspect the intermediate learning results before the planned full-100k fits. |
| Known curve: intermediate models | GPU job **8046588 completed 0:0**, including both fits and 2,000-case validation. All four rank screens pass; product-interval undercoverage remains. | Inspect [completed baseline evidence](../evidence/pilot-results/2026-10-28/known50k-completed/README.md), then conditional/reference and independent-price checks. The original pooled allowance charged 1.343 CPU-hours / 0.336 GPU-hours for this route. Reserved test outcomes remain unused. |
| Joint curve: initial bank | **Checked 20,000 training / 5,000 validation / 7,000 test**, with all 1,278 successful parts preserved. Both replacement ranges and both finalizers completed 0:0. | [Completed integrity/handoff evidence](../evidence/pilot-results/2026-10-28/joint-handoff-completed/README.md). Full phase cost, including failures, is **873.895 CPU-hours**, within the original 1,500-hour cap. |
| Joint curve: intermediate models | GPU job **8053365 completed 0:0**, including both fits and 2,000-case validation. All seven rank screens pass; coverage deviations and pricer outliers remain. | Inspect [completed joint20k evidence](../evidence/pilot-results/2026-10-28/joint20k-completed/README.md). Conditional reference, independent-price and product checks remain unfinished. |
| Joint curve: 100k append | **Generation active**, campaign PID **4070959**, first array **8053366**. At 15:08 UTC, **21,825 training surfaces** were complete; approximately **200 workers** are running. | Append training IDs 20,000–99,999 under the existing 3,000 CPU-hour cap. Five slow nodes are excluded. No failed allocations were observed in the active append at the live check. |

**Recovery and automatic handoff completed:** the two slow node2346 jobs timed out, and the reviewed continuation regenerated only their original missing ranges. Both succeeded, native publication completed, and the unchanged full integrity finalizer verified the entire bank. The bridge published readiness at **14:35:30 UTC**; by **14:37 UTC**, the existing operators had automatically started joint GPU training and 200-worker generation. No observer launches were needed. Both stopped histories and failed costs remain preserved. The [earlier 13:58 receipt](../evidence/pilot-results/2026-10-28/tail-live-status-1358.json) is retained as historical evidence, not current status.

## Why it stalled

1. **Joint workers:** all 21 timeouts were on node2339, dominated by operating-system CPU time. Workers on other nodes completed. Recovery retains the same simulator, sample identities, seeds, three-hour wall time and 1-CPU/8-GiB request, while excluding nodes2338, 2339, 2336 and 2344, which showed severe slowdowns. There was no observed out-of-memory or Python numerical failure in those timed-out workers.
2. **100k waiter:** its scheduler query used a UTC date as Slurm's local calendar-date filter. A completed metadata job submitted shortly after midnight UTC was invisible to that query. The fix queries one day earlier while retaining exact job IDs.
3. **Training waiter:** the finalizer publishes an ordinary `status="running"` receipt during its integrity scan. The waiter treated that transient state as a permanent failure. It now waits for successful completion and still blocks failed or incompatible receipts.

The original launch checks established that the processes were detached and advancing. They did not exercise these two later transitions, so they were insufficient evidence that the entire overnight chain would proceed. The regressions cover the actual date boundary and running-to-completed transition. All **23 relevant campaign, extension-waiter, training-operator and completion-bridge tests pass** in the pinned local core environment, including delayed read-only accounting and once-only finalizer submission.

## Data, budgets and provenance

The known-curve 50k snapshot manifest SHA-256 is `412f3cf437895b4764fa981bedbc1f8ad0f532d6f5b18e1b1fb957297cb627c4`.

The completed known-curve **100k** manifest SHA-256 is `f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313`. Its source/task/split identity and settled accounting are recorded in the [completion evidence](../evidence/pilot-results/2026-10-28/known100k-completed.json). The current trained rough model still uses the intermediate 50k snapshot.

The completed joint-curve **20k** manifest SHA-256 is `c0de030bafef6de3f728d75788cd7feddd672b476795b57c42ecd8cbb0843ce9`. Its [completion and original readiness records](../evidence/pilot-results/2026-10-28/joint-handoff-completed/completion.json) were verified byte-identical. Full prior/context/noise replay and generator-precision checks passed without nominal-precision exceptions; continuous-time grid/refinement acceptance remains separate.

Joint recovery begins with **662.385 allocated CPU-hours already consumed**, including the failed allocations, within the original cumulative **1,500 CPU-hour** cap. Its first wave of up to 120 workers fits within that cap even if every requested wall time is consumed. No cap is increased. Recovery retries the 21 reviewed failed identities once; it does not replace their seeds or regenerate successful shards. The remaining 288 identities had never been submitted.

The original intermediate-training phase is now **finished**, with both routes completed; its durable ledger remains `/oscar/scratch/afengler/sbi4finance-post-bank-baselines-recovery-v2-20261028/execution.json`. The active joint-100k ledger is `/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons/joint-100k/execution.json`. Completed known-100k and recovery ledgers remain unchanged at their recorded paths. The active generator is detached and does not require an interactive SSH session to stay open. A fresh authenticated connection is needed to inspect it.

Original failed journals remain available beside separately versioned recovery records. Scientific claims remain unchanged: the joint bank targets the recorded finite simulator, known-curve emulator/MCMC disagreement still needs review, and final 100k fitting follows review of the intermediate learning results. No notebook was run during recovery.

[15:09 live generation receipt](../evidence/pilot-results/2026-10-28/joint-generation-live-1509.json) · [Completed joint handoff](../evidence/pilot-results/2026-10-28/joint-handoff-completed/README.md) · [Completed joint model evidence](../evidence/pilot-results/2026-10-28/joint20k-completed/README.md) · [Two-range recovery history](../evidence/pilot-results/2026-10-28/bank-delivery/tail-recovery/README.md) · [Collaborator overview](collaborator-numerical-status.md).
