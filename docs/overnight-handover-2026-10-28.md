# Overnight setup: verified and armed

Verified at **01:15 UTC on 28 October 2026**. The processes below run detached on Oscar and do not depend on this chat or an open local terminal. Fresh shared-filesystem heartbeats were checked; process IDs are local to Oscar's load-balanced login hosts.

| Component | Verified state | Durable record |
| --- | --- | --- |
| Current bank generation | Active controller **2709727**, refilling known-curve 50k and joint 20k milestones. At the check, 333 new known shards and 13 new joint shards had completed: at least **28,325 / 10,250 training surfaces** respectively. | `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/execution-v2.json` |
| Both 100k extensions | Waiter **3792953**, verified on login010 with parent PID 1 and advancing heartbeats after SSH exit. Both campaigns are armed, not yet running pricing jobs. | `/oscar/scratch/afengler/sbi4finance-overnight-20261028/followons/waiter/execution.json` |
| Intermediate model training | Watcher **2843467**, with advancing heartbeat; waiting for each route's checked bank independently. It will train a posterior and forward emulator, then run validation-only diagnostics. | `/oscar/scratch/afengler/sbi4finance-post-bank-baselines-20261028/execution.json` |

## What will happen automatically

1. Finish the running known-curve **50k training** and joint **20k training** banks, including their fixed 5k/7k holdouts. Preserve and reuse existing surfaces; publish immutable checked snapshots.
2. Train the first larger posterior/emulator baselines when each intermediate snapshot passes its gate. Use the declared final-epoch posterior policy, native BayesFlow sampler and validation only. No reserved test outcomes or notebooks are evaluated automatically.
3. Append known-curve training IDs **50,000–99,999** after its checked 50k snapshot and settled workers, under a new **200 CPU-hour** phase limit and at most 64 workers.
4. Append joint training IDs **20,000–99,999** after its checked 20k snapshot and settled workers. This campaign additionally waits for known-100k completion so it can use up to **200 workers** without overlapping another 64-worker bank. Its new phase limit is **3,000 CPU-hours**. Initial phase caps remain unchanged.

The follow-ons are launched once, in bounded waves; this does not reserve every shard in Slurm upfront. Source, task, root, seed protocol, holdouts and completed data remain fixed. The final 100k fits are a later decision after inspecting the intermediate learning results; the armed training phase covers known50k and joint20k.

## Failure behavior and expectations

Unknown submissions, scientific/numerical failures, incompatible artifacts or exhausted allocations stop the affected operator for review; they do not cause replacement seeds or automatic budget increases. Completed data and partial training/evaluation outputs are retained. Read-only accounting delays are retried by the follow-on waiter without retrying launches.

This is a verified unattended setup, not a guarantee that every job succeeds or that joint100k finishes by morning. The finer joint simulation remains materially more expensive and may continue beyond the night. The current checkpoints, logs and receipts will make the morning review concrete.

[Verification and training receipts](pilot-results/2026-10-28/overnight-setup/verification.json) · [100k follow-on receipts and tests](pilot-results/2026-10-28/bank-followons-100k/README.md) · [Current generation evidence](pilot-results/2026-10-28/bank-delivery/README.md) · [Delivery work order](backbone-delivery-work-order.md) · [Complete interior plots](interior-corner-plots.md).
