# Post-bank training readiness correction

The bank finalizer writes an atomic `final.json` with `status="running"` before it performs the full integrity scan. The training waiter incorrectly treated every existing receipt other than `completed_checked_snapshot` as a terminal failure. Its outer loop then persisted `blocked` and never checked that route again.

This prevented the known-curve 50k training from starting. The bank itself is complete and checked, with snapshot SHA-256 `412f3cf437895b4764fa981bedbc1f8ad0f532d6f5b18e1b1fb957297cb627c4`. At the read-only diagnosis, its training route contained no attempt timestamp, scheduler command or job ID, its output directory did not exist, and an exact scheduler-history job-name query returned no rows. Joint training was still waiting. The [original state and source declaration](diagnosis-before.json) are preserved.

The minimal correction waits on the finalizer's known `running` state. Failed/unknown receipts and wrong producer/source/count/hash identities still block. A regression executes the real watcher path: an in-progress receipt must leave the route waiting and make no scheduler call; replacing that same receipt with a complete checked result must produce exactly one submission. The regression fails before the correction and all four scheduling/readiness tests pass afterward. [Regression record](regression.json).

## Reviewed recovery and actual training launch

The corrected source is committed at `c96a785fa1cc1a3e1ef234b8ddc81322776e5f4a`. A separate clean checkout was created at `/oscar/scratch/afengler/sbi4finance-training-recovery-v2-20261028/repo`, sharing existing Git objects and reusing the pinned GPU environment. The old `00faaef` checkout was not modified.

The [recovery receipt](recovery.json) proves that both original routes had no attempt timestamp, scheduler command, job ID or output directory, and that the exact job-name accounting query was empty. The old watcher PID 2843467 was checked on **login009**, including its exact `/proc` command, before stopping only that metadata process. Login hosts are load-balanced, so the recovery executed explicitly on login009 rather than acting on the same numeric PID on another host. The [old stopped receipt](original-execution-stopped.json) retains the blocked known route and waiting joint route; it is not rewritten as a success.

The new phase is `/oscar/scratch/afengler/sbi4finance-post-bank-baselines-recovery-v2-20261028`. Its [plan](plan-v2.json) records continuation of the same original unspent **9 CPU-hour / 2.25 GPU-hour** pool, not an additional allowance. A metadata-only `sbatch --test-only` [preflight](resource-preflight.txt) accepted the unchanged one-hour, four-CPU, one-L40S, 32-GiB request; that preflight submitted no real allocation.

**Known-curve 50k training was actually submitted once as job 8046588.** The durable replacement watcher PID is **2529399 on login009**, with joint training still waiting for its own checked snapshot. [The accepted execution ledger](execution-v2-submitted.json) and [first progress record](first-progress.json) bind the known50k bank SHA and show the worker fitting its posterior on the approved GPU allocation. This is not a claim of completed training or scientific acceptance.

The [one-time recovery source](recover_training_waiter.py) preserves receipts before mutation, stops on any submission uncertainty, and never deletes/replaces training data or models. No worker was cancelled, no uncertain job was retried, and no budget was raised. The original blocked receipt remains inspectable alongside the new versioned continuation.
