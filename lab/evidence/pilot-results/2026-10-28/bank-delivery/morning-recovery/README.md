# Joint-bank morning recovery

The joint bank stopped overnight when its controller correctly detected a timeout. Read-only inspection found **21 timeouts, all on node2339**, and no active original joint allocations remaining. Successful work was preserved: **971 parts / 24,275 surfaces**, comprising 19,475 training, 4,800 validation and no test rows yet.

The 21 failed original map indices are **404–424**. The other missing indices, **992–1279**, had never been submitted. This reviewed recovery therefore contains exactly **309 original 25-surface jobs**: one permitted same-ID retry of each failed range, plus the 288 never-submitted ranges. No parameter vector, seed, task, producer or numerical setting changes.

## Actual resumed work

- Durable controller PID **2563401**, launched once at 10:04:32 UTC.
- Metadata/preservation gate **8046673** completed `0:0` in 15 seconds. It hashes existing payloads and validates source/runtime/map and free overlap locks; it does not repeat historical seed sweeps or generate prices.
- Pricing array **8046699** has **120 actual RUNNING workers** in the launch receipt. No excluded node appears in that receipt.
- Same one-CPU, 8 GiB, three-hour worker policy; at most 120 nonterminal workers. Subsequent missing-ID waves refill automatically within the original cumulative ceiling.

The exact remote root is `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/morning-recovery-20261028`. The controller uses **`recovery-plan.json`**, SHA-256 `714f76271559aba1c13e4bb149cda1777ecf128d1b8bb00cc53252248b68a6c8`; the active journal is `execution.json`. The committed launch copy is a dated snapshot, not a live dashboard.

## Evidence and cost

Every original worker on node2339 timed out, while all submitted workers on other nodes completed. Example failed task 404 consumed almost three CPU-hours, with about four minutes of user CPU and 175 minutes of system CPU. RSS was approximately 0.6–0.7 GiB and logs show time-limit cancellation, not a numerical exception or OOM. This is strong node-associated evidence, not a definitive hardware diagnosis. Nodes2336 and2344 also showed unusually long runtimes; the reviewed phase excludes **2338, 2339, 2336 and 2344** rather than blindly increasing every wall limit.

The [inspection](inspection.json), [node diagnosis](node-diagnosis.json) and [launch accounting](launch-accounting.psv) retain the actual records. Old timeout logs and empty staging directories remain untouched. The [preserved inventory](preserved-parts.json) records all 971 successful parts; its actual payload hashes were checked again before recovery pricing.

The baseline charge is **2,384,586 allocated CPU-seconds (662.385 hours)**, including all previous successful and failed joint allocations. The **original 1,500 CPU-hour ceiling is unchanged**. Initial 120-worker full wall/cleanup exposure plus the two-hour checks reserve is about **1,032.4 cumulative CPU-hours**, within that ceiling. Every new queued/running allocation reserves its full three-hour limit plus 240 seconds cleanup before another wave. A new failure stops for review; there is no automatic second retry or replacement seed.

## Completion and downstream readiness

The stopped original `execution-v2.json` is preserved, not rewritten. New allocations are recorded separately in this recovery's `execution.json`. Once every new worker and the final integrity allocation is terminal `COMPLETED/0:0`, the controller freezes that journal and publishes `completion.json`, binding its SHA, the original stopped journal, the 971-part inventory, reviewed failed IDs, all 309 recovery IDs, and the checked snapshot identity/counts.

The same verified completion is atomically published at the original readiness location, `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/joint/final.json`, so training and the separate 100k follow-on can proceed with an explicit recovery dependency. They must not require historical TIMEOUT records to become successful or accept a snapshot manifest alone. The original target remains **20k training / 5k validation / 7k test**; the 100k campaign is separately bounded work after this checked snapshot.

Source and receipts: [controller](joint_recovery_controller.py), [metadata/finalizer wrapper](joint_recovery_stage.py), [plan](recovery-plan.json), [checks](recovery-checks.json), [launch](launch.json), [actual running-node receipt](running-receipt.json), [execution launch copy](execution.json). No notebook, training or MCMC execution belongs to this recovery unit.
