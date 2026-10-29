# Joint-100k gate after reviewed generation recovery

The joint follow-on cannot require historical TIMEOUT allocations to become successful. Its original bank journal remains unchanged, with 21 failed indices explicitly reviewed; the recovery preserves 971 completed parts and prices only the 309 missing original ranges.

The replacement **joint-only waiter PID 2629343** was armed on **login010** at **10:10:56 UTC, 28 October 2026**. A later process check found PPID 1 and an advancing heartbeat through **10:12:56 UTC**, after the deployment connection exited. Its current state is explicitly `waiting_for_resource_campaign:generation_active`; this is an armed dependency, not a claim that joint-100k workers have started.

It waits for both:

1. The already-running known-100k campaign to publish a checked 100k/5k/7k snapshot and settle all its recorded allocations successfully.
2. The joint 20k recovery to publish its explicit `completion.json`, binding the old journal SHA, preserved 971-part inventory, original job map, reviewed failed indices 404–424, exact 309 recovered/missing indices and successful terminal NEW allocations. The standard task/source/runtime/root/holdout/snapshot checks still apply.

The old failed allocations are retained, not rewritten or silently ignored. The new gate checks only the explicitly reviewed recovery's new successful receipts, with unchanged provenance bindings. The redundant central metadata waiter was preserved/stopped; **known campaign PID 2533833 and its immutable files were not changed**.

When those dependencies pass, the existing generic launcher will append only joint training IDs 20000–99999 under the original newly authorized **3,000 allocated CPU-hour** phase ceiling, with up to 200 one-CPU workers. The separate launcher copy includes the accounting-date fix and scheduler exclusions for `node2338`, `node2339`, `node2336` and `node2344`, avoiding the observed bad/slow nodes. These scheduling exclusions do not change numerical code, parameters or seeds. No new joint-100k pricing job was submitted by the deployer.

Source/configurations are `scripts/arm_bank_extensions.py` and `configs/campaigns/joint_100k_recovery*20261028.json`. The archive retains the reviewed launch command, source/config hashes, metadata-only stop receipt, replaced waiter state, launch receipt and observed armed state. Local tests cover recovery provenance/coverage, successful new allocations despite old failures, external resource readiness, transient read-only accounting failures and once-only process launch.
