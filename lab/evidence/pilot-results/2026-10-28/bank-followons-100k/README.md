# Both rough banks: 100k follow-ons armed

The user requested **100,000 training surfaces for each rough route**, with further incremental growth possible. The durable waiter was launched once at **01:12:46 UTC on 28 October 2026**, PID **3792953** on **login010**. A later check at 01:15:11 UTC found the same process reparented to PID 1, with shared-ledger heartbeats advancing after the deployment SSH session exited. Oscar's load-balanced login hosts explain why a PID may not appear in a new SSH session on another host.

This is an **armed follow-on**, not a claim that the new 100k tranches are already pricing. Current generation jobs continue under their original budgets and operators.

| Follow-on | Starts after | New generation allowance |
| --- | --- | --- |
| Known curve, 50k → 100k train | The current 50k/5k/7k snapshot passes its final checks and all current phase allocations settle successfully. | **200 allocated CPU-hours**, up to 64 one-CPU workers, 40 minutes and 8 GiB each. |
| Joint curve, 20k → 100k train | Its own current 20k/5k/7k snapshot passes and settles **and the known-curve 100k campaign completes and settles**. | **3,000 allocated CPU-hours**, up to 200 one-CPU workers, three hours and 8 GiB each. |

The joint resource dependency avoids an unrestricted overlap of 200 joint workers with 64 known workers. It is explicit: the routes are not advertised as fully independent starts. If known-curve follow-on generation fails, the joint follow-on records a blocked dependency rather than silently exceeding capacity. Original joint generation can still finish.

Both new tranches retain the existing numerical producer, task, root, collection and 5k/7k holdouts. Only missing training IDs are planned. Native whole-collection uint64 pricing-seed audits run; no claim is made that an older cross-collection initialized-state certificate covers these expanded ranges. The original 120-hour known and cumulative 1,500-hour joint phase ceilings remain unchanged; the new tranches have separate budgets and ledgers.

The waiter verifies the checked predecessor report, immutable manifest, exact collection/source/runtime/count identities and successful terminal accounting before launching the generic campaign. Attempts are persisted before any launch process. Ambiguous launch/submission outcomes are not retried. Transient read-only accounting timeouts remain waiting states.

The frozen generic launcher is from commit `00faaefe946c8e8fef3b55ff3156c2d0db61ffb3`. The reviewed standalone waiter/configs are separately hashed in the prospective deployment and launch receipt. No historical-state scan, numerical pilot or additional Slurm job was launched by the deployer.

Evidence:

- [Prospective deployment and frozen input hashes](deployment-prospective.json)
- [Actual deployment result](deployment-launch.json)
- [Detached waiter receipt](waiter-launch.json)
- [Observed waiter state](waiter-execution.json)
- [Host/heartbeat liveness observation](verification.json)
- [One-shot deployment source](deploy_followons.py)

Operator and configurations are in `scripts/arm_bank_extensions.py` and `configs/campaigns/*100k*20261028.json`. Five small dependency/once-only tests passed without numerical work or scheduler submission.
