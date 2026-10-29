# Tutorial backbone correction — 27 October 2026

The [work order](../../../tutorial-backbone-work-order.md) and [central plan](../../../../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) describe the three delivered routes: Heston, known-curve rough, then joint rough. These files record the Heston data milestone and known-curve software support. They do not claim a new accepted estimator or improved posterior accuracy.

| Evidence | Contents |
| --- | --- |
| [execution.json](execution.json) | Actual stage commands/job IDs, original and current limits, completed accounting and preserved identities. |
| [preparation.json](preparation.json) | Frozen initial recipe. Its proposed eight-worker throttle was raised to 32 in the execution record after only seven joint workers remained and further joint replenishment was deprioritized. The account CPU limit and six-hour exposure cap remain respected. |
| [source inventory](source-inventory.json.gz) | Exact original collection, driver/runtime and 60 part identities, including all original training and holdout payload hashes. |
| [seed audit](seed-audit.json.gz), [audit script](seed-audit-script.py) | No collisions for 80k added training IDs, 80k reduced pricing seeds and 240k initialized PCG64 states against the retained historical inventories, including all declared joint 32k streams. Includes conservative ungenerated historical declarations, not a claim those simulations ran. |
| [preflight](heston-preflight/preflight.json) | Frozen 9d903/Linux runtime check, candidate RNG replay, preservation of 60 original parts and actual 32-entry extension map. No financial pricing calls in this stage. |
| [generation](heston-generation/summary.json) | All 32 append jobs successful,80k new training surfaces; exact declared 100k/5k/5k coverage. Metadata inspection is distinct from the subsequent full snapshot checks. |
| [snapshots](heston-snapshots/summary.json) | Both full snapshot checks, manifest hashes and exact 50k→100k training prefix/holdout preservation. Models, acceptance evaluation and notebooks were not run. |
| [known-curve verification](known-curve-verification.json) | 32 full-surface software checks, including seven new checks and an actual tiny BF2.1 train/save/reload/sample. No financial simulation or scientific acceptance follows from those software checks. |
| [joint status](joint-status.json) | Timestamped read-only snapshot of the still-separate joint task: 9,750 published training surfaces, seven running shards and no hard failures at 20:02:56 UTC. No final snapshot or validation/test generation at that point. |
| [preflight/freeze driver](preflight-script.py), [submission operator](submission-operator.py) | One-off reviewed drivers; existing frozen producer supplies generation/planning/freezing. Not a new scheduler service. |

Both checked Heston snapshots live under `/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/`, as `snapshot-50k/` and `snapshot-100k/`. Their payloads reference the existing append-only collection. All 60 original parts are preserved, including 5k validation and 5k test. Current execution used 3,641 allocated CPU-seconds (1.01139 hours), no GPU, against the separate 6 CPU-hour limit. The original bank's generation cost is additional historical work, not charged again.

Bulk metadata is losslessly compressed with deterministic gzip. [compressed-artifacts.json](compressed-artifacts.json) records raw-byte and stored-file hashes; decompress to the original names before a driver expecting raw JSON is replayed. The generation/snapshot directories carry their own downloaded-file hash inventories and compressed-log metadata. Raw logs remain byte-preserved, including whitespace.

The data component is ready for the matched 50k Heston fit. Known-curve rough scientific training, reference comparisons and the final notebook fixtures remain to be delivered. The first rough lesson conditions on a known curve per case, with curves varying across the training bank; the joint lesson infers curve uncertainty.
