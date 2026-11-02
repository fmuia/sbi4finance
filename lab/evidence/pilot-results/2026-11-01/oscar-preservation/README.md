# Oscar output preservation — 1 November 2026

The preservation audit reconciles 41 original project run roots, 1,650 JSON
receipts and 15,491 original allocation records. Unique results are preserved
off Oscar, including historical failures and partial reference chains. Remote
originals remain in place. This record concerns preservation, not acceptance of
the scientific results.

The [completion summary](summary.json) binds the full verification receipts,
native-bank identities, split archive checksums, historical inventory and final
job accounting. Large files are in the existing
[path-workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k):

| Directory | Preserved contents |
| --- | --- |
| `training-banks-20261031/` | All three native million-history trajectory banks: 23,631 NPZ shards, 31,124,149,548 native bytes; 30 split archive parts, 29,171,828,752 compressed bytes. |
| `oscar-history-20261101/` | Four historical-output groups: 20,420 logical files, 7,672,992,354 source bytes; nine split archive parts, 7,060,858,995 compressed bytes. Source/config snapshots, deployment identities, exclusion proofs and audit receipts are also retained. |
| `oscar-history-20261101/numerical-pilot-20261101/` | The isolated Heston numerical pilot, its exact source, complete results and logs. |
| `development-diagnosis-20261101/` | The source-bound development diagnosis and reproducible calculation script. |

The prior option-bank distribution and all six scale-up fits were already
preserved; the audit reconciled those copies rather than duplicating them.
The full historical archive preserves original fixed-drift path pilots, repaired
models, reference failures/partial chains and intermediate option studies.
Excluded source payloads have SHA-256 matches in durable main history or a
separate source snapshot; installed environments and administrative caches are
identified explicitly in the exclusion proofs.

All source/archive identities match. Every newly transferred native trajectory
shard passes checksum, shape, finiteness, prior-support and canonical-ID checks,
using one decoded shard at a time. The original Gmail-account copy/readback
receipts are preserved. The last five bank parts were copied and independently
read back through the verified Brown-account FileProvider view. Fresh Brown
metadata/provider flags confirm every manifest item under the same exact Google
parent and workshop folder identities. The metadata continuation does not read,
rehydrate or freshly hash the older cold payloads. The summary explicitly binds
both kinds of evidence; original pending-upload reports are retained unchanged.
An independent cloud download was not performed. Sharing permissions were
unchanged, and the cause of the Gmail-path access denial remains unknown.

All 15,499 known allocations, including seven packaging jobs and one numerical
pilot, are terminal. The eight new CPU allocations used 784 CPU-seconds
(0.21778 CPU-hours; 0.75111 including the conservative cleanup allowance), and no
GPU. Historical failures remain in the accounting and were not blindly retried.
No unpreserved unique output or nonterminal allocation was identified in the
saved accounting scope. The last successful queue check (1 November, 06:02 UTC)
showed an empty queue. A later read-only attempt (16:15 UTC) failed with expired
SSH authentication, so the live queue was not freshly rechecked. No cluster
jobs were submitted after the numerical pilot; this local preservation closure
does not require another authentication or submission.

Use the [training guide](../../../../plans/training-data-guide.md#trajectory-banks-daily-histories-with-inferred-drift)
for extraction and bounded-memory training. Operational closure does not change
the trajectory candidates' failed marginal screens or resolve Heston's numerical
law, conditional references, option product checks or fresh acceptance. Those
follow-ups remain in the [path plan](../../../../plans/PATH-INFERENCE-PLAN.md#bounded-numerical-diagnosis-follow-up--1-november-2026)
and [application execution plan](../../../../../tutorial/curriculum/TUTORIAL-REVIEW.md).
