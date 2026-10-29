# Native-default check and rough-bank start: evidence guide

The human-readable report is [Native FlowMatching defaults and the first rough-bank shards](../../../native-default-and-bank-start.md). These files preserve execution provenance and development results; they do not mark a teaching checkpoint or complete bank as accepted.

## Heston native solver

| Location | Contents |
| --- | --- |
| [attempt 8010024](heston-native/attempt-8010024/) | Original input/default manifest, guard traceback, accounting and post-failure input-integrity audit. It failed before posterior sampling. |
| [attempt 8010270/run.json](heston-native/attempt-8010270/run.json) | Corrected run: source `343cb45`, installed native defaults, runtime/GPU, input hashes and completed case states. |
| [case 0](heston-native/attempt-8010270/case-0/comparison.json), [case 1](heston-native/attempt-8010270/case-1/comparison.json), [case 2](heston-native/attempt-8010270/case-2/comparison.json) | Conditional parameter/product metrics and paired changes against the retained Euler outputs. Each directory also contains its `(10000,4)` `native_draws.npz`. |
| [corrected accounting](heston-native/attempt-8010270/accounting.psv) | Slurm job/step records, actual allocation time, rounded time limit and CPU-frequency counters. These counters are not GPU clocks. |
| [downloaded-file hashes](heston-native/attempt-8010270/downloaded-files.sha256) | Checksums for the copied native outputs and logs. |

The retained Euler comparison/draw files and training metadata are in the earlier [FlowMatching follow-up](../followup/heston/flow-matching-500/). The three MCMC observations are from the original readiness bank, manifest `e5c12a5f…`; the model's training/validation snapshot is `1a7906f…`. Equal short observation labels do not identify the same collection row.

The model and older epoch/validation artifacts remain on Oscar at their recorded paths. The corrected completion guard verifies frozen inputs stayed unchanged. Only the three native comparisons ran: no extra solver variant, new fit, MCMC, 500-case validation/SBC, test evaluation or pricing belongs to this check. The first guard failure remains part of its cost. The 630-second request and Slurm's rounded 660-second limit are both retained; use the rounded value for reservation accounting.

## Rough collection and audit

| File or directory | Purpose |
| --- | --- |
| [execution.json](execution.json) | Authorizations, frozen source/deployment, submitted jobs, resource caps and collection identifiers. |
| [submission-operator.py](submission-operator.py) | Current submission-operator snapshot. Earlier ledger entries retain their original operator hashes; this copy does not reconstruct every historical version. |
| [collection.json](collection.json) | Frozen collection configuration, identity, generator hashes and versions. This is not a completed snapshot manifest. |
| [jobs-20k.json](jobs-20k.json.gz) | Declared 32k-surface job map: 1,280 shards of 25 rows, including fixed validation/test allocations. A job-map entry does not imply submission or completion. |
| [seed-audit.json](seed-audit.json.gz), [audit script](seed-audit-script.py) | All candidate IDs, possible block/rung streams, reduced seeds and initialized-state comparisons with the retained historical inventory. |
| [historical manifests](historical-manifests.json.gz), [Oscar inventory](prior-oscar-inventory.json.gz) | Retained audit inputs. Historical declared configurations can include IDs that were never generated. |
| [bank-preflight.json](bank-preflight.json), [preflight script](preflight-script.py) | Linux replay of candidate seed/state digests and collection/job-map preparation; no financial pricing. |
| [bank-wave-4-203](bank-wave-4-203/summary.json) | Launch checks for job8010762: original indices4–203, 200 running workers, resource/source/runtime identities and log inventory. No pricing payloads or outcomes were retrieved. |
| [bank-initial](bank-initial/) | Initial real production-shard manifests, Linux prior/quote reconstruction checks, progress journal and accounting snapshots. |
| [bank-initial/summary.json](bank-initial/summary.json), [progress.jsonl.gz](bank-initial/progress.jsonl.gz) | Checked initial-tranche summary and compressed progress history. Use the report's status paragraph when communicating a row count. |

Bank root: `/oscar/scratch/afengler/sbi4finance-native-bank-20261027`. Producer source: `c773ceb26e7273612b15c69fd63715d131778083`. Collection ID: `817de3664968de0c605ae4c73e2c99c26465344294af2fc5412f39fffa4baa50`.

The audit found no candidate/internal or retained-history collisions across 704,000 candidate reduced pricing seeds and 768,000 initialized states. Its scope is finite, explicit stream identities; it does not prove mathematical independence. Bank progress is incremental. The 200-worker and 1,500-allocated-CPU-hour limits govern future waves, not a claim that 32k rows are complete. Reserved test outcomes remain unopened.

Preserve raw run metadata and failed/incomplete states. Calculate allocated CPU from top-level elapsed time times allocated CPUs; do not add batch/extern allocation time again. Keep measured process CPU and GPU allocation time separate.

Large JSON inventories, job maps and the audit are retained as deterministic gzip files to keep the review compact. [compressed-artifacts.json](compressed-artifacts.json) gives original-byte and stored-file hashes. Decompress them to their original names in a scratch directory before replaying a driver that expects raw JSON; the execution ledger's hashes refer to those original bytes. The live Oscar files remain uncompressed and unchanged.
