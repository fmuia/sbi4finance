# Oscar preservation: local verification complete

All identified unique outputs in the audited project scope are preserved off
Oscar. Remote originals remain. The [local summary](local-summary.json) binds
the full receipts, native-bank manifests and archive-part checksums. This is
an integrity record, not scientific acceptance.

| Preserved material | Verified locally |
| --- | --- |
| Three harmonized trajectory banks | 1,000,000 training / 4,000 validation / 4,000 test histories per model, each with 1,260 daily returns. All 23,631 native shards pass hashes, schema, finiteness, prior-support and canonical-ID checks. |
| Native trajectory storage | 31,124,149,548 bytes, preserved in 30 split archive parts totaling 29,171,828,752 bytes. The checker retains one decoded shard; the largest decoded shard is 1,319,936 bytes. |
| Historical unique outputs | 20,420 logical files / 7,672,992,354 bytes, all payload hashes verified by streamed archive reads. Nine archive parts total 7,060,858,995 bytes. |
| Source and exclusions | Source/config snapshots and deployment identities retained; redundant source payloads matched against durable main history. No additional missing source/LFS payload identified. |

Historical material includes the original fixed-drift path banks, repaired
models, intermediate option studies, and failed/partial reference chains.
Previously published option banks and all six scale-up fits were already
preserved and reconciled separately. No scientific failure was blindly retried.
The seven packaging jobs and isolated numerical pilot all completed with exit
code `0:0`, using 784 allocated CPU-seconds and no GPU. The final queue
observation was empty.

**Drive completion is pending.** Before access was denied, 35 of the 40 bank
distribution files (24,424,933,428 of 29,176,952,403 bytes, including metadata)
were copied with destination readback checksums. Historical files were also
copied. macOS now returns `Operation not permitted` for identity metadata and
ordinary reads in the authorized Drive folder. Computer-use permission is
unavailable. The exact destination checks remain intact; strict resume monitors
wait for valid access. No complete-cloud-mirror or independent-cloud-download
claim follows from these partial copies.

Local archives, extracted banks and full receipts are under
`local-artifacts/oscar-closeout-20261101/`. Use the
[training guide](../../../../plans/training-data-guide.md#trajectory-banks-daily-histories-with-inferred-drift)
for extraction, verification and bounded-memory training. The target remains the
existing [path-workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k),
with banks in `training-banks-20261031/` and historical outputs in
`oscar-history-20261101/`.
