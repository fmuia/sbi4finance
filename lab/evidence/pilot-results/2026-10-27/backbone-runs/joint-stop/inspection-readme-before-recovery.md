# Joint-bank timeout review

All 397 submitted tasks are terminal: **394 completed shards / 9,850 published training surfaces**, and three timeouts. The collection is incomplete; validation/test pools and the final 32k snapshot have not been generated.

| Original map entry / job | Censored training range | Allocated elapsed | User CPU | System CPU |
| --- | --- | ---: | ---: | ---: |
| 378 / `8019485_378` | `[9450,9475)` | 10,801 s | 675 s | 10,062 s |
| 379 / `8019485_379` | `[9475,9500)` | 10,801 s | 823 s | 9,915 s |
| 380 / `8019485_380` | `[9500,9525)` | 10,801 s | 1,023 s | 9,714 s |

All three ran on **node2338**. Slurm reports the three-hour time limit, with no Python precision traceback or OOM. Each used 90–94% system CPU and under 0.72 GiB peak RSS against 8 GiB requested memory; raw frequency counters were `341K/300K/300K`. These observations suggest a runtime/node contribution, but do not establish its cause. They do not justify rejecting financial parameter values or narrowing the prior.

The three original staging directories remain intact and contain only empty `part/` directories. There are no payloads, manifests, row checkpoints or `failure.json` records. The frozen writer buffers rows until the complete 25-row shard is ready, so no exact slow row or in-place recovery state can be inferred. These are censored ranges, not 75 identified numerical failures.

## A bounded recovery for review

[recovery-proposal.json](recovery-proposal.json) specifies **one retry of the same three original map indices**, using the unchanged source, root seed, map, prior, grids and streams. The sole resource change is `--exclude=node2338`; retain one CPU, 8 GiB and three hours per task. No wider refill is included and no jobs have been submitted by this review.

All original jobs are terminal. Nonblocking probes verified the existing range locks are free and unchanged, and canonical destinations are absent. The producer creates a fresh uniquely named staging directory, ignores orphan staging for coverage, and serializes publication under its existing locks. Empty old `.tmp` directories and persistent lock files therefore do not require deletion. Preserve them and all old/new logs. Successful recovery publishes the same canonical ranges once; it is replay of missing artifacts, not new independent training data.

Full proposed exposure is **9.5 CPU-hours**: three full three-hour caps, four-minute cleanup allowance per worker, and two five-minute metadata stages with their cleanup allowances. Including **282.2811 hours already allocated** and the original **1.1-hour final snapshot reserve**, maximum bank exposure is **292.8811 hours**, within the existing 1,500-hour ceiling. Charge actual timeout spending; queue time and batch/extern records are not counted twice. A repeated timeout, numerical failure or ambiguous submission stops this bounded recovery for review.

[inspection.json](inspection.json), [lock checks](lock-check.json), [accounting](timeout-accounting.psv), [summary](summary.json) and the raw Slurm logs retain the evidence. [source-manifest.json](source-manifest.json) binds the exact frozen `c773ceb` sources in the lossless [source archive](frozen-producer-source.tar.gz). The remote source, original job map and collection hashes matched. No pricing, inference or bank mutation occurred during inspection.
