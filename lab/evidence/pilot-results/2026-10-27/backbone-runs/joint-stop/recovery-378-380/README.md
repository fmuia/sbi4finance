# One reviewed recovery of three original joint rough Bergomi ranges

This is an explicitly reviewed exception for three original TIMEOUT tasks, not an extension of the joint campaign. The original attempts and their empty staging directories remain preserved in the parent evidence directory and collection.

| Stage | Allocation | State |
| --- | --- | --- |
| Full baseline integrity preflight | 8030743, 5 minutes × 1 CPU | Completed, 12 allocated CPU-seconds |
| Same-ID recovery | 8030747_378, _379, _380; each 3 hours × 1 CPU, 8 GB | Completed, 4,064 allocated CPU-seconds total |
| Full post-recovery integrity check | 8030998, 5 minutes × 1 CPU | Completed, 11 allocated CPU-seconds |

The only worker change is excluding node2338. Source `c773ceb26e7273612b15c69fd63715d131778083`, the original map, collection, prior, numerical policy, runtime, root seed 271200, split and surface IDs are unchanged. The recovery ranges are train `[9450, 9475)`, `[9475, 9500)` and `[9500, 9525)`. Repeating their fixed seeds recovers the missing artifacts; it does not create 75 additional independent examples beyond the original allocation.

The preflight fully validated all 394 published parts (9,850 training surfaces), recorded every manifest and payload hash, confirmed the three missing canonical destinations and unheld range locks, and preserved the three original empty staging trees. Its [report](preflight.json) and [baseline hash inventory](preserved-parts.json) allow the postcheck to verify that recovery changed only the three intended destinations. Hashing ran on allocated compute; it made no financial pricing or inference calls.

The complete recovery reservation is **9.5 CPU-hours**: 3 × (10,800 + 240) worker seconds plus 2 × (300 + 240) integrity-check seconds, with cleanup included. Before submission the original joint jobs and original preflight had spent 1,016,212 allocated CPU-seconds. Adding the complete recovery reservation and the 3,960-second original final-snapshot reserve yields at most 1,054,372 seconds (**292.8811 CPU-hours**) against the existing 1,500-hour joint ceiling. The [execution ledger](execution.json) records plans before submission and keeps the original failures in the parent ledger.

The [postcheck](postcheck.json) passed: all 394 old manifests and payload hashes are unchanged, the original staging trees remain intact, and exactly the three missing canonical parts were published. The bank now has **397 parts / 9,925 training surfaces**. For all 75 recovered rows, exact prior draws and quote noise matched the original frozen-runtime seed derivation. The maximum reported pricing MCSE / quote sigma was **0.249859478**; none exceeded either the nominal 0.25 threshold or the hard 0.5 threshold. This is the existing simulator precision check, not a new grid-convergence or posterior-accuracy claim.

The recovery actually used **4,087 allocated CPU-seconds (1.13528 CPU-hours)**, including both full checks, below its 9.5-hour ceiling. The joint bank has cumulatively spent **1,020,299 CPU-seconds (283.41639 CPU-hours)**; adding the still-unused original 3,960-second final-snapshot reserve gives 284.51639 hours. No additional recovery or refill is reserved. Local and remote unit and parent ledgers agree byte-for-byte; the original three TIMEOUT attempts remain in history. All 15 downloaded remote files passed checksum verification; simulation payloads remain on Oscar.

The identical recovery tasks completed away from node2338 in 19–29 minutes. Their kernel CPU shares were about 15–23%, versus 90–94% in the original timeouts; see [accounting](recovery-accounting.psv). This supports an operational contribution to the old timeouts without identifying its cause or establishing any simulator speed claim.

There are no automatic retries or wider joint refills. Any recovery failure, repeated timeout, ambiguous submission or integrity mismatch stops this unit. This completes only the three-range recovery. It does not complete the 20k/5k/7k bank or validate an estimator. Reserved test outcomes remain unused.
