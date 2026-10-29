# Accounting-date failure and known-100k release

The original follow-on waiter remained armed overnight but did not release known-100k generation. Its Slurm query passed the ledger's UTC calendar date to `sacct`, which interprets a date-only filter in the cluster's local timezone. Metadata job **8032794** started at 00:36 UTC on 28 October, which was 20:36 EDT on 27 October. `-S 2026-10-28` returned no rows; `-S 2026-10-27` returned its successful completion. The rows were excluded before parsing. [Captured raw output](accounting-date-failure.json) reproduces this exact failure.

The generic launcher now queries from one day earlier, still restricted to the exact recorded job IDs. A regression uses the real captured output and checks that `.batch`/`.extern` records are excluded. This is an accounting fix, not a change to the financial simulator or numerical criteria.

The old no-attempt waiter **3792953** was preserved and stopped on login010. A versioned patched launcher outside the untouched frozen checkout released the existing known-curve 100k configuration once. Campaign **2533833** accepted array **8046496**; its first **64 workers were observed RUNNING**. Native planning emitted exactly 2,000 new 25-row shards, `train:50000` through `train:99999`. It did not reprice the existing 50k prefix or holdouts. The new phase retains its **200 allocated CPU-hour** ceiling and original source/root/collection.

The temporary central metadata waiter **2533826** was subsequently replaced by the [joint-only recovery-compatible gate](../joint100k-recovery-gate/README.md). The actual known-100k campaign, source and configuration were left unchanged. Its native metadata preparation took about 44 seconds; no historical-state reconstruction was added.

The archive preserves the old waiter plan/state/receipt, prospective recovery and launch records, exact deployment helper, known campaign receipt and first submission state. Fourteen launcher/dependency tests passed locally without scheduler submissions, including the real-text date regression and reviewed-recovery gate tests. Allocation accounting remains in the campaigns' live ledgers.
