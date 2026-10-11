# M2 Reproducible banks, offline loading, and CPU arrays

**Status:** M2.1–M2.3 implemented and locally checked for the fixed GBM task; [local evidence](../../gbm-bank-training.md). M2.4 passed a real one-CPU Oscar generation/reproduction job; [cluster instructions and run record](../../gbm-cluster-smoke.md). Larger banks/arrays and Linux neural training remain separate work. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md). Read [software and data](../notes/04-software-and-data.md).

## Purpose

Turn accepted task specifications into reusable training banks using the same generator locally and on Oscar. Establish correctness and resumability before requesting a substantial allocation.

Start with one compact bank and a thin generation/training script. This milestone does not require a data platform, custom loader or general job orchestrator. A few arrays, one metadata file and focused checks are sufficient for the first round trip; grow the storage and execution machinery when a measured experiment needs it.

## Tasks

1. Define immutable example identities and purpose-separated RNG streams using stable task/split/example identifiers and a root seed. Scheduler IDs, thread counts, shard boundaries, and retries must not define randomness. Kernels receive explicit streams or random inputs; do not share mutable generators unsafely across parallel loops. Reproducibility is within a pinned backend, not an unsupported cross-compiler bitwise promise.
2. Save compact `.npy` arrays or an NPZ bank with JSON metadata, stable IDs and basic integrity checks. Temporary writes followed by rename prevent reading unfinished files. Add many-example shards when size or cluster partitioning requires them. Preserve failed example IDs and reasons; never silently redraw numerical failures from the prior.
3. Generate independent base train/validation/test experiments before augmentation. Group each base parameter–surface example with its repeated quote-noise panels and target contracts; keep dependent overlapping return windows together. Record groups and enforce split disjointness. Preserve M0's origin, purpose, cutoff/selection, truth availability, and evidence metadata alongside bank identity.
4. Complete the first training round trip by loading the saved arrays into memory and using `fit_offline`. Implement a thin shard-aware `keras.utils.PyDataset` only when a measured bank no longer fits comfortably in memory. Use public BayesFlow interfaces and preserve training-only normalization, adapter and checkpoint behavior. Existing `fit_disk` cannot consume multi-example shards directly.
5. Start with a thin CPU job script invoking the same generator; use an array of predetermined ID ranges when scaling requires it. Set explicit thread/resource budgets and storage paths, and recheck actual cluster access/allocations before submission. Keep generation separate from GPU training and check bank integrity before dependent training.

## Ordered implementation subplan

M2.1–M2.3 establish local acceptance for independent GBM histories, and M2.4 now has measured execution on the actual cluster. Related option augmentations and dependent market windows are not implemented and will need grouped splitting when introduced. Larger arrays are triggered by a workload that needs them. Suggested destinations do not freeze filenames or public APIs.

| Step | Concrete output | Verification/handoff |
|---|---|---|
| M2.1 — Stable identities and saved bank | Explicit seed/ID derivation, compact array writer and JSON metadata, plus a small generation command using GBM. | Regenerate selected IDs independently of their batch assignment. Inspect provenance, context, labels and failure records. Sharding is an extension of the same format when needed. |
| M2.2 — Integrity and splits | Small checks for completeness, shapes, saved configuration and independent grouped splits, with safe writes and rerunning failed ID ranges. | Detect missing/duplicate IDs, changed or incomplete files and split leakage. A rerun preserves identity; related panels/windows remain grouped. This does not require a generalized recovery framework. |
| M2.3 — Load and tiny training round trip | A saved-bank → in-memory arrays → `fit_offline` demonstration, reloadable checkpoint and configuration links. Add a custom disk-batching bridge only if memory measurements require one. | Compare saved and loaded data; complete synthetic train/save/reload against GBM with every market fixture absent and provider access disabled. Check training-only normalization and no truth/diagnostic leakage. Record a simple runtime/memory measurement. |
| M2.4 — Actual cluster smoke and cost measurements | A thin CPU job script in `cluster/` invoking the same generator and checks, plus a short run record identifying code/environment, resources, storage and bank. | Execute a small real cluster job and verify output/selected ID reproduction. Measure cost before scaling. Add array partitioning when needed; actual remote execution remains pending if access is unavailable. |

## Deliverables

A small generation/checking command, saved arrays and metadata, an in-memory training/reload example, a CPU job script and local-to-cluster smoke instructions. Checkpoints reference bank/configuration and adapter versions. Shard-aware loading and larger arrays are conditional extensions, not prerequisites for accepting a small bank workflow.

## Acceptance and tests

Verify selected example identities across independent regeneration and supported batch/job partitions, and prevent incomplete writes from entering training. Detect duplicate/missing IDs, corruption, configuration mismatches, invalid rows and grouped split leakage with focused checks. Confirm loaded values and that labels/diagnostics cannot leak into conditioning. Complete the tiny synthetic train/save/reload round trip with provider access disabled and all market files absent. If a disk loader is added, compare its minibatches with the accepted in-memory path. Missing optional market assets must leave S executable, never trigger relabelling synthetic data as market data.

## Decision gate and deferred scope

Measure cold and steady-state generation, serialization, reader throughput, memory, and a small actual cluster run before sizing larger jobs. If cluster access is unavailable, local infrastructure can be accepted separately; remote execution remains pending. Market acquisition, usage permissions, and optional provider loaders are not S bank dependencies. Defer distributed shared-file writers, storage services, compression frameworks, and performance claims without measurements.
