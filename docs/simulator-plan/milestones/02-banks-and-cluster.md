# M2 Reproducible banks, offline loading, and CPU arrays

**Status:** proposed; implementation starts after review. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md). Read [software and data](../notes/04-software-and-data.md).

## Purpose

Turn accepted task specifications into reusable training banks using the same generator locally and on Oscar. Establish correctness and resumability before requesting a substantial allocation.

## Tasks

1. Define immutable example identities and purpose-separated RNG streams using stable task/split/example identifiers and a root seed. Scheduler IDs, thread counts, shard boundaries, and retries must not define randomness. Kernels receive explicit streams or random inputs; do not share mutable generators unsafely across parallel loops. Reproducibility is within a pinned backend, not an unsupported cross-compiler bitwise promise.
2. Implement many-example `.npy` shards, JSON metadata, checksums, temporary writes, same-filesystem atomic publication, and a verifier. Preserve failed example IDs and reasons; never silently redraw numerical failures from the prior.
3. Generate independent base train/validation/test experiments before augmentation. Group each base parameter–surface example with its repeated quote-noise panels and target contracts; keep dependent overlapping return windows together. Record groups and enforce split disjointness. Preserve M0's origin, purpose, cutoff/selection, truth availability, and evidence metadata alongside bank identity.
4. Support small banks through `fit_offline`; implement a thin shard-aware `keras.utils.PyDataset` for larger banks. Use public BayesFlow approximator interfaces and preserve build/compile, training-only normalization, adapter, and checkpoint behavior. Existing `fit_disk` treats each file as one example and cannot directly consume these shards.
5. Add a CPU-array wrapper selecting predetermined shard work, bounded concurrency, explicit thread budgets, and verified storage paths. Recheck actual cluster access/allocations before submission. Keep generation separate from GPU training and require manifest verification before dependent training.

## Ordered implementation subplan

All steps below are planned, not implemented. Work in order; each step uses the preceding step's output. M2.1–M2.3 establish local acceptance. M2.4 requires measured execution on the actual cluster and stays pending if access is unavailable; an offline rehearsal cannot satisfy it. Suggested destinations do not freeze filenames or public APIs.

| Step | Concrete output | Verification/handoff |
|---|---|---|
| M2.1 — Stable identities and shard generation | Identity/RNG derivation, manifest and shard writers in `datasets/`, plus a generation command in `scripts/`, using the accepted GBM task. | Generate the same example IDs with changed shard sizes and worker assignments; compare values within the pinned backend. Inspect provenance, context, labels and numerical-failure records. |
| M2.2 — Verification, retries and splits | A bank verifier, safe publication/retry behavior, and independent grouped train/validation/test assignment. | Exercise interrupted writes, duplicate/missing IDs, corruption, incompatible specifications and split leakage. Retrying must preserve example identity; related noise panels/windows must remain grouped. Hand only verified banks to training. |
| M2.3 — Shard loading and tiny training round trip | A shard-aware reader/BayesFlow bridge in `datasets/` and `adapters/`, with a small training demonstration in `scripts/` and checkpoint provenance. | Compare loaded and in-memory minibatches; complete synthetic train/save/reload against the GBM reference with every market fixture absent and provider access disabled. Check training-only normalization and that truth/diagnostics cannot leak into conditioning. Record local throughput and memory. |
| M2.4 — Actual cluster smoke and cost measurements | A CPU-array wrapper in `cluster/` invoking the same generation/verification commands, plus a measured run record identifying source/environment, resources, storage and bank manifest. | Recheck access and allocation, execute a small real cluster array, verify its output and test retry/reproducibility behavior there. Measure generation, serialization and reading costs before sizing larger jobs; record remote execution as pending if it cannot run. |

## Deliverables

Generation/verification CLIs, a manifest schema, shard reader, BayesFlow bridge, CPU-array template, and a documented local-to-cluster smoke procedure. Checkpoints reference the bank manifest and adapter/schema versions.

## Acceptance and tests

Generate identical examples under changed shard sizes, worker counts, interrupted writes, and retries. Detect duplicate/missing IDs, corruption, spec mismatches, invalid rows, and grouped split leakage. Compare a minibatch from the shard loader with the in-memory path; confirm labels and diagnostic arrays cannot leak into conditioning. Complete a tiny synthetic train/save/reload round trip against the GBM reference with provider access disabled and all market files absent. Missing optional market assets must leave S executable, never trigger relabelling synthetic data as market data.

## Decision gate and deferred scope

Measure cold and steady-state generation, serialization, reader throughput, memory, and a small actual cluster run before sizing larger jobs. If cluster access is unavailable, local infrastructure can be accepted separately; remote execution remains pending. Market acquisition, usage permissions, and optional provider loaders are not S bank dependencies. Defer distributed shared-file writers, storage services, compression frameworks, and performance claims without measurements.
