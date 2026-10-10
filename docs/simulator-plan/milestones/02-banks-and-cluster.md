# M2 Reproducible banks, offline loading, and CPU arrays

**Status:** proposed; implementation starts after review. **Depends on:** [M0](00-contracts.md) and [M1](01-gbm-reference.md). Read [software and data](../notes/04-software-and-data.md).

## Purpose

Turn accepted task specifications into reusable training banks using the same generator locally and on Oscar. Establish correctness and resumability before requesting a substantial allocation.

## Tasks

1. Define immutable example identities and purpose-separated RNG streams using stable task/split/example identifiers and a root seed. Scheduler IDs, thread counts, shard boundaries, and retries must not define randomness. Kernels receive explicit streams or random inputs; do not share mutable generators unsafely across parallel loops. Reproducibility is within a pinned backend, not an unsupported cross-compiler bitwise promise.
2. Implement many-example `.npy` shards, JSON metadata, checksums, temporary writes, same-filesystem atomic publication, and a verifier. Preserve failed example IDs and reasons; never silently redraw numerical failures from the prior.
3. Split independent base experiments before augmentations, repeated quote-noise draws, or overlapping windows. Record grouped reference cases and enforce split disjointness.
4. Support small banks through `fit_offline`; implement a thin shard-aware `keras.utils.PyDataset` for larger banks. Use public BayesFlow approximator interfaces and preserve build/compile, training-only normalization, adapter, and checkpoint behavior. Existing `fit_disk` treats each file as one example and cannot directly consume these shards.
5. Add a CPU-array wrapper selecting predetermined shard work, bounded concurrency, explicit thread budgets, and verified storage paths. Recheck actual cluster access/allocations before submission. Keep generation separate from GPU training and require manifest verification before dependent training.

## Deliverables

Generation/verification CLIs, a manifest schema, shard reader, BayesFlow bridge, CPU-array template, and a documented local-to-cluster smoke procedure. Checkpoints reference the bank manifest and adapter/schema versions.

## Acceptance and tests

Generate identical examples under changed shard sizes, worker counts, interrupted writes, and retries. Detect duplicate/missing IDs, corruption, spec mismatches, invalid rows, and grouped split leakage. Compare a minibatch from the shard loader with the in-memory path; confirm labels and diagnostic arrays cannot leak into conditioning. Complete a tiny train/save/reload round trip against the GBM reference.

## Decision gate and deferred scope

Measure cold and steady-state generation, serialization, reader throughput, memory, and a small actual cluster run before sizing larger jobs. If cluster access is unavailable, local infrastructure can be accepted separately; remote execution remains pending. Defer distributed shared-file writers, storage services, compression frameworks, and performance claims without measurements.
