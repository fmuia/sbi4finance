# Independent GPU work: accepted jobs and first progress

At 16:48:38 UTC on 28 October 2026, known-curve rough Bergomi 100k job **8059257** was running on gpu3001 (L40s). It had committed checkpoints at epochs 50 and 100 of 500. The unchanged frozen c96a785 worker retains final-epoch selection and the native BayesFlow sampler. This is progress evidence, not a completed evaluation.

The known100k job continues the original training allowance. Prior allocations consumed 9,188 CPU-seconds / 2,297 GPU-seconds. Its full reservation including cleanup is 15,360 / 3,840 seconds, so worst-case cumulative use is 24,548 / 6,137, within the original 32,400 / 8,100 ceilings.

Architecture job **8059371** was accepted once at 16:47:55 UTC and was awaiting GPU resources at the last recorded check. No cancellation, partition change or duplicate submission was made. The six variants use the same 20k Heston bank, 200 epochs and seed: four flow families with vector inputs, plus FlowMatching with DeepSet and SetTransformer. The separate single-job limit is 15,360 CPU-seconds / 3,840 GPU-seconds including cleanup. Its 200-case validation is a feasibility screen, not acceptance or a declared architecture winner; cross-family training losses are not comparable.

The first helper stopped before creating the pilot directory or submitting a job because its path named the parent of the immutable snapshot. The correction adds only `/snapshot-20k`; source bytes, hashes and a path-only comparison are retained. The existing snapshot is 20k training / 5k validation / 5k test, manifest SHA `1a7906f58a2b3e1e4cc868bc819c9c9308357f2f23f7fbb642c682860fb0074f`. No data were regenerated.

This development task represents four-parameter Heston through the zero-jump HKDE implementation, with kappa fixed at 2. Exact plans, executions, queue/accounting and first-progress records are embedded in `first-progress-check.json`; source and snapshot proofs are separate. These are timestamped receipts, not a live dashboard.
