# Paired Monte Carlo validation-loss audit

Job **8032144** completed the single authorized read-only comparison of the same 100,000-surface model's original selected epoch 402 and post-hoc final epoch 500. All 32 predeclared paired replicates completed. This is development evidence, not independent scientific acceptance.

| Checkpoint | Mean validation loss | Replicate SD |
| --- | ---: | ---: |
| Original selected 402 | 0.20877201701924206 | 0.0043565165657381425 |
| Post-hoc final 500 | 0.2070838402248919 | 0.004349081171869794 |

The paired difference, **final 500 minus selected 402**, is **−0.0016881767943501495**, with paired SD **0.00030356542188329577** and Monte Carlo standard error **0.0000536632920868584**. Final500 has lower averaged surrogate loss as well as the better posterior diagnostics found in the separate checkpoint audit. This supports noisy single-epoch checkpoint selection as a contributor; this comparison does not show that a higher mean surrogate loss was needed to improve the posterior diagnostics. It does not establish that final 500 is universally optimal or reconstruct the historical selection randomness exactly.

## What was evaluated

- The exact stored noisy quotes and targets for validation:0 through validation:4999 in snapshot100k, with no quote augmentation and no reserved-test access.
- BayesFlow 2.1.0's actual `BasicEstimator.compute_metrics` / `FlowMatching.compute_metrics` through `JAXEstimator.stateless_compute_metrics(stage="validation")`, without gradients, fitting, optimizer updates, pricing, posterior sampling, or MCMC.
- Each checkpoint's serialized adapter and saved standardizer moments. The two adapters produced identical transformed input bytes; standardizer moments were preserved separately and were never refitted.
- Seeds 281001 through 281032, with the same seed state for both checkpoints in each replicate. The FlowMatching generator controls both Gaussian base noise and sampled times. It advanced exactly twice per minibatch; every other non-trainable model variable remained unchanged.
- Batch size 256 with 19 full batches and a 136-row tail. Losses were weighted by row count, matching the framework's epoch loss tracker. A synthetic partial-tail check agreed with the actual framework test-step reduction within float32 rounding.

The audit used canonical fixed row order in place of the historical validation shuffle, and independent explicit replicate states in place of the historical evolving training/validation RNG stream. Python float64 accumulation can differ from the framework's float32 tracker at rounding precision. These differences are recorded in the machine-readable report.

The report verifies unchanged live model variables, standardizer state, frozen checkpoint/metadata files, and raw/adapted validation bytes after all replicates. It records actual API/method and source-file hashes, adapter configurations, standardizer values, checkpoint hashes, input hashes, seeds and every paired loss. The reviewed driver SHA256 is `f039389d8d118caa755a6bfefc0da8671034dff9bbdcacda268e659c22d65f09`; the frozen repository revision is `56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9`.

## Resources and provenance

The prospective ledger was frozen remotely before the only `sbatch` submission. The allocation was 4 CPU, one L40S GPU, 24 GiB, with a 5-minute Slurm limit, 270-second internal deadline and no requeue. Slurm reports **COMPLETED**, exit 0:0, elapsed **46 seconds**, hence **184 allocated CPU-seconds and 46 GPU-seconds**. The driver reports 42.18 seconds internally. The log's generic prolog prints "CPUs:1"; authoritative allocation accounting and the driver's `SLURM_CPUS_PER_TASK==4` guard establish 4 CPU.

Together with initial-controls job 8031924, actual use of the initial-controls pool is **856 CPU-seconds / 214 GPU-seconds**. Before submission, prior actual use plus this allocation's full 5-minute exposure and 240-second cleanup reserve was **2832 CPU-seconds / 708 GPU-seconds**, below the separate 7200 CPU-second / 900 GPU-second ceilings. No cleanup allocation, retry, or further job was used. Final-panel job 8032002 belongs to a separate pool.

Evidence files:

- [Full paired report](results/validation-loss-mc/run.json), including all 32 pairs and immutable-state checks.
- [Execution ledger](execution.json), [submitted ledger](execution-submitted.json), and [Slurm accounting](accounting.psv).
- [Raw log](logs/validation-loss-mc-8032144.out), [reviewed driver](heston_validation_loss_mc_20261028.py), and archived submission/download helpers.
- [Downloaded-file checksums](downloaded-files.json) and `sha256sums.txt`, which inventories every archived file except itself.

Both checkpoint and validation payloads remain on Oscar. Only this small report, metadata, source and log archive was downloaded. No repository commits were created by this audit worker.
