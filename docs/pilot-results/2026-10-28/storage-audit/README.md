# Measured bank storage — 28 October 2026

This is a read-only measurement during active generation. It counts completed `arrays.npz` payloads with published part manifests, not whole project directories. It reads ZIP metadata and the small NPY shape/dtype headers from 16 spread completed shards per collection; it does not load price arrays, test outcomes, simulation trajectories or all covariance matrices. Generation was not paused or changed.

| Bank at audit time | Completed rows | Actual compressed payload | Bytes per surface |
| --- | ---: | ---: | ---: |
| Known-curve rough | 22,625 train + 5,000 validation + 7,000 test | 308,303,905 bytes / 294.0 MiB | 8,904 |
| Joint-curve rough | 9,925 train; holdouts pending | 88,598,802 bytes / 84.5 MiB | 8,927 |
| Heston | 100,000 train + 5,000 validation + 5,000 test | 126,541,157 bytes / 120.7 MiB | 1,150 |

The actual bank-only counts are lower than broad directory totals, which also include code, logs, prior runs and metadata. Known/joint counts can increase immediately after the timestamp in [banks.json](banks.json); this is a snapshot, not a live status page.

## What is stored

The bank stores noisy observations, clean numerical price estimates, parameter vectors, example IDs and curve context where relevant. It also retains per-quote pricing standard errors, a 63×63 covariance matrix of **Monte Carlo price means**, path counts and refinement records. It does **not** retain the simulated stock/variance trajectories or individual Monte Carlo path payoffs.

About **88% of the sampled compressed rough payload is auxiliary diagnostic data**. Covariance alone is about 7.28 kB per surface, compared with about 1.05–1.08 kB for the roles requested by the projected posterior-training loader. The covariance has 31,752 uncompressed data bytes per surface, but block zeros/symmetry compress well. The covariance and error records support the scientific checks of shared-path pricing uncertainty; they are not network inputs. In the joint task, even the stored curve values remain auxiliary rather than inference conditions.

The trainers use projected loading and omit the large covariance tensors. Snapshots refer to the original shards and do not copy the price bank. Heston has zero Monte Carlo covariance, which compresses to very little; its refinement differences occupy more space than that zero matrix.

## Extrapolation to 100k training cases

Holding the current numerical target and measured compression rate fixed:

| Route | Training + fixed holdouts | Estimated full compressed bank |
| --- | ---: | ---: |
| Known-curve rough | 100,000 + 12,000 | 997,257,397 bytes / .929 GiB |
| Joint-curve rough | 100,000 + 12,000 | 999,805,121 bytes / .931 GiB |
| Heston | 100,000 + 10,000 | 126,541,157 bytes / .118 GiB, already measured |

These are extrapolations, not forecasts. A hypothetical rough export containing only the current projected training roles would be around .110–.112 GiB at those counts. We have not made such an export or removed diagnostics. Keeping the full existing scientific bank is inexpensive at this scale; no storage emergency follows from the bank measurements alone.

## Other measured retention

The bounded [checkpoint file audit](checkpoints.json) counts only published `.keras` models and immediate recovery pairs:

| Existing run | Recovery model files | Actual recovery model bytes |
| --- | ---: | ---: |
| Heston 100k | 20 | 244,929,700 / 233.6 MiB |
| Heston 50k | 20 | 244,929,680 / 233.6 MiB |
| Known-curve rough 20k | 24 | 294,273,960 / 280.6 MiB |

Each published model is about 12.25 MB; the new teaching export contains one model and no recovery copies. The three recovery trees total about 748 MiB. Older unreferenced recovery pairs, redundant combined-plus-partial evaluation arrays and duplicated environments are more useful future retention targets than removing the bank's scientific diagnostics. Before pruning, preserve the current recovery pointer's best/last files and every file referenced by published scientific provenance. **Nothing was deleted.**

Environment sizes and the personal scratch quota were **not measured** in this audit. A shared filesystem's free-space total is not the user's quota. No slow recursive environment/scratch scan was attempted here.

[Bank audit source](audit_bank_storage.py), [checkpoint audit source](audit_checkpoint_storage.py), [exact bank measurements/role sizes](banks.json) and [checkpoint measurements](checkpoints.json) make the calculation inspectable. The first Heston-path lookup failed because the collection is under `heston-development/collection`; the corrected successful audit took 2.48 seconds of remote metadata processing. No numerical calculation or Slurm allocation was added.
