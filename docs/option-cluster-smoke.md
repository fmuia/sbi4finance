# Option banks: measured Oscar smoke run

On 11 October 2026, the existing CPU job generated and verified all three
24-panel presets on Oscar. The returned files match independently retrieved
remote hashes. Selected panels also reproduce on the Mac within the recorded
portability tolerances. This completes the small option-workload generation
and retrieval check; it does not accept an option posterior or a larger bank.

Read the [machine-readable evidence](results/option-cluster-smoke.json) or open
`pixi run review-bergomi`: the final section of
[notebook 08](../notebooks/08_bergomi_models.py) explains the source-to-bank
workflow, measured costs and distinct verification questions. It reads cached
evidence and submits no jobs.

## What ran

Each job uses one CPU, 4 GB requested RAM and a ten-minute limit. The active
allocation was discovered before submission: `carney-mjfrank-condo2`, `batch`,
QOS `carney-condo2`. Each unchanged preset contains 16 training, four validation
and four test panels, each with nine synthetic European put quotes.

| Model | Job | Generation/write (s) | Exact Linux check (s) | Job elapsed (s) | Total CPU (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| HKDE | 7297096 | 2.45 | 0.20 | 4 | 1.50 | 144.48 |
| One-factor Bergomi | 7297097 | 111.95 | 44.88 | 158 | 155.33 | 76.16 |
| Rough Bergomi | 7297098 | 116.59 | 45.68 | 164 | 160.06 | 103.98 |

All three scheduler states are `COMPLETED`, exit `0:0`. Generation time includes
writing the bank; the job additionally loads and exactly regenerates the first,
middle and last IDs of each split in reverse order. That is nine regenerated
panels per model. Stored arrays occupy 23,648 bytes per bank; this excludes
pricing workspaces. Peak RSS is the scheduler's batch-step measurement.

The Linux environment was Python 3.12.8, NumPy 2.5.3 and SciPy 1.18.1, with
Numba 0.68.0 installed. The Mac used Python 3.12.15 and the same package pins.
Threads were capped at one in the job. These tiny jobs establish feasibility
and resource scale; they are not a throughput benchmark. In particular, job
elapsed time includes setup and regeneration, and different machines need not
give the same cost per surface. No GPU or training environment was needed.

## Source, transfer and numerical reproduction

The submitted archive came from commit
`25f4070753243c89ac46b365877cafdeb688410f`, with SHA-256
`b44c6f83383ca609ccf50f87107481e707f1c80ffde8d3eb7b3149b3da1f4f8c`.
The report lists its exact paths, configuration hashes, scientific source
hashes, manifest/split hashes and scheduler accounting. Only source, task
configurations and the thin job script were sent; no market fixture was used.

Three checks answer different questions:

1. **Transport:** compare remote and local SHA-256 for every manifest,
   verification JSON and NPZ file. All 15 files match byte for byte.
2. **Same-environment replay:** the Linux job regenerates selected panels
   exactly from their stored task, root seed, split and base IDs. All pass.
3. **Portable replay:** the local checker requires matching scientific source
   hashes and IDs, records environment differences, then compares regenerated
   floating arrays numerically. It reads the bank without modifying it.

| Model | Largest reference-price difference, Mac versus Linux (currency) | Original portable check | Revised currency-aware check |
|---|---:|---|---|
| HKDE | 1.54857e-9 | Failed | Passed |
| One-factor Bergomi | 1.24345e-14 | Passed | Passed |
| Rough Bergomi | 1.24345e-14 | Passed | Passed |

The original portable check used `rtol=1e-12, atol=1e-14` for every array. HKDE
exceeded this criterion. After inspecting that failure, the local checker was
changed to use `atol=max(1e-14, 1e-6 * quote_noise_std)` for currency arrays:
observed quotes, reference prices, sampling SEs and refinement differences.
Here this is **2.5e-7 currency**, one-millionth of the declared quote sigma.
The relative term remains `1e-12 * abs(regenerated value)`. Parameters and
currency-squared covariance retain `atol=1e-14`; IDs remain exact.

This portability rule was selected **after** the observed failure. The report
preserves the original outcomes and identifies the revised local checker by
its file hash separately from the submitted archive. No bank or scientific
kernel was changed, and the original Linux checks were not relabeled as runs
of the revised checker. HKDE's measured price difference is about 6.2e-9 of
quote sigma. Its separate Fourier refinement limit remains 0.001 of quote
sigma, or 0.00025 currency. This adjustment makes no change to numerical
pricing, posterior or empirical acceptance.

The measured Bergomi MCSE/quote-sigma maxima remain 0.0814 and 0.0741. Those
are sampling-precision checks. Portable agreement says nothing about remaining
asset-grid or rough-kernel bias; both machines can reproduce the same biased
approximation. The existing corner/center screens still do not prove bias
below 0.025 currency.

## Reproduce the route

Use the active allocation and a new scratch directory. Follow the existing
cluster instructions for authentication and environment discovery. The run
used a small Python 3.12 environment with the package pins above; installation
occurs before submission, never inside the job. Submit from the extracted
source checkout, with `OPTION_RUN_ROOT` pointing to that scratch directory:

```bash
for model in hkde bergomi rough_bergomi; do
  sbatch --parsable --account=carney-mjfrank-condo2 \
    --partition=batch --qos=carney-condo2 \
    --job-name="sbi-options-$model" \
    --output="$OPTION_RUN_ROOT/$model-%j.out" \
    cluster/option_bank.sbatch "$OPTION_RUN_ROOT/venv/bin/python" \
    "$OPTION_RUN_ROOT/$model-bank" "configs/tasks/${model}_option_bank.json"
done
```

Keep the archive/hash, job IDs, accounting, complete banks and remote file
hashes. Copy each bank to a new local directory and compare all file hashes
before numerical replay. With the updated checker, a local check is:

```bash
PYTHONPATH=src pixi run python scripts/generate_option_bank.py \
  --verify-existing /path/to/copied/hkde-bank --portable \
  --report /path/to/new/hkde-mac-verification.json
```

Repeat for the two Bergomi directories. The report path must be new and outside
the bank. Without `--portable`, the checker requires the recorded environment
and exact arrays; it never overwrites the bank. A source mismatch fails rather
than silently validating a different simulator.

## Next bounded milestone

Proceed to the existing two-parameter HKDE Q task: first build and refine its
two-dimensional posterior-grid reference, then train one small bank-backed
baseline and inspect conditional, joint, data-updating and downstream-price
checks. Keep validation cases and a fresh final test separate. The Bergomi
banks remain available as declared numerical pilots; tighter coupled-grid
analysis is recorded in [next analysis](simulator-analysis-next-steps.md).

The returns-versus-options curriculum remains open. Heston's unresolved
conditional inference checks remain recorded. This run completes another
infrastructure step toward the simulator ladder, not M5 posterior acceptance
or M8 participant delivery. Everything here is synthetic **S**; optional
market application **M** and empirical evaluation **E** remain separate.
