# CPU cluster bank smoke

This is **3/10 of the simulator-plan series**, implementing M2.4 after the [local bank/training round trip](gbm-bank-training.md). One short CPU job tests the real source → scheduler → storage → verified-bank path. It does not run neural training, require market observations or measure scaling.

## What is needed

1. An authenticated Oscar SSH session and a currently permitted CPU account/partition/QOS. Run `python3 cluster/discover_cluster.py --ssh-host oscar`; `cluster/oscar.local.json` is personal and stays untracked.
2. A source checkout or committed source archive, including the GBM modules, task configuration, generation/checking scripts and CPU job script.
3. A Python 3.12 interpreter with NumPy 2.5.3. The generator imports only NumPy and the standard library. An existing locked default Pixi environment can supply the interpreter; alternatively, a small separate venv suffices for this bank-only smoke. The latter is not a validation of the full Linux BayesFlow environment.
4. Writable scratch/data storage outside the home quota, with room for the small environment and approximately 20 MB bank. Choose a **new** output directory; existing banks are never overwritten.

The batch script requests one task, one CPU, 2 GB RAM and five minutes. BLAS/OMP thread counts are set to one. Resource/account/QOS flags can be changed explicitly after discovery; personal paths and allocation discoveries are not committed.

## Submit and inspect

Create a minimal environment on a login node if an appropriate one is not already available. For the measured run, `python/3.12.8-zr3c` was an available Oscar module; check available modules before reusing that name. Set `SMOKE_ROOT` to your own writable location.

```bash
module load python/3.12.8-zr3c
python3 -m venv "$SMOKE_ROOT/venv"
"$SMOKE_ROOT/venv/bin/python" -m pip install --no-cache-dir numpy==2.5.3

# From the source root, with SMOKE_ACCOUNT/PARTITION/QOS from discovery:
sbatch --parsable --account="$SMOKE_ACCOUNT" --partition="$SMOKE_PARTITION" --qos="$SMOKE_QOS" \
  --output="$SMOKE_ROOT/slurm-%j.out" \
  cluster/simulate.sbatch "$SMOKE_ROOT/venv/bin/python" "$SMOKE_ROOT/new-bank"

# SMOKE_JOB_ID is the submitted ID. Check the job and its .batch accounting row.
sacct -j "$SMOKE_JOB_ID" --parsable2 \
  --format=JobIDRaw,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocCPUS,ReqMem,NodeList
```

Load the required Python module before submission so Slurm inherits its runtime environment. The interpreter argument can instead point to the existing `.pixi/envs/default/bin/python`; compute jobs do not install packages. A third script argument optionally selects a different explicit bank configuration.

The job runs the same `generate_gbm_bank.py` as the laptop, then writes `verification.json` by calling `check_gbm_bank.py`. The generator verifies every saved split's checksum, shape, IDs, context and configuration. The additional checker requires the recorded generator source hashes and Python/NumPy versions to match, then exactly regenerates the first, middle and last example of each split in reverse order. Any failed check makes the job fail. Retain the bank manifest, verification file, scheduler log and accounting output with the source revision.

For a bank copied back from Linux to a different platform:

```bash
# Verify file transfer hashes against the remote manifest/verification first.
pixi run python scripts/check_gbm_bank.py --bank /path/to/copied-bank --portable
```

Portable mode still checks file/config integrity and generator source hashes. It records environment differences and uses `rtol=1e-12, atol=1e-14` for this GBM pilot's float arrays, while requiring exact IDs. Its report includes actual maximum differences and whether arrays happened to match exactly. Independently generated NPZ archive hashes are not a cross-platform scientific comparison. Same-environment exact regeneration and cross-platform numerical agreement are separate checks.

## Measured run

The actual run on **11 October 2026** passed, establishing M2.4 for this GBM bank workflow.

| Item | Observed result |
|---|---|
| Submitted source | `88673583b1b36f25a30af2f3531fbb0a8e376ed6`; source/config/scripts only, no market fixtures |
| Source archive SHA256 | `975d376584131ed31f422c46cba587a93abaa2ebf67530030545778d99674dae`, identical before/after upload |
| Slurm job / node | `7292853` / `node1829` |
| CPU lane | `carney-mjfrank-condo2` / `batch` / `carney-condo2` |
| Outcome | Job and batch step `COMPLETED`, exit code `0:0` |
| Allocated resources | 1 CPU, 2 GB RAM, 5-minute limit |
| Measured resources | 4 s elapsed; 2.178 s total CPU; batch MaxRSS `87692K` (about 86 MiB) |
| Environment | Linux x86_64; Python 3.12.8; NumPy 2.5.3; separate small venv |
| Bank | Default config: 8,192 training / 1,024 validation / 256 test histories; 19,574,688 loaded array bytes |
| Timings inside generation command | 1.702 s generation/write; 0.086 s verified load |
| Cluster regeneration | All nine selected examples matched exactly after independent, reordered requests |
| Download integrity | Manifest and verification hashes matched remote values; all NPZ hashes checked against the downloaded manifest |
| Mac comparison | Python 3.12.15 / NumPy 2.5.3, macOS arm64; all nine selected examples also matched exactly (maximum absolute difference 0) |

Bank manifest SHA256: `bc0eb3fb9c44ed5b84372d4875fbdde4500f641b1e01ecc032b60114feea8fd4`. Verification JSON SHA256: `8d4122f2b2bcfbe11c072f861cae18b7bcb8409fd6b0a559ea77ff38dbc33878`. Raw banks/logs remain in personal scratch and the local run directory rather than in Git. The downloaded bank was checked against the same source on the Mac using portable mode; exact agreement happened for these selected examples and is not a guarantee across all platforms/builds.

`bash -n` and focused verifier tests also pass, including source/version mismatches and a changed observation with a recomputed valid file hash. Numerical regeneration catches that change independently of storage integrity. This run validates the small generation/storage/reproduction path. It does **not** establish Linux neural-training compatibility, job-array scaling, large-bank performance or final tutorial calibration. No GPU was requested and no full BayesFlow environment was installed on Oscar for this run.
