# Small synthetic bank and training round trip

This implements local **M2.1–M2.3**, the second of ten envisioned simulator-plan PRs. It builds on the [GBM reference](gbm-reference.md). Three NPZ files, a JSON manifest and the existing BayesFlow `fit_offline` route are enough here. No registry, disk dataset framework or cluster orchestrator is introduced.

## Run

```bash
pixi install --locked
pixi run test-simulators
pixi run gbm-bank --out /tmp/gbm-bank
pixi run gbm-train --bank /tmp/gbm-bank --out /tmp/gbm-trained
```

Choose new output directories. The [bank configuration](../configs/tasks/gbm_bank.json) generates 8,192 training histories, 1,024 validation histories and 256 independent test histories. Each has 252 decimal log returns under P, known volatility 0.2, and a proportional drift drawn from `Normal(0, 0.1²)`. All dates/time intervals are synthetic; no actual-market files or provider access are used. Other sizes and task settings require an explicit configuration and retraining.

`generate_gbm_bank.py` writes and verifies the bank, then reports generation/write time, read time and array memory. `train_gbm.py` loads it into memory, trains a small flow-matching network for 40 epochs, reloads the best validation checkpoint, samples the held-out histories and compares with the analytic posterior. The checkpoint includes the adapter and training normalization. Output also includes `report.json` and `reference_checks.npz`; no generated banks/checkpoints are committed.

## Scientific and array boundary

For fixed known volatility and elapsed time, the sum of log returns is sufficient for the constant GBM drift. The neural network therefore receives one observed statistic, `sum_log_returns`, and learns the distribution of `mu`. It is trained on sampled generating drifts, **not analytic posterior answers**. The analytic posterior is reserved for evaluation. Full raw histories remain in the bank; this reduction does not transfer to SVJ or rough models.

Each split contains:

| Array | Role |
|---|---|
| `observations [N,T,1]` | Complete decimal log returns; observed conditioning |
| `dt_years [T]` | Known, positive year fractions; fixed design shared by the bank |
| `parameters [N,1]` | Synthetic drift truth; supervised target only |
| `example_id [N]` | Auxiliary, split-qualified base identity, e.g. `test:12`; never conditioning |

`conditioning_data` explicitly takes only observations, time intervals and the declared task. It rejects a different time design. `training_data` separately adds the supervised target. No ID, hidden diagnostic, future return or generating truth is passed as a condition. Neural normalization is fitted on training batches; neither held-out rows nor individual observed windows define normalization.

The checkpoint is specific to the saved task, including measure, prior, known volatility and observation design. Before later application, read its report and match these conventions; a parser accepting an array does not establish compatibility. A raw Keras checkpoint alone does not carry the task metadata, so retain `report.json` beside it. The data workstream's `LocalArtifact` can supply decimal returns and `dt_years` explicitly; this PR neither imports its branches nor claims completed D2 integration.

## Reproducibility and failure behavior

An identity consists of the task fingerprint, root seed, split and local example ID. PCG64 streams are derived separately for prior sampling and process shocks. Regenerating selected IDs in another order or batch gives identical arrays in the recorded environment; increasing split sizes preserves existing examples. Every history here is its own independent base group. Later quote augmentations/overlapping windows require their own grouped splitting rules.

The manifest records the configuration and checksum, task fingerprint, source-file hashes, NumPy/Python versions, origin, purpose, P measure, conditioning cutoff, role/truth/evidence declarations and file checksums. Reproduction is scoped to the recorded implementation/backend; it is not a cross-version floating-point guarantee.

The writer refuses an existing output directory. It renames each finished array file and publishes `manifest.json` last. The reader rejects incomplete banks, corrupted files, missing/misassigned IDs, invalid values/shapes and inconsistent configuration. A failed simulation leaves `failure.json` identifying the failed example; it never replaces that draw. Diagnose with `generate_examples(config, split, [id])`, then create a new bank. Generalized resume machinery and distributed writers are deferred.

## Local evidence and limits

All 16 reference, bank and inference tests pass in the locked environment. The full synthetic pipeline was also run from an isolated source/configuration copy containing no market fixtures, with Python network connections blocked. Focused checks cover stable examples across batching/reordering/count changes, corruption/incomplete writes, independent splits, explicit conditioning selection, training-only normalization, and identical seeded draws from independently reloaded checkpoints.

The 19.6 MB array bank took about 0.36 s to generate/write and 0.012 s to verify/load in the isolated local run. CPU training took about 2.1 s, with 256 × 4,096 posterior draws taking about 16.6 s. Those times include the respective JAX compilation, exclude Python/framework import startup, and are single-machine measurements. Sampling 16 cases per batch kept peak process memory near 1.3 GB; the framework and inference intermediates dominate the small bank. No custom training-data streaming is needed.

On these held-out cases, mean error was about **0.045 reference posterior SD RMS** (maximum 0.162), and median absolute SD error about **2.0%**. The predeclared conditional-moment pilot limits are RMS mean error 0.10 SD, maximum mean error 0.30 SD and median SD error 10%. Same-seed reload draws match exactly.

The empirical coverage of nominal 90% intervals was **84.4%**, versus **86.3% for the exact posterior on those same 256 cases**; nominal binomial standard error is about 1.9 percentage points. The inferred intervals contained an average 89.1% exact posterior probability. These are reported observations, not a full SBC/calibration certificate. The small bank round trip and conditional-moment pilot pass; broader calibration, tail/function checks and selected tutorial acceptance remain later work. No tuning on this held-out test bank is claimed.

M2.4 is next: run the same CPU generator on the actual cluster and verify selected IDs and resource use. It is not satisfied by local tests or a job template. The returns-versus-options teaching choice remains open; market acquisition cannot block this S path.
