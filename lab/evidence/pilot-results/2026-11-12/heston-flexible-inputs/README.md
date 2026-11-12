# Heston summary and flexible-input development

This separate experiment extends the actual63 joint-`c` notebook in PR #89.
The original notebook, producers, bank and checkpoints retain their contracts.
The experiment plan is [here](../../../../plans/HESTON-FLEXIBLE-INPUTS-PLAN.md).

## Implemented first increment

- `lab/notebooks/heston_flexible_inputs.py`: saved-run comparisons, learning curves,
  full-grid posterior sampling, and clearly identified mask/geometry previews.
- `tutorial/.../heston_options/experimental_inference.py`: a serializable learned
  MLP bottleneck, with the same raw six targets and FlowMatching posterior head.
- `lab/scripts/heston_flexible_experiment.py`: verified-bank training, atomic
  progress/history, final and validation-minimum archives, checked reload and
  paired development diagnostics using the original full MCMC chains.
- `tutorial/.../heston_options/flexible_quotes.py`: geometry-bearing quote tokens,
  training-only shared scaling, availability masks, carry interpolation and
  bounded shared-expiry geometry proposals that require actual repricing.

**Trained mask-aware and varying-grid inference are later increments.** Their
notebook controls currently visualize the proposed observation designs. They
never feed incomplete or moved observations to a complete-grid checkpoint.
The initial summary screen is an empirical compression study, not a proof of
sufficiency or scientific acceptance.

## Run and inspect

Obtain the verified actual63 development bundle through the existing
[training-data guide](../../../../plans/training-data-guide.md). Large outputs
remain in a caller-owned directory outside Git. For example, from `lab/`:

```sh
export KERAS_BACKEND=jax
export PYTHONPATH=.:src:../tutorial/src
export MPLCONFIGDIR=/private/tmp/heston-flex-mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

pixi run -e core-reference python scripts/heston_flexible_experiment.py train \
  --bundle "$HESTON_BUNDLE" --out "$HESTON_EXPERIMENT_ROOT/mlp32-seed9011" \
  --rows 300000 --epochs 300 --summary-dim 32 --seconds 3600 --evaluate

pixi run -e core-reference marimo edit notebooks/heston_flexible_inputs.py
```

`HESTON_EXPERIMENT_ROOT` contains a concurrency-safe `study.json` and per-run
directories. The notebook reads this index for private local results. Each run
saves task/preprocessing identities, native versions, source hashes, optimizer
configuration, history, checkpoints and checksums. A new output directory is
required per fit. Intermediate checkpoints preserve inference state; exact
training continuation is not implemented or claimed.

Final-epoch evaluation is primary. A distinct minimum-validation checkpoint is
evaluated separately when `--evaluate` is requested. Every raw posterior draw is
retained: no clipping, rejection or resampling at prior boundaries. The existing
100-row validation and historical 512-case population are development panels;
neither a smoke fit nor repeated use of these panels establishes fresh acceptance.

The first full-size candidate uses a 32-dimensional MLP summary on all 300k
stored noisy observations. It reuses the exact frozen ZCA matrix, five physical
priors, inferred-`c` prior, batch size, 300 epochs and posterior-network settings
of the direct baseline. Further dimensions follow the bounded sequence in the
plan after inspecting the initial result. No Bergomi allocation is changed.
