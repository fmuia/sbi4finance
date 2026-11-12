# Heston summary and flexible-input development

This separate experiment extends the actual63 joint-`c` notebook in PR #89.
The original notebook, producers, bank and checkpoints retain their contracts.
The experiment plan is [here](../../../../plans/HESTON-FLEXIBLE-INPUTS-PLAN.md).

[Open the self-contained comparison report](report.html). Its portable
[study index](study.json) also loads in the experimental notebook without private
bank or model files. Final and validation-minimum endpoints remain separately
labelled. Only aggregate metrics and source hashes are included.

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

Append completed dimensions to the portable report from `lab/` with:

```sh
pixi run -e core-reference python scripts/heston_flexible_report.py \
  --run "$HESTON_EXPERIMENT_ROOT/mlp32-seed9011" \
  --out evidence/pilot-results/2026-11-12/heston-flexible-inputs
```

Repeat `--run` for additional completed dimensions. The exporter checks bank,
population, reference, model and draw identities, preserves endpoint roles,
and rejects changed evidence for an existing endpoint. It performs no training,
pricing or sampling.

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

## First full-size result and current sequence

The 32-dimensional MLP completed all 300 epochs on the 300k stored noisy
observations in 37.8 minutes. Its declared final checkpoint and validation
minimum (epoch 258) both completed their evaluations. Independent reconstruction
from the saved raw draws reproduces the metrics. The bank, frozen ZCA, priors,
noise, seed, posterior head, optimizer schedule, 351,600 updates and 90 million
presentations match the direct baseline.

| Reused development population, primary endpoint | Direct 63 inputs | MLP summary 32 |
| --- | ---: | ---: |
| Six marginal SBC screens passed | 5/6 | 6/6 |
| Initial-variance nominal-90% coverage | 97.46% | 96.29% |
| `c` nominal-90% coverage | 90.23% | 89.65% |
| `c` median-estimate RMSE | 0.13494 | 0.13461 |
| Raw posterior draws outside prior | 3.043% | 2.324% |

Pooled recovery is reasonably preserved. Conditional fidelity is not established:
MLP32 synthetic physical intervals remain 1.02–2.29 times their MCMC widths,
with material mean errors. All three complete conditional panels still fail the
declared criteria. The market comparison worsens, including a `c` interval about
99 times its boundary-concentrated reference width versus about 59.5 for the
direct network. The validation-minimum checkpoint does not resolve these errors.
Passing population screens is not a sufficient-statistic result.

The user's clarified priority is a clean **32 / 16 / 8 / 4 / 64** fixed-grid MLP
comparison, prioritizing the smaller dimensions to locate performance loss. At
the launch snapshot in [campaign.json](campaign.json), 16 dimensions was running;
8 and 4 follow automatically, then a full 64-dimensional control. All use the same
300k rows, 300 epochs, seed
9011, two 128-wide SiLU summary layers and unchanged posterior/training settings.
Each fit has a one-hour training cap and separate final/minimum evaluations.
One early 64-dimensional launch was stopped while priorities were clarified;
its partial outputs remain private and are excluded from endpoint comparisons.

**SetTransformer and masking training are held until this comparison is assessed.**
The existing design previews remain available. No Bergomi allocation is changed.
