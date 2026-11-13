# Heston summary and flexible-input development

This separate experiment extends the actual63 joint-`c` notebook in PR #89.
The original notebook, producers, bank and checkpoints retain their contracts.
The experiment plan is [here](../../../../plans/HESTON-FLEXIBLE-INPUTS-PLAN.md).

[Open the self-contained comparison report](report.html). Its portable
[study index](study.json) also loads in the experimental notebook without private
bank or model files. Final and validation-minimum endpoints remain separately
labelled. Portable metrics, source hashes and rendered recovery plots are included;
raw posterior samples and simulated parameter arrays stay outside Git.

## Standalone experiment lesson

`lab/notebooks/heston_summary_experiments.py` gives the explanatory rundown intended
for later harvesting into the SPX tutorial: task and priors, whitening versus
compression, the matched recipe, interactive recovery/calibration comparisons,
conditional references and checkpoint selection. A rounded-box schematic follows
the existing Heston notebook artwork and contrasts the jointly trained MLP route
with the direct baseline. A model selector displays all six true-versus-estimated
parameter recovery panels, including posterior medians and central 90% intervals
for all 512 paired cases. All seven final models use the same plot limits.

The lesson reads only the checked-in study and recovery images, and imports no
pricing or neural runtime. The [recovery manifest](recovery/manifest.json) binds
each figure to its final-checkpoint evaluation and raw-draw hashes.
`lab/scripts/export_heston_summary_recovery.py` regenerates those images from the
original private payloads after reproducing recovery metrics and support counts. No bank, model download
or environment-variable configuration is required.

From `lab/`, launch it with:

```sh
pixi run -e core-reference marimo edit notebooks/heston_summary_experiments.py
```

The broader experiment dashboard in `heston_flexible_inputs.py` remains separate.
Its masking and geometry controls are still previews. The original SPX notebook
is not modified by the new lesson.

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

## Intermediate dimensions 7, 6 and 5: running

After inspecting the parameter-recovery plots, the user requested all three
intermediate integer dimensions in the 8-to-4 bracket. A background campaign trains **7 locally and 6/5 on separate Oscar GPUs**,
with the same 300k observations, frozen ZCA,
seed 9011, two 128-wide SiLU layers, unchanged posterior head and 300 epochs.
GPU checkpoints return unchanged for evaluation against the local reference panel.
CPU/GPU kernels can change optimization numerics; backend provenance is shown in
the lesson and a stable cutoff needs common-backend and seed confirmation.
Each fit retains final-300 and validation-minimum evaluations; no recovery result
is assumed in advance. Checkpoints and raw diagnostics stay outside Git. The
[campaign record](campaign.json) binds the launch and preserved recipe.

The completed-six-size plots remain the available evidence until those extra
endpoints finish. An observed cutoff is specific to these fitted models and
this development population; even testing every integer does not prove an exact
minimal sufficient statistic or a reproducible threshold across seeds.

## Completed initial screen and dimension-two extension

The 32-dimensional MLP completed all 300 epochs on the 300k stored noisy
observations in 37.8 minutes. Its declared final checkpoint and validation
minimum (epoch 258) both completed their evaluations. Independent reconstruction
from the saved raw draws reproduces the metrics. The bank, frozen ZCA, priors,
noise, seed, posterior head, optimizer schedule, 351,600 updates and 90 million
presentations match the direct baseline.

All six fits and all twelve checkpoint evaluations completed successfully. The
following table uses final epoch 300 only, on the same reused 512-case population.

| Representation | SBC screens | `c` RMSE | Initial-variance RMSE | Median `c` 90% width | `c` 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct 63 inputs | 5/6 | 0.13494 | 0.000111 | 0.4343 | 90.23% |
| MLP 64 | 5/6 | 0.13506 | 0.000121 | 0.4503 | 89.45% |
| MLP 32 | 6/6 | 0.13461 | 0.000128 | 0.4527 | 89.65% |
| MLP 16 | 5/6 | 0.13666 | 0.000126 | 0.4605 | 89.45% |
| MLP 8 | 6/6 | 0.13593 | 0.000131 | 0.4494 | 90.04% |
| MLP 4 | 6/6 | 0.43967 | 0.007525 | 1.3472 | 90.82% |
| MLP 2 | 6/6 | 0.43945 | 0.018809 | 1.3505 | 90.63% |

**The observed deterioration is between 8 and 4 dimensions in this recipe.**
Eight is the smallest tested representation preserving similar pooled recovery
to the direct baseline. Four sharply degrades physical recovery and returns
nearly prior-width uncertainty for `c` (the prior central-90% width is 1.35),
despite passing all six marginal SBC screens. This illustrates why calibration
alone does not establish that an estimator retains the observations' information.
One seed per dimension does not establish a reproducible minimal dimension or
a sufficient statistic; follow-up seed confirmation is needed.

Pooled recovery is reasonably preserved for dimensions 8–64. Conditional fidelity is not established:
MLP32 synthetic physical intervals remain 1.02–2.29 times their MCMC widths,
with material mean errors. All three complete conditional panels still fail the
declared criteria. The market comparison worsens, including a `c` interval about
99 times its boundary-concentrated reference width versus about 59.5 for the
direct network. The validation-minimum checkpoint does not resolve these errors.
Passing population screens is not a sufficient-statistic result.

The user's clean **32 / 16 / 8 / 4 / 64** fixed-grid MLP sequence is complete;
[campaign.json](campaign.json) retains its launch snapshot and final accounting.
All use the same 300k rows, 300 epochs, seed
9011, two 128-wide SiLU summary layers and unchanged posterior/training settings.
Each fit has a one-hour training cap and separate final/minimum evaluations.
One early 64-dimensional launch was stopped while priorities were clarified;
its partial outputs remain private and are excluded from endpoint comparisons.

Conditional fidelity remains insufficient for every size: all three complete
reference panels fail the declared criteria. Smaller-summary pooled recovery
should not be described as absolute posterior correctness. All validation-minimum
outcomes remain available separately in the report and notebook.

**SetTransformer and masking training remain held while this comparison is assessed.**
The existing design previews remain available. No Bergomi allocation is changed.

## Completed dimension-two extension

The two-dimensional summary completed the identical 300k/300-epoch recipe in
26.7 minutes; training, both endpoint evaluations and report export took about
31 minutes overall. Its validation minimum was epoch 218.

The primary `c` RMSE is 0.43945 and its median 90% width is 1.35048, essentially
the prior's 1.35 width. Initial-variance RMSE rises to 0.01881, about 143 times
the dimension-8 value. The validation-minimum checkpoint retains the deterioration.
The final checkpoint nevertheless passes all six marginal SBC screens, showing
why calibration alone does not establish information retention.

This strengthens the severe-compression example without proving a universal
phase transition or failure of every size below 8: dimension 6 is untested and
independent-seed confirmation remains pending. Masking stays on hold.
