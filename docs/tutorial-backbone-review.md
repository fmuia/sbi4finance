# Reviewing the current tutorial backbones

Open `tutorial_final/notebooks/backbone_review.py` to inspect the new Heston and
rough Bergomi artifacts. It defaults to cached observations and posterior
draws, with full corner plots and provenance. Models, raw banks and cluster
access are optional. No notebook was executed while preparing this integration.

From the repository root, use the existing collaborative marimo route:

```bash
pixi run --locked -e collaboration collaborate tutorial_final/notebooks/backbone_review.py
```

For optional fresh inference, open it in the BayesFlow 2.1 core environment:

```bash
pixi run --locked -e core marimo edit tutorial_final/notebooks/backbone_review.py
```

Opening the notebook loads neither a neural backend nor a model. The separate
live section requires a matching local checkpoint directory and an explicit
button press. It generates fresh draws for the exact cached observation; it
does not train, reprice the surface or run MCMC.

## What each example supports

| Example | Inferred parameters | Inference conditions | Cached comparison |
|---|---|---|---|
| Heston final500 | Initial variance, long-run variance, vol-of-vol, correlation | 63 complete quotes, fixed design/noise | Interior development case; 10,000 neural draws, four PyMC chains, paired analytic variance rates |
| Rough known50k | H, eta, rho | 63 complete quotes plus seven observed forward-variance levels | Fixed validation:0 replay; 1,000 saved neural draws, no conditional reference |
| Rough joint20k | H, eta, rho, short variance, long variance, decay | 63 complete quotes; the generating curve is excluded | Fixed validation:0 replay; 1,000 saved neural draws, no conditional reference |

The rough examples are software replay fixtures selected by row identity, with
no search for attractive posteriors. The joint fixture approaches prior edges
in H, rho and decay; the notebook flags this and displays every inferred
parameter. The Heston example is the checked interior showcase. No rough PyMC
agreement or product-price acceptance is implied by these caches.

The catalogue at `data/synthetic/full-surface-backbones/artifacts.json` pins the
cache, task and model hashes. The NumPy-only loader also checks file hashes,
observation identity, parameter dimensions and prior bounds. Joint curve
labels cannot be passed through the rough replay as observed context. Existing
Heston analytic products retain their original per-draw pairing and chain axes.

```python
from sbi4finance.calibration.full_surface_teaching import load_tutorial_artifact

task, provenance, arrays = load_tutorial_artifact(
    "data/synthetic/full-surface-backbones/artifacts.json", "rough-known50k"
)
print(arrays["observations"].shape)   # (63,)
print(arrays["context_values"].shape) # (7,), observed only in the known-curve task
print(arrays["neural_draws"].shape)   # (1000, 3)
```

The rough bundles add about 235 kB of data and metadata. Large models and banks
remain on Oscar. All scientific-acceptance flags remain false; historical
`ACCEPTED.json` files are unchanged. The 2,000-case validation reports preserve
their complete rank and coverage results, including deviations from nominal
coverage and pricer outliers.

## Copying a model for the live role

The catalogue and notebook show the exact Oscar directory for each model. For
example, copy only the known50k model and its matching metadata:

```bash
remote_checkpoint=/oscar/scratch/afengler/sbi4finance-post-bank-baselines-recovery-v2-20261028/known/posterior
local_checkpoint=/tmp/sbi4finance-known50k-final500
mkdir -p "$local_checkpoint"
scp "oscar:$remote_checkpoint/model.keras" \
    "oscar:$remote_checkpoint/task.json" \
    "oscar:$remote_checkpoint/training.json" "$local_checkpoint/"
```

Enter that local directory in the notebook and press its live-inference button.
The loader rejects a checkpoint with the wrong task or model hash. Cached draws
retain their original batched sampling provenance; new draws need not be
bit-identical to that batch. There is no automatic download or silent fallback
to another model.

For Heston, use the exported `checkpoint-final500` directory identified in the
[artifact handover](full-surface-artifact-handover.md), which correctly pairs the
final500 model with derived metadata and preserves the historical selected402
record. Never attach selected402 metadata to the final500 model.

## Known curves with summary networks

The complete-design inference factory now supports vector, DeepSet and
SetTransformer inputs for the known-curve rough task. A set encoder summarizes
63 quote tokens containing scaled price and fixed contract coordinates. The
seven log curve levels remain separate global observed context, concatenated
with the learned quote summary by BayesFlow. They are never parameter targets
or repeated curve labels inside quote tokens.

Fresh quote noise changes the price channel only. Joint-curve inference still
receives no generating curve. Existing vector checkpoint inputs, architecture
defaults and BayesFlow's native adaptive FlowMatching sampler are unchanged.
Tiny actual BayesFlow 2.1 tests cover training, exact seeded save/reload,
permutation invariance through the trained adapter and curve/noise isolation;
these establish software behavior, not posterior quality of the new encoders.

## Compatibility with the proposed repository layout

[PR49](https://github.com/fmuia/sbi4finance/pull/49) is still open. Its verified
head on 2026-10-28 was `3cf59b552b0e1a4fc8050c54d1ce715b4483a014`.
It separates course material, a self-contained tutorial and research code. This
integration does not merge it or move its files.

| Current integration | PR49 migration point |
|---|---|
| `tutorial_final/notebooks/backbone_review.py` | `tutorial/notebooks/backbone_review.py`; notebook links remain relative |
| `sbi4finance.calibration.full_surface_teaching` | NumPy cache reader can be copied into `tutorial/src/sbi4f_tutorial/`; live BF2.1 factory remains an explicit port from `lab/src/sbi4finance/calibration/` |
| `data/synthetic/full-surface-backbones` and Heston cache | Copy selected small bundles together under `tutorial/assets/`; update relative catalogue paths without changing their bytes/hashes |
| `SBI4F_BACKBONE_MANIFEST` | Explicit catalogue override; distinct from PR49's `SBI4F_TUTORIAL_ASSETS` root |
| Root core environment, BayesFlow 2.1 | PR49's independent tutorial environment must explicitly adopt BF2.1 before enabling these live models; historical BF2.0 checkpoints stay separate |

The proposed tutorial's loaders currently read historical accepted study
artifacts. These new development bundles must retain their explicit roles and
limitations when ported; they must not acquire historical acceptance markers.

## Regenerating the small rough replays

This command copies existing outputs only. It fixes the case to validation:0,
checks the matching bank row, and never loads a model or computes new inference:

```bash
pixi run --locked -e core python scripts/prepare_full_surface_teaching.py rough-replay \
  --run-root /path/to/completed/known-or-joint-run \
  --snapshot /path/to/its/immutable-snapshot --out /new/path/replay
```

For joint replay, the builder deliberately omits auxiliary generating curves.
The source hashes, exact task, training record, full validation report and saved
joint draws travel with the cache. Rebuilding is optional; collaborators can use
the committed small replay directly.
