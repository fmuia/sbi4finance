# Loading the new tutorial artifacts

> **Historical operational record (superseded 1 November 2026).** The repository-root
> commands and Oscar/SCP paths below describe the original artifact export, not
> the current collaborator setup. Start with the [tutorial README](../../tutorial/README.md),
> [current data guide](training-data-guide.md) and
> [handover execution plan](../../tutorial/curriculum/TUTORIAL-REVIEW.md#g-collaborator-handover-before-the-next-feedback-round).
> Published native model downloads now exist; retain this record for the original
> checkpoint/reference provenance.


The [backbone review guide](tutorial-backbone-review.md) now connects these
artifacts to an actual marimo notebook with explicit cached and live roles,
alongside fixed known-curve and joint-curve rough development replays.

The new Heston model is BayesFlow 2.1's **100k-training-surface final epoch 500**, not the older course model or the same run's minimum-loss epoch 402. Its model SHA-256 is `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`.

It passed the frozen 2,000-case marginal calibration gate and the recorded analytic variance-product comparisons. This supports one candidate; it is not full tutorial, training-seed, path-product or market acceptance. Historical checkpoints and acceptance files remain unchanged.

## Small offline cache

The primary cache at `data/synthetic/full-surface-heston-final500/interior-midpoint/` contains the new ex-ante interior development observation, 10,000 neural draws, four-chain PyMC reference draws and paired analytic variance products. Its reference convergence, Fourier refinement and registered interior criteria passed. All four central-99% parameter intervals are well inside the prior box. It is a development example, not another reserved-test acceptance campaign.

The earlier `validation-0/` cache remains a separate integration fixture and boundary example; it has not been relabelled as the interior case.

From the repository root:

```bash
pixi run --locked -e collaboration python - <<'PY'
from sbi4finance.calibration.full_surface_teaching import load_cache
task, report, arrays = load_cache("data/synthetic/full-surface-heston-final500/interior-midpoint")
print(report["case_id"], task["parameter_names"])
print(arrays["observations"].shape)      # (63,), prices in spot-one forward units
print(arrays["neural_draws"].shape)     # (10000, 4), joint parameter rows
print(arrays["reference_chains"].shape) # (4, 8000, 4), chain axes retained
PY
```

The loader uses NumPy and file hashes; it does not import BayesFlow, Keras, JAX or SciPy, initialize a model, price options, or access the raw bank. Missing/corrupt files fail explicitly. Optional expected task and checkpoint identities can be supplied when connecting a live model:

```python
task, report, arrays = load_cache(
    "data/synthetic/full-surface-heston-final500/interior-midpoint",
    expected_task=checkpoint_task,
    expected_checkpoint_sha256=checkpoint_record["checkpoint_sha256"],
)
```

The complete ordered 63-price observation is the inference input. Reference chains, true parameters if later supplied, product values and example IDs are not inference conditions. Analytic cached products are annualized variance rates, not payoff draws or path-dependent prices.

## Optional live inference

The large exported model remains on Oscar:

```text
/oscar/scratch/afengler/sbi4finance-heston-teaching-20261028/checkpoint-final500/
```

Copy it to a directory outside the checkout, using your existing Oscar access:

```bash
scp -r oscar:/oscar/scratch/afengler/sbi4finance-heston-teaching-20261028/checkpoint-final500 \
  /tmp/sbi4finance-heston-final500
```

Then use the isolated **core** environment, not the archived BayesFlow 2.0 environment:

```bash
pixi run --locked -e core python - <<'PY'
from sbi4finance.calibration import full_surface
from sbi4finance.calibration.full_surface_teaching import load_cache
task, workflow, record = full_surface.load_checkpoint("/tmp/sbi4finance-heston-final500")
_, _, cached = load_cache(
    "data/synthetic/full-surface-heston-final500/interior-midpoint",
    expected_task=task, expected_checkpoint_sha256=record["checkpoint_sha256"],
)
draws = full_surface.sample(workflow, task, cached["observations"][None, :], 1000, seed=0)
print(draws.shape) # (1, 1000, 4)
PY
```

This is an explicit instructor/collaborator command; opening the cache does not run it. Sampling uses BayesFlow's native adaptive solver. A changed observation requires fresh inference: do not reuse cached posterior draws for edited prices.

An actual CPU roundtrip compared this ordinary exported-checkpoint loader with a direct load of the pinned final-500 model: ten joint draws at the same seed were bit-identical and all model/metadata bytes remained unchanged. [Runtime verification](../evidence/pilot-results/2026-10-28/heston-teaching-export/README.md) records the exact check. It does not promote the candidate to broader scientific acceptance.

## Instructor regeneration

`scripts/prepare_full_surface_teaching.py export-checkpoint` copies the final model into a **new** directory, preserves original `training.json` bytes as `origin-training.json`, and writes a derived loadable record with `exported_candidate_epoch=500` and `original_validation_selected_epoch=402`. The selection label states that this is a post-hoc development-selected candidate frozen before its held-out gate. It creates no `ACCEPTED.json` and leaves scientific-validation flags false. A copied final model must never be paired with unchanged epoch-402 metadata.

```bash
pixi run --locked -e core python scripts/prepare_full_surface_teaching.py export-checkpoint \
  --source /path/to/heston-100k-native/checkpoint --out /new/path/checkpoint-final500 \
  --gate /path/to/acceptance/evaluation.json --product-checks /path/to/products/products.json

pixi run --locked -e core python scripts/prepare_full_surface_teaching.py cache \
  --checkpoint /new/path/checkpoint-final500 --declaration /path/to/frozen/manifest.json \
  --case-index 0 --out /new/path/validation-0
```

The cache builder checks exact source hashes in the declaration. It copies existing development draws rather than sampling, training, pricing or reading the bank. Reference chain structure and parameter/product pairing are checked on replay. No notebook was executed while preparing this handover.

For the new interior case, use its own completed source/run manifest, not the old case0 product evidence:

```bash
pixi run --locked -e core python scripts/prepare_full_surface_teaching.py cache-development \
  --checkpoint /new/path/checkpoint-final500 \
  --case-root /oscar/scratch/afengler/sbi4finance-interior-corners-20261028 \
  --out /new/path/interior-midpoint
```

This copies the completed checked reference and neural output. Its analytic product summaries are **descriptive for this new case**, with no inherited boundary-case quantile-gate verdict. The normal core checkpoint loader and a direct candidate load have also passed an actual local CPU ten-draw same-seed roundtrip; [the verification record](../evidence/pilot-results/2026-10-28/heston-teaching-export/README.md) retains its scope and exact bytes.

## The rough lessons

The known50k route supplies seven observed curve levels alongside the 63 quotes and infers H, eta and rho. The joint20k route infers six parameters and receives no generating curve as input. The new review notebook loads one small fixed development replay from each completed run. These contain saved neural draws and provenance, with no fabricated reference chains or product checks. Independent rough reference, pricing and product acceptance remain pending.

The historical SPY notebooks retain their own artifacts and link to the new review notebook. The cache catalogue pins the exact model/task for an optional live load; it does not overwrite historical checkpoints or acceptance files. See the [review guide](tutorial-backbone-review.md), [collaborator overview](collaborator-numerical-status.md) and [work order](tutorial-backbone-work-order.md).
