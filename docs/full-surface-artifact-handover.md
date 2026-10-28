# Loading the new tutorial artifacts

The new Heston model is BayesFlow 2.1's **100k-training-surface final epoch 500**, not the older course model or the same run's minimum-loss epoch 402. Its model SHA-256 is `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`.

It passed the frozen 2,000-case marginal calibration gate and the recorded analytic variance-product comparisons. This supports one candidate; it is not full tutorial, training-seed, path-product or market acceptance. Historical checkpoints and acceptance files remain unchanged.

## Small offline cache

The initial cache at `data/synthetic/full-surface-heston-final500/validation-0/` contains one previously inspected development observation, 10,000 neural draws, four-chain PyMC reference draws and paired analytic variance products. It is a reproducible integration fixture, **not the new interior posterior showcase**. The new interior case will receive its own cache after its reference and sampling checks finish.

From the repository root:

```bash
pixi run --locked -e collaboration python - <<'PY'
from sbi4finance.calibration.full_surface_teaching import load_cache
task, report, arrays = load_cache("data/synthetic/full-surface-heston-final500/validation-0")
print(report["case_id"], task["parameter_names"])
print(arrays["observations"].shape)      # (63,), prices in spot-one forward units
print(arrays["neural_draws"].shape)     # (10000, 4), joint parameter rows
print(arrays["reference_chains"].shape) # (4, 8000, 4), chain axes retained
PY
```

The loader uses NumPy and file hashes; it does not import BayesFlow, Keras, JAX or SciPy, initialize a model, price options, or access the raw bank. Missing/corrupt files fail explicitly. Optional expected task and checkpoint identities can be supplied when connecting a live model:

```python
task, report, arrays = load_cache(
    "data/synthetic/full-surface-heston-final500/validation-0",
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
    "data/synthetic/full-surface-heston-final500/validation-0",
    expected_task=task, expected_checkpoint_sha256=record["checkpoint_sha256"],
)
draws = full_surface.sample(workflow, task, cached["observations"][None, :], 1000, seed=0)
print(draws.shape) # (1, 1000, 4)
PY
```

This is an explicit instructor/collaborator command; opening the cache does not run it. Sampling uses BayesFlow's native adaptive solver. A changed observation requires fresh inference: do not reuse cached posterior draws for edited prices.

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

## The rough lessons

The known-curve route will supply seven curve levels alongside the 63 quotes and infer H, eta and rho. The joint route will infer six parameters and must not receive its generating curve as input. Their full tutorial bundles remain pending the ongoing bank, estimator and independent reference checks. The Heston loader deliberately rejects those tasks rather than silently accepting an incomplete rough bundle.

The current notebook loaders still point to the historical course artifacts. Connecting these new bundles is a separate, explicit integration change; this handover does not overwrite them. See the [collaborator overview](collaborator-numerical-status.md) for overall status and the [work order](tutorial-backbone-work-order.md) for active numerical work.
