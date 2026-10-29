# Load the tutorial banks and train your own estimator

Use the saved simulation banks to train or compare estimators without running
the financial simulators again. The three current banks contain complete
63-quote surfaces for Heston, rough Bergomi with a supplied known curve, and
rough Bergomi with jointly inferred curve coefficients.

Download the versioned archives from the
[29 October numerical-backbone release](https://github.com/fmuia/sbi4finance/releases/tag/tutorial-backbones-2026-10-29).
The [artifact inventory](pilot-results/2026-10-29/local-backbone-transfer/README.md)
records their verified identities. Download only the bank you need; the separate
38 MB pretrained-and-cases ZIP supports inference/review without a training bank.
The repository contains code, compact examples and provenance; release assets
contain the full payloads. A Google Drive mirror of the identical versioned files
is being arranged; its link will be added once the destination is confirmed. These are synthetic research data, and the release
retains the recorded numerical and scientific limitations.

For example, with the GitHub CLI installed, download Heston and verify it:

```sh
mkdir -p /absolute/path/to/sbi-bank-downloads
cd /absolute/path/to/sbi-bank-downloads
gh release download tutorial-backbones-2026-10-29 --repo fmuia/sbi4finance \
  --pattern sbi4finance-bank-heston-100k-20261029.tar.gz --pattern SHA256SUMS
python3 - <<'PY'
import hashlib
from pathlib import Path
name = "sbi4finance-bank-heston-100k-20261029.tar.gz"
expected = next(line.split()[0] for line in Path("SHA256SUMS").read_text().splitlines()
                if line.split()[1] == name)
hasher = hashlib.sha256()
with Path(name).open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
actual = hasher.hexdigest()
if actual != expected:
    raise SystemExit("Archive checksum differs; do not extract or use it")
print("SHA256 verified:", name)
PY
mkdir -p ../sbi-banks
tar -xzf sbi4finance-bank-heston-100k-20261029.tar.gz -C ../sbi-banks
```

Browser downloads work too. The other bank filenames are
`sbi4finance-bank-rough-known-100k-20261029.tar.gz` and
`sbi4finance-bank-rough-joint-100k-20261029.tar.gz`; substitute the chosen name
in the download, verification and extraction commands. Extract all banks into
the same `sbi-banks` directory if you download more than one.

Snapshot directories under that common extraction root are:

| Bank | Snapshot directory |
| --- | --- |
| Heston | `sbi4finance-heston-backbone-20261027/snapshot-100k` |
| Rough, known curve | `sbi4finance-rough-known-backbone-20261027/snapshot-100k` |
| Rough, joint curve | `sbi4finance-native-bank-20261027/snapshot-100k` |

| Bank | Training surfaces | Validation surfaces | Reserved test surfaces | Compressed archive size |
| --- | ---: | ---: | ---: | ---: |
| Heston | 100,000 | 5,000 | 5,000 | 122,873,069 bytes (~123 MB) |
| Rough, known curve | 100,000 | 5,000 | 7,000 | 899,951,783 bytes (~900 MB) |
| Rough, joint curve | 100,000 | 5,000 | 7,000 | 933,622,627 bytes (~934 MB) |

The archives retain all native arrays and splits, including reserved test
payloads. **Use training and validation for development; do not use test for
current model selection.** Copying or checking file hashes does not evaluate
the held-out outcomes. For your own experiment splits, subdivide the training
pool and keep the original surface IDs with every derived row.

## 1. Install the locked environment and keep the archive layout

Run commands from the repository root:

```sh
pixi install --locked -e core
pixi run --locked -e core core-check
export SBI_BANK=/absolute/path/to/extracted/snapshot-100k
export SBI_RUN=/absolute/path/to/new-training-run
```

`SBI_BANK` is the directory containing the chosen snapshot's `manifest.json`,
not the archive file, collection directory or model checkpoint. A snapshot is
a small manifest pointing to a collection's `collection.json` and immutable
`parts/*/{manifest.json,arrays.npz}`. Extract the complete archive and preserve
its relative directory layout. Moving the whole extracted tree is fine;
moving only `manifest.json` or rewriting its paths breaks the original hash
binding. Compare the archive's SHA256 with the release's `SHA256SUMS` before
extracting it; the native loader also checks referenced part identities.

Use `core` on a CPU/Mac. For a supported NVIDIA Linux GPU use
`pixi install --locked -e core-gpu` and select `-e core-gpu` below, inside an
allocated GPU job. Both use BayesFlow **2.1.0**, Keras 3.15.1 and JAX 0.11.1.
The repository's archived default environment uses BayesFlow 2.0.14 and is not
the environment for these current estimators. See the
[environment and cluster instructions](full-surface-calibration.md).

## 2. Inspect one part, then load only what you need

This executable example reads one verified training part. It imports no
BayesFlow/JAX model and performs no pricing or training:

```sh
pixi run --locked -e core python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from sbi4finance.calibration import bank_collection as banks

snapshot = Path(os.environ["SBI_BANK"])
manifest = json.loads((snapshot / "manifest.json").read_text())
task = manifest["configuration"]["task"]
known_curve = task["task_id"] == "rough_calibration_v1"
roles = ("observations", "reference_prices", "parameters", "example_id")
if known_curve:
    roles += ("context_values",)

part = next(banks.iter_snapshot(snapshot, "train", roles=roles))
print("task:", task["task_id"])
print("parameters:", task["parameter_names"])
print("snapshot SHA256:", hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest())
print("collection:", manifest["collection_id"])
print("rows:", part["example_id"][0], "...", part["example_id"][-1])
for name, values in part.items():
    print(name, values.shape, values.dtype)
PY
```

To materialize complete development splits in Python, use the same `snapshot`
and `roles` variables:

```python
manifest, splits = banks.load_bank(
    snapshot,
    splits=("train", "validation"),
    roles=roles,
)
train = splits["train"]
validation = splits["validation"]
```

Always specify `splits`: omitting it loads test too. Always select `roles`:
omitting them also decodes large covariance tensors that posterior training
does not need. `load_bank` preallocates the selected arrays and fills them in
part order; it is an in-memory loader. Use `iter_snapshot` for inspection or
your own part-by-part processing. Projected reads verify the selected part's
file hashes but do not decode unrequested NPZ members. The existing trainer
already selects its required roles and never opens test payloads.

## 3. Understand one training example

An example is one whole surface, not one independent row per option. Contracts
are ordered by maturity then strike: seven maturities (5, 10, 21, 42, 63, 126
and 252 trading days) times nine strikes. There are 252 trading days per year.
Spot is one and carry is zero; prices and quote-noise standard deviations are
in normalized forward-price units, not dollar or implied-volatility units.
Use the exact contract order, priors and noise vector in the task metadata.

| Array | Shape for N surfaces | Meaning |
| --- | --- | --- |
| `observations` | `[N,63,1]` | One saved noisy observed surface per example |
| `reference_prices` | `[N,63]` | Numerical price estimates before adding quote noise |
| `parameters` | `[N,P]` | Supervised physical parameters, in `task["parameter_names"]` order |
| `example_id` | `[N]` | Stable IDs such as `train:42`; preserve them in derived data |
| `context_values` | `[N,7]` for rough; `[N,0]` for Heston | Supplied known curve only in the known-curve route; auxiliary labels in joint inference |
| `pricing_standard_errors` | `[N,63]` | Estimated MC error of the numerical price means; Heston has zeros |
| `pricing_covariance` | `[N,63,63]` | MC covariance of those price means, not observation-noise covariance |

The core task boundaries are:

| Route | P and target order | Curve conditioning |
| --- | --- | --- |
| Heston | 4: `initial_variance, theta, vol_of_vol, rho` | No supplied curve; kappa and zero jump intensity are fixed by the task |
| Rough, known curve | 3: `hurst, eta, rho` | Positive `context_values[N,7]` are known per case and vary across training |
| Rough, joint curve | 6: `hurst, eta, rho, short_variance, long_variance, decay` | Infer the curve coefficients; **do not feed the auxiliary seven curve levels to the network** |

Variance levels are annualized variances: 0.04 corresponds to 20% volatility.
Joint decay is an inverse-year rate, and its initial curve is
`xi(t) = long + (short - long) * exp(-decay * t)`. Stored joint curve levels
are interval averages derived from these parameters. The numerical pricer
uses the analytic curve on its fine grid; the seven levels are not independent
observations. The shared trainer and inference adapter enforce this known/joint
distinction automatically.

**Do not add noise twice.** Saved observations are `reference_prices` plus
independent Gaussian quote errors with the fixed per-contract sigma in the
task. The posterior trainer starts from `reference_prices`, divides by sigma,
then adds fresh standard-normal noise to each training batch. It uses the
saved noisy observations for validation and inference. The known curve stays
fixed during that noise augmentation. Prices are not clipped. Numerical MC
uncertainty and grid bias are distinct from this deliberately added quote
noise; `reference_prices` are not exact market truth.

## 4. Train and evaluate with the existing scripts

The same real trainer handles all three banks by reading the task. A small
CPU **software smoke fit** is:

```sh
JAX_PLATFORMS=cpu pixi run --locked -e core python scripts/train_full_surface.py \
  --bank "$SBI_BANK" --out "$SBI_RUN-cpu-smoke" \
  --family flow_matching --encoder vector --widths 16,16 \
  --epochs 2 --batch-size 256 --chunk-epochs 1 --seed 0 \
  --checkpoint-selection final
```

This uses the supplied training/validation rows with a deliberately small
network and schedule; it does not establish inference quality. It still loads
the complete selected splits. A full baseline recipe, intended for an allocated
GPU job, is:

```sh
pixi run --locked -e core-gpu core-check --require-gpu
pixi run --locked -e core-gpu python scripts/train_full_surface.py \
  --bank "$SBI_BANK" --out "$SBI_RUN" \
  --family flow_matching --encoder vector --widths 256,256,256,256 \
  --epochs 500 --batch-size 256 --chunk-epochs 50 --seed 0 \
  --checkpoint-selection final
```

Omitting `--flow-steps` keeps BayesFlow's native adaptive Tsit5 posterior
sampler. That solver is part of the inference network, not the financial
simulator. For CPU training, the same command works with `-e core` and
`JAX_PLATFORMS=cpu`; the full CPU training duration has not been benchmarked.
The [local inference measurements](pilot-results/2026-10-29/laptop-inference-benchmark/README.md)
measure pretrained sampling rather than full training.

Evaluate a declared development subset explicitly:

```sh
pixi run --locked -e core python scripts/evaluate_full_surface.py \
  --bank "$SBI_BANK" --checkpoint "$SBI_RUN" \
  --out "$SBI_RUN-validation" --split validation \
  --first 0 --surfaces 100 --draws 1000 --batch-size 10 --seed 0
```

For architecture experiments, the trainer also accepts `--family affine`,
`spline` or `mean_flow`, and `--encoder deepset` or `set_transformer`. Use a new
output directory when changing the recipe. These choices do not add
missing-quote support; the current core inputs remain complete surfaces.

## Reproducibility and what to share

Keep the original archive and manifest hashes, task, collection ID and stable
surface IDs. The bank root seed/prior/pricing/noise streams identify existing
simulation rows; the trainer's `--seed` controls network training and fresh
quote augmentation. Changing a training seed does not require another bank.
Never move noisy copies of one base surface into different experiment splits.

An interrupted fit can resume with the **identical command, source, environment
and bank** in its original output directory. The trainer saves committed
optimizer and noise-RNG state. A larger bank snapshot or changed architecture,
schedule or source gets a new run directory; it is not an exact resume of the
old fit. Preserve `training.json`, `task.json`, model/checkpoint files and the
declared validation results when sharing a fit. Freeze a model and evaluation
criteria before using reserved test outcomes.

A **bank** supplies examples for retraining. A **posterior checkpoint**
(`model.keras`, `task.json`, `training.json`) supplies fast inference without
the bank. A **forward-pricer checkpoint** is a separate deterministic map from
parameters to numerical prices. Its existing training entry point is
[`scripts/train_full_surface_pricer.py`](../scripts/train_full_surface_pricer.py);
it learns `reference_prices`, with no observation-noise augmentation, and does
not replace the posterior estimator. The
[full-surface workflow guide](full-surface-calibration.md) covers that route
and optional PyMC references. The numerical generation task and independent
pricing checks keep their recorded scope when a bank or model is copied.
