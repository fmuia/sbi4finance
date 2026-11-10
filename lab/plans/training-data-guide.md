# Load the tutorial banks and train your own estimator

Choose the smallest tier needed for your work. No Oscar login, owner-specific
SSH setup or Drive Desktop mount is needed for any of these routes.

| Goal | Required files | Where they belong |
| --- | --- | --- |
| Read, edit and review the three application lessons | Fresh repository checkout, its shipped small assets and the locked tutorial environment | `tutorial/`; no additional model or bank download |
| Draw fresh approximate posteriors for the packaged observations | The matching native model archive below | Ignored `tutorial/assets/local-checkpoints/`; no training bank |
| Retrain or compare estimators | The selected option or trajectory bank and locked lab environment | A caller-owned directory outside Git; preserve the complete native tree |

The current [participant instructions](../../tutorial/README.md) give the
three-application launch and review sequence. An older exported workshop kit or
HTML preview does not contain subsequent notebook edits; use the agreed Git
revision for collaboration. Model and bank distribution does not change any
recorded numerical or scientific limitation.

## Source and environment for existing banks

Keep the complete current checkout when installing the locked lab environment:
lab workers reuse the numerical runtime under the sibling `tutorial/` directory.
The participant-only notebook dependencies are separate, so the lab can retain
its recorded BayesFlow environment. Runtime editing locations are in the
[tutorial source map](../../tutorial/README.md#author-or-change-a-lesson).

The source-ownership refactor does not rewrite downloaded banks, task files,
checkpoints or their identities. Existing banks can be read and used for a new
fit from the current checkout. Extending an existing collection or resuming its
generation additionally requires its original producer source/runtime identity;
use the matching historical checkout/environment for that operation. A newly
generated bank records the new source identity. Keep native manifests unchanged.

## Access and local paths

The [29 October option-backbone release](https://github.com/fmuia/sbi4finance/releases/tag/tutorial-backbones-2026-10-29)
and its [Drive mirror](https://drive.google.com/drive/folders/1Zez6mSu8Y_B7IIjdRYWJlDxdAl49fTLq)
contain the option archives and `SHA256SUMS`. The [artifact inventory](../evidence/pilot-results/2026-10-29/local-backbone-transfer/README.md)
records their identities. The trajectory archive and banks use the separate
[path-workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k).

This GitHub repository is private. Use your own account with repository read
access; releases inherit that access. Run `gh auth login` if needed. An
unauthenticated 404 does not establish that a release asset is missing. Drive
uses your own separately granted folder access; GitHub and Drive permissions do
not confer access to one another. A browser download is sufficient; neither a
Drive filesystem mount nor an owner's account or SSH key is required. Request
access to the relevant service if needed. Do not regenerate data to repair a
missing or corrupt download.

For the commands below, set these to your own checkout and
download directory, using absolute paths:

```sh
export SBI_CHECKOUT="/absolute/path/to/your/sbi4finance"
export SBI_DOWNLOADS="/absolute/path/to/your/downloads"
mkdir -p "$SBI_DOWNLOADS"
```

## Fixed rough-option diagnostic inputs

The rough notebook includes **200 validation markets**, each with 63 noisy
option prices, seven supplied curve levels and generating parameters for the
checks. They live in
[`tutorial/assets/diagnostics/rough-known-200/`](../../tutorial/assets/diagnostics/rough-known-200/):
`task.json`, `panel.json` and `inputs.npz`. The compressed inputs are about
**305 KB**. This panel contains no posterior draws.

Train the tutorial network or load the matching pretrained network, select it
under **Network to check**, then press **Check this network on the 200 stored
markets**. This draws 200 samples per market and plots native BayesFlow recovery
and rank ECDFs. It neither reprices the options nor retrains. These are fixed
validation cases used for development, not an untouched final test.

On one Apple M5 Max CPU worker, generating all 200 markets took **688.45 s**
wall time and **685.88 CPU-s**. The original path ladder and precision policy
were retained; every quote met MC SE ≤ 0.25 quote SD, with no failed or replaced
markets. Pretrained inference took **33.63 s** on its first call and **33.46 s**
warm; model loading took **0.86 s** and plotting **1.23 s**. Imports took another
**9.38 s** in a fresh process. These are measured local costs, not guarantees
for another machine or evidence of posterior accuracy.

### Optional regeneration

Keep generation outside the notebook. From `lab/`, with the locked `core`
environment installed, choose a new output directory outside Git:

```sh
export SBI_DIAGNOSTIC_OUT="/absolute/path/to/new/rough-diagnostic-shard"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 pixi run --locked -e core python - <<'PY_DIAGNOSTIC_INPUTS'
import json
import os
from pathlib import Path
from sbi4finance.calibration import bank

panel = Path("../tutorial/assets/diagnostics/rough-known-200")
task = json.loads((panel / "task.json").read_text())
config = {
    "task": task,
    "root_seed": 20261110,
    "splits": {"train": 1, "validation": 200, "test": 1},
}
bank.validate_config(config)
record = bank.write_shard(
    Path(os.environ["SBI_DIAGNOSTIC_OUT"]), config, "validation", 0, 200)
print(f"Generated {record['examples']} markets in {record['generation_cpu_seconds']:.1f} CPU-s")
PY_DIAGNOSTIC_INPUTS
```

Only validation IDs 0–199 are generated; the train/test declarations are required
configuration fields. The new shard contains `arrays.npz` and its native
`manifest.json`; it does not overwrite the bundled panel. A pricing failure
stops generation and records the failed ID. Do not discard difficult rows or
loosen the task to make a run pass. This command starts a new shard. After an
interruption, keep the same task, seed and producer environment and rerun into
a new directory. Publish lesson assets through the lab tools.

## Optional native posterior models

Each native model needs its unchanged `model.keras`, `task.json` and
`training.json` together. The tutorial loaders bind the task and model to the
selected cache, preserve its recorded sampler and reject mismatches. Extraction
is into the explicitly ignored download cache; never replace the shipped replay
assets or edit model metadata to make a different checkpoint load.

### Option models: Heston and known/joint-curve rough Bergomi

Download `sbi4finance-pretrained-and-cases-20261029.zip` and `SHA256SUMS` from
the option release or mirror. The ZIP is **38,046,255 bytes**, SHA-256
`8d3a146312efdeaffcf797dee89158a491cde0961689649bbe17b6a3ae4bf51c`.
The CLI alternative, after authenticating with your own GitHub account, is:

```sh
gh release download tutorial-backbones-2026-10-29 --repo fmuia/sbi4finance \
  --dir "$SBI_DOWNLOADS" \
  --pattern sbi4finance-pretrained-and-cases-20261029.zip --pattern SHA256SUMS
python3 - <<'PY_OPTION_HASH'
import hashlib
import os
from pathlib import Path
archive = Path(os.environ["SBI_DOWNLOADS"]) / "sbi4finance-pretrained-and-cases-20261029.zip"
expected = "8d3a146312efdeaffcf797dee89158a491cde0961689649bbe17b6a3ae4bf51c"
hasher = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
actual = hasher.hexdigest()
if actual != expected:
    raise SystemExit("Archive checksum differs; do not extract or use it")
print("SHA256 verified:", archive.name)
PY_OPTION_HASH
mkdir -p "$SBI_CHECKOUT/tutorial/assets/local-checkpoints"
python3 -m zipfile -e "$SBI_DOWNLOADS/sbi4finance-pretrained-and-cases-20261029.zip" \
  "$SBI_CHECKOUT/tutorial/assets/local-checkpoints"
```

The ZIP also contains packaged cases and forward-pricer artifacts. Leave them
in this ignored cache; the checkout supplies its own teaching cases. Use the
`pretrained/` directories below for posterior inference. The rough notebook can
also load the joint forward pricer for its point-calibration example below.

From `tutorial/`, open `pixi run --locked backbones`. Select the cached example
and enter the **absolute** checkpoint directory in **Local matching checkpoint
directory**, then press **Load the matching model and draw 1,000 fresh samples**.
The field is a filesystem path; it does not expand shell variables.

| Selected example | Checkpoint directory below `tutorial/assets/local-checkpoints/` |
| --- | --- |
| `heston-final500` | `pretrained/heston100k-final500/` |
| `rough-known100k-validation0` | `pretrained/rough-known100k-final500/` |
| Any of the four `rough-joint100k-*` examples | `pretrained/rough-joint100k-final500/` |

For example, print the Heston path to paste into the field:

```sh
printf '%s\n' "$SBI_CHECKOUT/tutorial/assets/local-checkpoints/pretrained/heston100k-final500"
```

This optional review action draws a new approximate posterior for the exact
packaged observation. It does not run MCMC, train a model or update the cached
lesson result. The current option estimator is BayesFlow 2.1; archived 2.0
models are separate historical artifacts.

### Joint rough-Bergomi pricer: optional point calibration

The same ZIP contains `pricers/rough-joint100k/`. Keep `pricer.npz`, `task.json`
and `pricer.json` together under
`tutorial/assets/local-checkpoints/pricers/rough-joint100k/`.

In the rough notebook, use **Load the joint pricer and fit this market**.
The default field names that archive location. The loader verifies the selected
task and the model's recorded payload identity; the notebook checks the bank
identity too. It then fits the selected market by local multi-start least
squares and carries that point into a fair variance-swap strike beside the
approximate posterior distribution.

This pricer maps six parameters to expected option prices; it is not a posterior
network. The example runs no joint MCMC reference. Saved posterior propagation
works without the download. No training bank is needed.

### Trajectory models: daily return histories

Download `path-workshop-native-models-20261031.tar.gz` from the
[path-workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k).
It is **6,354,318 bytes**, SHA-256
`261c36550da52948aedcc2680d51bcb756db485626a6a2b8ee918fa9d6fff4cf`.
After saving it in `SBI_DOWNLOADS`, verify and extract:

```sh
python3 - <<'PY_PATH_HASH'
import hashlib
import os
from pathlib import Path
archive = Path(os.environ["SBI_DOWNLOADS"]) / "path-workshop-native-models-20261031.tar.gz"
expected = "261c36550da52948aedcc2680d51bcb756db485626a6a2b8ee918fa9d6fff4cf"
hasher = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
actual = hasher.hexdigest()
if actual != expected:
    raise SystemExit("Archive checksum differs; do not extract or use it")
print("SHA256 verified:", archive.name)
PY_PATH_HASH
tar -xzf "$SBI_DOWNLOADS/path-workshop-native-models-20261031.tar.gz" \
  -C "$SBI_CHECKOUT/tutorial/assets"
```

The archive already includes its `local-checkpoints/` parent. Its native folders
are `path-heston/`, `path-rough-known/` and `path-rough-joint/` under that parent.
From `tutorial/`, open `pixi run --locked path-inference`, select a model and case,
then choose a history length. Leave **Checkpoint folder relative to
tutorial/assets/ (optional)** empty to use the matching default path. To override
it, enter e.g. `local-checkpoints/path-rough-known`, **not** an option-model path.
Press **Run CPU inference at the exact observed history length** explicitly.
No bank is loaded. The cached views remain available when models are absent.

Exact-length denotes the observation prefix, not an exact posterior. The three
trajectory candidates failed their declared final marginal-calibration screen;
Heston also retains its atom-bearing numerical law. Native publication files,
including any `ACCEPTED.json` provenance receipt, do not change that scientific
status. The notebook and [trajectory plan](PATH-INFERENCE-PLAN.md) retain the
limitations and follow-up protocol.

## Full option training banks

Use saved banks to train or compare estimators without rerunning the financial
simulators. These three banks contain complete 63-quote surfaces for Heston,
known-curve rough Bergomi and joint-curve rough Bergomi. Download only the bank
you need, plus `SHA256SUMS`, from the option release or mirror. A browser download
does not require the GitHub CLI. The following example keeps its download and
extracted tree outside the checkout.

For example, with the GitHub CLI installed, download Heston and verify it:

```sh
export SBI_BANK_DOWNLOADS="/absolute/path/to/your/sbi-bank-downloads"
export SBI_BANKS="/absolute/path/to/your/sbi-banks"
mkdir -p "$SBI_BANK_DOWNLOADS" "$SBI_BANKS"
cd "$SBI_BANK_DOWNLOADS"
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
tar -xzf sbi4finance-bank-heston-100k-20261029.tar.gz -C "$SBI_BANKS"
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

Run commands from `lab/`:

```sh
pixi install --locked -e core
pixi run --locked -e core core-check
export SBI_BANK="/absolute/path/to/extracted/snapshot-100k"
export SBI_RUN="/absolute/path/to/new-training-run"
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
The [local inference measurements](../evidence/pilot-results/2026-10-29/laptop-inference-benchmark/README.md)
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

## Trajectory banks: daily histories with inferred drift

The physical-measure trajectory application has separate native banks for
Heston, known-curve rough Bergomi and joint-curve rough Bergomi. Each contains
1,000,000 training, 4,000 validation and 4,000 test histories with 1,260 daily
log returns (five trading years). The files store observed returns, target
parameters and stable trajectory IDs; latent volatility is not an inference
input. Prefixes of a history stay in its original split.

**Preservation complete, 1 November:** all source shards, 30 transferred archive
parts and 23,631 extracted native shards passed their integrity checks. Bound
copy-time receipts verify the original Gmail copies and the last five parts
copied through the Brown-account view. Fresh Brown provider metadata confirms
all 40 distribution files uploaded under the same Google folder identities;
old cold payloads were not re-read or rehashed. The [preservation record](../evidence/pilot-results/2026-11-01/oscar-preservation/README.md)
binds the manifests, archive parts, native checks and upload receipts. An
independent cloud download was not performed. Download every part and verify it
locally before use.

The Drive destination is `training-banks-20261031/` within the existing
[path-workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k).
Download one model's entire folder, including all ten archive parts and its
`sha256sums.txt`, `bank-manifest.json` and `archive-receipt.json`.

| Folder | Native NPZ payload | Compressed split archive | Native manifest SHA-256 |
| --- | ---: | ---: | --- |
| `heston` | 10,372,028,516 B | 9,643,832,260 B | `bcddef79dddd753df6a07035124c7e0043e309dd84a8e693fb489555f76477a9` |
| `rough-known` | 10,363,964,516 B | 9,752,018,975 B | `0de676d0da84a1fd9a1984f54fbc9d8205edeae276fa3c6a783a2679044272ea` |
| `rough-joint` | 10,388,156,516 B | 9,775,977,517 B | `1e48dc63aa700dd98039d84d815f4b1a607e9fbdeacaa5c999b1c35299228fce` |

The archive sizes above come from the completed source-part receipts. Native
bank bytes and identities are unchanged. After downloading
all parts, verify their checksums and stream extraction (requires `zstd`):

```sh
export SBI_PATH_DOWNLOADS="/absolute/path/to/downloaded/heston"
cd "$SBI_PATH_DOWNLOADS"
shasum -a 256 -c sha256sums.txt
cat path-bank-heston-1000000-native.tar.zst.part* | zstd -d -c | tar -xf -
```

The extracted `heston/` contains `manifest.json` and the native split directories.
Use the analogous filename and directory for either rough bank. From `lab/`,
check every native shard without materializing a whole split:

```sh
export SBI_PATH_BANK="$SBI_PATH_DOWNLOADS/heston"
pixi run --locked -e core python scripts/path_bank.py check \
  --bank "$SBI_PATH_BANK"
```

This verifies checksums, array schemas, finite values, prior support and canonical
split-qualified identities. It is not posterior evaluation or scientific
acceptance. The checker retains one decoded shard at a time; avoid calling
`load_split` on the million-history training split for an integrity audit.

For a fresh streaming fit using the published Heston recipe, choose a new
output directory and run from `lab/` in the locked BF2.1 environment:

```sh
export SBI_PATH_RUN="/absolute/path/to/new-heston-fit"
pixi run --locked -e core python scripts/train_path_posterior.py \
  --bank "$SBI_PATH_BANK" \
  --out "$SBI_PATH_RUN" \
  --streaming --training-limit 1000000 --epochs 5 --batch-size 128 \
  --prefix-policy uniform_log_mixture --checkpoint-selection final \
  --hidden-dim 64 --summary-dim 32 --depth 6 --bins 8 --width 64 \
  --learning-rate 0.0005 --cache-shards 2 --max-seconds 3300 --seed 20261031301
```

The matched comparison uses the first 100,000 training histories for 50 epochs,
also five million presentations. Known-curve and joint-curve training use their
own banks and recorded seeds `20261031302` and `20261031303`. Runtime, backend,
source and selected checkpoint identity remain part of reproducibility; matching
this command is not a guarantee of bitwise agreement across devices. These are
fresh trajectory fits; the option trainer's optimizer-resume contract above does
not apply to them. On a slower CPU, the time cap may stop a run before all five
epochs. Inspect the recorded status, completed epochs and actual training views;
an interrupted budget is not a completed comparison.

All three selected trajectory candidates failed their declared final marginal
calibration screen; Heston also retains its documented atom-bearing numerical
law. Bank distribution does not change that status. All 4,000 test histories
have been characterized already. Use development data for changes and the
[bounded follow-up protocol](PATH-INFERENCE-PLAN.md#bounded-numerical-diagnosis-follow-up--1-november-2026)
for a new target, refit and fresh acceptance cohort. Native checkpoints alone,
in the parent workshop folder, remain sufficient for participant inference.
