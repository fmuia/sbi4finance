# Growing the training banks without repeating simulations

The Heston and rough-Bergomi training banks can grow in stages. Keep one collection of checked shards, then publish immutable snapshots of the data available at each stage. A 300k snapshot reuses the first 100k surfaces and adds 200k; it does not regenerate the first tranche or concatenate all prices and covariance arrays into another large archive.

This implements the data-growth part of the [numerical experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md). It does not submit jobs, change the numerical task, migrate BayesFlow, or certify that a particular bank size is sufficient.

**Current rough-task status:** the plan now calls for a six-target joint rough model with a Heston-shaped initial curve. The rough config and CLI examples below still describe the earlier seven-level task and illustrate storage mechanics only. Do not queue that old full bank as data for the new model. Complete the [curve-family check and task/prior freeze](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md#rough-curve-baseline-and-bounded-flexibility-check), implement the new generator/adapter, then use its compatible config. Existing Heston data and this incremental machinery remain reusable.

## What makes a surface unique

A collection fixes the complete task, root seed, numerical generation source hashes and package versions. Its identity is independent of its eventual training size. Validation and test counts are fixed when the collection is created.

Within the collection, the base identity is **split + surface ID**. For example:

| Snapshot | Training IDs | New pricing work |
| --- | --- | --- |
| 100k | `train:0` through `train:99999` | First 100k training surfaces, plus the fixed holdouts |
| 300k | `train:0` through `train:299999` | Only IDs 100000–299999 |
| 1M | `train:0` through `train:999999` | Only IDs 300000–999999 |

The existing generator derives separate prior, pricing, quote-noise and curve streams from the task fingerprint, root seed, split, ID and purpose. It does not use the worker number, chunk size, job-array index, completion order or total bank size as the random identity. Changing any of those scheduling choices therefore preserves the same financial experiments.

**Keep the root seed fixed when extending a collection.** Increasing the ID range is how we obtain new surfaces. Starting another bank with the same root and IDs would repeat the data; choosing a different root creates a separate campaign rather than an extension. Collections reject shards from a different root, task, generation source or runtime.

The legacy numerical pricer receives a derived 64-bit seed. Different base IDs are not a mathematical guarantee that this reduced seed never collides. Planning and snapshot publication audit the actual pricing seeds across all included splits and reject any collision. The implementation keeps the existing seed protocol to preserve compatible historical data; it does not silently change how old surfaces are generated. Root seeds and IDs have an explicit 32-bit domain so entropy components have unambiguous widths. One million IDs are well inside that limit.

Repeated exposure to a base surface across epochs, noise/mask augmentations, architecture comparisons or nested learning curves is intentional reuse. Count it once when reporting unique training surfaces. New quote noise does not create new parameter/curve locations or average away the stored Monte Carlo pricing error.

## Start or adopt a collection

Run commands from one frozen source checkout with the locked environment. The config's training count is the initial allocation; later plans specify a larger training target without editing that config or its old manifests.

```bash
pixi run --locked python scripts/calibration_bank_collection.py init \
  --config configs/tasks/calibration_rough_bank.json \
  --out /path/to/rough-collection
```

To preserve existing compatible generation work, adopt its finished shards before planning more data:

```bash
pixi run --locked python scripts/calibration_bank_collection.py adopt \
  --collection /path/to/rough-collection \
  --shards /path/to/existing-rough-shards
```

Adoption checks the original manifests, identities, hashes, ID ranges, arrays and numerical error budgets. It preserves the payload bytes without repricing. Keep the original data and provenance. An incompatible old pilot, including one with a different curve prior, belongs to a separate collection. Adoption currently requires the original shard directories; an assembled legacy bank alone is not a shard collection.

## Generate only the missing ranges

```bash
pixi run --locked python scripts/calibration_bank_collection.py plan \
  --collection /path/to/rough-collection --train 100000 --chunk 50 \
  --out /path/to/jobs-100k.json
```

The command verifies finished shards and prints the number of missing tasks. Existing complete ranges are omitted, including adopted shards. Save a new job map for each expansion. A map with zero jobs needs no submission.

For a fresh rough collection with the current 5k validation and 7k test allocations, this first map contains 2,240 tasks. Submit only after confirming the account, partition and concurrency available to you:

```bash
sbatch --account YOUR_ACCOUNT --partition YOUR_PARTITION --array=0-2239%400 \
  cluster/calibration_collection_array.sbatch \
  /absolute/path/to/locked/python /path/to/jobs-100k.json
```

The final array index is **number of jobs minus one**; do not reuse 2239 after adoption or for another target. The `%400` is a concurrency example, not a claim about your allocation. If `MaxArraySize` is smaller than the map, submit parts with the wrapper's optional offset, as in the [cluster runbook](../tutorial_final/CLUSTER-RUNS.md#r1-s1-the-full-bank).

Workers verify the frozen generation identity, lock their range, generate into a temporary directory and publish completed shards atomically. Retrying a verified completed task does no pricing work. A corrupt or incompatible finished shard causes an error; the worker does not silently delete it and regenerate. An interrupted temporary shard can be retried without touching other finished work. Locks rely on the shared filesystem's POSIX locking support.

When all required ranges are complete:

```bash
pixi run --locked python scripts/calibration_bank_collection.py freeze \
  --collection /path/to/rough-collection --train 100000 \
  --out /path/to/snapshots/rough-100k
```

Publication rejects gaps, overlapping/duplicate ranges, changed payloads and repeated pricing seeds. It writes a manifest referring to the checked shards; it does not copy the whole bank. Keep the collection and its snapshots available together. Every training run records its particular snapshot manifest hash, so later growth cannot change the data that run used.

To grow to 300k, run `plan` again with `--train 300000` and a new `--out`, submit its missing tasks, then `freeze` into a new snapshot directory. Repeat for one million. Choose planned snapshot sizes on shard boundaries; a requested prefix through the middle of an existing shard is rejected. Validation and test surfaces are reused byte for byte and never moved into training.

## Train and evaluate a particular snapshot

The study runners accept a snapshot directory in `--bank`. Training loads only the arrays used for fitting, one shard at a time into preallocated selected arrays. It omits the large per-surface covariance tensors and does not open test payloads. This is a projected in-memory loader, not fully streaming GPU training; selected arrays and network tensors must still fit in memory. Prefer fresh noise to materializing many fixed augmented copies.

```bash
pixi run --locked python tutorial_final/studies/train_calibration.py \
  --bank /path/to/snapshots/rough-100k --fresh-noise \
  --epochs 40 --chunk-epochs 5 --out /path/to/training/rough-100k-seed0
```

Interruption recovery requires the same snapshot, layout, settings, source and runtime identities. At each completed chunk the runner saves immutable last/best checkpoint copies, then atomically advances the resume pointer. An interrupted later chunk leaves that committed recovery point usable, even if it changed a working file. Recovery restores the saved optimizer and quote-augmentation RNG state; it does not certify a bitwise-identical trajectory to uninterrupted training across every backend and shuffle implementation. The runner rejects a changed contract before modifying the run directory. It also rejects a nonempty directory without a valid resume state instead of deleting its files. Committed checkpoint copies are retained and need a disk allocation. A run lock prevents concurrent retries from writing to the same training output.

**A larger snapshot gets a new training run directory.** New data change batch counts, the learning-rate schedule and potentially fitted standardization. This implementation does not pretend that dataset growth is an exact resume, and does not add a warm-start mode. Fresh fits on nested snapshots provide interpretable learning curves while reusing all numerical generation work.

For learning curves and architecture selection, use `evaluate_calibration.py --split validation` or `evaluate_thin.py --split validation`. Their default remains `test` for compatibility, so specify the development split explicitly. Freeze the final model and criteria before opening reserved acceptance data. Splitting data mechanically does not make an already inspected test set untouched again.

## Checks before the first large campaign

The focused unit checks use a mocked pricer to exercise IDs, seed dispatch, extensions, adoption, corruption rejection and projected loading. They do not train a network or submit a numerical job:

```bash
PYTHONPATH=src pixi run --locked python -m unittest discover -s tests \
  -p 'test_calibration_collection*.py' -v
PYTHONPATH=src pixi run --locked python -m unittest discover -s tests \
  -p 'test_calibration_resume.py' -v
```

Run the Oscar numerical/throughput pilot separately before a large allocation. Snapshot publication checks numerical budgets one shard at a time. The existing full S1 `bank_quality.py` plotting/report study still materializes all requested arrays and needs a high-memory allocation or a separate streaming-report update at million-surface scale. If the task, numerical code or generator runtime changes, create a new collection and assess numerical compatibility explicitly rather than mixing versions into an existing one.
