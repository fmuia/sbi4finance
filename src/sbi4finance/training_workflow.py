"""Shared training lifecycle; task adapters and neural architectures remain local.

Only training arrays reach optimizer/normalizer fitting. Validation selects the
saved epoch. Reserved test arrays have no argument in this executable lifecycle.
"""

from pathlib import Path
import time

import numpy as np


def reserve_output_dir(path):
    """Create a new run directory; never replace a checkpoint or study."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_training_splits(splits):
    """Check split identities before a task adapter selects fit/selection arrays."""
    if not {"train", "validation"}.issubset(splits):
        raise ValueError("training requires separate train and validation splits")
    identities = set()
    for name, split in splits.items():
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {name}")
        ids = np.asarray(split.get("example_id", ()))
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError(f"{name} needs nonempty one-dimensional example identities")
        current = set(ids.tolist())
        if len(current) != len(ids) or identities.intersection(current):
            raise ValueError("example identities must be unique within and across splits")
        identities.update(current)


def _example_count(data, name):
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{name} must be a nonempty mapping of task arrays")
    sizes = set()
    for value in data.values():
        array = np.asarray(value)
        if array.ndim < 2 or not np.isfinite(array).all():
            raise ValueError(f"{name} arrays must be finite and have a batch dimension")
        sizes.add(len(array))
    if len(sizes) != 1 or next(iter(sizes)) == 0:
        raise ValueError(f"{name} arrays need the same nonzero batch size")
    return sizes.pop()


def fit_and_select(workflow, *, training_data, validation_data, epochs,
                   batch_size, seed, num_batches_per_epoch=None, verbose=0):
    """Fit online/offline, then load the epoch minimizing validation loss.

The caller constructs its own workflow in a newly reserved output directory.
BayesFlow updates standardization during training, freezes it for validation
and inference, and saves both with the selected network. No final test data
are accepted. The returned record separates finite examples from exposures.
"""
    for label, value in (("epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    n_validation = _example_count(validation_data, "validation")
    if not workflow.checkpoint_filepath or workflow.save_weights_only:
        raise ValueError("training requires a full-model checkpoint directory")
    # This executable policy is shared by SVJ and HKDE.
    workflow.save_best_only = True
    start = time.perf_counter()
    if training_data is None:
        if type(num_batches_per_epoch) is not int or num_batches_per_epoch <= 0:
            raise ValueError("online training needs a positive number of batches")
        mode = "online"
        unique = None  # Framework setup may request additional simulated batches.
        exposures = epochs * num_batches_per_epoch * batch_size
        history = workflow.fit_online(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size, validation_data=validation_data, verbose=verbose,
        )
    else:
        if num_batches_per_epoch is not None:
            raise ValueError("offline training consumes each finite training example per epoch")
        mode = "offline"
        unique = _example_count(training_data, "training")
        if any(np.shares_memory(train, validation) for train in training_data.values()
               for validation in validation_data.values()):
            raise ValueError("training and validation arrays must not overlap")
        exposures = epochs * unique
        history = workflow.fit_offline(
            data=training_data, validation_data=validation_data,
            epochs=epochs, batch_size=batch_size, verbose=verbose,
        )
    fit_seconds = time.perf_counter() - start
    losses = np.asarray(history.history.get("val_loss", ()), dtype=float)
    if len(losses) != epochs or not np.isfinite(losses).all():
        raise RuntimeError("finite validation loss is required for every trained epoch")
    workflow.load_approximator()
    return history, {
        "schema_version": 1, "mode": mode, "seed": seed,
        "normalization_fit_split": "train", "optimizer_fit_split": "train",
        "selection_split": "validation", "selection_metric": "val_loss",
        "selection_rule": "minimum; first epoch on ties", "selected_epoch": int(np.argmin(losses)) + 1,
        "epochs": epochs, "batch_size": batch_size,
        "unique_training_examples": unique, "training_exposures": exposures,
        "validation_examples": n_validation, "reserved_test_evaluated": False,
        "fit_seconds": fit_seconds,
        "scientific_acceptance": False,
    }


def verify_reload(workflow, *, restore, sample, save=None):
    """Optionally finalize a selected checkpoint and compare same-seed draws."""
    if save is not None:
        save(workflow)
    before = sample(workflow)
    restored = restore()
    after = sample(restored)
    if isinstance(before, dict):
        equal = (isinstance(after, dict) and before.keys() == after.keys()
                 and all(np.array_equal(before[key], after[key]) for key in before))
    else:
        equal = np.array_equal(before, after)
    if not equal:
        raise RuntimeError("checkpoint reload changed same-seed joint draws")
    return True
