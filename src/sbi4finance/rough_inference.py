"""Restricted rough-Bergomi inference under the declared finite-MC quote law.

Only nine ordered noisy currency prices condition the joint H/eta flow. The
full saved task (including curve, contracts and numerical settings) must match
at application time: array shape and adapter ID do not establish compatibility.
No framework import is needed for the array boundary or quote augmentation.
"""

import hashlib

import numpy as np

from .gbm_bank import _integer, _json
from .option_bank import validate_config


ADAPTER_ID = "rough_ordered_currency_panel_v1"
PARAMETER_NAMES = ("hurst", "eta")


def _task_design(task):
    if (task.get("task_id") != "rough_option_panel_v1"
            or task.get("model") != "rough" or task.get("measure") != "Q"
            or task.get("parameter_names") != list(PARAMETER_NAMES)):
        raise ValueError("expected the rough Q task with ordered hurst, eta targets")
    validate_config({"task": task, "root_seed": 0,
                     "splits": {"train": 1, "validation": 1, "test": 1}})
    if len(task["context"]["contracts"]) != 9:
        raise ValueError("the restricted rough adapter requires nine ordered contracts")
    return np.asarray(task["prior_bounds"], dtype=np.float64)


def conditioning_data(observations, task):
    """Select noisy observations [B,9,1] without per-panel normalization.

    Gaussian negative quotes are valid. Labels, clean prices, pricing errors,
    base identities and future outcomes are not conditioning variables.
    """
    _task_design(task)
    prices = np.asarray(observations, dtype=np.float64)
    if (prices.ndim != 3 or prices.shape[0] == 0 or prices.shape[1:] != (9, 1)
            or not np.isfinite(prices).all()):
        raise ValueError("complete finite [batch,9,1] currency prices are required")
    return {"observations": prices[..., 0]}


def training_data(split, task):
    """Add separate synthetic parameter labels to unaugmented observations."""
    result = conditioning_data(split["observations"], task)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    parameters = np.asarray(split["parameters"], dtype=np.float64)
    if (parameters.shape != (len(result["observations"]), 2)
            or not np.isfinite(parameters).all()):
        raise ValueError("synthetic parameter targets must be finite [batch,2]")
    if np.any(parameters <= bounds[:, 0]) or np.any(parameters >= bounds[:, 1]):
        raise ValueError("parameter targets must lie strictly inside prior_bounds")
    return {**result, **{name: parameters[:, index:index + 1]
                        for index, name in enumerate(PARAMETER_NAMES)}}


def augmented_training_data(split, task, *, augmentations, seed):
    """Draw fresh quote errors around each TRAIN base's clean MC panel.

    This conditions repeatedly on the same (theta,U) base, with independent
    Gaussian quote errors. It does not create independent pricing paths or
    independent parameter draws. All augmentations retain their base's split.
    The returned rows are base-major, then augmentation-major. PCG64 streams
    depend on task, explicit seed and split-qualified base ID, never row order.
    Selection and audit panels must use training_data without augmentation.
    """
    count = _integer(augmentations, "augmentations", minimum=1)
    seed = _integer(seed, "augmentation seed")
    base = training_data(split, task)
    n = len(base["observations"])
    ids = np.asarray(split["example_id"])
    if (ids.shape != (n,) or ids.dtype.kind not in "US"
            or len(set(ids.tolist())) != n
            or any(not identifier.startswith("train:") for identifier in ids.tolist())):
        raise ValueError("augmentation requires unique split-qualified train base IDs")
    clean = np.asarray(split["reference_prices"], dtype=np.float64)
    if clean.shape != (n, 9) or not np.isfinite(clean).all():
        raise ValueError("clean reference_prices must be finite [batch,9]")
    sigma = task["context"]["quote_noise_std"]
    task_digest = hashlib.sha256(_json(task).encode()).hexdigest()
    task_words = [int(task_digest[i:i + 8], 16) for i in range(0, 64, 8)]
    observations = np.empty((n, count, 9), dtype=np.float64)
    for index, identifier in enumerate(ids.tolist()):
        digest = hashlib.sha256(identifier.encode()).hexdigest()
        id_words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
        rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([0x52415547, seed, *task_words, *id_words])))
        observations[index] = clean[index] + sigma * rng.standard_normal((count, 9))
    return {"observations": observations.reshape(n * count, 9),
            **{name: np.repeat(base[name], count, axis=0) for name in PARAMETER_NAMES}}


def build_workflow(task, checkpoint_dir=None):
    """One small coupling flow; float64 logits precede neural float32 casting.

    Global normalization uses training data only and freezes at inference.
    Joint samples preserve paired H/eta draws in their shared sample axis.
    """
    bounds = _task_design(task)
    import bayesflow as bf

    adapter = bf.Adapter()
    for name, (lower, upper) in zip(PARAMETER_NAMES, bounds, strict=True):
        adapter = adapter.constrain(name, lower=float(lower), upper=float(upper),
                                    inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(list(PARAMETER_NAMES), into="inference_variables")
               .rename("observations", "inference_conditions"))
    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.CouplingFlow(
            depth=4, transform="spline", subnet_kwargs={"widths": [64, 64]},
            transform_kwargs={"bins": 8},
        ),
        standardize="all",
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )
