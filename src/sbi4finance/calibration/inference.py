"""Neural posterior estimation for the calibration tasks, generalized from rough_inference.

The network conditions on each quote divided by its noise standard deviation,
y_c / sigma_c, which is defined for any value including negative noisy quotes,
and, for rough Bergomi, on the log forward-variance levels of the example.
Targets are the task's parameters, constrained to their prior box by a logit
inside the adapter. Generating parameters, clean prices and pricing errors are
never conditioning variables.
"""

import hashlib

import numpy as np

from ..gbm_bank import _integer, _json
from . import tasks

EULER_STEPS = 50  # fixed-step sampler, as in the SVJ design; SBC validates this sampler


def _checked_task(task):
    tasks.validate_task(task)
    return task


def conditioning_data(observations, context_values, task):
    """Scaled quotes [B, C] and, if the task has one, log curve levels [B, K]."""
    _checked_task(task)
    prices = np.asarray(observations, dtype=np.float64)
    c, k = len(task["context"]["contracts"]), tasks.context_size(task)
    if prices.ndim == 3:
        prices = prices[..., 0]
    if prices.ndim != 2 or prices.shape[1] != c or not len(prices) or not np.isfinite(prices).all():
        raise ValueError(f"complete finite quotes [batch,{c}] required")
    data = {"observations": prices / tasks.sigma(task)}
    if k:
        levels = np.asarray(context_values, dtype=np.float64)
        if levels.shape != (len(prices), k) or not np.isfinite(levels).all() or np.any(levels <= 0):
            raise ValueError(f"positive finite forward-variance levels [batch,{k}] required")
        data["context"] = np.log(levels)
    return data


def training_data(split, task):
    """Unaugmented conditions plus the parameter targets, one [B, 1] array per name."""
    data = conditioning_data(split["observations"], split.get("context_values"), task)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    parameters = np.asarray(split["parameters"], dtype=np.float64)
    if parameters.shape != (len(data["observations"]), len(bounds)) or not np.isfinite(parameters).all():
        raise ValueError("finite parameter targets [batch, parameters] required")
    if np.any(parameters <= bounds[:, 0]) or np.any(parameters >= bounds[:, 1]):
        raise ValueError("parameter targets must lie strictly inside prior_bounds")
    return {**data, **{name: parameters[:, i:i + 1] for i, name in enumerate(task["parameter_names"])}}


def augmented_training_data(split, task, *, augmentations, seed):
    """Fresh quote noise around each training surface's clean prices.

    Rows are base-major then augmentation-major. Each base's stream depends on
    the task, the seed and its split-qualified ID, never on row order, as in
    rough_inference. Augmentations keep their base's split.
    """
    count = _integer(augmentations, "augmentations", minimum=1)
    seed = _integer(seed, "augmentation seed")
    base = training_data(split, task)
    n, c = base["observations"].shape
    ids = np.asarray(split["example_id"])
    if ids.shape != (n,) or len(set(ids.tolist())) != n or any(not i.startswith("train:") for i in ids.tolist()):
        raise ValueError("augmentation requires unique split-qualified train base IDs")
    clean = np.asarray(split["reference_prices"], dtype=np.float64)
    if clean.shape != (n, c) or not np.isfinite(clean).all():
        raise ValueError("clean reference_prices must be finite [batch, contracts]")
    sigma = tasks.sigma(task)
    digest = hashlib.sha256(_json(task).encode()).hexdigest()
    task_words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    quotes = np.empty((n, count, c))
    for row, identifier in enumerate(ids.tolist()):
        id_digest = hashlib.sha256(identifier.encode()).hexdigest()
        id_words = [int(id_digest[i:i + 8], 16) for i in range(0, 64, 8)]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x43414C42, seed, *task_words, *id_words])))
        quotes[row] = clean[row] + sigma * rng.standard_normal((count, c))
    result = {"observations": quotes.reshape(n * count, c) / sigma}
    if "context" in base:
        result["context"] = np.repeat(base["context"], count, axis=0)
    for name in task["parameter_names"]:
        result[name] = np.repeat(base[name], count, axis=0)
    return result


def build_workflow(task, checkpoint_dir=None, *, subnet_widths=(256, 256, 256, 256)):
    """FlowMatching on logit-constrained targets; conditions standardized with training statistics."""
    _checked_task(task)
    import bayesflow as bf

    names = task["parameter_names"]
    adapter = bf.Adapter()
    for name, (lower, upper) in zip(names, task["prior_bounds"], strict=True):
        adapter = adapter.constrain(name, lower=float(lower), upper=float(upper), inclusive="none")
    conditions = ["observations"] + (["context"] if tasks.context_size(task) else [])
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(list(names), into="inference_variables")
               .concatenate(conditions, into="inference_conditions"))
    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": list(subnet_widths)},
                                                   integrate_kwargs={"method": "euler", "steps": EULER_STEPS}),
        standardize=["inference_variables", "inference_conditions"],
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )


def sample(workflow, task, observations, context_values, num_samples, seed=0):
    """Posterior draws [B, N, P] in the task's parameter order and units."""
    conditions = conditioning_data(observations, context_values, task)
    drawn = workflow.sample(num_samples=num_samples, conditions=conditions, seed=int(seed))
    return np.concatenate([np.asarray(drawn[name], dtype=np.float64) for name in task["parameter_names"]], axis=-1)
