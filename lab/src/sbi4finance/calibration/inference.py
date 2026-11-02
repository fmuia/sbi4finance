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

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.inference import (
    EULER_STEPS,
    SAMPLER_STEPS,
    SAMPLER_DESCRIPTION,
    sampler_times,
    use_sampler,
    conditioning_data,
    build_workflow,
    sample,
    _checked_task,
)


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


def clean_training_data(split, task):
    """Clean prices over their noise sd as conditions, with the targets; train with fresh_noise.

    The conditions are y / sigma, so the quote error of the declared model is a
    standard normal in these units, and adding one per batch (fresh_noise)
    gives every training step new quotes around the same priced surfaces.
    """
    return training_data({**split, "observations": np.asarray(split["reference_prices"], dtype=np.float64)}, task)


def fresh_noise(seed):
    """BayesFlow batch augmentation: a new standard-normal quote error on every scaled quote of every batch."""
    rng = np.random.default_rng(np.random.SeedSequence([0x46524553, _integer(seed, "noise seed")]))

    def augment(batch):
        batch = dict(batch)
        batch["observations"] = batch["observations"] + rng.standard_normal(np.shape(batch["observations"]))
        return batch

    augment.rng = rng  # its state is saved with a resumable training run
    return augment


def offline_optimizer(total_steps, initial_learning_rate=5e-4):
    """BayesFlow 2.0's default for offline training, built once for a run of total_steps.

    AdamW (weight decay 5e-3, gradient norm clipped at 1.5); the learning rate
    warms up from a tenth of its peak over the first 5% of steps, then decays to
    zero along a cosine. Built explicitly so that a run trained in chunks keeps
    one schedule across the chunks and across restarts.
    """
    import keras

    warmup = int(0.05 * total_steps)
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1 * initial_learning_rate, warmup_target=initial_learning_rate,
        warmup_steps=warmup, decay_steps=total_steps - warmup, alpha=0)
    return keras.optimizers.AdamW(schedule, weight_decay=5e-3, clipnorm=1.5)
