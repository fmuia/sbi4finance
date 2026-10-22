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

EULER_STEPS = 50  # the sampler stored with a trained network's configuration; sampling uses sampler_times()
SAMPLER_STEPS = 100
SAMPLER_DESCRIPTION = (f"{SAMPLER_STEPS} Euler steps on t_i = 1 - (1 - i/{SAMPLER_STEPS})^2, "
                       "dense near the data end")


def sampler_times(steps=SAMPLER_STEPS):
    """The Euler time grid of every posterior draw: t_i = 1 - (1 - i/n)^2, dense where the flow ends.

    Uniform steps narrow a sharp posterior. For a Gaussian posterior of standard
    deviation s in the network's standardized coordinates, with the exact
    straight-line velocity field, 50 uniform Euler steps return a standard
    deviation of 0.65 s at s = 0.02 and 0.91 s at s = 0.1; 100 steps on this grid
    return 0.96 s and 0.98 s. Heston's v0 posterior has s near 0.02, and with
    50 uniform steps its 90% intervals covered the truth on 73% of held-out
    smiles; on this grid, 93% with 50 steps.
    """
    return (1.0 - (1.0 - np.linspace(0.0, 1.0, steps + 1)) ** 2).astype(np.float32)


def use_sampler(workflow):
    """Set the sampler on a built or loaded workflow; the saved configuration keeps its own."""
    network = workflow.approximator.inference_network
    network.integrate_kwargs = {**network.integrate_kwargs, "method": "euler", "steps": sampler_times()}
    return workflow


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


def build_workflow(task, checkpoint_dir=None, *, subnet_widths=(256, 256, 256, 256), optimizer=None,
                   time_power_law_alpha=0.5):
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
                                                   integrate_kwargs={"method": "euler", "steps": EULER_STEPS},
                                                   time_power_law_alpha=time_power_law_alpha),
        standardize=["inference_variables", "inference_conditions"],
        optimizer=optimizer,
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )


def sample(workflow, task, observations, context_values, num_samples, seed=0):
    """Posterior draws [B, N, P] in the task's parameter order and units."""
    conditions = conditioning_data(observations, context_values, task)
    use_sampler(workflow)
    drawn = workflow.sample(num_samples=num_samples, conditions=conditions, seed=int(seed))
    return np.concatenate([np.asarray(drawn[name], dtype=np.float64) for name in task["parameter_names"]], axis=-1)
