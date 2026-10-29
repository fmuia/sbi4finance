"""Neural posterior estimation for the calibration tasks: the network of slots 2 to 5, and its sampler.

The network conditions on each quote divided by its noise standard deviation,
y_c / sigma_c, which is defined for any value including negative noisy quotes,
and, for rough Bergomi, on the log forward-variance levels of the example.
Targets are the task's parameters, constrained to their prior box by a logit
inside the adapter. Generating parameters, clean prices and pricing errors are
never conditioning variables. This is the lab's sbi4finance.calibration.inference
without its training helpers; the lab trains the networks, the notebooks load them.
"""

import numpy as np

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
    network = workflow.inference_network
    network.integrate_kwargs = {**network.integrate_kwargs, "method": "euler", "steps": sampler_times()}
    return workflow


def _checked_task(task):
    """The keys a task must have to condition and build a network; the lab validates the rest."""
    missing = {"parameter_names", "prior_bounds", "context", "model_context"} - set(task)
    if missing:
        raise ValueError(f"task lacks {sorted(missing)}")
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
