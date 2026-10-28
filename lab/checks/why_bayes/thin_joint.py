"""The thin-chain network for the six-target joint task: H, eta, rho and the curve's three coefficients.

It mirrors sbi4finance.calibration.thin, written for the ten-target task with the seven-level curve: the same
adapter, FlowMatching network, sampler and noise model, with the targets replaced by the joint task's six
parameters, each constrained to its prior box. A thin joint bank (make_thin_joint_config.py) holds only the
layout's contracts, so its prices are read whole. The conditions are the quotes over their noise sd and the log of
a per-surface noise scale s; training draws s log-uniform on thin.SCALE_BOUNDS and a fresh error for every batch
(thin.thin_noise).

Needs the joint task of PR #48.
"""

import numpy as np

from sbi4finance.calibration import inference, tasks, thin

NAMES = list(tasks.JOINT_ROUGH_PARAMETERS)
_VALIDATION_TAG = 0x544A56  # "TJV"


def bounds(task):
    return np.asarray(task["prior_bounds"], dtype=float)


def noise(task):
    return np.asarray(task["context"]["quote_noise_std"], dtype=float)


def clean_data(split, task):
    """Clean prices over their noise sd, and the six targets as [B, 1] arrays; train with thin.thin_noise."""
    targets = np.asarray(split["parameters"], dtype=np.float64)
    box = bounds(task)
    if targets.shape[1] != len(NAMES) or np.any(targets <= box[:, 0]) or np.any(targets >= box[:, 1]):
        raise ValueError("six targets strictly inside the prior box required")
    clean = np.asarray(split["reference_prices"], dtype=np.float64) / noise(task)
    return {"clean_scaled": clean, **{name: targets[:, i:i + 1] for i, name in enumerate(NAMES)}}


def validation_data(split, task, seed):
    """The validation split with one fixed draw of the noise scale and the quote errors."""
    data = clean_data(split, task)
    clean = data.pop("clean_scaled")
    rng = np.random.default_rng(np.random.SeedSequence([_VALIDATION_TAG, int(seed)]))
    log_s = rng.uniform(np.log(thin.SCALE_BOUNDS[0]), np.log(thin.SCALE_BOUNDS[1]), (len(clean), 1))
    data.update(observations=clean / np.exp(log_s) + rng.standard_normal(clean.shape), log_scale=log_s)
    return data


def conditions(quotes, task, scale=1.0):
    """Conditions for quotes [B, L] in forward units, with noise scale s (scalar or [B])."""
    quotes = np.atleast_2d(np.asarray(quotes, dtype=np.float64))
    sd = noise(task)
    if quotes.shape[1] != len(sd) or not np.isfinite(quotes).all():
        raise ValueError(f"finite quotes [batch, {len(sd)}] required")
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64).reshape(-1, 1), (len(quotes), 1))
    if np.any(s <= 0):
        raise ValueError("the noise scale must be positive")
    return {"observations": quotes / (s * sd), "log_scale": np.log(s)}


def build_workflow(task, checkpoint_dir=None, *, subnet_widths=(256, 256, 256, 256), optimizer=None,
                   time_power_law_alpha=0.5):
    """thin.build_thin_workflow with the six joint targets."""
    import bayesflow as bf

    adapter = bf.Adapter()
    for name, (lower, upper) in zip(NAMES, bounds(task), strict=True):
        adapter = adapter.constrain(name, lower=float(lower), upper=float(upper), inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(NAMES, into="inference_variables")
               .concatenate(["observations", "log_scale"], into="inference_conditions"))
    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": list(subnet_widths)},
                                                   integrate_kwargs={"method": "euler",
                                                                     "steps": inference.EULER_STEPS},
                                                   time_power_law_alpha=time_power_law_alpha),
        standardize=["inference_variables", "inference_conditions"],
        optimizer=optimizer,
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )


def load(checkpoint_dir, task, *, subnet_widths=(256, 256, 256, 256)):
    workflow = build_workflow(task, subnet_widths=subnet_widths)
    workflow.load_approximator(str(checkpoint_dir) + "/model.keras")
    return workflow


def sample(workflow, task, quotes, num_samples, scale=1.0, seed=0):
    """Posterior draws [B, N, 6] in natural units, on the lab's Euler grid (inference.sampler_times)."""
    inference.use_sampler(workflow)
    drawn = workflow.sample(num_samples=num_samples, conditions=conditions(quotes, task, scale), seed=int(seed))
    return np.concatenate([np.asarray(drawn[name], dtype=np.float64) for name in NAMES], axis=-1)


def forward_variance(theta, start, end):
    """The joint curve's forward variance from start to end years, for draws [..., 6]."""
    s, l, k = (np.asarray(theta, dtype=float)[..., j] for j in (3, 4, 5))
    w = [l * t + (s - l) * -np.expm1(-k * t) / k for t in (start, end)]
    return (w[1] - w[0]) / (end - start)


def product_curve(theta, flat_after=None):
    """The joint curve on the thin products' one-year grid; with flat_after = (a, b), held beyond b at its average
    over [a, b], the classical extrapolation of the last quoted interval."""
    from sbi4finance.calibration import pricing, thin_products

    steps = thin_products.SPEC["steps_per_year"]
    times = np.arange(steps + 1) / steps
    curve = pricing.heston_variance_curve(times, *np.asarray(theta, dtype=float)[3:])
    if flat_after is not None:
        a, b = flat_after
        level = float(forward_variance(theta, a, b))
        curve["values"] = np.where(times <= b + 1e-12, curve["values"], level).tolist()
    return curve


def products(task, theta, n_paths=8192, seed=0, flat_after=None):
    """The six thin products (thin_products.NAMES) for one parameter vector, on common random numbers.

    thin_products.price, with the curve given by the joint task's coefficients instead of the seven levels."""
    from sbi4finance.calibration import products as pr, thin_products as tp

    parameters = tasks.model_parameters(task, theta)
    steps = tp.SPEC["steps_per_year"]
    paths = pr.rough_paths(parameters, product_curve(theta, flat_after), 1.0, steps, n_paths, seed)
    rho = parameters["rho"]
    out = {"variance_swap_6m": pr.variance_swap(paths, 0.5),
           "volatility_swap_6m": pr.volatility_swap(paths, 0.5),
           "forward_start_3m_6m": pr.forward_start_call(paths, rho, 0.25, 0.5),
           "put_6m_80": pr.vanilla(paths, rho, tp.SPEC["put_strike"], 0.5, "put"),
           "forward_start_6m_1y": pr.forward_start_call(paths, rho, 0.5, 1.0)}
    idx = np.arange(0, 6 * tp.SPEC["month_steps"] + 1, tp.SPEC["month_steps"])
    di = np.diff(paths["I"][:, idx], axis=1)
    dj = np.diff(paths["J"][:, idx], axis=1)
    r = rho * dj - 0.5 * di + np.sqrt((1 - rho**2) * di) * tp._period_normals(n_paths, 6, seed)
    monthly = np.expm1(r)
    out["cliquet_6m"] = pr._controlled_mean(
        np.maximum(np.minimum(monthly, tp.SPEC["cliquet_cap"]).sum(axis=1), 0.0), monthly.sum(axis=1)[:, None])
    return np.array([out[n][0] for n in tp.NAMES])
