"""PyMC references for deterministic full-surface likelihoods.

The first route is Heston with the checked Fourier price ladder. A black-box
log likelihood has no supplied gradients, so use PyMC's DEMetropolisZ rather
than NUTS. This is not a wrapper for stochastic Monte Carlo log likelihoods.
"""

import math
import multiprocessing
import os

import numpy as np

from . import tasks


def heston_log_likelihood(task, quotes):
    """Build the declared Gaussian observation likelihood on all 63 contracts.

    Numerical failures propagate: a proposal that could not be priced to the
    declared tolerance must not silently acquire zero posterior probability.
    """
    from .full_surface import observations, validate_task

    validate_task(task)
    if task["model"] != "hkde":
        raise ValueError("this reference requires the deterministic Heston task")
    if np.iscomplexobj(quotes):
        raise ValueError("real complete quotes are required")
    if np.ma.isMaskedArray(quotes) and np.ma.getmaskarray(quotes).any():
        raise ValueError("missing quotes are unsupported by the complete-surface reference")
    values = np.asarray(quotes, dtype=np.float64)
    if values.shape == (63,):
        values = values[None, :]
    y = observations(values, task)
    if len(y) != 1:
        raise ValueError("one complete observed surface is required")
    y = y[0]
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    sigma = tasks.sigma(task)
    normalization = np.log(2 * np.pi * sigma**2).sum()

    def log_likelihood(parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (len(bounds),) or not np.isfinite(parameters).all():
            raise ValueError("one finite parameter vector in task order is required")
        if np.any(parameters < bounds[:, 0]) or np.any(parameters > bounds[:, 1]):
            return -np.inf
        prices, *_ = tasks.price(task, parameters, np.empty(0), seed=0)
        residual = (y - prices) / sigma
        return float(-0.5 * (residual @ residual + normalization))

    return log_likelihood


def reference_initialization(bounds, chains, seed):
    """Record physical prior starts and PyMC 6's native per-chain seed protocol."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
            or type(chains) is not int or chains < 2 or type(seed) is not int or seed < 0):
        raise ValueError("finite prior bounds, at least two chains and a nonnegative seed required")
    rng = np.random.default_rng(np.random.SeedSequence([0x50594D43, seed]))
    starts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(chains, len(bounds)))
    children = np.random.default_rng(seed).spawn(chains)
    return {"prior_starts": starts.tolist(), "start_protocol": "PCG64 SeedSequence([0x50594D43, seed]); independent uniform prior draws",
            "pymc_random_seed": seed,
            "pymc_chain_rng_protocol": "PyMC 6: default_rng(seed).spawn(chains); child model seed drawn before sampling",
            "chains": [{"chain": chain, "spawn_key": list(child.bit_generator.seed_seq.spawn_key),
                        "model_initialization_seed": int(child.integers(2**30))}
                       for chain, child in enumerate(children)]}


def validate_parallelism(cores, chains, mp_context):
    """Respect the Slurm allocation and use start methods safe beside JAX."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        available = min(available, int(os.environ["SLURM_CPUS_PER_TASK"]))
    if type(cores) is not int or not 1 <= cores <= min(chains, available):
        raise ValueError(f"cores must be between 1 and min(chains, allocated CPUs)={min(chains, available)}")
    if mp_context not in ("spawn", "forkserver") or mp_context not in multiprocessing.get_all_start_methods():
        raise ValueError("use an available spawn or forkserver multiprocessing context; fork is unsupported")


def sample_deterministic_reference(log_likelihood, bounds, parameter_names, *, draws=2000,
                                   tune=2000, chains=4, seed=0, progressbar=True, cores=1,
                                   mp_context="spawn", callback=None):
    """Sample a bounded independent-uniform prior with a deterministic likelihood.

    Retains chain/draw structure in PyMC 6's xarray DataTree. PyMC handles
    bounded-variable transforms and their Jacobians. Each chain starts at a
    separate prior draw; no neural estimator supplies its initialization.
    Cores are explicit and bounded by the allocation. Spawn/forkserver avoids
    forking a process that may already have initialized JAX threads. A native
    PyMC callback can retain partial draws; it does not preserve sampler state.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    names = list(parameter_names)
    if (bounds.shape != (len(names), 2) or not names or len(set(names)) != len(names)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("unique parameter names and increasing finite bounds are required")
    if any(type(n) is not int or n < 1 for n in (draws, tune, chains)) or chains < 2:
        raise ValueError("positive draws/tune and at least two chains are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validate_parallelism(cores, chains, mp_context)
    starts = np.asarray(reference_initialization(bounds, chains, seed)["prior_starts"])
    for start in starts:
        first, repeated = float(log_likelihood(start)), float(log_likelihood(start))
        if not math.isfinite(first) or first != repeated:
            raise ValueError("reference log likelihood must be finite and deterministic at prior starts")

    import pymc as pm
    from .pymc_reference_op import DeterministicLikelihoodOp
    likelihood_op = DeterministicLikelihoodOp(log_likelihood)

    with pm.Model(coords={"parameter": names}):
        parameters = pm.Uniform("parameters", lower=bounds[:, 0], upper=bounds[:, 1],
                                dims="parameter")
        pm.Potential("quote_log_likelihood", likelihood_op(parameters))
        step = pm.DEMetropolisZ(vars=[parameters])
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, step=step,
                         initvals=[{"parameters": row} for row in starts],
                         random_seed=seed, progressbar=progressbar,
                         return_inferencedata=True, discard_tuned_samples=True,
                         mp_ctx=mp_context, blas_cores=cores, callback=callback)


def chain_diagnostics(inference_data):
    """Report statistical checks; passing them alone is not scientific acceptance."""
    import arviz_stats as az

    frame = az.summary(inference_data, var_names=["parameters"], round_to="none")
    records = {}
    for name, row in frame.iterrows():
        records[str(name)] = {str(key): float(value) if np.isfinite(value) else None
                              for key, value in row.items()}
    passed = bool(len(records)) and all(
        row.get("r_hat") is not None and row["r_hat"] <= 1.01
        and row.get("ess_bulk") is not None and row["ess_bulk"] >= 400
        and row.get("ess_tail") is not None and row["ess_tail"] >= 400
        for row in records.values())
    return {"parameters": records, "mcmc_diagnostics_passed": passed,
            "scientific_acceptance": "not_assessed",
            "requirements": {"r_hat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400},
            "limitations": "Also assess numerical-price refinement, products and reference uncertainty."}
