"""PyMC references for deterministic full-surface likelihoods.

Heston uses the checked Fourier price ladder. Rough tasks can optionally use a
fixed, task-bound forward-pricer emulator. That reference targets the emulated
price model and is not independent evidence of numerical model accuracy.
Black-box likelihoods have no supplied gradients: use native DEMetropolisZ,
not ordinary MCMC with a fresh stochastic Monte Carlo log likelihood.
"""

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path

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


def load_forward_pricer(directory, task):
    """Load an explicitly bound rough pricer; legacy unbound files remain archival.

    The binding establishes task/file identity, not emulator accuracy. Training
    validation and an MCMC convergence pass do not make this an exact-model
    reference without independent pricing and refinement checks.
    """
    from .full_surface import validate_task
    from .pricer_net import Pricer
    from ..gbm_bank import _json

    validate_task(task)
    if task["model"] != "rough":
        raise ValueError("the optional forward-pricer reference supports rough tasks; Heston uses Fourier")
    directory = Path(directory).resolve()
    paths = {name: directory / name for name in ("pricer.npz", "pricer.json", "task.json")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    recorded_task = json.loads(paths["task.json"].read_text())
    record = json.loads(paths["pricer.json"].read_text())
    fingerprint = hashlib.sha256(_json(task).encode()).hexdigest()
    if (recorded_task != task or record.get("task_fingerprint") != fingerprint
            or record.get("pricer_sha256") != hashes["pricer.npz"]):
        raise ValueError("forward-pricer task/fingerprint/checkpoint binding differs from the reference task")
    pricer = Pricer.load(directory)
    return pricer, {"directory": str(directory), "task_fingerprint": fingerprint,
                    "input_sha256": {str(paths[name]): digest for name, digest in hashes.items()},
                    "activation": pricer.activation, "training_record": record,
                    "interpreted_output_units": "forward prices divided by the task's quote-noise standard deviations",
                    "target_scope": "approximate deterministic emulator likelihood; independent pricing accuracy not assessed by this run"}


def pricer_log_likelihood(task, quotes, pricer, context_values=None):
    """Physical Gaussian quote likelihood only, around a fixed rough emulator.

    The pricer returns prices divided by quote sigma. Restore the Gaussian
    normalization in physical quote units. No prior or logit Jacobian enters:
    PyMC's physical Uniform variables supply those transformations once.
    Emulator/training-pricer errors are not included in this approximate law.
    """
    from .full_surface import observations, uses_known_curve, validate_task
    from .pricer_net import features, _log_known_curve

    validate_task(task)
    if task["model"] != "rough":
        raise ValueError("the emulator likelihood requires a known-curve or joint rough task")
    if np.iscomplexobj(quotes) or (np.ma.isMaskedArray(quotes) and np.any(np.ma.getmaskarray(quotes))):
        raise ValueError("real complete unmasked quotes are required")
    values = np.asarray(quotes, dtype=np.float64)
    if values.shape == (63,):
        values = values[None, :]
    y = observations(values, task)
    if len(y) != 1:
        raise ValueError("one complete observed surface is required")
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    sigma = tasks.sigma(task)
    scaled_quotes = y[0] / sigma
    normalization = np.log(2 * np.pi * sigma**2).sum()
    known_curve = None
    if uses_known_curve(task):
        # Freeze the observed curve. It is a condition, not another MCMC state.
        _log_known_curve(task, context_values, 1)
        known_curve = np.asarray(context_values, dtype=np.float64).reshape(1, -1).copy()
    # Joint auxiliaries are deliberately ignored, even if they are supplied.
    probe = np.asarray(pricer(features(task, bounds.mean(axis=1)[None, :], known_curve)))
    if probe.shape != (1, 63) or np.iscomplexobj(probe) or not np.isfinite(probe).all():
        raise ValueError("the bound forward pricer must return one finite 63-quote vector in noise units")

    def log_likelihood(parameters):
        if np.iscomplexobj(parameters) or (np.ma.isMaskedArray(parameters) and np.any(np.ma.getmaskarray(parameters))):
            raise ValueError("one finite real unmasked physical parameter vector is required")
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (len(bounds),) or not np.isfinite(parameters).all():
            raise ValueError("one finite parameter vector in task order is required")
        # Logit features are defined on the open interior. Boundaries have zero
        # prior mass and PyMC's bounded transform stays in that interior.
        if np.any(parameters <= bounds[:, 0]) or np.any(parameters >= bounds[:, 1]):
            return -np.inf
        predicted = np.asarray(pricer(features(task, parameters[None, :], known_curve)))
        if predicted.shape != (1, 63) or np.iscomplexobj(predicted) or not np.isfinite(predicted).all():
            raise FloatingPointError("forward-pricer evaluation failed; do not replace it with zero density")
        residual = scaled_quotes - predicted[0]
        return float(-0.5 * (residual @ residual + normalization))

    return log_likelihood


def rough_derived_diagnostics(inference_data, task):
    """MCMC diagnostics for rho*eta, with chain/draw axes preserved."""
    import xarray as xr

    if task["model"] != "rough":
        raise ValueError("rho*eta diagnostics require a rough task")
    names = task["parameter_names"]
    draws = inference_data["posterior"]["parameters"].transpose("chain", "draw", "parameter").values
    product = draws[..., names.index("rho")] * draws[..., names.index("eta")]
    tree = xr.DataTree.from_dict({"posterior": xr.Dataset(
        {"parameters": (("chain", "draw", "parameter"), product[..., None])}, coords={"parameter": ["rho_eta"]})})
    result = chain_diagnostics(tree)
    result["scope"] = "derived rho*eta chain diagnostics; not product pricing or independent emulator validation"
    return result


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


def data_fit_initialization(log_likelihood, bounds, chains, seed, **settings):
    """Opt-in data-only MAP starts and proposal geometry, with native chain seeds."""
    from .reference_initialization import data_fit_initialization as fit

    result = fit(log_likelihood, bounds, chains, seed, **settings)
    seeds = reference_initialization(bounds, chains, seed)
    result.update({key: value for key, value in seeds.items()
                   if key in ("pymc_random_seed", "pymc_chain_rng_protocol", "chains")})
    return result


def sample_deterministic_reference(log_likelihood, bounds, parameter_names, *, draws=2000,
                                   tune=2000, chains=4, seed=0, progressbar=True, cores=1,
                                   mp_context="spawn", callback=None, initialization=None):
    """Sample a bounded independent-uniform prior with a deterministic likelihood.

    Retains chain/draw structure in PyMC 6's xarray DataTree. PyMC handles
    bounded-variable transforms and their Jacobians. Default starts remain
    independent prior draws. An explicit data-only initialization record can
    supply dispersed starts and a covariance in those same value coordinates.
    No amortized posterior supplies starts or proposal geometry.
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
    covariance = None
    if initialization is None:
        starts = np.asarray(reference_initialization(bounds, chains, seed)["prior_starts"])
    else:
        if (initialization.get("kind") != "data-fit" or initialization.get("bounds") != bounds.tolist()
                or initialization.get("seed") != seed):
            raise ValueError("data-only initialization must match the sampler bounds and root seed")
        starts = np.asarray(initialization["chain_starts"], dtype=float)
        covariance = np.asarray(initialization["proposal"]["S"], dtype=float)
        if (starts.shape != (chains, len(names)) or not np.isfinite(starts).all()
                or np.any(starts <= bounds[:, 0]) or np.any(starts >= bounds[:, 1])
                or covariance.shape != (len(names), len(names)) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)
                or np.linalg.eigvalsh(covariance).min() <= 0):
            raise ValueError("finite interior starts and positive definite value-coordinate covariance required")
    for start in starts:
        first, repeated = float(log_likelihood(start)), float(log_likelihood(start))
        if not math.isfinite(first) or first != repeated:
            raise ValueError("reference log likelihood must be finite and deterministic at chain starts")

    import pymc as pm
    from .pymc_reference_op import DeterministicLikelihoodOp
    likelihood_op = DeterministicLikelihoodOp(log_likelihood)

    with pm.Model(coords={"parameter": names}):
        parameters = pm.Uniform("parameters", lower=bounds[:, 0], upper=bounds[:, 1],
                                dims="parameter")
        pm.Potential("quote_log_likelihood", likelihood_op(parameters))
        if covariance is None:
            step = pm.DEMetropolisZ(vars=[parameters])
        else:
            from pymc.step_methods.metropolis import MultivariateNormalProposal
            step = pm.DEMetropolisZ(vars=[parameters], S=covariance, proposal_dist=MultivariateNormalProposal,
                                    scaling=1.0, tune="scaling")
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


def variance_functional_diagnostics(inference_data, task):
    """Independent MCMC diagnostics for analytic Heston variance products."""
    import arviz_stats as az
    import xarray as xr
    from .variance_functionals import heston_variance_functionals

    draws = inference_data["posterior"]["parameters"].transpose("chain", "draw", "parameter").values
    values = heston_variance_functionals(task, draws)
    tree = xr.DataTree.from_dict({"posterior": xr.Dataset({name: (("chain", "draw"), value)
                                                        for name, value in values.items()})})
    frame = az.summary(tree, var_names=list(values), kind="diagnostics", round_to="none")
    names = {str(name): {str(key): float(value) if np.isfinite(value) else None
                         for key, value in row.items()} for name, row in frame.iterrows()}
    passed = bool(names) and all(
        row.get("r_hat") is not None and row["r_hat"] <= 1.01
        and row.get("ess_bulk") is not None and row["ess_bulk"] >= 400
        and row.get("ess_tail") is not None and row["ess_tail"] >= 400 for row in names.values())
    return {"names": names, "mcmc_diagnostics_passed": passed,
            "requirements": {"r_hat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400},
            "scope": "analytic variance-unit functionals; a product pass does not imply parameter or scientific acceptance"}
