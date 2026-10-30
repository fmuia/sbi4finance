"""Masked, arbitrary-prefix BayesFlow inference from observed returns only.

All lengths 1..T are supported. Every training batch draws an independent
integer prefix for every trajectory, pads to T, and forwards a boolean mask
through BayesFlow 2.1 to the causal GRU. Fixed physical scaling preserves
between-history return amplitude; no data-dependent standardizer is fitted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "path_posterior_v1"
DEFAULT_NETWORK = {"hidden_dim": 64, "summary_dim": 32, "depth": 6,
                   "bins": 8, "widths": [64, 64]}
LOGISTIC_SD = np.pi / np.sqrt(3.0)
PREFIX_POLICIES = {
    "uniform": "uniform_integer_per_trajectory_1_to_max_with_mask",
    "uniform_log_mixture": "half_uniform_half_log_uniform_integer_per_trajectory_1_to_max_with_mask",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_task(task):
    names = task.get("parameter_names", [])
    bounds = np.asarray(task.get("prior_bounds", []), dtype=np.float64)
    kinds = task.get("prior_types", [])
    if (not names or len(set(names)) != len(names) or bounds.shape != (len(names), 2)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0])
            or len(kinds) != len(names) or any(k not in ("uniform", "log_uniform") for k in kinds)):
        raise ValueError("invalid path-task parameter/prior definition")
    for kind, (lower, _) in zip(kinds, bounds, strict=True):
        if kind == "log_uniform" and lower <= 0:
            raise ValueError("log-uniform prior must be positive")
    integer(task["n_observations"], "n_observations")
    if not np.isfinite(task["dt_years"]) or task["dt_years"] <= 0 or task.get("measure") != "P":
        raise ValueError("positive time step and an explicit P task required")
    if any(integer(n, "prefix length") > task["n_observations"] for n in task["prefix_lengths"]):
        raise ValueError("validation landmark exceeds maximum history")
    return task


def _bounds(task):
    validate_task(task)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64).copy()
    logged = np.asarray([kind == "log_uniform" for kind in task["prior_types"]])
    bounds[logged] = np.log(bounds[logged])
    return bounds, logged


def transform_parameters(values, task):
    """Prior-CDF logits, with fixed logistic-SD scaling, preserve box support."""
    values = np.asarray(values, dtype=np.float64)
    bounds, logged = _bounds(task)
    physical = np.asarray(task["prior_bounds"], dtype=np.float64)
    if (values.shape[-1:] != (len(bounds),) or not np.isfinite(values).all()
            or np.any(values <= physical[:, 0]) or np.any(values >= physical[:, 1])):
        raise ValueError("parameter rows must be finite and strictly inside prior bounds")
    coordinates = values.copy()
    coordinates[..., logged] = np.log(coordinates[..., logged])
    u = (coordinates - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return (np.log(u) - np.log1p(-u)) / LOGISTIC_SD


def inverse_parameters(values, task):
    values = np.asarray(values, dtype=np.float64)
    bounds, logged = _bounds(task)
    if values.shape[-1:] != (len(bounds),) or not np.isfinite(values).all():
        raise ValueError("invalid unconstrained posterior array")
    # Stable sigmoid. Do not clip samples, reject them, or redraw them.
    z = values * LOGISTIC_SD
    u = np.exp(-np.logaddexp(0.0, -z))
    result = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
    result[..., logged] = np.exp(result[..., logged])
    return result


def conditioning_data(values, task, *, lengths=None):
    """Pad returns and known time coordinates, exposing neither latent states nor labels.

    ``values`` may contain any complete prefix, or a full history with explicit
    per-row lengths. Values after those lengths are ignored, even if nonfinite.
    """
    validate_task(task)
    if np.iscomplexobj(values) or np.ma.isMaskedArray(values):
        raise ValueError("real unmasked return arrays required")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    maximum = task["n_observations"]
    if values.ndim != 3 or values.shape[2] != 1 or not len(values) or not 1 <= values.shape[1] <= maximum:
        raise ValueError("returns must be [batch,time,1] with 1 <= time <= task maximum")
    if lengths is None:
        lengths = np.full(len(values), values.shape[1], dtype=np.int64)
    lengths = np.asarray(lengths)
    if (lengths.shape != (len(values),) or lengths.dtype.kind not in "iu"
            or np.any(lengths < 1) or np.any(lengths > values.shape[1])):
        raise ValueError("each integer length must lie within its supplied history")
    mask = np.arange(maximum)[None, :] < lengths[:, None]
    observed_mask = mask[:, :values.shape[1]]
    # Keep the full bank in float64, but materialize neural inputs directly in
    # float32. Chunked float64 scaling avoids whole-bank temporary arrays.
    tokens = np.zeros((len(values), maximum, 2), dtype=np.float32)
    scale = np.sqrt(0.04 * task["dt_years"])
    times = (np.arange(maximum) + 1) / maximum
    for start in range(0, len(values), 1024):
        stop = start + 1024
        present = observed_mask[start:stop]
        raw = values[start:stop, :, 0]
        if not np.isfinite(raw[present]).all():
            raise ValueError("observed returns must be finite")
        tokens[start:stop, :values.shape[1], 0] = np.where(present, raw / scale, 0.0)
        tokens[start:stop, :, 1] = np.where(mask[start:stop], times[None, :], 0.0)
    if not np.isfinite(tokens).all():
        raise ValueError("return scaling overflowed neural float32 storage")
    context = np.column_stack([lengths / maximum,
                               np.full(len(values), task["dt_years"] * 252.0)]).astype(np.float32)
    return {"observations": tokens, "time_context": context, "summary_mask": mask}


def training_data(split, task, *, lengths=None):
    result = conditioning_data(split["observations"], task, lengths=lengths)
    parameters = transform_parameters(split["parameters"], task).astype(np.float32)
    if parameters.shape != (len(result["observations"]), len(task["parameter_names"])):
        raise ValueError("parameters and observed trajectories must align")
    return {**result, "parameters": parameters}


def random_prefixes(task, seed, policy="uniform"):
    """Pre-adapter prefix sampling with full integer support and stable shapes.

    The short-history refinement mixes equal probabilities of Uniform{1..T}
    and floor(exp(Uniform[0,log(T+1)))). This changes training exposure only;
    neither the model's parameter prior nor the observed-path law changes.
    """
    if policy not in PREFIX_POLICIES:
        raise ValueError(f"prefix policy must be one of {tuple(PREFIX_POLICIES)}")
    maximum = integer(task["n_observations"], "n_observations")
    rng = np.random.default_rng(integer(seed, "prefix seed", minimum=0))
    exposure = np.zeros(maximum + 1, dtype=np.int64)

    def augment(batch):
        result = dict(batch)
        tokens = np.array(batch["observations"], copy=True)
        lengths = rng.integers(1, maximum + 1, len(tokens))
        if policy == "uniform_log_mixture":
            choose_log = rng.random(len(tokens)) < .5
            log_lengths = np.minimum(maximum, np.floor(np.exp(
                rng.uniform(0.0, np.log(maximum + 1), len(tokens)))).astype(np.int64))
            lengths = np.where(choose_log, log_lengths, lengths)
        mask = np.arange(maximum)[None, :] < lengths[:, None]
        tokens = np.where(mask[..., None], tokens, 0.0)
        result["observations"] = tokens
        result["summary_mask"] = mask
        result["time_context"] = np.column_stack([lengths / maximum,
                                                    np.full(len(tokens), task["dt_years"] * 252.0)]).astype(np.float32)
        exposure[:] += np.bincount(lengths, minlength=maximum + 1)
        return result

    augment.rng, augment.exposure, augment.policy = rng, exposure, policy
    return augment


def build_workflow(task, *, network=None, optimizer=None):
    validate_task(task)
    import bayesflow as bf
    if bf.__version__ != "2.1.0":
        raise RuntimeError("path inference requires the locked BayesFlow 2.1.0 core environment")
    settings = dict(DEFAULT_NETWORK if network is None else network)
    adapter = (bf.Adapter().convert_dtype("float64", "float32")
               .rename("parameters", "inference_variables")
               .rename("observations", "summary_variables")
               .rename("time_context", "inference_conditions"))
    summary = bf.networks.RecurrentNetwork(
        hidden_dim=settings["hidden_dim"], summary_dim=settings["summary_dim"],
        recurrent_type="gru", bidirectional=False, time_axis=1,
        time_embed_dim=4, dropout=0.0, return_sequences=False)
    density = bf.networks.CouplingFlow(
        depth=settings["depth"], transform="spline",
        subnet_kwargs={"widths": settings["widths"]},
        transform_kwargs={"bins": settings["bins"]})
    return bf.BasicWorkflow(adapter=adapter, summary_network=summary, inference_network=density,
                            optimizer=optimizer, standardize=None)


def sample(workflow, task, values, draws, *, lengths=None, seed=0):
    conditions = conditioning_data(values, task, lengths=lengths)
    output = workflow.sample(num_samples=integer(draws, "draws"), conditions=conditions,
                             sample_shape=(), seed=integer(seed, "seed", minimum=0))
    result = inverse_parameters(np.asarray(output["parameters"], dtype=np.float64), task)
    expected = (len(conditions["observations"]), draws, len(task["parameter_names"]))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"invalid posterior sample shape/value: expected {expected}, got {result.shape}")
    return result


def load_checkpoint(directory):
    directory = Path(directory)
    task = validate_task(json.loads((directory / "task.json").read_text()))
    report = json.loads((directory / "training.json").read_text())
    policy = report.get("prefix_policy", "uniform")
    if (report.get("format") != FORMAT or report.get("task_sha256") != fingerprint(task)
            or report.get("model_sha256") != sha256(directory / "model.keras")
            or report.get("normalization") != "fixed_prior_units_no_fitted_standardizer_v1"
            or policy not in PREFIX_POLICIES
            or report.get("history_training") != PREFIX_POLICIES[policy]):
        raise ValueError("checkpoint task/model/normalization contract mismatch")
    workflow = build_workflow(task, network=report["network"])
    workflow.load_estimator(directory / "model.keras")
    return task, workflow, report


def prior_samples(task, count, seed):
    rng = np.random.default_rng(seed)
    bounds, logged = _bounds(task)
    result = rng.uniform(bounds[:, 0], bounds[:, 1], (count, len(bounds)))
    result[..., logged] = np.exp(result[..., logged])
    return result


def diagnostics(samples, truth, prior, *, seed=0):
    """Physical-coordinate diagnostics. No variance expansion is clipped away."""
    samples, truth, prior = (np.asarray(v, dtype=np.float64) for v in (samples, truth, prior))
    if samples.ndim != 3 or samples.shape[0] != len(truth) or samples.shape[2] != truth.shape[1]:
        raise ValueError("posterior/truth shapes do not align")
    mean = samples.mean(axis=1)
    prior_mean, prior_variance = prior.mean(axis=0), prior.var(axis=0)
    mse = np.mean((mean - truth) ** 2, axis=0)
    baseline = np.mean((prior_mean - truth) ** 2, axis=0)
    rng = np.random.default_rng(seed)
    wrong = truth[rng.permutation(len(truth))]
    ranks = np.sum(samples < truth[:, None, :], axis=1)
    u = (ranks + rng.uniform(size=ranks.shape)) / (samples.shape[1] + 1)
    ordered = np.sort(u, axis=0)
    n = len(truth)
    ks = np.maximum(np.max(np.arange(1, n + 1)[:, None] / n - ordered, axis=0),
                    np.max(ordered - np.arange(n)[:, None] / n, axis=0))
    coverage = {}
    for level in (.50, .80, .90, .95):
        lower, upper = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
        hits = (truth >= lower) & (truth <= upper)
        rate = hits.mean(axis=0)
        coverage[str(level)] = {"coverage": rate.tolist(),
                                "binomial_se": np.sqrt(rate * (1 - rate) / n).tolist()}
    lower, upper = np.quantile(samples, [.05, .95], axis=1)
    prior_width = np.diff(np.quantile(prior, [.05, .95], axis=0), axis=0)[0]
    corr = []
    for j in range(truth.shape[1]):
        corr.append(float(np.corrcoef(mean[:, j], truth[:, j])[0, 1])
                    if np.std(mean[:, j]) > 0 and np.std(truth[:, j]) > 0 else None)
    return {"cases": n, "draws_per_case": samples.shape[1], "rank_ks": ks.tolist(),
            "coverage": coverage, "mean_squared_error": mse.tolist(),
            "prior_mean_squared_error": baseline.tolist(),
            "mse_to_prior_mean_ratio": (mse / baseline).tolist(),
            "wrong_pair_mean_squared_error": np.mean((mean - wrong) ** 2, axis=0).tolist(),
            "posterior_mean_truth_correlation": corr,
            "mean_posterior_to_prior_variance": (samples.var(axis=1).mean(axis=0) / prior_variance).tolist(),
            "mean_90_interval_to_prior_width": ((upper - lower).mean(axis=0) / prior_width).tolist(),
            "rank_screen_note": "descriptive marginal diagnostics; no conditional/joint accuracy certification"}
