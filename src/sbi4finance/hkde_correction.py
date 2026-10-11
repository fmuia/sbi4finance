"""Defensive importance correction of the frozen two-parameter HKDE flow.

The proposal mixes the existing conditional flow with the declared uniform prior.
Every row is weighted using the *whole mixture density* and the original refined
Fourier/Gaussian likelihood. Correction targets that numerical posterior; it is
not a retrained or newly validated flow. Finite weighted pools, resampling and
numerical pricing each retain approximation error.
"""

import time

import numpy as np
from scipy.special import logsumexp

from . import hkde_inference as inference, hkde_reference as reference


_PROPOSAL_TAG = 0x484B434F


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _rows(values, *, columns=2, minimum=1, name="parameters"):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != columns or len(values) < minimum
            or not np.isfinite(values).all()):
        raise ValueError(f"finite {name} [N,{columns}], N>={minimum}, required")
    return values


def _weights(values, count):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (count,) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.any(values > 0)):
        raise ValueError("finite nonnegative weights [N] with positive total required")
    values = values / values.max()
    return values / values.sum()


def _context(task, observation):
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (9, 1):
        raise ValueError("one complete observation [9,1] required")
    conditions = inference.conditioning_data(observation[None], task)
    return np.asarray(task["prior_bounds"], dtype=np.float64), conditions


def _inside(parameters, bounds):
    if np.any(parameters <= bounds[:, 0]) or np.any(parameters >= bounds[:, 1]):
        raise ValueError("proposal parameters must be strictly inside declared bounds; no clipping")


def _flow_log_prob(workflow, task, observation, parameters):
    """Raw-unit density, including BayesFlow's adapter and normalizer Jacobians.

    Targets are deliberately flattened [N,1], with repeated [N,9] observations.
    Constrain.log_det_jac sums all non-batch axes in BayesFlow 2.0.14, so passing
    an extra sample axis would give the wrong per-row Jacobian. The public
    approximator already adds logit AND fitted target-standardizer Jacobians.
    """
    bounds, conditions = _context(task, observation)
    parameters = _rows(parameters)
    _inside(parameters, bounds)
    data = {name: parameters[:, d:d+1] for d, name in enumerate(inference.PARAMETER_NAMES)}
    data["observations"] = np.repeat(conditions["observations"], len(parameters), axis=0)
    log_prob = np.asarray(workflow.log_prob(data), dtype=np.float64)
    if (log_prob.shape != (len(parameters),) or np.isnan(log_prob).any()
            or np.isposinf(log_prob).any()):
        raise ValueError("flow log_prob must return finite or -infinity raw-unit values [N]")
    return log_prob


def propose(workflow, task, observation, num_samples, seed, prior_fraction=0.1):
    """IID flow/prior mixture proposals and their full raw-unit log densities.

    N flow draws are generated, then IID Bernoulli-selected rows are replaced
    by independent prior draws. Fixed N avoids variable-count compilation; unused
    flow draws are discarded. Component identities are diagnostics, never a
    substitute for the mixture denominator. Endpoints/nonfinite flow draws stop
    the calculation instead of being clipped or silently redrawn.
    """
    num_samples = _integer(num_samples, "num_samples")
    seed = _integer(seed, "seed", minimum=0)
    if (isinstance(prior_fraction, (bool, np.bool_)) or np.ndim(prior_fraction) != 0
            or not np.isfinite(prior_fraction) or not 0 < prior_fraction < 1):
        raise ValueError("prior_fraction must lie strictly between zero and one")
    bounds, conditions = _context(task, observation)
    streams = [np.random.SeedSequence([_PROPOSAL_TAG, seed, purpose]) for purpose in range(3)]
    start = time.perf_counter()
    is_prior = np.random.default_rng(streams[0]).random(num_samples) < prior_fraction
    prior_rows = np.random.default_rng(streams[1]).uniform(bounds[:, 0], bounds[:, 1], (num_samples, 2))
    network_seed = int(streams[2].generate_state(1, dtype=np.uint32)[0])
    sampled = workflow.sample(num_samples=num_samples, conditions=conditions, seed=network_seed)
    pieces = [np.asarray(sampled[name], dtype=np.float64) for name in inference.PARAMETER_NAMES]
    if any(p.shape != (1, num_samples, 1) or not np.isfinite(p).all() for p in pieces):
        raise ValueError("flow must return matched finite named draws [1,N,1]")
    neural_rows = np.concatenate(pieces, axis=-1)[0]
    _inside(neural_rows, bounds)
    parameters = np.where(is_prior[:, None], prior_rows, neural_rows)
    _inside(parameters, bounds)
    sampled_at = time.perf_counter()
    log_neural = _flow_log_prob(workflow, task, observation, parameters)
    log_prior = -float(np.log(np.diff(bounds, axis=1)[:, 0]).sum())
    log_proposal = np.logaddexp(np.log1p(-prior_fraction) + log_neural,
                               np.log(prior_fraction) + log_prior)
    return {"parameters": parameters, "log_proposal": log_proposal, "is_prior": is_prior,
            "timing_seconds": {"sample": sampled_at-start, "density": time.perf_counter()-sampled_at}}


def correct(parameters, log_proposal, observation, task, *, prices=None):
    """Self-normalized importance weights for the original numerical likelihood.

    Optional prices must be the original refined Fourier expectation labels
    [N,9], in observation-contract order. A surrogate, noisy panel or single
    payoff is not a valid replacement for this declared likelihood calculation.
    Gaussian normalizing constants are retained in log_likelihood. The uniform
    prior density is included before normalization, in the same raw units as q.
    """
    bounds, _ = _context(task, observation)
    parameters = _rows(parameters)
    _inside(parameters, bounds)
    log_proposal = np.asarray(log_proposal, dtype=np.float64)
    if log_proposal.shape != (len(parameters),) or not np.isfinite(log_proposal).all():
        raise ValueError("finite full-mixture log_proposal [N] required")
    if prices is None:
        prices = reference.price_parameters(task, parameters)
    prices = _rows(prices, columns=9, name="prices")
    if len(prices) != len(parameters):
        raise ValueError("one nine-contract likelihood price row per parameter row required")
    sigma = task["context"]["quote_noise_std"]
    observed = np.asarray(observation, dtype=np.float64)[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        log_likelihood = -.5*np.sum(((prices-observed)/sigma)**2, axis=1) - 9*np.log(sigma*np.sqrt(2*np.pi))
    log_prior = -float(np.log(np.diff(bounds, axis=1)[:, 0]).sum())
    log_weight = log_likelihood + log_prior - log_proposal
    normalizer = logsumexp(log_weight)
    if not np.isfinite(normalizer):
        raise ValueError("nonfinite importance normalization")
    weights = np.exp(log_weight-normalizer)
    weights /= weights.sum()
    ess = float(1 / np.sum(weights**2))
    return {"weights": weights, "ess": ess, "ess_fraction": ess/len(parameters),
            "max_weight": float(weights.max()), "log_likelihood": log_likelihood}


def _moments(parameters, weights):
    mean = weights @ parameters
    centered = parameters-mean
    covariance = (centered*weights[:, None]).T @ centered
    std = np.sqrt(np.diag(covariance))
    if not np.isfinite(covariance).all() or np.any(std <= 0):
        raise ValueError("nondegenerate weighted parameter marginals required")
    correlation = float(np.clip(covariance[0, 1]/np.prod(std), -1., 1.))
    return mean, std, covariance, correlation


def summarize(parameters, weights, target_prices, reference_summary=None,
              probabilities=(.05, .5, .95), blocks=8):
    """Weighted joint and target-price summaries, plus finite-pool diagnostics.

    Blocks are independent equal-sized proposal groups from the IID mixture.
    Their self-normalized means/quantiles have finite-block bias. Dividing the
    between-block standard deviation by sqrt(blocks) estimates full-pool MCSE;
    this is a descriptive approximation, not a quantile confidence interval or
    bound on inference/numerical error. No resampling is needed for these means,
    quantiles or reference comparisons.
    """
    parameters = _rows(parameters, minimum=4)
    targets = _rows(target_prices, name="target_prices", minimum=4)
    if targets.shape != parameters.shape:
        raise ValueError("two ordered target prices per paired parameter row required")
    weights = _weights(weights, len(parameters))
    blocks = _integer(blocks, "blocks", minimum=2)
    if len(parameters) % blocks or len(parameters)//blocks < 2:
        raise ValueError("equal independent blocks with at least two rows each required")
    probs = np.asarray(probabilities, dtype=np.float64)
    if (probs.ndim != 1 or not len(probs) or not np.isfinite(probs).all()
            or np.any((probs <= 0) | (probs >= 1))):
        raise ValueError("finite nonempty quantile probabilities strictly inside (0,1) required")
    mean, std, covariance, correlation = _moments(parameters, weights)
    parameter_quantiles = reference.weighted_quantiles(parameters, weights, probs)
    target_quantiles = reference.weighted_quantiles(targets, weights, probs)
    intervals = reference.weighted_quantiles(parameters, weights, [.05, .95])
    target_mean = weights @ targets
    block_means, block_target_means, block_target_quantiles = [], [], []
    for index in np.split(np.arange(len(parameters)), blocks):
        local_weights = _weights(weights[index], len(index))
        block_means.append(local_weights @ parameters[index])
        block_target_means.append(local_weights @ targets[index])
        block_target_quantiles.append(reference.weighted_quantiles(targets[index], local_weights, probs))
    block_means = np.asarray(block_means)
    block_target_means, block_target_quantiles = np.asarray(block_target_means), np.asarray(block_target_quantiles)
    mean_mcse = np.sqrt(len(parameters)/(len(parameters)-1)*np.sum(
        (weights[:, None]*(parameters-mean))**2, axis=0))
    target_mean_mcse = np.sqrt(len(parameters)/(len(parameters)-1)*np.sum(
        (weights[:, None]*(targets-target_mean))**2, axis=0))
    result = {"mean": mean.tolist(), "std": std.tolist(), "covariance": covariance.tolist(),
              "correlation": correlation, "probabilities": probs.tolist(),
              "parameter_quantiles": parameter_quantiles.tolist(), "parameter_intervals_90": intervals.tolist(),
              "target_mean": target_mean.tolist(), "target_quantiles": target_quantiles.tolist(),
              "uncertainty": {"blocks": blocks, "rows_per_block": len(parameters)//blocks,
                "mean_snis_mcse": mean_mcse.tolist(), "target_mean_snis_mcse": target_mean_mcse.tolist(),
                "parameter_mean_block_mcse": (block_means.std(axis=0, ddof=1)/np.sqrt(blocks)).tolist(),
                "target_mean_block_mcse": (block_target_means.std(axis=0, ddof=1)/np.sqrt(blocks)).tolist(),
                "target_quantile_block_mcse": (block_target_quantiles.std(axis=0, ddof=1)/np.sqrt(blocks)).tolist(),
                "block_parameter_means": block_means.tolist(), "block_target_means": block_target_means.tolist(),
                "block_target_quantiles": block_target_quantiles.tolist(),
                "max_abs_block_parameter_mean_deviation_in_posterior_sd": float(np.max(np.abs(block_means-mean)/std)),
                "max_abs_block_target_quantile_deviation_currency": float(np.max(np.abs(block_target_quantiles-target_quantiles))),
                "scope": "IID SNIS delta-method mean SE and independent-block approximate SE; finite-pool/quantile bias and numerical bias are not covered; no support certificate"}}
    if reference_summary is not None:
        ref_mean, ref_std = [np.asarray(reference_summary[key], dtype=float) for key in ("mean", "std")]
        ref_correlation = reference_summary["correlation"]
        ref_targets = np.asarray(reference_summary["target_quantiles"], dtype=float)
        if (ref_mean.shape != (2,) or ref_std.shape != (2,) or np.any(ref_std <= 0)
                or not np.isfinite(ref_mean).all() or not np.isfinite(ref_std).all()
                or ref_correlation is None or not np.isfinite(ref_correlation)
                or ref_targets.shape != (2, len(probs)) or not np.isfinite(ref_targets).all()
                or not np.array_equal(probs, reference_summary["probabilities"])):
            raise ValueError("finite nondegenerate reference with matching quantile probabilities required")
        axes, cdfs = reference_summary["marginal_axes"], reference_summary["marginal_cdfs"]
        if len(axes) != 2 or len(cdfs) != 2:
            raise ValueError("two integrated reference marginal CDFs required")
        masses = []
        for d, (axis, cdf) in enumerate(zip(axes, cdfs, strict=True)):
            axis, cdf = np.asarray(axis, dtype=float), np.asarray(cdf, dtype=float)
            if (axis.ndim != 1 or len(axis) < 3 or cdf.shape != axis.shape
                    or not np.isfinite(axis).all() or not np.isfinite(cdf).all()
                    or np.any(np.diff(axis) <= 0) or np.any(np.diff(cdf) < 0)
                    or not np.allclose(cdf[[0,-1]], [0,1], rtol=0, atol=1e-12)):
                raise ValueError("increasing reference axes and integrated CDFs from zero to one required")
            endpoints = np.interp(intervals[d], axis, cdf, left=0., right=1.)
            masses.append(float(endpoints[1]-endpoints[0]))
        result.update(mean_error_in_ref_sd=((mean-ref_mean)/ref_std).tolist(),
                      relative_std_error=(std/ref_std-1).tolist(),
                      correlation_error=float(correlation-ref_correlation),
                      target_quantile_error_currency=(target_quantiles-ref_targets).tolist(),
                      reference_mass_of_90_interval=masses,
                      mean_mcse_in_ref_sd=(mean_mcse/ref_std).tolist())
    return result


def resample(parameters, weights, num_samples, seed):
    """Joint SIR draws from one finite weighted pool, preserving paired targets.

    Draws are conditionally IID given the pool but share its approximation error.
    Repeated indices are expected and must not be counted as new independent
    proposals or used to claim the original flow passed calibration.
    """
    parameters = _rows(parameters)
    weights = _weights(weights, len(parameters))
    num_samples = _integer(num_samples, "num_samples")
    seed = _integer(seed, "seed", minimum=0)
    indices = np.random.default_rng(seed).choice(len(parameters), size=num_samples, p=weights)
    return parameters[indices].copy()
