"""Descriptive neural-versus-reference comparisons on inspected one-day cases.

Uses all stored importance weights and the conditional drift Normal mixture,
not resampled reference draws. Discrepancy flags were chosen after observing
the study and are explicitly post hoc. They are not acceptance thresholds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import binom, truncnorm

from sbi4finance.path_inference import bank, teaching, teaching_reference


PROBABILITIES = np.array([.05, .5, .95])
POSTHOC_FLAGS = {"absolute_mean_difference_in_reference_sd": .20,
                "absolute_neural90_reference_mass_error": .05}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prior_summary(task):
    means, variances, quantiles = [], [], []
    for (low, high), kind in zip(task["prior_bounds"], task["prior_types"], strict=True):
        if kind == "uniform":
            mean, variance = (low + high) / 2, (high - low) ** 2 / 12
            q = low + PROBABILITIES * (high - low)
        else:
            ratio = np.log(high / low)
            mean = (high - low) / ratio
            variance = (high ** 2 - low ** 2) / (2 * ratio) - mean ** 2
            q = low * np.exp(PROBABILITIES * ratio)
        means.append(mean)
        variances.append(variance)
        quantiles.append(q)
    return {"mean": np.asarray(means), "sd": np.sqrt(variances), "quantiles": np.asarray(quantiles)}


def prior_cdf(value, bounds, kind):
    low, high = bounds
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return float((value - low) / (high - low) if kind == "uniform" else np.log(value / low) / np.log(high / low))


def drift_moments(location, scale, bounds):
    """Stable component moments, independently checked with 64 smooth nodes."""
    low, high = bounds
    lower, upper = (low - location) / scale, (high - location) / scale
    smooth = (upper - lower < .5) & (np.maximum(np.abs(lower), np.abs(upper)) * (upper - lower) < 1)
    mean, variance = np.empty_like(location), np.empty_like(location)
    if np.any(~smooth):
        mean[~smooth], variance[~smooth] = truncnorm.stats(lower[~smooth], upper[~smooth],
                                                       loc=location[~smooth], scale=scale[~smooth], moments="mv")
    if np.any(smooth):
        nodes, weights = np.polynomial.legendre.leggauss(64)
        values = low + (nodes + 1) * (high - low) / 2
        exponent = -.5 * ((values[None, :] - location[smooth, None]) / scale[smooth, None]) ** 2
        normalized = np.exp(exponent - logsumexp(exponent, b=weights, axis=1)[:, None]) * weights
        mean[smooth] = normalized @ values
        variance[smooth] = np.sum(normalized * (values - mean[smooth, None]) ** 2, axis=1)
    if (not np.isfinite(mean).all() or not np.isfinite(variance).all()
            or np.any(variance < 0) or np.any(mean < low) or np.any(mean > high)):
        raise FloatingPointError("reference component drift-moment precision failed")
    return mean, variance


def component_values(payload, parameter, value, drift_bounds):
    """Conditional CDF integrands; drift uses its entire truncated mixture."""
    if parameter < payload["parameters_nondrift"].shape[1]:
        return (payload["parameters_nondrift"][:, parameter] <= value).astype(float)
    low, high = drift_bounds
    if value <= low:
        return np.zeros(len(payload["weights"]))
    if value >= high:
        return np.ones(len(payload["weights"]))
    location, scale = payload["drift_location"], payload["drift_scale"]
    values = truncnorm.cdf(value, (low - location) / scale, (high - location) / scale,
                          loc=location, scale=scale)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise FloatingPointError("reference conditional drift CDF precision failed")
    return values


def weighted_mcse(weights, values):
    """Plug-in self-normalized IS MCSE; not a guarantee of tail exploration."""
    center = weights @ values
    return float(np.sqrt(np.sum(weights ** 2 * (values - center) ** 2)))


def component_summary(payload, task):
    weights = payload["weights"]
    mean_mu, variance_mu = drift_moments(payload["drift_location"], payload["drift_scale"], task["prior_bounds"][-1])
    mean_integrands = np.column_stack((payload["parameters_nondrift"], mean_mu))
    second_integrands = np.column_stack((payload["parameters_nondrift"] ** 2, mean_mu ** 2 + variance_mu))
    mean = weights @ mean_integrands
    variance = weights @ second_integrands - mean ** 2
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise FloatingPointError("degenerate reference marginal variance")
    quantiles = np.empty((len(mean), 3))
    for col in range(len(mean) - 1):
        values = payload["parameters_nondrift"][:, col]
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        quantiles[col] = values[order[np.minimum(np.searchsorted(cumulative, PROBABILITIES), len(order) - 1)]]
    for j, probability in enumerate(PROBABILITIES):
        low, high = task["prior_bounds"][-1]
        for _ in range(44):
            midpoint = (low + high) / 2
            cdf = weights @ component_values(payload, len(mean) - 1, midpoint, task["prior_bounds"][-1])
            if cdf < probability:
                low = midpoint
            else:
                high = midpoint
        quantiles[-1, j] = (low + high) / 2
    return {"mean": mean, "sd": np.sqrt(variance), "quantiles": quantiles,
            "mean_mcse": np.array([weighted_mcse(weights, values) for values in mean_integrands.T])}


def likelihood_upper_bound(task):
    """One-day bound from the positive deterministic first internal variance."""
    if task["model"] == "heston":
        initial, steps = task["fixed"]["initial_variance"], task["numerics"]["substeps"]
    else:
        initial = (task["fixed"]["variance_level"] if task["model"] == "rough-known" else
                   task["prior_bounds"][task["parameter_names"].index("short_variance")][0])
        steps = task["numerics"]["observation_substeps"]
    rho = task["prior_bounds"][task["parameter_names"].index("rho")]
    conditional_variance_minimum = (1 - max(abs(r) for r in rho) ** 2) * initial * task["dt_years"] / steps
    return float(1 / np.sqrt(2 * np.pi * conditional_variance_minimum))


def load_reference_payload(directory, summary, task, observations):
    path = (directory / summary["payload"]["file"]).resolve()
    if not path.is_relative_to(directory.resolve()) or sha256(path) != summary["payload"]["sha256"]:
        raise ValueError("reference importance payload hash/path mismatch")
    with np.load(path, allow_pickle=False) as source:
        payload = {k: source[k] for k in source.files}
    count, dimension = summary["samples"], len(task["parameter_names"])
    expected = {"parameters_nondrift": (count, dimension - 1), "weights": (count,),
                "drift_location": (count,), "drift_scale": (count,), "observations": (1,)}
    if (set(payload) != set(expected) | {"posterior_draws"}
            or any(payload[k].shape != shape for k, shape in expected.items())
            or any(not np.isfinite(x).all() for x in payload.values())
            or payload["posterior_draws"].ndim != 2 or payload["posterior_draws"].shape[1] != dimension
            or not np.array_equal(payload["observations"], observations)):
        raise ValueError("reference payload shape/data differs from bound one-day observations")
    weights = payload["weights"]
    bounds = np.asarray(task["prior_bounds"])
    if (np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
            or np.any(payload["drift_scale"] <= 0)
            or np.any(payload["parameters_nondrift"] < bounds[:-1, 0])
            or np.any(payload["parameters_nondrift"] > bounds[:-1, 1])):
        raise ValueError("invalid reference weights, drift scale or prior support")
    if not np.isclose(1 / np.sum(weights ** 2), summary["ess"], rtol=1e-10):
        raise ValueError("reference ESS differs from stored weights")
    if not np.isclose(weights.max(), summary["maximum_weight"], rtol=1e-10):
        raise ValueError("reference maximum weight differs from stored weights")
    largest_likelihood = float(weights.max() * count * np.exp(summary["log_evidence_estimate"]))
    if largest_likelihood > likelihood_upper_bound(task) * (1 + 1e-10):
        raise ValueError("reference likelihood exceeds the exact one-day Gaussian bound")
    actual = component_summary(payload, task)
    if (not np.allclose(actual["mean"], summary["posterior_mean"], atol=1e-9, rtol=1e-8)
            or not np.allclose(actual["sd"], summary["posterior_sd"], atol=1e-9, rtol=1e-8)
            or not np.allclose(actual["quantiles"], summary["posterior_quantiles"], atol=1e-9, rtol=1e-8)):
        raise ValueError("reference weighted moments differ from summary")
    return payload, actual, largest_likelihood


def pool_runs(payloads, summaries):
    """Pool independent prior IS draws with their unnormalized evidence masses."""
    log_masses = np.array([np.log(r["samples"]) + r["log_evidence_estimate"] for r in summaries])
    shares = np.exp(log_masses - logsumexp(log_masses))
    output = {key: np.concatenate([p[key] for p in payloads], axis=0)
              for key in ("parameters_nondrift", "drift_location", "drift_scale")}
    output["weights"] = np.concatenate([share * p["weights"] for share, p in zip(shares, payloads, strict=True)])
    return output, shares


def quantile_order_intervals(draws, bounds):
    """Distribution-free order-statistic 95% intervals for iid-draw quantiles."""
    values = np.sort(draws, axis=0)
    n, dimension = draws.shape
    output = np.empty((dimension, 3, 2))
    for j, probability in enumerate(PROBABILITIES):
        # K = number below the population q quantile is Binomial(n,q).
        lower = int(binom.ppf(.025, n, probability))
        upper = int(binom.ppf(.975, n, probability)) + 1
        output[:, j, 0] = bounds[:, 0] if lower == 0 else values[lower - 1]
        output[:, j, 1] = bounds[:, 1] if upper > n else values[upper - 1]
    return output


def compare_draws(task, draws, payload, reference_summary, replicate_payloads=()):
    draws = np.asarray(draws, dtype=float)
    dimension = len(task["parameter_names"])
    bounds = np.asarray(task["prior_bounds"])
    if (draws.ndim != 2 or draws.shape[1] != dimension or len(draws) < 20 or not np.isfinite(draws).all()
            or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
        raise ValueError("invalid one-day neural posterior draws")
    n = len(draws)
    neural_mean, neural_sd = draws.mean(axis=0), draws.std(axis=0, ddof=1)
    neural_mean_se = neural_sd / np.sqrt(n)
    neural_quantiles = np.quantile(draws, PROBABILITIES, axis=0).T
    order_intervals = quantile_order_intervals(draws, bounds)
    prior = prior_summary(task)
    result = {}
    for col, name in enumerate(task["parameter_names"]):
        low, high = neural_quantiles[col, [0, 2]]
        lower_values = component_values(payload, col, low, bounds[-1])
        upper_values = component_values(payload, col, high, bounds[-1])
        mass_values = upper_values - lower_values
        mass = float(payload["weights"] @ mass_values)
        mean_difference = float((neural_mean[col] - reference_summary["mean"][col]) / reference_summary["sd"][col])
        mass_mcse = weighted_mcse(payload["weights"], mass_values)
        reference_mass_replicates = [float(p["weights"] @ (component_values(p, col, high, bounds[-1])
                                                         - component_values(p, col, low, bounds[-1])))
                                     for p in replicate_payloads]
        result[name] = {
            "reference_mean": float(reference_summary["mean"][col]), "reference_sd": float(reference_summary["sd"][col]),
            "reference_mean_is_mcse": float(reference_summary["mean_mcse"][col]),
            "reference_quantiles05_50_95": reference_summary["quantiles"][col].tolist(),
            "neural_mean": float(neural_mean[col]), "neural_sd": float(neural_sd[col]),
            "neural_mean_iid_mcse": float(neural_mean_se[col]),
            "mean_difference_in_reference_sd": mean_difference,
            "mean_difference_combined_mcse_in_reference_sd": float(np.hypot(neural_mean_se[col], reference_summary["mean_mcse"][col]) / reference_summary["sd"][col]),
            "neural_sd_to_reference_sd": float(neural_sd[col] / reference_summary["sd"][col]),
            "neural_quantiles05_50_95": neural_quantiles[col].tolist(),
            "quantile_differences_in_reference_sd": ((neural_quantiles[col] - reference_summary["quantiles"][col]) / reference_summary["sd"][col]).tolist(),
            "neural_quantile_order_statistic_95pct_intervals": order_intervals[col].tolist(),
            "reference_cdf_at_neural90_endpoints": [float(payload["weights"] @ lower_values), float(payload["weights"] @ upper_values)],
            "reference_mass_inside_neural90": mass, "reference_mass_is_mcse": mass_mcse,
            "reference_mass_inside_neural90_by_independent_run": reference_mass_replicates,
            "iid_asymptotic_nominal90_probability_mcse": float(np.sqrt(.9 * .1 / n)),
            "prior_mean": float(prior["mean"][col]), "prior_sd": float(prior["sd"][col]),
            "prior_quantiles05_50_95": prior["quantiles"][col].tolist(),
            "prior_mass_inside_neural90": (prior_cdf(high, bounds[col], task["prior_types"][col])
                                           - prior_cdf(low, bounds[col], task["prior_types"][col])),
            "reference_mean_minus_prior_mean_in_prior_sd": float((reference_summary["mean"][col] - prior["mean"][col]) / prior["sd"][col]),
            "neural_mean_minus_prior_mean_in_prior_sd": float((neural_mean[col] - prior["mean"][col]) / prior["sd"][col]),
            "reference_sd_to_prior_sd": float(reference_summary["sd"][col] / prior["sd"][col]),
            "posthoc_descriptive_flags": {
                "mean_difference_at_least_point2_reference_sd": abs(mean_difference) >= POSTHOC_FLAGS["absolute_mean_difference_in_reference_sd"],
                "neural90_reference_mass_error_at_least_point05": abs(mass - .9) >= POSTHOC_FLAGS["absolute_neural90_reference_mass_error"]}}
    return result


def run(replay, reference_directory, out, *, cases=(0, 1)):
    replay, reference_directory, out = Path(replay).resolve(), Path(reference_directory).resolve(), Path(out)
    if out.exists():
        raise FileExistsError("comparison output must be new")
    task, bundle, arrays = teaching.load_cache(replay)
    teaching_reference.validate_task(task)
    if 1 not in bundle["cached_lengths"]:
        raise ValueError("verified replay has no cached one-day neural draws")
    report_path = reference_directory / "reference-summary.json"
    reference = json.loads(report_path.read_text())
    source = reference["source"]
    evaluation = json.loads((replay / "evaluation.json").read_text())
    if (reference["task_sha256"] != bank.task_fingerprint(task)
            or bank.task_fingerprint(reference["task"]) != bank.task_fingerprint(task)
            or source["bank_manifest_sha256"] != evaluation["bank_manifest_sha256"]
            or source.get("split") != "validation" or source.get("reserved_test_evaluated") is not False
            or source.get("unseen_validation_evaluated") is not False):
        raise ValueError("reference task/bank/development cohort identity mismatch")
    if ((source.get("kind") == "development_replay" or "source_case_panel_sha256" in source)
            and source.get("source_case_panel_sha256") != bundle["source_case_panel_sha256"]):
        raise ValueError("reference and replay source case-panel hash mismatch")
    if (not cases or len(set(cases)) != len(cases) or any(not isinstance(i, int) or not 0 <= i < 1024 for i in cases)):
        raise ValueError("unique inspected validation case indices required")
    rows = {str(identity): row for row, identity in enumerate(arrays["trajectory_id"])}
    output = {"format": "one_day_neural_reference_comparison_v1", "model": task["model"],
              "task_sha256": bank.task_fingerprint(task), "checkpoint_sha256": bundle["checkpoint_sha256"],
              "source_case_panel_sha256": bundle["source_case_panel_sha256"],
              "replay_bundle_sha256": sha256(replay / "bundle.json"), "reference_summary_sha256": sha256(report_path),
              "reference_comparison_source_sha256": sha256(__file__),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "length": 1, "posthoc_descriptive_flag_values": POSTHOC_FLAGS,
              "scientific_acceptance": False, "reserved_test_evaluated": False, "unseen_validation_evaluated": False,
              "pooling": "Two largest independent prior-importance runs; pooled by sample-count times estimated evidence, not reference resampling",
              "uncertainty": "Neural mean SE and quantile order intervals assume iid conditional neural draws. Nominal90 probability SE is asymptotic under equality of neural/reference laws. IS MCSE is a plug-in estimate and can miss unexplored tails; independent-run mass estimates are also retained.",
              "limitations": "Two inspected one-day cases only. Posthoc flags describe discrepancies; they do not define acceptance or establish general conditional, joint, or long-history accuracy.",
              "cases": {}}
    for index in cases:
        case = reference["cases"][str(index)]
        if (case["trajectory_id"] not in rows
                or not case["trajectory_id"].endswith(f":validation:{index}")):
            raise ValueError("reference trajectory identity absent from replay")
        row = rows[case["trajectory_id"]]
        observations = arrays["observations"][row, :1, 0]
        if teaching_reference.observations_sha256(arrays["observations"][row, :, 0]) != case["full_observations_sha256"]:
            raise ValueError("reference full observations hash mismatch")
        current = case["lengths"]["1"]
        recomputed_precision = teaching_reference.compare_runs(current["runs"])
        if (current["comparison"]["status"] != "short_history_precision_checks_pass"
                or recomputed_precision["status"] != "short_history_precision_checks_pass"):
            raise ValueError("reference has not passed fixed one-day precision checks")
        largest_count = max(r["samples"] for r in current["runs"])
        selected = [r for r in current["runs"] if r["samples"] == largest_count]
        if len(selected) < 2 or len({r["seed"] for r in selected}) != len(selected):
            raise ValueError("two independent largest reference runs required")
        selected = selected[:2]
        payloads, likelihood_maxima = [], []
        for item in selected:
            if (item["task_sha256"] != output["task_sha256"] or item["length"] != 1
                    or item["observations_sha256"] != teaching_reference.observations_sha256(observations)
                    or item["trajectory_id"] != case["trajectory_id"] or item["parameter_names"] != task["parameter_names"]):
                raise ValueError("reference run task/observations/trajectory identity mismatch")
            payload, _summary, maximum = load_reference_payload(reference_directory, item, task, observations)
            payloads.append(payload)
            likelihood_maxima.append(maximum)
        pooled, shares = pool_runs(payloads, selected)
        summary = component_summary(pooled, task)
        output["cases"][str(index)] = {
            "trajectory_id": case["trajectory_id"], "observations_sha256": teaching_reference.observations_sha256(observations),
            "neural_draws": len(arrays["draws_1"][row]), "reference_draws_before_importance_weighting": sum(x["samples"] for x in selected),
            "reference_pooled_ess": float(1 / np.sum(pooled["weights"] ** 2)),
            "reference_runs": [{"seed": x["seed"], "payload": x["payload"], "pool_share": float(share)}
                               for x, share in zip(selected, shares, strict=True)],
            "reference_precision": recomputed_precision,
            "global_one_day_likelihood_upper_bound": likelihood_upper_bound(task),
            "largest_stored_component_likelihood": max(likelihood_maxima),
            "parameters": compare_draws(task, arrays["draws_1"][row], pooled, summary, payloads)}
    output["status"] = "descriptive_comparison_complete_no_scientific_acceptance"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    output = run(args.replay, args.reference, args.out, cases=args.cases)
    print(json.dumps({"status": output["status"], "model": output["model"], "cases": list(output["cases"])}))


if __name__ == "__main__":
    main()
