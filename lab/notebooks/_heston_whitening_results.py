"""Read-only plotting evidence for the matched historical whitening pilot.

Only NumPy and the standard library are used. This does not load a simulator,
bank, checkpoint or posterior chain, and does not fit preprocessing on viewers.
"""

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np


NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"]
BOUNDS = [[.01, .16], [.1, 10.], [.01, .16], [.1, 3.], [-.95, -.1], [.5, 2.]]
MODES = ("diagonal", "zca")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate whitening-evidence key: " + key)
        result[key] = value
    return result


def _finite(value, shape, label):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(label + " must be numeric") from error
    _require(array.shape == shape and np.isfinite(array).all(), label + " has invalid shape or nonfinite values")
    return array


def _public(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _require(key not in {"draws", "samples", "row_id", "row_ids", "physical_rows", "bank_rows", "theta"},
                     "only compact noise-scale plotting coordinates may be published")
            _public(item)
    elif isinstance(value, list):
        for item in value:
            _public(item)
    elif isinstance(value, str):
        _require(not value.startswith(("/", "~", "file:", "\\"))
                 and not any(part in value.lower() for part in ("/users/", "/scratch/", "/home/", "../")),
                 "private or escaping paths are not public plotting evidence")
    elif isinstance(value, (int, float)):
        _require(np.isfinite(value), "nonfinite values are not public plotting evidence")


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(payload).hexdigest()


def load_results(root):
    """Load the frozen 54-quote CouplingFlow comparison, not the host estimator."""
    try:
        record = json.loads((Path(root) / "results.json").read_text(), object_pairs_hook=_unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))
        _validate(record)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("incomplete or malformed historical whitening evidence") from error
    return record


def _validate(record):
    _public(record)
    _require(record["format"] == "heston_whitening_lesson_v1" and record["schema_version"] == 1
             and record["scientific_acceptance"] is False, "unsupported or incorrectly accepted whitening evidence")
    study, training, population = record["study"], record["training"], record["population"]
    _require(study["date"] == "2026-11-10" and study["quote_count"] == 54
             and study["parameter_names"] == NAMES and study["prior_bounds"] == BOUNDS
             and study["distinct_from_current_63_quote_flow_matching"] is True,
             "the historical pilot must not be relabelled as the current 63-quote estimator")
    _require(study["observation_model"] == {"family": "gaussian", "sd": "c*h_i", "independent": True,
                                           "c_bounds": [.5, 2.]}, "the noise law or c units changed")
    _require(study["conditioning"] == "noisy quotes divided by fixed observed half-spreads"
             and study["architecture"] == {"family": "CouplingFlow", "design": "spline", "depth": 6,
                                           "conditioner_widths": [128, 128], "dropout": 0.},
             "the controlled spline architecture or input coordinates changed")
    expected = {"rows": 10000, "monitor_rows": 100, "epochs": 200, "optimizer_updates": 8000,
                "batch_size": 256, "tail_batch_size": 16, "presentations": 2000000, "seed": 9011,
                "noise": "fixed noisy rows repeated each pass", "checkpoint_selection": "predeclared_final",
                "optimizer": {"class_name": "Adam", "schedule": {"class_name": "CosineDecay",
                    "initial_learning_rate": .0005, "decay_steps": 8000, "alpha": .1}}}
    _require(training == expected, "the fixed paired training budget or checkpoint rule changed")
    _require(record["pairing"] == {"only_contract_differences": ["mode", "preprocessing_fingerprint"],
                                   "same_initial_state": True, "same_training_inputs_and_targets": True,
                                   "same_evaluation_datasets_and_order": True}, "the preprocessing-only control is not matched")
    _require(population["case_count"] == 512 and population["posterior_draws_per_case"] == 512
             and population["seed"] == 261112, "the declared common evaluation population changed")
    truth = _finite(population["c_truth"], (512,), "shared c truths")
    _require(np.all((truth >= .5) & (truth <= 2.)), "generating c values violate the frozen prior")
    sources = record["source_bindings"]
    _require(isinstance(sources, dict) and len(sources) >= 18, "source identities are missing")
    for label, value in sources.items():
        _require(re.fullmatch("[a-z_]+", label) is not None
                 and isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
                 "source identities must be logical labels and SHA-256 values")
    _require(sources["evaluation_c"] == _fingerprint(population["c_truth"]), "the paired generating c cache changed")
    _require(set(record["arms"]) == set(MODES), "both preprocessing arms are required")
    expected_sbc = {"diagonal": [False, True, False, True, False, True], "zca": [False] * 6}
    for mode in MODES:
        arm = record["arms"][mode]
        q = _finite(arm["c_q05_q50_q95"], (3, 512), mode + " c quantiles")
        _require(np.all(np.diff(q, axis=0) >= 0) and np.all(q[2] > q[0]), "posterior quantiles are out of order")
        metrics = {"median_rmse": np.sqrt(np.mean((q[1] - truth) ** 2)),
                   "truth_median_correlation": np.corrcoef(truth, q[1])[0, 1],
                   "median_width90": np.median(q[2] - q[0]),
                   "coverage90": np.mean((q[0] <= truth) & (truth <= q[2]))}
        _require(set(arm["metrics"]) == set(metrics)
                 and all(np.isclose(arm["metrics"][key], value, rtol=1e-12, atol=1e-12)
                         for key, value in metrics.items()), "metrics differ from the displayed paired quantiles")
        _require(arm["sbc_passed"] == expected_sbc[mode], "the original calibration failures must remain visible")
        coverage = _finite(arm["physical_coverage90"], (5,), mode + " physical coverage")
        _require(np.all((coverage >= 0) & (coverage <= 1)), "invalid descriptive coverage")
        if mode == "zca":
            _require(np.max(coverage) < .62 and metrics["coverage90"] < .75,
                     "the final whitened pilot is not a calibrated posterior")
        preprocessing = arm["preprocessing"]
        _require(preprocessing["mode"] == mode and preprocessing["fit_rows"] == 10000
                 and preprocessing["fit_split"] == "train" and preprocessing["all_features_retained"] is True
                 and preprocessing["eigenvalues_floored"] == 0,
                 "preprocessing must be frozen, training-only and full rank")
        condition = preprocessing["covariance_condition_number"]
        _require(np.isfinite(condition) and (condition > 1e6 if mode == "diagonal" else abs(condition - 1) < 1e-6),
                 "the frozen input-conditioning summaries differ")
        checkpoint = arm["checkpoint"]
        _require(checkpoint["epoch"] == 200 and checkpoint["optimizer_updates"] == 8000
                 and checkpoint["selection"] == "predeclared_final"
                 and checkpoint["best_validation_weights_saved"] is False,
                 "the reported comparison uses final weights, not an earlier selected checkpoint")
        _require(type(checkpoint["validation_min_epoch"]) is int and 1 <= checkpoint["validation_min_epoch"] <= 200
                 and np.isfinite(checkpoint["validation_min_loss"])
                 and np.isfinite(checkpoint["final_validation_loss"]), "invalid loss-curve qualification")
    _require(record["arms"]["zca"]["checkpoint"]["validation_min_epoch"] == 83,
             "the recorded ZCA validation minimum changed")
