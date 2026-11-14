"""Validate compact, read-only evidence for the rough-Bergomi closing section.

The loader uses only the standard library and NumPy. It never loads a bank,
network or posterior chain, and performs no pricing, sampling or writes. The
record contains market aggregates and synthetic recovery coordinates, not
simulated quote payloads or raw joint draws. Saved quantiles describe the
original joint posterior draws, not a repaired noise
conditional. The reference is a fixed-emulator posterior, not exact Bergomi.
"""

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np


PARAMETER_NAMES = ("hurst", "eta", "rho", "short_variance", "long_variance", "decay", "c")
PARAMETER_BOUNDS = ((.02, .35), (.5, 3.5), (-.99, -.2), (.01, .16),
                    (.01, .16), (.25, 8.), (.5, 50.))
METHODS = ("mcmc", "amortizer")
MEAN_FUNCTION_SHA256 = "fbc7c5ad60205112a42b739a3ebb3a3fbb5f36a3f8b46723674895c15966053a"
GEOMETRY_FLOATS = ("observed_raw", "half_spread_raw", "maturity_years", "strike_raw",
                   "forward_raw", "discount", "log_moneyness", "standardized_log_moneyness")
GEOMETRY_KEYS = frozenset((*GEOMETRY_FLOATS, "option_type", "expiry_index", "expiry_labels"))


def canonical_sha256(value):
    """SHA-256 of canonical JSON; this is an identity check, not a signature."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate result key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def _finite(value, shape, label, *, positive=False, nonnegative=False):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("non-numeric values in " + label) from error
    _require(array.shape == shape and np.isfinite(array).all(),
             f"{label} must have shape {shape} with finite values")
    if positive:
        _require(np.all(array > 0), label + " must be positive")
    if nonnegative:
        _require(np.all(array >= 0), label + " must be nonnegative")
    return array


def _quantiles(value, columns, label):
    array = _finite(value, (3, columns), label)
    _require(np.all(np.diff(array, axis=0) >= 0), label + " quantiles are out of order")
    return array


def _hash(value, label):
    _require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
             label + " must be a lowercase SHA-256 digest")


def _safe_public_tree(value, label="record"):
    """Reject private paths, unreviewed raw draws and non-finite JSON values."""
    if isinstance(value, dict):
        forbidden = {"draws", "theta", "samples", "posterior_draws", "physical_draws", "bank_rows",
                     "row_ids", "parameter_rows", "model_weights", "chain_values", "raw_chain"}
        for key, item in value.items():
            _require(key not in forbidden, "raw/private payload is not compact evidence: " + label + "." + key)
            _safe_public_tree(item, label + "." + key)
            if key.endswith("sha256"):
                _hash(item, label + "." + key)
    elif isinstance(value, list):
        for item in value:
            _safe_public_tree(item, label)
    elif isinstance(value, str):
        lowered = value.lower()
        _require(not value.startswith(("/", "~", "file:", "\\"))
                 and not re.match(r"^[A-Za-z]:[\\/]", value)
                 and not any(fragment in lowered for fragment in
                             ("/users/", "/home/", "/scratch/", "../", "oscar:/")),
                 "private or escaping path in " + label)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _require(np.isfinite(value), "non-finite number in " + label)


def _same_numeric(actual, expected, label):
    expected = _finite(expected, (63,), "expected " + label)
    _require(np.allclose(actual, expected, rtol=2e-14, atol=1e-12),
             "notebook and capstone " + label + " differ")


def _check_notebook_geometry(geometry, expected_contracts, expected_spreads,
                             expected_observations, expected_design):
    if expected_spreads is not None:
        _same_numeric(geometry["half_spread_raw"], expected_spreads, "half-spreads")
    if expected_observations is not None:
        _same_numeric(geometry["observed_raw"], expected_observations, "observations")
    if expected_design is not None:
        _require(isinstance(expected_design, dict), "expected_design must be a mapping")
        if "selected_quotes" in expected_design:
            _require(expected_contracts is None, "supply contracts or a selected-quotes design, not both")
            expected_contracts = expected_design["selected_quotes"]
        else:
            _require(set(expected_design) <= GEOMETRY_KEYS and bool(expected_design),
                     "expected_design must contain geometry fields or selected_quotes")
            for key, value in expected_design.items():
                if key in GEOMETRY_FLOATS:
                    _same_numeric(geometry[key], value, key)
                else:
                    _require(geometry[key] == list(value), "notebook and capstone " + key + " differ")
    if expected_contracts is None:
        return
    _require(isinstance(expected_contracts, (list, tuple)) and len(expected_contracts) == 63,
             "the notebook must supply all 63 ordered contracts")
    # Host-notebook selected quotes retain these actual contracts and carry values.
    fields = {"strike": "strike_raw", "forward": "forward_raw", "discount": "discount",
              "maturity": "maturity_years", "mid": "observed_raw", "half_spread": "half_spread_raw"}
    for source, destination in fields.items():
        _same_numeric(geometry[destination], [quote[source] for quote in expected_contracts], destination)
    _require(geometry["option_type"] == [quote["option_type"] for quote in expected_contracts],
             "notebook and capstone option types or contract order differ")
    expiries = [quote["expiration"] for quote in expected_contracts]
    labels = sorted(set(expiries))
    _require(geometry["expiry_labels"] == labels
             and geometry["expiry_index"] == [labels.index(expiry) for expiry in expiries],
             "notebook and capstone expiry order differs")


def load_results(root, *, expected_contracts=None, expected_spreads=None,
                 expected_observations=None, expected_design=None):
    """Return validated compact evidence, optionally bound to the host's quotes.

    ``root`` is the directory containing ``results.json``. Optional spreads and
    observations are raw price units. ``expected_contracts`` is the host design's
    ordered ``selected_quotes``; ``expected_design`` may instead be that complete
    design or a mapping of the compact geometry fields. No private data is read.
    """
    directory = Path(root).resolve()
    path = directory / "results.json"
    _require(path.resolve().parent == directory, "results.json must remain inside the evidence directory")
    try:
        record = json.loads(path.read_text(), object_pairs_hook=_unique_keys,
                            parse_constant=_invalid_constant)
        _validate(record, expected_contracts, expected_spreads, expected_observations, expected_design)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("incomplete or malformed Bergomi capstone evidence") from error
    return record


def _validate(record, expected_contracts, expected_spreads, expected_observations, expected_design):
    # The remaining checks bind the scientific task as well as the plotted data.
    _safe_public_tree(record)
    allowed = {"format", "schema_version", "status", "observation_model", "host_heston_c_bounds",
               "comparison_heston_c_bounds", "parameter_names", "parameter_bounds", "task", "geometry",
               "geometry_sha256", "fit_comparison", "posteriors", "predictives", "predictive_protocol",
               "training", "reference", "source_bindings", "numerical_qualification_passed",
               "scientific_acceptance", "scope", "caveats", "recovery"}
    _require(set(record) == allowed, "only the reviewed compact aggregate fields may be published")
    _require(record.get("format") == "bergomi_capstone_results_v2" and record.get("schema_version") == 2,
             "unsupported Bergomi capstone record")
    _require(record.get("status") == "development"
             and record.get("scientific_acceptance") is False
             and record.get("numerical_qualification_passed") is False,
             "the saved experiment remains development evidence, not qualified acceptance")
    _require(tuple(record["parameter_names"]) == PARAMETER_NAMES
             and record["parameter_bounds"] == [list(pair) for pair in PARAMETER_BOUNDS],
             "Bergomi parameter identities or immutable prior bounds differ")
    geometry = record["geometry"]
    _require(set(geometry) == GEOMETRY_KEYS, "exact actual63 geometry fields are required")
    arrays = {name: _finite(geometry[name], (63,), "geometry " + name,
                           positive=name not in {"log_moneyness", "standardized_log_moneyness"})
              for name in GEOMETRY_FLOATS}
    _require(np.all(arrays["discount"] <= 1.05), "discount factors are not raw discount factors")
    _require(np.allclose(arrays["log_moneyness"],
                        np.log(arrays["strike_raw"] / arrays["forward_raw"]), rtol=0, atol=1e-12),
             "log-moneyness and raw contract geometry disagree")
    _require(np.allclose(arrays["standardized_log_moneyness"],
                        arrays["log_moneyness"] / (.16 * np.sqrt(arrays["maturity_years"])),
                        rtol=0, atol=1e-9), "fixed-16% standardized moneyness differs")
    _require(geometry["expiry_index"] == np.repeat(np.arange(7), 9).tolist(),
             "the seven ordered nine-contract expiries are required")
    labels = geometry["expiry_labels"]
    _require(len(labels) == 7 and len(set(labels)) == 7 and labels == sorted(labels),
             "seven distinct ordered expiry labels are required")
    _require(len(geometry["option_type"]) == 63
             and set(geometry["option_type"]) <= {"put", "call"}, "invalid actual63 option types")
    for name in ("maturity_years", "forward_raw", "discount"):
        panel = arrays[name].reshape(7, 9)
        _require(np.allclose(panel, panel[:, :1], rtol=0, atol=1e-12),
                 "carry or maturity varies inside a single expiry")
    _require(np.all(np.diff(arrays["maturity_years"].reshape(7, 9)[:, 0]) > 0),
             "expiry maturities must increase")
    _require(np.all(np.diff(arrays["strike_raw"].reshape(7, 9), axis=1) > 0),
             "contract strike order must increase within every expiry")
    _require(record["geometry_sha256"] == canonical_sha256(geometry), "geometry checksum differs")
    _check_notebook_geometry(geometry, expected_contracts, expected_spreads,
                             expected_observations, expected_design)
    noise = record["observation_model"]
    _require(noise["family"] == "gaussian" and noise["formula"] == "Y_i=P_i(phi)+c*h_i*Z_i"
             and noise["c_bounds"] == [.5, 50.] and noise["h_units"] == "raw price half-spread"
             and noise["independent"] is True,
             "the capstone requires independent Gaussian c-times-half-spread noise in raw prices")
    task = record["task"]
    _require(task["id"] == "actual63_fixed_emulator_noise_task_v1"
             and task["mean_function"] == "fixed_deterministic_price_emulator"
             and task["conditioning"] == "whitened_Y_over_h"
             and tuple(task["target_names"]) == PARAMETER_NAMES,
             "the seven-target actual63 frozen-emulator task is required")
    _hash(task["fingerprint"], "task fingerprint")
    _require(record["host_heston_c_bounds"] == [.5, 2.]
             and record["comparison_heston_c_bounds"] == [.5, 50.],
             "host and broad-c comparison priors must remain distinct")
    training = record["training"]
    expected = {"rows": 200000, "epochs": 300, "batch_size": 256,
                "optimizer_updates": 234600, "checkpoint_epoch": 300}
    _require(all(training[key] == value for key, value in expected.items()),
             "the primary saved final-300 checkpoint uses 200k rows and the recorded update budget")
    _require(training["checkpoint_role"] == "primary_final",
             "the capstone must use the primary final checkpoint")
    _require(training["seed"] == 9011 and training["preprocessing_fit"] == "training rows only"
             and training["noise_refresh"] == "fixed offline noise realization",
             "training-only preprocessing and the offline-noise protocol must remain explicit")
    sources = record["source_bindings"]
    _require(isinstance(sources, dict) and bool(sources), "source hash bindings are required")
    for label, digest in sources.items():
        _require(re.fullmatch("[a-z][a-z0-9_]*", label) is not None,
                 "source identities must be logical names, not paths")
        _hash(digest, "source identity " + label)
    _require(sources["geometry"] == record["geometry_sha256"],
             "geometry and source identity differ")
    _require(sources["control_task"] == task["fingerprint"]
             and sources["observations"] == canonical_sha256(geometry["observed_raw"])
             and sources["half_spreads"] == canonical_sha256(geometry["half_spread_raw"]),
             "task, quote and half-spread source identities differ")
    _require(sources["same_emulator_pricer"] == MEAN_FUNCTION_SHA256,
             "the immutable saved emulator mean-function identity differs")
    _require(training["mean_function_sha256"] == sources["same_emulator_pricer"],
             "the amortizer must target the shared frozen mean function")
    reference = record["reference"]
    _require(reference["fixed_emulator"] is True and reference["exact_bergomi"] is False
             and reference["mean_function_sha256"] == sources["same_emulator_pricer"],
             "MCMC must target the matching approximate fixed-emulator posterior")
    _require(reference["sampler"] == "PyMC NUTS" and reference["chains"] == 4
             and reference["draws_per_chain"] == 2000
             and reference["balanced_draws_per_chain"] == 256,
             "the recorded four-chain MCMC and balanced predictive budget are required")
    requirements = {"r_hat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400}
    _require(reference["requirements"] == requirements,
             "the recorded reference diagnostic thresholds must remain fixed")
    diagnostics = _finite([reference[key] for key in ("max_r_hat", "min_ess_bulk", "min_ess_tail")],
                          (3,), "measured reference diagnostics", positive=True)
    _require(diagnostics[0] <= requirements["r_hat_max"]
             and diagnostics[1] >= requirements["ess_bulk_min"]
             and diagnostics[2] >= requirements["ess_tail_min"]
             and reference["divergences"] == 0
             and reference["mcmc_diagnostics_passed"] is True
             and reference["gradient_checks_passed"] is True,
             "the displayed approximate reference requires passing measured diagnostics and gradients")
    _validate_recovery(record["recovery"], sources)

    _require(set(record["posteriors"]) == set(METHODS)
             and set(record["predictives"]) == set(METHODS),
             "both matched posterior and predictive methods are required")
    predictive_protocol = record["predictive_protocol"]
    _require(predictive_protocol["quantile_levels"] == [.05, .5, .95]
             and predictive_protocol["shared_gaussian_innovations"] is True
             and predictive_protocol["raw_price_conversion"] == "P_raw=P_forward*D*F exactly once",
             "predictive quantiles, Gaussian innovations and one raw-price conversion are required")
    for method in METHODS:
        posterior = record["posteriors"][method]
        _require(posterior["summaries_use_all_raw_draws"] is True
                 and posterior["support_clipped"] is False,
                 "posterior summaries must retain raw support violations without clipping")
        _quantiles(posterior["q05_q50_q95"], 7, method + " parameter")
        _finite(posterior["mean"], (7,), method + " posterior mean")
        _finite(posterior["sd"], (7,), method + " posterior SD", positive=True)
        correlation = _finite(posterior["correlation"], (7, 7), method + " posterior correlation")
        _require(np.allclose(correlation, correlation.T, rtol=0, atol=1e-12)
                 and np.allclose(np.diag(correlation), 1, rtol=0, atol=1e-12)
                 and np.all(np.abs(correlation) <= 1 + 1e-12)
                 and np.linalg.eigvalsh(correlation).min() >= -1e-10,
                 "invalid posterior correlation matrix")
        expected_draw_count = 8000 if method == "mcmc" else 4096
        _require(posterior["raw_draw_count"] == expected_draw_count
                 and type(posterior["out_of_prior_count"]) is int
                 and 0 <= posterior["out_of_prior_count"] <= expected_draw_count,
                 "the raw posterior draw count and unmodified support audit are required")
        _require(set(posterior["histograms"]) == {"c", "hurst"},
                 "the c and H marginal histogram summaries are required")
        for parameter, histogram in posterior["histograms"].items():
            edges = np.asarray(histogram["edges"], dtype=float)
            _require(edges.ndim == 1 and 2 <= len(edges) <= 201 and np.isfinite(edges).all()
                     and np.all(np.diff(edges) > 0), "invalid histogram edges for " + parameter)
            density = _finite(histogram["density"], (len(edges) - 1,), parameter + " histogram density",
                              nonnegative=True)
            _require(np.isclose(np.dot(density, np.diff(edges)), 1., rtol=1e-10, atol=1e-10),
                     "posterior histogram density must integrate to one")
        predictive = record["predictives"][method]
        _require(predictive["draw_count"] == 1024
                 and predictive["paired_parameter_and_c_draws"] is True
                 and predictive["conditional_c_redraw"] is False,
                 "predictives require original paired physical/curve/c draws at a matched draw budget")
        _require(type(predictive["raw_draw_count"]) is int
                 and type(predictive["support_exclusions"]) is int
                 and predictive["raw_draw_count"] >= 1024
                 and 0 <= predictive["support_exclusions"] <= predictive["raw_draw_count"] - 1024,
                 "raw posterior support exclusions must be explicit")
        _require(predictive["raw_draw_count"] == posterior["raw_draw_count"]
                 and predictive["support_exclusions"] == posterior["out_of_prior_count"]
                 and predictive["viewed_quote_count"] == 63,
                 "predictive support exclusions and the raw posterior audit differ")
        _quantiles(predictive["clean_q05_q50_q95_raw"], 63, method + " clean-price")
        noisy = _quantiles(predictive["noisy_q05_q50_q95_raw"], 63, method + " noisy-quote")
        _finite(predictive["clean_mean_raw"], (63,), method + " clean-price mean")
        contained = int(np.sum((noisy[0] <= arrays["observed_raw"]) & (arrays["observed_raw"] <= noisy[2])))
        _require(type(predictive["noisy_in_band_count"]) is int
                 and predictive["noisy_in_band_count"] == contained,
                 "descriptive in-band quote count differs from the saved noisy bands")

    comparison = record["fit_comparison"]
    rms_values = {}
    for method in ("heston", "bergomi"):
        fit = comparison[method]
        prices = _finite(fit["price_mean_raw"], (63,), method + " posterior-average prices")
        _quantiles(fit["price_q05_q50_q95_raw"], 63, method + " comparison price")
        residual = _finite(fit["residual_mean_minus_observed_over_h"], (63,), method + " price residual")
        _require(np.allclose(residual, (prices - arrays["observed_raw"]) / arrays["half_spread_raw"],
                             rtol=1e-10, atol=1e-10), "comparison residual sign or units differ")
        c_quantiles = _finite(fit["c_q05_q50_q95"], (3,), method + " comparison c interval")
        _require(np.all(np.diff(c_quantiles) >= 0) and fit["draw_count"] == 1024,
                 "ordered c intervals and the matched mean-fit draw budget are required")
        rms = float(np.sqrt(np.mean(((prices - arrays["observed_raw"]) / arrays["half_spread_raw"]) ** 2)))
        _require(np.isclose(fit["rms_over_h"], rms, rtol=1e-10, atol=1e-10),
                 "posterior-average price RMS and units differ for " + method)
        rms_values[method] = rms
    _require(np.isclose(comparison["rms_reduction_fraction"],
                        1 - rms_values["bergomi"] / rms_values["heston"], rtol=1e-10, atol=1e-10),
             "the stated RMS reduction and saved prices differ")


def _validate_recovery(recovery, sources):
    _require(recovery["case_count"] == 800 and recovery["posterior_draws_per_case"] == 256
             and recovery["population"] == "held_out_amortizer_development"
             and recovery["availability_selected"] is True
             and recovery["prior_wide_acceptance"] is False
             and recovery["used_for_amortizer_training"] is False
             and recovery["used_for_training_monitor"] is False
             and recovery["raw_draws_clipped"] is False,
             "recovery must retain the unfiltered, held-out development population and its limitations")
    truth = _finite(recovery["truth"], (800, 7), "development generating values")
    q = _finite(recovery["q05_q50_q95"], (3, 800, 7), "development marginal quantiles")
    bounds = np.asarray(PARAMETER_BOUNDS)
    _require(np.all((truth >= bounds[:, 0]) & (truth <= bounds[:, 1]))
             and np.all(np.diff(q, axis=0) >= 0),
             "development truths or quantile ordering differ")
    _require(recovery["truth_sha256"] == canonical_sha256(recovery["truth"])
             and recovery["quantiles_sha256"] == canonical_sha256(recovery["q05_q50_q95"]),
             "development generating values or marginal summaries changed")
    expected = {
        "median_rmse": np.sqrt(np.mean((q[1] - truth) ** 2, axis=0)),
        "coverage90": np.mean((q[0] <= truth) & (truth <= q[2]), axis=0),
        "median_width90": np.median(q[2] - q[0], axis=0),
    }
    for key, values in expected.items():
        actual = _finite(recovery[key], (7,), "development " + key, nonnegative=True)
        _require(np.allclose(actual, values, rtol=1e-12, atol=1e-14),
                 "development recovery metrics differ from the plotted summaries: " + key)
    counts = _finite(recovery["raw_support_count"], (7,), "development raw support counts", nonnegative=True)
    fractions = _finite(recovery["raw_support_fraction"], (7,), "development raw support fractions", nonnegative=True)
    total = 800 * 256
    _require(np.all(counts == np.floor(counts)) and np.all(counts <= total)
             and np.allclose(fractions, counts / total, rtol=0, atol=1e-15)
             and type(recovery["raw_joint_support_count"]) is int
             and counts.max() <= recovery["raw_joint_support_count"] <= min(total, counts.sum()),
             "raw support violations must be counted without clipping")
    _require(recovery["source_checkpoint_sha256"] == sources["control_checkpoint"],
             "development recovery must use the same displayed checkpoint")
