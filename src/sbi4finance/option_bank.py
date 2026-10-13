"""Small fixed-panel option banks: explicit model dispatch, NPZ arrays and JSON.

This is a synthetic Q generation boundary, not a neural adapter or general data
platform. Pricing truth is an accurate expectation estimate, never one payoff.
"""

import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from .gbm_bank import _integer, _json, _sha256, _write_json
from .option_panel import noisy_panel, validate_context

SPLITS = {"train": 0, "validation": 1, "test": 2}
MODEL_PARAMETERS = {
    "hkde": {"kappa", "theta", "vol_of_vol", "rho", "jump_intensity",
             "jump_up_probability", "jump_up_rate", "jump_down_rate"},
    "one_factor": {"kappa", "eta", "rho"},
    "rough": {"hurst", "eta", "rho"},
}


def validate_config(config):
    if set(config) != {"task", "root_seed", "splits"}:
        raise ValueError("config requires task, root_seed, splits")
    task = config["task"]
    if (set(task) != {"task_id", "model", "measure", "context", "model_context",
                     "parameter_names", "prior_bounds", "fixed_parameters", "numerics"}
            or task["model"] not in MODEL_PARAMETERS or task["measure"] != "Q"
            or task["task_id"] != task["model"] + "_option_panel_v1"):
        raise ValueError("unsupported fixed-panel Q task")
    validate_context(task["context"])
    names = task["parameter_names"]
    bounds = np.asarray(task["prior_bounds"], dtype=float)
    fixed = task["fixed_parameters"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not isinstance(fixed, dict) or set(names) & set(fixed)
            or set(names) | set(fixed) != MODEL_PARAMETERS[task["model"]]
            or bounds.shape != (len(names), 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("explicit parameter order and finite independent-uniform bounds required")
    # Supported domain restrictions are scalar bounds (including the jump
    # moment restriction below), so checking both ends covers the entire box.
    for corner in bounds.T:
        parameters = dict(fixed, **dict(zip(names, corner)))
        if task["model"] == "hkde":
            from .hkde import validate_parameters
            validate_parameters(parameters)
        else:
            from .bergomi import _model_parameters
            _model_parameters(parameters, task["model"])
    if task["model"] == "hkde":
        lower = dict(fixed, **dict(zip(names, bounds[:, 0])))
        upper = dict(fixed, **dict(zip(names, bounds[:, 1])))
        if upper["jump_intensity"] * upper["jump_up_probability"] > 0 and lower["jump_up_rate"] <= 1:
            raise ValueError("entire prior box needs a finite upward exponential moment")
    numerics = task["numerics"]
    if task["model"] == "hkde":
        if set(task["model_context"]) != {"initial_variance"}:
            raise ValueError("HKDE requires explicit known initial_variance")
        v = task["model_context"]["initial_variance"]
        if isinstance(v, bool) or np.ndim(v) != 0 or not np.isfinite(v) or v <= 0:
            raise ValueError("initial_variance must be positive")
        if set(numerics) != {"nodes", "upper_limit", "refined_nodes", "refined_upper_limit", "max_refinement_over_quote_sigma"}:
            raise ValueError("HKDE requires base and refined Fourier settings")
        for key in ("nodes", "refined_nodes"):
            _integer(numerics[key], key, minimum=16)
        for key in ("upper_limit", "refined_upper_limit"):
            if not np.isfinite(numerics[key]) or numerics[key] <= 0:
                raise ValueError("Fourier upper limits must be positive")
        if (numerics["refined_nodes"] <= numerics["nodes"]
                or numerics["refined_upper_limit"] <= numerics["upper_limit"]):
            raise ValueError("refinement requires more nodes and a larger upper limit")
        budget = numerics["max_refinement_over_quote_sigma"]
    else:
        from .bergomi import curve_values
        horizon = max(c["maturity"] for c in task["context"]["contracts"])
        curve_values(task["model_context"], [0, horizon])
        if set(numerics) != {"n_paths", "n_steps", "method", "max_mc_se_over_quote_sigma"}:
            raise ValueError("Bergomi requires n_paths, n_steps, method")
        _integer(numerics["n_paths"], "n_paths", minimum=2)
        _integer(numerics["n_steps"], "n_steps", minimum=1)
        allowed = {"exact_ou"} if task["model"] == "one_factor" else {"hybrid", "covariance"}
        if numerics["method"] not in allowed:
            raise ValueError("unsupported driver method")
        if numerics["method"] == "covariance" and numerics["n_steps"] > 128:
            raise ValueError("covariance reference is limited to 128 steps")
        indices = np.array([c["maturity"] for c in task["context"]["contracts"]]) * numerics["n_steps"] / horizon
        if not np.allclose(indices, np.rint(indices), rtol=0, atol=1e-9):
            raise ValueError("contracts must land on the declared uniform grid")
        budget = numerics["max_mc_se_over_quote_sigma"]
    if isinstance(budget, bool) or np.ndim(budget) != 0 or not np.isfinite(budget) or budget <= 0:
        raise ValueError("numerical precision budget must be positive")
    _integer(config["root_seed"], "root_seed")
    if set(config["splits"]) != set(SPLITS):
        raise ValueError("train, validation and test required")
    for name, count in config["splits"].items():
        _integer(count, name, minimum=1)
    _json(config)


def task_fingerprint(config):
    return hashlib.sha256(_json(config["task"]).encode()).hexdigest()


def price_panel(task, parameters, seed):
    context, numerics = task["context"], task["numerics"]
    contracts = context["contracts"]
    if task["model"] == "hkde":
        from .hkde import european_price
        estimates = []
        for prefix in ("", "refined_"):
            estimates.append(np.array([
                european_price(context["spot"] * np.exp((context["rate"] - context["dividend"]) * c["maturity"]),
                               c["strike"], np.exp(-context["rate"] * c["maturity"]), c["maturity"],
                               parameters, task["model_context"]["initial_variance"], kind=c["option_type"],
                               nodes=numerics[prefix + "nodes"], upper_limit=numerics[prefix + "upper_limit"])
                for c in contracts], dtype=float))
        # Refined values are labels; the difference is an indicator, not a bound.
        n = len(contracts)
        return estimates[1], np.zeros(n), np.zeros((n, n)), estimates[1] - estimates[0]
    from .bergomi import option_panel
    result = option_panel(task["model"], parameters, task["model_context"], contracts,
                          spot=context["spot"], rate=context["rate"], dividend=context["dividend"],
                          n_paths=numerics["n_paths"], n_steps=numerics["n_steps"], seed=seed,
                          method=numerics["method"])
    return (result["prices"], result["standard_errors"], result["estimator_covariance"],
            np.zeros(len(contracts)))


def generate_examples(config, split, example_ids):
    """Independent whole-panel IDs, stable under ordering and split-size changes."""
    validate_config(config)
    ids = [_integer(i, "example ID") for i in example_ids]
    if split not in SPLITS or not ids or len(set(ids)) != len(ids):
        raise ValueError("known split and nonempty unique local IDs required")
    task = config["task"]
    bounds = np.asarray(task["prior_bounds"])
    c, p, b = len(task["context"]["contracts"]), len(bounds), len(ids)
    arrays = {"observations": np.empty((b, c, 1)), "parameters": np.empty((b, p)),
              "reference_prices": np.empty((b, c)), "pricing_standard_errors": np.empty((b, c)),
              "pricing_covariance": np.empty((b, c, c)), "refinement_difference": np.empty((b, c)),
              "example_id": np.array([f"{split}:{i}" for i in ids])}
    digest = task_fingerprint(config)
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    for row, example_id in enumerate(ids):
        identity = [0x4F5054, config["root_seed"], *words, SPLITS[split], example_id]
        streams = [np.random.SeedSequence([*identity, purpose]) for purpose in range(3)]
        prior_rng = np.random.Generator(np.random.PCG64(streams[0]))
        quote_rng = np.random.Generator(np.random.PCG64(streams[2]))
        seed = int(streams[1].generate_state(1, dtype=np.uint64)[0])
        parameters = prior_rng.uniform(bounds[:, 0], bounds[:, 1])
        model = dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], parameters)))
        try:
            prices, se, covariance, difference = price_panel(task, model, seed)
            sigma = task["context"]["quote_noise_std"]
            if task["model"] == "hkde":
                if np.max(np.abs(difference)) > sigma * task["numerics"]["max_refinement_over_quote_sigma"]:
                    raise ValueError("Fourier refinement budget failed")
            elif np.max(se) > sigma * task["numerics"]["max_mc_se_over_quote_sigma"]:
                raise ValueError("Monte Carlo standard-error budget failed")
            arrays["observations"][row] = noisy_panel(prices, task["context"], quote_rng)
            arrays["parameters"][row] = parameters
            for key, value in zip(("reference_prices", "pricing_standard_errors", "pricing_covariance", "refinement_difference"),
                                  (prices, se, covariance, difference)):
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite {key}")
                arrays[key][row] = value
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise ValueError(f"failed example {split}:{example_id}: {error}") from error
    return arrays


def source_files(model):
    return ["gbm_bank.py", "option_panel.py", "option_bank.py"] + {
        "hkde": ["hkde.py", "heston.py", "gbm.py"], "one_factor": ["bergomi.py"],
        "rough": ["bergomi.py", "rough_bergomi.py"]}[model]


def write_bank(path, config):
    """New directory only; a final manifest marks success. Failures are not redrawn."""
    validate_config(config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model = config["task"]["model"]
    manifest = {
        "format": "option_bank_v1", "configuration": config,
        "configuration_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
        "task_fingerprint": task_fingerprint(config), "data_origin": "synthetic",
        "evidence_scope": "S", "measure": "Q", "purpose": "synthetic fixed-panel inference training/evaluation",
        "conditioning_cutoff_years": 0.0, "contract_selection": "fixed ordered synthetic contracts in task context",
        "roles": {"observations": "conditioning: noisy currency prices [batch,contract,1]",
                  "context": "known synthetic contracts/spot/rates/quote sigma; model_context also known; stored in configuration",
                  "parameters": "ordered synthetic training targets only; never observed conditioning",
                  "reference_prices": "auxiliary numerical expectation estimates, not exact market truth",
                  "pricing_standard_errors": "auxiliary MC standard errors (zero for deterministic Fourier)",
                  "pricing_covariance": "auxiliary covariance of MC price MEANS; zero for Fourier",
                  "refinement_difference": "auxiliary refined-minus-base Fourier estimate; MC zeros mean not calculated",
                  "example_id": "auxiliary split-qualified base-panel identity; future augmentations must keep base grouping"},
        "truth_available": config["task"]["parameter_names"], "future_outcomes_in_conditioning": False,
        "prior": "independent uniforms in explicit parameter_names order",
        "noise": "independent additive Gaussian currency quote errors, no clipping",
        "numerical_scope": {"mc_se_excludes_bias": True, "fourier_difference_is_error_bound": False,
                            "refinement_difference_available": model == "hkde", "posterior_validated": False},
        "rng": "PCG64; SeedSequence(option tag, root, task digest words, split, ID, purpose); purposes 0 prior,1 pricing,2 quotes",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()},
        "source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in source_files(model)},
        "splits": {},
    }
    try:
        for split, count in config["splits"].items():
            arrays = generate_examples(config, split, range(count))
            destination = path / f"{split}.npz"
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as file:
                np.savez(file, **arrays)
            os.replace(temporary, destination)
            manifest["splits"][split] = {"file": destination.name, "examples": count, "sha256": _sha256(destination)}
        _write_json(path / "manifest.json", manifest)
    except Exception as error:
        _write_json(path / "failure.json", {"configuration": config, "error": str(error)})
        raise
    return manifest


def load_bank(path):
    """Verify stored identity, shapes and numerical summaries; no inference claim."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    config = manifest["configuration"]
    validate_config(config)
    if (manifest["format"] != "option_bank_v1" or manifest["data_origin"] != "synthetic"
            or manifest["evidence_scope"] != "S" or manifest["measure"] != "Q"
            or manifest["configuration_sha256"] != hashlib.sha256(_json(config).encode()).hexdigest()
            or manifest["task_fingerprint"] != task_fingerprint(config)
            or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("incompatible or changed manifest")
    task = config["task"]
    c, p = len(task["context"]["contracts"]), len(task["parameter_names"])
    result = {}
    bounds = np.asarray(task["prior_bounds"])
    for split, count in config["splits"].items():
        record = manifest["splits"][split]
        if record["file"] != f"{split}.npz" or record["examples"] != count:
            raise ValueError(f"{split}: inconsistent manifest")
        source = path / record["file"]
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"{split}: checksum mismatch")
        with np.load(source, allow_pickle=False) as saved:
            arrays = dict(saved)
        shapes = {"observations": (count, c, 1), "parameters": (count, p), "reference_prices": (count, c),
                  "pricing_standard_errors": (count, c), "pricing_covariance": (count, c, c),
                  "refinement_difference": (count, c), "example_id": (count,)}
        if set(arrays) != set(shapes):
            raise ValueError(f"{split}: unexpected arrays")
        for key, shape in shapes.items():
            a = arrays[key]
            if a.shape != shape or (key != "example_id" and (a.dtype != np.float64 or not np.isfinite(a).all())):
                raise ValueError(f"{split}: invalid {key}")
        if not np.array_equal(arrays["example_id"], [f"{split}:{i}" for i in range(count)]):
            raise ValueError(f"{split}: changed base-panel identity")
        if np.any(arrays["parameters"] < bounds[:, 0]) or np.any(arrays["parameters"] > bounds[:, 1]):
            raise ValueError(f"{split}: parameters outside prior")
        covariance = arrays["pricing_covariance"]
        se = arrays["pricing_standard_errors"]
        if (np.any(se < 0) or not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-12)
                or not np.allclose(se ** 2, np.diagonal(covariance, axis1=-2, axis2=-1), rtol=1e-10, atol=1e-14)
                or np.linalg.eigvalsh(covariance).min() < -1e-10):
            raise ValueError(f"{split}: invalid MC covariance/standard errors")
        result[split] = arrays
    return manifest, result
