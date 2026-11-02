"""Verified offline option replays and explicit native model loading."""
import hashlib
import json
from pathlib import Path
import numpy as np


NAMES = ["initial_variance", "theta", "vol_of_vol", "rho"]


CACHE_FILES = {"task.json", "posterior.npz", "reference.json", "diagnostics.json", "products.json"}


ROUGH_NAMES = {"rough_calibration_v1": ["hurst", "eta", "rho"],
               "rough_joint_calibration_v1": ["hurst", "eta", "rho", "short_variance", "long_variance", "decay"]}


NEURAL_FILES = {"task.json", "posterior.npz", "evaluation.json", "training.json"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(task):
    return hashlib.sha256((json.dumps(task, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def heston_task(task):
    require(task.get("model") == "hkde" and task.get("parameter_names") == NAMES,
            "this cache supports the four-target Heston task only")
    context = task.get("context", {})
    contracts = context.get("contracts", [])
    noise = np.asarray(context.get("quote_noise_std"), dtype=float)
    bounds = np.asarray(task.get("prior_bounds"), dtype=float)
    require(len(contracts) == 63 and noise.shape == (63,) and np.isfinite(noise).all() and np.all(noise > 0),
            "complete ordered 63-contract design and positive quote noise required")
    require(bounds.shape == (4, 2) and np.isfinite(bounds).all() and np.all(bounds[:, 0] < bounds[:, 1]),
            "invalid parameter bounds")
    require(context.get("spot") == 1 and context.get("rate") == 0 and context.get("dividend") == 0,
            "expected spot-one zero-carry task")
    require(task.get("fixed_parameters", {}).get("kappa", 0) > 0
            and task["fixed_parameters"].get("jump_intensity") == 0, "zero-jump fixed-kappa Heston required")
    for contract in contracts:
        require(contract.get("option_type") in ("call", "put")
                and np.isfinite([contract.get("strike", np.nan), contract.get("maturity", np.nan)]).all()
                and min(contract["strike"], contract["maturity"]) > 0, "invalid contract")
    return task


def variance_products(task, draws):
    """Analytic annualized variance rates, preserving chain/draw axes."""
    kappa = task["fixed_parameters"]["kappa"]
    v0, theta = draws[..., 0], draws[..., 1]
    def rate(start, end):
        weight = np.exp(-kappa * start) * -np.expm1(-kappa * (end - start)) / (kappa * (end - start))
        return weight * v0 + (1 - weight) * theta
    return {"variance_swap_1y": rate(0, 1), "forward_variance_0_5_to_1y": rate(.5, 1)}


def load_cache(path, *, expected_task=None, expected_checkpoint_sha256=None):
    """Return task, report and cached arrays offline; requires NumPy, never BayesFlow."""
    path = Path(path)
    report = json.loads((path / "bundle.json").read_text())
    require(report.get("format") == "full_surface_teaching_v1" and set(report.get("files", {})) == CACHE_FILES,
            "expected complete full_surface_teaching_v1 cache")
    for name, sha in report["files"].items():
        require(digest(path / name) == sha, f"teaching cache checksum mismatch: {name}")
    task = heston_task(json.loads((path / "task.json").read_text()))
    require(report["task_fingerprint"] == fingerprint(task) and report["parameter_names"] == NAMES,
            "cache task identity differs")
    require(expected_task is None or task == expected_task, "cache differs from expected complete task")
    require(expected_checkpoint_sha256 is None or report["checkpoint_sha256"] == expected_checkpoint_sha256,
            "cache differs from expected checkpoint")
    require(report.get("scientific_acceptance") is False and report.get("role") == "cached_development_example"
            and report.get("checkpoint_status") == "frozen_candidate_gate_passed_not_full_acceptance"
            and not report["case_id"].startswith("test:"), "cached development/acceptance scope changed")
    with np.load(path / "posterior.npz", allow_pickle=False) as data: arrays = dict(data)
    count, shape = report["neural_draw_count"], report["reference_chain_shape"]
    require(type(count) is int and count > 1 and len(shape) == 3 and shape[-1] == 4
            and shape[0] > 1 and shape[1] > 1, "invalid draw counts or chain axes")
    shapes = {"observations": (63,), "neural_draws": (count, 4), "reference_chains": tuple(shape)}
    for name in ("variance_swap_1y", "forward_variance_0_5_to_1y"):
        shapes["neural_" + name] = (count,); shapes["reference_" + name] = tuple(shape[:2])
    require(set(arrays) == set(shapes), "cache arrays differ from complete replay contract")
    for name, expected in shapes.items():
        require(arrays[name].shape == expected and np.isfinite(arrays[name]).all(), f"invalid cached {name}")
    bounds = np.asarray(task["prior_bounds"])
    for name in ("neural_draws", "reference_chains"):
        require(np.all(arrays[name] >= bounds[:, 0]) and np.all(arrays[name] <= bounds[:, 1]),
                "cached parameter draws outside prior")
    reference = json.loads((path / "reference.json").read_text())
    require(reference["task"] == task and reference["surface_id"] == report["case_id"]
            and np.array_equal(reference["observed_prices"], arrays["observations"]),
            "cache observation/reference identity differs")
    for prefix, draws in (("neural_", arrays["neural_draws"]), ("reference_", arrays["reference_chains"])):
        for name, expected in variance_products(task, draws).items():
            require(np.allclose(arrays[prefix + name], expected, rtol=1e-12, atol=1e-14),
                    "parameter/product pairing changed")
    return task, report, arrays


def rough_task(task):
    names = ROUGH_NAMES.get(task.get("task_id"))
    require(task.get("model") == "rough" and names is not None and task.get("parameter_names") == names,
            "expected complete known-curve or joint-curve rough task")
    context = task.get("context", {}); contracts = context.get("contracts", [])
    noise = np.asarray(context.get("quote_noise_std"), dtype=float)
    bounds = np.asarray(task.get("prior_bounds"), dtype=float)
    require(len(contracts) == 63 and noise.shape == (63,) and np.isfinite(noise).all() and np.all(noise > 0)
            and bounds.shape == (len(names), 2) and np.isfinite(bounds).all()
            and np.all(bounds[:, 0] < bounds[:, 1]), "invalid full-surface rough task")
    return task


def load_neural_replay(path, *, expected_task=None, expected_checkpoint_sha256=None):
    """Offline rough replay; never fabricate missing reference chains or products."""
    path = Path(path); report = json.loads((path / "bundle.json").read_text())
    require(report.get("format") == "full_surface_neural_replay_v1" and set(report.get("files", {})) == NEURAL_FILES,
            "expected complete rough neural replay")
    for name, sha in report["files"].items(): require(digest(path / name) == sha, "replay checksum mismatch: " + name)
    task = rough_task(json.loads((path / "task.json").read_text()))
    training = json.loads((path / "training.json").read_text()); evaluation = json.loads((path / "evaluation.json").read_text())
    require(report["task_fingerprint"] == fingerprint(task) == training["task_fingerprint"] == evaluation["task_fingerprint"]
            and report["parameter_names"] == task["parameter_names"]
            and training.get("format") == "full_surface_training_v1"
            and report["selected_epoch"] == training["selected_epoch"]
            and report["checkpoint_sha256"] == training["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
            and report["bank_manifest_sha256"] == evaluation["bank_manifest_sha256"], "replay task/model/bank identity differs")
    require(expected_task is None or task == expected_task, "replay differs from expected complete task")
    require(expected_checkpoint_sha256 is None or report["checkpoint_sha256"] == expected_checkpoint_sha256,
            "replay differs from expected checkpoint")
    ids = evaluation.get("example_ids", [])
    case_id = report.get("case_id")
    require(isinstance(case_id, str) and case_id.startswith("validation:")
            and case_id.removeprefix("validation:").isdigit()
            and case_id == f"validation:{int(case_id.removeprefix('validation:'))}"
            and len(ids) == len(set(ids)) and case_id in ids, "development replay case identity differs")
    require(report.get("validation_case_index", int(case_id.split(":")[1])) == int(case_id.split(":")[1])
            and report.get("evaluation_row", ids.index(case_id)) == ids.index(case_id), "development replay case identity differs")
    require(report.get("role") == "cached_development_neural_replay"
            and report.get("scientific_acceptance") is False and report.get("reserved_test_evaluated") is False
            and report.get("conditional_reference_available") is False and evaluation.get("evaluated_split") == "validation"
            and evaluation.get("reserved_test_evaluated") is False, "development replay scope changed")
    with np.load(path / "posterior.npz", allow_pickle=False) as saved: arrays = dict(saved)
    count, dimension = report["neural_draw_count"], len(task["parameter_names"])
    known = task["task_id"] == "rough_calibration_v1"
    require(report.get("known_curve_is_observed") == known, "known/joint curve boundary changed")
    shapes = {"observations": (63,), "truth": (dimension,), "neural_draws": (count, dimension)}
    if known: shapes["context_values"] = (7,)
    require(type(count) is int and count > 1 and set(arrays) == set(shapes), "invalid replay arrays or draw count")
    for name, shape in shapes.items():
        require(arrays[name].shape == shape and np.isfinite(arrays[name]).all(), "invalid replay array: " + name)
    if known: require(np.all(arrays["context_values"] > 0), "observed curve levels must be positive")
    bounds = np.asarray(task["prior_bounds"])
    for name in ("truth", "neural_draws"):
        require(np.all(arrays[name] >= bounds[:, 0]) and np.all(arrays[name] <= bounds[:, 1]), "replay parameters outside prior")
    return task, report, arrays


def load_tutorial_artifact(manifest, name):
    """Pinned cached role for the review notebook; does not download or load models."""
    manifest = Path(manifest); catalogue = json.loads(manifest.read_text())
    require(catalogue.get("format") == "full_surface_tutorial_artifacts_v1"
            and catalogue.get("scientific_acceptance") is False, "invalid tutorial artifact catalogue")
    entry = catalogue["artifacts"][name]; path = (manifest.parent / entry["cache"]).resolve()
    require(digest(path / "bundle.json") == entry["bundle_sha256"], "catalogue/cache identity differs")
    require(entry["cache_format"] in ("full_surface_teaching_v1", "full_surface_neural_replay_v1"),
            "unknown tutorial cache format")
    loader = load_cache if entry["cache_format"] == "full_surface_teaching_v1" else load_neural_replay
    task, report, arrays = loader(path, expected_checkpoint_sha256=entry["checkpoint_sha256"])
    require(fingerprint(task) == entry["task_fingerprint"], "catalogue/task identity differs")
    if loader is load_cache:
        arrays["truth"] = np.asarray(json.loads((path / "reference.json").read_text())["generating_parameters"])
    return task, {**report, "artifact_role": "cached", "model_source": entry["model_source"]}, arrays


def load_tutorial_checkpoint(manifest, name, directory):
    """Explicit live role; BF2.1 initializes only when this function is called."""
    task, report, _ = load_tutorial_artifact(manifest, name)
    from . import full_surface
    live_task, workflow, record = full_surface.load_checkpoint(directory)
    require(live_task == task and record["checkpoint_sha256"] == report["checkpoint_sha256"],
            "live checkpoint differs from the selected tutorial artifact")
    return live_task, workflow, record


def rough_variance_products(task, draws, *, context_values=None):
    """Continuous mean-curve variance rates, preserving all leading draw axes.

    Known curves use seven observed piecewise-constant levels. Joint curves use
    each paired draw's exponential-curve coefficients; an observed/generating
    curve is not accepted for that task. These analytic functionals establish
    neither Monte Carlo product-price nor posterior accuracy.
    """
    task = rough_task(task)
    values = np.asarray(draws, dtype=float)
    bounds = np.asarray(task["prior_bounds"], dtype=float)
    require(values.ndim >= 1 and values.shape[-1] == len(task["parameter_names"])
            and values.size > 0 and np.isfinite(values).all(), "invalid rough parameter draws")
    require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
            "rough parameter draws outside task prior")
    context = task.get("model_context", {})
    knots = np.asarray(context.get("forward_variance_knots"), dtype=float)
    require(task.get("measure") == "Q" and knots.shape == (8,) and np.isfinite(knots).all()
            and knots[0] == 0 and np.all(np.diff(knots) > 0) and knots[-1] >= 1,
            "rough curve knots must cover the one-year product horizon")
    if task["task_id"] == "rough_calibration_v1":
        levels = np.asarray(context_values, dtype=float)
        require(levels.shape == (7,) and np.isfinite(levels).all() and np.all(levels > 0),
                "seven positive finite observed curve levels required")
        def rate(start, end):
            widths = np.maximum(0, np.minimum(knots[1:], end) - np.maximum(knots[:-1], start))
            return np.full(values.shape[:-1], np.dot(levels, widths) / (end - start))
    else:
        require(context_values is None and context.get("curve_family") == "heston_exponential_v1",
                "joint rough products require paired inferred curve coefficients only")
        short, long, decay = values[..., 3], values[..., 4], values[..., 5]
        require(np.all(values[..., 3:] > 0), "positive joint curve levels and decay required")
        def rate(start, end):
            x = decay * (end - start)
            average = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
            weight = np.exp(-decay * start) * average
            return weight * short + (1 - weight) * long
    products = {"variance_swap_1y": rate(0, 1), "forward_variance_0_5_to_1y": rate(.5, 1)}
    require(all(np.isfinite(value).all() and np.all(value > 0) for value in products.values()),
            "nonpositive or nonfinite rough variance product")
    return products
