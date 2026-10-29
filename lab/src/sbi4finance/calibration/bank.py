"""Calibration banks: the option_bank_v1 arrays and roles, plus the per-example curve context.

This mirrors option_bank and option_bank_shards rather than editing them, because
archived evidence pins their source hashes. Arrays, roles, seed streams, the
fail-closed precision rule and the chunk/assemble discipline are the same; the
additions are per-contract quote noise and three arrays:

* context_values [B, K]: the forward-variance levels the example was priced with
  (conditioning; K = 0 for the Heston check);
* context_parameters [B, 2]: the auxiliary draws behind the curve, (sigma0, slope) for linear_wobble and
  (sigma0, exponent b) for power_law_walk;
* pricing_paths [B, blocks]: Monte Carlo paths used per maturity block (auxiliary).

For the new six-target joint rough task, context_values are analytic interval
averages derived from the targets and context_parameters contain its three
curve coefficients. Both arrays are auxiliary, never inference conditions;
the simulator prices the curve on its finer grids rather than those buckets.

Stream purposes: 0 prior, 1 pricing, 2 quotes (as in option_bank), 3 curve context.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from ..gbm_bank import _integer, _json, _sha256, _write_json
from . import tasks

FORMAT = "calibration_bank_v1"
SHARD_FORMAT = "calibration_bank_shard_v1"
SPLITS = {"train": 0, "validation": 1, "test": 2}
PACKAGE = Path(__file__).resolve().parents[1]


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"task", "root_seed", "splits"}:
        raise ValueError("config requires task, root_seed, splits")
    tasks.validate_task(config["task"])
    _integer(config["root_seed"], "root_seed")
    if set(config["splits"]) != set(SPLITS):
        raise ValueError("train, validation and test required")
    for name, count in config["splits"].items():
        _integer(count, name, minimum=1)
    _json(config)


def task_fingerprint(config):
    return hashlib.sha256(_json(config["task"]).encode()).hexdigest()


def source_files(task):
    model = {"hkde": ["hkde.py", "heston.py", "gbm.py"], "rough": ["bergomi.py", "rough_bergomi.py"]}[task["model"]]
    return ["gbm_bank.py", *model, "calibration/surface.py", "calibration/pricing.py",
            "calibration/tasks.py", "calibration/bank.py"]


def source_hashes(task):
    return {name: _sha256(PACKAGE / name) for name in source_files(task)}


def versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()}


def array_shapes(config, count):
    task = config["task"]
    c, p = len(task["context"]["contracts"]), len(task["parameter_names"])
    return {"observations": (count, c, 1), "parameters": (count, p), "reference_prices": (count, c),
            "pricing_standard_errors": (count, c), "pricing_covariance": (count, c, c),
            "refinement_difference": (count, c), "example_id": (count,),
            "context_values": (count, tasks.context_size(task)),
            "context_parameters": (count, tasks.context_parameter_size(task)),
            "pricing_paths": (count, tasks.block_count(task))}


def check_precision(task, arrays):
    sigma = tasks.sigma(task)
    numerics = task["numerics"]
    if task["model"] == "hkde":
        if np.any(np.abs(arrays["refinement_difference"]) > sigma * numerics["max_refinement_over_quote_sigma"]):
            raise ValueError("Fourier refinement budget failed")
    elif np.any(arrays["pricing_standard_errors"] > sigma * numerics.get("max_mc_se_over_quote_sigma_last_rung",
                                                                         numerics["max_mc_se_over_quote_sigma"])):
        raise ValueError("Monte Carlo standard-error budget failed")


def generate_examples(config, split, example_ids):
    """Independent whole-surface IDs, stable under ordering and split-size changes."""
    validate_config(config)
    ids = [_integer(i, "example ID") for i in example_ids]
    if split not in SPLITS or not ids or len(set(ids)) != len(ids):
        raise ValueError("known split and nonempty unique local IDs required")
    task = config["task"]
    bounds = np.asarray(task["prior_bounds"])
    sigma = tasks.sigma(task)
    arrays = {key: np.empty(shape) for key, shape in array_shapes(config, len(ids)).items() if key != "example_id"}
    arrays["example_id"] = np.array([f"{split}:{i}" for i in ids])
    digest = task_fingerprint(config)
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    for row, example_id in enumerate(ids):
        identity = [0x4F5054, config["root_seed"], *words, SPLITS[split], example_id]
        streams = [np.random.SeedSequence([*identity, purpose]) for purpose in range(4)]
        prior_rng = np.random.Generator(np.random.PCG64(streams[0]))
        quote_rng = np.random.Generator(np.random.PCG64(streams[2]))
        context_rng = np.random.Generator(np.random.PCG64(streams[3]))
        seed = int(streams[1].generate_state(1, dtype=np.uint64)[0])
        parameters = prior_rng.uniform(bounds[:, 0], bounds[:, 1])
        context_values, context_parameters = (tasks.derived_context(task, parameters)
                                               if tasks.is_joint_curve_task(task)
                                               else tasks.draw_context(task, context_rng))
        try:
            prices, se, covariance, difference, used = tasks.price(task, parameters, context_values, seed)
            values = {"reference_prices": prices, "pricing_standard_errors": se, "pricing_covariance": covariance,
                      "refinement_difference": difference, "context_values": context_values,
                      "context_parameters": context_parameters, "pricing_paths": used}
            for key, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite {key}")
            check_precision(task, values)
            arrays["observations"][row] = (prices + sigma * quote_rng.standard_normal(prices.shape))[:, None]
            arrays["parameters"][row] = parameters
            for key, value in values.items():
                arrays[key][row] = value
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise ValueError(f"failed example {split}:{example_id}: {error}") from error
    return arrays


def check_arrays(arrays, config, split, start, stop):
    """Shapes, identities, prior support, MC covariance and the precision budget."""
    count = stop - start
    task = config["task"]
    shapes = array_shapes(config, count)
    if set(arrays) != set(shapes):
        raise ValueError(f"{split}: unexpected arrays")
    for key, shape in shapes.items():
        a = arrays[key]
        if a.shape != shape or (key != "example_id" and (a.dtype != np.float64 or not np.isfinite(a).all())):
            raise ValueError(f"{split}: invalid {key}")
    if not np.array_equal(arrays["example_id"], [f"{split}:{i}" for i in range(start, stop)]):
        raise ValueError(f"{split}: changed surface identity")
    bounds = np.asarray(task["prior_bounds"])
    if np.any(arrays["parameters"] < bounds[:, 0]) or np.any(arrays["parameters"] > bounds[:, 1]):
        raise ValueError(f"{split}: parameters outside prior")
    if tasks.is_joint_curve_task(task):
        for row, parameters in enumerate(arrays["parameters"]):
            levels, coefficients = tasks.derived_context(task, parameters)
            if (not np.allclose(arrays["context_values"][row], levels, rtol=1e-12, atol=1e-14)
                    or not np.array_equal(arrays["context_parameters"][row], coefficients)):
                raise ValueError(f"{split}: joint curve auxiliaries differ from parameter targets")
    covariance, se = arrays["pricing_covariance"], arrays["pricing_standard_errors"]
    if (np.any(se < 0) or not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-12)
            or not np.allclose(se**2, np.diagonal(covariance, axis1=-2, axis2=-1), rtol=1e-10, atol=1e-14)
            or (len(covariance) and np.linalg.eigvalsh(covariance).min() < -1e-10)):
        raise ValueError(f"{split}: invalid MC covariance/standard errors")
    check_precision(task, arrays)


def _manifest(config, identity, generation_versions):
    task = config["task"]
    manifest = {
        "format": FORMAT, "configuration": config,
        "configuration_sha256": identity["configuration_sha256"],
        "task_fingerprint": identity["task_fingerprint"], "data_origin": "synthetic",
        "evidence_scope": "S", "measure": "Q", "purpose": "synthetic smile-calibration training/evaluation",
        "conditioning_cutoff_years": 0.0, "contract_selection": "fixed ordered design grid in task context",
        "roles": {"observations": "conditioning: noisy forward-unit prices [batch,contract,1]",
                  "context": "known contracts, forward units and per-contract quote sigma; stored in configuration",
                  "parameters": "ordered synthetic training targets only; never observed conditioning",
                  "reference_prices": "auxiliary numerical expectation estimates, not exact market truth",
                  "pricing_standard_errors": "auxiliary MC standard errors (zero for deterministic Fourier)",
                  "pricing_covariance": "auxiliary covariance of MC price MEANS; zero for Fourier",
                  "refinement_difference": "auxiliary refined-minus-base Fourier estimate; MC zeros mean not calculated",
                  "example_id": "auxiliary split-qualified surface identity; augmentations must keep base grouping",
                  "context_values": "known per-example forward-variance levels on the knot intervals; conditioning",
                  "context_parameters": "auxiliary draws behind the curve (sigma0 and the slope or exponent); never conditioning",
                  "pricing_paths": "auxiliary Monte Carlo paths used per maturity block"},
        "truth_available": task["parameter_names"], "future_outcomes_in_conditioning": False,
        "prior": "independent uniforms in explicit parameter_names order; curve context from context_prior",
        "noise": "independent additive Gaussian forward-unit quote errors, one sd per contract, no clipping",
        "numerical_scope": {"mc_se_excludes_bias": True, "fourier_difference_is_error_bound": False,
                            "posterior_validated": False},
        "rng": "PCG64; SeedSequence(option tag, root, task digest words, split, ID, purpose); "
               "purposes 0 prior, 1 pricing, 2 quotes, 3 curve context",
        "versions": generation_versions, "source_sha256": identity["source_sha256"], "splits": {},
    }
    if tasks.is_joint_curve_task(task):
        manifest["roles"].update(
            context_values="auxiliary analytic interval-average forward variances derived from targets; never conditioning",
            context_parameters="auxiliary short_variance, long_variance, decay copied from targets; never conditioning")
        manifest["prior"] = "six independent uniforms in explicit parameter_names order; curve is derived from targets"
        manifest["numerical_scope"]["curve_evaluation"] = (
            "analytic heston_exponential_v1 at every simulator node; left-cell integration and matching discrete "
            "expected-integrated-variance controls; stored interval averages are auxiliary only")
    return manifest


def _identity(config):
    return {"configuration_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
            "task_fingerprint": task_fingerprint(config), "source_sha256": source_hashes(config["task"])}


def _save(path, arrays):
    """Compressed: the [B, C, C] covariance is zero for Fourier and block-sparse for Monte Carlo."""
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def write_bank(path, config):
    """New directory only; one process; a final manifest marks success."""
    validate_config(config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(config, _identity(config), versions())
    try:
        for split, count in config["splits"].items():
            destination = path / f"{split}.npz"
            _save(destination, generate_examples(config, split, range(count)))
            manifest["splits"][split] = {"file": destination.name, "examples": count, "sha256": _sha256(destination)}
        _write_json(path / "manifest.json", manifest)
    except Exception as error:
        _write_json(path / "failure.json", {"configuration": config, "error": str(error)})
        raise
    return manifest


def load_bank(path, splits=None):
    """The manifest and the checked arrays of every split, or only of the splits named (the full bank's train split
    holds a [100000, 63, 63] covariance: a study of the test split should not load it)."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    config = manifest["configuration"]
    validate_config(config)
    if (manifest["format"] != FORMAT or manifest["measure"] != "Q" or manifest["evidence_scope"] != "S"
            or manifest["configuration_sha256"] != hashlib.sha256(_json(config).encode()).hexdigest()
            or manifest["task_fingerprint"] != task_fingerprint(config) or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("incompatible or changed manifest")
    result = {}
    for split, count in config["splits"].items():
        if splits is not None and split not in splits:
            continue
        record = manifest["splits"][split]
        if record["file"] != f"{split}.npz" or record["examples"] != count:
            raise ValueError(f"{split}: inconsistent manifest")
        source = path / record["file"]
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"{split}: checksum mismatch")
        with np.load(source, allow_pickle=False) as saved:
            arrays = dict(saved)
        check_arrays(arrays, config, split, 0, count)
        result[split] = arrays
    return manifest, result


def write_shard(path, config, split, start, stop):
    """Generate IDs [start, stop) of one split in a new directory; failures keep their ID."""
    validate_config(config)
    start, stop = _integer(start, "start"), _integer(stop, "stop", minimum=1)
    if split not in SPLITS or not start < stop <= config["splits"][split]:
        raise ValueError("chunk must be a nonempty range within a configured split")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    record = {"format": SHARD_FORMAT, "configuration": config, **_identity(config), "versions": versions(),
              "split": split, "start": start, "stop": stop, "file": "arrays.npz"}
    begun = time.process_time()
    try:
        arrays = generate_examples(config, split, range(start, stop))
        _save(path / record["file"], arrays)
        record.update(sha256=_sha256(path / record["file"]), examples=stop - start,
                      generation_cpu_seconds=time.process_time() - begun)
        _write_json(path / "manifest.json", record)
    except Exception as error:
        _write_json(path / "failure.json", {**record, "error": str(error),
                                            "generation_cpu_seconds": time.process_time() - begun})
        raise
    return record


def assemble_bank(path, config, shard_paths):
    """Check exact split coverage, identities and every array; then write one bank."""
    validate_config(config)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    identity = _identity(config)
    parts = {split: [] for split in SPLITS}
    provenance, generation_versions = [], None
    for shard_path in map(Path, shard_paths):
        record = json.loads((shard_path / "manifest.json").read_text())
        if record.get("format") != SHARD_FORMAT or record.get("configuration") != config:
            raise ValueError("incompatible shard configuration")
        for key, value in identity.items():
            if record.get(key) != value:
                raise ValueError(f"shard {key} mismatch")
        if generation_versions is None:
            generation_versions = record["versions"]
        elif record["versions"] != generation_versions:
            raise ValueError("shards use different generation versions")
        source = shard_path / record["file"]
        if _sha256(source) != record["sha256"]:
            raise ValueError("shard checksum mismatch")
        with np.load(source, allow_pickle=False) as saved:
            arrays = dict(saved)
        check_arrays(arrays, config, record["split"], record["start"], record["stop"])
        parts[record["split"]].append((record["start"], record["stop"], arrays))
        provenance.append({key: record[key] for key in ("split", "start", "stop", "sha256", "generation_cpu_seconds")})
    for split, chunks in parts.items():
        cursor = 0
        for start, stop, _ in sorted(chunks, key=lambda item: item[0]):
            if start != cursor:
                raise ValueError(f"{split}: overlapping, duplicate or missing shard IDs")
            cursor = stop
        if cursor != config["splits"][split]:
            raise ValueError(f"{split}: missing shard IDs")
    manifest = _manifest(config, identity, generation_versions)
    manifest["shard_assembly"] = {"assembly_versions": versions(),
                                  "chunks": sorted(provenance, key=lambda p: (SPLITS[p["split"]], p["start"]))}
    path.mkdir(parents=True, exist_ok=False)
    try:
        for split, chunks in parts.items():
            chunks.sort(key=lambda item: item[0])
            arrays = {key: np.concatenate([chunk[2][key] for chunk in chunks], axis=0) for key in chunks[0][2]}
            destination = path / f"{split}.npz"
            _save(destination, arrays)
            manifest["splits"][split] = {"file": destination.name, "examples": config["splits"][split],
                                         "sha256": _sha256(destination)}
        _write_json(path / "manifest.json", manifest)
    except Exception as error:
        _write_json(path / "failure.json", {"configuration": config, "error": str(error)})
        raise
    return manifest


def job_map(config, chunk_size):
    """[{split, start, stop}] covering every split in chunks of at most chunk_size surfaces."""
    validate_config(config)
    chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
    return [{"split": split, "start": start, "stop": min(start + chunk_size, count)}
            for split, count in config["splits"].items() for start in range(0, count, chunk_size)]
