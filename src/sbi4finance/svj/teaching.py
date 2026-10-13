"""Hash-bound synthetic lesson evidence; cached loading needs no neural backend."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..datasets.local import load_artifact, write_artifact
from ..datasets.returns import prepare_return_window
from . import model
from .adapters import prepared_returns
from .bank import TASK_ID, default_task, task_fingerprint

FORMAT = "svj_teaching_v1"
BUNDLE_FILES = ("illustration.npz", "report.json", "window/manifest.json", "window/observations.csv")
CHECKPOINT_FILES = ("artifact.json", "model.keras", "training.json", "misspecification.npz")
ROLES = {"returns": "observation_percent_log_return",
         "variance": "auxiliary_truth_percent_squared_per_trading_day",
         "jumps": "auxiliary_truth_jump_indicator", "truth": "auxiliary_truth_native_targets",
         "posterior": "inference_output_equal_weight_joint_rows",
         "conditional_var": "downstream_output_percent_log_loss"}
SOURCE_FILES = (
    "inference_workflow.py", "training_workflow.py", "datasets/local.py", "datasets/returns.py",
    "svj/model.py", "svj/bank.py", "svj/adapters.py", "svj/artifacts.py", "svj/amortized.py",
    "svj/checks.py", "svj/filtering.py", "svj/audit.py", "svj/pmmh.py", "svj/pipeline.py",
    "svj/risk.py", "svj/teaching.py",
)


def synthetic_case():
    """The contribution's illustrative seed-3 history, with explicit data roles.

    These hand-chosen parameters illustrate a conditional task. They are not a
    prior-predictive calibration sample, and their truth never conditions the
    network or the particle audit.
    """
    coordinates = np.array([8., np.log(.18), np.log(35), np.log(.55), np.arctanh(-.6),
                            np.log(2.), -2.5, np.log(3.5)])
    simulation = model.simulate_from_coordinates(coordinates[None], np.random.default_rng(3))
    boundaries = np.busday_offset("2020-01-01", np.arange(model.WINDOW + 1)).astype(str).tolist()
    records = [{"interval_start": a, "interval_end": b, "value": float(value)}
               for a, b, value in zip(boundaries, boundaries[1:], simulation["returns"][0])]
    metadata = {
        "artifact_id": "svj_synthetic_illustration_seed3", "task_id": TASK_ID,
        "scientific_task": default_task(),
        "measure": "P", "data_origin": "synthetic", "purpose": "conditioning",
        "conventions": {"currency": "not_applicable", "adjustment_policy": "synthetic_no_adjustment",
                        "calendar": "synthetic_weekdays", "time_basis": "252_trading_days_per_year"},
        "provenance": {"source": "daily_svj_simulator", "series_id": "illustration:3",
                       "seed": 3, "evidence_scope": "S", "selection": "contributed_illustration_not_calibration_sample"},
        "truth": {"target_names": list(model.TARGET_NAMES), "targets": simulation["targets"][0].tolist()},
    }
    artifact = prepare_return_window(records, metadata=metadata, source_kind="log_returns",
                                     source_unit="percent", return_kind="log", boundaries=boundaries,
                                     dt_years=[1 / model.DAYS_PER_YEAR] * model.WINDOW,
                                     cutoff=boundaries[-1])
    return artifact, simulation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    """Bind the known implementation without importing inference backends."""
    package = Path(__file__).parents[1]
    return {name: sha(package / name) for name in SOURCE_FILES}


def demo_config():
    """Explicit opt-in reproduction budget; constructing it imports inference."""
    from .pipeline import PipelineConfig

    return PipelineConfig(n_draws=128, n_proposals=256, audit_particles=128,
                          n_paths=4, variance_split=False, seed=7301)


def write_bundle(destination, checkpoint):
    """Generate one bounded illustrative run; never overwrite earlier evidence.

    The manifest is published last. An interrupted/incomplete directory cannot
    be loaded as completed evidence and must be inspected before regeneration.
    """
    from . import amortized, artifacts, checks, pipeline

    destination, checkpoint = Path(destination), Path(checkpoint)
    if destination.exists():
        raise FileExistsError(destination)
    artifact, simulation = synthetic_case()
    binding = artifacts.validate_checkpoint(checkpoint, expected_task=default_task())
    posterior = amortized.AmortizedPosterior.load(checkpoint)
    reference = checks.MisspecificationTest.load(checkpoint / "misspecification.npz")
    report, draws = pipeline.run_prepared_window(artifact, posterior, reference, demo_config())
    destination.mkdir(parents=True)
    write_artifact(destination / "window", artifact.metadata, artifact.rows)
    np.savez_compressed(destination / "illustration.npz", returns=simulation["returns"][0],
                        variance=simulation["variance"][0], jumps=simulation["jumps"][0],
                        truth=simulation["targets"][0], posterior=draws,
                        conditional_var=pipeline.risk.conditional_var(draws))
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {
        "format": FORMAT, "evidence_scope": "S", "scientific_acceptance": False,
        "description": "One fixed synthetic illustration, not a calibration or acceptance study",
        "task_fingerprint": task_fingerprint({"task": default_task()}),
        "checkpoint": {"files": {name: sha(checkpoint / name) for name in CHECKPOINT_FILES},
                       "task_fingerprint": binding["task_sha256"]},
        "source_sha256": source_hashes(),
        "files": {name: sha(destination / name) for name in BUNDLE_FILES},
        "roles": ROLES,
        "seeds": {"simulation": 3, "conditional_workflow": 7301},
        "run_config": report["config"],
    }
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    load_bundle(destination, checkpoint=checkpoint)
    return manifest


def load_bundle(directory, *, checkpoint=None):
    """Load compatible local cached evidence without BayesFlow/Keras/JAX.

    Both cached outputs and their source/checkpoint identity are checked. A
    changed model or workflow requires a fresh cache in a new directory.
    """
    from .artifacts import validate_checkpoint

    directory = Path(directory)
    checkpoint = Path(checkpoint) if checkpoint is not None else Path(__file__).parents[3] / "checkpoints/svj"
    try:
        manifest = json.loads((directory / "bundle.json").read_text())
        expected_task = task_fingerprint({"task": default_task()})
        if (manifest["format"] != FORMAT or manifest["scientific_acceptance"] is not False
                or manifest["evidence_scope"] != "S" or manifest["roles"] != ROLES
                or manifest["task_fingerprint"] != expected_task
                or set(manifest["files"]) != set(BUNDLE_FILES)):
            raise ValueError("unsupported teaching artifact roles, task or evidence scope")
        if manifest["source_sha256"] != source_hashes():
            raise ValueError("teaching artifact source identity changed; regenerate into a new directory")
        binding = validate_checkpoint(checkpoint, expected_task=default_task(), check_versions=False)
        if (manifest["checkpoint"]["task_fingerprint"] != binding["task_sha256"]
                or manifest["checkpoint"]["files"] != {name: sha(checkpoint / name) for name in CHECKPOINT_FILES}):
            raise ValueError("teaching artifact checkpoint identity changed")
        for name, digest in manifest["files"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"teaching artifact changed: {name}")
        window = load_artifact(directory / "window")
        percent_returns = prepared_returns(window, checkpoint_task=default_task())
        if window.metadata["data_origin"] != "synthetic":
            raise ValueError("teaching artifact must use a synthetic history")
        with np.load(directory / "illustration.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if set(arrays) != set(ROLES):
            raise ValueError("unexpected teaching arrays")
        count = manifest["run_config"]["n_draws"]
        shapes = {"returns": (model.WINDOW,), "variance": (model.WINDOW,), "jumps": (model.WINDOW,),
                  "truth": (model.N_TARGETS,), "posterior": (count, model.N_TARGETS), "conditional_var": (count,)}
        for name, shape in shapes.items():
            value = arrays[name]
            if (value.shape != shape or not np.isfinite(value).all()
                    or (name == "jumps" and value.dtype != np.bool_)
                    or (name != "jumps" and value.dtype != np.float64)):
                raise ValueError(f"invalid teaching {name} shape, dtype or values")
        if np.any(arrays["variance"] < model.V_MIN):
            raise ValueError("invalid synthetic variance truth")
        if (not np.allclose(arrays["returns"], percent_returns, rtol=1e-14, atol=1e-14)
                or not np.array_equal(arrays["truth"], window.metadata["truth"]["targets"])
                or arrays["truth"][-1] != model.log_spot_vol(arrays["variance"][-1])):
            raise ValueError("synthetic observation/truth mapping changed")
        report = json.loads((directory / "report.json").read_text())
        workflow = report["workflow"]
        if (workflow["schema_version"] != "inference_workflow_v1"
                or workflow["application"] != "svj_daily_returns_v1"
                or workflow["scientific_acceptance"] != "not_assessed_by_this_run"
                or report["config"] != manifest["run_config"]
                or workflow["reliability"] not in {"flagged", "locally_checked", "unassessed"}
                or report["status"] != workflow["reliability"]
                or report["risk"]["reliability"] != workflow["reliability"]
                or report["risk"]["unit"] != "percent_log_loss"
                or report["risk"]["definition"] != "negative_cumulative_percent_log_return_v1"
                or report["scientific_acceptance"] != workflow["scientific_acceptance"]
                or report["risk"]["scientific_acceptance"] != workflow["scientific_acceptance"]
                or report["risk"]["diagnostic_reasons"] != workflow["diagnostic_reasons"]
                or workflow["provenance"]["observation"]["sha256"]
                    != hashlib.sha256(percent_returns.astype("<f8").tobytes()).hexdigest()):
            raise ValueError("inconsistent cached inference workflow report")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"incomplete teaching bundle: {error}") from error
    return manifest, window, arrays, report
