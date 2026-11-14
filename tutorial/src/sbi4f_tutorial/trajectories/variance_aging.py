"""Bound simulation-design evidence about the four preserved rough histories.

Loading reads accepted local JSON/PNG and the original replay only. It runs no
new simulations, posterior sampling or network import. This is a model-law
illustration, not a new calibration result or replacement history selection.
"""

import json

import numpy as np

from ..shared.paths import ASSETS
from ..shared.full_surface_teaching import digest as asset_digest
from . import latent_assets, lesson, simulation


ASSET_FOLDER = "rough-variance-aging-v1"
REPORT_SHA256 = "68645f59e6ad2d3c02c1ad40dd721884e94e0598c5a852c95ee23db022a4fd6b"
EVENT = "late_mean_V_below_2pct_vol"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _asset_file(path):
    result = path.resolve()
    _require(result.is_relative_to(ASSETS.resolve()), "variance-aging inputs must remain under tutorial/assets")
    return result


def _sha(path):
    return asset_digest(path)


def _wilson(count, n):
    z = 1.959963984540054
    p, denominator = count/n, 1+z*z/n
    centre = (p+z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return np.array([centre-radius, centre+radius])


def _rate(row):
    n, count = row.get("n"), row.get("count")
    _require(type(n) is int and n == 512 and type(count) is int and 0 <= count <= n
             and row.get("fraction") == count/n
             and np.allclose(row.get("wilson95", []), _wilson(count, n), rtol=2e-13, atol=1e-15),
             "variance-aging count, fraction or binomial interval disagree")


def load(task):
    """Return the verified report and figure for its exact rough-known design."""
    simulation.validate_task(task)
    folder = ASSETS / "diagnostics" / ASSET_FOLDER
    report_path = _asset_file(folder / "report.json")
    _require(report_path.is_file() and _sha(report_path) == REPORT_SHA256,
             "variance-aging report is missing or differs from its pinned identity")
    report = json.loads(report_path.read_text())
    _require(report.get("format") == "trajectory_rough_variance_aging_v1"
             and report.get("scope") == "unchanged_simulator_variance_aging_illustration"
             and report.get("scientific_acceptance") is False
             and report.get("posterior_calibration_claim") is False
             and report.get("simulator_changed") is False and report.get("prior_changed") is False
             and report.get("checkpoint_changed") is False and report.get("model") == "rough-known"
             and report.get("measure") == "P",
             "variance-aging illustration scope differs; it cannot become a calibration claim")
    _require(report.get("task") == task
             and report.get("task_sha256") == simulation.task_fingerprint(task),
             "variance-aging task differs from this history design")
    original_task, replay, arrays = lesson.load_example("rough-known")
    base = lesson.catalogue()["artifacts"]["rough-known"]
    _require(task == original_task and report.get("checkpoint_sha256") == replay["checkpoint_sha256"]
             == base["checkpoint_sha256"]
             and report.get("source_replay_bundle_sha256") == base["bundle_sha256"]
             and report.get("source_posterior_file_sha256") == replay["files"]["posterior.npz"]
             and report.get("paired_latent_bundle_sha256") == latent_assets.BUNDLE_SHA256,
             "variance-aging checkpoint, original replay or paired latent identity differs")
    files = report.get("files", {})
    _require(set(files) == {"variance-aging.png"}, "unexpected variance-aging figure files")
    figure_path = _asset_file(folder / "variance-aging.png")
    _require(figure_path.is_file() and _sha(figure_path) == files[figure_path.name],
             "variance-aging figure differs from its bound file hash")
    receipt = json.loads(_asset_file(folder / "ACCEPTED.json").read_text())
    _require(receipt.get("changes") == []
             and receipt.get("source_sha256", {}).get("report.json") == REPORT_SHA256
             and receipt.get("source_sha256", {}).get("variance-aging.png") == files[figure_path.name],
             "variance-aging copy receipt does not preserve the accepted bytes")
    event = report.get("event", {})
    _require(event == {"name": EVENT, "threshold_variance": .0004, "annual_volatility_equivalent": .02,
                       "boundary_day_range_inclusive": [1009, 1260], "boundary_count": 252,
                       "comparison": "strictly_less_than", "statistic": "arithmetic_mean_latent_boundary_variance"},
             "variance-aging final-year event differs from the declared definition")
    rule = report.get("frozen_rule", {})
    prior = report.get("prior_ensemble", {})
    _require(rule.get("prior_ensemble_counts", {}).get("rough-known") == 512
             and rule.get("fixed_known_case_ensemble_count") == 512 and EVENT in rule.get("events", {})
             and prior.get("all_paths_retained") is True and prior.get("split") == "validation"
             and prior.get("start_id") == 0 and prior.get("seed") == rule.get("prior_seed", -2)+1,
             "variance-aging ensemble or frozen sampling rule differs")
    _rate(prior)
    values = np.asarray(prior.get("final_year_mean_variance", []), dtype=np.float64)
    edges = np.asarray(prior.get("log10_histogram_edges", []), dtype=np.float64)
    counts = np.asarray(prior.get("histogram_counts", []))
    _require(values.shape == (512,) and np.isfinite(values).all() and np.all(values > 0)
             and int(np.count_nonzero(values < event["threshold_variance"])) == prior["count"],
             "variance-aging final-year values disagree with the event count")
    _require(edges.ndim == 1 and len(edges) >= 2 and np.isfinite(edges).all() and np.all(np.diff(edges) > 0)
             and counts.shape == (len(edges)-1,) and counts.dtype.kind in "iu"
             and np.array_equal(np.histogram(np.log10(values), edges)[0], counts) and int(counts.sum()) == 512,
             "variance-aging histogram must retain every simulated history")
    saved, fixed = report.get("saved_cases", []), report.get("fixed_parameter_ensembles", [])
    _require(len(saved) == len(fixed) == 4, "variance-aging illustration must retain all four original cases")
    for case, (observed, repeated) in enumerate(zip(saved, fixed)):
        _rate(repeated)
        paired = latent_assets.load_history(task, arrays, case)
        measured = float(paired["variance"][-252:].mean())
        _require(observed.get("case") == repeated.get("case") == case
                 and observed.get("case_id") == str(arrays["trajectory_id"][case])
                 and observed.get("observation_sha256") == latent_assets._array_sha(arrays["observations"][case])
                 and observed.get("parameter_sha256") == latent_assets._array_sha(arrays["truth"][case])
                 and np.array_equal(observed.get("parameters"), arrays["truth"][case])
                 and np.array_equal(repeated.get("parameters"), arrays["truth"][case])
                 and repeated.get("seed") == rule.get("fixed_seed", -10)+case
                 and repeated.get("start_id") == rule.get("fixed_case_start_ids", [None]*4)[case],
                 "variance-aging saved case, parameters or fixed-ensemble random stream differ")
        _require(np.isclose(observed.get("final_year_mean_variance", np.nan), measured, rtol=2e-13, atol=1e-16)
                 and observed.get("event") is bool(measured < .0004)
                 and edges[0] <= np.log10(measured) <= edges[-1],
                 "variance-aging saved marker must show its actual paired latent history, including extremes")
    return {"report": report, "figure_path": figure_path}
