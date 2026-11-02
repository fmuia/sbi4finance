"""Bounded Heston numerical diagnosis; never a production task or posterior test.

Run from lab: PYTHONPATH=src python checks/path_projected_euler.py --out DIRECTORY
The fixed panel compares signed full truncation and projected Euler, with the
same left-variance price increment and coupled Brownian increments at 8/16/32
steps/day. Projection is sometimes called absorption; zero is NOT absorbing
when kappa*theta>0. Removing a return atom does not establish weak accuracy.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numba
import numpy as np
from numba import njit

from sbi4finance import heston
from sbi4finance.path_inference.bank import task_fingerprint, teaching_task


PROFILE_ID = "heston_projected_euler_development_panel_v1"
GRIDS = (8, 16, 32)
PARAMETER_NAMES = ("drift", "kappa", "theta", "vol_of_vol", "rho")
METRICS = ("total_log_return", "realized_variance", "integrated_variance",
           "terminal_effective_variance", "time_mean_effective_variance",
           "normalized_terminal_price", "zero_integrated_days", "zero_left_steps",
           "minimum_daily_integrated_variance", "minimum_daily_independent_variance")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_panel(seed=2026110101):
    """48 marginally stratified prior rows plus 16 declared extreme corners.

    Each prior coordinate occupies every one of 48 equal-probability strata
    once. Corners span kappa/theta/vol-of-vol/rho endpoints, with drift fixed at
    its prior mean (.05). This is a diagnostic design, not an iid prior sample.
    """
    task = teaching_task("heston")
    rng = np.random.default_rng(seed)
    columns = {}
    for name, bounds, kind in zip(task["parameter_names"], task["prior_bounds"], task["prior_types"]):
        u = (rng.permutation(48) + rng.random(48)) / 48
        lo, hi = bounds
        columns[name] = np.exp(np.log(lo) + u * np.log(hi / lo)) if kind == "log_uniform" else lo + u * (hi - lo)
    rows = np.column_stack([columns["drift"], columns["kappa"], columns["long_variance"],
                            columns["vol_of_vol"], columns["rho"]])
    bounds = dict(zip(task["parameter_names"], task["prior_bounds"]))
    corners = [[.05, bounds["kappa"][k], bounds["long_variance"][v],
                bounds["vol_of_vol"][s], bounds["rho"][r]]
               for k, v, s, r in itertools.product((0, 1), repeat=4)]
    return np.concatenate([rows, np.asarray(corners)]), ["stratified_prior"] * 48 + ["corner"] * 16


def coupled_normals(fine, substeps):
    """Aggregate independent finest-grid dW channels without changing leverage."""
    fine = np.asarray(fine, dtype=np.float64)
    if (fine.ndim != 4 or fine.shape[-1] != 2 or substeps not in GRIDS
            or fine.shape[2] != 32 or not np.isfinite(fine).all()):
        raise ValueError("need finite [batch,days,32,2] innovations and grid 8/16/32")
    factor = 32 // substeps
    if factor == 1:
        return fine
    return np.ascontiguousarray(fine.reshape(*fine.shape[:2], substeps, factor, 2).sum(axis=3) / math.sqrt(factor))


# Keep this isolated check importable both as a CLI and by file-based tests.
# Numba disk caches can encode the dynamic test module name.
@njit(cache=False, fastmath=False)
def _steps(parameters, dt, initial, normals, projected):
    batch, days, substeps, _ = normals.shape
    returns = np.empty((batch, days, 1))
    states = np.empty((batch, days + 1, 1))
    integrated = np.empty((batch, days))
    zero_left = np.zeros((batch, days), dtype=np.int64)
    for row in range(batch):
        drift, kappa, theta, xi, rho = parameters[row]
        independent_weight = math.sqrt(1 - rho * rho)
        variance = initial[row]
        states[row, 0, 0] = variance
        for day in range(days):
            h = dt[day] / substeps
            total, iv = 0.0, 0.0
            for step in range(substeps):
                positive = max(variance, 0.0)
                scale = math.sqrt(positive * h)
                zv, independent = normals[row, day, step]
                total += (drift - 0.5 * positive) * h + scale * (rho * zv + independent_weight * independent)
                iv += positive * h
                if positive == 0.0:
                    zero_left[row, day] += 1
                variance += kappa * (theta - positive) * h + xi * scale * zv
                if projected:
                    variance = max(variance, 0.0)
            returns[row, day, 0] = total
            states[row, day + 1, 0] = variance
            integrated[row, day] = iv
    return returns, states, integrated, zero_left


def diagnose(parameters, dt, initial, normals, *, projected):
    """Checked wrapper around the isolated analysis kernel; no redraw/floor."""
    p = np.asarray(parameters, dtype=np.float64)
    z = np.asarray(normals, dtype=np.float64)
    t = np.asarray(dt, dtype=np.float64)
    v = np.broadcast_to(np.asarray(initial, dtype=np.float64), (len(p),)).copy()
    if (p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all()
            or np.any(p[:, 1:4] <= 0) or np.any(np.abs(p[:, 4]) >= 1)
            or t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(t <= 0)
            or z.ndim != 4 or z.shape[:2] != (len(p), len(t)) or z.shape[2] < 1
            or z.shape[-1] != 2 or not np.isfinite(z).all() or not np.isfinite(v).all()
            or (projected and np.any(v < 0))):
        raise ValueError("invalid diagnostic inputs")
    result = _steps(np.ascontiguousarray(p), t, v, np.ascontiguousarray(z), bool(projected))
    if not all(np.isfinite(array).all() for array in result):
        raise ValueError("nonfinite diagnostic trajectory; no redraw permitted")
    return dict(zip(("log_returns", "variance_state", "integrated_variance", "zero_left_steps"), result))


def assert_production_parity(parameters, dt, initial, normals, baseline):
    p = dict(zip(PARAMETER_NAMES, np.asarray(parameters).T))
    shape = normals.shape[:2]
    innovations = {"brownian": normals, "jump_counts": np.zeros(shape, dtype=np.int64),
                   "jump_normals": np.zeros(shape)}
    production = heston.simulate(p, dt, initial, innovations, backend="numba")
    for key in ("log_returns", "variance_state"):
        if not np.array_equal(baseline[key], production[key]):
            raise AssertionError(f"signed diagnostic kernel differs from production {key}")


def path_metrics(result, parameters, dt):
    r = result["log_returns"][:, :, 0]
    iv = result["integrated_variance"]
    normalized_log_price = r.sum(axis=1) - parameters[:, 0] * dt.sum()
    with np.errstate(over="raise", invalid="raise"):
        normalized_price = np.exp(normalized_log_price)
    return np.column_stack([
        r.sum(axis=1), np.square(r).sum(axis=1), iv.sum(axis=1),
        np.maximum(result["variance_state"][:, -1, 0], 0), iv.sum(axis=1) / dt.sum(),
        normalized_price, (iv == 0).sum(axis=1), result["zero_left_steps"].sum(axis=1),
        iv.min(axis=1), ((1 - parameters[:, 4] ** 2)[:, None] * iv).min(axis=1),
    ])


def mean_se(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "replicate_se": (float(values.std(ddof=1) / math.sqrt(len(values)))
                                                            if len(values) > 1 else None)}


def analytical_scope():
    h = 1 / (252 * 8)
    return {
        "argument": "Projected V>=0. If V_j=0, V_(j+1)=kappa*theta*h>0. With >=2 left nodes, I=h*sum(V_j)>0. Conditional on all variance-driving normals, the independent price variance is (1-rho^2)*I>0; mixing these Gaussians gives no structural daily drift atom.",
        "unconditional_uniform_lower_bound": 0.0,
        "lower_bound_caveat": "Positive states can be arbitrarily small for supported innovations; nonconsecutive zero states do not imply a uniform four-rebound lower bound. Real arithmetic positivity does not prove every floating-point path is safe.",
        "exact_zero_rebound_minimum_at_8": .25 * .01 * h,
        "first_day_integrated_variance_lower_bound_at_8": .04 * h,
        "first_day_independent_variance_lower_bound_at_8": (1 - .95 ** 2) * .04 * h,
        "finite_precision": "An exact-zero rebound is representable in float64 across this prior. Record actual finite values and I; do not introduce a floor, noise, tolerance-defined atom mass or redraw to hide arithmetic failures.",
        "primary_scheme_source": {"authors": "Lord, Koekkoek and van Dijk", "title": "A comparison of biased simulation schemes for stochastic volatility models", "preprint_url": "https://papers.tinbergen.nl/06046.pdf", "published_doi": "10.1080/14697680802392496", "locations": "Table 1 and section 4.1: absorption uses f1=f2=f3=x+; full truncation f1=x,f2=f3=x+. Section 4.2 discusses absorption upward pricing bias. Projected storage has the same observable left states as their absorption scheme."},
        "bias_caveat": "Projection changes the latent transition and weak bias near zero. The projected latent state retains an atom at zero. Atom-free daily observations are not a Heston-accuracy, posterior-accuracy or reference-validation result.",
        "reference_caveat": "Any reference must use this projected intraday law, retain leverage and integrate independent price noise with its actual left-state sum. Signed-FTE references and old checkpoints/banks are not matched targets.",
        "martingale_caveat": "The left-Gaussian price step preserves conditional expected price growth exp(mu*h) analytically. Four finite histories per parameter cannot certify a price-martingale estimate; rare tails can dominate it.",
    }


def run_panel(out, *, smoke=False, days=1260, seed=2026110102):
    if isinstance(days, bool) or not isinstance(days, int) or days < 1 or (not smoke and days != 1260):
        raise ValueError("the full profile has 1260 days; --days overrides require --smoke")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() or (out / "path-metrics.npz").exists():
        raise FileExistsError("diagnosis output already exists; preserve prior results")
    started = time.perf_counter()
    panel, groups = parameter_panel()
    selected = [50] if smoke else list(range(64))
    replicates = 1 if smoke else 4
    task = teaching_task("heston")
    dt = np.full(days, task["dt_years"])
    rows, metrics, paired, moments = [], [], [], []
    aggregate = {}
    max_buffer_bytes = 0
    for index in selected:
        parameters = np.repeat(panel[index:index + 1], replicates, axis=0)
        ids = [index * 4 + replicate for replicate in range(replicates)]
        fine = heston.draw_innovations(dt, 32, seed=seed, example_ids=ids,
                                       stream_id=0x50324531)["brownian"]
        record = {}
        for grid in GRIDS:
            normals = coupled_normals(fine, grid)
            max_buffer_bytes = max(max_buffer_bytes, fine.nbytes + (0 if normals is fine else normals.nbytes))
            for scheme in ("signed_full_truncation", "projected_euler"):
                result = diagnose(parameters, dt, .04, normals, projected=scheme == "projected_euler")
                if scheme == "signed_full_truncation":
                    assert_production_parity(parameters, dt, .04, normals, result)
                values = path_metrics(result, parameters, dt)
                record[scheme, grid] = values
                drift, kappa, theta, xi, rho = panel[index]
                horizon = dt.sum()
                exact_mean_iv = theta * horizon + (.04 - theta) * (-math.expm1(-kappa * horizon)) / kappa
                continuous_first_moments = {
                    "terminal_effective_variance": theta + (.04 - theta) * math.exp(-kappa * horizon),
                    "integrated_variance": exact_mean_iv,
                    "total_log_return": drift * horizon - .5 * exact_mean_iv,
                    "normalized_terminal_price": 1.0,
                }
                moments.append({"parameter_index": index, "group": groups[index],
                                "scheme": scheme, "substeps": grid,
                                "path_metric_mean_and_replicate_se": {name: mean_se(values[:, col]) for col, name in enumerate(METRICS)},
                                "continuous_heston_first_moment_comparators": continuous_first_moments,
                                "scope": "Known continuous-process first moments are weak comparators, not a transition/reference posterior. Four histories cannot certify agreement or a price martingale."})
                for rep in range(replicates):
                    rows.append([index, rep, grid, 0 if scheme == "signed_full_truncation" else 1])
                    metrics.append(values[rep])
                key = f"{scheme}/{grid}"
                entry = aggregate.setdefault(key, {"zero_integrated_days": 0, "all_zero_left_days": 0,
                                                    "zero_left_steps": 0, "negative_end_states": 0,
                                                    "floating_equal_return_drift_dt_days": 0,
                                                    "minimum_daily_integrated_variance": None,
                                                    "minimum_daily_independent_variance": None})
                entry["zero_integrated_days"] += int(np.count_nonzero(result["integrated_variance"] == 0))
                entry["all_zero_left_days"] += int(np.count_nonzero(result["zero_left_steps"] == grid))
                entry["zero_left_steps"] += int(result["zero_left_steps"].sum())
                entry["negative_end_states"] += int(np.count_nonzero(result["variance_state"][:, 1:] < 0))
                entry["floating_equal_return_drift_dt_days"] += int(np.count_nonzero(
                    result["log_returns"][:, :, 0] == parameters[:, 0, None] * dt[None, :]))
                for field in ("minimum_daily_integrated_variance", "minimum_daily_independent_variance"):
                    value = float(values[:, METRICS.index(field)].min())
                    entry[field] = value if entry[field] is None else min(entry[field], value)
        # Paired differences are within parameter and innovation. Do not treat
        # time steps, deliberately selected corners, or grid pairs as iid cases.
        comparisons = [("projected_euler", g, "signed_full_truncation", g) for g in GRIDS]
        comparisons += [(s, finer, s, coarse) for s in ("signed_full_truncation", "projected_euler")
                        for coarse, finer in ((8, 16), (16, 32))]
        for scheme_a, grid_a, scheme_b, grid_b in comparisons:
            delta = record[scheme_a, grid_a] - record[scheme_b, grid_b]
            paired.append({"parameter_index": index, "group": groups[index],
                           "a": f"{scheme_a}/{grid_a}", "b": f"{scheme_b}/{grid_b}",
                           "difference_a_minus_b": {name: mean_se(delta[:, col])
                                                    for col, name in enumerate(METRICS)}})
    rows, metrics = np.asarray(rows, dtype=np.int64), np.asarray(metrics)
    payload = out / "path-metrics.npz"
    np.savez_compressed(payload, row_keys=rows, metrics=metrics, parameters=panel[selected],
                        parameter_indices=np.asarray(selected), metric_names=np.asarray(METRICS),
                        parameter_names=np.asarray(PARAMETER_NAMES))
    source = Path(__file__).resolve()
    root = source.parents[1]
    sources = {str(path.relative_to(root)): sha256(path) for path in
               (source, Path(heston.__file__), root / "src/sbi4finance/path_inference/bank.py")}
    report = {
        "schema": "heston_numerical_development_diagnosis_v1", "profile_id": PROFILE_ID,
        "complete_prespecified_panel": not smoke, "smoke": smoke,
        "purpose": "numerical development only; no target selection, production bank, refit or posterior accuracy claim",
        "original_teaching_task": task, "original_teaching_task_sha256": task_fingerprint(task),
        "prototype_task_published": False, "source_sha256": sources,
        "design": {"parameter_panel": "48 marginally stratified prior rows plus 16 kappa/theta/vol_of_vol/rho corners at drift .05",
                   "panel_seed": 2026110101, "innovation_seed": seed, "innovation_stream": 0x50324531,
                   "normal_coupling": "32-step independent channels; coarse standard normals are consecutive sums/sqrt(32/grid)",
                   "parameter_indices": selected, "groups": [groups[i] for i in selected],
                   "replicates_per_parameter": replicates, "days": days, "dt_years": 1 / 252,
                   "initial_variance": .04, "substeps": list(GRIDS),
                   "price_update": "left variance; leverage rho*Zv+sqrt(1-rho^2)*Zind; drift is annual proportional price drift"},
        "signed_baseline_bitwise_production_parity": True,
        "all_recorded_values_finite": bool(np.isfinite(metrics).all()),
        "characteristic_function_comparison_performed": False,
        "floating_equality_scope": "Exact stored-return equality to mu*dt is a floating-point diagnostic, separately counted from structural zero integrated variance. It is not a posterior atom weight or a population atom-rate estimate.",
        "aggregate_events": aggregate, "moment_comparisons": moments, "paired_comparisons": paired,
        "interpretation": analytical_scope(),
        "uncertainty": "Paired SE uses independent innovation replicates within each fixed parameter (n=4 in full panel). The panel is not iid prior data; rows/corners, time steps and grid comparisons are not independent replications. Refinement differences diagnose sensitivity, not error versus exact Heston.",
        "payload": {"path": payload.name, "bytes": payload.stat().st_size, "sha256": sha256(payload)},
        "resources": {"elapsed_seconds_including_initial_jit": time.perf_counter() - started,
                      "largest_explicit_innovation_buffer_bytes": max_buffer_bytes,
                      "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "peak_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
                      "python": platform.python_version(), "numpy": np.__version__,
                      "numba": numba.__version__, "platform": platform.platform()},
    }
    temporary = out / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--days", type=int, default=1260)
    parser.add_argument("--seed", type=int, default=2026110102)
    args = parser.parse_args()
    report = run_panel(args.out, smoke=args.smoke, days=args.days, seed=args.seed)
    print(json.dumps({"summary": str(args.out / "summary.json"), "complete_prespecified_panel": report["complete_prespecified_panel"],
                      "elapsed_seconds": report["resources"]["elapsed_seconds_including_initial_jit"]}))


if __name__ == "__main__":
    main()
