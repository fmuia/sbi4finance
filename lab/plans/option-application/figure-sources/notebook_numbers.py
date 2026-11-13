"""Recompute the numbers and figure data that the Heston notebook prints.

The notebook is `lab/notebooks/heston_spx_calibration.py`. This script repeats its computations
without marimo or BayesFlow, from the same recorded inputs, seeds and code, and writes
`notebook.json`. The documentation quotes these numbers, and `make_figures.py` draws the
notebook's figures from this file.

Run it from the `lab/` folder of a checkout of the notebook's branch, with that checkout's lab,
lab sources and tutorial runtime on the path:

    cd <checkout>/lab
    PYTHONPATH=.:src:../tutorial/src python \
        <this folder>/notebook_numbers.py --out <this folder>/notebook.json --commit <commit>

Needs numpy and scipy. What is recomputed, and what is read:
- Everything that does not need the trained network is recomputed as the notebook computes it:
  the design, the four features of Heston, the prior predictive check, the whitening example,
  the PyMC reference tables, the costs, the posterior predictive check, the calibration band,
  the recovery table, the misfit analysis and the product prices of Section 6.
- The network's posteriors on the three data sets are read from the recorded evaluation
  (`evaluation.json`): 4,096 draws per data set, with the seed the notebook uses for its own draws.
- The posterior predictive check uses the PyMC draws alone, as the notebook does when the
  pretrained estimator is not loaded.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binom

from sbi4f_tutorial.shared import surface
from sbi4f_tutorial.shared.whitening import fit_whitening
from sbi4finance.calibration import products
from sbi4finance.calibration.heston_grid import HestonGridPilot
from scripts import audit_heston_grid_coverage as grid_rules
from scripts import reproduce_heston_pr87_grid as recipe
from scripts.run_heston_spread_reference import expected_variance

LAB = Path(".").resolve()
EVIDENCE = LAB / "evidence" / "pilot-results"
GRID = EVIDENCE / "2026-11-09" / "heston-spread-baseline" / "revised-grid"
EVALUATION = GRID / "recovery-bank-independent" / "300k-300" / "evaluation"
REFERENCES = EVIDENCE / "2026-11-11" / "heston-actual63-notebook"
NOTEBOOK = LAB / "notebooks" / "heston_spx_calibration.py"
CASE_ORDER = ("synthetic-market-like", "synthetic-broad", "market")  # the notebook's order of data sets
FORWARD_START = (0.25, 0.5)  # Section 6: strike fixed after three months, paid after six
PATHS, SHOWN = 4096, 40      # Monte Carlo paths per price, posterior draws priced (Section 6)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(value, digits=6):
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return rounded(value.tolist(), digits)
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="commit of the notebook's checkout")
    parser.add_argument("--branch", default="claude/awesome-fermat-544u9v")
    args = parser.parse_args()
    out = {}

    # ---- the training record (Section 3.2)
    run = EVALUATION.parent
    contract = json.loads((run / "protocol.json").read_text())["contract"]
    optimizer = contract["resolved_defaults"]["optimizer"]
    rate = optimizer["config"]["learning_rate"]["config"]
    jobs = {job["id"]: job for job in json.loads((run.parent / "campaign.json").read_text())["jobs"]}
    evaluated = json.loads((EVALUATION / "evaluation.json").read_text())
    out["training"] = {
        "rows": contract["train_rows"], "validation_rows": contract["validation_rows"],
        "epochs": contract["epochs"], "batch_size": contract["batch_size"],
        "updates": contract["expected_optimizer_updates"], "seed": contract["seed"],
        "optimizer": optimizer["class_name"], "weight_decay": optimizer["config"]["weight_decay"],
        "clipnorm": optimizer["config"]["clipnorm"], "rate_start": rate["initial_learning_rate"],
        "rate_peak": rate["warmup_target"], "warmup_updates": rate["warmup_steps"],
        "whitening_rows": contract["bank_contract"]["whitening_rows"],
        "fresh_noise_per_pass": contract["bank_contract"]["fresh_noise_per_pass"],
        "platform": contract["platform"], "training_seconds": jobs[f"{run.name}_training"]["seconds"],
        "bank_seconds": jobs["bank"]["seconds"],
        "evaluation": {"rows": evaluated["population"]["rows"],
                       "draws_per_case": evaluated["population"]["draws_per_case"]},
    }

    # ---- the data (before Section 1)
    task = recipe.read_task(GRID / "cv-preparation" / "task.json")
    pilot = HestonGridPilot(task["records"], infer_c=True, c_bounds=(0.5, 2.0))
    pilot.validate_task(task["target"])
    names = list(pilot.names)
    bounds = np.asarray(pilot.bounds)
    half = np.asarray(pilot.records["half_spread"])
    design = json.loads((GRID / "cv-preparation" / "design.json").read_text())
    quotes = design["selected_quotes"]
    records = pilot.records
    forward, strike, discount, maturity = (np.asarray(records[k]) for k in ("forward", "strike", "discount", "maturity"))
    kind = np.asarray(records["option_type"])
    mid = np.asarray(records["mid"])
    ref_vol = float(design["reference_volatility"])
    delta = np.log(strike / forward) / (ref_vol * np.sqrt(maturity))
    expiry_dates = sorted({q["expiration"] for q in quotes})
    expiry = np.asarray([expiry_dates.index(q["expiration"]) for q in quotes])
    days = [int(round(365 * maturity[expiry == e][0])) for e in range(len(expiry_dates))]

    def implied_vols(prices):
        rows = np.atleast_2d(np.asarray(prices, dtype=float))
        return surface.implied_vol(rows / (discount * forward), np.log(strike / forward), maturity, kind)

    def unit_pricer(log_moneyness, mat, option_kind):
        n = np.size(log_moneyness)
        return HestonGridPilot({"forward": np.ones(n), "strike": np.exp(log_moneyness), "discount": np.ones(n),
                                "maturity": np.asarray(mat, dtype=float), "option_type": np.asarray(option_kind),
                                "mid": np.full(n, 0.01), "half_spread": np.full(n, 1e-4)},
                               infer_c=True, c_bounds=(0.5, 2.0))

    def atm_shape(log_moneyness, vols):
        near = np.argsort(np.abs(log_moneyness))[:5]
        c2, c1, _ = np.polyfit(log_moneyness[near], vols[near], 2)
        return c1, 2 * c2

    market_iv = implied_vols(mid)[0]
    vega = discount * forward * surface.bs_vega(np.log(strike / forward), maturity, market_iv)
    spread_points = 100 * half / vega
    relative = 100 * half / mid
    worst = int(np.argmax(relative))
    by_expiry = lambda values, pick: [float(values[expiry == e][pick(delta[expiry == e])]) for e in range(len(days))]
    out["data"] = {
        "snapshot_utc": design["valuation_time_utc"], "expiry_dates": expiry_dates, "expiry_days": days,
        "puts": int((kind == "put").sum()), "calls": int((kind == "call").sum()),
        "delta_range": [float(delta.min()), float(delta.max())],
        "half_spread_vol_points": {"min": float(np.nanmin(spread_points)), "median": float(np.nanmedian(spread_points)),
                                   "max": float(np.nanmax(spread_points))},
        "relative_half_spread_percent": {"min": float(relative.min()), "max": float(relative.max()),
                                         "max_contract": {"kind": str(kind[worst]), "mid": float(mid[worst]),
                                                          "days": days[expiry[worst]], "delta": float(delta[worst])}},
        "atm_iv_percent": by_expiry(100 * market_iv, lambda d: np.argmin(np.abs(d))),
        "deepest_put_iv_percent": by_expiry(100 * market_iv, np.argmin),
        "target_days": list(grid_rules.TARGET_DAYS), "max_relative_half_spread": grid_rules.MAX_RELATIVE_HALFSPREAD,
        "max_target_distance_allowed": grid_rules.MAX_Z_DISTANCE,
        "max_target_distance": max(abs(q["target_distance"]) for q in quotes),
        "target_override": design["target_overrides"][0],
    }

    # ---- the three data sets and the PyMC references (Section 4)
    panel = json.loads((GRID / "cv-preparation" / "conditional-panel.json").read_text())
    audit = json.loads((GRID / "cv-preparation" / "conditional-truth-audit.json").read_text())
    cases = {case["id"]: np.asarray(case["observations"], dtype=float) for case in panel["cases"]}
    truths = {case_id: None for case_id in cases}
    clean = {}
    for case in audit["cases"]:
        truths[case["id"]] = np.r_[case["physical_truth"], case["generating_c"]]
        clean[case["id"]] = np.asarray(case["clean_prices"], dtype=float)
    horizons = panel["products"]  # years: "control" (now to 21 days) and "affected" (81 to 325 days)
    summary = json.loads((REFERENCES / "reference-comparison.json").read_text())
    with np.load(REFERENCES / "reference-display-draws.npz", allow_pickle=False) as archive:
        draws = {c: archive[f"{c}__parameters"].reshape(-1, 6) for c in cases}
        window = {c: archive[f"{c}__products"].reshape(-1, 2) for c in cases}
    out["truths"] = {c: truths[c] for c in CASE_ORDER}
    out["horizons_days"] = {k: [365 * t for t in v] for k, v in horizons.items()}

    # ---- Section 1.2: the four features, at the market-like parameters
    base = truths["synthetic-market-like"]
    v0, kappa, v_inf, xi, rho, _ = base
    t30 = 30 / 365
    d40 = np.linspace(delta.min(), delta.max(), 40)
    k40 = d40 * ref_vol * np.sqrt(t30)
    kind40 = np.where(d40 <= 0, "put", "call")
    smile_pricer = unit_pricer(k40, np.full(d40.size, t30), kind40)

    def smiles(index, values):
        rows = np.tile(base, (len(values), 1))
        rows[:, index] = values
        return [100 * surface.implied_vol(smile_pricer.prices(row)[None], k40, np.full(d40.size, t30), kind40)[0]
                for row in rows]

    rhos, xis = [-0.3, rho, -0.9], [xi / 2, xi, min(1.5 * xi, bounds[3, 1])]
    rho_smiles, xi_smiles = smiles(4, rhos), smiles(3, xis)
    grid_t = np.geomspace(1 / 365, 1.0, 36)
    grid_d = np.array([-1.25, -0.5, 0.0, 0.5, 1.0])
    tt, dd = np.meshgrid(grid_t, grid_d, indexing="ij")
    gk, gt, gkind = (dd * ref_vol * np.sqrt(tt)).ravel(), tt.ravel(), np.where(dd.ravel() <= 0, "put", "call")
    grid_pricer = unit_pricer(gk, gt, gkind)

    def skew_curve(theta):
        iv = surface.implied_vol(grid_pricer.prices(theta)[None], gk, gt, gkind)[0].reshape(tt.shape)
        return np.array([atm_shape(grid_d * ref_vol * np.sqrt(t), iv[i]) for i, t in enumerate(grid_t)])

    feature_curve = skew_curve(base)
    ceiling = abs(rho * xi / (4 * np.sqrt(v0)))
    skew_at = lambda day: float(np.interp(np.log(day), np.log(365 * grid_t), np.abs(feature_curve[:, 0])))
    out["features"] = {
        "parameters": base, "half_life_days": 365 * np.log(2) / kappa,
        "rho": {"values": rhos, "deep_put_iv": [rho_smiles[0][0], rho_smiles[-1][0]],
                "far_call_iv": [rho_smiles[0][-1], rho_smiles[-1][-1]]},
        "sigma_v": {"values": xis, "deep_put_iv": [xi_smiles[0][0], xi_smiles[-1][0]],
                    "atm_iv": [float(np.interp(0.0, d40, xi_smiles[0])), float(np.interp(0.0, d40, xi_smiles[-1]))]},
        "smile_delta_range": [float(d40[0]), float(d40[-1])],
        "skew": {"365": skew_at(365), "30": skew_at(30), "7": skew_at(7)}, "ceiling": ceiling,
        "within_10_percent_of_ceiling_below_days": float(365 * grid_t[np.abs(feature_curve[:, 0]) >= 0.9 * ceiling].max()),
    }
    median = {n: summary["cases"]["market"]["parameters"][n]["regenerated_q05_q50_q95"][1] for n in names}
    out["feller"] = {"market_like": {"2kappa_vinf": 2 * kappa * v_inf, "sigma_v2": xi ** 2},
                     "spx_median": {"2kappa_vinf": 2 * median["kappa"] * median["theta"],
                                    "sigma_v2": median["vol_of_vol"] ** 2}}

    # ---- Sections 2.2 and 2.3: 1,000 prior simulations in the order bf.make_simulator draws them
    rng = np.random.default_rng(20261111)
    started = time.perf_counter()
    theta = np.array([rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(1000)])  # the prior, once per data set
    x = np.array([(pilot.prices(row) + row[5] * half * rng.standard_normal(half.size)) / half for row in theta])
    prior_seconds = time.perf_counter() - started
    percentile = 100 * np.mean(x * half < mid, axis=0)
    lowest = int(np.argmin(percentile))
    split = int(0.7 * len(x))
    fitted = fit_whitening(x[:split])
    values, vectors = np.linalg.eigh(np.cov(x[:split], rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    scores = (x[split:] - x[:split].mean(axis=0)) @ vectors / np.sqrt(values)
    link = np.array([abs(np.corrcoef(np.abs(scores[:, j]), theta[split:, 5])[0, 1]) for j in range(63)])
    off = lambda m: np.abs(m[~np.eye(63, dtype=bool)])
    out["prior_predictive"] = {
        "data_sets": len(x), "seconds_on_this_machine": prior_seconds,
        "market_percentile": {"min": float(percentile.min()), "max": float(percentile.max()),
                              "contracts_outside": int(np.sum((percentile == 0) | (percentile == 100))),
                              "lowest": {"days": days[expiry[lowest]], "kind": str(kind[lowest]), "delta": float(delta[lowest])}},
        "without_implied_vol_percent": float(100 * np.isnan(implied_vols(x * half)).mean()),
    }
    out["whitening_example"] = {
        "fit_rows": split, "shown_rows": len(x) - split, "largest_variance": values[0], "smallest_variance": values[-1],
        "variance_ratio": values[0] / values[-1], "c_link_last_30": float(np.mean(link[-30:])),
        "c_link_first_10": float(np.mean(link[:10])),
        "median_abs_correlation_before": float(np.median(off(np.corrcoef(x[split:], rowvar=False)))),
        "max_abs_correlation_after": float(np.max(off(np.corrcoef(fitted.transform(x[split:]), rowvar=False)))),
    }

    # ---- Section 4.1: the PyMC reference runs
    limits = json.loads((REFERENCES / "reference-protocol.json").read_text())["settings"]
    runs = {}
    for case_id in CASE_ORDER:
        record = json.loads((REFERENCES / "references" / case_id / "run.json").read_text())
        stats = json.loads((REFERENCES / "references" / case_id / "diagnostics.json").read_text())["parameters"]
        attempts = record["attempts"]
        runs[case_id] = {
            "selected_attempt": record["selected_attempt"], "chains": record["settings"]["chains"],
            "attempts": [{"tune": a["tune"], "draws": a["draws"], "likelihood_evaluations": a["likelihood_evaluations"],
                          "wall_seconds": a["wall_seconds"], "passed": a.get("mcmc_diagnostics_passed")} for a in attempts],
            "max_rhat": max(s["r_hat"] for s in stats.values()),
            "min_ess_bulk": min(s["ess_bulk"] for s in stats.values()),
            "min_ess_tail": min(s["ess_tail"] for s in stats.values()),
            "likelihood_evaluations": sum(a["likelihood_evaluations"] for a in attempts),
            "wall_seconds": sum(a["wall_seconds"] for a in attempts),
        }
    out["pymc"] = {"r_hat_max": limits["r_hat_max"], "ess_min": limits["ess_bulk_min"], "runs": runs}

    # ---- Section 4.2: each data set, network against PyMC
    per_case = {}
    for case_id in CASE_ORDER:
        par = summary["cases"][case_id]["parameters"]
        comparison = evaluated["cases"][case_id]["comparison"]
        net_q, net_mean = np.asarray(comparison["neural_quantiles_05_50_95"]), np.asarray(comparison["neural_mean"])
        width = lambda n: par[n]["regenerated_q05_q50_q95"][2] - par[n]["regenerated_q05_q50_q95"][0]
        entry = {
            "pymc_q05_q50_q95": [par[n]["regenerated_q05_q50_q95"] for n in names],
            "pymc_mean": [par[n]["regenerated_mean"] for n in names], "pymc_sd": [par[n]["regenerated_sd"] for n in names],
            "network_q05_q50_q95": net_q.T, "network_mean": net_mean,
            "centre_gap": [(net_mean[k] - par[n]["regenerated_mean"]) / par[n]["regenerated_sd"] for k, n in enumerate(names)],
            "width_ratio": [(net_q[2, k] - net_q[0, k]) / width(n) for k, n in enumerate(names)],
            "network_outside_prior_percent": 100 * evaluated["cases"][case_id]["support"]["outside_prior"]["sample_fraction"],
            "network_product_width_ratio_recorded": comparison["derived_products"]["width_ratio"],
            "variance_strike_vol_percent": {}}
        for k, (product, horizon) in enumerate(horizons.items()):
            vol = 100 * np.sqrt(window[case_id][:, k])
            entry["variance_strike_vol_percent"][product] = {
                "mean": float(vol.mean()), "q05_q10_q50_q95": np.quantile(vol, [.05, .10, .5, .95]),
                "truth": None if truths[case_id] is None else float(100 * np.sqrt(expected_variance(truths[case_id], horizon)))}
        reference = draws[case_id]
        corr = np.corrcoef(reference, rowvar=False)
        hi, lo = max(((i, j) for i in range(6) for j in range(i)), key=lambda pair: abs(corr[pair]))
        entry["strongest_correlation"] = [names[lo], names[hi], float(corr[hi, lo])]
        if truths[case_id] is not None:
            centre, sd = reference.mean(axis=0), reference.std(axis=0, ddof=1)
            z = (truths[case_id] - centre) / sd
            precision = np.linalg.inv(np.cov(reference, rowvar=False))
            distance = lambda v: np.einsum("...i,ij,...j->...", v - centre, precision, v - centre)
            entry["truth_in_posterior_sd"] = z
            entry["truth_joint_percentile"] = float(100 * np.mean(distance(reference) < distance(truths[case_id])))
        per_case[case_id] = entry
    out["data_sets"] = per_case
    like = truths["synthetic-market-like"]
    out["market_like_within_percent_of_spx_medians"] = float(np.ceil(100 * max(abs(like[j] / median[n] - 1) for j, n in enumerate(names[:5]))))

    # ---- Section 4.3: costs
    pymc_seconds = np.array([runs[c]["wall_seconds"] for c in CASE_ORDER])
    counts = {"calibration": evaluated["population"]["raw_draws_used"] // evaluated["population"]["draws_per_case"],
              "validation": evaluated["native_validation"]["raw_draws_used"] // evaluated["native_validation"]["draws_per_case"],
              "data_sets": len(evaluated["cases"])}
    out["costs"] = {"pymc_seconds": dict(zip(CASE_ORDER, pymc_seconds)), "posteriors_in_evaluation": counts,
                    "evaluation_seconds": evaluated["seconds"],
                    "network_seconds_per_posterior": evaluated["seconds"] / sum(counts.values()),
                    "pymc_hours_for_calibration_sets": counts["calibration"] * pymc_seconds.mean() / 3600}

    # ---- Section 5.1: the posterior predictive check at the 81-day expiry, PyMC draws
    at = expiry == 4
    order81 = np.argsort(delta[at])
    ppc = {"days": days[4], "delta": delta[at][order81]}
    for case_id in CASE_ORDER:
        generator = np.random.default_rng(20261112)
        sample = draws[case_id][generator.choice(len(draws[case_id]), SHOWN, replace=False)]
        y = cases[case_id]
        prices = np.array([pilot.prices(row) for row in sample])
        replicated = prices + sample[:, -1:] * half * generator.standard_normal(prices.shape)
        model_dev = np.quantile((prices - y)[:, at] / half[at], [.05, .95], axis=0)[:, order81]
        rep_dev = np.quantile((replicated - y)[:, at] / half[at], [.05, .95], axis=0)[:, order81]
        ppc[case_id] = {"model_minus_observed_q05_q95": model_dev, "replicated_minus_observed_q05_q95": rep_dev,
                        "true_minus_observed": None if case_id not in clean else ((clean[case_id] - y) / half)[at][order81],
                        "misfit_rms": float(np.sqrt(np.mean((((prices.mean(axis=0) - y) / half)[at]) ** 2))),
                        "median_c": float(np.median(sample[:, -1]))}
    out["posterior_predictive"] = ppc

    # ---- Section 5.2: calibration, with the simultaneous band of the notebook
    population = evaluated["population"]
    ranks = np.asarray(population["fractional_ranks"])
    grid = np.linspace(0, 1, min(len(ranks), 1000) + 2)[1:-1]
    uniform = np.random.default_rng(20261112).uniform(size=(10_000, len(ranks)))
    first = np.searchsorted(grid, uniform)
    rows = np.arange(10_000)[:, None] * (grid.size + 1)
    tally = np.bincount((rows + first).ravel(), minlength=10_000 * (grid.size + 1))
    tally = tally.reshape(10_000, grid.size + 1).cumsum(axis=1)[:, :grid.size]
    extremes = 2 * np.minimum(binom.cdf(tally, len(ranks), grid), binom.sf(tally - 1, len(ranks), grid)).min(axis=1)
    gamma = np.quantile(extremes, 0.05 / len(names))
    low = binom.ppf(gamma / 2, len(ranks), grid) / len(ranks) - grid
    high = binom.ppf(1 - gamma / 2, len(ranks), grid) / len(ranks) - grid
    curves = [np.searchsorted(np.sort(ranks[:, j]), grid, side="right") / len(ranks) - grid for j in range(len(names))]
    inside = [bool(np.all((c >= low) & (c <= high))) for c in curves]
    coverage = population["coverage"]
    wilson = coverage["0.9"]["wilson"]
    good = [j for j in range(len(names)) if inside[j] and wilson[j][0] <= 0.9 <= wilson[j][1]]
    shapes = {}
    for j in set(range(len(names))) - set(good):
        left, right = (np.mean(ranks[:, j] <= v) - v for v in (0.25, 0.75))
        shapes[names[j]] = "too wide" if left < 0 < right else "too narrow" if right < 0 < left else "biased"
    support = population["support"]["outside_prior"]
    out["calibration"] = {
        "band": {"grid": grid[::10], "low": low[::10], "high": high[::10], "level_per_panel": 1 - 0.05 / len(names)},
        "inside_band": dict(zip(names, inside)), "well_calibrated": [names[j] for j in good], "exceptions": shapes,
        "coverage": {level: dict(zip(names, coverage[level]["coverage"])) for level in ("0.5", "0.8", "0.9", "0.95")},
        "coverage_90_wilson": dict(zip(names, wilson)),
        "outside_prior_percent": 100 * support["sample_fraction"],
        "outside_prior_percent_by_parameter": dict(zip(names, 100 * np.asarray(support["per_dimension_count"])
                                                       / population["support"]["samples"])),
        "mean_posterior_sd": dict(zip(names, population["mean_posterior_sd"])),
        "rank_ks": dict(zip(names, population["rank_dkw"]["rank_ks"])), "dkw_bound": population["rank_dkw"]["dkw_bound"],
    }

    # ---- Section 5.3: recovery
    with np.load(EVALUATION / "recovery.npz", allow_pickle=False) as archive:
        recovery_truth, recovery_q = archive["truth"].astype(float), archive["quantiles"].astype(float)
    out["recovery"] = {
        "correlation": dict(zip(names, [float(np.corrcoef(recovery_q[1, :, j], recovery_truth[:, j])[0, 1]) for j in range(6)])),
        "mean_width_90_share_of_prior_percent": dict(zip(names, [float(100 * np.mean(recovery_q[2, :, j] - recovery_q[0, :, j])
                                                                       / (bounds[j, 1] - bounds[j, 0])) for j in range(6)])),
        "median_rmse": dict(zip(names, np.sqrt(np.mean((recovery_q[1] - recovery_truth) ** 2, axis=0)))),
    }

    # ---- Section 5.4: summary of the comparison on the two simulated data sets
    gaps = np.abs(np.concatenate([per_case[c]["centre_gap"] for c in CASE_ORDER[:2]]))
    ratios = np.concatenate([per_case[c]["width_ratio"] for c in CASE_ORDER[:2]])
    out["comparison"] = {"agree_both": int(np.sum((gaps < 0.3) & (np.abs(ratios - 1) < 0.25))), "posteriors": len(gaps),
                         "median_gap": float(np.median(gaps)), "max_gap": float(gaps.max()),
                         "width_ratio_range": [float(ratios.min()), float(ratios.max())],
                         "wider_than_1_25": int(np.sum(ratios > 1.25)), "narrower_than_1": int(np.sum(ratios < 1)),
                         "spx_max_gap": float(np.max(np.abs(per_case["market"]["centre_gap"])))}

    # ---- Section 5.5 and "What Heston misses": the best fit is the PyMC posterior mean
    best = draws["market"].mean(axis=0)
    fit = pilot.prices(best)
    residual = (cases["market"] - fit) / half
    fit_iv = implied_vols(fit)[0]
    misfit_vp = 100 * (market_iv - fit_iv)
    rows_by_expiry = [np.flatnonzero(expiry == e)[np.argsort(delta[expiry == e])] for e in range(len(days))]
    grid_vp = np.array([misfit_vp[r] for r in rows_by_expiry])
    deep = np.array([r[0] for r in rows_by_expiry])
    atm = np.array([r[np.argmin(np.abs(delta[r]))] for r in rows_by_expiry])
    rms = [float(np.sqrt(np.mean(residual[r] ** 2))) for r in rows_by_expiry]
    wing = delta < -3
    feller = 2 * best[1] * best[2]
    market_net_c = per_case["market"]["network_q05_q50_q95"][5]
    market_ref_c = per_case["market"]["pymc_q05_q50_q95"][5]
    out["market"] = {
        "best_fit_parameters": best, "rms_half_spreads": float(np.sqrt(np.mean(residual ** 2))),
        "rms_vol_points": float(np.sqrt(np.mean(misfit_vp ** 2))), "max_abs_vol_points": float(np.max(np.abs(misfit_vp))),
        "median_c": float(np.median(draws["market"][:, -1])),
        "network_c_width_ratio": float((market_net_c[2] - market_net_c[0]) / (market_ref_c[2] - market_ref_c[0])),
        "network_c_above_bound_percent": per_case["market"]["network_outside_prior_percent"],
        "rms_half_spreads_by_expiry": dict(zip(map(str, days), rms)),
        "largest_by_expiry": [{"days": days[e], "half_spreads": float(residual[r][np.argmax(np.abs(residual[r]))]),
                               "delta": float(delta[r][np.argmax(np.abs(residual[r]))])} for e, r in enumerate(rows_by_expiry)],
        "deepest_put_gap_vol_points": grid_vp[:, 0], "misfit_grid_vol_points": grid_vp,
        "put_wing_rise_spx": 100 * (market_iv[deep] - market_iv[atm]), "put_wing_rise_fit": 100 * (fit_iv[deep] - fit_iv[atm]),
        "deep_wing_puts": int(wing.sum()), "deep_wing_share_of_squared_misfit_percent": float(100 * np.sum(residual[wing] ** 2) / np.sum(residual ** 2)),
        "elsewhere_max_abs_vol_points": float(np.max(np.abs(misfit_vp[~wing]))),
        "fit_sigma_v": float(best[3]), "fit_2kappa_vinf": float(feller), "feller_factor": float(best[3] ** 2 / feller),
    }
    k = delta * ref_vol * np.sqrt(maturity)
    spx_shape = np.array([atm_shape(k[expiry == e], market_iv[expiry == e]) for e in range(len(days))])
    fit_shape = np.array([atm_shape(k[expiry == e], fit_iv[expiry == e]) for e in range(len(days))])
    curve = skew_curve(best)
    fit_ceiling = abs(best[4] * best[3] / (4 * np.sqrt(best[0])))
    power = lambda v: np.polyfit(np.log(np.asarray(days) / 365), np.log(np.abs(v)), 1)
    slope, intercept = power(spx_shape[:, 0])
    negative = 365 * grid_t[curve[:, 1] < 0]
    short = expiry == 0
    out["missing_feature"] = {
        "days_grid": 365 * grid_t, "fit_skew_grid": curve[:, 0], "fit_curvature_grid": curve[:, 1],
        "spx_skew": spx_shape[:, 0], "spx_curvature": spx_shape[:, 1], "fit_skew": fit_shape[:, 0], "fit_curvature": fit_shape[:, 1],
        "ceiling": fit_ceiling, "spx_power": float(slope), "spx_power_intercept": float(intercept),
        "fit_power": float(power(fit_shape[:, 0])[0]),
        "fit_skew_within_percent": float(100 * np.max(np.abs(fit_shape[:, 0] / spx_shape[:, 0] - 1))),
        "shortest_skew_percent_of_ceiling": float(100 * abs(fit_shape[0, 0]) / fit_ceiling),
        "fit_curvature_peak_days": days[int(np.argmax(fit_shape[:, 1]))],
        "fit_curvature_negative_below_days": float(negative.max()) if negative.size else None,
        "shortest": {"delta": delta[short][np.argsort(delta[short])], "spx_iv": 100 * market_iv[short][np.argsort(delta[short])],
                     "fit_iv": 100 * fit_iv[short][np.argsort(delta[short])],
                     "near_money_max_abs_vol_points": float(np.max(np.abs(misfit_vp[short & (np.abs(delta) < 1.3)])))},
    }

    # ---- Section 6: a forward-start call and a volatility swap under posterior draws and refits
    def price(row):
        v0_, kappa_, v_inf_, xi_, rho_ = row[:5]
        paths = products.heston_paths(dict(kappa=kappa_, theta=v_inf_, vol_of_vol=xi_), v0_, FORWARD_START[1], 252, PATHS,
                                      seed=20261115)
        return (100 * products.forward_start_call(paths, rho_, *FORWARD_START, 1.0)[0],
                100 * products.volatility_swap(paths, FORWARD_START[0])[0])

    def posterior_prices(sample, seed):
        inside = sample[np.all((sample >= bounds[:, 0]) & (sample <= bounds[:, 1]), axis=1)]
        pick = np.random.default_rng(seed).choice(len(inside), min(SHOWN, len(inside)), replace=False)
        return np.array([price(row) for row in inside[pick]])

    priced = {c: posterior_prices(draws[c], [20261115, i, 0]) for i, c in enumerate(("synthetic-market-like", "market"))}
    subsets = {"all 63 quotes": np.ones(len(half), bool),
               f"the 3 shortest expiries ({days[0]}–{days[2]} days)": expiry < 3,
               f"the 4 longest expiries ({days[3]}–{days[-1]} days)": expiry >= 3,
               "quotes near the money, |δ| ≤ 1.3": np.abs(delta) <= 1.3}
    refits = {}
    for name, mask in subsets.items():
        def misses(p, mask=mask):
            try:
                return ((cases["market"] - pilot.prices(np.r_[p, 1.0])) / half)[mask]
            except (FloatingPointError, ValueError):
                return np.full(int(mask.sum()), 1e3)
        solution = least_squares(misses, best[:5], bounds=(bounds[:5, 0], bounds[:5, 1]), x_scale="jac", max_nfev=400)
        refits[name] = {"parameters": solution.x, "prices": price(solution.x), "quotes": int(mask.sum()),
                        "rms_half_spreads_on_subset": float(np.sqrt(np.mean(solution.fun ** 2)))}
    spx_range = np.quantile(priced["market"], [.05, .95], axis=0)
    refit_prices = np.array([r["prices"] for r in refits.values()])
    out["products"] = {
        "forward_start": list(FORWARD_START), "paths": PATHS, "steps": 252, "draws_priced": SHOWN,
        "true_price": price(truths["synthetic-market-like"]),
        "posterior_prices": priced,
        "market_like_q05_q95": np.quantile(priced["synthetic-market-like"], [.05, .95], axis=0),
        "spx_q05_q95": spx_range, "refits": refits,
        "refit_spread_over_posterior_range": (refit_prices.max(axis=0) - refit_prices.min(axis=0)) / (spx_range[1] - spx_range[0]),
    }

    out["source"] = {"repository": "fmuia/sbi4finance", "branch": args.branch, "commit": args.commit,
                     "notebook": {"path": "lab/notebooks/heston_spx_calibration.py", "sha256": sha256(NOTEBOOK)},
                     "files": {str(path.relative_to(LAB.parent)): sha256(path) for path in (
                         GRID / "cv-preparation" / "design.json", GRID / "cv-preparation" / "conditional-panel.json",
                         GRID / "cv-preparation" / "conditional-truth-audit.json", EVALUATION / "evaluation.json",
                         EVALUATION / "recovery.npz", REFERENCES / "reference-comparison.json",
                         REFERENCES / "reference-display-draws.npz")},
                     "written_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z"}
    args.out.write_text(json.dumps(rounded(out), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
