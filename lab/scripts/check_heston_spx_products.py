"""Check the SPX lesson's exact product prices and record what the presented Monte Carlo got wrong.

Section 6 of notebooks/heston_spx_calibration.py prices a forward-start call, max(S_6m/S_3m - 1, 0),
and a 3-month volatility swap under posterior draws. The version presented on 15 November 2026
priced 40 draws per case by full-truncation Euler (252 steps over six months) with 4,096
antithetic paths and common random numbers. The notebook now prices every inside-prior draw
exactly (sbi4finance.calibration.heston_exact_products). This script, run from lab/, records:

1. for four parameter sets, the exact prices against Monte Carlo with 2^20 paths at 252, 1,008
   and 4,032 steps over six months, with standard errors, and against an independent route for
   the forward start (Lewis integrals of the tutorial runtime's characteristic function, averaged
   over 20,000 midpoint quantiles of the exact law of the variance at the reset);
2. the presented 40-draw Monte Carlo ranges against exact prices of the same draws and of all
   inside-prior draws, and the refit spreads, under the lesson's c <= 2 prior;
3. how much wider the posterior price bands are when c is free (the Bergomi capstone's Heston
   reference, c in [0.5, 50]), which scales the refit ratio for c free.

No MCMC, training or private input. Output: a compact results.json.
"""

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtri
from scipy.stats import ncx2, norm

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from sbi4f_tutorial.heston_options import model  # noqa: E402
from sbi4finance.calibration import heston_exact_products as exact  # noqa: E402
from sbi4finance.calibration import products  # noqa: E402
from sbi4finance.calibration.heston_grid import HestonGridPilot  # noqa: E402

EVIDENCE = LAB / "evidence/pilot-results"
GRID = EVIDENCE / "2026-11-09/heston-spread-baseline/revised-grid"
REFERENCES = EVIDENCE / "2026-11-11/heston-actual63-notebook"
CAPSTONE = EVIDENCE / "2026-11-14/bergomi-capstone/results.json"
PREDICTIVES = LAB / "notebooks/data/heston_market_predictives/results.json"
OUT = EVIDENCE / "2026-11-19/heston-spx-product-pricing"
RESET, EXPIRY = .25, .5  # forward start: strike fixed at three months, paid at six; the swap runs to RESET
TENOR = EXPIRY - RESET
PRESENTED = {"draws": 40, "paths": 4096, "steps": 252, "seed": 20261115}
JUMPS = dict(jump_intensity=0., jump_up_probability=.5, jump_up_rate=10., jump_down_rate=10.)
NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def points(price):
    """Black-equivalent forward-start volatility points of a price in units of notional."""
    return 100 * 2 / np.sqrt(TENOR) * ndtri((1 + np.asarray(price)) / 2)


def points_per_price(price):
    return 100 / (np.sqrt(TENOR) * norm.pdf(ndtri((1 + np.asarray(price)) / 2)))


def as_parameters(draws):
    draws = np.atleast_2d(draws)
    return dict(kappa=draws[:, 1], theta=draws[:, 2], vol_of_vol=draws[:, 3], rho=draws[:, 4]), draws[:, 0]


def exact_prices(draws):
    """Exact forward start (volatility points) and volatility swap (points), with quadrature changes."""
    parameters, v0 = as_parameters(draws)
    call, call_change = exact.forward_start_call(parameters, v0, RESET, EXPIRY)
    swap, swap_change = exact.volatility_swap({key: parameters[key] for key in ("kappa", "theta", "vol_of_vol")},
                                              v0, RESET)
    change = np.maximum(call_change * points_per_price(call), 100 * swap_change)
    return np.column_stack([points(call), 100 * swap]), float(change.max())


def presented_prices(theta):
    """The presented notebook's Monte Carlo price of one draw, with its standard errors."""
    v0, kappa, long_run, vol_of_vol, rho = theta[:5]
    paths = products.heston_paths(dict(kappa=kappa, theta=long_run, vol_of_vol=vol_of_vol), v0, EXPIRY,
                                  PRESENTED["steps"], PRESENTED["paths"], seed=PRESENTED["seed"])
    call, call_se = products.forward_start_call(paths, rho, RESET, EXPIRY, 1.0)
    swap, swap_se = products.volatility_swap(paths, RESET)
    return np.array([points(call), 100 * swap, call_se * points_per_price(call), 100 * swap_se])


def monte_carlo(theta, steps, paths, seed, chunk=4096):
    """Chunked full-truncation Euler estimates (products.py) with independent streams per chunk."""
    v0, kappa, long_run, vol_of_vol, rho = theta[:5]
    estimates = []
    for index in range(paths // chunk):
        simulated = products.heston_paths(dict(kappa=kappa, theta=long_run, vol_of_vol=vol_of_vol), v0, EXPIRY,
                                          steps, chunk, seed=[seed, steps, index])
        estimates.append((*products.forward_start_call(simulated, rho, RESET, EXPIRY, 1.0),
                          *products.volatility_swap(simulated, RESET)))
    estimates = np.array(estimates)
    count = len(estimates)
    call, call_se = estimates[:, 0].mean(), np.sqrt(np.sum(estimates[:, 1] ** 2)) / count
    swap, swap_se = estimates[:, 2].mean(), np.sqrt(np.sum(estimates[:, 3] ** 2)) / count
    return {"forward_start": float(points(call)), "forward_start_se": float(call_se * points_per_price(call)),
            "volatility_swap": float(100 * swap), "volatility_swap_se": float(100 * swap_se)}


def quantile_average(theta, count=20_000, near=256, far=512):
    """The forward start as Lewis prices, Gauss-Legendre on [0, 25] and [25, 1500], averaged over V(RESET)."""
    v0, kappa, long_run, vol_of_vol, rho = theta[:5]
    scale = vol_of_vol**2 * -np.expm1(-kappa * RESET) / (4 * kappa)
    law = ncx2(4 * kappa * long_run / vol_of_vol**2, v0 * np.exp(-kappa * RESET) / scale, scale=scale)
    variances = law.ppf((np.arange(count) + .5) / count)
    roots, weights = np.polynomial.legendre.leggauss(near)
    u = [12.5 * (roots + 1)]
    w = [12.5 * weights]
    roots, weights = np.polynomial.legendre.leggauss(far)
    u.append(25 + 737.5 * (roots + 1))
    w.append(737.5 * weights)
    u, w = np.concatenate(u), np.concatenate(w)
    parameters = dict(kappa=kappa, theta=long_run, vol_of_vol=vol_of_vol, rho=rho, **JUMPS)
    calls = [1 - (np.real(model.characteristic_function(u - .5j, TENOR, parameters, float(v))) / (u * u + .25)) @ w
             / np.pi for v in variances]
    return float(points(np.mean(calls)))


def quantiles(values, digits):
    low, high = np.quantile(values, [.05, .95])
    return [round(float(low), digits), round(float(high), digits)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--paths-exponent", type=int, default=20)
    parser.add_argument("--steps", default="252,1008,4032")
    args = parser.parse_args()
    started = time.time()

    task = json.loads((GRID / "cv-preparation/task.json").read_text())
    task = task.get("task", task)
    pilot = HestonGridPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    pilot.validate_task(task["target"])
    if list(pilot.names)[:5] != list(NAMES):
        raise ValueError("unexpected parameter order")
    bounds = np.asarray(pilot.bounds)
    half_spread = np.asarray(pilot.records["half_spread"])
    panel = json.loads((GRID / "cv-preparation/conditional-panel.json").read_text())
    audit = json.loads((GRID / "cv-preparation/conditional-truth-audit.json").read_text())
    observations = {case["id"]: np.asarray(case["observations"], float) for case in panel["cases"]}
    truth = next(np.asarray(case["physical_truth"], float) for case in audit["cases"]
                 if case["id"] == "synthetic-market-like")
    display = REFERENCES / "reference-display-draws.npz"
    comparison = json.loads((REFERENCES / "reference-comparison.json").read_text())
    if digest(display) != comparison["display_cache"]["sha256"]:
        raise ValueError("the reference draws differ from their recorded identity")
    with np.load(display, allow_pickle=False) as archive:
        draws = {case: archive[f"{case}__parameters"].reshape(-1, 6) for case in ("synthetic-market-like", "market")}

    # The four least-squares refits of the notebook, weighted by the half-spreads (c fixed).
    design = json.loads((GRID / "cv-preparation/design.json").read_text())
    records = pilot.records
    forward, strike, maturity = (np.asarray(records[key]) for key in ("forward", "strike", "maturity"))
    delta = np.log(strike / forward) / (float(design["reference_volatility"]) * np.sqrt(maturity))
    dates = sorted({quote["expiration"] for quote in design["selected_quotes"]})
    expiry = np.asarray([dates.index(quote["expiration"]) for quote in design["selected_quotes"]])
    subsets = {"all 63 quotes": np.ones(len(half_spread), bool), "3 shortest expiries": expiry < 3,
               "4 longest expiries": expiry >= 3, "near the money": np.abs(delta) <= 1.3}
    start = draws["market"].mean(axis=0)
    refits = {}
    for name, mask in subsets.items():
        def misfit(p, mask=mask):
            try:
                return ((observations["market"] - pilot.prices(np.r_[p, 1.0])) / half_spread)[mask]
            except (FloatingPointError, ValueError):
                return np.full(int(mask.sum()), 1e3)
        refits[name] = least_squares(misfit, start[:5], bounds=(bounds[:5, 0], bounds[:5, 1]), x_scale="jac",
                                     max_nfev=400).x

    # 1. Convergence at four parameter sets.
    sets = {"market-like generating values": truth[:5], "SPX posterior median": np.median(draws["market"], 0)[:5],
            "refit to the 3 shortest expiries": refits["3 shortest expiries"],
            "refit to the 4 longest expiries": refits["4 longest expiries"]}
    steps = [int(value) for value in args.steps.split(",")]
    convergence = []
    for name, theta in sets.items():
        prices, change = exact_prices(theta)
        row = {"name": name, "parameters": dict(zip(NAMES, map(float, theta))),
               "exact": {"forward_start": float(prices[0, 0]), "volatility_swap": float(prices[0, 1]),
                         "quadrature_change": change},
               "quantile_average_forward_start": quantile_average(theta),
               "presented_monte_carlo": dict(zip(("forward_start", "volatility_swap", "forward_start_se",
                                                  "volatility_swap_se"), map(float, presented_prices(theta)))),
               "monte_carlo": {}}
        for count in steps:
            row["monte_carlo"][str(count)] = monte_carlo(theta, count, 2 ** args.paths_exponent, 20261119)
            print(name, count, row["monte_carlo"][str(count)], flush=True)
        convergence.append(row)

    # 2. Ranges under the c <= 2 posterior: the presented selection, its exact prices, and all draws.
    ranges, width = {}, {}
    for index, case in enumerate(("synthetic-market-like", "market")):
        inside = draws[case][np.all((draws[case] >= bounds[:, 0]) & (draws[case] <= bounds[:, 1]), axis=1)]
        chosen = np.random.default_rng([PRESENTED["seed"], index, 0]).choice(len(inside), PRESENTED["draws"],
                                                                             replace=False)
        presented = np.array([presented_prices(theta) for theta in inside[chosen]])
        same, _ = exact_prices(inside[chosen])
        everything, change = exact_prices(inside)
        ranges[case] = {
            "inside_prior_draws": len(inside),
            "presented": {"forward_start": quantiles(presented[:, 0], 3), "volatility_swap": quantiles(presented[:, 1], 2),
                          "median_se": [round(float(np.median(presented[:, 2])), 3),
                                        round(float(np.median(presented[:, 3])), 3)]},
            "exact_same_draws": {"forward_start": quantiles(same[:, 0], 3), "volatility_swap": quantiles(same[:, 1], 3),
                                 "correlation_with_presented": [round(float(np.corrcoef(presented[:, j], same[:, j])[0, 1]), 2)
                                                                for j in range(2)],
                                 "width_ratio_presented_over_exact": [
                                     round(float(np.ptp(np.quantile(presented[:, j], [.05, .95]))
                                                 / np.ptp(np.quantile(same[:, j], [.05, .95]))), 2) for j in range(2)]},
            "exact_all_draws": {"forward_start": quantiles(everything[:, 0], 3),
                                "volatility_swap": quantiles(everything[:, 1], 3), "quadrature_change": change},
        }
        if case == "market":  # posterior 90% widths, unrounded, for the refit ratios
            width = {"presented": np.ptp(np.quantile(presented[:, :2], [.05, .95], axis=0), axis=0),
                     "exact": np.ptp(np.quantile(everything, [.05, .95], axis=0), axis=0)}
    truth_prices = {"presented": [round(float(value), 3) for value in presented_prices(truth)[:2]],
                    "exact": [round(float(value), 4) for value in exact_prices(truth)[0][0]]}
    refit_presented = np.array([presented_prices(theta)[:2] for theta in refits.values()])
    refit_exact, _ = exact_prices(np.array(list(refits.values())))
    spread = {"presented": np.ptp(refit_presented, axis=0), "exact": np.ptp(refit_exact, axis=0)}

    # 3. c free: how much wider the clean-price bands are under the capstone's Heston reference.
    capstone = json.loads(CAPSTONE.read_text())["fit_comparison"]["heston"]
    bands = {"free": np.asarray(capstone["price_q05_q50_q95_raw"]),
             "bounded": np.asarray(json.loads(PREDICTIVES.read_text())["methods"]["pymc"]["clean_q05_q50_q95_raw"])}
    factor = (bands["free"][2] - bands["free"][0]) / (bands["bounded"][2] - bands["bounded"][0])
    shift = np.abs(bands["free"][1] - bands["bounded"][1]) / half_spread
    ratio = spread["exact"] / width["exact"]
    result = {
        "format": "heston_spx_product_pricing_check_v1",
        "contract": {"forward_start": "max(S(0.5)/S(0.25) - 1, 0), zero carry, Black-equivalent volatility points",
                     "volatility_swap": "100 E[sqrt(I(0.25)/0.25)], annualized volatility points"},
        "presented_method": PRESENTED,
        "monte_carlo": {"paths": 2 ** args.paths_exponent, "steps_over_six_months": steps, "seed": 20261119,
                        "scheme": "products.heston_paths full-truncation Euler, antithetic, chunks of 4,096 paths"},
        "convergence": convergence,
        "ranges_c_at_most_2": ranges,
        "market_like_truth": truth_prices,
        "refits": {name: {"parameters": dict(zip(NAMES, map(float, theta))),
                          "presented": [round(float(v), 3) for v in refit_presented[i]],
                          "exact": [round(float(v), 4) for v in refit_exact[i]]}
                   for i, (name, theta) in enumerate(refits.items())},
        "refit_ratio_c_at_most_2": {"presented": [round(float(v), 1) for v in spread["presented"] / width["presented"]],
                                    "exact_all_draws": [round(float(v), 1) for v in ratio]},
        "c_free": {"capstone_c_q05_q50_q95": capstone["c_q05_q50_q95"],
                   "band_width_factor_median": round(float(np.median(factor)), 2),
                   "band_width_factor_iqr": [round(float(v), 2) for v in np.quantile(factor, [.25, .75])],
                   "band_width_factor_range": [round(float(factor.min()), 2), round(float(factor.max()), 2)],
                   "median_shift_half_spreads": round(float(np.median(shift)), 2),
                   "refit_ratio_estimate": [round(float(v), 1) for v in ratio / np.median(factor)]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": metadata.version("scipy"), "platform": platform.platform()},
        "sources": {name: digest(path) for name, path in {
            "script": Path(__file__), "exact_products": Path(exact.__file__), "monte_carlo_products": Path(products.__file__),
            "reference_display_draws": display, "capstone_results": CAPSTONE, "market_predictives": PREDICTIVES}.items()},
        "seconds": round(time.time() - started),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in ("ranges_c_at_most_2", "market_like_truth",
                                                    "refit_ratio_c_at_most_2", "c_free")}, indent=1))


if __name__ == "__main__":
    main()
