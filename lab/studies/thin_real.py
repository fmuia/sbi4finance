"""A real thin chain (study S11): EWG's posterior, its products and reserves, against the classical fits.

    python lab/studies/thin_real.py --checkpoint S2T_RUN --observation OBSERVATION.json --out NEW_DIR \
        [--draws 1000 --proposals 1024 --rounds 4 --paths 8192 --product-draws 512 --product-paths 8192 \
         --scales 1,0.5,2 --predictive-draws 200 --workers 4 --seed 0]

The data are the chain's own cleaned quotes at their own strikes and trading days (OBSERVATION.json, written by
lab/checks/thin_chains/build_layout.py); each quote's noise sd is its half-spread in price units,
floored at the design noise. The thin network reads the chain on its layout and gives a proposal; importance
sampling with the Monte Carlo likelihood at the chain's own quotes gives the posterior (thin.importance_rounds:
every proposal on its own seed, the unbiased estimate N(y; P_hat, diag(noise^2) - S), rounds refitted to the
weights and pooled with the balance heuristic until k-hat is at most 0.5 or --rounds is reached).

Each noise scale s of --scales multiplies every noise sd (s = 1: the half-spread is one sd); the network is
given the same s. Reported per scale: the posterior of (H, eta, rho) and the curve; the six thin products
(sbi4finance.calibration.thin_products) under the posterior and under the network alone, with one-sided 90%
bounds; the Gauss-Newton fit (started from the quotes and the prior box, not from the posterior), with the curve
beyond the last quote flat or extrapolated by a power law, and the fit with H fixed at 0.10, each with
delta-method bounds. The quotes end at 126 trading days, so the likelihood does not depend on the last
forward-variance level (126 to 252 days): its posterior given the other targets is the prior's. The forward start
from six months to one year is therefore also reported with that level flat, and with its prior increment doubled,
which are the posteriors under those two other priors for the last level. At s = 1, the posterior predictive
check at every quote. Writes real.json, draws.npz, products.png and predictive.png.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import correction, surface, thin, thin_products  # noqa: E402
from thin_coverage import COLOURS, finite, truth_arms  # noqa: E402

ONE_SIDED = 1.2815515655446004
H_FIXED = 0.10
LONG_PRODUCT = "forward_start_6m_1y"
_TAG = 0x53313131  # "S111"


def chain_data(observation):
    """Contracts, prices and noise sds of the chain's own quotes, in forward units."""
    quotes = observation["quotes"]
    k = np.array([q["k"] for q in quotes])
    T = np.array([q["trading_days"] for q in quotes]) / surface.TRADING_DAYS
    delta = k / (surface.SIGMA_REF * np.sqrt(T))
    iv = np.array([q["implied_vol"] for q in quotes])
    half = np.array([q["half_spread_vol"] for q in quotes])
    # Each quote's half-spread in price units, floored at the design noise in price units.
    noise = np.maximum(surface.bs_vega(k, T, iv) * half,
                       surface.bs_vega(k, T, surface.SIGMA_REF) * surface.implied_vol_noise(delta))
    contracts = [{"strike": float(np.exp(ki)), "maturity": float(t), "option_type": q["kind"]}
                 for ki, t, q in zip(k, T, quotes)]
    prices = np.array([q["price_forward_units"] for q in quotes])
    return contracts, prices, noise, [q["trading_days"] for q in quotes]


def product_job(job):
    task, target, paths, seed = job
    return thin_products.price(task, target[:3], target[3:], paths, seed)[0]


def seed_of(seed, purpose, j):
    return int(np.random.SeedSequence([_TAG, int(seed), int(purpose), int(j)]).generate_state(1, dtype=np.uint32)[0])


def q10_50_90(values):
    return np.quantile(np.asarray(values, dtype=float), [0.1, 0.5, 0.9], axis=0)


def analyse(task, layout, workflow, observation, y, contracts, noise, days, scale, args, pool):
    names = thin.target_names(task)
    seed = seed_of(args.seed, 10, int(round(1000 * scale)))
    network = thin.sample_thin(workflow, task, layout, np.array([observation["prices_forward_units"]]), args.draws,
                               scale=scale, seed=seed % 2**31)[0]
    price_one = partial(thin.priced_for_audit, task=task, contracts=contracts, noise=noise * scale, paths=args.paths)
    audit = thin.importance_rounds(network, task, price_one, y, noise * scale, args.proposals, seed,
                                   max_rounds=args.rounds, khat_target=0.5, map_fn=partial(pool.map, chunksize=8))
    for r in audit["rounds"]:
        print(json.dumps({"scale": scale, **r}), flush=True)
    summary = correction.weighted_summary(audit["targets"], audit["weights"])
    posterior = {n: {"mean": float(summary["mean"][i]), "sd": float(summary["sd"][i]),
                     "q05_q50_q95": summary["quantiles"][i].tolist()} for i, n in enumerate(names)}
    network_summary = {n: {"mean": float(network[:, i].mean()), "sd": float(network[:, i].std()),
                           "q05_q50_q95": np.quantile(network[:, i], [0.05, 0.5, 0.95]).tolist()}
                       for i, n in enumerate(names)}

    # products under the posterior and under the network alone, common random numbers
    product_seed = seed_of(args.seed, 1, 0)

    def prices_at(rows):
        return np.array(list(pool.map(product_job, [(task, t, args.product_paths, product_seed) for t in rows])))

    picks = correction.resample(audit["targets"], audit["weights"], args.product_draws, seed_of(seed, 2, 0))
    post_prices = prices_at(picks)
    net_pick = network[np.random.default_rng(seed_of(seed, 3, 0)).choice(len(network), args.product_draws,
                                                                          replace=False)]
    net_prices = prices_at(net_pick)
    products = {name: {"posterior_q10_q50_q90": q10_50_90(post_prices[:, j]).tolist(),
                       "network_q10_q50_q90": q10_50_90(net_prices[:, j]).tolist()}
                for j, name in enumerate(thin_products.NAMES)}
    j = list(thin_products.NAMES).index(LONG_PRODUCT)
    long_end = {"prior": products[LONG_PRODUCT]["posterior_q10_q50_q90"]}
    for arm in ("flat", "wide"):
        moved = np.array([truth_arms(task, row)[arm] for row in picks])
        long_end[arm] = q10_50_90(prices_at(moved)[:, j]).tolist()

    # the classical fits on the chain's own quotes, started from the quotes and the prior box
    def product_fn(row):
        return thin_products.price(task, row[:3], row[3:], args.product_paths, product_seed)[0]

    fit_seed = seed_of(args.seed, 4, 0)
    starts, chi2_starts = thin.classical_screen(task, y, contracts, noise * scale, days, args.paths, fit_seed,
                                                candidates=args.screen)
    fit = thin.classical_fit(task, y, contracts, noise * scale, starts[0], days, args.paths, fit_seed)
    if not fit["converged"]:
        second = thin.classical_fit(task, y, contracts, noise * scale, starts[1], days, args.paths, fit_seed)
        fit = second if second["chi2"] < fit["chi2"] else fit
    fixed_start = fit["row"].copy()
    fixed_start[task["parameter_names"].index("hurst")] = H_FIXED
    fit_h = thin.classical_fit(task, y, contracts, noise * scale, fixed_start, days, args.paths, fit_seed,
                               fixed_hurst=H_FIXED)
    classical = {}
    for label, f, row_fn in (("gauss_newton", fit, None),
                             ("gauss_newton_power", fit, lambda r: thin.power_law_extension(task, r, days)),
                             ("h_fixed", fit_h, None)):
        centre, sd = thin.delta_method(task, f, product_fn if row_fn is None else (lambda r: product_fn(row_fn(r))))
        classical[label] = {"chi2": f["chi2"], "converged": f["converged"], "nfev": f["nfev"],
                            "fit": dict(zip(names, f["row"].tolist())),
                            "products": {name: {"centre": float(centre[k]), "sd": float(sd[k]),
                                                "lower": float(centre[k] - ONE_SIDED * sd[k]),
                                                "upper": float(centre[k] + ONE_SIDED * sd[k])}
                                         for k, name in enumerate(thin_products.NAMES)}}
    classical["gauss_newton"]["screen_chi2"] = chi2_starts[:3].tolist()
    result = {"scale": scale, "importance_sampling": audit["rounds"], "khat": audit["khat"], "ess": audit["ess"],
              "unpriced_proposals": audit["unpriced"], "posterior": posterior, "network": network_summary,
              "products": products, "long_end_laws": long_end, "classical": classical}
    arrays = {"targets": audit["targets"], "weights": audit["weights"], "network": network,
              "posterior_products": post_prices, "network_products": net_prices}
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--product-draws", type=int, default=512)
    parser.add_argument("--product-paths", type=int, default=8192)
    parser.add_argument("--scales", default="1", help="comma-separated noise scales; 1 first")
    parser.add_argument("--screen", type=int, default=16)
    parser.add_argument("--predictive-draws", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from evaluate_thin import load_thin_network

    task, layout, workflow, record = load_thin_network(args.checkpoint)
    observation = json.loads(args.observation.read_text())
    if observation["ticker"] != layout["ticker"] or observation["asof"] != layout["asof"]:
        raise ValueError("the observation and the network's layout come from different chains")
    if observation.get("forward") != layout.get("forward"):
        raise ValueError("the observation and the layout were read with different forwards")
    scales = [float(x) for x in args.scales.split(",")]
    if scales[0] != 1.0:
        raise ValueError("the first scale must be 1: the predictive check and the figures use it")
    lo, hi = thin.SCALE_BOUNDS
    if any(not lo <= s <= hi for s in scales):
        raise ValueError(f"noise scales must lie in the network's training range {thin.SCALE_BOUNDS}")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contracts, y, noise, days = chain_data(observation)
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    results, arrays = [], {}
    for scale in scales:
        result, arr = analyse(task, layout, workflow, observation, y, contracts, noise, days, scale, args, pool)
        results.append(result)
        arrays.update({f"{k}_scale_{scale:g}": v for k, v in arr.items()})
        # after every scale, so a later failure keeps the s = 1 draws that R9 needs
        np.savez_compressed(args.out / "draws.npz", **arrays)

    # the posterior predictive check at every quote, at the chain's own noise (s = 1)
    base = arrays["targets_scale_1"], arrays["weights_scale_1"]
    pp_rows = correction.resample(base[0], base[1], args.predictive_draws, seed_of(args.seed, 5, 0))
    price_one = partial(thin.priced_for_audit, task=task, contracts=contracts, noise=noise, paths=args.paths)
    seeds = [seed_of(args.seed, 6, j) for j in range(len(pp_rows))]
    pp = np.array([p for p, _, _ in pool.map(price_one, pp_rows, seeds, chunksize=4) if p is not None])
    pool.shutdown()
    rng = np.random.default_rng(seed_of(args.seed, 7, 0))
    replicated = pp + noise * rng.standard_normal(pp.shape)
    low, mid, high = np.quantile(replicated, [0.05, 0.5, 0.95], axis=0)
    standardized = (y - replicated.mean(axis=0)) / replicated.std(axis=0)
    inside = (y >= low) & (y <= high)
    predictive = {"quotes": len(y), "share_inside_90": float(np.mean(inside)),
                  "standardized_residuals": standardized.tolist(),
                  "by_expiry": {str(d): {"share_inside_90": float(np.mean(inside[np.array(days) == d])),
                                         "max_abs_standardized": float(np.max(np.abs(standardized[np.array(days) == d])))}
                                for d in sorted(set(days))}}
    report = {"study": "S11", "ticker": observation["ticker"], "asof": observation["asof"],
              "source": observation["source"], "forward": observation.get("forward"), "quotes": len(y),
              "trading_days": sorted(set(days)),
              "checkpoint_sha256": hashlib.sha256((args.checkpoint / "model.keras").read_bytes()).hexdigest(),
              "settings": {k: getattr(args, k) for k in ("draws", "proposals", "rounds", "paths", "product_draws",
                                                         "product_paths", "scales", "screen", "predictive_draws",
                                                         "seed")},
              "by_scale": results, "posterior_predictive": predictive,
              "wall_seconds": time.perf_counter() - started}
    np.savez_compressed(args.out / "draws.npz", **arrays)  # again before the report, which could still fail
    (args.out / "real.json").write_text(json.dumps(finite(report), indent=2, allow_nan=False) + "\n")
    figures(results[0], report, y, low, high, days, args.out)
    print(json.dumps({"khat_by_scale": [r["khat"] for r in results],
                      "predictive_share_inside_90": predictive["share_inside_90"]}, indent=1))


def figures(result, report, y, low, high, days, out):
    from sbi4finance.calibration.plots import INK, SERIES, plt, style

    style()
    names = list(thin_products.NAMES)
    fig, ax = plt.subplots(figsize=(11.5, 3.8), constrained_layout=True)
    for j, name in enumerate(names):
        p = result["products"][name]
        price = p["posterior_q10_q50_q90"][1]
        rows = [("network", p["network_q10_q50_q90"], COLOURS["network"]),
                ("posterior", p["posterior_q10_q50_q90"], COLOURS["posterior"])]
        for label, key in (("Gauss–Newton, flat", "gauss_newton"), ("Gauss–Newton, power law", "gauss_newton_power"),
                           ("H fixed", "h_fixed")):
            c = result["classical"][key]["products"][name]
            rows.append((label, [c["lower"], c["centre"], c["upper"]], COLOURS[key]))
        for i, (label, (lo, mid, hi), colour) in enumerate(rows):
            x = j + (i - 2) * 0.15
            ax.plot([x, x], [lo / price, hi / price], color=colour, lw=3, solid_capstyle="butt",
                    label=label if j == 0 else None)
            ax.plot(x, mid / price, "o", color=colour, ms=4)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax.set_ylabel("price / posterior median")
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    ax.set_title(f"{report['ticker']} on {report['asof']}: one-sided 90% bounds by method", loc="left", fontsize=10,
                 color=INK)
    fig.savefig(out / "products.png", dpi=160)
    fig, axes = plt.subplots(1, len(set(days)), figsize=(12, 3.4), constrained_layout=True, squeeze=False)
    for ax, d in zip(axes[0], sorted(set(days))):
        sel = np.array(days) == d
        x = np.arange(sel.sum())
        ax.fill_between(x, low[sel] / y[sel], high[sel] / y[sel], color=SERIES[1], alpha=0.3, label="90% predictive")
        ax.plot(x, np.ones(sel.sum()), "o", color="0.2", ms=3, label="quote")
        ax.set_title(f"{d} trading days", fontsize=9, loc="left")
        ax.set_xlabel("quote, by strike")
    axes[0][0].set_ylabel("price / quote")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.savefig(out / "predictive.png", dpi=160)


if __name__ == "__main__":
    main()
