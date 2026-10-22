"""Smile facts for the lesson's opening acts (figures S2-1 and S2-2 of the design document).

    python tutorial_final/studies/smile_facts.py example --model rough --out NEW_DIR [--seed 0]
    python tutorial_final/studies/smile_facts.py example --model heston --out NEW_DIR [--seed 0]
    python tutorial_final/studies/smile_facts.py prior-predictive --bank BANK --out NEW_DIR [--surfaces 1000]

example: one surface at the centre of the prior box (rough Bergomi with a flat
25% forward-variance curve, or Heston), priced by the task's own pricer, with
one draw of quote noise. Reported: the implied volatilities of the clean and of
the noisy quotes, the at-the-money level, skew and curvature per maturity, and
the power law fitted to the skew across maturities; for rough Bergomi also the
short-maturity formula of the design document, maturity by maturity.

prior-predictive: four statistics over held-out surfaces of a bank, from the
clean model prices and from the noisy quotes: the 1-month ATM implied
volatility, the 1-month ATM skew, the skew's power-law exponent across the 7
maturities and the 1-month curvature. This is the prior predictive
distribution in which study S8 places the real smile; the real design grid is
read off a smoothing spline, so it is compared with the clean distribution.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank, surface, tasks  # noqa: E402
from heston_check import jsonable  # noqa: E402

ONE_MONTH = 21 / surface.TRADING_DAYS
TENORS = {5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m", 126: "6m", 252: "1y"}
SPX_EXPONENT = -0.4  # Bayer, Friz and Gatheral (2016): SPX ATM skew close to a power law with exponent about -0.4


def rough_skew_formula(parameters, maturities):
    """Short-maturity ATM skew of rough Bergomi, rho eta sqrt(2H) / (2 (H + 1/2)(H + 3/2)) T^(H - 1/2)."""
    h, eta, rho = parameters["hurst"], parameters["eta"], parameters["rho"]
    return rho * eta * np.sqrt(2 * h) / (2 * (h + 0.5) * (h + 1.5)) * np.asarray(maturities) ** (h - 0.5)


def implied(task, prices):
    k, maturity, kind = surface.contract_arrays(task["context"]["contracts"])
    return surface.implied_vol(prices, k, maturity, kind)


def statistics(task, implied_vols):
    k, maturity, _ = surface.contract_arrays(task["context"]["contracts"])
    maturities, atm, skew, curvature = surface.smile_statistics(implied_vols, k, maturity)
    try:
        exponent, scale = surface.skew_power_law(maturities, skew)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        exponent, scale = np.nan, np.nan
    month = int(np.argmin(np.abs(maturities - ONE_MONTH)))
    return {"maturities": maturities, "atm": atm, "skew": skew, "curvature": curvature,
            "skew_exponent": exponent, "skew_scale": scale,
            "one_month": {"atm": atm[month], "skew": skew[month], "curvature": curvature[month]}}


def task_for(model):
    return tasks.rough_task() if model == "rough" else tasks.heston_task()


def example(args):
    task = task_for(args.model)
    theta = np.asarray(task["prior_bounds"], dtype=float).mean(axis=1)
    context = np.full(tasks.context_size(task), 0.25**2)
    diagnostics = [] if args.model == "rough" else None
    started = time.process_time()
    prices, errors, _, difference, used = tasks.price(task, theta, context, args.seed, diagnostics)
    cpu = time.process_time() - started
    sigma = tasks.sigma(task)
    noisy = prices + sigma * np.random.default_rng([args.seed, 0x534D494C]).standard_normal(len(prices))
    clean_iv, noisy_iv = implied(task, prices), implied(task, noisy)
    k, maturity, _ = surface.contract_arrays(task["context"]["contracts"])
    iv_noise = sigma / surface.bs_vega(k, maturity, clean_iv)
    clean_stats, noisy_stats = statistics(task, clean_iv), statistics(task, noisy_iv)
    parameters = tasks.model_parameters(task, theta)
    report = {
        "study": "smile_facts:example", "model": args.model, "task_id": task["task_id"],
        "parameters": dict(zip(task["parameter_names"], theta.tolist())),
        "forward_variance_levels": context.tolist(), "seed": args.seed,
        "pricing": {"cpu_seconds": cpu, "paths_per_block": used.tolist(),
                    "max_mc_se_over_sigma": float(np.max(errors / sigma)),
                    "median_mc_se_over_sigma": float(np.median(errors / sigma)),
                    "max_quadrature_difference_over_sigma": float(np.max(np.abs(difference) / sigma))},
        "prices": prices.tolist(), "noisy_quotes": noisy.tolist(), "quote_noise_sd": sigma.tolist(),
        "implied_vol_clean": clean_iv.tolist(), "implied_vol_noisy": noisy_iv.tolist(),
        "implied_vol_noise_sd": iv_noise.tolist(),
        "clean": clean_stats, "noisy": noisy_stats,
    }
    if args.model == "rough":
        report["short_maturity_formula_skew"] = rough_skew_formula(parameters, clean_stats["maturities"]).tolist()
        ratio = np.empty(len(prices))
        for block in diagnostics:
            ratio[block["contracts"]] = block["controlled_variance"] / block["plain_variance"]
        report["control_variates"] = {
            "variance_ratio_with_over_without": ratio.tolist(),
            "quantiles": dict(zip(("min", "q25", "median", "q75", "max"),
                                  np.quantile(ratio, [0, 0.25, 0.5, 0.75, 1]).tolist())),
            "note": "per quote, the variance of the antithetic pair means with the two controls per maturity over "
                    "the variance without them (1 - R^2), at the paths used"}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "example.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    example_figure(task, report, args.out / "example-surface.png")
    print(json.dumps(jsonable({"cpu_seconds": cpu, "one_month": clean_stats["one_month"],
                               "skew_exponent": clean_stats["skew_exponent"],
                               "pricing": report["pricing"]}), indent=1))


def example_figure(task, report, path):
    from sbi4finance.calibration.plots import BASELINE, INK, INK_2, SERIES, plt, style

    style()
    k, maturity, _ = surface.contract_arrays(task["context"]["contracts"])
    delta = surface.standardized_moneyness(k, maturity)
    clean, noisy = np.array(report["implied_vol_clean"]), np.array(report["implied_vol_noisy"])
    bars = np.array(report["implied_vol_noise_sd"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    days = sorted({int(round(t * surface.TRADING_DAYS)) for t in maturity})
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(days)))
    for color, day in zip(shades, days):
        chosen = np.isclose(maturity * surface.TRADING_DAYS, day)
        left.plot(delta[chosen], 100 * clean[chosen], color=color, linewidth=1.3, label=TENORS.get(day, f"{day}d"))
        left.errorbar(delta[chosen], 100 * noisy[chosen], yerr=100 * bars[chosen], fmt="o", ms=3.2, color=color,
                      elinewidth=0.8, capsize=0)
    left.set_xlabel("standardized moneyness δ = ln(K/F) / (0.2 √T)")
    left.set_ylabel("implied volatility (%)")
    left.legend(title="maturity", fontsize=7.5, title_fontsize=7.5, ncol=2, loc="upper right")
    name = "rough Bergomi" if report["model"] == "rough" else "Heston"
    left.set_title(f"{name} at the centre of the prior: model smile (lines) and 63 noisy quotes (±1 sd)",
                   loc="left", fontsize=8.5)
    stats = report["clean"]
    t, skew = np.array(stats["maturities"]), np.abs(np.array(stats["skew"]))
    right.loglog(t * surface.TRADING_DAYS, skew, "o", color=SERIES[0], ms=5, label="model, from the clean smile")
    grid = np.geomspace(t.min(), t.max(), 50)
    right.loglog(grid * surface.TRADING_DAYS, stats["skew_scale"] * grid ** stats["skew_exponent"], "--",
                 color=SERIES[0], linewidth=1, label=f"power-law fit, exponent {stats['skew_exponent']:.2f}")
    if "short_maturity_formula_skew" in report:
        theta = report["parameters"]
        formula = np.abs(rough_skew_formula(theta, grid))
        right.loglog(grid * surface.TRADING_DAYS, formula, ":", color=INK_2, linewidth=1.2,
                     label=f"short-maturity formula, exponent H − ½ = {theta['hurst'] - 0.5:.2f}")
    noisy_skew = np.abs(np.array(report["noisy"]["skew"]))
    right.loglog(t * surface.TRADING_DAYS, noisy_skew, "x", color=SERIES[1], ms=5, label="from the noisy quotes")
    right.set_xticks(days, [TENORS.get(d, str(d)) for d in days])
    right.minorticks_off()
    low, high = right.get_ylim()
    ticks = np.unique(np.round(np.geomspace(low * 1.05, high / 1.05, 5), 2))
    right.set_yticks(ticks, [f"{v:.2f}" for v in ticks])
    right.set_xlabel("maturity")
    right.set_ylabel("|ATM skew|, |∂σ/∂ ln K| at the money")
    right.legend(fontsize=7.5, loc="lower left")
    right.set_title("the skew's term structure, log–log", loc="left", fontsize=8.5)
    for ax in (left, right):
        ax.grid(color=BASELINE, linewidth=0.4, alpha=0.6)
    fig.savefig(path, dpi=160)


def _bank_statistics(task, prices):
    vols = implied(task, prices)
    rows = []
    for r in range(len(prices)):
        s = statistics(task, vols[r])
        rows.append([s["one_month"]["atm"], s["one_month"]["skew"], s["skew_exponent"], s["one_month"]["curvature"]])
    return np.array(rows, dtype=float), int(np.sum(~np.isfinite(vols)))


def _quantiles(values, names):
    summary = {}
    for j, name in enumerate(names):
        finite = values[np.isfinite(values[:, j]), j]
        summary[name] = {"finite": int(len(finite)),
                         "quantiles": dict(zip(("q01", "q05", "q25", "q50", "q75", "q95", "q99"),
                                               np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist()))}
    return summary


def prior_predictive(args):
    manifest, splits = bank.load_bank(args.bank)
    task = manifest["configuration"]["task"]
    split = splits[args.split]
    rows = np.arange(min(args.surfaces, len(split["parameters"])))
    started = time.process_time()
    clean, clean_missing = _bank_statistics(task, split["reference_prices"][rows])
    noisy, noisy_missing = _bank_statistics(task, split["observations"][rows, :, 0])
    names = ["atm_vol_1m", "atm_skew_1m", "skew_exponent", "atm_curvature_1m"]
    report = {"study": "smile_facts:prior_predictive", "task_id": task["task_id"],
              "task_fingerprint": manifest["task_fingerprint"], "split": args.split, "surfaces": len(rows),
              "statistics": names,
              "clean": {"summary": _quantiles(clean, names), "quotes_without_implied_vol": clean_missing,
                        "values": clean.tolist()},
              "noisy": {"summary": _quantiles(noisy, names), "quotes_without_implied_vol": noisy_missing,
                        "values": noisy.tolist()},
              "note": ("clean: the model's smiles; noisy: the quotes the network sees. A real design grid is read "
                       "off a smoothing spline, so it compares with the clean distribution."),
              "cpu_seconds": time.process_time() - started, "real_smile": "pending study S8"}
    if task["model"] == "rough":
        hurst = np.asarray(split["parameters"])[rows, task["parameter_names"].index("hurst")]
        gap = clean[:, 2] - (hurst - 0.5)
        finite = np.isfinite(gap)
        report["skew_exponent_minus_h_minus_half"] = {
            "from": "clean prices", "finite": int(finite.sum()),
            "quantiles": dict(zip(("q05", "q25", "q50", "q75", "q95"),
                                  np.quantile(gap[finite], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist())),
            "share_negative": float(np.mean(gap[finite] < 0)),
            "correlation_exponent_with_hurst": float(np.corrcoef(clean[finite, 2], hurst[finite])[0, 1])}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "prior-predictive.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    prior_predictive_figure(task, clean, noisy, names, args.out / "prior-predictive.png")
    print(json.dumps(jsonable({"clean": report["clean"]["summary"], "noisy": report["noisy"]["summary"]}), indent=1))


def prior_predictive_figure(task, clean, noisy, names, path):
    from sbi4finance.calibration.plots import INK_2, SERIES, plt, style

    style()
    titles = {"atm_vol_1m": "1-month ATM implied volatility (%)", "atm_skew_1m": "1-month ATM skew",
              "skew_exponent": "exponent of the skew's term structure", "atm_curvature_1m": "1-month ATM curvature"}
    scale = {"atm_vol_1m": 100.0}
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.1), constrained_layout=True)
    for j, (ax, name) in enumerate(zip(axes, names)):
        a = clean[np.isfinite(clean[:, j]), j] * scale.get(name, 1.0)
        b = noisy[np.isfinite(noisy[:, j]), j] * scale.get(name, 1.0)
        lo, hi = np.quantile(np.concatenate([a, b]), [0.01, 0.99])
        bins = np.linspace(lo, hi, 41)
        ax.hist(a[(a >= lo) & (a <= hi)], bins=bins, color=SERIES[0], alpha=0.75, label="model smiles (clean)")
        ax.hist(b[(b >= lo) & (b <= hi)], bins=bins, histtype="step", color=SERIES[1], linewidth=1.1,
                label="noisy quotes")
        ax.axvline(np.median(a), color=SERIES[0], linestyle="--", linewidth=1)
        if name == "skew_exponent":
            line = ax.axvline(SPX_EXPONENT, color=INK_2, linestyle=":", linewidth=1.2)
            ax.legend([line], ["SPX, about −0.4 (BFG 2016)"], fontsize=7, loc="upper left")
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(titles[name], loc="left", fontsize=8.5)
        ax.set_yticks([])
    model = "rough Bergomi" if task["model"] == "rough" else "Heston"
    fig.suptitle(f"Prior predictive check, {model}: {len(clean)} held-out surfaces "
                 "(central 98% shown; dashed: median of the clean smiles)", x=0.01, ha="left", fontsize=9)
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("example")
    one.add_argument("--model", choices=("rough", "heston"), required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--seed", type=int, default=0)
    many = sub.add_parser("prior-predictive")
    many.add_argument("--bank", type=Path, required=True)
    many.add_argument("--out", type=Path, required=True)
    many.add_argument("--split", default="test")
    many.add_argument("--surfaces", type=int, default=1000)
    args = parser.parse_args()
    (example if args.command == "example" else prior_predictive)(args)


if __name__ == "__main__":
    main()
