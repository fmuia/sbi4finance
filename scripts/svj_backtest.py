"""VaR and ES backtests with a known truth on held-out simulated series (feasibility result 6).

    pixi run svj-backtest                       # 500 series
    pixi run svj-backtest --n-series 100 --out /tmp/svj-backtest

On a simulated series the true parameters and today's true variance are known, so the probability that
the next H-day loss exceeds a reported 99% VaR is computed from 50,000 simulated futures rather than
counted on one path. The expected Acerbi-Szekely Z2 score of the 97.5% ES uses the same futures; it is
0 for a correct ES and negative when the ES understates the loss beyond the VaR. Three methods report:

- predictive: 20 forward paths from each of 1,000 network posterior draws;
- plug-in: parameters fixed at the posterior median, today's variance from a 2,000-particle filter there;
- check: the true parameters with a filtered state, which tests the test itself.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

from sbi4finance.svj import amortized, model, plots, risk

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 10)
METHODS = ("check: true parameters, filtered state", "predictive (amortized posterior)", "plug-in at posterior median")


def risk_report(targets, n_paths, seed):
    """{H: (VaR99, VaR97.5, ES97.5)} of the distribution simulated from targets."""
    pooled = risk.horizon_returns(targets, HORIZONS, n_paths, seed)
    out = {}
    for h, r in pooled.items():
        var99, _ = risk.var_es(r)
        var975, es975 = risk.var_es(r, var_level=0.975)
        out[h] = (var99, var975, es975)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--n-series", type=int, default=500)
    parser.add_argument("--n-futures", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results" / "svj-backtest"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    sims = model.simulate(args.n_series, seed=args.seed)
    start = time.time()
    draws = posterior.sample(sims["returns"], num_samples=1_000, seed=args.seed + 1)
    p_exc = np.zeros((args.n_series, len(METHODS), len(HORIZONS)))
    z2 = np.zeros_like(p_exc)
    es_low = np.zeros_like(p_exc, dtype=bool)
    for i in range(args.n_series):
        y, truth = sims["returns"][i], sims["targets"][i]
        seed = args.seed + 10 * i
        reports = (
            risk_report(risk.point_forecast_draws(truth, y, seed), 1, seed + 1),
            risk_report(draws[i], 20, seed + 2),
            risk_report(risk.plugin_draws(draws[i], y, seed + 3), 1, seed + 4),
        )
        futures = risk.horizon_returns(truth[None], HORIZONS, args.n_futures, seed + 5)
        for k, h in enumerate(HORIZONS):
            loss = -futures[h]
            true_var975 = np.quantile(loss, 0.975)
            true_es = loss[loss >= true_var975].mean()
            for m, report in enumerate(reports):
                var99, var975, es975 = report[h]
                p_exc[i, m, k] = np.mean(loss > var99)
                z2[i, m, k] = 1 - np.mean(loss * (loss > var975)) / (0.025 * es975)
                es_low[i, m, k] = es975 < 0.9 * true_es
        if (i + 1) % 50 == 0:
            print(f"{i + 1} series, {time.time() - start:.0f} s")

    def mean_ci(x):
        return float(x.mean()), (float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))), float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x))))

    summary = {"n_series": args.n_series, "n_futures": args.n_futures, "horizons": {}}
    exc_rows, z2_rows = [], []
    for k, h in enumerate(HORIZONS):
        block = {}
        for m, method in enumerate(METHODS):
            rate, rate_ci = mean_ci(p_exc[:, m, k])
            score, score_ci = mean_ci(z2[:, m, k])
            block[method] = {
                "expected_exceedance_99": rate,
                "expected_exceedance_99_ci95": rate_ci,
                "expected_exceptions_in_250_days": 250 * rate,
                "prob_leave_green_zone_in_250_days": float(binom.sf(4, 250, p_exc[:, m, k]).mean()),
                "z2_975": score,
                "z2_975_ci95": score_ci,
                "share_es_more_than_10pct_low": float(es_low[:, m, k].mean()),
            }
            exc_rows.append((f"{h}-day · {method}", 100 * rate, tuple(100 * np.array(rate_ci))))
            z2_rows.append((f"{h}-day · {method}", score, score_ci))
        diff = p_exc[:, 2, k] - p_exc[:, 1, k]
        dz = z2[:, 2, k] - z2[:, 1, k]
        block["plugin_minus_predictive"] = {
            "exceedance_pp": float(100 * diff.mean()),
            "exceedance_pp_se": float(100 * diff.std(ddof=1) / np.sqrt(len(diff))),
            "z2": float(dz.mean()),
            "z2_se": float(dz.std(ddof=1) / np.sqrt(len(dz))),
        }
        summary["horizons"][str(h)] = block
    (out / "backtest.json").write_text(json.dumps(summary, indent=1))
    plots.interval_dots(exc_rows, 1.0, "expected exceedance rate of the 99% VaR (%), 95% interval",
                        f"VaR backtest on {args.n_series} held-out series (nominal 1%)").savefig(out / "var_exceedance.png", dpi=150)
    plots.interval_dots(z2_rows, 0.0, "expected Z2 score of the 97.5% ES (0 = correct, negative = too low)",
                        f"ES backtest on {args.n_series} held-out series").savefig(out / "es_z2.png", dpi=150)

    for k, h in enumerate(HORIZONS):
        print(f"{h}-day horizon:")
        for method in METHODS:
            b = summary["horizons"][str(h)][method]
            print(f"  {method:<42} exceedance {100 * b['expected_exceedance_99']:.2f}%  Z2 {b['z2_975']:+.3f}  "
                  f"ES >10% low in {b['share_es_more_than_10pct_low']:.0%}")
        d = summary["horizons"][str(h)]["plugin_minus_predictive"]
        print(f"  plug-in minus predictive: exceedance {d['exceedance_pp']:+.3f} pp (se {d['exceedance_pp_se']:.3f}), Z2 {d['z2']:+.3f} (se {d['z2_se']:.3f})")


if __name__ == "__main__":
    main()
