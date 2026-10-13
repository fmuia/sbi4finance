"""VaR and ES backtests with a known truth on held-out simulated series (feasibility result 6).

    pixi run svj-backtest --out /tmp/svj-backtest-new  # 500 series
    pixi run svj-backtest --n-series 100 --out /tmp/svj-backtest
    pixi run svj-backtest --from-saved docs/results/svj-backtest --out /tmp/svj-backtest-replay

On a simulated series the true parameters and today's true variance are known, so the probability that
the next H-day loss exceeds a reported 99% VaR is computed from 50,000 simulated futures rather than
counted on one path. The expected Acerbi-Szekely Z2 score of the 97.5% ES uses the same futures; it is
0 for a correct ES and negative when the ES understates the loss beyond the VaR. Three methods report:

- predictive: 20 forward paths from each of 1,000 network posterior draws;
- plug-in: parameters fixed at the posterior median, today's variance from a 2,000-particle filter there;
- check: the true parameters with a filtered state, which tests the test itself.

VaR/ES retain the historical negative cumulative percent-log-return definition.
Saved replay reads its source and writes only a separate new output directory.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

from sbi4finance.training_workflow import reserve_output_dir
from sbi4finance.svj import amortized, artifacts, model, plots, risk

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 10)
METHODS = ("check: true parameters, filtered state", "predictive (amortized posterior)", "plug-in at posterior median")
COLORS = dict(zip(METHODS, (plots.INK_2, plots.SERIES[0], plots.SERIES[1])))


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
    parser.add_argument("--out", required=True, help="new output directory; existing studies are never overwritten")
    parser.add_argument("--from-saved", type=Path, help="read-only input NPZ or directory for summary/figure replay")
    args = parser.parse_args()
    out = reserve_output_dir(args.out)
    if args.from_saved:
        source = args.from_saved / "backtest_series.npz" if args.from_saved.is_dir() else args.from_saved
        with np.load(source, allow_pickle=False) as saved:
            if "risk_definition" in saved and str(saved["risk_definition"]) != risk.RISK_DEFINITION:
                raise ValueError("saved backtest uses a different risk definition")
            n_futures = int(saved["n_futures"]) if "n_futures" in saved else args.n_futures
            write_report(saved["p_exc"], saved["z2"], saved["es_low"], n_futures, out,
                         provenance={"source": str(source), "source_sha256": artifacts.sha256(source),
                                     "mode": "saved_replay", "n_futures_from_archive": "n_futures" in saved})
        return

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

    np.savez_compressed(out / "backtest_series.npz", p_exc=p_exc, z2=z2, es_low=es_low,
                        methods=np.array(METHODS), horizons=np.array(HORIZONS), n_futures=args.n_futures,
                        risk_definition=risk.RISK_DEFINITION, risk_unit=risk.RISK_UNIT, seed=args.seed)
    write_report(p_exc, z2, es_low, args.n_futures, out, provenance={"mode": "new_simulation", "seed": args.seed})


def write_report(p_exc, z2, es_low, n_futures, out, *, provenance=None):
    """Summary JSON, figures and a printed table from per-series exceedance, Z2 and ES-shortfall arrays."""
    n_series = len(p_exc)

    def mean_ci(x):
        se = x.std(ddof=1) / np.sqrt(len(x))
        return float(x.mean()), (float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se))

    summary = {"n_series": n_series, "n_futures": n_futures, "horizons": {},
               "risk_definition": risk.RISK_DEFINITION, "risk_unit": risk.RISK_UNIT,
               "scientific_acceptance": False, "provenance": provenance}
    levels_exc, levels_z2, paired_exc, paired_z2 = {}, {}, {}, {}
    for k, h in enumerate(HORIZONS):
        row = f"{h}-day"
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
            levels_exc[(row, method)] = (100 * rate, 100 * rate_ci[0], 100 * rate_ci[1])
            levels_z2[(row, method)] = (score, *score_ci)
            if m > 0:  # paired against the check on the same series
                d_rate, d_rate_ci = mean_ci(p_exc[:, m, k] - p_exc[:, 0, k])
                d_score, d_score_ci = mean_ci(z2[:, m, k] - z2[:, 0, k])
                block[method]["minus_check"] = {"exceedance_pp": 100 * d_rate, "exceedance_pp_ci95": [100 * v for v in d_rate_ci],
                                                "z2": d_score, "z2_ci95": list(d_score_ci)}
                paired_exc[(row, method)] = (100 * d_rate, 100 * d_rate_ci[0], 100 * d_rate_ci[1])
                paired_z2[(row, method)] = (d_score, *d_score_ci)
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
    rows = [f"{h}-day" for h in HORIZONS]
    plots.method_panels(
        [("expected exceedance of the 99% VaR (%)", 1.0, levels_exc), ("expected Z2 of the 97.5% ES", 0.0, levels_z2)],
        rows, list(METHODS), f"Percent-log-loss VaR/ES on {n_series} held-out series: levels, 95% intervals", colors=COLORS,
    ).savefig(out / "backtest_levels.png", dpi=150)
    plots.method_panels(
        [("99% VaR exceedance minus the check (pp)", 0.0, paired_exc), ("97.5% ES Z2 minus the check", 0.0, paired_z2)],
        rows, list(METHODS[1:]), "Paired against the true-parameter check on the same series, 95% intervals", colors=COLORS,
    ).savefig(out / "backtest_paired.png", dpi=150)

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
