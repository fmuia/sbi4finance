"""The classical baseline against the posterior: Wald coverage on held-out series (feasibility result 5).

    pixi run svj-classical                        # 40 series, two starts each; a few hours on 4 CPU cores
    pixi run svj-classical --n-series 8 --out /tmp/svj-classical
    pixi run svj-classical --shard 0 --shards 2 & pixi run svj-classical --shard 1 --shards 2; pixi run svj-classical --merge

For each series, approximate ML maximizes the grid-filter likelihood with L-BFGS-B from the prior mean and
from the amortized posterior median, inside bounds at the prior mean +/- 6 prior sd, and builds 95% Wald
intervals in the unbounded coordinates. The same series get 95% posterior intervals from the network.
The script also scores the plug-in forecast at the MLE (today's variance filtered at the estimate) with the
known-truth backtest of `svj_backtest.py`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sbi4finance.svj import amortized, classical, model, plots, risk

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--n-series", type=int, default=40)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results" / "svj-classical"))
    parser.add_argument("--shard", type=int, default=0, help="process series i with i %% shards == shard")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="only summarize the shard files already written")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.merge:
        fit_series(args, out)
    if args.merge or args.shards == 1:
        summarize(args, out)


def fit_series(args, out):
    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    sims = model.simulate(args.n_series, seed=args.seed)
    draws = posterior.sample(sims["returns"], num_samples=1_000, seed=args.seed + 1)
    truths = sims["targets"][:, : model.N_PARAMS]
    shard_file = out / f"classical_series_{args.shard}_of_{args.shards}.json"
    records = json.loads(shard_file.read_text()) if shard_file.exists() else []  # resume an interrupted shard
    done = {r["series"] for r in records}
    for i in range(args.shard, args.n_series, args.shards):
        if i in done:
            continue
        y, truth = sims["returns"][i], truths[i]
        starts = [classical.prior_mean(), np.median(draws[i, :, : model.N_PARAMS], axis=0)]
        fit = classical.fit_mle(y, starts, n_cells=args.n_cells)
        lo, hi = np.quantile(draws[i, :, : model.N_PARAMS], [0.025, 0.975], axis=0)
        # plug-in at the MLE, scored against 50,000 futures from the true parameters and today's true variance
        plugin = risk.point_forecast_draws(fit.coords, y, seed=i)
        futures = {h: -r for h, r in risk.horizon_returns(sims["targets"][i][None], (1, 10), 50_000, seed=i + 1).items()}
        scores = {}
        for h, r in risk.horizon_returns(plugin, (1, 10), 1, seed=i + 2).items():
            var99, _ = risk.var_es(r)
            var975, es975 = risk.var_es(r, var_level=0.975)
            loss = futures[h]
            scores[str(h)] = {
                "exceedance_99": float(np.mean(loss > var99)),
                "z2_975": float(1 - np.mean(loss * (loss > var975)) / (0.025 * es975)),
            }
        records.append({
            "series": i,
            "mle": fit.coords.tolist(),
            "se": fit.se.tolist(),
            "wald_covers": ((fit.lower <= truth) & (truth <= fit.upper)).tolist(),
            "wald_width": (fit.upper - fit.lower).tolist(),
            "posterior_covers": ((lo <= truth) & (truth <= hi)).tolist(),
            "posterior_width": (hi - lo).tolist(),
            "on_bound": fit.on_bound.tolist(),
            "hessian_positive_definite": fit.hessian_positive_definite,
            "loglik": fit.loglik,
            "cpu_s": fit.cpu_s,
            "truth": truth.tolist(),
            "plugin_at_mle": scores,
        })
        print(f"series {i + 1}/{args.n_series}: {fit.cpu_s:.0f} CPU-s, Hessian PD {fit.hessian_positive_definite}, "
              f"on bound {[model.QUANTITY_LABELS[j] for j in np.flatnonzero(fit.on_bound)]}", flush=True)
        shard_file.write_text(json.dumps(records, indent=1))


def scores(forecast_targets, n_paths, losses, seed):
    """Exceedance of the 99% VaR and Z2 of the 97.5% ES of a forecast, against the true future losses."""
    out = {}
    for h, r in risk.horizon_returns(forecast_targets, (1, 10), n_paths, seed).items():
        var99, _ = risk.var_es(r)
        var975, es975 = risk.var_es(r, var_level=0.975)
        loss = losses[h]
        out[str(h)] = {
            "exceedance_99": float(np.mean(loss > var99)),
            "z2_975": float(1 - np.mean(loss * (loss > var975)) / (0.025 * es975)),
        }
    return out


def paired_references(args, records):
    """Score the true-parameter check and the predictive forecast on the same series and futures as the MLE plug-in."""
    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    sims = model.simulate(args.n_series, seed=args.seed)
    draws = posterior.sample(sims["returns"], num_samples=1_000, seed=args.seed + 1)
    for r in records:
        i = r["series"]
        y, truth = sims["returns"][i], sims["targets"][i]
        losses = {h: -v for h, v in risk.horizon_returns(truth[None], (1, 10), 50_000, seed=i + 1).items()}
        r["check"] = scores(risk.point_forecast_draws(truth, y, seed=i + 3), 1, losses, i + 4)
        r["predictive"] = scores(draws[i], 20, losses, i + 5)


def summarize(args, out):
    records = sorted(
        (r for f in sorted(out.glob("classical_series_*_of_*.json")) for r in json.loads(f.read_text())),
        key=lambda r: r["series"],
    )
    paired_references(args, records)
    wald = np.array([r["wald_covers"] for r in records])
    post = np.array([r["posterior_covers"] for r in records])
    bound = np.array([r["on_bound"] for r in records])
    pd_ok = np.array([r["hessian_positive_definite"] for r in records])
    jsd_width = np.exp(np.array([r["wald_width"] for r in records])[:, 7])  # ratio upper / lower of the jump-size sd
    summary = {
        "n_series": len(records),
        "median_cpu_s_per_fit": float(np.median([r["cpu_s"] for r in records])),
        "fits_with_a_parameter_on_a_bound": int(bound.any(axis=1).sum()),
        "bound_hits_by_parameter": dict(zip(model.QUANTITY_LABELS, bound.sum(0).tolist())),
        "hessians_not_positive_definite": int((~pd_ok).sum()),
        "wald_coverage_95": dict(zip(model.QUANTITY_LABELS, wald.mean(0).round(3).tolist())),
        "posterior_coverage_95": dict(zip(model.QUANTITY_LABELS, post.mean(0).round(3).tolist())),
        "jump_sd_wald_intervals_spanning_factor_100": int(np.sum(jsd_width[np.isfinite(jsd_width)] > 100)),
        "forecasts_on_these_series": {
            method: {
                h: {
                    "expected_exceedance_99": float(np.mean([r[method][h]["exceedance_99"] for r in records])),
                    "z2_975": float(np.mean([r[method][h]["z2_975"] for r in records])),
                }
                for h in ("1", "10")
            }
            for method in ("check", "predictive", "plugin_at_mle")
        },
        "plugin_at_mle_minus_check": {
            h: {
                key: {
                    "mean": float(np.mean(d := np.array([r["plugin_at_mle"][h][key] - r["check"][h][key] for r in records]))),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                }
                for key in ("exceedance_99", "z2_975")
            }
            for h in ("1", "10")
        },
        "plugin_at_mle_minus_predictive": {
            h: {
                key: {
                    "mean": float(np.mean(d := np.array([r["plugin_at_mle"][h][key] - r["predictive"][h][key] for r in records]))),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                }
                for key in ("exceedance_99", "z2_975")
            }
            for h in ("1", "10")
        },
        "predictive_minus_check": {
            h: {
                key: {
                    "mean": float(np.mean(d := np.array([r["predictive"][h][key] - r["check"][h][key] for r in records]))),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                }
                for key in ("exceedance_99", "z2_975")
            }
            for h in ("1", "10")
        },
    }
    (out / "classical.json").write_text(json.dumps(summary, indent=1))
    n = len(records)
    def wilson(p, z=1.96):
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return centre - half, centre + half

    labels = list(model.QUANTITY_LABELS[: model.N_PARAMS])
    methods = ("Wald (approximate ML)", "posterior (amortized)")
    values = {}
    for j, label in enumerate(labels):
        for method, cover in zip(methods, (wald[:, j].mean(), post[:, j].mean())):
            lo, hi = wilson(cover)
            values[(label, method)] = (100 * cover, 100 * lo, 100 * hi)
    plots.method_panels(
        [("coverage of 95% intervals (%), with 95% Wilson intervals", 95.0, values)],
        labels, list(methods), f"Wald against posterior intervals on {n} held-out series", height_per_row=0.55,
        colors={methods[0]: plots.SERIES[1], methods[1]: plots.SERIES[0]},
    ).savefig(out / "coverage.png", dpi=150)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
