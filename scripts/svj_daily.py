"""Run the SVJ daily process on one window of real returns and write a JSON report.

    pixi run svj-daily --ticker VOO                                   # last 1,000 days of the bundled CSV
    pixi run svj-daily --ticker SPMO --end 2025-04-30 --pmmh-on-flag  # a past window; flagged windows go to particle MCMC
    pixi run svj-daily --ticker VOO --no-audit                        # network posterior only, about a second

Steps: network posterior -> misspecification test -> importance-sampling audit (-> particle MCMC) -> VaR / ES.
"""

import argparse
import json
from pathlib import Path

from sbi4finance.svj import amortized, checks, data, model, pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=str(ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv"))
    parser.add_argument("--ticker", default="VOO")
    parser.add_argument("--kind", choices=("simple", "log"), default="simple", help="return type stored in the CSV")
    parser.add_argument("--end", help="last day of the window (default: last day in the file)")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--threshold", type=float, default=pipeline.DEFAULT_THRESHOLD, help="audit efficiency below which a window is flagged")
    parser.add_argument("--pmmh-on-flag", action="store_true", help="run particle MCMC when the audit flags the window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="JSON report path (default: results/svj/<ticker>_<end>.json)")
    args = parser.parse_args()

    returns = data.window(data.load_returns(args.csv, args.ticker, args.kind), args.end)
    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    reference = Path(args.checkpoint) / "misspecification.npz"
    misspecification = checks.MisspecificationTest.load(reference) if reference.exists() else None
    config = pipeline.PipelineConfig(
        run_audit=not args.no_audit, audit_threshold=args.threshold, pmmh_on_flag=args.pmmh_on_flag, seed=args.seed
    )
    report, _ = pipeline.run_window(returns.to_numpy(), posterior, misspecification, config)
    report["window"] = {
        "ticker": args.ticker,
        "first_day": str(returns.index[0].date()),
        "last_day": str(returns.index[-1].date()),
        "n_days": len(returns),
        "annualized_vol": float(returns.std() * model.DAYS_PER_YEAR**0.5 / 100),
    }

    out = Path(args.out) if args.out else ROOT / "results" / "svj" / f"{args.ticker}_{report['window']['last_day']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    w, risk = report["window"], report["risk"]
    print(f"{w['ticker']} {w['first_day']} to {w['last_day']}, realized vol {w['annualized_vol']:.1%}; route: {report['route']}")
    if "misspecification_pvalue" in report:
        print(f"  misspecification p-value {report['misspecification_pvalue']:.3f}")
    if "audit" in report:
        a = report["audit"]
        print(f"  audit efficiency {a['efficiency']:.1%} ({a['ess']:.0f} effective draws, {a['n_particles']} particles)"
              + ("  FLAGGED" if a["flagged"] else ""))
    spot = report["posterior"]["spot vol"]
    print(f"  spot vol {spot['median']:.1%} [{spot['q05']:.1%}, {spot['q95']:.1%}]")
    for h in sorted(risk["predictive"], key=int):
        p, g = risk["predictive"][h], risk["plugin_at_posterior_median"][h]
        print(f"  {h:>2}-day VaR99 {p['var']:.2f}% ES97.5 {p['es']:.2f}%   (plug-in {g['var']:.2f}% / {g['es']:.2f}%)")
    c = risk["conditional_var_1d"]
    print(f"  1-day conditional VaR 90% interval {c['interval_90'][0]:.2f}% to {c['interval_90'][1]:.2f}%"
          + (f"; today's variance carries {c['state_share']:.0%} of its variance" if "state_share" in c else ""))
    print(f"  report: {out}")


if __name__ == "__main__":
    main()
