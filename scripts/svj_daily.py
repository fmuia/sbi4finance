"""Run the shared inference workflow on a prepared SVJ return-window artifact.

    pixi run svj-daily --artifact data/synthetic/svj-workflow/window --out /tmp/svj-case.json
    pixi run svj-daily --artifact /path/to/market-window --pmmh-on-flag --out /tmp/svj-market.json
    pixi run svj-daily --legacy-csv --ticker VOO --out /tmp/svj-historical-replay.json

Steps: network posterior -> misspecification test -> importance-sampling audit (-> particle MCMC) -> VaR / ES.
"""

import argparse
import json
from pathlib import Path

from sbi4finance.datasets.local import load_artifact
from sbi4finance.svj import adapters, amortized, artifacts, checks, data, model, pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--artifact", type=Path, help="prepared complete P return-window artifact directory")
    inputs.add_argument("--legacy-csv", action="store_true", help="explicit historical permissive CSV reproduction path")
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

    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    reference = Path(args.checkpoint) / "misspecification.npz"
    if reference.exists():
        artifacts.validate_reference(reference, checkpoint=args.checkpoint)
    misspecification = checks.MisspecificationTest.load(reference) if reference.exists() else None
    config = pipeline.PipelineConfig(
        run_audit=not args.no_audit, audit_threshold=args.threshold, pmmh_on_flag=args.pmmh_on_flag, seed=args.seed
    )
    if args.legacy_csv:
        returns = data.window(data.load_returns(args.csv, args.ticker, args.kind), args.end)
        report, _ = pipeline.run_window(returns.to_numpy(), posterior, misspecification, config,
                                        provenance={"input_mode": "historical_permissive_csv", "evidence_scope": "M"})
        first, last, label, values = str(returns.index[0].date()), str(returns.index[-1].date()), args.ticker, returns.to_numpy()
    else:
        artifact = load_artifact(args.artifact or ROOT / "data/synthetic/svj-workflow/window")
        values = adapters.prepared_returns(artifact, checkpoint_task=posterior.task)
        report, _ = pipeline.run_prepared_window(artifact, posterior, misspecification, config)
        conventions = artifact.metadata["conventions"]
        first, last = conventions["window_start"], conventions["conditioning_cutoff"]
        label = artifact.metadata["artifact_id"]
    report["window"] = {"ticker": label, "first_day": first, "last_day": last,
                        "n_days": len(values), "annualized_vol": float(values.std() * model.DAYS_PER_YEAR**0.5 / 100)}

    out = Path(args.out) if args.out else ROOT / "results" / "svj" / f"{args.ticker}_{report['window']['last_day']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        handle.write(json.dumps(report, indent=1, allow_nan=False) + "\n")

    w, risk = report["window"], report["risk"]
    print(f"{w['ticker']} {w['first_day']} to {w['last_day']}, realized vol {w['annualized_vol']:.1%}; route: {report['route']}")
    print(f"  local reliability: {report['status']}; scientific acceptance: {report['scientific_acceptance']}; units: percent log-loss")
    if "misspecification_pvalue" in report:
        print(f"  misspecification p-value {report['misspecification_pvalue']}")
    if "audit" in report:
        a = report["audit"]
        print(f"  audit: {a}")
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
