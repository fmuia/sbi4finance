"""Rolling amortized posteriors on real returns, with the misspecification test (feasibility result 7).

    pixi run svj-rolling --out /tmp/svj-rolling-new  # VOO and SPMO, one window per week
    pixi run svj-rolling --tickers VOO --step 1 --out /tmp/svj-rolling-daily  # every trading day

Every window holds the last 1,000 daily log returns; the network gives the joint posterior in about a
second per window, and the misspecification test compares its learned summaries with simulated ones.
Spot volatility is compared with the realized volatility of the last 20 days.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbi4finance.training_workflow import reserve_output_dir
from sbi4finance.svj import amortized, checks, data, model, plots

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=str(ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv"))
    parser.add_argument("--tickers", nargs="+", default=["VOO", "SPMO"])
    parser.add_argument("--kind", choices=("simple", "log"), default="simple")
    parser.add_argument("--step", type=int, default=5, help="trading days between window ends")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--out", required=True, help="new output directory; historical studies are never overwritten")
    args = parser.parse_args()
    out = reserve_output_dir(args.out)

    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    test = checks.MisspecificationTest.load(Path(args.checkpoint) / "misspecification.npz")
    test.assert_compatible(posterior)
    summary = {}
    for ticker in args.tickers:
        returns = data.load_returns(args.csv, ticker, args.kind)
        ends = data.rolling_ends(returns, args.step)
        windows = np.stack([data.window(returns, end).to_numpy() for end in ends])
        draws = model.quantities(posterior.sample(windows, num_samples=1_000, seed=0))
        pvalues = test.pvalue(posterior.summaries(windows))
        q = lambda j, p: np.quantile(draws[:, :, j], p, axis=1)
        realized = returns.rolling(20).std().loc[ends].to_numpy() * np.sqrt(model.DAYS_PER_YEAR) / 100
        frame = pd.DataFrame(
            {
                "end": ends.date,
                "spot_vol_median": q(8, 0.5), "spot_vol_q05": q(8, 0.05), "spot_vol_q95": q(8, 0.95),
                "realized_vol_20d": realized,
                "jumps_per_year_median": q(5, 0.5), "jumps_per_year_q05": q(5, 0.05), "jumps_per_year_q95": q(5, 0.95),
                "mean_jump_median": q(6, 0.5),
                "leverage_median": q(4, 0.5),
                "long_run_vol_median": q(1, 0.5),
                "misspecification_pvalue": pvalues,
            }
        )
        frame.to_csv(out / f"{ticker}_rolling.csv", index=False, float_format="%.5g")
        dates = pd.to_datetime(frame["end"])
        flagged = pvalues < 0.05
        plots.rolling_panels(
            dates,
            [
                {"ylabel": "spot volatility", "median": frame.spot_vol_median, "lo": frame.spot_vol_q05,
                 "hi": frame.spot_vol_q95, "extra": [("realized vol, last 20 days", frame.realized_vol_20d, plots.SERIES[1])]},
                {"ylabel": "jumps per year", "median": frame.jumps_per_year_median, "lo": frame.jumps_per_year_q05,
                 "hi": frame.jumps_per_year_q95},
                {"ylabel": "misspecification p-value", "pvalues": pvalues},
            ],
            f"{ticker}: amortized posterior on rolling 1,000-day windows (one every {args.step} trading days)",
        ).savefig(out / f"{ticker}_rolling.png", dpi=150)
        summary[ticker] = {
            "scientific_acceptance": False,
            "inference_status": "network_only; summary-space screening is not a conditional posterior audit",
            "n_windows": len(ends),
            "first_end": str(ends[0].date()),
            "last_end": str(ends[-1].date()),
            "flagged_at_5pct": int(flagged.sum()),
            "flagged_ends": [str(d.date()) for d in ends[flagged]],
            "corr_spot_vol_vs_realized_20d": float(np.corrcoef(frame.spot_vol_median, realized)[0, 1]),
            "jumps_per_year_median_range": [float(frame.jumps_per_year_median.min()), float(frame.jumps_per_year_median.max())],
            "leverage_median_range": [float(frame.leverage_median.min()), float(frame.leverage_median.max())],
        }
        print(f"{ticker}: {len(ends)} windows {ends[0].date()} to {ends[-1].date()}; flagged {int(flagged.sum())}; "
              f"corr(spot vol, 20-day realized) {summary[ticker]['corr_spot_vol_vs_realized_20d']:.2f}")
    (out / "rolling.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
