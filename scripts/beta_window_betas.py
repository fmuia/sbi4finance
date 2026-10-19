"""Window betas for slot 1's real-data check: least-squares betas in consecutive, non-overlapping windows.

    pixi run beta-window-betas --prices weekly_prices.csv --market SPY --out docs/results/intro-betas/window_betas.csv

The input is a CSV of weekly adjusted closes with a `date` column and one column per symbol,
including the market's. The output has one row per stock and window: the window's length in weeks,
its number (consecutive numbers are adjacent windows), its first and last week, the stock, its beta
against the market and the beta's standard error. The windows end at the latest week. Slot 1 tests
on these rows which estimator best predicts the next window's beta (`beta_shrinkage.blume_test`).

The committed table was built from Alpha Vantage weekly adjusted closes for 20 large US stocks and
SPY, November 1999 to October 2026, in windows of 52 and 104 weeks. The prices themselves are not
in the repository.
"""

import argparse
from pathlib import Path

import pandas as pd

from sbi4finance import beta_shrinkage as bs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--market", default="SPY")
    parser.add_argument("--weeks", type=int, nargs="+", default=[52, 104])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prices = pd.read_csv(args.prices, parse_dates=["date"], index_col="date").sort_index()
    frames = []
    for weeks in args.weeks:
        rows = bs.window_betas(prices, args.market, weeks)
        if not rows:
            parser.error(f"no {weeks}-week window has three stocks and the market complete")
        frames.append(pd.DataFrame(rows).assign(window_weeks=weeks))
    table = pd.concat(frames, ignore_index=True)
    table["window_start"] = table["window_start"].dt.date
    table["window_end"] = table["window_end"].dt.date
    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["window_weeks", "window", "window_start", "window_end", "stock", "beta", "se"]
    table[columns].to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(table)} rows, {table.stock.nunique()} stocks -> {args.out}")


if __name__ == "__main__":
    main()
