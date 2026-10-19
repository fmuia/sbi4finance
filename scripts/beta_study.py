"""The numbers behind slot 1: betas across many stocks, simulated universes and real windows.

    pixi run beta-study --out docs/results/intro-betas/summary.json  # a few seconds

For each data frequency in `beta_shrinkage.SETTINGS`, 200 simulated universes of 500 stocks compare
least squares, Bloomberg's rule and Bayes' rule (with the prior learned from the cross-section, and
with the true population). Two smaller studies repeat it for a low-beta universe and for five
stocks. The real-data part runs Blume's test on `docs/results/intro-betas/window_betas.csv`
(see `scripts/beta_window_betas.py`).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from sbi4finance import beta_shrinkage as bs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-betas", type=Path, default=ROOT / "docs" / "results" / "intro-betas" / "window_betas.csv")
    args = parser.parse_args()
    summary = {
        "population": bs.POPULATION,
        "settings": {key: vars(setting) for key, setting in bs.SETTINGS.items()},
        "simulated": {key: bs.study(setting, n_universes=200, seed=1) for key, setting in bs.SETTINGS.items()},
        "utilities": bs.study(bs.SETTINGS["weekly"], n_universes=200, n_stocks=60, top=10, seed=5,
                              population={"beta_mean": 0.6, "beta_sd": 0.2, "idio_dispersion": 0.35}),
        "five_stocks": bs.study(bs.SETTINGS["weekly"], n_universes=400, n_stocks=5, top=1, seed=5),
    }
    if args.window_betas.exists():
        table = pd.read_csv(args.window_betas)
        summary["real"] = {
            "stocks": sorted(table.stock.unique()), "first_week": str(table.window_start.min()), "last_week": str(table.window_end.max()),
            "mean_beta": float(table.beta.mean()),
            "blume": {int(w): bs.blume_test(g.to_dict("records"), n_boot=2000, seed=0) for w, g in table.groupby("window_weeks")},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
