"""The numbers behind slot 1: betas across many stocks, simulated universes and real windows.

    pixi run beta-study --out docs/results/intro-betas/summary.json  # about ten seconds

For each data frequency in `beta_shrinkage.SETTINGS`, 200 simulated universes of 500 stocks compare
least squares, Bloomberg's rule and Bayes' rule (with the prior learned from the cross-section, one
prior per volatility group, and the true population). The same study with weekly data runs on the
populations in `beta_shrinkage.MISSPECIFIED`, where the prior's picture of the stocks is wrong, and
two smaller studies repeat it for a low-beta universe and for five stocks. The real-data part runs
Blume's test on `docs/results/intro-betas/window_betas.csv` (see `scripts/beta_window_betas.py`).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sbi4finance import beta_shrinkage as bs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-betas", type=Path, default=ROOT / "docs" / "results" / "intro-betas" / "window_betas.csv")
    args = parser.parse_args()
    if not args.window_betas.exists():
        parser.error(f"{args.window_betas} does not exist; build it with scripts/beta_window_betas.py")
    weekly = bs.SETTINGS["weekly"]
    table = pd.read_csv(args.window_betas)
    summary = {
        "population": asdict(bs.POPULATION),
        "settings": {key: asdict(setting) for key, setting in bs.SETTINGS.items()},
        "simulated": {key: bs.study(setting, n_universes=200, seed=1) for key, setting in bs.SETTINGS.items()},
        "misspecified": {key: {**bs.study(weekly, population, n_universes=200, seed=2), "population_parameters": asdict(population)}
                         for key, population in bs.MISSPECIFIED.items()},
        "utilities": bs.study(weekly, bs.UTILITIES, n_universes=200, n_stocks=60, top=10, seed=5),
        "five_stocks": bs.study(weekly, n_universes=400, n_stocks=5, top=1, seed=5),
        "real": {
            "stocks": sorted(table.stock.unique()), "first_week": str(table.window_start.min()), "last_week": str(table.window_end.max()),
            "mean_beta": float(table.beta.mean()),
            "blume": {int(w): {"first_week": str(g.window_start.min()), "last_week": str(g.window_end.max()),
                               **bs.blume_test(g.to_dict("records"), n_boot=2000, seed=0)}
                      for w, g in table.groupby("window_weeks")},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
