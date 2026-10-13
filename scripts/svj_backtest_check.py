"""Check the backtest itself: with the true parameters and a filtered state, the 99% VaR must be exceeded 1% of the time.

    pixi run python scripts/svj_backtest_check.py                     # 2,000 fresh series
    pixi run python scripts/svj_backtest_check.py --n-series 500 --seed 4242   # the backtest's own series

For the one-day horizon both sides are exact: the reported VaR is the 1% quantile of the mixture averaged
over the particle filter's cloud for v_T, and its exceedance probability is the mixture CDF at the true
v_T. Averaged over series from the prior the result must be 1%; the per-series values are right-skewed
(most series sit below 1%, a few far above), so a batch of 500 series can land well below 1% by chance.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sbi4finance.svj import filtering, model, risk

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-series", type=int, default=2_000)
    parser.add_argument("--n-particles", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results" / "svj-backtest" / "check_exact_1day.json"))
    args = parser.parse_args()

    sims = model.simulate(args.n_series, seed=args.seed)
    p_exc = np.empty(args.n_series)
    for i in range(args.n_series):
        params = model.model_params(sims["targets"][i])
        _, cloud = filtering.filtered_variance(params, sims["returns"][i], seed=args.seed + i, n_particles=args.n_particles)
        p_exc[i] = risk.one_day_cdf(params, sims["variance"][i, -1], risk.cloud_quantile(params, cloud, 0.01))
    se = p_exc.std(ddof=1) / np.sqrt(args.n_series)
    batches = [float(b.mean()) for b in np.array_split(p_exc, max(1, args.n_series // 500))]
    result = {
        "n_series": args.n_series,
        "seed": args.seed,
        "n_particles": args.n_particles,
        "mean_exceedance": float(p_exc.mean()),
        "ci95": [float(p_exc.mean() - 1.96 * se), float(p_exc.mean() + 1.96 * se)],
        "median_exceedance": float(np.median(p_exc)),
        "share_below_1pct": float(np.mean(p_exc < 0.01)),
        "batch_means_of_500": batches,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"exact 1-day check on {args.n_series} series: {100 * p_exc.mean():.3f}% "
          f"(95% CI {100 * result['ci95'][0]:.3f} to {100 * result['ci95'][1]:.3f}); median {100 * result['median_exceedance']:.3f}%; "
          f"batch means {[round(100 * b, 3) for b in batches]}")


if __name__ == "__main__":
    main()
