"""Desk case: the front-month quotes along H. For each H on a grid, least squares on (eta, rho), and the products.

    python profile_h.py OUT_DIR

This is the profile likelihood in H, and at the same time desk practice with every choice of H: each row is the
calibration a desk would get by fixing H at that value. The chi-square shows how much the 27 front-month quotes
care about H; the products show how much the 1-year marks depend on it. Same pricer and products as desk_case.py.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_case as dc  # noqa: E402
import desk_products as dp  # noqa: E402
import rough_spy_is as base  # noqa: E402

GRID = (0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30)


def row(h):
    start = np.array([h, 2.9, -0.65])
    theta, chi2, n = dc.least_squares_fit(dc.FRONT, start, [1, 2])
    full_chi2 = float(np.sum(((base.surface(theta, dc.SEED)[0] - base.Y) / base.SIGMA) ** 2))
    return {"h": h, "theta": theta.tolist(), "chi2_front": chi2, "chi2_all_63": full_chi2, "evaluations": n,
            "products": dc.product_job((theta, dc.SEED)).tolist()}


def main(out):
    started = time.time()
    with ProcessPoolExecutor(2) as pool:
        rows = list(pool.map(row, GRID))
    report = {"grid": list(GRID), "products": list(dp.NAMES), "rows": rows, "wall_seconds": time.time() - started}
    Path(out, "profile_h.json").write_text(json.dumps(report, indent=1))
    for r in rows:
        print(f"H {r['h']:.3f}: eta {r['theta'][1]:.3f} rho {r['theta'][2]:+.3f}  chi2 front {r['chi2_front']:6.2f}  "
              f"chi2 all 63 {r['chi2_all_63']:8.1f}  products {np.round(r['products'], 5)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
