"""CPU-seconds of one audit evaluation on EWG's layout and quotes, and of one evaluation of the six thin products.

    python timing.py

At a parameter set near the middle of the prior (H 0.10, eta 1.8, rho -0.75, volatility levels 18% to 22%), after
one warm-up call, each timing is the median of three calls. Writes timing.json next to this file. The figures
plan the cluster runs (lab/plans/CLUSTER-RUNS.md) and the costs of course/v2-plan.md, Section 2.4.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))
sys.path.insert(0, str(ROOT / "lab" / "studies"))
from sbi4finance.calibration import correction, thin, thin_products  # noqa: E402
from thin_real import chain_data  # noqa: E402


def timed(fn, repeats=3):
    fn()
    runs = []
    for _ in range(repeats):
        started = time.process_time()
        out = fn()
        runs.append(time.process_time() - started)
    return float(np.median(runs)), out


def main():
    task = json.loads((ROOT / "lab/configs" / "tasks" / "calibration_rough_bank.json").read_text())["task"]
    layout = thin.load_layout(ROOT / "lab/configs" / "thin" / "ewg_layout.json", task)
    _, noise = thin.layout_arrays(layout)
    contracts = thin.layout_contracts(task, layout)
    quotes, _, quote_noise, _ = chain_data(json.loads((HERE / "EWG_observation.json").read_text()))
    row = np.concatenate([[0.10, 1.8, -0.75], 2 * np.log([0.18, 0.19, 0.20, 0.205, 0.21, 0.215, 0.22])])
    report = {"parameters": row.tolist(), "layout": {}, "quotes": {}}
    for name, cs, sd in (("layout", contracts, noise), ("quotes", quotes, quote_noise)):
        for paths in (4096, 8192, 32768):
            seconds, (p, se, cov) = timed(lambda: thin.price_contracts(task, row, cs, sd, [paths], 1, budget=np.inf,
                                                                       cap=None, covariance=True))
            report[name][str(paths)] = {"cpu_seconds": seconds, "contracts": len(cs),
                                        "largest_se_over_noise": float(np.max(se / sd)),
                                        "trace": float(np.sum((se / sd) ** 2)),
                                        "largest_eigenvalue": correction.relative_mc_eigenvalue(sd, cov)}
    seconds, _ = timed(lambda: thin_products.price(task, row[:3], row[3:], 8192, 1))
    report["products_8192_paths_cpu_seconds"] = seconds
    (HERE / "timing.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
