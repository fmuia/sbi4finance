"""CPU cost of the pieces a thin-chain study prices per posterior draw, for planning S10 (docs/course/v2-plan.md).

    python timing.py OUT.json [--repeats 3]

For one parameter set (the first desk-case proposal), it times, in CPU-seconds of this process:
  the 27 front-month quotes at 32,768 paths (two of the pricer's three blocks);
  the 63 design quotes at 32,768 paths;
  the four desk products (desk_products.py) at 8,192 and 16,384 paths;
  slot 5's four products (rough_spy_is.py) at 16,384 paths.
The median over the repeats is stored. The numbers depend on the machine; they were measured in the container.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_products as dp  # noqa: E402
import rough_spy_is as base  # noqa: E402
from rough_spy_is_cases import _DAYS  # noqa: E402
from sbi4finance.calibration import pricing, tasks  # noqa: E402


def cpu(fn, repeats):
    out = []
    for _ in range(repeats):
        t0 = time.process_time()
        fn()
        out.append(time.process_time() - t0)
    return float(np.median(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    theta = np.load(HERE / "desk_case_combined.npz")["theta"][0]
    params = tasks.model_parameters(base.TASK, theta)
    front = np.isin(_DAYS, [21, 42, 63])
    contracts = [c for c, f in zip(base.CONTRACTS, front) if f]
    blocks = [b for b in base.BLOCKS if b["horizon"] <= 0.25 + 1e-12]
    result = {
        "theta": theta.tolist(),
        "front_27_quotes_32768_paths": cpu(lambda: pricing.rough_surface(
            params, base.CURVE, contracts, blocks, [32768], base.SIGMA[front], np.inf, 0), args.repeats),
        "design_63_quotes_32768_paths": cpu(lambda: base.surface(theta, 0), args.repeats),
        "desk_products_8192_paths": cpu(lambda: dp.price(params, base.CURVE, 8192, 0), args.repeats),
        "desk_products_16384_paths": cpu(lambda: dp.price(params, base.CURVE, 16384, 0), args.repeats),
        "slot5_products_16384_paths": cpu(lambda: base.product_job((theta, 0)), args.repeats),
        "unit": "CPU-seconds, median of the repeats",
        "repeats": args.repeats,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
