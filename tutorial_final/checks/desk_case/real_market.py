"""Desk case: the real full SPY surface against the front-month calibration.

    python real_market.py OUT_DIR

Chi-squares of three fits on the 27 front-month quotes, on the other 36 quotes and on all 63: the full-surface
fit (rough_spy_is.py, seed 0), the front-month fit and the front-month fit with H fixed at 0.10 (desk_case.py).
Same pricer as everywhere in the case (32,768 paths per block, seed 0).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_case as dc  # noqa: E402
import rough_spy_is as base  # noqa: E402


def main(out):
    r = json.loads((Path(out) / "desk_case.json").read_text())
    full = json.loads((HERE.parent / "rough_spy_is" / "rough_spy_is_seed0.json").read_text())
    fits = {"full_surface": full["fit"]["theta"], "front_months": r["fit_front"]["theta"],
            "front_months_h_fixed": r["fit_h_fixed"]["theta"]}
    report = {}
    for name, theta in fits.items():
        p = base.surface(np.asarray(theta), dc.SEED)[0]
        z2 = ((p - base.Y) / base.SIGMA) ** 2
        report[name] = {"theta": theta, "chi2_front_27": float(z2[dc.FRONT].sum()),
                        "chi2_other_36": float(z2[~dc.FRONT].sum()), "chi2_all_63": float(z2.sum())}
        print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in report[name].items()})
    Path(out, "real_market.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
