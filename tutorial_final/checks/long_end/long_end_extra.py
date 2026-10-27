"""Prior sensitivity of the long-end ranges, and Heston's long end against SPY's LEAPs (long_end_extra.json).

    python long_end_extra.py        (after long_end.py; reads long_end.json and ../heston_checks-full_grid_kappa_free.npy)
"""

import json
from pathlib import Path

import numpy as np

import long_end as L

HERE = Path(__file__).resolve().parent


def main():
    r = json.loads((HERE / "long_end.json").read_text())
    g = np.array(r["gradient"])
    p = np.array(r["price_posterior_mean_inputs"])
    flat = np.log(np.concatenate([L.XI_SHORT, np.full(4, L.XI_SHORT[-1])]))
    out = {}
    for s in (0.1, 0.2, 0.3):
        for vs in (0.25, 0.5, 1.0):
            L.S_PRIOR, L.VS_SD = s, vs / 100
            _, c = L.posterior(flat)
            out[f"S={s},vs_sd={vs}"] = (100 * 2 * 1.645 * np.sqrt(np.einsum("pi,ij,pj->p", g, c, g)) / p).round(1).tolist()
    d = np.load(HERE.parent / "heston_checks-full_grid_kappa_free.npy").astype(float)
    v0, th, kap = d[:, 0], d[:, 1], d[:, 4]
    t = np.array([1.0, 1.20, 1.30, 1.71, 2.23, 3.0, 5.0])
    vs = np.sqrt(th[:, None] + (v0 - th)[:, None] * (1 - np.exp(-kap[:, None] * t)) / (kap[:, None] * t))
    q = np.quantile(100 * vs, [0.05, 0.5, 0.95], axis=0)
    leap = {str(round(tt, 2)): round(100 * np.sqrt(w / tt), 2) for tt, w in L.LEAPS}
    (HERE / "long_end_extra.json").write_text(json.dumps(
        {"sensitivity_range90_percent": out, "heston_kappa_free_vs_vol_q": q.tolist(), "T": t.tolist(), "leaps": leap},
        indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
