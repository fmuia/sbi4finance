"""A candidate curve prior for the full rough-Bergomi bank (gate G0), checked on SPY's curve of 21 October 2026.

    python curve_prior.py OUT.json FIGURE.png

The pilot's prior draws each forward-variance level as sigma0^2 (1 + s t_j) e^{u_j}, with sigma0 ~ U(0.10, 0.40),
s ~ U(-0.3, 0.6) and u_j ~ U(-0.2, 0.2) (tasks.rough_task). Its last-to-first ratio is at most 2.15; SPY's is 2.93.

The candidate is a power law in time with a random walk in log time:

    log xi_j = 2 log sigma0 + b (log t_j - log t_1) + W_j,   W_1 = 0,   W_{j+1} - W_j ~ N(0, s_w^2 (log t_{j+1} - log t_j)),

with t_j the midpoint of the j-th design interval, sigma0 log-uniform on (0.08, 0.60), b ~ U(-0.6, 0.6) and s_w a fixed
scale (0.10, 0.15 or 0.25 here). b > 0 is a rising (calm-market) curve, b < 0 an inverted (stressed) one.

Checks:
  1. SPY against a power law: exponent and residuals.
  2. Prior predictive: the central 90% envelope of curves relative to their first level, old prior against candidate.
  3. Extrapolation: with the first five levels (to 63 days) known, or the first six (to 126 days), the Gaussian
     predictive of the later levels (b approximated by a normal with the uniform's standard deviation), against SPY's
     levels and against flat extrapolation.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sbi4finance.calibration.plots import plt, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SPY = json.loads((ROOT / "tutorial_final" / "results" / "calibration" / "spy" / "surface.json").read_text())
KNOTS = np.array([0, 5, 10, 21, 42, 63, 126, 252]) / 252
MID = 0.5 * (KNOTS[1:] + KNOTS[:-1])
LT = np.log(MID)
X = LT - LT[0]
B_BOUNDS = (-0.6, 0.6)
SIGMA0_BOUNDS = (0.08, 0.60)
SCALES = (0.10, 0.15, 0.25)


def old_prior(rng, n):
    s0 = rng.uniform(0.10, 0.40, n)
    s = rng.uniform(-0.3, 0.6, n)
    u = rng.uniform(-0.2, 0.2, (n, len(MID)))
    return 2 * np.log(s0)[:, None] + np.log1p(s[:, None] * MID[None, :]) + u


def candidate(rng, n, s_w):
    s0 = np.exp(rng.uniform(*np.log(SIGMA0_BOUNDS), n))
    b = rng.uniform(*B_BOUNDS, n)
    steps = rng.normal(0, 1, (n, len(MID) - 1)) * s_w * np.sqrt(np.diff(LT))
    w = np.hstack([np.zeros((n, 1)), np.cumsum(steps, axis=1)])
    return 2 * np.log(s0)[:, None] + b[:, None] * X[None, :] + w


def predictive(rel, known, s_w):
    """Gaussian predictive of the levels after `known`, given levels 1..known exactly (relative to level 1)."""
    sb = (B_BOUNDS[1] - B_BOUNDS[0]) / np.sqrt(12)
    idx = np.arange(1, len(MID))
    cov = sb ** 2 * np.outer(X[idx], X[idx]) + s_w ** 2 * np.minimum.outer(X[idx], X[idx])
    obs = idx < known
    c_oo, c_uo, c_uu = cov[np.ix_(obs, obs)], cov[np.ix_(~obs, obs)], cov[np.ix_(~obs, ~obs)]
    mean = c_uo @ np.linalg.solve(c_oo, rel[idx][obs])
    var = np.diag(c_uu - c_uo @ np.linalg.solve(c_oo, c_uo.T))
    return idx[~obs], mean, np.sqrt(var)


def main(out_path, fig_path):
    xi = np.asarray(SPY["forward_variance_levels"])
    y = np.log(xi)
    rel = y - y[0]
    vol = lambda r: 100 * np.sqrt(xi[0] * np.exp(r))  # noqa: E731
    design = np.vstack([np.ones_like(X), X]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    out = {"spy_levels_vol": list(100 * np.sqrt(xi)), "power_law": {"exponent": coef[1],
           "residual_sd": float(resid.std(ddof=2)), "max_abs_residual": float(np.abs(resid).max())},
           "spy_last_over_first": float(xi[-1] / xi[0]), "old_prior_bound": 2.15, "envelopes": {}, "extrapolation": []}

    rng = np.random.default_rng(0)
    old = old_prior(rng, 200_000)
    env = {"old": old - old[:, :1]}
    for s_w in SCALES:
        draws = candidate(rng, 200_000, s_w)
        env[f"s_w={s_w}"] = draws - draws[:, :1]
    for k, r in env.items():
        q = np.quantile(r, [0.05, 0.95], axis=0)
        out["envelopes"][k] = {"q05": list(q[0]), "q95": list(q[1]),
                               "spy_inside_central_90_at_every_level": bool(np.all((rel >= q[0]) & (rel <= q[1]))),
                               "share_with_ratio_at_least_spy": float(np.mean(np.exp(r[:, -1]) >= xi[-1] / xi[0])),
                               "share_inverted_as_much": float(np.mean(np.exp(r[:, -1]) <= xi[0] / xi[-1]))}

    for known in (5, 6):
        for s_w in SCALES:
            levels, mean, sd = predictive(rel, known, s_w)
            for j, m, s in zip(levels, mean, sd):
                out["extrapolation"].append({"known_to_days": int(round(KNOTS[known] * 252)), "s_w": s_w,
                                             "level": int(j + 1), "q05_vol": vol(m - 1.645 * s), "median_vol": vol(m),
                                             "q95_vol": vol(m + 1.645 * s), "spy_vol": vol(rel[j]),
                                             "flat_vol": vol(rel[known - 1]), "z_spy": (rel[j] - m) / s,
                                             "z_flat": (rel[known - 1] - m) / s})
    Path(out_path).write_text(json.dumps(out, indent=1))

    style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(MID * 252, 100 * np.sqrt(xi), "o-", color="0.15", label="SPY, 21 Oct 2026")
    ax.plot(MID * 252, 100 * np.sqrt(np.exp(design @ coef)), "--", color="#D97732",
            label=f"power law, exponent {coef[1]:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("interval midpoint (business days, log scale)")
    ax.set_ylabel("forward volatility (%)")
    ax.set_title("SPY's curve is close to a power law in time", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for k, colour in [("old", "0.6"), ("s_w=0.15", "#087F8C")]:
        e = out["envelopes"][k]
        ax.fill_between(MID * 252, np.exp(e["q05"]), np.exp(e["q95"]), color=colour, alpha=0.35,
                        label="pilot prior, central 90%" if k == "old" else "candidate (s_w = 0.15), central 90%")
    ax.plot(MID * 252, np.exp(rel), "o-", color="0.15", label="SPY")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("interval midpoint (business days, log scale)")
    ax.set_ylabel("level / first level (log scale)")
    ax.set_title("Prior curves relative to their first level", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[2]
    rows = [r for r in out["extrapolation"] if r["known_to_days"] == 63]
    for i, s_w in enumerate(SCALES):
        for r in (r for r in rows if r["s_w"] == s_w):
            x = r["level"] + (i - 1) * 0.12
            ax.plot([x, x], [r["q05_vol"], r["q95_vol"]], lw=4, color=["#9ACFD4", "#087F8C", "#04474E"][i],
                    solid_capstyle="butt", label=f"candidate, s_w = {s_w}" if r["level"] == 6 else None)
    for level in (6, 7):
        r = next(r for r in rows if r["level"] == level)
        ax.plot(level + 0.3, r["spy_vol"], "o", color="0.15", label="SPY" if level == 6 else None)
        ax.plot(level + 0.3, r["flat_vol"], "x", color="#B33E43", ms=8, label="flat extrapolation" if level == 6 else None)
    ax.set_xticks([6, 7])
    ax.set_xticklabels(["63-126 days", "126-252 days"])
    ax.set_ylabel("forward volatility (%)")
    ax.set_title("Given the curve to 63 days: 90% bands", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(fig_path, dpi=160)
    print(json.dumps({k: out[k] for k in ("power_law", "spy_last_over_first")}, indent=1))
    for k, e in out["envelopes"].items():
        print(k, e["spy_inside_central_90_at_every_level"], round(e["share_with_ratio_at_least_spy"], 3),
              round(e["share_inverted_as_much"], 3))
    for r in out["extrapolation"]:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
