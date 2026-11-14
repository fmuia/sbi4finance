"""Heston Fisher information per grid column at SPY's exact-posterior mean, and for hypothetical deeper puts."""
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "tutorial/src"))
from sbi4f_tutorial.shared import surface as S, pricing, tasks  # noqa: E402

task = json.loads((ROOT / "tutorial/assets/backbones/heston-final500/task.json").read_text())
spy_exact = json.loads((ROOT / "tutorial/assets/data/heston-spy-exact.json").read_text())
names = task["parameter_names"]
theta0 = np.array([spy_exact["summary"][n]["mean"] for n in names])
print("theta0", dict(zip(names, theta0)))


def price(theta, contracts, nodes=2048, cut=1500.0):
    p, v0 = tasks._heston_split(tasks.model_parameters(task, theta))
    return pricing.heston_surface(p, v0, contracts, nodes=nodes, upper_limit=cut)


def jac(contracts):
    steps = np.array([2e-4, 5e-4, 5e-3, 5e-3])
    cols = []
    for i, h in enumerate(steps):
        e = np.zeros(4); e[i] = h
        cols.append((price(theta0 + e, contracts) - price(theta0 - e, contracts)) / (2 * h))
    return np.column_stack(cols)


np.set_printoptions(linewidth=200, precision=3, suppress=True)
deltas = np.array(S.DESIGN_DELTAS)
contracts = task["context"]["contracts"]
sig = np.array(task["context"]["quote_noise_std"])
# numerics check
p1 = price(theta0, contracts, 1024, 600.0); p2 = price(theta0, contracts, 4096, 3000.0)
print("max |p1024-p4096|/sigma:", np.max(np.abs(p1 - p2) / sig))
J = jac(contracts)
Js = J / sig[:, None]
per = (Js**2).reshape(7, 9, 4)
col = per.sum(0)
print("Heston: share of diagonal Fisher information by column", deltas)
for i, n in enumerate(names):
    print(f"{n:16s}", np.round(col[:, i] / col[:, i].sum(), 3))


def sd(mask):
    Jm = Js[mask]
    return np.sqrt(np.diag(np.linalg.inv(Jm.T @ Jm)))


full = np.ones(63, bool)
print("Laplace sd full:", sd(full), " posterior sd (exact):", [spy_exact["summary"][n]["sd"] for n in names])
for label, m in [("drop delta>=1 (21)", np.tile(deltas < 1, 7)), ("drop delta>=1.5 (14)", np.tile(deltas < 1.5, 7)),
                 ("drop delta<=-1.5 (14)", np.tile(deltas > -1.5, 7)), ("drop delta=-2 (7)", np.tile(deltas > -2, 7))]:
    print(f"{label:24s} sd ratio:", sd(m) / sd(full))

# Hypothetical deeper put columns at delta (20% units) = -3, -4, -6, -8, under two noise conventions.
extra_d = np.array([-3.0, -4.0, -6.0, -8.0])
days = np.array(S.DESIGN_DAYS)
ext = []
for dd in days:
    T = dd / 252
    for d in extra_d:
        ext.append({"strike": float(np.exp(d * 0.2 * np.sqrt(T))), "maturity": T, "option_type": "put"})
k, T, kind = S.contract_arrays(ext)
pe = price(theta0, ext)
ive = S.implied_vol(pe, k, T, kind)
sig_design = np.array(S.quote_noise(ext))  # vega at 20% x s_iv capped at 1 pt
sig_actual = S.bs_vega(k, T, ive) * 0.01      # 1 vol point at the model's own implied vol
print("\nHypothetical deep puts, model iv (%) rows=days cols=", extra_d)
print(100 * ive.reshape(7, 4))
print("design-convention noise in iv units at model iv (vol pts):")
print(100 * (sig_design / S.bs_vega(k, T, ive)).reshape(7, 4))
Je = jac(ext)
for label, s in [("design noise (vega@20%)", sig_design), ("1 vol pt at model iv", sig_actual)]:
    Jx = np.vstack([Js, Je / s[:, None]])
    Jm = Jx
    sd_new = np.sqrt(np.diag(np.linalg.inv(Jm.T @ Jm)))
    print(f"add 28 deep puts, {label}: sd ratio vs current grid", sd_new / sd(full))
# replace delta>=1 calls by delta=-3,-4 puts (same count 14 vs 21: use -3,-4,-6)
for label, s in [("design noise (vega@20%)", sig_design), ("1 vol pt at model iv", sig_actual)]:
    keep = np.tile(deltas < 1, 7)
    sel = np.tile(np.isin(extra_d, [-3.0, -4.0, -6.0]), 7)
    Jm = np.vstack([Js[keep], (Je / s[:, None])[sel]])
    print(f"swap 21 calls (delta>=1) for 21 puts (-3,-4,-6), {label}: sd ratio", np.sqrt(np.diag(np.linalg.inv(Jm.T @ Jm))) / sd(full))
