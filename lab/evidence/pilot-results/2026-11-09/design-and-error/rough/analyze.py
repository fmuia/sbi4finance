"""Assemble Jacobians from the cached pricing runs and compute Fisher/Laplace summaries per grid and noise.

Usage: python analyze.py N_PATHS [CHECK_POINT CHECK_SEED_OFFSET]
Writes results.json next to this file.
"""

import json
import os
import sys

import numpy as np

import rough_fisher as rf
from sbi4f_tutorial.shared import surface, tasks

CACHE = rf.CACHE
NAMES = rf.NAMES
SIDES = ("put", "atm", "call")


def load(name, n_paths, seed_offset=0):
    pts = dict(rf.points())
    union = json.load(open(os.path.join(CACHE, f"union_{name}.json")))
    seed = 1000 + seed_offset + list(pts).index(name)
    pre = os.path.join(CACHE, f"stage2_{name}_{n_paths}_s{seed}_")
    base = np.load(pre + "base.npz")
    h = np.array(union["steps"])
    J = np.empty((len(union["contracts"]), 6))
    se = []
    for j, p in enumerate(NAMES):
        plus, minus = np.load(pre + f"{p}_p.npz"), np.load(pre + f"{p}_m.npz")
        J[:, j] = (plus["prices"] - minus["prices"]) / (2 * h[j])
        se.append(np.maximum(plus["se"], minus["se"]))
    return union, base["prices"], base["se"], J


def noise_sd(union, base_prices, grid, convention):
    meta = union["meta"][grid]
    idx = np.array(meta["index"])
    c = [union["contracts"][i] for i in idx]
    k, T, kind = surface.contract_arrays(c)
    delta = np.array(meta["delta"])
    if convention == "design":
        return surface.bs_vega(k, T, np.array(meta["sigma_scale"])) * surface.implied_vol_noise(delta)
    s_atm = np.array(union["sigma_atm"])[np.array(meta["maturity_index"])]
    iv = surface.implied_vol(base_prices[idx], k, T, kind)
    if not np.isfinite(iv).all():
        raise ValueError(f"model implied vol undefined on {grid}")
    return surface.bs_vega(k, T, iv) * surface.implied_vol_noise(k / (s_atm * np.sqrt(T)))


def vs_vol_gradient(theta):
    """1y variance-swap vol sqrt(int_0^1 xi) and its gradient (curve parameters only)."""
    s, l, d = theta[3:]
    w = -np.expm1(-d) / d
    var = w * s + (1 - w) * l
    dw = (np.exp(-d) * d - (1 - np.exp(-d))) / d**2
    g = np.zeros(6)
    g[3], g[4], g[5] = w, 1 - w, dw * (s - l)
    vol = np.sqrt(var)
    return vol, g / (2 * vol)


def summarise(union, base_prices, J, grid, convention):
    meta = union["meta"][grid]
    idx = np.array(meta["index"])
    sd = noise_sd(union, base_prices, grid, convention)
    Jw = J[idx] / sd[:, None]
    norms = np.linalg.norm(Jw, axis=0)
    active = norms > 1e-6 * norms.max()          # decay is exactly unidentified when short == long
    A = Jw[:, active]
    F = A.T @ A
    Finv = np.linalg.inv(F)
    post = np.full(6, np.nan)
    post[active] = np.sqrt(np.diag(Finv))
    lev = np.einsum("ij,jk,ik->i", A, Finv, A)
    delta = np.array(meta["delta"])
    rank = active.sum()
    shares = {"put": lev[delta < 0].sum() / rank, "atm": lev[delta == 0].sum() / rank,
              "call": lev[delta > 0].sum() / rank}
    theta = np.array(union["theta"])
    _, g_vs = vs_vol_gradient(theta)
    j_atm1y = [i for i, (m, d) in enumerate(zip(meta["maturity_index"], delta)) if m == 6 and d == 0][0]
    T1 = rf.MATS[6]
    g_atm = J[idx[j_atm1y]] / surface.bs_vega(0.0, T1, union["sigma_atm"][6])
    prod = {}
    for label, g in (("vs_vol_1y", g_vs), ("atm_iv_1y", g_atm)):
        gg = g[active]
        prod[label] = float(np.sqrt(gg @ Finv @ gg))
    return {"post_sd": post.tolist(), "active": active.tolist(), "leverage_share": shares,
            "products_sd": prod, "cond_F_corr": float(np.linalg.cond(F / np.sqrt(np.outer(np.diag(F), np.diag(F))))),
            "median_noise_volpts": None}


def analyse_point(name, n_paths, seed_offset=0):
    union, base, base_se, J = load(name, n_paths, seed_offset)
    # Sanity: G0 design noise equals the task's quote_noise_std (same contract order).
    g0 = noise_sd(union, base, "G0", "design")
    assert np.allclose(g0, tasks.sigma(rf.TASK), rtol=1e-10), "G0 design noise != task quote_noise_std"
    out = {"theta": union["theta"], "sigma_atm": union["sigma_atm"], "steps": union["steps"],
           "vs_vol_1y": float(vs_vol_gradient(np.array(union["theta"]))[0]), "grids": {}}
    for conv in ("design", "local"):
        for g in rf.GRIDS:
            out["grids"][f"{g}/{conv}"] = summarise(union, base, J, g, conv)
    # MC standard error of the base prices relative to the noise, for the record
    for conv in ("design", "local"):
        r = []
        for g in rf.GRIDS:
            idx = np.array(union["meta"][g]["index"])
            r.append(float(np.max(base_se[idx] / noise_sd(union, base, g, conv))))
        out[f"max_base_se_over_noise_{conv}"] = dict(zip(rf.GRIDS, r))
    return out, union, base, J


def mc_check(name, n_paths, offset_b):
    ua, ba, _, Ja = load(name, n_paths, 0)
    ub, bb, _, Jb = load(name, n_paths, offset_b)
    res = {}
    for conv in ("design", "local"):
        for g in rf.GRIDS:
            idx = np.array(ua["meta"][g]["index"])
            sd = noise_sd(ua, ba, g, conv)
            Wa, Wb = Ja[idx] / sd[:, None], Jb[idx] / sd[:, None]
            col = np.linalg.norm(Wa - Wb, axis=0) / np.linalg.norm(Wa, axis=0)
            sa = summarise(ua, ba, Ja, g, conv)["post_sd"]
            sb = summarise(ub, bb, Jb, g, conv)["post_sd"]
            res[f"{g}/{conv}"] = {
                # ||J_w(seed A) - J_w(seed B)|| / ||J_w(seed A)|| per parameter column
                "jacobian_col_rel_diff": dict(zip(NAMES, col.tolist())),
                "post_sd_rel_diff": dict(zip(NAMES, (np.array(sb) / np.array(sa) - 1).tolist()))}
    return res


def main():
    n_paths = int(sys.argv[1])
    names = [n for n, _ in rf.points()
             if os.path.exists(os.path.join(CACHE, f"stage2_{n}_{n_paths}_s{1000 + [m for m, _ in rf.points()].index(n)}_decay_m.npz"))]
    results = {"n_paths_per_block": n_paths, "points": {}, "api": {
        "pricer": "sbi4f_tutorial.shared.pricing.rough_surface (ladder [N], budget inf, sigma 1)",
        "curve": "sbi4finance.calibration.tasks.joint_curve",
        "parameters": "sbi4f_tutorial.shared.tasks.model_parameters",
        "iv_vega_noise": "sbi4f_tutorial.shared.surface.implied_vol, bs_vega, implied_vol_noise"}}
    for n in names:
        results["points"][n], *_ = analyse_point(n, n_paths)
    if len(sys.argv) > 3:
        results["mc_check"] = {"point": sys.argv[2], "seed_offset": int(sys.argv[3]),
                               "result": mc_check(sys.argv[2], n_paths, int(sys.argv[3]))}
    # ratios vs G0 and medians
    ratio_tab = {}
    for conv in ("design", "local"):
        for g in rf.GRIDS:
            key = f"{g}/{conv}"
            rows = {}
            for n in names:
                a = np.array(results["points"][n]["grids"][key]["post_sd"])
                b = np.array(results["points"][n]["grids"][f"G0/{conv}"]["post_sd"])
                rows[n] = (a / b).tolist()
                pa = results["points"][n]["grids"][key]["products_sd"]
                pb = results["points"][n]["grids"][f"G0/{conv}"]["products_sd"]
                rows[n] += [pa["vs_vol_1y"] / pb["vs_vol_1y"], pa["atm_iv_1y"] / pb["atm_iv_1y"]]
            M = np.array(list(rows.values()))
            ratio_tab[key] = {"per_point": rows,
                              "median_all": np.nanmedian(M, axis=0).tolist(),
                              "median_excl_spy": np.nanmedian(np.array([rows[n] for n in names if n != "spy_like"]), axis=0).tolist(),
                              "min": np.nanmin(M, axis=0).tolist(), "max": np.nanmax(M, axis=0).tolist(),
                              "spy_like": rows.get("spy_like")}
    results["ratio_vs_G0"] = {"columns": NAMES + ["vs_vol_1y", "atm_iv_1y"], "table": ratio_tab}
    # G0 local vs G0 design
    results["G0_local_over_design"] = {n: (np.array(results["points"][n]["grids"]["G0/local"]["post_sd"])
                                           / np.array(results["points"][n]["grids"]["G0/design"]["post_sd"])).tolist()
                                       for n in names}
    json.dump(results, open(os.path.join(rf.HERE, "results.json"), "w"), indent=1)
    # print compact tables
    cols = NAMES + ["vsvol", "atmiv"]
    print("points:", names)
    for key, t in ratio_tab.items():
        print(f"{key:10s} median", " ".join(f"{x:6.3f}" for x in t["median_all"]),
              "| spy", " ".join(f"{x:6.3f}" for x in (t["spy_like"] or [np.nan] * 8)))
    print("cols:", cols)
    for n in names:
        print(n, {k: {s: round(v, 3) for s, v in g["leverage_share"].items()}
                  for k, g in results["points"][n]["grids"].items()})


if __name__ == "__main__":
    main()
