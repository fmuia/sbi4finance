"""Fisher information of four option grids for the joint rough-Bergomi task (scratch experiment).

Pricing reuses the runtime: sbi4f_tutorial.shared.pricing.rough_surface (hybrid scheme, antithetic pairs,
control variates, maturity blocks from task numerics) with the analytic joint curve from
sbi4finance.calibration.tasks.joint_curve and parameters from sbi4f_tutorial.shared.tasks.model_parameters.
A single-rung ladder [N] with an infinite budget makes every call use exactly N paths per block.
Stream SeedSequence([seed, block, 0]) is shared by +h and -h (common random numbers).

Usage: python rough_fisher.py stage1|stage2 N_PATHS POINT_IDS... [--seed-offset S]
"""

import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src"), str(pathlib.Path(__file__).resolve().parents[6] / "lab" / "src")]
import numpy as np  # noqa: E402

from sbi4f_tutorial.shared import pricing, surface, tasks  # noqa: E402
from sbi4finance.calibration import tasks as lab_tasks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
TASK = json.load(open(str(pathlib.Path(__file__).resolve().parents[6]) + "/tutorial/assets/backbones/rough-joint100k-validation0/task.json"))
NAMES = TASK["parameter_names"]
BOUNDS = np.array(TASK["prior_bounds"])

D_REF = np.array(surface.DESIGN_DELTAS)                                   # -2..2 step .5
D_ATM = np.array([-4.0, -3.25, -2.5, -1.75, -1.0, -0.5, 0.0, 0.5, 1.0])   # proposed list
DAYS = np.array(surface.DESIGN_DAYS)
MATS = DAYS / surface.TRADING_DAYS
# Grid name -> (delta list, scale kind)
GRIDS = {"G0": (D_REF, "ref"), "G1": (D_ATM, "atm"), "G2": (D_ATM, "ref"), "G3": (D_REF, "atm")}


def points():
    """Evaluation points (name, 6-vector). Fixed seed for the random prior draws."""
    centre = BOUNDS.mean(axis=1)
    spy = np.array([0.10, 1.9, -0.8, 0.018, 0.05, 2.0])
    assert np.all((spy >= BOUNDS[:, 0]) & (spy <= BOUNDS[:, 1]))
    rng = np.random.default_rng(20261109)
    draws = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(6, 6))
    out = [("centre", centre), ("spy_like", spy)] + [(f"draw{i}", d) for i, d in enumerate(draws)]
    return out


def steps(theta):
    """Central-difference half-steps: absolute for hurst/eta/rho, 2% relative for curve coefficients."""
    h = np.array([0.005, 0.03, 0.01, 0.0, 0.0, 0.0])
    h[3:] = 0.02 * theta[3:]
    return h


def price(theta, contracts, n_paths, seed):
    params = tasks.model_parameters(TASK, theta)
    curve = lab_tasks.joint_curve(TASK, theta)
    p, s, _, _ = pricing.rough_surface(params, curve, contracts, TASK["numerics"]["blocks"], [int(n_paths)],
                                       np.ones(len(contracts)), np.inf, int(seed))
    return p, s


def atm_contracts():
    return [{"strike": 1.0, "maturity": float(t), "option_type": "put"} for t in MATS]


def build_union(sigma_atm):
    """Union of the four grids' contracts (deduplicated by maturity and log-strike) and per-grid metadata."""
    keys, contracts = {}, []
    meta = {}
    for g, (deltas, scale) in GRIDS.items():
        idx, dl, sc, mat = [], [], [], []
        for j, t in enumerate(MATS):
            s = surface.SIGMA_REF if scale == "ref" else sigma_atm[j]
            for d in deltas:
                k = float(d * s * np.sqrt(t))
                key = (j, round(k, 12))
                if key not in keys:
                    keys[key] = len(contracts)
                    contracts.append({"strike": float(np.exp(k)), "maturity": float(t),
                                      "option_type": "put" if d <= 0 else "call"})
                idx.append(keys[key]); dl.append(float(d)); sc.append(float(s)); mat.append(j)
        meta[g] = {"index": idx, "delta": dl, "sigma_scale": sc, "maturity_index": mat}
    return contracts, meta


def _cached(tag, fn):
    path = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = time.time()
    out = fn()
    out["seconds"] = np.array(time.time() - t0)
    np.savez(path, **out)
    return out


def stage1_job(args):
    name, theta, n_paths = args
    def run():
        p, s = price(theta, atm_contracts(), n_paths, 777)
        iv = surface.implied_vol(p, np.zeros(len(MATS)), MATS, "put")
        return {"prices": p, "se": s, "sigma_atm": iv}
    return name, _cached(f"stage1_{name}_{n_paths}", run)


def stage2_job(args):
    name, theta, j, sign, n_paths, seed, contracts = args
    def run():
        th = np.array(theta, dtype=float)
        if j >= 0:
            th[j] += sign * steps(np.asarray(theta))[j]
        p, s = price(th, contracts, n_paths, seed)
        return {"prices": p, "se": s, "theta": th}
    tag = f"stage2_{name}_{n_paths}_s{seed}_" + ("base" if j < 0 else f"{NAMES[j]}_{'p' if sign > 0 else 'm'}")
    return tag, _cached(tag, run)


def main():
    stage, n_paths = sys.argv[1], int(sys.argv[2])
    seed_offset = 0
    rest = sys.argv[3:]
    if "--seed-offset" in rest:
        i = rest.index("--seed-offset"); seed_offset = int(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
    pts = dict(points())
    wanted = rest or list(pts)
    with Pool(2) as pool:
        if stage == "stage1":
            for name, out in pool.imap_unordered(stage1_job, [(n, pts[n], n_paths) for n in wanted]):
                print(name, np.round(out["sigma_atm"], 4), f"{float(out['seconds']):.1f}s", flush=True)
            return
        jobs = []
        for pi, name in enumerate(wanted):
            s1 = dict(np.load(os.path.join(CACHE, f"stage1_{name}_8192.npz")))
            contracts, meta = build_union(s1["sigma_atm"])
            json.dump({"contracts": contracts, "meta": meta, "sigma_atm": s1["sigma_atm"].tolist(),
                       "theta": pts[name].tolist(), "steps": steps(pts[name]).tolist()},
                      open(os.path.join(CACHE, f"union_{name}.json"), "w"))
            seed = 1000 + seed_offset + list(pts).index(name)
            jobs.append((name, pts[name], -1, 0, n_paths, seed, contracts))
            for j in range(6):
                for sign in (1, -1):
                    jobs.append((name, pts[name], j, sign, n_paths, seed, contracts))
        t0 = time.time()
        for k, (tag, out) in enumerate(pool.imap_unordered(stage2_job, jobs)):
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {tag} {float(out['seconds']):.1f}s", flush=True)


if __name__ == "__main__":
    main()
