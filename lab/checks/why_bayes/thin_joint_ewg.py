"""EWG's own chain of 22 October 2026 under the six-target thin task: each method's 90% bounds for the thin products.

The input is EWG read onto its layout (checks/thin_chains/EWG_observation.json, prices_forward_units), the
network's input; the noise is the layout's. `classical` fits least squares from the prior box and the quotes, with
and without H fixed, and writes the delta-method bounds of thin_joint_breach.py's three classical methods.
`posterior` samples the trained network and prices the products under its draws. There is no truth: this shows
how far the methods' answers differ on a real chain, not which is right.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/thin_joint_ewg.py classical --config THIN_JOINT_CONFIG.json --out DIR
python checks/why_bayes/thin_joint_ewg.py posterior --config THIN_JOINT_CONFIG.json --checkpoint RUN --out DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sbi4finance.calibration import thin_products

import thin_joint
from thin_joint_breach import HURST_FIXED, LAST_QUOTED, Fit, delta_bounds

OBSERVATION = Path("checks/thin_chains/EWG_observation.json")


def load(args):
    task = json.loads(args.config.read_text())["task"]
    quotes = np.asarray(json.loads(OBSERVATION.read_text())["prices_forward_units"], dtype=float)
    if len(quotes) != len(task["context"]["contracts"]):
        raise ValueError("the observation does not match the task's layout")
    return task, quotes


def classical(args):
    task, quotes = load(args)
    box = thin_joint.bounds(task)
    fitter = Fit(task, quotes, args.fit_paths, args.seed + 2)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 3, 99]))
    candidates = np.vstack([box.mean(axis=1), box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((32, 6))])
    chi2 = np.array([fitter.chi2(c) for c in candidates])
    order = np.argsort(chi2)
    fit = fitter.run(candidates[order[:3]])
    fixed = Fit(task, quotes, args.fit_paths, args.seed + 2, fixed_hurst=HURST_FIXED)
    fit_h = fixed.run([np.r_[HURST_FIXED, fit["theta"][1:]]])
    product = lambda theta, flat=None: thin_joint.products(task, theta, args.product_paths, args.seed + 1, flat)
    result = {"products": list(thin_products.NAMES), "quotes": len(quotes)}
    for name, f, fn, flat in (("gauss_newton", fit, fitter, None), ("hurst_fixed", fit_h, fixed, None),
                              ("flat_curve", fit, fitter, LAST_QUOTED)):
        mark, lower, upper = delta_bounds(f, fn, lambda theta: product(theta, flat))
        result[name] = {"mark": mark.tolist(), "lower": lower.tolist(), "upper": upper.tolist(), "chi2": f["chi2"],
                        "theta": f["theta"].tolist(), "condition": f["condition"], "on_bound": f["on_bound"]}
        print(f"{name}: chi2 {f['chi2']:.1f} on {len(quotes)} quotes, theta {np.round(f['theta'], 4).tolist()}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ewg-classical.json").write_text(json.dumps(result, indent=1) + "\n")


def posterior(args):
    task, quotes = load(args)
    workflow = thin_joint.load(args.checkpoint, task)
    draws = thin_joint.sample(workflow, task, quotes[None], args.draws, seed=args.seed + 5)[0]
    priced = np.array([thin_joint.products(task, d, args.product_paths, args.seed + 1) for d in draws])
    result = {"products": list(thin_products.NAMES), "draws": args.draws,
              "parameters": {n: np.quantile(draws[:, j], [0.05, 0.5, 0.95]).tolist() for j, n in enumerate(thin_joint.NAMES)},
              "lower": np.quantile(priced, 0.1, axis=0).tolist(), "upper": np.quantile(priced, 0.9, axis=0).tolist(),
              "median": np.median(priced, axis=0).tolist()}
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "ewg-posterior-draws.npy", draws)
    (args.out / "ewg-posterior.json").write_text(json.dumps(result, indent=1) + "\n")
    for j, n in enumerate(thin_joint.NAMES):
        print(f"  {n:15s} 5/50/95% " + " ".join(f"{v:.4f}" for v in result["parameters"][n]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("classical", classical), ("posterior", posterior)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--product-paths", type=int, default=8192)
        if name == "classical":
            p.add_argument("--fit-paths", type=int, default=4096)
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--draws", type=int, default=400)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
