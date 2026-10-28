"""SBC, coverage and contraction of the six-target thin network on held-out surfaces, at one noise scale.

Each surface's quotes are its clean layout prices plus a fresh error with sd scale x noise, one stream per row,
independent of the bank's stored noise. Two derived targets are checked as well: rho x eta, and the forward
variance from six months to one year, which lies beyond the last quote and depends on the curve's coefficients.
Contraction is reported for the six box targets, in the logit coordinates of their uniform prior.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/thin_joint_sbc.py --checkpoint RUN --bank BANK --out OUT.json [--split validation
    --surfaces 2000 --draws 500 --scale 1.0 --seed 0]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sbi4finance.calibration import bank, validation

import thin_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--surfaces", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank, splits=(args.split,))
    task = manifest["configuration"]["task"]
    data = splits[args.split]
    rows = np.arange(min(args.surfaces, len(data["parameters"])))
    sd = thin_joint.noise(task)
    quotes = np.stack([data["reference_prices"][r] + args.scale * sd * np.random.default_rng(
        np.random.SeedSequence([args.seed, int(r)])).standard_normal(len(sd)) for r in rows])
    workflow = thin_joint.load(args.checkpoint, task)
    started = time.perf_counter()
    draws = np.concatenate([thin_joint.sample(workflow, task, quotes[s:s + 250], args.draws, args.scale,
                                              seed=args.seed + s) for s in range(0, len(rows), 250)])
    seconds = time.perf_counter() - started
    truth = np.asarray(data["parameters"], dtype=float)[rows]
    names = thin_joint.NAMES + ["rho_eta", "forward_variance_6m_1y"]
    extra = lambda x: np.stack([x[..., 2] * x[..., 1], thin_joint.forward_variance(x, 0.5, 1.0)], axis=-1)
    draws_all, truth_all = np.concatenate([draws, extra(draws)], -1), np.concatenate([truth, extra(truth)], -1)
    sbc = validation.sbc(draws_all, truth_all, seed=args.seed)
    cover = validation.coverage(draws_all, truth_all)
    shrink = validation.contraction(draws, thin_joint.bounds(task))
    report = {"checkpoint": str(args.checkpoint), "bank": str(args.bank), "split": args.split, "surfaces": len(rows),
              "draws": args.draws, "noise_scale": args.scale, "seed": args.seed, "targets": names,
              "sbc": {k: v for k, v in sbc.items() if k != "fractional_ranks"}, "coverage": cover,
              "contraction": dict(zip(thin_joint.NAMES, shrink)), "seconds_per_surface": seconds / len(rows)}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(rows)} surfaces x {args.draws} draws at noise x{args.scale:g}, {seconds / len(rows):.3f} s a surface;"
          f" DKW bound {sbc['dkw_bound']:.4f}")
    for j, name in enumerate(names):
        c90 = cover["0.9"]["coverage"][j]
        shrink_j = f", contraction {shrink[j]:.2f}" if j < len(shrink) else ""
        print(f"  {name:24s} rank KS {sbc['rank_ks'][j]:.4f} {'pass' if sbc['passed'][j] else 'FAIL'};"
              f" 90% coverage {c90:.3f}{shrink_j}")


if __name__ == "__main__":
    main()
