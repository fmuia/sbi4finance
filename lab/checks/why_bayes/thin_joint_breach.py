"""How often each method's one-sided 90% bounds for the thin products are breached, on simulated thin chains.

The joint-task version of study S10's design (studies/thin_coverage.py), for EWG's layout. Each chain is a test
surface of a thin joint bank: its clean layout prices plus a fresh quote error with the layout's noise (seed per
row, as in thin_joint_sbc.py). Its truth is the six thin products priced at the surface's parameters. Six methods
give a lower and an upper 90% bound for each product:

* posterior: the 10% and 90% quantiles of the products priced under the network's draws;
* gauss_newton: least squares in the same six-parameter model, the Gauss-Newton covariance in the logit
  coordinates of the prior box, and the delta method: the price at the fit plus or minus 1.2816 sd;
* hurst_fixed: the same with H fixed at 0.10, the course's baseline (Lecture 6, Section 6.7);
* flat_curve: the gauss_newton fit, with the curve held flat beyond the last quote at its average over the last
  quoted interval (thin_joint.product_curve);
* laplace: the posterior's mode and the Laplace approximation there, in the same logit coordinates, where the
  uniform prior on the box is a logistic density in each coordinate (Gauss-Newton curvature for the quotes, the
  exact curvature for the prior); the 10% and 90% quantiles of the products under 200 draws from that normal,
  each mapped back into the box;
* laplace_delta: the delta method at the same mode with the same covariance.

A long breach is a true price below the lower bound, a short breach one above the upper bound; a correct method
breaches each side 10% of the time. Every product is priced with the same paths (common random numbers), so the
spreads come from the parameters. Least squares starts from the prior box and the quotes only, never from the
posterior: the best of the box's centre and 16 prior draws by chi-square, and a second start from the next best.
The mode starts from the least-squares fit, each logit coordinate clipped to [-4, 4].

Needs the joint task of PR #48. Run from lab/; the classical arm needs no network and can run first, the laplace
arm after it:
python checks/why_bayes/thin_joint_breach.py run --arm classical --bank BANK --out DIR --start 0 --stop 25
python checks/why_bayes/thin_joint_breach.py sample --checkpoint RUN --bank BANK --out DIR --start 0 --stop 100
python checks/why_bayes/thin_joint_breach.py run --arm posterior --bank BANK --out DIR --start 0 --stop 25
python checks/why_bayes/thin_joint_breach.py run --arm laplace --bank BANK --out DIR --start 0 --stop 25
python checks/why_bayes/thin_joint_breach.py summarize --out DIR [--arms classical posterior laplace]
(one run per core and row range)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from sbi4finance.calibration import bank, pricing, tasks, thin_products

import thin_joint

METHODS = ("posterior", "gauss_newton", "hurst_fixed", "flat_curve", "laplace", "laplace_delta")
ARM_METHODS = {"classical": ("gauss_newton", "hurst_fixed", "flat_curve"), "posterior": ("posterior",),
               "laplace": ("laplace", "laplace_delta")}
Z90 = 1.2815515655446004  # one-sided 90%
HURST_FIXED = 0.10
LAST_QUOTED = (63 / 252, 126 / 252)


def chain_quotes(data, task, row, seed):
    sd = thin_joint.noise(task)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(row)]))
    return data["reference_prices"][row] + sd * rng.standard_normal(len(sd))


def sample(args):
    manifest, splits = bank.load_bank(args.bank, splits=(args.split,))
    task = manifest["configuration"]["task"]
    data = splits[args.split]
    rows = np.arange(args.start, args.stop)
    quotes = np.stack([chain_quotes(data, task, r, args.seed) for r in rows])
    workflow = thin_joint.load(args.checkpoint, task)
    draws = np.concatenate([thin_joint.sample(workflow, task, quotes[s:s + 100], args.draws, seed=args.seed + 7 + s)
                            for s in range(0, len(rows), 100)])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"posterior-{args.start}-{args.stop}.npz", rows=rows, quotes=quotes, draws=draws,
             truth=np.asarray(data["parameters"], dtype=float)[rows])
    print(f"{len(rows)} chains x {args.draws} draws written")


class Fit:
    """Least squares in the logit coordinates of the prior box, with the Monte Carlo pricer on one seed. With
    prior=True the box's uniform density joins the objective, so the fit is the posterior's mode."""

    def __init__(self, task, quotes, paths, seed, fixed_hurst=None, prior=False):
        self.task, self.y, self.paths, self.seed, self.fixed = task, np.asarray(quotes, dtype=float), paths, seed, fixed_hurst
        self.box = thin_joint.bounds(task)
        self.free = [0, 1, 2, 3, 4, 5] if fixed_hurst is None else [1, 2, 3, 4, 5]
        self.sd = thin_joint.noise(task)
        self.prior = prior

    def theta(self, z):
        theta = np.empty(6)
        theta[self.free] = self.box[self.free, 0] + (self.box[self.free, 1] - self.box[self.free, 0]) * expit(z)
        if self.fixed is not None:
            theta[0] = self.fixed
        return theta

    def z(self, theta):
        unit = (np.asarray(theta, dtype=float)[self.free] - self.box[self.free, 0]) / (self.box[self.free, 1] - self.box[self.free, 0])
        return logit(np.clip(unit, 1e-4, 1 - 1e-4))

    def prices(self, theta):
        p, _, _, _ = pricing.rough_surface(tasks.model_parameters(self.task, theta), tasks.joint_curve(self.task, theta),
                                           self.task["context"]["contracts"], self.task["numerics"]["blocks"],
                                           [self.paths], self.sd, np.inf, self.seed, cap=np.inf)
        return p

    def residuals(self, z):
        r = (self.prices(self.theta(z)) - self.y) / self.sd
        if not self.prior:
            return r
        # Half the square of each extra residual is the logistic density's negative log, less its value at 0.
        excess = np.logaddexp(0.0, z) + np.logaddexp(0.0, -z) - 2 * np.log(2.0)
        return np.r_[r, np.sign(z) * np.sqrt(2 * np.maximum(excess, 0.0))]

    def chi2(self, theta):
        return float(np.sum(((self.prices(theta) - self.y) / self.sd) ** 2))

    def run(self, starts, max_nfev=40):
        """max_nfev counts iterations' evaluations only: scipy does not count the Jacobian's, about seven a step."""
        best = None
        for start in starts:
            r = least_squares(self.residuals, self.z(start), method="trf", diff_step=1e-3, x_scale="jac",
                              max_nfev=max_nfev, bounds=(-12.0, 12.0))
            if best is None or r.cost < best.cost:
                best = r
        jac = best.jac[: len(self.y)]
        curvature = jac.T @ jac
        if self.prior:
            u = expit(best.x)
            curvature += np.diag(2 * u * (1 - u))
        return {"z": best.x, "theta": self.theta(best.x), "chi2": float(np.sum(best.fun[: len(self.y)] ** 2)),
                "covariance": np.linalg.pinv(curvature), "condition": float(np.linalg.cond(curvature)),
                "on_bound": bool(np.any(np.abs(best.x) > 8)), "evaluations": int(best.nfev)}


def delta_bounds(fit, fitter, product_fn, step=1e-2):
    """The price at the fit and the one-sided 90% bounds of the delta method, by central differences in z."""
    z = fit["z"]
    base = product_fn(fit["theta"])
    gradient = np.empty((len(base), len(z)))
    for j in range(len(z)):
        up, down = z.copy(), z.copy()
        up[j] += step
        down[j] -= step
        gradient[:, j] = (product_fn(fitter.theta(up)) - product_fn(fitter.theta(down))) / (2 * step)
    sd = np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", gradient, fit["covariance"], gradient), 0.0))
    return base, base - Z90 * sd, base + Z90 * sd


def records(out, arm):
    return {r["row"]: r for f in sorted(out.glob(f"rows-{arm}-*.jsonl")) for r in thin_joint.read_rows(f)}


def posterior_draws(out):
    """The network's draws by row, from every sample file in out; a row may come from one file only."""
    files = sorted(out.glob("posterior-*.npz"))
    if not files:
        raise FileNotFoundError(f"no posterior-*.npz in {out}: run the sample step first")
    draws = {}
    for f in files:
        with np.load(f) as data:
            for row, d in zip(data["rows"], data["draws"]):
                if int(row) in draws:
                    raise ValueError(f"row {int(row)} is in more than one sample file")
                draws[int(row)] = d
    return draws


def run(args):
    """Skips a row that any run of the same arm has written, checked before each row, so row ranges may overlap."""
    manifest, _ = bank.load_bank(args.bank, splits=())
    task = manifest["configuration"]["task"]
    out_file = args.out / f"rows-{args.arm}-{args.start}-{args.stop}.jsonl"
    box = thin_joint.bounds(task)
    product = lambda theta, flat=None: thin_joint.products(task, theta, args.product_paths, args.seed + 1, flat)
    if args.arm == "posterior":
        post = posterior_draws(args.out)
    else:
        _, splits = bank.load_bank(args.bank, splits=(args.split,))
        data = splits[args.split]
    args.out.mkdir(parents=True, exist_ok=True)
    for row in range(args.start, args.stop):
        if row in records(args.out, args.arm):
            continue
        t0 = time.process_time()
        if args.arm == "posterior":
            if row not in post:
                continue
            priced = np.array([product(d) for d in post[row][: args.draws]])
            record = {"row": row, "posterior": {"lower": np.quantile(priced, 0.1, axis=0).tolist(),
                                                "upper": np.quantile(priced, 0.9, axis=0).tolist(),
                                                "median": np.median(priced, axis=0).tolist()},
                      "cpu_seconds": {"posterior": time.process_time() - t0}}
            thin_joint.append_row(out_file, record)
            print(f"row {row}: posterior {record['cpu_seconds']['posterior']:.0f} CPU-s", flush=True)
            continue
        if args.arm == "laplace":
            least = records(args.out, "classical").get(row)
            if least is None:
                continue
            fitter = Fit(task, chain_quotes(data, task, row, args.seed), args.fit_paths, args.seed + 2, prior=True)
            fit = fitter.run([fitter.theta(np.clip(fitter.z(least["gauss_newton"]["theta"]), -4.0, 4.0))])
            timings = {"fit": time.process_time() - t0}
            t0 = time.process_time()
            mark, lower, upper = delta_bounds(fit, fitter, product)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, 5, row]))
            z = rng.multivariate_normal(fit["z"], fit["covariance"], size=args.draws)
            priced = np.array([product(fitter.theta(zi)) for zi in z])
            timings["products"] = time.process_time() - t0
            record = {"row": row,
                      "laplace": {"lower": np.quantile(priced, 0.1, axis=0).tolist(),
                                  "upper": np.quantile(priced, 0.9, axis=0).tolist(),
                                  "median": np.median(priced, axis=0).tolist(), "chi2": fit["chi2"],
                                  "condition": fit["condition"], "on_bound": fit["on_bound"],
                                  "theta": fit["theta"].tolist(), "sd_z": np.sqrt(np.diag(fit["covariance"])).tolist(),
                                  "evaluations": fit["evaluations"]},
                      "laplace_delta": {"mark": mark.tolist(), "lower": lower.tolist(), "upper": upper.tolist()},
                      "cpu_seconds_laplace": timings}
            thin_joint.append_row(out_file, record)
            print(f"row {row}: laplace {sum(timings.values()):.0f} CPU-s; chi2 at the mode {fit['chi2']:.1f}", flush=True)
            continue
        quotes = chain_quotes(data, task, row, args.seed)
        truth = np.asarray(data["parameters"], dtype=float)[row]
        record = {"row": row, "truth_parameters": truth.tolist(), "truth": product(truth).tolist()}
        timings = {"truth": time.process_time() - t0}
        t0 = time.process_time()
        fitter = Fit(task, quotes, args.fit_paths, args.seed + 2)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 3, row]))
        candidates = np.vstack([box.mean(axis=1), box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((16, 6))])
        chi2 = np.array([fitter.chi2(c) for c in candidates])
        order = np.argsort(chi2)
        fit = fitter.run(candidates[order[:2]])
        fixed = Fit(task, quotes, args.fit_paths, args.seed + 2, fixed_hurst=HURST_FIXED)
        fit_h = fixed.run([np.r_[HURST_FIXED, fit["theta"][1:]]])
        timings["fits"] = time.process_time() - t0
        t0 = time.process_time()
        for name, f, fn, flat in (("gauss_newton", fit, fitter, None), ("hurst_fixed", fit_h, fixed, None),
                                  ("flat_curve", fit, fitter, LAST_QUOTED)):
            mark, lower, upper = delta_bounds(f, fn, lambda theta: product(theta, flat))
            record[name] = {"mark": mark.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
                            "chi2": f["chi2"], "condition": f["condition"], "on_bound": f["on_bound"],
                            "theta": f["theta"].tolist(), "evaluations": f["evaluations"]}
        timings["delta"] = time.process_time() - t0
        record["cpu_seconds"] = timings
        thin_joint.append_row(out_file, record)
        print(f"row {row}: {sum(timings.values()):.0f} CPU-s; GN chi2 {fit['chi2']:.1f}, H fixed {fit_h['chi2']:.1f}",
              flush=True)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def summarize(args):
    """Joins the arms' records by row; only rows that every arm in --arms has enter the table. The split by the true
    H, within 0.05 of 0.10 or not, is a diagnostic of the fixed-H baseline, not a stratum a desk could observe."""
    by_arm = {arm: records(args.out, arm) for arm in args.arms}
    rows = sorted(set.intersection(*(set(found) for found in by_arm.values())))
    joined = [{k: v for arm in args.arms for k, v in by_arm[arm][r].items()} for r in rows]
    methods = [m for m in METHODS if any(m in ARM_METHODS[arm] for arm in args.arms)]
    truth = np.array([r["truth"] for r in joined])
    near = np.abs(np.array([r["truth_parameters"][0] for r in joined]) - HURST_FIXED) <= 0.05
    n = len(joined)
    report = {"chains": n, "arms": args.arms, "products": list(thin_products.NAMES), "nominal_breach_rate": 0.1,
              "true_hurst_near_0.10": int(near.sum()), "methods": {}}
    print(f"{n} chains; breach rates of the one-sided 90% bounds (nominal 10% each side)")
    print(f"  {'product':22s} " + " ".join(f"{m:>22s}" for m in methods))
    for m in methods:
        lower = np.array([r[m]["lower"] for r in joined])
        upper = np.array([r[m]["upper"] for r in joined])
        long_hit, short_hit = truth < lower, truth > upper
        long_b, short_b = long_hit.sum(axis=0), short_hit.sum(axis=0)
        width = np.median((upper - lower) / np.abs(truth), axis=0)
        report["methods"][m] = {"long_breaches": long_b.tolist(), "short_breaches": short_b.tolist(),
                                "long_wilson": [wilson(k, n) for k in long_b], "short_wilson": [wilson(k, n) for k in short_b],
                                "median_relative_width": width.tolist(),
                                "by_true_hurst": {name: {"chains": int(mask.sum()),
                                                         "long_breaches": long_hit[mask].sum(axis=0).tolist(),
                                                         "short_breaches": short_hit[mask].sum(axis=0).tolist()}
                                                  for name, mask in (("near_0.10", near), ("far", ~near))}}
    for j, p in enumerate(thin_products.NAMES):
        cells = []
        for m in methods:
            r = report["methods"][m]
            cells.append(f"{r['long_breaches'][j] / n:5.2f}/{r['short_breaches'][j] / n:4.2f} w{r['median_relative_width'][j]:5.3f}")
        print(f"  {p:22s} " + " ".join(f"{c:>22s}" for c in cells))
    for m in ("gauss_newton", "hurst_fixed", "laplace"):
        if m in methods:
            on_bound = sum(r[m]["on_bound"] for r in joined)
            report["methods"][m]["fits_on_bound"] = on_bound
            print(f"  {m}: fits with a coordinate on the prior bound {on_bound} of {n}")
    (args.out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("sample", sample), ("run", run), ("summarize", summarize)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--out", type=Path, required=True)
        if name != "summarize":
            p.add_argument("--bank", type=Path, required=True)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int, default=100)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--draws", type=int, default=200)
        if name != "summarize":
            p.add_argument("--split", default="test")
        if name == "sample":
            p.add_argument("--checkpoint", type=Path, required=True)
        if name == "summarize":
            p.add_argument("--arms", nargs="+", choices=list(ARM_METHODS), default=["classical", "posterior"])
        if name == "run":
            p.add_argument("--arm", choices=list(ARM_METHODS), required=True,
                           help="classical needs only the bank; posterior needs the sample step's draws; laplace "
                                "starts from the classical arm's fits")
            p.add_argument("--product-paths", type=int, default=8192)
            p.add_argument("--fit-paths", type=int, default=4096)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
