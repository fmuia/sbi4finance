"""Reserves on simulated thin chains and how often each method's reserve is breached (study S10).

    python lab/studies/thin_coverage.py run --bank BANK --checkpoint S2T_RUN --out DIR \
        --start 3000 --stop 3100 [--proposals 512 --rounds 2 --audit-paths 4096 --product-draws 256 \
        --product-paths 8192 --fit-paths 8192 --screen 16 --seed-check-every 50 --workers 4 --seed 0]
    python lab/studies/thin_coverage.py run --truths S11_DIR/draws.npz --checkpoint S2T_RUN --out DIR2 \
        --start 0 --stop 100 [same options]
    python lab/studies/thin_coverage.py summarize --out DIR

Each test surface of rows [start, stop) becomes a thin chain: its clean prices on the network's layout (the EWG
layout) plus a fresh quote error with the layout's noise. The truth (H, eta, rho and the forward-variance curve)
is the surface's own; no method is given the curve. For each chain and each of the six thin products
(sbi4finance.calibration.thin_products), every method gives a centre and one-sided 90% bounds:

  network       the thin network's draws alone: products at product_draws of them;
  posterior     the audit (thin.importance_rounds): proposals from the draws (Student t, 10% prior), each priced
                on the layout with its own seed, the unbiased likelihood N(y; P_hat, diag(noise^2) + C - S) with the
                proposal's Monte Carlo covariance S and the bank's C (the chain's clean prices are the bank's Monte
                Carlo prices); a second round refitted to the weights when k-hat exceeds 0.7; products
                at product_draws resampled by the weights; centre the median, bounds the 10% and 90% quantiles;
  gauss_newton  least squares on (H, eta, rho) and one forward-variance level per quoted maturity interval, flat
                beyond the last quote; started at the lowest chi-square of the prior box's centre and screen prior
                draws (the network is not used); the delta method gives each product's sd, and the bounds are the
                fit's price -/+ 1.2816 sd;
  gauss_newton_power  the same fit, with the levels beyond the last quote extrapolated by a power law in time
                fitted to the quoted levels (thin.power_law_extension) instead of flat;
  h_fixed       the Gauss-Newton fit with H fixed at 0.10, started from the Gauss-Newton fit.

The truth and every method price the products on one seed per chain and the same path count, so the comparison
is not blurred by product Monte Carlo error; one chain in seed_check_every also gives the posterior's bounds on a
second product seed, to measure that error.

Only the forward start from six months to one year depends on the curve beyond the last quote (126 days). Its
truth under the prior law is correct for the posterior by construction, and its flat truth is correct for the
flat Gauss-Newton fit by construction. So each chain also prices that truth under two other laws for the last
level, with the quotes unchanged (they end at 126 days): "flat" (the last level equal to the one before) and
"wide" (the prior's increment over the last interval doubled in log variance, clipped to vol_bounds). The
posterior's prior is not changed.

With --truths (thin_real.py's draws.npz), the truths are resampled from EWG's audited posterior instead, and their
layout quotes are priced as the bank prices; the posterior's prior is then not the truths' law, so its coverage
is not guaranteed by construction: the run tests every method near the real chain. Resampled truths repeat when
the posterior's effective sample size is small; summarize reports the number of distinct truths and a bootstrap
over them, since chains sharing a truth are not independent. Run summarize on its own directory.

A long breach is a true price below the lower bound, a short breach one above the upper bound; a correct method
breaches each side 10% of the time. Each finished chain is appended to DIR/rows-START-STOP.jsonl, so a rerun
continues where it stopped; summarize pools every rows-*.jsonl in DIR into coverage.json and coverage.png, on all
chains and on the chains whose audit is reliable (final k-hat at most 0.7 and every proposal priced). Test rows
3000 to 3999 are S10's
(lab/studies/README.md).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank, correction, thin, thin_products  # noqa: E402

ONE_SIDED = 1.2815515655446004
H_FIXED = 0.10
KHAT_RELIABLE = 0.7
_TAG = 0x53313030  # "S100"
METHODS = ("network", "posterior", "gauss_newton", "gauss_newton_power", "h_fixed")
COLOURS = {"network": "#2a78d6", "posterior": "#eb6834", "gauss_newton": "#1baf7a", "gauss_newton_power": "#7a5af5",
           "h_fixed": "#555555"}  # the first three are the course's validated series colours
ARMS = ("prior", "flat", "wide")
LONG_PRODUCT = "forward_start_6m_1y"


def read_jsonl(path):
    """The records of a JSON-lines file; a line cut short by a killed job is skipped and reported."""
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"{path}: line {number} is incomplete (a job was killed while writing it) and is skipped",
                      file=sys.stderr, flush=True)
    return records


def end_with_newline(path):
    """Before appending: a file cut short mid-line gets a newline, so the next record starts on its own line."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with path.open("a") as f:
                f.write("\n")


def check_settings(out, settings):
    """A directory pools chains of one configuration only: the first run writes settings.json, later runs compare."""
    path = Path(out) / "settings.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != settings:
            changed = sorted(k for k in set(stored) | set(settings) if stored.get(k) != settings.get(k))
            raise ValueError(f"{out} holds chains run with other settings ({', '.join(changed)}): use a new --out")
        return
    temporary = path.with_name(f"settings.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(settings, indent=1) + "\n")
    temporary.replace(path)


def chain_seed(seed, row, purpose):
    return int(np.random.SeedSequence([_TAG, int(seed), int(row), int(purpose)]).generate_state(1, dtype=np.uint32)[0])


def quantile_bounds(values):
    q = np.quantile(np.asarray(values, dtype=float), [0.1, 0.5, 0.9], axis=0)
    return {"lower": q[0].tolist(), "centre": q[1].tolist(), "upper": q[2].tolist()}


def product_prices(task, rows, paths, seed):
    """Prices [N, 6] of the products at each row of targets, common random numbers (one seed)."""
    return np.array([thin_products.price(task, r[:3], r[3:], paths, seed)[0] for r in rows])


def truth_arms(task, truth):
    """The truth under the three laws for the levels beyond the layout's last maturity (126 days)."""
    p = len(task["parameter_names"])
    low, high = 2 * np.log(np.asarray(task["context_prior"]["forward_variance"]["vol_bounds"], dtype=float))
    flat, wide = truth.copy(), truth.copy()
    flat[-1] = truth[-2]
    wide[-1] = np.clip(truth[-2] + 2 * (truth[-1] - truth[-2]), low, high)
    assert len(truth) == p + 7
    return {"prior": truth, "flat": flat, "wide": wide}


def classical_bounds(task, fit, product_fn, row_fn=None):
    centre, sd = thin.delta_method(task, fit, product_fn if row_fn is None else (lambda r: product_fn(row_fn(r))))
    return {"lower": (centre - ONE_SIDED * sd).tolist(), "centre": centre.tolist(),
            "upper": (centre + ONE_SIDED * sd).tolist()}


def chain_job(job):
    """One chain; an error is recorded with the chain's row instead of stopping the run."""
    try:
        return chain_job_inner(job)
    except Exception as error:  # noqa: BLE001 - every failure is reported in the summary
        return {"test_row": int(job[2]), "failed": f"{type(error).__name__}: {error}"}


def chain_job_inner(job):
    (task, layout, row, truth, clean, clean_cov, draws, args, seed) = job
    started = time.process_time()
    timings = {}
    idx, noise = thin.layout_arrays(layout)
    contracts = thin.layout_contracts(task, layout)
    days = thin.contract_days(contracts)
    rng = np.random.default_rng(chain_seed(seed, row, 0))
    y = clean + noise * rng.standard_normal(len(noise))
    product_seed = chain_seed(seed, row, 1)
    out = {"test_row": int(row)}

    def products_of(rows, product_seed=product_seed):
        return product_prices(task, rows, args["product_paths"], product_seed)

    # the network alone
    t0 = time.process_time()
    pick = np.random.default_rng(chain_seed(seed, row, 2)).choice(len(draws), args["product_draws"], replace=False)
    out["network"] = quantile_bounds(products_of(draws[pick]))
    timings["network_products"] = time.process_time() - t0

    # the audit: each proposal on its own seed, a second round if k-hat > 0.7
    t0 = time.process_time()
    price_one = partial(thin.priced_for_audit, task=task, contracts=contracts, noise=noise,
                        paths=args["audit_paths"])
    audit = thin.importance_rounds(draws, task, price_one, y, noise, args["proposals"], chain_seed(seed, row, 3),
                                   max_rounds=args["rounds"], khat_target=KHAT_RELIABLE, data_covariance=clean_cov)
    timings["audit"] = time.process_time() - t0
    t0 = time.process_time()
    picks = correction.resample(audit["targets"], audit["weights"], args["product_draws"], chain_seed(seed, row, 4))
    out["posterior"] = quantile_bounds(products_of(picks))
    out["posterior"].update(ess=audit["ess"], efficiency=audit["efficiency"], khat=audit["khat"],
                            rounds=audit["rounds"], unpriced_proposals=audit["unpriced"])
    if args["seed_check"]:
        out["posterior_second_seed"] = quantile_bounds(products_of(picks, chain_seed(seed, row, 7)))
    timings["posterior_products"] = time.process_time() - t0

    # the classical fits: a start from the quotes and the prior box, never from the network
    t0 = time.process_time()
    starts, chi2_starts = thin.classical_screen(task, y, contracts, noise, days, args["fit_paths"],
                                                chain_seed(seed, row, 5), candidates=args["screen"])
    fit = thin.classical_fit(task, y, contracts, noise, starts[0], days, args["fit_paths"], chain_seed(seed, row, 5))
    if not fit["converged"]:
        second = thin.classical_fit(task, y, contracts, noise, starts[1], days, args["fit_paths"],
                                    chain_seed(seed, row, 5))
        fit = second if second["chi2"] < fit["chi2"] else fit
    fixed_start = fit["row"].copy()
    fixed_start[task["parameter_names"].index("hurst")] = H_FIXED
    fit_h = thin.classical_fit(task, y, contracts, noise, fixed_start, days, args["fit_paths"],
                               chain_seed(seed, row, 5), fixed_hurst=H_FIXED)
    timings["classical_fits"] = time.process_time() - t0
    t0 = time.process_time()
    for name, f, row_fn in (("gauss_newton", fit, None),
                            ("gauss_newton_power", fit, lambda r: thin.power_law_extension(task, r, days)),
                            ("h_fixed", fit_h, None)):
        out[name] = classical_bounds(task, f, lambda r: products_of([r])[0], row_fn)
        out[name].update(chi2=f["chi2"], converged=f["converged"], nfev=f["nfev"], fit=f["row"].tolist())
    out["gauss_newton"]["screen_chi2"] = chi2_starts[:3].tolist()
    timings["classical_products"] = time.process_time() - t0

    # the truth on the methods' seed and paths, under three laws for the last level; and on an independent seed
    t0 = time.process_time()
    arms = truth_arms(task, truth)
    out["truth"] = {"targets": truth.tolist(),
                    "products": {arm: products_of([arms[arm]])[0].tolist() for arm in ARMS},
                    "products_independent_seed": products_of([truth], chain_seed(seed, row, 6))[0].tolist()}
    timings["truth"] = time.process_time() - t0
    out["cpu_seconds"] = {**timings, "total": time.process_time() - started}
    return out


TRUTH_ROW_OFFSET = 1_000_000  # chain identities of truths drawn from a posterior, apart from the bank's test rows
TRUTH_POOL = 10_000  # truths resampled once from the posterior; --start and --stop slice this list


def price_truth(job):
    """A truth's clean layout prices at the bank's precision (budget 0.25 of the design noise) and their covariance."""
    task, truth, contracts, sigma, seed = job
    numerics = task["numerics"]
    prices, _, cov = thin.price_contracts(task, truth, contracts, sigma, numerics["paths_ladder"], seed,
                                          budget=numerics["max_mc_se_over_quote_sigma"],
                                          cap=numerics["max_mc_se_over_quote_sigma_last_rung"], covariance=True)
    return prices, cov


def posterior_truths(task, layout, path, start, stop, seed, workers):
    """Truths resampled from a real chain's audited posterior (thin_real.py's draws.npz, noise scale 1), each
    with its layout quotes priced as the bank would price them: the coverage near that chain, where the prior is
    not the truth's law."""
    if not 0 <= start < stop <= TRUTH_POOL:
        raise ValueError(f"--start and --stop must index the first {TRUTH_POOL} resampled truths")
    saved = np.load(path)
    targets, weights = saved["targets_scale_1"], saved["weights_scale_1"]
    picks = correction.resample(targets, weights, TRUTH_POOL, chain_seed(seed, TRUTH_ROW_OFFSET, 8))[start:stop]
    contracts = thin.layout_contracts(task, layout)
    idx, _ = thin.layout_arrays(layout)
    from sbi4finance.calibration import tasks as task_tools

    sigma = task_tools.sigma(task)[idx]
    jobs = [(task, t, contracts, sigma, chain_seed(seed, TRUTH_ROW_OFFSET + start + j, 9)) for j, t in enumerate(picks)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        priced = list(pool.map(price_truth, jobs))
    clean = np.array([p for p, _ in priced])
    clean_cov = np.array([c for _, c in priced])
    rows = [TRUTH_ROW_OFFSET + start + j for j in range(len(picks))]
    return rows, picks, clean, clean_cov


def run(args):
    from evaluate_thin import load_thin_network

    task, layout, workflow, record = load_thin_network(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731
    check_settings(args.out, {
        "checkpoint_sha256": digest(args.checkpoint / "model.keras"),
        "truths_sha256": digest(args.truths) if args.truths else None,
        "bank": None if args.truths else str(args.bank.resolve()),
        **{k: getattr(args, k) for k in ("draws", "proposals", "rounds", "audit_paths", "product_draws",
                                         "product_paths", "fit_paths", "screen", "seed_check_every", "seed")}})
    tag = "truths-" if args.truths else ""
    log_path = args.out / f"rows-{tag}{args.start}-{args.stop}.jsonl"
    done = set()
    if log_path.exists():
        done = {r["test_row"] for r in read_jsonl(log_path)}
        end_with_newline(log_path)
    idx, noise = thin.layout_arrays(layout)
    if args.truths:
        rows, truth, clean, clean_cov = posterior_truths(task, layout, args.truths, args.start, args.stop, args.seed,
                                                         args.workers)
        keep = [i for i, r in enumerate(rows) if r not in done]
        pending, truth, clean, clean_cov = [rows[i] for i in keep], truth[keep], clean[keep], clean_cov[keep]
    else:
        manifest, splits = bank.load_bank(args.bank, splits=("test",))
        if task != manifest["configuration"]["task"]:
            raise ValueError("checkpoint and bank tasks differ")
        test = splits["test"]
        rows = np.arange(args.start, min(args.stop, len(test["parameters"])))
        pending = [int(r) for r in rows if int(r) not in done]
        clean = test["reference_prices"][pending][:, idx]
        clean_cov = test["pricing_covariance"][pending][:, idx][:, :, idx]  # the reference prices' Monte Carlo error
        truth = np.hstack([test["parameters"][pending], np.log(test["context_values"][pending])])
    if not pending:
        print("nothing to do")
        return
    # the network's input: the same simulated chain as chain_job draws (same seed and stream)
    quotes = np.array([c + noise * np.random.default_rng(chain_seed(args.seed, r, 0)).standard_normal(len(noise))
                       for c, r in zip(clean, pending)])
    draws = np.concatenate([thin.sample_thin(workflow, task, layout, quotes[s:s + 100], args.draws,
                                             seed=args.seed + s) for s in range(0, len(pending), 100)])
    settings = {k: getattr(args, k) for k in ("proposals", "rounds", "audit_paths", "product_draws", "product_paths",
                                             "fit_paths", "screen")}
    jobs = [(task, layout, r, truth[i], clean[i], clean_cov[i], draws[i],
             {**settings, "seed_check": int(r) % args.seed_check_every == 0}, args.seed) for i, r in enumerate(pending)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool, \
            log_path.open("a") as log:
        for result in pool.map(chain_job, jobs):
            log.write(json.dumps(result) + "\n")
            log.flush()
            if "failed" in result:
                print(f"row {result['test_row']}: FAILED {result['failed']}", flush=True)
            else:
                print(f"row {result['test_row']}: {result['cpu_seconds']['total']:.0f} CPU-s, "
                      f"k-hat {result['posterior']['khat']:.2f}", flush=True)


def finite(value):
    """Non-finite numbers (k-hat with a short tail, a Wilson interval on no chains) become null."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    return value


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def cluster_interval(breached, clusters, draws=2000, seed=0):
    """95% interval of the breach rate per product by a bootstrap over clusters (chains sharing a truth).

    Truths resampled from a posterior repeat, and chains with the same truth are not independent, so the Wilson
    interval, which assumes independent chains, is too narrow; resampling whole clusters keeps their dependence."""
    breached = np.asarray(breached, dtype=float)
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.zeros((len(labels), breached.shape[1]))
    np.add.at(sums, inverse, breached)
    sizes = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(labels), (draws, len(labels)))
    rates = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)[:, None]
    return np.quantile(rates, [0.025, 0.975], axis=0).T.tolist()


def breach_table(records, truth, clusters=None):
    n = len(records)
    table = {}
    for m in METHODS:
        lower = np.array([r[m]["lower"] for r in records])
        upper = np.array([r[m]["upper"] for r in records])
        centre = np.array([r[m]["centre"] for r in records])
        long_b, short_b = (truth < lower).sum(axis=0), (truth > upper).sum(axis=0)
        table[m] = {"long_breaches": long_b.tolist(), "short_breaches": short_b.tolist(),
                    "long_breach_rate": (long_b / n).tolist(), "short_breach_rate": (short_b / n).tolist(),
                    "long_wilson": [wilson(k, n) for k in long_b], "short_wilson": [wilson(k, n) for k in short_b],
                    "median_width_over_price": np.median((upper - lower) / np.abs(truth), axis=0).tolist(),
                    "median_abs_centre_error_over_price": np.median(np.abs(centre - truth) / np.abs(truth),
                                                                    axis=0).tolist()}
        if clusters is not None and len(set(clusters)) < n:
            table[m]["long_cluster_bootstrap"] = cluster_interval(truth < lower, clusters)
            table[m]["short_cluster_bootstrap"] = cluster_interval(truth > upper, clusters)
    return table


def truth_clusters(records):
    """One label per chain; chains whose truths are identical share it."""
    keys = [tuple(np.round(r["truth"]["targets"], 12)) for r in records]
    index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return [index[k] for k in keys]


def summarize(args):
    records = []
    for path in sorted(args.out.glob("rows-*.jsonl")):
        records += read_jsonl(path)
    failed = [r for r in records if "failed" in r]
    records = [r for r in records if "failed" not in r]
    if not records:
        raise ValueError("no finished chains found")
    records.sort(key=lambda r: r["test_row"])
    khat = np.array([np.inf if r["posterior"]["khat"] is None else r["posterior"]["khat"] for r in records], dtype=float)
    # Reliable: a usable tail (k-hat) and every proposal priced (an unpriced proposal's zero weight biases the audit).
    reliable = [r for r, k in zip(records, khat)
                if k <= KHAT_RELIABLE and r["posterior"]["unpriced_proposals"] == 0]
    clusters = truth_clusters(records)
    report = {"study": "S10", "chains": len(records), "reliable_chains": len(reliable),
              "distinct_truths": len(set(clusters)),
              "failed_chains": [{"test_row": r["test_row"], "error": r["failed"]} for r in failed],
              "test_rows": [records[0]["test_row"], records[-1]["test_row"] + 1],
              "products": list(thin_products.NAMES), "nominal_breach_rate": 0.1, "arms": list(ARMS),
              "reliable_means": f"final k-hat at most {KHAT_RELIABLE} and no unpriced proposal",
              "intervals": ("Wilson 95% intervals assume independent chains; when truths repeat (truths resampled "
                            "from a posterior), the cluster bootstrap over distinct truths is the one to read"),
              "all": {}, "reliable": {}}
    for label, subset in (("all", records), ("reliable", reliable)):
        subset_clusters = truth_clusters(subset) if subset else None
        for arm in ARMS:
            if subset:
                truth = np.array([r["truth"]["products"][arm] for r in subset])
                report[label][arm] = breach_table(subset, truth, subset_clusters)
    report["audit"] = {"khat_quantiles": dict(zip(("q50", "q90", "max"), np.quantile(khat, [0.5, 0.9, 1.0]).tolist())),
                       "share_khat_above_0_7": float(np.mean(khat > KHAT_RELIABLE)),
                       "share_with_second_round": float(np.mean([len(r["posterior"]["rounds"]) > 1 for r in records])),
                       "ess_median": float(np.median([r["posterior"]["ess"] for r in records])),
                       "unpriced_proposals_total": int(sum(r["posterior"]["unpriced_proposals"] for r in records))}
    report["classical"] = {"gauss_newton_not_converged": int(sum(not r["gauss_newton"]["converged"] for r in records)),
                           "h_fixed_not_converged": int(sum(not r["h_fixed"]["converged"] for r in records))}
    checked = [r for r in records if "posterior_second_seed" in r]
    if checked:
        a = np.array([[r["posterior"]["lower"], r["posterior"]["upper"]] for r in checked])
        b = np.array([[r["posterior_second_seed"]["lower"], r["posterior_second_seed"]["upper"]] for r in checked])
        width = np.array([np.array(r["posterior"]["upper"]) - np.array(r["posterior"]["lower"]) for r in checked])
        change = np.abs(a - b).max(axis=1)
        ratio = np.divide(change, width, out=np.full_like(change, np.nan), where=width > 0)
        report["product_seed_check"] = {
            "chains": len(checked), "chains_with_zero_width": int(np.sum(np.any(width <= 0, axis=1))),
            "median_bound_change_over_width": [float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else None
                                               for c in ratio.T]}
    report["cpu_seconds_per_chain_median"] = float(np.median([r["cpu_seconds"]["total"] for r in records]))
    (args.out / "coverage.json").write_text(json.dumps(finite(report), indent=2, allow_nan=False) + "\n")
    figure(report, args.out / "coverage.png")
    print(json.dumps({m: {"long": np.round(v["long_breach_rate"], 3).tolist(),
                          "short": np.round(v["short_breach_rate"], 3).tolist()}
                      for m, v in report["all"]["prior"].items()}, indent=1))


def figure(report, path):
    from scipy.stats import binom
    from sbi4finance.calibration.plots import INK, plt, style

    style()
    n = report["chains"]
    lo, hi = binom.ppf(0.025, n, 0.1) / n, binom.ppf(0.975, n, 0.1) / n
    labels = {"network": "network alone", "posterior": "posterior (audited)", "gauss_newton": "Gauss–Newton, flat",
              "gauss_newton_power": "Gauss–Newton, power law", "h_fixed": "H fixed at 0.10"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.2), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [2.2, 1.0]})
    marks = ["o", "s", "v", "D", "P", "X"]
    for ax in (left, right):
        ax.axhspan(lo, hi, color="0.9", zorder=0)
        ax.axhline(0.1, color="0.3", lw=0.8)
    table = report["all"]["prior"]
    for i, m in enumerate(METHODS):
        for s, side in enumerate(("long", "short")):
            for j, product in enumerate(report["products"]):
                x = i + (s - 0.5) * 0.4 + (j - 2.5) * 0.05
                rate = table[m][f"{side}_breach_rate"][j]
                left.scatter(x, rate, marker=marks[j], s=30, edgecolor=COLOURS[m],
                             facecolor=COLOURS[m] if side == "long" else "white", lw=1.1, zorder=3)
    left.set_xticks(range(len(METHODS)))
    left.set_xticklabels([labels[m] for m in METHODS], fontsize=8)
    left.set_ylabel("share of chains whose truth breaches the 90% bound")
    left.legend(handles=[plt.Line2D([], [], marker=mk, ls="", color="0.3", label=p.replace("_", " "))
                         for mk, p in zip(marks, report["products"])]
                + [plt.Line2D([], [], marker="o", ls="", color="0.3", label="long side filled, short side open")],
                fontsize=7, frameon=False, loc="upper left")
    left.set_title(f"All six products, truths from the prior: {n} simulated chains with the EWG layout", loc="left",
                   fontsize=9.5, color=INK)
    j = report["products"].index(LONG_PRODUCT)
    for a, arm in enumerate(ARMS):
        for i, m in enumerate(METHODS):
            for s, side in enumerate(("long", "short")):
                x = a + (i - 2) * 0.14 + (s - 0.5) * 0.05
                rate = report["all"][arm][m][f"{side}_breach_rate"][j]
                right.scatter(x, rate, marker="o", s=26, edgecolor=COLOURS[m],
                              facecolor=COLOURS[m] if side == "long" else "white", lw=1.1, zorder=3,
                              label=labels[m] if (a == 0 and s == 0) else None)
    right.set_xticks(range(len(ARMS)))
    right.set_xticklabels(["last level from the prior", "last level flat", "last increment doubled"], fontsize=8)
    right.legend(fontsize=7, frameon=False, loc="upper left")
    right.set_title("Forward start 6m to 1y: three laws for the unquoted level", loc="left", fontsize=9.5,
                    color=INK)
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--bank", type=Path, help="the bank whose test rows are the truths (or give --truths)")
    p.add_argument("--truths", type=Path, help="thin_real.py's draws.npz: truths resampled from that posterior; "
                                               "--start and --stop then index the resampled truths")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--proposals", type=int, default=512)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--audit-paths", type=int, default=4096)
    p.add_argument("--product-draws", type=int, default=256)
    p.add_argument("--product-paths", type=int, default=8192)
    p.add_argument("--fit-paths", type=int, default=8192)
    p.add_argument("--screen", type=int, default=16)
    p.add_argument("--seed-check-every", type=int, default=50,
                   help="chains whose row is a multiple of this are also priced on a second product seed")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run" and (args.bank is None) == (args.truths is None):
        parser.error("give exactly one of --bank and --truths")
    run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    main()
