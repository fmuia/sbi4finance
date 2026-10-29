"""Check the numbers of course/v2-plan.md, and of the lectures' statement of the argument, against the stored
results.

    python course/scripts/check_v2_plan.py

Each check reads a committed file and compares it with the value the plan prints, at the plan's precision.
The script prints one line per check and exits with status 1 if any check fails.
"""

from datetime import date
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))
from sbi4finance.calibration import chain  # noqa: E402

CHECKS = ROOT / "lab" / "checks"
RESULTS = ROOT / "tutorial" / "assets" / "results" / "calibration"
FAILED = []


def load(path):
    return json.loads(Path(path).read_text())


def check(label, value, printed, tol):
    ok = abs(value - printed) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: stored {value:.4g}, printed {printed}")
    if not ok:
        FAILED.append(label)


def section_1():
    layout = load(ROOT / "lab/configs/thin/ewg_layout.json")
    observation = load(CHECKS / "thin_chains/EWG_observation.json")
    diag = observation["diagnostics"]
    april = next(r for r in layout["expiry_map"].values() if r["expiry"] == "2027-04-16")
    check("EWG April expiry: NYSE trading days", april["trading_days"], 120, 0)
    check("EWG April expiry: weekdays", diag["weekdays_to_each_expiry"]["2027-04-16"], 126, 0)
    start = date(2026, 10, 21)
    for end, printed in ((date(2027, 1, 15), 5), (date(2027, 4, 16), 5), (date(2027, 10, 15), 4)):
        excess = 100 * (chain.trading_days(start, end) / chain.trading_days(start, end, chain.NYSE_HOLIDAYS) - 1)
        check(f"SPY's weekday clock to {end}: excess (%)", excess, printed, 0.5)
    settings = load(CHECKS / "thin_chains/settings.json")
    check("rate (%)", 100 * settings["EWG"]["rate"], 4.22, 1e-9)
    gaps = [100 * (1 - v["parity"] / v["carry"]) for v in diag["forward_by_design_maturity"].values()]
    check("parity forward below carry, smallest (%)", min(gaps), 0.15, 0.005)
    check("parity forward below carry, largest (%)", max(gaps), 0.46, 0.005)
    moves = [abs(q["half_spreads"]) for q in diag["other_forward_moves_quotes"]]
    check("parity forward: largest move of a quote (half-spreads)", max(moves), 1.24, 0.005)
    puts = [q["vol_points"] for q in diag["other_forward_moves_quotes"] if q["kind"] == "put"]
    calls = [q["vol_points"] for q in diag["other_forward_moves_quotes"] if q["kind"] == "call"]
    check("parity forward moves puts down", float(np.mean(puts) < 0), 1, 0)
    check("parity forward moves calls up", float(np.mean(calls) > 0), 1, 0)
    ewg = load(CHECKS / "thin_chains/EWG_2026-10-22.json")
    premia = {e["days"]: max(q["early_exercise_over_half_spread"] for q in e["quotes"]
                             if q["two_sided"] and q["cp"] == "P") for e in ewg["expiries"] if e["days"] <= 176}
    check("EWG puts: largest early-exercise value (half-spreads)", max(premia.values()), 0.31, 0.005)
    check("... at the six-month expiry", float(max(premia, key=premia.get) == 176), 1, 0)
    calls_ee = [q["early_exercise_over_half_spread"] for e in ewg["expiries"] if e["days"] <= 176
                for q in e["quotes"] if q["two_sided"] and q["cp"] == "C"]
    check("EWG calls: early-exercise value", max(calls_ee), 0.0, 1e-9)
    check("vega at 20% understates the half-spread, largest factor", max(diag["vega_at_own_vol_over_vega_at_sigma_ref"]),
          3.6, 0.05)
    report = load(CHECKS / "grid_bias_minipilot/grid-bias.json")
    grid = report["summary"]
    check("grid bias check: surfaces", grid["surfaces"], 12, 0)
    q = grid["abs_difference_over_sigma_quantiles"]
    check("grid bias: median |move| (design sd)", q["q50"], 0.02, 0.005)
    check("grid bias: 90th percentile", q["q90"], 0.15, 0.005)
    check("grid bias: largest", q["max"], 1.0, 0.05)
    worst = max(report["surfaces"], key=lambda r: max(abs(v) for v in r["difference_over_sigma"]))
    check("grid bias: the worst surface's eta", worst["parameters"][1], 3.4, 0.05)
    check("grid bias: the worst surface's rho", worst["parameters"][2], -0.96, 0.005)
    return grid


def section_2():
    ev = load(ROOT / "course/images/v2-evidence-numbers.json")
    cases = ev["classical_vs_posterior"]
    heston = [p["ratio"] for c in cases if c["case"].startswith("Heston") for p in c["points"]]
    check("Heston ratios, lowest", min(heston), 0.92, 0.005)
    check("Heston ratios, highest", max(heston), 1.21, 0.005)
    rough = [p["ratio"] for c in cases if c["case"] == "rough Bergomi 63 quotes" for p in c["points"]]
    check("rough 63 quotes, lowest", min(rough), 0.93, 0.005)
    check("rough 63 quotes, highest", max(rough), 1.16, 0.005)
    # the same ratios at the course's reserve, the 10% and 90% quantiles, from the stored draws
    arrays = np.load(CHECKS / "rough_spy_is/rough_spy_is_seed0.npz")
    prod, weights, mark = arrays["products"], arrays["w_post"], arrays["product_mark"]
    grad, cov = arrays["product_gradient"], arrays["cov"]
    sd = np.sqrt(np.einsum("pi,ij,pj->p", grad, cov, grad))
    ratios90 = []
    for k in np.flatnonzero(sd > 0):
        order = np.argsort(prod[:, k])
        cdf = np.cumsum(weights[order]) / np.sum(weights)
        q10, q90 = (prod[order[min(np.searchsorted(cdf, a), len(cdf) - 1)], k] for a in (0.10, 0.90))
        ratios90 += [1.2815516 * sd[k] / (mark[k] - q10), 1.2815516 * sd[k] / (q90 - mark[k])]
    check("rough 63 quotes at the 90% quantile, lowest", min(ratios90), 0.94, 0.005)
    check("rough 63 quotes at the 90% quantile, highest", max(ratios90), 1.13, 0.005)
    flat = next(c for c in cases if "flat" in c["case"])
    nonzero = [p["ratio"] for p in flat["points"] if p["ratio"] > 0]
    check("long end, flat: lowest non-zero", min(nonzero), 0.30, 0.005)
    check("long end, flat: highest", max(nonzero), 0.67, 0.005)
    vs = next(p["ratio"] for p in flat["points"] if p["product"] == "variance swap 3y")
    check("long end, flat: three-year variance swap", vs, 0.0, 1e-12)
    extra = load(CHECKS / "long_end/long_end_extra.json")["sensitivity_range90_percent"]
    products = load(CHECKS / "long_end/long_end.json")["products"]
    i = products.index("forward-start 2y-3y")
    check("long end: forward start range at S = 0.1 (%)", extra["S=0.1,vs_sd=0.5"][i], 19.4, 0.05)
    check("long end: forward start range at S = 0.3 (%)", extra["S=0.3,vs_sd=0.5"][i], 36.8, 0.05)
    cov, width = ev["bs_hierarchy_coverage"], ev["bs_hierarchy_width"]
    small, large = "2 liquid + 6 illiquid", "8 liquid + 32 illiquid"
    for method, c, w, c_large in (("per name", 89.7, 9.9, 89.6), ("proxy", 59.5, 8.2, 83.3),
                                  ("empirical Bayes", 69.7, 5.5, 87.5), ("hierarchical Bayes", 88.0, 7.6, 89.4)):
        check(f"sector 2 + 6, {method}: coverage (%)", 100 * cov[small][method], c, 0.05)
        check(f"sector 2 + 6, {method}: width", width[small][method], w, 0.05)
        check(f"sector 8 + 32, {method}: coverage (%)", 100 * cov[large][method], c_large, 0.05)
    costs = load(RESULTS / "costs/costs.json")
    check("exact rough chain (CPU-hours)", costs["exact_rough_posterior"]["cpu_hours"], 16.9, 0.05)
    evaluation = costs["rough_likelihood_evaluation_cpu_s"]["median"]
    check("one rough likelihood evaluation (CPU-seconds)", evaluation, 5.0, 0.05)
    spy = load(CHECKS / "rough_spy_is/rough_spy_is_seed0.json")
    check("SPY importance sampling: proposals", spy["proposals"], 1024, 0)
    check("SPY importance sampling (CPU-minutes)", spy["proposals"] * evaluation / 60, 85, 1)
    network = costs["rough_amortized_posterior_1000_draws_cpu_s"]
    check("network, 1,000 draws (CPU-seconds)", network["median"], 3.6, 0.05)
    check("network, 1,000 draws (seconds)", network["wall_median"], 1.4, 0.05)
    timing = load(CHECKS / "thin_chains/timing.json")
    one = timing["layout"]["4096"]["cpu_seconds"]
    check("EWG layout evaluation at 4,096 paths (CPU-seconds)", one, 0.39, 0.005)
    check("EWG audit, 512 evaluations (CPU-minutes)", 512 * one / 60, 3, 0.5)
    check("SBC on 2,000 smiles with the network (CPU-hours)", 2000 * network["median"] / 3600, 2, 0.05)
    check("SBC on 2,000 smiles at 85 CPU-minutes each (CPU-hours)", 2000 * 85 / 60, 2800, 50)


def section_3():
    ev = load(ROOT / "course/images/v2-evidence-numbers.json")["thin_candidates"]

    def rows(ticker, days=None):
        return [e for e in ev[ticker]["expiries"] if days is None or e["calendar_days"] in days]

    for ticker, days, quotes, widths in (("EWI", [29, 57, 148, 238], (7, 11), (20, 36)),
                                         ("EWG", [29, 57, 85, 176], (7, 9), (2.3, 4.0)),
                                         ("EWQ", [29, 57, 148, 238], (2, 5), (8.0, 36))):
        r = rows(ticker)
        check(f"{ticker}: expiries", float([e["calendar_days"] for e in r] == days), 1, 0)
        q = [e["two_sided_near_money"] for e in r]
        w = [e["median_near_money_width_vol_points"] for e in r]
        check(f"{ticker}: fewest two-sided quotes", min(q), quotes[0], 0)
        check(f"{ticker}: most two-sided quotes", max(q), quotes[1], 0)
        check(f"{ticker}: narrowest width", min(w), widths[0], 0.05 if widths[0] < 10 else 0.5)
        check(f"{ticker}: widest width", max(w), widths[1], 0.05 if widths[1] < 10 else 0.5)
    monthly = rows("FEZ", [28, 56, 84])
    others = [e for e in rows("FEZ") if e["calendar_days"] not in (28, 56, 84)]
    check("FEZ monthly: fewest quotes", min(e["two_sided_near_money"] for e in monthly), 12, 0)
    check("FEZ monthly: most quotes", max(e["two_sided_near_money"] for e in monthly), 17, 0)
    check("FEZ monthly: narrowest", min(e["median_near_money_width_vol_points"] for e in monthly), 2.1, 0.05)
    check("FEZ monthly: widest", max(e["median_near_money_width_vol_points"] for e in monthly), 3.1, 0.05)
    check("FEZ others: narrowest", min(e["median_near_money_width_vol_points"] for e in others), 10, 0.5)
    check("FEZ others: widest", max(e["median_near_money_width_vol_points"] for e in others), 42, 0.5)
    check("FEZ others: last expiry", max(e["calendar_days"] for e in others), 210, 0)
    ewg = [e["median_near_money_width_vol_points"] for e in rows("EWG")]
    check("EWG half-spread, smallest (vol points)", min(ewg) / 2, 1.2, 0.05)
    check("EWG half-spread, largest (vol points)", max(ewg) / 2, 2.0, 0.05)
    check("EWG half-spread over SPY's design noise, smallest", min(ewg) / 2 / 0.3, 4, 0.5)
    check("EWG half-spread over SPY's design noise, largest", max(ewg) / 2 / 0.3, 7, 0.5)
    layout = load(ROOT / "lab/configs/thin/ewg_layout.json")
    check("EWG layout contracts", len(layout["contract_indices"]), 27, 0)
    counts = {d: layout["maturity_days"].count(d) for d in set(layout["maturity_days"])}
    check("EWG layout: seven at 21, 42 and 63 days", float(counts[21] == counts[42] == counts[63] == 7), 1, 0)
    check("EWG layout: six at 126 days", counts[126], 6, 0)
    days = sorted(r["trading_days"] for r in layout["expiry_map"].values())
    check("EWG expiries in trading days", float(days == [21, 40, 58, 120]), 1, 0)
    task = load(ROOT / "lab/configs/tasks/calibration_rough_bank.json")["task"]
    from sbi4finance.calibration import tasks
    ratio = np.array(layout["noise_sd"]) / tasks.sigma(task)[layout["contract_indices"]]
    check("EWG noise over design noise, smallest", ratio.min(), 2.0, 0.05)
    check("EWG noise over design noise, largest", ratio.max(), 15.0, 0.05)
    check("EWG noise over design noise, median", np.median(ratio), 3.8, 0.05)
    check("EWG cleaned quotes", len(load(CHECKS / "thin_chains/EWG_observation.json")["quotes"]), 61, 0)


def section_5():
    mini = load(CHECKS / "minipilot/spy-against-prior.json")["statistics"]
    for name, printed in (("atm_vol_1m", 0.28), ("atm_skew_1m", 0.25), ("skew_exponent", 0.79),
                          ("atm_curvature_1m", 0.83)):
        check(f"mini-pilot: SPY's rank, {name}", mini[name]["spy_rank"], printed, 0.005)
    quality = load(CHECKS / "minipilot/bank-quality.json")
    check("mini-pilot: CPU-seconds per surface", quality["generation"]["cpu_seconds_per_surface"], 11.5, 0.05)
    check("mini-pilot: share of surfaces with a quote above budget", quality["share_of_surfaces_with_a_quote_above_budget"],
          0, 0)
    pilot = load(RESULTS / "rough-pilot/s1/bank-quality.json")["generation"]["cpu_seconds_per_surface"]
    check("pilot: CPU-seconds per surface", pilot, 11.9, 0.05)
    prior = load(CHECKS / "curve_prior/curve_prior.json")["envelopes"]["s_w=0.15"]
    check("SPY's curve inside the prior's central 90% at every level", float(prior["spy_inside_central_90_at_every_level"]),
          1, 0)


def lectures():
    """Numbers that the lectures' statement of the argument quotes (Lectures 15 and 16, v2-plan Section 2.2)."""
    end = load(CHECKS / "long_end/long_end.json")
    names = end["products"]
    flat, post = np.array(end["range90_desk_delta"]), np.array(end["range90_posterior_mc"])
    beyond = np.array(end["variance_shares"]["ξ0 2.25 to 5 years (prior only)"])
    for name, ratio, share in (("forward-start 2y-3y", 0.32, 0.64), ("put 5y 80%", 0.30, 0.76),
                               ("variance swap 3y", 0.0, 0.59), ("autocallable 3y", 0.60, 0.02),
                               ("cliquet 3y", 0.67, 0.14)):
        i = names.index(name)
        check(f"long end, {name}: flat range over the posterior's", flat[i] / post[i], ratio, 0.005)
        check(f"long end, {name}: variance share beyond 2.25 years", beyond[i], share, 0.005)
    seeds = [load(CHECKS / f"rough_spy_is/rough_spy_is_seed{s}.json")["parameters"] for s in (0, 1)]
    shift = [abs(seeds[0][k]["posterior_mean"] - seeds[1][k]["posterior_mean"]) / seeds[0][k]["posterior_sd"]
             for k in seeds[0]]
    check("SPY importance sampling: second seed's shift of the means, smallest (sd)", min(shift), 0.07, 0.005)
    check("SPY importance sampling: second seed's shift of the means, largest (sd)", max(shift), 0.21, 0.005)
    rows = load(RESULTS / "rough-pilot/s7-across/across-models.json")["surfaces"]
    fs = rows[2]["products"]["forward_start_call"]
    check("pilot smile 292: forward-start reserve under rough Bergomi (%)", 100 * fs["rough"]["reserve_short_relative"],
          6.0, 0.05)
    check("pilot smile 292: forward-start reserve under Heston (%)", 100 * fs["heston"]["reserve_short_relative"], 3.2,
          0.05)
    for m, printed in ((512, 0.63), (1024, 0.67)):
        check(f"k-hat sample-size threshold at {m} proposals", min(1 - 1 / np.log10(m), 0.7), printed, 0.005)
    local = {s["xi0_log_sd"]: s for s in load(CHECKS / "rough_checks.json")["scenarios"] if s["n_quotes"] == 15}
    for sd, lo, hi in ((0.0, 0.61, 0.65), (0.1, 0.12, 0.16)):
        c = sorted(local[sd][seed]["contraction"][2] for seed in ("seed1", "seed2"))
        check(f"local check, 15 quotes, curve sd {sd}: contraction of rho, lower seed", c[0], lo, 0.005)
        check(f"local check, 15 quotes, curve sd {sd}: contraction of rho, higher seed", c[1], hi, 0.005)


def main():
    grid = section_1()
    section_2()
    section_3()
    section_5()
    lectures()
    print(f"grid bias (for the plan's Section 1): {json.dumps(grid)}")
    if FAILED:
        print(f"{len(FAILED)} checks failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
