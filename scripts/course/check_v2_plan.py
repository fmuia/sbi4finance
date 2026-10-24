"""Check the numbers of docs/course/v2-plan.md against the stored results.

    python scripts/course/check_v2_plan.py

Each check reads a committed file and compares it with the value the plan prints, at the plan's precision.
The script prints one line per check and exits with status 1 if any check fails.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "tutorial_final" / "checks"
RESULTS = ROOT / "tutorial_final" / "results" / "calibration"
FAILED = []


def load(path):
    return json.loads(Path(path).read_text())


def check(label, value, printed, tol):
    ok = abs(value - printed) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: stored {value:.4g}, printed {printed}")
    if not ok:
        FAILED.append(label)


def ratio_range(cases, prefix):
    values = [p["ratio"] for c in cases if c["case"].startswith(prefix) for p in c["points"]]
    return min(values), max(values)


def main():
    ev = load(ROOT / "docs/course/images/v2-evidence-numbers.json")
    cases = ev["classical_vs_posterior"]

    # Section 1.3, Figure 1
    lo, hi = ratio_range(cases, "Heston")
    check("Heston ratios, lowest", lo, 0.92, 0.005)
    check("Heston ratios, highest", hi, 1.21, 0.005)
    for prefix, a, b in (("rough 63 quotes", 0.93, 1.16), ("rough 63 quotes 3", 0.89, 1.20),
                         ("rough 15 quotes", 0.93, 1.73), ("rough front months 27", 0.78, 1.44),
                         ("rough front months H fixed", 0.20, 1.10), ("rough long end", 0.0, 0.67)):
        group = [c for c in cases if c["case"].startswith(prefix)
                 and not (prefix == "rough 63 quotes" and "3×" in c["case"])]
        values = [p["ratio"] for c in group for p in c["points"]]
        check(f"{prefix}: lowest", min(values), a, 0.005)
        check(f"{prefix}: highest", max(values), b, 0.005)
    rough15 = next(c for c in cases if c["case"].startswith("rough 15"))
    top = max(rough15["points"], key=lambda p: p["ratio"])
    check("15 quotes: the largest is the put's short side", float(top["product"] == "put_out_of_grid"
                                                                  and top["side"] == "short"), 1, 0)
    h_fixed = next(c for c in cases if "H fixed" in c["case"])
    fs_long = next(p["ratio"] for p in h_fixed["points"] if p["product"] == "forward_start" and p["side"] == "long")
    check("H fixed: forward start long, a fifth", fs_long, 0.20, 0.005)
    long_end = next(c for c in cases if "long end" in c["case"])
    nonzero = [p["ratio"] for p in long_end["points"] if p["ratio"] > 0]
    check("long end, flat: lowest non-zero", min(nonzero), 0.30, 0.005)
    extra = load(CHECKS / "long_end/long_end_extra.json")["sensitivity_range90_percent"]
    products = load(CHECKS / "long_end/long_end.json")["products"]
    i = products.index("forward-start 2y-3y")
    check("long end: forward start range at S = 0.1", extra["S=0.1,vs_sd=0.5"][i], 19.4, 0.05)
    check("long end: forward start range at S = 0.3", extra["S=0.3,vs_sd=0.5"][i], 36.8, 0.05)

    # Figure 2
    rows = ev["desk_case_breach_rates"]["rows"]
    rate = {(r["method"], r["side"], r["product"]): r["breach_rate"] for r in rows}
    post = [v for (m, _, _), v in rate.items() if m == "posterior"]
    check("posterior breaches, lowest (%)", 100 * min(post), 1.25, 1e-9)
    check("posterior breaches, highest (%)", 100 * max(post), 10.0, 0.05)
    check("Gauss–Newton, forward start long (%)", 100 * rate[("Gauss–Newton", "long", "forward_start")], 17.5, 0.05)
    check("Gauss–Newton, put long (%)", 100 * rate[("Gauss–Newton", "long", "put_1y_80")], 15.0, 0.05)
    profile_long = [rate[("profile", "long", p)] for p in ("forward_start", "put_1y_80", "cliquet")]
    check("profile long, three products (%)", 100 * max(profile_long), 16.25, 1e-9)
    check("profile long, three products equal", max(profile_long) - min(profile_long), 0.0, 1e-12)
    check("H fixed, forward start long (%)", 100 * rate[("desk: H = 0.10", "long", "forward_start")], 81, 0.5)
    check("H fixed, put long (%)", 100 * rate[("desk: H = 0.10", "long", "put_1y_80")], 71, 0.5)
    band = ev["desk_case_breach_rates"]["binomial_95_band"]
    check("binomial band, lower (%)", 100 * band[0], 3.75, 1e-9)
    check("binomial band, upper (%)", 100 * band[1], 17.5, 1e-9)
    cov = ev["bs_hierarchy_coverage"]["2 liquid + 6 illiquid"]
    width = ev["bs_hierarchy_width"]["2 liquid + 6 illiquid"]
    check("sector: proxy coverage (%)", 100 * cov["proxy"], 59.5, 0.05)
    check("sector: empirical Bayes coverage (%)", 100 * cov["empirical Bayes"], 69.7, 0.05)
    check("sector: hierarchical Bayes coverage (%)", 100 * cov["hierarchical Bayes"], 88.0, 0.05)
    check("sector: hierarchical width", width["hierarchical Bayes"], 7.6, 0.05)
    check("sector: per-name width", width["per name"], 9.9, 0.05)

    # Section 1.3, conclusion 5: model error on the front-month case
    combined = load(CHECKS / "desk_case/desk_case_combined.json")
    fs = combined["scales"]["1.0"]["products"]["forward_start"]
    full_median = combined["full_surface"]["forward_start"][2]
    check("full surface over front-month mark, forward start (%)", 100 * (full_median / fs["mark"] - 1), 27, 0.5)
    upper = max(fs["mark"] + fs["posterior"]["short"], fs["mark"] + fs["delta"]["short"],
                fs["mark"] + fs["profile"]["short"], fs["mark"] + fs["truncated_gaussian"]["short"],
                fs["mark_h_fixed"] + fs["h_fixed_delta"]["short"])
    check("full-surface price above every method's 90% point", float(full_median > upper), 1, 0)
    market = load(CHECKS / "desk_case/real_market.json")
    check("front-month fit: chi-square on the 36 later quotes", market["front_months"]["chi2_other_36"], 216, 0.5)
    check("full-surface fit: chi-square on the 36 later quotes", market["full_surface"]["chi2_other_36"], 48, 0.5)

    # Section 1.4: costs
    costs = load(RESULTS / "costs/costs.json")
    check("exact rough chain (CPU-hours)", costs["exact_rough_posterior"]["cpu_hours"], 16.9, 0.05)
    check("network, 1,000 draws (CPU-seconds)", costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"], 3.6, 0.05)
    check("network, 1,000 draws (seconds)", costs["rough_amortized_posterior_1000_draws_cpu_s"]["wall_median"], 1.4,
          0.05)
    audit = load(RESULTS / "rough-pilot/s5/audit.json")["summary"]["cpu_seconds_per_surface_median"]
    check("audit (CPU-seconds)", audit, 425, 0.5)
    desk = load(CHECKS / "desk_case/desk_case.json")["wall_seconds"] + \
        load(CHECKS / "desk_case/desk_case_combined.json")["wall_seconds"]
    check("front-month importance sampling, two rounds on 2 cores (CPU-hours)", 2 * desk / 3600, 4.0, 0.05)
    spy = load(CHECKS / "rough_spy_is/rough_spy_is_seed0.json")
    check("SPY importance sampling: proposals", spy["proposals"], 1024, 0)

    # Figure 3 and lecture A2: EWI
    ewi = load(CHECKS / "thin_chains/ewi_2026-10-22.json")
    for e, printed in zip(ewi["expiries"], (36, 23, 24, 21)):
        check(f"EWI {e['expiry']}: median width (vol points)", 100 * e["median_iv_spread"], printed, 0.5)
    check("EWI: two-sided quotes", sum(e["two_sided"] for e in ewi["expiries"]), 45, 0)
    check("EWI: out-of-the-money quotes", sum(e["out_of_the_money"] for e in ewi["expiries"]), 94, 0)
    check("EWI: contracts traded", ewi["volume_total"], 132, 0)
    first, last = ewi["expiries"][0], ewi["expiries"][-1]
    check("EWI first expiry: strip on bids (%)", 100 * first["strip_vol_on_bids"], 14, 0.5)
    check("EWI first expiry: strip on asks (%)", 100 * first["strip_vol_on_asks"], 52, 0.5)
    check("EWI last expiry: strip on bids (%)", 100 * last["strip_vol_on_bids"], 14, 0.5)
    check("EWI last expiry: strip on asks (%)", 100 * last["strip_vol_on_asks"], 38, 0.5)
    rates = [100 * e["parity_regression"]["implied_rate"] for e in ewi["expiries"]]
    check("EWI: lowest parity-implied rate (%)", min(rates), -4.6, 0.05)
    check("EWI: highest parity-implied rate (%)", max(rates), 56, 0.5)

    def eep(expiry, strike, half_spread, printed):
        e = next(e for e in ewi["expiries"] if e["expiry"] == expiry)
        q = min((q for q in e["quotes"] if q["cp"] == "C"),
                key=lambda q: abs(e["forward"] * 2.718281828459045 ** q["k"] - strike))
        check(f"EWI {expiry} {strike} call: early-exercise premium", q["early_exercise_over_half_spread"] * half_spread,
              printed, 0.005)

    eep("2026-12-18", 57, 0.125, 0.18)  # half-spreads from the raw file: (1.00 - 0.75) / 2 and (5.90 - 2.40) / 2
    eep("2027-06-17", 55, 1.75, 0.50)

    # Section 6.1, Figure 4
    cp = load(CHECKS / "curve_prior/curve_prior.json")
    check("SPY last/first level", cp["spy_last_over_first"], 2.93, 0.005)
    check("power-law exponent", cp["power_law"]["exponent"], 0.244, 0.0005)
    check("power-law residual sd", cp["power_law"]["residual_sd"], 0.025, 0.0005)
    env = cp["envelopes"]
    check("pilot prior: SPY inside at every level", float(env["old"]["spy_inside_central_90_at_every_level"]), 0, 0)
    check("candidate: SPY inside at every level", float(env["s_w=0.15"]["spy_inside_central_90_at_every_level"]), 1, 0)
    check("candidate: share rising as much (%)", 100 * env["s_w=0.15"]["share_with_ratio_at_least_spy"], 29, 0.5)
    check("candidate: share falling as much (%)", 100 * env["s_w=0.15"]["share_inverted_as_much"], 29, 0.5)
    r = next(r for r in cp["extrapolation"] if r["known_to_days"] == 63 and r["s_w"] == 0.15 and r["level"] == 7)
    check("126-252 days: band low (%)", r["q05_vol"], 19.6, 0.05)
    check("126-252 days: band high (%)", r["q95_vol"], 27.2, 0.05)
    check("126-252 days: median (%)", r["median_vol"], 23.1, 0.05)
    check("126-252 days: SPY (%)", r["spy_vol"], 23.4, 0.05)
    check("126-252 days: flat (%)", r["flat_vol"], 19.9, 0.05)
    check("126-252 days: flat z", r["z_flat"], -1.49, 0.005)

    # Sections 6.2 and 6.8: banks and training
    heston = load(RESULTS / "heston/s1/bank-quality.json")["generation"]
    check("Heston bank (CPU-minutes)", heston["cpu_seconds"] / 60, 46, 0.5)
    training = load(ROOT / "tutorial_final/checkpoints/heston/training.json")
    check("Heston training on 4 cores (minutes)", training["timing_seconds"]["training_including_compile"] / 60, 32, 0.5)
    check("velocity network width", training["training"]["subnet_widths"][0], 256, 0)

    # Sections 6.2 and 6.6: per-draw costs (timing.json) and the budgets built on them
    timing = load(CHECKS / "desk_case/timing.json")
    check("27 front-month quotes, 32,768 paths (CPU-s)", timing["front_27_quotes_32768_paths"], 1.8, 0.05)
    check("desk products, 8,192 paths (CPU-s)", timing["desk_products_8192_paths"], 0.25, 0.005)
    check("desk products, 16,384 paths (CPU-s)", timing["desk_products_16384_paths"], 1.0, 0.05)
    check("slot 5 products, 16,384 paths (CPU-s)", timing["slot5_products_16384_paths"], 2.5, 0.05)
    product_bank = 5000 * (timing["desk_products_16384_paths"] + timing["slot5_products_16384_paths"]) / 3600
    check("product bank, 5,000 surfaces (CPU-hours)", product_bank, 5, 0.5)
    check("full bank, 112,000 surfaces at 11.9 CPU-s (CPU-hours)", 112000 * 11.9 / 3600, 370, 1)
    draws = 256 * (timing["front_27_quotes_32768_paths"] + timing["desk_products_8192_paths"]) / 60
    check("256 draws per chain (CPU-minutes)", draws, 9, 0.5)
    batches = [load(CHECKS / f"desk_case/validate_thin_x1_batch{b}.json")["report"] for b in (0, 1)]
    per_chain = 2 * sum(b["wall_seconds"] for b in batches) / sum(b["replicas"] for b in batches) / 60
    check("classical fits and truths per chain (CPU-minutes)", per_chain, 3.3, 0.05)
    check("S10, 1,000 chains (CPU-hours)", 1000 * (draws + per_chain) / 60, 200, 10)
    check("S5 audits, liquid network (CPU-hours)", 500 * audit / 3600, 59, 0.5)
    check("S5 audits, S2-T at most (CPU-hours)", 400 * audit / 3600, 47, 0.5)
    check("SBC on 2,000 smiles, amortized (CPU-hours)", 2000 * 3.6 / 3600, 2, 0)
    check("SBC on 2,000 smiles at 75 CPU-minutes each (CPU-hours)", 2000 * 75 / 60, 2500, 0)
    check("mini-pilot, 200 surfaces (CPU-minutes)", 200 * 11.9 / 60, 40, 0.5)

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall checks pass")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
