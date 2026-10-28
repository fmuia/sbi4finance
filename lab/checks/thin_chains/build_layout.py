"""A thin layout and its observation from a real thin chain: the design contracts it quotes, and their noise.

    python build_layout.py RAW_CHAIN.json TICKER LAYOUT.json OBSERVATION.json [--task CONFIG]

The chain is read with the SPY pipeline's conventions (sbi4finance.calibration.chain): valuation at the
underlying's last trade, out-of-the-money quotes, and a smoothing spline in standardized moneyness per expiration.
Three choices differ from SPY's pipeline:

* maturities are NYSE trading days over 252 (chain.NYSE_HOLIDAYS removed), the model's clock; SPY's surface counted
  weekdays, which makes the April 2027 expiry 126 days instead of 120;

* the forward (--forward carry, the default) is the spot net of the dividends of settings.json, carried at the flat
  rate in calendar time, with the discount in calendar time too. Parity (--forward parity, SPY's choice) is biased
  low on American options: the puts in the near-money pairs carry early-exercise value. On EWG of 22 October the
  parity forward is 0.2% to 0.5% below the carry forward, which moves put volatilities down and call volatilities
  up by up to 1.3 half-spreads. EWG paid no December distribution in 2024 or 2025 (Alpha Vantage DIVIDENDS), so
  no dividend falls before its six-month expiry. The printout reports the other choice's effect.
* every two-sided quote is kept (relative spread below MAX_SPREAD); the half-spreads weight them.

Each design maturity is matched to the chain's expiration within MATCH_DAYS trading days, if any. At a matched
maturity, a design point (T, delta) belongs to the layout if its log-moneyness k = delta sigma_ref sqrt(T) lies
inside the strikes the expiration quotes (in the expiration's own standardized moneyness). The point's implied
volatility is the expiration's at k; its noise sd is the local half-spread in price units (the half-spread in
volatility units times the vega at the point's own volatility), floored at the design noise. Raw quotes are not
written; LAYOUT.json and OBSERVATION.json hold derived values only.
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import bank, chain, surface, tasks, thin  # noqa: E402

MATCH_DAYS = 7
MAX_SPREAD = 2.0  # (ask - bid) / mid < 2 for every two-sided quote: keep them all; their half-spreads weight them


def carry(settings, valuation, expiry, rate):
    """Discount and forward in calendar time: F = (S - PV(dividends before expiry)) / D."""
    tau = (expiry - valuation).days / 365.0
    r = rate(tau)
    discount = float(np.exp(-r * tau))
    pv = sum(a * np.exp(-r * (d - valuation).days / 365.0)
             for d, a in dividend_dates(settings) if valuation < d < expiry)
    return discount, float((settings["spot"] - pv) / discount)


def dividend_dates(settings):
    one_year = [(date.fromisoformat(d), float(a)) for d, a in settings["dividends"]]
    return [(d.replace(year=d.year + y), a) for y in range(int(settings["repeat_years"])) for d, a in one_year]


def read_layout(raw_text, settings, task, forward_kind="carry"):
    rows = chain.parse_cboe_json(raw_text)
    valuation = rows[0]["date"]
    rate = chain.rate_curve([(0.25, settings["rate"])])
    contracts = task["context"]["contracts"]
    k_all, T_all, kind_all = surface.contract_arrays(contracts)
    sigma = tasks.sigma(task)
    by_expiry = {}
    for r in rows:
        if r["expiration"] > valuation:
            by_expiry.setdefault(r["expiration"], []).append(r)
    expirations = []
    for expiry, rows_e in sorted(by_expiry.items()):
        if expiry > chain.NYSE_HOLIDAYS_THROUGH:
            continue  # beyond the published holiday calendar, and beyond the design grid's year
        days = chain.trading_days(valuation, expiry, chain.NYSE_HOLIDAYS)
        if days < 1:
            continue
        T = days / surface.TRADING_DAYS
        if forward_kind == "carry":
            discount, forward = carry(settings, valuation, expiry, rate)
        else:
            discount = float(np.exp(-rate(T) * T))
            forward = chain.forward_given_discount(rows_e, discount)
        if forward is None:
            continue
        quotes = chain.clean_expiration(rows_e, forward, discount, T, max_spread=MAX_SPREAD)
        if len(quotes) >= 5:
            expirations.append({"expiry": expiry, "days": days, "T": T, "forward": forward, "discount": discount,
                                "quotes": quotes, "spline": chain.smile_spline(quotes, T)})
    chosen, expiry_map = [], {}
    for design_days in sorted(set(int(round(t * surface.TRADING_DAYS)) for t in T_all)):
        near = [e for e in expirations if abs(e["days"] - design_days) <= MATCH_DAYS]
        if not near:
            continue
        e = min(near, key=lambda e: abs(e["days"] - design_days))
        expiry_map[str(design_days)] = {"expiry": e["expiry"].isoformat(), "trading_days": e["days"],
                                        "forward": e["forward"], "discount": e["discount"],
                                        "quotes_kept": len(e["quotes"])}
        e["design_days"] = design_days
        spline, (dmin, dmax), (deltas_q, half_q) = e["spline"]
        T_design = design_days / surface.TRADING_DAYS
        for i in np.flatnonzero(np.isclose(T_all, T_design)):
            delta = k_all[i] / (surface.SIGMA_REF * np.sqrt(T_design))
            d_e = k_all[i] / (surface.SIGMA_REF * np.sqrt(e["T"]))  # the same log-moneyness at the expiry
            if not dmin <= d_e <= dmax:
                continue
            iv = float(spline(d_e))
            half = float(np.interp(d_e, deltas_q, half_q))
            # The half-spread in price units at the reading: the local volatility half-spread times the vega at
            # the reading's own volatility (not at sigma_ref, which understates it threefold in the put wing).
            noise = float(surface.bs_vega(k_all[i], T_design, iv) * half)
            chosen.append({"index": int(i), "maturity_days": design_days, "delta": float(delta), "iv": iv,
                           "half_width_vol": half, "noise_sd": max(noise, float(sigma[i])),
                           "price": float(surface.bs_price(k_all[i], T_design, iv, kind_all[i]))})
    quotes = [{"expiry": e["expiry"].isoformat(), "trading_days": e["days"], "design_days": e["design_days"],
               "k": q["k"], "kind": q["kind"], "price_forward_units": q["price"], "implied_vol": q["iv"],
               "half_spread_vol": q["iv_half_spread"]}
              for e in expirations if "design_days" in e for q in e["quotes"]]
    return valuation, chosen, expiry_map, quotes


def expiry_map_forward(expiry_map, quote):
    """The forward of the quote's expiry, from an expiry map keyed by design maturity."""
    return next(r["forward"] for r in expiry_map.values() if r["expiry"] == quote["expiry"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("ticker")
    parser.add_argument("layout_out")
    parser.add_argument("observation_out")
    parser.add_argument("--task", default=str(ROOT / "configs" / "tasks" / "calibration_rough_bank.json"))
    parser.add_argument("--forward", choices=("carry", "parity"), default="carry")
    args = parser.parse_args()
    config = json.loads(Path(args.task).read_text())
    task = config["task"]
    settings = json.loads((HERE / "settings.json").read_text())[args.ticker]
    raw_text = Path(args.raw).read_text()
    valuation, chosen, expiry_map, quotes = read_layout(raw_text, settings, task, args.forward)
    other = "parity" if args.forward == "carry" else "carry"
    _, chosen_other, expiry_other, quotes_other = read_layout(raw_text, settings, task, other)
    source = (f"Cboe delayed quotes for {args.ticker} (data/SOURCES.md), chain sha256 "
              f"{hashlib.sha256(raw_text.encode()).hexdigest()}; quotes of the {valuation.isoformat()} close")
    layout = {"format": thin.LAYOUT_FORMAT, "ticker": args.ticker, "asof": valuation.isoformat(), "source": source,
              "contract_indices": [c["index"] for c in chosen], "noise_sd": [c["noise_sd"] for c in chosen],
              "half_width_vol": [c["half_width_vol"] for c in chosen],
              "maturity_days": [c["maturity_days"] for c in chosen], "delta": [c["delta"] for c in chosen],
              "expiry_map": expiry_map, "forward": args.forward,
              "task_fingerprint": bank.task_fingerprint(config)}
    thin.validate_layout(layout, task)
    Path(args.layout_out).write_text(json.dumps(layout, indent=1) + "\n")
    by_index = {c["index"]: c for c in chosen_other}
    k_all, T_all, _ = surface.contract_arrays(task["context"]["contracts"])
    strike = {(q["expiry"], q["kind"], round(float(np.exp(q["k"]) * expiry_map_forward(expiry_map, q)), 4)): q
              for q in quotes}
    shifts = []
    for q in quotes_other:
        key = (q["expiry"], q["kind"], round(float(np.exp(q["k"]) * expiry_map_forward(expiry_other, q)), 4))
        if key in strike:
            mine = strike[key]
            shifts.append({"expiry": q["expiry"], "kind": q["kind"],
                           "vol_points": 100 * (q["implied_vol"] - mine["implied_vol"]),
                           "half_spreads": (q["implied_vol"] - mine["implied_vol"]) / mine["half_spread_vol"]})
    diagnostics = {
        "forward_other": other,
        "forward_by_design_maturity": {d: {args.forward: r["forward"], other: expiry_other[d]["forward"]}
                                       for d, r in expiry_map.items()},
        "other_forward_moves_layout_readings_in_half_widths": [
            (by_index[c["index"]]["iv"] - c["iv"]) / c["half_width_vol"] for c in chosen if c["index"] in by_index],
        "other_forward_moves_quotes": shifts,
        "vega_at_own_vol_over_vega_at_sigma_ref": [
            float(surface.bs_vega(k_all[c["index"]], T_all[c["index"]], c["iv"])
                  / surface.bs_vega(k_all[c["index"]], T_all[c["index"]], surface.SIGMA_REF)) for c in chosen],
        "clock": "NYSE trading days over 252 (sbi4finance.calibration.chain.NYSE_HOLIDAYS)",
        "weekdays_to_each_expiry": {r["expiry"]: chain.trading_days(valuation, date.fromisoformat(r["expiry"]))
                                    for r in expiry_map.values()}}
    observation = {"ticker": args.ticker, "asof": valuation.isoformat(), "source": source, "forward": args.forward,
                   "prices_forward_units": [c["price"] for c in chosen], "implied_vol": [c["iv"] for c in chosen],
                   "quotes": quotes, "diagnostics": diagnostics,
                   "note": "prices_forward_units: the chain read onto its own layout, the network's input. quotes: "
                           "the cleaned out-of-the-money quotes of the matched expirations at their own strikes and "
                           "trading days, the audit's likelihood data (price and half-spread in forward units "
                           "follow from implied_vol and half_spread_vol with the Black formula)"}
    Path(args.observation_out).write_text(json.dumps(observation, indent=1) + "\n")
    for days, record in expiry_map.items():
        points = [c for c in chosen if c["maturity_days"] == int(days)]
        print(f"design {days} d <- {record['expiry']} ({record['trading_days']} d): {len(points)} points, deltas "
              f"{min(c['delta'] for c in points):.1f} to {max(c['delta'] for c in points):.1f}, half-widths "
              + " ".join(f"{100 * c['half_width_vol']:.1f}" for c in points))
    print(f"{len(chosen)} layout contracts")
    by_index = {c["index"]: c for c in chosen_other}
    shift = [(c["iv"] - by_index[c["index"]]["iv"]) / c["half_width_vol"] for c in chosen if c["index"] in by_index]
    for days, record in expiry_map.items():
        print(f"forward at {days} d: {args.forward} {record['forward']:.3f}, {other} "
              f"{expiry_other[days]['forward']:.3f}")
    print(f"the {other} forward moves the {len(shift)} common layout readings by {min(shift):+.2f} to "
          f"{max(shift):+.2f} noise half-widths")


if __name__ == "__main__":
    main()
