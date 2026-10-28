"""From an end-of-day option chain to the design-grid observation (design document, Appendix I).

Steps, each recorded in the returned diagnostics:

1. Parse rows (expiration, strike, type, bid, ask) from an Alpha Vantage
   HISTORICAL_OPTIONS CSV or a Cboe delayed-quotes JSON.
2. Per expiration: forward and discount from put–call parity on the pairs
   nearest the money; mids in forward units (price / (D F)).
3. Keep out-of-the-money quotes with a positive bid, a relative spread at most
   max_spread and a mid above intrinsic value; implied volatility per quote,
   and the half-spread in volatility units.
4. Total variance w(T) per expiration by log-contract replication over the
   cleaned strip; forward-variance levels on the design intervals from w
   interpolated linearly in T at the design maturities.
5. The design grid: at each design maturity, implied volatility at each delta is
   read from a smoothing spline per bracketing expiration and interpolated in
   total variance between expirations; the quote-noise sd is the vega at
   sigma_ref times the local half-spread, floored at the design noise.

Only derived quantities leave this module; raw quotes stay with the caller.
"""

import csv
from datetime import date
import io
import json

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import surface


def _day(text):
    return date.fromisoformat(str(text)[:10])


# NYSE full-day closures as published by the exchange: "NYSE Group Announces 2026, 2027 and 2028 Holiday and
# Early Closings Calendar", Intercontinental Exchange press release, 2025
# (https://ir.theice.com/press/news-details/2025/NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/).
NYSE_HOLIDAYS = tuple(date.fromisoformat(d) for d in (
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23",
    "2028-12-25"))
NYSE_HOLIDAYS_THROUGH = date(2028, 12, 31)


def trading_days(valuation, expiration, holidays=None):
    """Business days from valuation to expiration: weekends excluded, and the given exchange holidays.

    holidays=None counts weekdays (SPY's surface of 21 October 2026 was read so); NYSE_HOLIDAYS removes the NYSE
    closures, which makes a calendar year about 252 days, the model's clock. Beyond NYSE_HOLIDAYS_THROUGH the
    published list ends and the count falls back to weekdays."""
    if holidays is not None and expiration > NYSE_HOLIDAYS_THROUGH and holidays is NYSE_HOLIDAYS:
        raise ValueError(f"no published NYSE holidays after {NYSE_HOLIDAYS_THROUGH}")
    return int(np.busday_count(valuation, expiration, holidays=[] if holidays is None else list(holidays)))


def parse_alpha_vantage_csv(text):
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        rows.append({"expiration": _day(record["expiration"]), "strike": float(record["strike"]),
                     "type": record["type"].strip().lower(), "bid": float(record["bid"]), "ask": float(record["ask"]),
                     "date": _day(record["date"])})
    return rows


def parse_cboe_json(text):
    """Cboe delayed quotes: OCC symbols ROOT YYMMDD C/P strike*1000 (8 digits).

    The valuation date is the session of the underlying's last trade: a file
    fetched before the open carries the next day's timestamp but the previous
    session's closing quotes, and counting maturities from the timestamp would
    shorten every one of them by a trading day.
    """
    payload = json.loads(text)
    data = payload.get("data", payload)
    valuation = _day(data.get("last_trade_time") or payload.get("timestamp", ""))
    rows = []
    for option in data["options"]:
        symbol = option["option"]
        tail = symbol[-15:]
        rows.append({"expiration": date(2000 + int(tail[:2]), int(tail[2:4]), int(tail[4:6])),
                     "strike": int(tail[7:]) / 1000.0, "type": "call" if tail[6] == "C" else "put",
                     "bid": float(option["bid"]), "ask": float(option["ask"]), "date": valuation})
    return rows


def forward_and_discount(rows, spot_guess=None, pairs=6):
    """Parity regression C - P = D (F - K) on the pairs of one expiration nearest the money."""
    calls = {r["strike"]: 0.5 * (r["bid"] + r["ask"]) for r in rows if r["type"] == "call" and r["bid"] > 0}
    puts = {r["strike"]: 0.5 * (r["bid"] + r["ask"]) for r in rows if r["type"] == "put" and r["bid"] > 0}
    common = sorted(set(calls) & set(puts))
    if len(common) < 3:
        return None
    differences = np.array([calls[k] - puts[k] for k in common])
    centre = common[int(np.argmin(np.abs(differences)))] if spot_guess is None else spot_guess
    chosen = sorted(common, key=lambda k: abs(k - centre))[:pairs]
    return surface.parity_forward(chosen, [calls[k] for k in chosen], [puts[k] for k in chosen])


def forward_given_discount(rows, discount, pairs=6):
    """Parity with a known discount: F = K + (C - P) / D, the median over the pairs nearest the money.

    For American options (SPY) the regression of forward_and_discount is tilted by the
    early-exercise value of the in-the-money puts, which grows with the strike, and
    returns discounts above one; a discount from the rate curve avoids it.
    """
    calls = {r["strike"]: 0.5 * (r["bid"] + r["ask"]) for r in rows if r["type"] == "call" and r["bid"] > 0}
    puts = {r["strike"]: 0.5 * (r["bid"] + r["ask"]) for r in rows if r["type"] == "put" and r["bid"] > 0}
    common = sorted(set(calls) & set(puts))
    if len(common) < 3:
        return None
    differences = np.array([calls[k] - puts[k] for k in common])
    centre = common[int(np.argmin(np.abs(differences)))]
    chosen = sorted(common, key=lambda k: abs(k - centre))[:pairs]
    return float(np.median([k + (calls[k] - puts[k]) / discount for k in chosen]))


def rate_curve(points):
    """Continuously compounded rate r(T) from (T, bond-equivalent yield) points, linear in T, flat outside."""
    points = sorted(points)
    times = np.array([p[0] for p in points], dtype=float)
    rates = 2 * np.log1p(np.array([p[1] for p in points], dtype=float) / 2)
    return lambda T: float(np.interp(T, times, rates))


def clean_expiration(rows, forward, discount, maturity, max_spread=0.5):
    """Out-of-the-money quotes in forward units with their implied volatilities and half-spreads."""
    kept = []
    for r in rows:
        k = np.log(r["strike"] / forward)
        is_put = r["type"] == "put"
        if (is_put and k > 0) or (not is_put and k <= 0) or r["bid"] <= 0 or r["ask"] <= r["bid"]:
            continue
        mid = 0.5 * (r["bid"] + r["ask"])
        if (r["ask"] - r["bid"]) / mid > max_spread:
            continue
        price = mid / (discount * forward)
        kind = "put" if is_put else "call"
        iv = float(surface.implied_vol(price, k, maturity, kind))
        if not np.isfinite(iv):
            continue
        half = 0.5 * (r["ask"] - r["bid"]) / (discount * forward)
        vega = float(surface.bs_vega(k, maturity, iv))
        kept.append({"k": k, "price": price, "kind": kind, "iv": iv, "iv_half_spread": half / vega if vega > 0 else np.inf})
    kept.sort(key=lambda q: q["k"])
    # Implied volatility must be finite and total variance continuous across the money; drop isolated outliers.
    if len(kept) >= 5:
        iv = np.array([q["iv"] for q in kept])
        smooth = np.convolve(np.pad(iv, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        kept = [q for q, s in zip(kept, smooth) if abs(q["iv"] - s) < 0.25 * s]
    return kept


def smile_spline(quotes, maturity, sigma_ref=surface.SIGMA_REF):
    delta = np.array([q["k"] for q in quotes]) / (sigma_ref * np.sqrt(maturity))
    iv = np.array([q["iv"] for q in quotes])
    weight = 1.0 / np.maximum(np.array([q["iv_half_spread"] for q in quotes]), 1e-4)
    spline = UnivariateSpline(delta, iv, w=weight, k=3, s=len(delta))
    half = np.array([q["iv_half_spread"] for q in quotes])
    return spline, (delta.min(), delta.max()), (delta, half)


def build_surface(rows, *, valuation=None, design_days=surface.DESIGN_DAYS, deltas=surface.DESIGN_DELTAS,
                  sigma_ref=surface.SIGMA_REF, max_spread=0.5, rate=None, replication_deltas=(-8.0, 1.8)):
    """The 63 design prices, their noise sds and the 7 forward-variance levels, with diagnostics.

    rate, a function of T (see rate_curve), sets each expiration's discount; the forward then
    comes from parity on the pairs nearest the money. Without it both come from the parity
    regression, which suits European options only.

    The variance-swap term structure, and with it the forward-variance levels, uses only the
    expirations whose cleaned strip reaches the standardized moneyness replication_deltas on
    both sides: a strip cut short in the put wing misses the tail's variance, which on SPY's
    end-of-month expirations is three to four volatility points. Every expiration serves the
    smile itself.
    """
    valuation = valuation or rows[0]["date"]
    by_expiry = {}
    for r in rows:
        if r["expiration"] > valuation:
            by_expiry.setdefault(r["expiration"], []).append(r)
    expirations = []
    for expiry, chain in sorted(by_expiry.items()):
        days = trading_days(valuation, expiry)
        fd = forward_and_discount(chain)
        if days < 1 or fd is None:
            continue
        T = days / surface.TRADING_DAYS
        parity_discount = fd[1]
        if rate is None:
            forward, discount = fd
        else:
            discount = float(np.exp(-rate(T) * T))
            forward = forward_given_discount(chain, discount)
        quotes = clean_expiration(chain, forward, discount, T, max_spread)
        if len(quotes) < 8:
            continue
        k = np.array([q["k"] for q in quotes])
        w = surface.total_variance_by_replication(k, [q["price"] for q in quotes])
        reach = (float(k.min() / (sigma_ref * np.sqrt(T))), float(k.max() / (sigma_ref * np.sqrt(T))))
        expirations.append({"expiration": expiry.isoformat(), "days": days, "T": T, "forward": forward,
                            "discount": discount, "parity_regression_discount": parity_discount, "quotes": quotes,
                            "w_replicated": w, "strip_deltas": reach,
                            "replication_covered": reach[0] <= replication_deltas[0] and reach[1] >= replication_deltas[1]})
    if not expirations:
        raise ValueError("no usable expiration after cleaning")
    Ts = np.array([e["T"] for e in expirations])
    covered = [e for e in expirations if e["replication_covered"]]
    T_covered = np.array([e["T"] for e in covered])
    w_covered = np.array([e["w_replicated"] for e in covered])
    design_T = np.array(design_days) / surface.TRADING_DAYS
    if design_T[0] < Ts[0] or design_T[-1] > Ts[-1]:
        raise ValueError("the chain does not bracket every design maturity")
    if not len(covered) or design_T[0] < T_covered[0] - 1e-12 or design_T[-1] > T_covered[-1]:
        raise ValueError("the expirations whose strips cover the tails do not bracket every design maturity")
    w_design = np.interp(design_T, T_covered, w_covered)
    levels = surface.forward_variances(design_T, w_design)
    splines = [smile_spline(e["quotes"], e["T"], sigma_ref) for e in expirations]
    prices, noise, implied, missing = [], [], [], []
    def covers(idx, T, delta):
        _, (dmin, dmax), _ = splines[idx]
        return dmin <= delta * np.sqrt(T / Ts[idx]) <= dmax

    readings = {"two expirations": 0, "one expiration": 0, "extrapolated": 0}
    for T in design_T:
        j = int(np.searchsorted(Ts, T))
        nearest = (j, j) if np.isclose(Ts[min(j, len(Ts) - 1)], T) else (j - 1, j)
        nearest = (max(nearest[0], 0), min(nearest[1], len(Ts) - 1))
        for delta in deltas:
            # The nearest expirations on each side whose quotes reach this point: an expiration whose
            # strip stops short would extrapolate its spline, which is not data.
            below = [i for i in range(len(Ts)) if Ts[i] <= T + 1e-12 and covers(i, T, delta)]
            above = [i for i in range(len(Ts)) if Ts[i] >= T - 1e-12 and covers(i, T, delta)]
            inside = bool(below or above)
            if below and above:
                lo, hi = below[-1], above[0]
            elif inside:
                lo = hi = below[-1] if below else above[0]
            else:
                lo, hi = nearest
            readings["extrapolated" if not inside else "two expirations" if lo != hi else "one expiration"] += 1
            values, halves = [], []
            for idx in sorted({lo, hi}):
                spline, _, (deltas_q, half_q) = splines[idx]
                d = delta * np.sqrt(T / Ts[idx])  # the same log-moneyness at the expiration's own maturity
                values.append(float(spline(d)) ** 2 * Ts[idx])
                halves.append(float(np.interp(d, deltas_q, half_q)))
            if len(values) == 2:
                weight = (T - Ts[lo]) / (Ts[hi] - Ts[lo])
                total = (1 - weight) * values[0] + weight * values[1]
                half = (1 - weight) * halves[0] + weight * halves[1]
            else:
                total, half = values[0] * T / Ts[lo], halves[0]  # the expiration's volatility at this log-moneyness
            iv = np.sqrt(total / T)
            k = delta * sigma_ref * np.sqrt(T)
            kind = "put" if delta <= 0 else "call"
            implied.append(iv)
            prices.append(float(surface.bs_price(k, T, iv, kind)))
            vol_noise = max(half, float(surface.implied_vol_noise(delta)))
            noise.append(float(surface.bs_vega(k, T, sigma_ref) * vol_noise))
            missing.append(not inside)
    diagnostics = {
        "valuation": valuation.isoformat(),
        "discount_source": "rate curve" if rate is not None else "parity regression",
        "replication_deltas": list(replication_deltas),
        "expirations": [{key: e[key] for key in ("expiration", "days", "forward", "discount",
                                                 "parity_regression_discount", "w_replicated", "strip_deltas",
                                                 "replication_covered")}
                        | {"quotes_kept": len(e["quotes"])} for e in expirations],
        "design_total_variance": w_design.tolist(),
        "grid_points_outside_quoted_strikes": int(np.sum(missing)),
        "grid_point_readings": readings,
        "implied_vol_grid": implied,
    }
    return {"prices": np.array(prices), "noise_sd": np.array(noise), "forward_variance_levels": levels,
            "missing": np.array(missing), "diagnostics": diagnostics}
