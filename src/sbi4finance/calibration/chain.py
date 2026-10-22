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


def trading_days(valuation, expiration):
    """Business days from valuation to expiration (weekends excluded; exchange holidays are not)."""
    return int(np.busday_count(valuation, expiration))


def parse_alpha_vantage_csv(text):
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        rows.append({"expiration": _day(record["expiration"]), "strike": float(record["strike"]),
                     "type": record["type"].strip().lower(), "bid": float(record["bid"]), "ask": float(record["ask"]),
                     "date": _day(record["date"])})
    return rows


def parse_cboe_json(text):
    """Cboe delayed quotes: OCC symbols ROOT YYMMDD C/P strike*1000 (8 digits)."""
    payload = json.loads(text)
    data = payload.get("data", payload)
    valuation = _day(payload.get("timestamp", data.get("last_trade_time", "")))
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
                  sigma_ref=surface.SIGMA_REF, max_spread=0.5):
    """The 63 design prices, their noise sds and the 7 forward-variance levels, with diagnostics."""
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
        forward, discount = fd
        T = days / surface.TRADING_DAYS
        quotes = clean_expiration(chain, forward, discount, T, max_spread)
        if len(quotes) < 8:
            continue
        k = np.array([q["k"] for q in quotes])
        w = surface.total_variance_by_replication(k, [q["price"] for q in quotes])
        expirations.append({"expiration": expiry.isoformat(), "days": days, "T": T, "forward": forward,
                            "discount": discount, "quotes": quotes, "w_replicated": w})
    if not expirations:
        raise ValueError("no usable expiration after cleaning")
    Ts = np.array([e["T"] for e in expirations])
    ws = np.array([e["w_replicated"] for e in expirations])
    design_T = np.array(design_days) / surface.TRADING_DAYS
    if design_T[0] < Ts[0] or design_T[-1] > Ts[-1]:
        raise ValueError("the chain does not bracket every design maturity")
    w_design = np.interp(design_T, Ts, ws)
    levels = surface.forward_variances(design_T, w_design)
    splines = [smile_spline(e["quotes"], e["T"], sigma_ref) for e in expirations]
    prices, noise, implied, missing = [], [], [], []
    for T in design_T:
        j = int(np.searchsorted(Ts, T))
        lo, hi = (j, j) if np.isclose(Ts[min(j, len(Ts) - 1)], T) else (j - 1, j)
        lo, hi = max(lo, 0), min(hi, len(Ts) - 1)
        for delta in deltas:
            values, halves, inside = [], [], True
            for idx in sorted({lo, hi}):
                spline, (dmin, dmax), (deltas_q, half_q) = splines[idx]
                d = delta * np.sqrt(T / Ts[idx])  # the same log-moneyness at the expiration's own maturity
                inside &= dmin <= d <= dmax
                values.append(float(spline(d)) ** 2 * Ts[idx])
                halves.append(float(np.interp(d, deltas_q, half_q)))
            if len(values) == 2:
                weight = (T - Ts[lo]) / (Ts[hi] - Ts[lo])
                total = (1 - weight) * values[0] + weight * values[1]
                half = (1 - weight) * halves[0] + weight * halves[1]
            else:
                total, half = values[0], halves[0]
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
        "expirations": [{key: e[key] for key in ("expiration", "days", "forward", "discount", "w_replicated")}
                        | {"quotes_kept": len(e["quotes"])} for e in expirations],
        "design_total_variance": w_design.tolist(),
        "grid_points_outside_quoted_strikes": int(np.sum(missing)),
        "implied_vol_grid": implied,
    }
    return {"prices": np.array(prices), "noise_sd": np.array(noise), "forward_variance_levels": levels,
            "missing": np.array(missing), "diagnostics": diagnostics}
