"""A thin chain: the EWI (iShares MSCI Italy) options of the 22 October 2026 close, in implied-volatility units.

    python ewi_chain.py RAW_CHAIN.json OUT.json

RAW_CHAIN.json is the Cboe delayed-quotes file for EWI (the format of `spy/surface.json`'s source). It is not
committed (decision D6: derived data only). OUT.json keeps, per out-of-the-money quote, the log-moneyness, the
implied volatilities of the bid, the ask and the mid, the vendor's implied volatility, the open interest and
the day's volume, and the early-exercise premium in units of the half-spread; per expiry, it keeps the
variance-swap strip over the quoted strikes, computed on the bids and on the asks, and the rate and forward a
put-call parity regression would imply.

Assumptions, stated once:
  * spot: the 22 Oct close, 55.02 (the file's after-hours bid and ask, 51.95 and 56.73, are not used);
  * rate: 4.22%, the 3-month Treasury yield of 22 Oct 2026, flat (parity on this chain cannot pin a rate:
    a regression of C - P on K returns implied rates from -4.6% to 56%);
  * dividends: 0.70 ex 15 Dec 2026 and 1.17 ex 14 Jun 2027, the last December and June amounts paid
    (Alpha Vantage DIVIDENDS); the ex-dates are assumed;
  * forwards: (S - PV(dividends)) e^{rT};
  * American exercise: a binomial tree with escrowed dividends, 600 steps, at each quote's own mid implied
    volatility (25% where the quote has no bid).
"""

import json
import sys
from datetime import date

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ASOF = date(2026, 10, 22)
SPOT = 55.02
RATE = 0.0422
DIVIDENDS = [(date(2026, 12, 15), 0.70), (date(2027, 6, 14), 1.17)]


def tau(d):
    return (d - ASOF).days / 365.0


def forward(T):
    pv = sum(a * np.exp(-RATE * tau(d)) for d, a in DIVIDENDS if tau(d) < T)
    return (SPOT - pv) * np.exp(RATE * T)


def black(F, K, T, s, cp, D):
    if s <= 0:
        return D * max((F - K) if cp == "C" else (K - F), 0.0)
    d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2)) if cp == "C" else D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied(p, F, K, T, cp, D):
    if p <= black(F, K, T, 0.0, cp, D) + 1e-10:
        return float("nan")
    try:
        return brentq(lambda s: black(F, K, T, s, cp, D) - p, 1e-4, 5.0)
    except ValueError:
        return float("nan")


def tree(K, T, s, cp, american, n=600):
    """CRR tree on the spot net of the dividends before T; early exercise against the cum-dividend spot."""
    pv = sum(a * np.exp(-RATE * tau(d)) for d, a in DIVIDENDS if tau(d) < T)
    dt = T / n
    u = np.exp(s * np.sqrt(dt))
    dn = 1 / u
    p = (np.exp(RATE * dt) - dn) / (u - dn)
    disc = np.exp(-RATE * dt)
    j = np.arange(n + 1)
    x = (SPOT - pv) * u ** (n - j) * dn ** j
    v = np.maximum(x - K, 0) if cp == "C" else np.maximum(K - x, 0)
    for i in range(n - 1, -1, -1):
        t = i * dt
        x = (SPOT - pv) * u ** (i - np.arange(i + 1)) * dn ** np.arange(i + 1)
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        if american:
            pvt = sum(a * np.exp(-RATE * (tau(d) - t)) for d, a in DIVIDENDS if t < tau(d) < T)
            v = np.maximum(v, (x + pvt - K) if cp == "C" else (K - x - pvt))
    return float(v[0])


def parity_regression(rows, T):
    """Weighted least squares of C_mid - P_mid = D F - D K over the strikes where both have a bid.

    Weights are the inverse of the summed half-spreads. Returns the number of pairs, the implied rate -log(D)/T
    and the implied forward, which the conversion does not use: on a thin chain they are not credible.
    """
    pairs = []
    for K in sorted({r["K"] for r in rows}):
        c = next((r for r in rows if r["K"] == K and r["cp"] == "C"), None)
        p = next((r for r in rows if r["K"] == K and r["cp"] == "P"), None)
        if c and p and c["bid"] > 0 and p["bid"] > 0:
            y = 0.5 * (c["bid"] + c["ask"]) - 0.5 * (p["bid"] + p["ask"])
            w = 1.0 / (0.5 * (c["ask"] - c["bid"]) + 0.5 * (p["ask"] - p["bid"]))
            pairs.append((K, y, w))
    k, y, w = (np.array(v) for v in zip(*pairs))
    (dF, D), *_ = np.linalg.lstsq(np.vstack([np.ones_like(k), -k]).T * w[:, None], y * w, rcond=None)
    return {"pairs": len(pairs), "implied_rate": float(-np.log(D) / T), "implied_forward": float(dF / D)}


def strip_vols(strikes, bids, asks, T, D):
    """The variance-swap strip over the quoted strikes only, (2/T) sum dK P(K)/K^2 on forward prices, on bids and on asks.

    The quotes are American; their early-exercise premia are small against the spreads (see the quotes' field).
    The strikes beyond the quoted ones are left out, so both values understate the full strip.
    """
    dk = np.gradient(strikes)
    return [float(np.sqrt(max(2 / T * np.sum(dk * (p / D) / strikes ** 2), 0.0))) for p in (bids, asks)]


def main(raw_path, out_path):
    raw = json.load(open(raw_path))
    rows = []
    for q in raw["data"]["options"]:
        s = q["option"]
        rows.append(dict(exp=date(2000 + int(s[3:5]), int(s[5:7]), int(s[7:9])), cp=s[9], K=int(s[10:]) / 1000,
                         bid=q["bid"], ask=q["ask"], vendor=q["iv"], oi=q["open_interest"], volume=q["volume"]))
    out = {"source": "Cboe delayed quotes for EWI, file timestamp " + raw["timestamp"] + "; quotes of the 2026-10-22 close",
           "raw_quotes_committed": False, "spot": SPOT, "rate": RATE,
           "dividends": [[str(d), a] for d, a in DIVIDENDS],
           "options_in_file": len(rows), "volume_total": float(sum(r["volume"] for r in rows)),
           "open_interest_total": float(sum(r["oi"] for r in rows)), "expiries": []}
    for exp in sorted({r["exp"] for r in rows}):
        T = tau(exp)
        D = np.exp(-RATE * T)
        F = forward(T)
        quotes, strikes, bids, asks = [], [], [], []
        for r in sorted((r for r in rows if r["exp"] == exp), key=lambda r: r["K"]):
            if not ((r["cp"] == "P" and r["K"] <= F) or (r["cp"] == "C" and r["K"] > F)):
                continue
            strikes.append(r["K"])
            bids.append(r["bid"])
            asks.append(r["ask"])
            two_sided = r["bid"] > 0
            iv_mid = implied(0.5 * (r["bid"] + r["ask"]), F, r["K"], T, r["cp"], D) if two_sided else float("nan")
            s_ref = iv_mid if np.isfinite(iv_mid) else 0.25
            eep = tree(r["K"], T, s_ref, r["cp"], True) - tree(r["K"], T, s_ref, r["cp"], False)
            quotes.append(dict(cp=r["cp"], k=float(np.log(r["K"] / F)), two_sided=bool(two_sided),
                               iv_bid=implied(r["bid"], F, r["K"], T, r["cp"], D) if two_sided else float("nan"),
                               iv_ask=implied(r["ask"], F, r["K"], T, r["cp"], D), iv_mid=iv_mid,
                               vendor_iv=r["vendor"] if r["vendor"] > 0 else float("nan"),
                               early_exercise_over_half_spread=eep / (0.5 * (r["ask"] - r["bid"])),
                               open_interest=r["oi"], volume=r["volume"]))
        spreads = [q["iv_ask"] - q["iv_bid"] for q in quotes if q["two_sided"]]
        strip = strip_vols(np.array(strikes), np.array(bids), np.array(asks), T, D)
        out["expiries"].append(dict(expiry=str(exp), days=(exp - ASOF).days, T=T, forward=F, discount=float(D),
                                    out_of_the_money=len(quotes), two_sided=int(sum(q["two_sided"] for q in quotes)),
                                    median_iv_spread=float(np.nanmedian(spreads)),
                                    strip_vol_on_bids=strip[0], strip_vol_on_asks=strip[1],
                                    strike_range_log_moneyness=[float(np.log(min(strikes) / F)),
                                                                float(np.log(max(strikes) / F))],
                                    parity_regression=parity_regression([r for r in rows if r["exp"] == exp], T),
                                    quotes=quotes))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        return o

    json.dump(clean(out), open(out_path, "w"), indent=1)
    for e in out["expiries"]:
        print(f"{e['expiry']} ({e['days']} d): forward {e['forward']:.2f}, {e['out_of_the_money']} OTM quotes, "
              f"{e['two_sided']} with a bid, median bid-ask width {100 * e['median_iv_spread']:.1f} vol points, "
              f"strip vol on bids {100 * e['strip_vol_on_bids']:.1f}%, on asks {100 * e['strip_vol_on_asks']:.1f}%, "
              f"parity-implied rate {100 * e['parity_regression']['implied_rate']:.1f}% "
              f"({e['parity_regression']['pairs']} pairs)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
