"""Thin option chains in implied-volatility units (Cboe delayed quotes): EWI, EWG, EWQ and FEZ.

    python thin_chain.py TICKER RAW_CHAIN.json OUT.json

RAW_CHAIN.json is the Cboe delayed-quotes file; it is not committed (decision D6), and data/SOURCES.md gives the
URL each file was downloaded from. settings.json gives, per ticker, the as-of date, spot, rate and dividends,
with their sources. OUT.json keeps derived data only. Per out-of-the-money quote: the log-moneyness, the implied
volatilities of the bid, the ask and the mid, the vendor's implied volatility, the open interest, the day's
volume and the early-exercise premium in units of the half-spread. Per expiry: the counts, the median bid-ask
width in volatility units, the variance-swap strip over the quoted strikes on bids and on asks, and the rate and
forward a put-call parity regression would imply. A summary gives the numbers used to choose the thin example.

Assumptions, stated once:
  * forwards: (S - PV(dividends before T)) e^{rT}, with the flat rate of settings.json;
  * American exercise: a binomial tree with escrowed dividends, 600 steps, at each quote's own mid implied
    volatility (25% where the quote has no bid);
  * expiries closer than three calendar days are dropped.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
NEAR_MONEY = 0.10  # |log-moneyness| of the quotes summarized as "near the money"
MIN_DAYS = 3


class Chain:
    def __init__(self, settings):
        self.asof = date.fromisoformat(settings["asof"])
        self.spot = float(settings["spot"])
        self.rate = float(settings["rate"])
        one_year = [(date.fromisoformat(d), float(a)) for d, a in settings["dividends"]]
        self.dividends = [(d.replace(year=d.year + y), a) for y in range(int(settings["repeat_years"]))
                          for d, a in one_year]

    def tau(self, d):
        return (d - self.asof).days / 365.0

    def pv_dividends(self, start, end):
        return sum(a * np.exp(-self.rate * (self.tau(d) - start)) for d, a in self.dividends
                   if start < self.tau(d) < end)

    def forward(self, T):
        return (self.spot - self.pv_dividends(0.0, T)) * np.exp(self.rate * T)

    def tree(self, K, T, s, cp, american, n=600):
        """CRR tree on the spot net of the dividends before T; early exercise against the cum-dividend spot."""
        pv = self.pv_dividends(0.0, T)
        dt = T / n
        u = np.exp(s * np.sqrt(dt))
        dn = 1 / u
        p = (np.exp(self.rate * dt) - dn) / (u - dn)
        disc = np.exp(-self.rate * dt)
        j = np.arange(n + 1)
        x = (self.spot - pv) * u ** (n - j) * dn ** j
        v = np.maximum(x - K, 0) if cp == "C" else np.maximum(K - x, 0)
        for i in range(n - 1, -1, -1):
            t = i * dt
            x = (self.spot - pv) * u ** (i - np.arange(i + 1)) * dn ** np.arange(i + 1)
            v = disc * (p * v[:-1] + (1 - p) * v[1:])
            if american:
                pvt = self.pv_dividends(t, T)
                v = np.maximum(v, (x + pvt - K) if cp == "C" else (K - x - pvt))
        return float(v[0])


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


def parity_regression(rows, T):
    """Weighted least squares of C_mid - P_mid = D F - D K over the strikes where both have a bid.

    Weights are the inverse of the summed half-spreads. Returns the number of pairs, the implied rate -log(D)/T
    and the implied forward, which the conversion does not use: on a thin chain they are not credible.
    """
    pairs = []
    for K in sorted({r["K"] for r in rows}):
        c = next((r for r in rows if r["K"] == K and r["cp"] == "C"), None)
        p = next((r for r in rows if r["K"] == K and r["cp"] == "P"), None)
        if c and p and c["bid"] > 0 and p["bid"] > 0 and c["ask"] > c["bid"] and p["ask"] > p["bid"]:
            y = 0.5 * (c["bid"] + c["ask"]) - 0.5 * (p["bid"] + p["ask"])
            w = 1.0 / (0.5 * (c["ask"] - c["bid"]) + 0.5 * (p["ask"] - p["bid"]))
            pairs.append((K, y, w))
    if len(pairs) < 3:
        return {"pairs": len(pairs), "implied_rate": None, "implied_forward": None}
    k, y, w = (np.array(v) for v in zip(*pairs))
    (dF, D), *_ = np.linalg.lstsq(np.vstack([np.ones_like(k), -k]).T * w[:, None], y * w, rcond=None)
    return {"pairs": len(pairs), "implied_rate": float(-np.log(D) / T), "implied_forward": float(dF / D)}


def strip_vols(strikes, bids, asks, T, D):
    """The variance-swap strip over the quoted strikes only, (2/T) sum dK P(K)/K^2 on forward prices.

    Computed on bids and on asks. The quotes are American; their early-exercise premia are small against the
    spreads (see the quotes' field). Strikes beyond the quoted ones are left out, so both values understate the
    full strip.
    """
    dk = np.gradient(strikes) if len(strikes) > 1 else np.zeros(len(strikes))
    return [float(np.sqrt(max(2 / T * np.sum(dk * (p / D) / strikes ** 2), 0.0))) for p in (bids, asks)]


def parse(raw):
    rows = []
    for q in raw["data"]["options"]:
        s = q["option"]
        root = len(s) - 15  # ticker, then YYMMDD, C or P, strike x 1000 on eight digits
        rows.append(dict(exp=date(2000 + int(s[root:root + 2]), int(s[root + 2:root + 4]), int(s[root + 4:root + 6])),
                         cp=s[root + 6], K=int(s[root + 7:]) / 1000, bid=q["bid"], ask=q["ask"], vendor=q["iv"],
                         oi=q["open_interest"], volume=q["volume"]))
    return rows


def convert(ticker, raw, settings):
    chain = Chain(settings)
    rows = parse(raw)
    out = {"ticker": ticker, "name": settings["name"],
           "source": f"Cboe delayed quotes for {ticker}, file timestamp {raw['timestamp']}; "
                     f"quotes of the {settings['asof']} close",
           "raw_quotes_committed": False, "spot": chain.spot, "rate": chain.rate,
           "dividends": [[str(d), a] for d, a in chain.dividends],
           "options_in_file": len(rows), "volume_total": float(sum(r["volume"] for r in rows)),
           "open_interest_total": float(sum(r["oi"] for r in rows)), "expiries": []}
    for exp in sorted({r["exp"] for r in rows}):
        if (exp - chain.asof).days < MIN_DAYS:
            continue
        T = chain.tau(exp)
        D = np.exp(-chain.rate * T)
        F = chain.forward(T)
        quotes, strikes, bids, asks = [], [], [], []
        for r in sorted((r for r in rows if r["exp"] == exp), key=lambda r: r["K"]):
            if not ((r["cp"] == "P" and r["K"] <= F) or (r["cp"] == "C" and r["K"] > F)) or r["ask"] <= 0:
                continue
            strikes.append(r["K"])
            bids.append(r["bid"])
            asks.append(r["ask"])
            two_sided = r["bid"] > 0
            iv_mid = implied(0.5 * (r["bid"] + r["ask"]), F, r["K"], T, r["cp"], D) if two_sided else float("nan")
            s_ref = iv_mid if np.isfinite(iv_mid) else 0.25
            eep = chain.tree(r["K"], T, s_ref, r["cp"], True) - chain.tree(r["K"], T, s_ref, r["cp"], False)
            half = 0.5 * (r["ask"] - r["bid"])
            quotes.append(dict(cp=r["cp"], k=float(np.log(r["K"] / F)), two_sided=bool(two_sided),
                               iv_bid=implied(r["bid"], F, r["K"], T, r["cp"], D) if two_sided else float("nan"),
                               iv_ask=implied(r["ask"], F, r["K"], T, r["cp"], D), iv_mid=iv_mid,
                               vendor_iv=r["vendor"] if r["vendor"] > 0 else float("nan"),
                               early_exercise_over_half_spread=eep / half if half > 0 else float("nan"),
                               open_interest=r["oi"], volume=r["volume"]))
        if not quotes:
            continue
        spreads = [q["iv_ask"] - q["iv_bid"] for q in quotes if q["two_sided"]]
        near = [q["iv_ask"] - q["iv_bid"] for q in quotes if q["two_sided"] and abs(q["k"]) <= NEAR_MONEY]
        strip = strip_vols(np.array(strikes), np.array(bids), np.array(asks), T, D)
        out["expiries"].append(dict(
            expiry=str(exp), days=(exp - chain.asof).days, T=T, forward=F, discount=float(D),
            out_of_the_money=len(quotes), two_sided=int(sum(q["two_sided"] for q in quotes)),
            two_sided_near_money=len(near),
            median_iv_spread=float(np.nanmedian(spreads)) if spreads else float("nan"),
            median_iv_spread_near_money=float(np.nanmedian(near)) if near else float("nan"),
            strip_vol_on_bids=strip[0], strip_vol_on_asks=strip[1],
            strike_range_log_moneyness=[float(np.log(min(strikes) / F)), float(np.log(max(strikes) / F))],
            parity_regression=parity_regression([r for r in rows if r["exp"] == exp], T),
            quotes=quotes))
    usable = [e for e in out["expiries"] if e["days"] <= 366 and e["two_sided_near_money"] >= 5]
    out["summary"] = {
        "near_money_definition": f"two-sided out-of-the-money quotes with |log-moneyness| <= {NEAR_MONEY}",
        "expiries_to_one_year_with_five_near_money_quotes": len(usable),
        "their_days": [e["days"] for e in usable],
        "their_median_near_money_width_vol_points": [100 * e["median_iv_spread_near_money"] for e in usable]}
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


def main(ticker, raw_path, out_path):
    settings = json.loads((HERE / "settings.json").read_text())[ticker]
    out = convert(ticker, json.loads(Path(raw_path).read_text()), settings)
    Path(out_path).write_text(json.dumps(clean(out), indent=1))
    for e in out["expiries"]:
        width = e["median_iv_spread_near_money"]
        print(f"{ticker} {e['expiry']} ({e['days']} d): {e['out_of_the_money']} OTM quotes, {e['two_sided']} two-sided, "
              f"{e['two_sided_near_money']} near the money; median near-money width "
              f"{100 * width:.1f} vol points" if np.isfinite(width) else
              f"{ticker} {e['expiry']} ({e['days']} d): {e['out_of_the_money']} OTM quotes, no two-sided quote near the money")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
