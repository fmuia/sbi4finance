"""SPY's quotes against the design grid read from them (course, Lecture 5, Section 5.4).

    python checks/spy_quotes/spy_quotes.py RAW_CHAIN.json OUT.json [--rate-curve 0.25:0.0426,2:0.0479]

RAW_CHAIN.json is a Cboe delayed-quotes file; it is not committed, and data/SOURCES.md gives the URL. The rate
curve defaults to the one the tutorial's SPY surface was built with. The steps are those of
sbi4finance.calibration.chain.build_surface, rerun per expiration so that the quotes it reads can be kept.
OUT.json keeps derived data only:

  * per expiration: maturity, forward, discount, the strip's reach in standardized moneyness and whether it
    covers the replication range;
  * per out-of-the-money quote with a positive bid below its ask: log-moneyness, the implied volatilities of the
    bid, the mid and the ask, and whether cleaning kept it; the count of quotes with no bid;
  * the smoothing spline of chain.smile_spline on 201 points across the kept quotes;
  * the 63 design implied volatilities and the SHA-256 of the raw file.

Run from lab/. course/scripts/lecture05.py draws the figure from OUT.json and checks that its SHA-256 is the one
of the tutorial's SPY surface.
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from sbi4finance.calibration import chain, surface

DEFAULT_RATE_CURVE = "0.25:0.0426,2:0.0479"  # tutorial/assets/results/calibration/spy/surface.json, rate_curve


def quote_vols(rows, forward, discount, maturity):
    """Implied volatilities of the bid, the mid and the ask of every out-of-the-money quote with a usable bid."""
    out, no_bid = [], 0
    for r in rows:
        k = float(np.log(r["strike"] / forward))
        kind = "put" if k <= 0 else "call"
        if r["type"] != kind:
            continue
        if r["bid"] <= 0:
            no_bid += 1
            continue
        if r["ask"] <= r["bid"]:
            continue
        vols = [float(surface.implied_vol(p / (discount * forward), k, maturity, kind))
                for p in (r["bid"], 0.5 * (r["bid"] + r["ask"]), r["ask"])]
        out.append({"k": k, "iv_bid": vols[0], "iv_mid": vols[1], "iv_ask": vols[2]})
    out.sort(key=lambda q: q["k"])
    return out, no_bid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw")
    parser.add_argument("out")
    parser.add_argument("--rate-curve", default=DEFAULT_RATE_CURVE, help="T:yield pairs (bond-equivalent yields)")
    args = parser.parse_args()
    raw = Path(args.raw).read_bytes()
    rows = chain.parse_cboe_json(raw.decode())
    points = [tuple(float(x) for x in pair.split(":")) for pair in args.rate_curve.split(",")]
    result = chain.build_surface(rows, rate=chain.rate_curve(points))
    diag = result["diagnostics"]
    by_expiry = {}
    for r in rows:
        by_expiry.setdefault(r["expiration"].isoformat(), []).append(r)

    expirations = []
    for e in diag["expirations"]:
        T = e["days"] / surface.TRADING_DAYS
        quotes, no_bid = quote_vols(by_expiry[e["expiration"]], e["forward"], e["discount"], T)
        kept = chain.clean_expiration(by_expiry[e["expiration"]], e["forward"], e["discount"], T)
        kept_k = np.array([q["k"] for q in kept])
        for q in quotes:
            q["kept"] = bool(kept_k.size and np.min(np.abs(kept_k - q["k"])) < 1e-12)
        spline, (dmin, dmax), _ = chain.smile_spline(kept, T)
        delta = np.linspace(dmin, dmax, 201)
        expirations.append({
            "expiration": e["expiration"], "days": e["days"], "forward": e["forward"], "discount": e["discount"],
            "strip_deltas": e["strip_deltas"], "replication_covered": e["replication_covered"],
            "quotes_no_bid": no_bid,
            "quotes": [{key: round(v, 6) if isinstance(v, float) else v for key, v in q.items()} for q in quotes],
            "spline": {"k": np.round(delta * surface.SIGMA_REF * np.sqrt(T), 6).tolist(),
                       "iv": np.round(spline(delta), 6).tolist()},
        })

    out = {
        "valuation": diag["valuation"],
        "chain_sha256": hashlib.sha256(raw).hexdigest(),
        "rate_curve": {"points_bond_equivalent": points},
        "replication_deltas": diag["replication_deltas"],
        "design": {"days": list(surface.DESIGN_DAYS), "deltas": list(surface.DESIGN_DELTAS),
                   "sigma_ref": surface.SIGMA_REF},
        "implied_vol_grid": [round(float(v), 6) for v in diag["implied_vol_grid"]],
        "expirations": expirations,
        "written": date.today().isoformat(),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(expirations)} expirations, {sum(len(e['quotes']) for e in expirations)} quotes -> {args.out}")


if __name__ == "__main__":
    main()
