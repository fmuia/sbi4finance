"""Why a Cboe chain fails the full-design gate: each expiration's cleaned strip reach against the replication deltas.

The SPY pipeline (chain.build_surface) replicates variance swaps only on expirations whose cleaned strip reaches
the replication deltas on both sides, and needs such expirations on both sides of every design maturity. This
prints, for the short and the long end, each expiration's reach, beside the 21 October SPY diagnostics.

Run from lab/: python checks/why_bayes/chain_reach.py CHAIN.json ../tutorial/assets/results/calibration/spy/surface.json
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np

from sbi4finance.calibration import chain, surface

defaults = {k: v.default for k, v in inspect.signature(chain.build_surface).parameters.items()
            if v.default is not inspect.Parameter.empty}
rows = chain.parse_cboe_json(Path(sys.argv[1]).read_text())
rate = chain.rate_curve([(0.25, 0.0426), (2.0, 0.0479)])  # the 20 October Treasury yields; a diagnostic, not a study
print("rows", len(rows), "valuation", rows[0]["date"], "replication deltas", defaults["replication_deltas"])
by_expiry = {}
for r in rows:
    if r["expiration"] > rows[0]["date"]:
        by_expiry.setdefault(r["expiration"], []).append(r)
print("this chain, expirations to 12 and beyond 200 trading days:")
for expiry, rows_e in sorted(by_expiry.items()):
    days = chain.trading_days(rows[0]["date"], expiry)
    if 12 < days < 200:
        continue
    fd = chain.forward_and_discount(rows_e)
    if fd is None:
        print(f"  {expiry} {days}d: no forward")
        continue
    T = days / surface.TRADING_DAYS
    discount = float(np.exp(-rate(T) * T))
    forward = chain.forward_given_discount(rows_e, discount)
    quotes = chain.clean_expiration(rows_e, forward, discount, T, defaults["max_spread"])
    two_sided = sum(1 for r in rows_e if r.get("bid", 0) > 0 and r.get("ask", 0) > 0)
    if len(quotes) < 8:
        print(f"  {expiry} {days}d: {len(rows_e)} rows, {two_sided} two-sided, {len(quotes)} cleaned quotes")
        continue
    k = np.array([q["k"] for q in quotes])
    reach = (k.min() / (defaults["sigma_ref"] * np.sqrt(T)), k.max() / (defaults["sigma_ref"] * np.sqrt(T)))
    print(f"  {expiry} {days}d: {len(rows_e)} rows, {two_sided} two-sided, {len(quotes)} cleaned, "
          f"reach {reach[0]:.2f}..{reach[1]:.2f}")
spy = json.loads(Path(sys.argv[2]).read_text())
print("21 October SPY, the same expirations:")
for e in spy["diagnostics"]["expirations"]:
    if e["days"] <= 12 or e["days"] >= 200:
        print(f"  {e['expiration']} {e['days']}d: reach {e['strip_deltas'][0]:.2f}..{e['strip_deltas'][1]:.2f}, "
              f"covered {e['replication_covered']}")
