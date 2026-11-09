"""A conservative SPXW snapshot reader for the European Heston pilot.

The legacy chain reader intentionally remains unchanged. This reader retains
actual quotes and contract identities; it neither interpolates smiles nor
estimates forwards, discounts, model noise or quote freshness. Cboe's delayed
JSON exposes last *trade* times, not bid/ask update times.
"""

from collections import Counter
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import re
from zoneinfo import ZoneInfo


CONTRACT_SOURCE = "https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications"
CALENDAR_SOURCE = "https://www.cboe.com/about/hours/us-options"
# The pilot excludes early-close expirations instead of pretending they settle
# at 16:00. Published exchange calendars cover these dates through 2028.
EARLY_CLOSE_EXPIRIES = frozenset((
    date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26),
    date(2028, 7, 3), date(2028, 11, 24),
))
_OCC = re.compile(r"(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})\Z")


def _aware_time(value, zone):
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=ZoneInfo(zone)) if parsed.tzinfo is None else parsed


def parse_spxw_snapshot(raw, *, timestamp_zone, trade_time_zone="America/New_York",
                        valuation_time=None, excluded_expiries=EARLY_CLOSE_EXPIRIES):
    """Return JSON-serializable metadata and positive, noncrossed SPXW quotes.

    ``timestamp_zone`` is required because the public payload timestamp is
    naive. For the downloaded pilot snapshot UTC is an explicit assumption,
    not a verified vendor guarantee. ``valuation_time`` may override that
    timestamp, but must be aware. Every maturity uses actual elapsed UTC
    seconds divided by 365 days (ACT/365F) to 16:00 America/New_York.

    Both call and put rows survive for separate parity estimation. Same-day,
    past, early-close, nonfinite, zero-bid and locked/crossed quotes are
    excluded with counts. No maturity-dependent spread floor is introduced.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    payload = json.loads(raw)
    data = payload["data"]
    if payload.get("symbol") not in ("_SPX", "SPX", "^SPX"):
        raise ValueError("expected a Cboe SPX snapshot")
    snapshot_time = _aware_time(payload["timestamp"], timestamp_zone)
    asof = snapshot_time if valuation_time is None else valuation_time
    if not isinstance(asof, datetime) or asof.tzinfo is None or asof.utcoffset() is None:
        raise ValueError("valuation_time must be a timezone-aware datetime")
    asof = asof.astimezone(timezone.utc)
    valuation_day = asof.astimezone(ZoneInfo("America/New_York")).date()
    quotes, excluded, roots, seen = [], Counter(), Counter(), set()
    for option in data["options"]:
        symbol = option["option"]
        match = _OCC.fullmatch(symbol)
        if match is None:
            raise ValueError(f"invalid OCC option symbol: {symbol!r}")
        root = match["root"]
        roots[root] += 1
        if root != "SPXW":
            excluded["other_root"] += 1
            continue
        if symbol in seen:
            raise ValueError(f"duplicate SPXW symbol: {symbol}")
        seen.add(symbol)
        expiry = datetime.strptime(match["expiry"], "%y%m%d").date()
        if expiry <= valuation_day:
            excluded["same_day_or_past_expiry"] += 1
            continue
        if expiry in excluded_expiries:
            excluded["excluded_early_close_expiry"] += 1
            continue
        if expiry.weekday() >= 5:
            raise ValueError(f"SPXW expiration falls on a weekend: {symbol}")
        bid, ask = float(option["bid"]), float(option["ask"])
        strike = int(match["strike"]) / 1000.0
        if not all(math.isfinite(x) for x in (bid, ask, strike)) or strike <= 0:
            excluded["nonfinite_or_invalid_market"] += 1
            continue
        if bid <= 0 or ask <= bid:
            excluded["zero_bid_or_locked_crossed_market"] += 1
            continue
        settlement = datetime.combine(expiry, time(16), ZoneInfo("America/New_York")).astimezone(timezone.utc)
        last_trade_raw = option.get("last_trade_time")
        quote = {
            "symbol": symbol, "root": root, "expiration": expiry.isoformat(),
            "option_type": "call" if match["kind"] == "C" else "put",
            "strike": strike, "bid": bid, "ask": ask,
            "mid": 0.5 * (bid + ask), "half_spread": 0.5 * (ask - bid),
            "maturity": (settlement - asof).total_seconds() / (365.0 * 86400.0),
            "settlement_time_utc": settlement.isoformat(),
            "last_trade_time_raw": last_trade_raw,
            "last_trade_time_utc": (_aware_time(last_trade_raw, trade_time_zone).astimezone(timezone.utc).isoformat()
                                    if last_trade_raw else None),
        }
        for key in ("bid_size", "ask_size", "open_interest", "volume"):
            if key in option and option[key] is not None and math.isfinite(float(option[key])):
                quote[key] = float(option[key])
        quotes.append(quote)
    quotes.sort(key=lambda q: (q["expiration"], q["strike"], q["option_type"]))
    if not quotes:
        raise ValueError("no usable future SPXW quotes")
    return {
        "schema": "spxw_actual_quotes_v1",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_timestamp_raw": payload["timestamp"],
        "snapshot_timestamp_utc": snapshot_time.astimezone(timezone.utc).isoformat(),
        "valuation_time_utc": asof.isoformat(),
        "underlying_last_trade_time_raw": data.get("last_trade_time"),
        "underlying_price": data.get("current_price"),
        "timestamp_zone_assumption": timestamp_zone,
        "trade_time_zone_assumption": trade_time_zone,
        "time_convention": "ACT/365F: actual UTC seconds / (365 * 86400)",
        "contract_convention": "SPXW European cash settlement at 16:00 America/New_York; early closes excluded",
        "contract_source": CONTRACT_SOURCE, "calendar_source": CALENDAR_SOURCE,
        "excluded_early_close_dates": sorted(d.isoformat() for d in excluded_expiries),
        "quality_limitations": [
            "Delayed intraday snapshot; no per-contract bid/ask update timestamps.",
            "Last-trade timestamps do not establish bid/ask freshness or synchronization.",
            "Timezone assignments for naive vendor fields are explicit assumptions.",
            "Snapshot generation time is used as valuation time unless explicitly overridden.",
        ],
        "input_quote_count": len(data["options"]), "root_counts": dict(sorted(roots.items())),
        "exclusion_counts": dict(sorted(excluded.items())), "quotes": quotes,
    }
