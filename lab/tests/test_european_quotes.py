import hashlib
import json
from datetime import datetime, timezone
import unittest

from sbi4finance.calibration.european_quotes import parse_spxw_snapshot


def payload(options, timestamp="2026-11-09 20:28:37"):
    return json.dumps({"symbol": "_SPX", "timestamp": timestamp,
                       "data": {"last_trade_time": "2026-11-09T15:13:35", "current_price": 7571.54,
                                "options": options}}).encode()


def option(symbol="SPXW261120C07500000", bid=100.0, ask=101.0):
    return {"option": symbol, "bid": bid, "ask": ask, "last_trade_time": "2026-11-09T10:29:50"}


class EuropeanQuotesTests(unittest.TestCase):
    def test_retains_exact_actual_quote_and_only_spxw_root(self):
        raw = payload([option(), option("SPX261120C07500000"), option("SPXO261120C07500000"),
                       option("SPXW261120P07500000", 99.0, 100.2)])
        result = parse_spxw_snapshot(raw, timestamp_zone="UTC")
        self.assertEqual(result["raw_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(result["quotes"]), 2)
        self.assertEqual(result["exclusion_counts"]["other_root"], 2)
        q = result["quotes"][0]
        self.assertEqual((q["symbol"], q["strike"], q["bid"], q["ask"], q["mid"], q["half_spread"]),
                         ("SPXW261120C07500000", 7500.0, 100.0, 101.0, 100.5, 0.5))
        self.assertEqual(q["settlement_time_utc"], "2026-11-20T21:00:00+00:00")
        self.assertEqual(q["last_trade_time_raw"], "2026-11-09T10:29:50")
        self.assertEqual(q["last_trade_time_utc"], "2026-11-09T15:29:50+00:00")
        expected = (datetime(2026, 11, 20, 21, tzinfo=timezone.utc)
                    - datetime(2026, 11, 9, 20, 28, 37, tzinfo=timezone.utc)).total_seconds() / (365 * 86400)
        self.assertAlmostEqual(q["maturity"], expected, places=15)

    def test_excludes_early_close_same_day_and_bad_markets(self):
        raw = payload([option(), option("SPXW261109C07500000"), option("SPXW261127C07500000"),
                       option("SPXW261121C07500000", 0, 1),
                       option("SPXW261120C07600000", 0, 1),
                       option("SPXW261120C07700000", 1, 1),
                       option("SPXW261120C07800000", 2, 1),
                       option("SPXW261120C07900000", float("nan"), 2)])
        # Remove the deliberately malformed weekend expiry; a strict parser
        # rejects contract-convention errors, separately from market filters.
        with self.assertRaisesRegex(ValueError, "weekend"):
            parse_spxw_snapshot(raw, timestamp_zone="UTC")
        data = json.loads(raw)
        data["data"]["options"].pop(3)
        result = parse_spxw_snapshot(json.dumps(data), timestamp_zone="UTC")
        self.assertEqual(len(result["quotes"]), 1)
        self.assertEqual(result["exclusion_counts"], {
            "same_day_or_past_expiry": 1, "excluded_early_close_expiry": 1,
            "zero_bid_or_locked_crossed_market": 3, "nonfinite_or_invalid_market": 1})

    def test_accounts_for_daylight_saving_and_aware_override(self):
        raw = payload([option("SPXW270319C07500000")])
        override = datetime(2026, 11, 9, 20, 13, 37, tzinfo=timezone.utc)
        result = parse_spxw_snapshot(raw, timestamp_zone="UTC", valuation_time=override)
        q = result["quotes"][0]
        self.assertEqual(q["settlement_time_utc"], "2027-03-19T20:00:00+00:00")
        self.assertEqual(result["valuation_time_utc"], override.isoformat())
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            parse_spxw_snapshot(raw, timestamp_zone="UTC", valuation_time=datetime(2026, 11, 9))

    def test_duplicate_symbols_and_malformed_symbols_fail(self):
        for options, message in (([option(), option()], "duplicate"), ([option("not-an-OCC-symbol")], "invalid OCC")):
            with self.assertRaisesRegex(ValueError, message):
                parse_spxw_snapshot(payload(options), timestamp_zone="UTC")

    def test_empty_usable_selection_and_wrong_underlying_fail(self):
        with self.assertRaisesRegex(ValueError, "no usable"):
            parse_spxw_snapshot(payload([option("SPX261120C07500000")]), timestamp_zone="UTC")
        data = json.loads(payload([option()]))
        data["symbol"] = "SPY"
        with self.assertRaisesRegex(ValueError, "SPX snapshot"):
            parse_spxw_snapshot(json.dumps(data), timestamp_zone="UTC")


if __name__ == "__main__":
    unittest.main()
