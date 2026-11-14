import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_heston_grid_coverage.py"
spec = importlib.util.spec_from_file_location("heston_grid_coverage_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def quote(strike, kind, *, forward=100., discount=.98, maturity=.5, symbol=None, half=.1):
    mid = 10. + discount * max((forward - strike) if kind == "call" else (strike - forward), 0.)
    return {"symbol": symbol or f"Q{strike:g}{kind}", "root": "SPXW", "expiration": "2027-01-29",
            "strike": strike, "option_type": kind, "maturity": maturity,
            "bid": mid - half, "ask": mid + half, "mid": mid, "half_spread": half}


class GridCoverageTests(unittest.TestCase):
    def sample_designs(self):
        rows = []
        for i, day in enumerate(audit.TARGET_DAYS):
            expiry = f"2026-11-{10+i:02d}"
            rows.extend(dict(quote(k, kind, maturity=day/252., discount=np.exp(-.04*day/252.),
                                   symbol=f"{expiry}-{k:g}-{kind}"),
                             expiration=expiry)
                        for k in np.linspace(95., 105., 91) for kind in ("call", "put"))
        first = [q for q in rows if q["expiration"] == "2026-11-10"]
        carry = audit.parity_carry(first, 100.)
        symbols = set(carry["anchor_symbols"])
        carry["anchor_quotes"] = [q for q in first if q["symbol"] in symbols]
        carry["forward"] = 100.123456789  # preserve this recorded estimate, never silently refit it
        available = [q for q in first if q["symbol"] not in symbols]
        baseline = {"source_sha256": "frozen", "expiries": ["2026-11-10"],
                    "selected_quotes": [available[0]], "carry": {"2026-11-10": carry}}
        comparison = copy.deepcopy(baseline)
        comparison["selected_quotes"] = [available[1]]
        scored = {"source_sha256": "frozen", "selected_quotes": [available[2]]}
        snapshot = {"quotes": rows, "underlying_price": 100., "raw_sha256": "frozen",
                    "valuation_time_utc": audit.VALUATION_TIME, "time_convention": "ACT/365F",
                    "exclusion_counts": {}}
        return snapshot, baseline, comparison, scored

    def test_full_candidate_preserves_old_carry_and_excludes_union_from_all_quotes(self):
        snapshot, baseline, comparison, scored = self.sample_designs()
        candidate, summary = audit.build_candidate(snapshot, baseline, comparison, scored)
        self.assertEqual(candidate["union_carry"]["2026-11-10"],
                         audit.compact_carry(baseline["carry"]["2026-11-10"]))
        anchors = {s for c in candidate["union_carry"].values() for s in c["anchor_symbols"]}
        used = {q["symbol"] for d in (baseline, comparison, scored, candidate) for q in d["selected_quotes"]}
        self.assertTrue(anchors.isdisjoint(used))
        self.assertTrue(summary["carry_disjoint_from_candidate_old_fitted_and_scored"])
        self.assertEqual(len(candidate["union_carry"]), 7)

    def test_changed_historical_quote_or_source_is_rejected(self):
        snapshot, baseline, comparison, scored = self.sample_designs()
        altered = copy.deepcopy(scored)
        altered["source_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "another raw snapshot"):
            audit.build_candidate(snapshot, baseline, comparison, altered)
        altered = copy.deepcopy(scored)
        altered["selected_quotes"][0]["mid"] += 1.
        with self.assertRaisesRegex(ValueError, "selected quote differs"):
            audit.build_candidate(snapshot, baseline, comparison, altered)

    def test_revision_is_explicit_and_preserves_original_design_inputs(self):
        snapshot, baseline, comparison, scored = self.sample_designs()
        original, _ = audit.build_candidate(snapshot, baseline, comparison, scored)
        revision, summary = audit.build_candidate(snapshot, baseline, comparison, scored, last_call_target=1.5)
        self.assertEqual(original["target_overrides"], [])
        self.assertEqual(original["selected_quotes"][-1]["target_z"], 2.)
        self.assertEqual(revision["selected_quotes"][-1]["target_z"], 1.5)
        self.assertEqual(len(revision["target_overrides"]), 1)
        self.assertFalse(summary["original_one_year_and_plus2_coverage_criteria_satisfied"])
        self.assertFalse(summary["scientific_acceptance"])

    def test_parity_recovers_known_carry_and_reserves_disjoint_pairs(self):
        rows = [quote(k, kind) for k in np.linspace(95., 105., 91) for kind in ("call", "put")]
        forbidden = {rows[0]["symbol"], rows[1]["symbol"]}
        carry = audit.parity_carry(rows, 100., forbidden)
        self.assertAlmostEqual(carry["forward"], 100., places=11)
        self.assertAlmostEqual(carry["discount"], .98, places=11)
        self.assertTrue(set(carry["anchor_symbols"]).isdisjoint(forbidden))
        chosen, report = audit.select_bucket(rows, carry, set(carry["anchor_symbols"]), set(), set(), targets=(-.1, 0., .1))
        self.assertEqual(len(chosen), 3)
        self.assertTrue({q["symbol"] for q in chosen}.isdisjoint(carry["anchor_symbols"]))
        self.assertTrue(all(t["target_supported"] for t in report["targets"]))

    def test_nearest_expiry_retains_exact_maturity_and_flags_duplicate(self):
        rows = [quote(100., "put", maturity=.02),
                dict(quote(100., "put", maturity=.9), expiration="2027-09-30")]
        rows[0]["expiration"] = "2026-11-16"
        buckets = audit.nearest_expiries(rows, target_days=(5, 10, 252))
        self.assertEqual([b["maturity"] for b in buckets], [.02, .02, .9])
        self.assertFalse(buckets[0]["duplicate_nearest_expiry"])
        self.assertTrue(buckets[1]["duplicate_nearest_expiry"])
        self.assertAlmostEqual(buckets[-1]["maturity_offset_calendar_days"], -36.5)

    def test_known_fixed16_coordinate_is_not_atm_adaptive(self):
        q = quote(100. * np.exp(.16 * np.sqrt(.5) * -3.5), "put")
        self.assertAlmostEqual(audit.standardized_z(q, 100.), -3.5)
        q["atm_iv"] = .8
        self.assertAlmostEqual(audit.standardized_z(q, 100.), -3.5)

    def test_spread_quality_scoring_and_duplicate_exclusions_remain_visible(self):
        rows = [quote(100., "put", symbol="anchor"), quote(100., "put", symbol="score"),
                quote(100., "put", symbol="wide", half=3.), quote(101., "put", symbol="itm"),
                quote(105., "call", symbol="good")]
        carry = {"forward": 100., "discount": .98}
        selected, report = audit.select_bucket(rows, carry, {"anchor"}, {"score"}, set(), targets=(1., 1.))
        self.assertEqual([q["symbol"] for q in selected], ["good"])
        self.assertEqual(report["selection_rejections"], {"carry_anchor": 1,
            "preserved_scoring_quote": 1, "halfspread_over_mid_exceeds_25pct": 1, "not_otm": 1})
        self.assertFalse(report["targets"][1]["target_supported"])
        self.assertEqual(report["targets"][1]["failure"], "no_unused_quality_eligible_actual_quote")

    def test_unavailable_wing_is_flagged_without_interpolation_or_replacement(self):
        rows = [quote(101., "call", symbol="first"), quote(102., "call", symbol="second")]
        selected, report = audit.select_bucket(rows, {"forward": 100., "discount": .98}, set(), set(), set(), targets=(1., 2.))
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({q["symbol"] for q in selected}), 2)
        self.assertFalse(report["targets"][-1]["target_supported"])
        self.assertEqual(selected[-1]["strike"], 101.)

    def test_source_hash_and_size_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, source = Path(tmp) / "raw.json", Path(tmp) / "source.json"
            raw.write_bytes(b"frozen")
            source.write_text(json.dumps({"raw_sha256": audit.digest(raw), "raw_size_bytes": 6}))
            self.assertEqual(audit.load_bound_raw(raw, source)[0], b"frozen")
            raw.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "hash or byte count"):
                audit.load_bound_raw(raw, source)

    def test_real_cli_help_does_not_require_historical_sources(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--historical-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
