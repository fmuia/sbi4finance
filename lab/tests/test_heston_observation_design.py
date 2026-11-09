"""Parity recovery and information separation for the frozen SPXW pilot."""

import json
from pathlib import Path
import unittest

import numpy as np

from scripts import run_heston_observation_pilot as runner
from sbi4finance.calibration.heston_pilot import black_price


DESIGNS = Path(__file__).resolve().parents[1] / "evidence/pilot-results/2026-11-09/heston-observations/designs"


class HestonObservationDesignTests(unittest.TestCase):
    def parity_fixture(self, scale=1.):
        spot, forward, discount, maturity = 100. * scale, 101. * scale, np.exp(-.03 * .5), .5
        quotes = []
        for i, strike in enumerate(np.linspace(96., 104., 49) * scale):
            for kind in ("call", "put"):
                mid = float(black_price(forward, strike, discount, maturity, .2, kind))
                half = (.02 + i * .0001) * scale
                quotes.append({"symbol": f"fixture:{i}:{kind}", "strike": float(strike),
                               "option_type": kind, "mid": mid, "half_spread": half,
                               "bid": mid - half, "ask": mid + half, "maturity": maturity})
        return spot, forward, discount, quotes

    def test_spread_weighted_parity_recovers_known_forward_discount_and_units(self):
        for scale in (1., 100.):
            spot, forward, discount, quotes = self.parity_fixture(scale)
            result = runner.carry_anchors(quotes, spot)
            self.assertAlmostEqual(result["forward"], forward, delta=1e-9 * scale)
            self.assertAlmostEqual(result["discount"], discount, places=12)
            self.assertAlmostEqual(result["rate"], .03, places=11)
            self.assertEqual(result["anchor_pairs"], 17)
            self.assertLess(result["parity_max_residual_over_combined_halfspread"], 1e-9)
            covariance = np.asarray(result["intercept_discount_covariance"])
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_sparse_parity_panel_fails_instead_of_claiming_precise_carry(self):
        spot, _, _, quotes = self.parity_fixture()
        with self.assertRaisesRegex(ValueError, "at least 15"):
            runner.carry_anchors(quotes[:20], spot)

    def test_frozen_layouts_share_carry_and_separate_its_quotes_from_smile(self):
        designs = {name: json.loads((DESIGNS / (name + ".json")).read_text()) for name in runner.TARGETS}
        self.assertEqual(designs["symmetric"]["carry"], designs["put_heavy"]["carry"])
        self.assertEqual(designs["symmetric"]["source_sha256"], designs["put_heavy"]["source_sha256"])
        for design in designs.values():
            self.assertEqual(len(design["selected_quotes"]), 54)
            self.assertEqual(sum(design["short_mask"]), 27)
            symbols = [q["symbol"] for q in design["selected_quotes"]]
            self.assertEqual(len(set(symbols)), len(symbols))
            anchors = {symbol for carry in design["carry"].values() for symbol in carry["anchor_symbols"]}
            self.assertFalse(set(symbols) & anchors)
            for q, short in zip(design["selected_quotes"], design["short_mask"]):
                self.assertEqual(q["root"], "SPXW")
                self.assertEqual(short, q["expiration"] in runner.EXPIRIES[:3])
                self.assertAlmostEqual(q["mid"], .5 * (q["bid"] + q["ask"]))
                self.assertAlmostEqual(q["half_spread"], .5 * (q["ask"] - q["bid"]))
                self.assertEqual(q["option_type"], "put" if q["strike"] <= q["forward"] else "call")

    def test_unavailable_symmetric_wings_remain_flagged_as_actual_contracts(self):
        for name in runner.TARGETS:
            design = json.loads((DESIGNS / (name + ".json")).read_text())
            unavailable = []
            for q in design["selected_quotes"]:
                self.assertEqual(q["target_supported"], abs(q["z"] - q["target_z"]) <= .35)
                if not q["target_supported"]:
                    unavailable.append((q["expiration"], q["target_z"]))
            self.assertEqual(unavailable, [] if name == "put_heavy" else [
                ("2027-06-30", 2), ("2027-09-30", 2)])


if __name__ == "__main__":
    unittest.main()
