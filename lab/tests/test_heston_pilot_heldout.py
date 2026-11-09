import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import ndtri

from scripts import check_heston_pilot_heldout as heldout


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence/pilot-results/2026-11-09/heston-observations"


class HestonPilotHeldoutTests(unittest.TestCase):
    def test_common_scoring_quotes_are_disjoint_from_both_fits_and_carry(self):
        primary, symmetric = (json.loads((EVIDENCE / "designs" / (name + ".json")).read_text())
                              for name in ("put_heavy", "symmetric"))
        frozen = json.loads((EVIDENCE / "heldout_design.json").read_text())
        snapshot = {"raw_sha256": primary["source_sha256"], "valuation_time_utc": frozen["valuation_time_utc"],
                    "quotes": frozen["selected_quotes"] + primary["selected_quotes"] + symmetric["selected_quotes"]}
        selected = heldout.select_common_quotes(snapshot, primary, symmetric)
        self.assertEqual(len(selected["selected_quotes"]), 30)
        self.assertEqual([q["symbol"] for q in selected["selected_quotes"]], [q["symbol"] for q in frozen["selected_quotes"]])
        symbols = {q["symbol"] for q in selected["selected_quotes"]}
        self.assertEqual(len(symbols), 30)
        excluded = {q["symbol"] for d in (primary, symmetric) for q in d["selected_quotes"]}
        excluded.update(s for carry in primary["carry"].values() for s in carry["anchor_symbols"])
        self.assertFalse(symbols & excluded)
        self.assertTrue(selected["all_targets_supported"])

    def test_predictive_mixture_quantiles_match_identical_normal_oracle(self):
        means = np.tile([10., -4.], (16, 1))
        sigmas = np.tile([2., .5], (16, 1))
        actual = heldout.mixture_interval(means, sigmas)
        expected = np.array([[10., -4.] + ndtri(p) * np.array([2., .5]) for p in (.05, .95)])
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_mismatched_reference_layout_rejected_before_loading_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"design_sha256": "wrong-design"}))
            (root / "chains.npz").write_bytes(b"do not read this mismatched payload")
            (root / "diagnostics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "layout hash"):
                heldout.score_reference(root, "expected-design", {})


if __name__ == "__main__":
    unittest.main()
