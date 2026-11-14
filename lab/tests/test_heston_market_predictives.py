"""Saved market bands preserve source identities and paired raw noise scales."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prepare_heston_market_predictives import predict


LAB = Path(__file__).resolve().parents[1]
GRID = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid"
DATA = LAB / "notebooks/data/heston_market_predictives"
spec = importlib.util.spec_from_file_location("market_predictives_under_test", LAB / "notebooks/_heston_market_predictives.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class HestonMarketPredictiveTests(unittest.TestCase):
    def setUp(self):
        evaluation = json.loads((GRID / "recovery-bank-independent/300k-300/evaluation/evaluation.json").read_text())
        task = json.loads((GRID / "cv-preparation/task.json").read_text())
        panel = json.loads((GRID / "cv-preparation/conditional-panel.json").read_text())
        references = json.loads((LAB / "evidence/pilot-results/2026-11-11/heston-actual63-notebook/reference-comparison.json").read_text())
        self.arguments = dict(evaluation=evaluation, task=task,
                              observations=next(c for c in panel["cases"] if c["id"] == "market")["observations"],
                              half_spread=task["records"]["half_spread"], reference_sha256=references["display_cache"]["sha256"])

    def test_saved_bands_bind_to_actual63_and_preserve_raw_support_violations(self):
        record = loader.load_results(DATA, **self.arguments)
        neural = record["methods"]["amortizer"]
        self.assertEqual((neural["raw_draw_count"], neural["priced_draw_count"], neural["outside_prior_count"]), (4096, 256, 853))
        self.assertEqual(neural["outside_pricing_bounds_count"], 0)
        self.assertGreater(neural["selected_c_outside_prior_count"], 0)
        self.assertGreater(neural["c_q05_q50_q95"][-1], 2)
        self.assertEqual(record["methods"]["pymc"]["raw_draw_count"], 2000)

    def test_mismatched_sources_or_display_payload_are_rejected(self):
        for key in ("reference_sha256", "observations", "half_spread", "evaluation", "task"):
            arguments = deepcopy(self.arguments)
            if key == "reference_sha256":
                arguments[key] = "0" * 64
            elif key in ("observations", "half_spread"):
                arguments[key][0] += .01
            elif key == "evaluation":
                arguments[key]["training_rows"] -= 1
            else:
                arguments[key]["target"]["prior_bounds"][-1][-1] = 3
            with self.subTest(key=key), self.assertRaises(ValueError):
                loader.load_results(DATA, **arguments)
        record = json.loads((DATA / "results.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            record["methods"]["amortizer"]["clean_q05_q50_q95_raw"][0][0] = float("nan")
            (Path(temporary) / "results.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                loader.load_results(temporary, **self.arguments)

    def test_noise_uses_each_physical_rows_own_c_including_outside_prior(self):
        class Pilot:
            records = {"half_spread": np.array([1., 2.])}
            def prices(self, row):
                return np.array([row[0], 2 * row[0]])
        draws = np.array([[1, 1, 1, 1, -.5, .8], [2, 1, 1, 1, -.5, 1.2],
                          [3, 1, 1, 1, -.5, 2.5], [4, 1, 1, 1, -.5, 3.]])
        bounds = np.array([[.1, 5], [.1, 5], [.1, 5], [.1, 5], [-.9, -.1], [.5, 2.]])
        result = predict(draws, bounds, Pilot(), seed=17, method_index=1, count=4)
        selected = np.random.default_rng([17, 1, 0]).choice(4, 4, replace=False)
        paired = draws[selected]
        clean = np.array([Pilot().prices(row) for row in paired])
        noisy = clean + paired[:, -1:] * Pilot.records["half_spread"] * np.random.default_rng([17, 1, 1]).standard_normal(clean.shape)
        np.testing.assert_array_equal(result["noisy_q05_q50_q95_raw"], np.quantile(noisy, [.05, .5, .95], axis=0))
        self.assertEqual(result["outside_prior_count"], 2)
        self.assertEqual(result["selected_c_outside_prior_count"], 2)
        self.assertEqual(result["priced_draw_count"], 4)


if __name__ == "__main__":
    unittest.main()
