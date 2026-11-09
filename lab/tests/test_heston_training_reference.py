"""Synthetic references retain the declared Gaussian support and target."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import norm

from scripts.run_heston_pilot_training_reference import SyntheticLikelihood, load_case


class FixedPricePilot:
    names = ["parameter"]
    bounds = np.array([[0., 2.]])
    records = {"mid": np.ones(2)}

    def prices(self, point):
        return np.array([.05, .2])

    def observation_sigma(self, point):
        return np.array([.1, .3])


class SyntheticReferenceTests(unittest.TestCase):
    def test_negative_observations_use_untruncated_normal_and_no_prior_jacobian(self):
        observations = np.array([-.2, .4])
        pilot = FixedPricePilot()
        likelihood = SyntheticLikelihood(pilot, observations)
        expected = norm.logpdf(observations, pilot.prices([1]), pilot.observation_sigma([1])).sum()
        self.assertAlmostEqual(likelihood([1]), expected)
        self.assertEqual(likelihood([.1]), likelihood([1.9]))
        self.assertEqual(likelihood([-1]), -np.inf)
        observations[:] = 9
        self.assertAlmostEqual(likelihood([1]), expected)

    def test_panel_design_mismatch_is_rejected_before_pricing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "design.json").write_text("{}")
            (root / "panel.json").write_text(json.dumps({"design_sha256": "a" * 64}))
            with self.assertRaisesRegex(ValueError, "differs from frozen quote design"):
                load_case(root / "design.json", root / "panel.json", 0)


if __name__ == "__main__":
    unittest.main()
