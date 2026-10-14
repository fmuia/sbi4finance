"""Tests for the advanced lesson's helpers: python -m unittest tests/test_svj_tutorial.py"""

import json
import unittest
from pathlib import Path

import numpy as np

from sbi4finance.svj import filtering, model, teaching, tutorial

ROOT = Path(__file__).resolve().parents[1]


class FilteredPath(unittest.TestCase):
    def test_tracks_the_true_spot_volatility(self):
        _, simulation = teaching.synthetic_case()
        returns, truth = simulation["returns"][0], simulation["targets"][0]
        spot_vol, jump_prob = tutorial.filtered_path(model.model_params(truth), returns, seed=0, n_particles=500)
        self.assertEqual(spot_vol.shape, (model.WINDOW, 3))
        self.assertTrue(np.all(np.diff(spot_vol, axis=1) >= 0))
        self.assertTrue(np.all((jump_prob >= 0) & (jump_prob <= 1)))
        true_vol = model.spot_vol_from_variance(simulation["variance"][0])
        self.assertGreater(np.corrcoef(spot_vol[:, 1], true_vol)[0, 1], 0.6)
        inside = (true_vol >= spot_vol[:, 0]) & (true_vol <= spot_vol[:, 2])
        self.assertTrue(0.8 < inside.mean() < 0.97)  # a 90% band, day by day
        jump_days = np.flatnonzero(simulation["jumps"][0])
        self.assertGreater(jump_prob[jump_days].mean(), 10 * jump_prob.mean())

    def test_runs_the_pipeline_filter(self):
        _, simulation = teaching.synthetic_case()
        params = model.model_params(simulation["targets"][0])
        returns = simulation["returns"][0]
        probs = (0.1, 0.5, 0.9)
        spot_vol, _ = tutorial.filtered_path(params, returns, seed=4, n_particles=300, probs=probs)
        _, cloud = filtering.filtered_variance(params, returns, seed=4, n_particles=300)
        np.testing.assert_allclose(
            spot_vol[-1], model.spot_vol_from_variance(np.quantile(cloud, probs)), rtol=1e-5
        )


class AgreementSeries(unittest.TestCase):
    def test_matches_the_recorded_study(self):
        recorded = json.loads((ROOT / "docs/results/svj-agreement/agreement.json").read_text())["series"]
        for k, record in enumerate(recorded):
            returns, targets, name = tutorial.agreement_series(k)
            self.assertEqual(name, record["series"])
            self.assertEqual(returns.shape, (model.WINDOW,))
            np.testing.assert_allclose(
                model.quantities(targets), list(record["truth"].values()), atol=1e-4, rtol=0
            )

    def test_rejects_unknown_series(self):
        with self.assertRaises(ValueError):
            tutorial.agreement_series(4)


if __name__ == "__main__":
    unittest.main()
