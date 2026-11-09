"""The bounded pilot preserves quote/noise pairing across fixed layouts."""
import unittest

import numpy as np

from scripts import train_heston_pilot as trainer


class PilotTrainingTests(unittest.TestCase):
    def setUp(self):
        self.records = {"forward": np.full(12, 100.), "strike": np.tile([90., 100., 110.], 4),
                        "discount": np.ones(12), "maturity": np.repeat([.1, .2, .3, .5], 3),
                        "option_type": ["put"]*12, "mid": np.full(12, 5.),
                        "half_spread": np.linspace(.1, .3, 12)}

    def test_short_layout_omits_later_values_and_records_noise_explicitly(self):
        mask = trainer.layout_mask(self.records, "short")
        self.assertEqual(mask.tolist(), [True]*9+[False]*3)
        quotes = np.arange(24, dtype=float).reshape(2, 12)
        first = trainer.condition_data(quotes, np.array([.5, 2.]), self.records, mask)
        quotes[:, ~mask] = 1e10
        second = trainer.condition_data(quotes, np.array([.5, 2.]), self.records, mask)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first["log_noise"][1]-first["log_noise"][0], np.log(4.))
        with self.assertRaises(ValueError):
            trainer.condition_data(quotes, np.array([.49, 2.]), self.records, mask)

    def test_noise_is_fresh_per_step_probe_invariant_and_paired_before_selection(self):
        h = np.asarray(self.records["half_spread"])
        vega = np.full(12, 20.)
        clean = np.full((5, 12), 3.)
        theta = np.full((5, 6), .01)
        first, c = trainer.noise_batch(clean, theta, h, vega, 7, 10)
        again, again_c = trainer.noise_batch(clean, theta, h, vega, 7, 10)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(c, again_c)
        changed, _ = trainer.noise_batch(clean, theta, h, vega, 8, 10)
        self.assertFalse(np.array_equal(first, changed))
        mask = trainer.layout_mask(self.records, "short")
        full = trainer.condition_data(first, c, self.records, np.ones(12, dtype=bool))
        short = trainer.condition_data(first, c, self.records, mask)
        np.testing.assert_array_equal(full["observations"][:, mask], short["observations"])

    def test_gaussian_simulator_uses_spread_and_discrepancy_in_quadrature(self):
        count = 50000
        theta = np.tile([.02, 2., .04, .5, -.7, .01], (count, 1))
        quotes, c = trainer.noise_batch(np.zeros((count, 2)), theta,
                                      np.array([.2, .4]), np.array([20., 40.]), 0, 15, multiplier=1.)
        expected = np.sqrt(np.array([.2, .4])**2 + (.01*np.array([20., 40.]))**2)
        np.testing.assert_allclose(quotes.std(axis=0), expected, rtol=.015)
        np.testing.assert_array_equal(c, np.ones(count))


if __name__ == "__main__":
    unittest.main()
