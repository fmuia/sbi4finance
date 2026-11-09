"""Only successful Fourier rungs may be reused in the adjacent comparison."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4f_tutorial.shared import pricing, tasks


def task():
    return {"context": {"contracts": [{"maturity": .25, "strike": 1., "option_type": "put"}],
                        "quote_noise_std": [1.]},
            "numerics": {"max_refinement_over_quote_sigma": .1,
                         "quadrature_ladder": [[16, 1.], [32, 2.], [64, 3.], [128, 4.]]}}


class RefinementReuseTests(unittest.TestCase):
    def test_successful_refined_rung_is_next_base_without_repricing(self):
        values = {16: np.array([0.]), 32: np.array([.2]), 64: np.array([.21])}
        with patch.object(pricing, "heston_surface", side_effect=lambda *a, nodes, **k: values[nodes]) as price:
            prices, discrepancy = tasks.heston_prices(task(), {}, .04)
        self.assertEqual([call.kwargs["nodes"] for call in price.call_args_list], [16, 32, 64])
        assert_array_equal(prices, values[64])
        assert_array_equal(discrepancy, values[64] - values[32])

    def test_failed_refined_rung_is_retried_from_scratch(self):
        with patch.object(pricing, "heston_surface", side_effect=[np.array([0.]), ValueError("bounds"),
                                                                  np.array([.2]), np.array([.21])]) as price:
            prices, discrepancy = tasks.heston_prices(task(), {}, .04)
        self.assertEqual([call.kwargs["nodes"] for call in price.call_args_list], [16, 32, 32, 64])
        assert_array_equal(prices, [.21])
        assert_array_equal(discrepancy, np.array([.21]) - np.array([.2]))

    def test_failed_base_is_not_retained_and_exhaustion_keeps_failure_message(self):
        with patch.object(pricing, "heston_surface", side_effect=[ValueError("bounds"),
                                                                  np.array([.2]), np.array([.21])]) as price:
            tasks.heston_prices(task(), {}, .04)
        self.assertEqual([call.kwargs["nodes"] for call in price.call_args_list], [16, 32, 64])
        with patch.object(pricing, "heston_surface", side_effect=ValueError("bounds")) as price:
            with self.assertRaisesRegex(ValueError, "bounds on every quadrature rung"):
                tasks.heston_prices(task(), {}, .04)
        self.assertEqual([call.kwargs["nodes"] for call in price.call_args_list], [16, 32, 64])
        with patch.object(pricing, "heston_surface", side_effect=lambda *a, nodes, **k: np.array([nodes])):
            with self.assertRaisesRegex(ValueError, "Fourier refinement budget failed on every quadrature rung"):
                tasks.heston_prices(task(), {}, .04)

    def test_a_subset_keeps_the_existing_first_successful_pair_rule(self):
        contracts = [{"maturity": .25, "strike": 1., "option_type": "put"}] * 2
        with patch.object(pricing, "heston_surface", side_effect=[np.array([0., 0.]), np.array([9., 9.])]) as price:
            prices, discrepancy = tasks.heston_prices(task(), {}, .04, contracts)
        self.assertEqual(price.call_count, 2)
        assert_array_equal(prices, [9., 9.])
        assert_array_equal(discrepancy, [9., 9.])


if __name__ == "__main__":
    unittest.main()
