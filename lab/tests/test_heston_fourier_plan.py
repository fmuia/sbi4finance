"""Content-addressed grid plans retain the general Fourier pricing contract."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4f_tutorial.heston_options import model
from test_hkde_maturity_reuse import original_price, model as parameters


class FourierPlanTests(unittest.TestCase):
    def setUp(self):
        model._fourier_plan.cache_clear()

    def test_equal_content_reuses_plan_but_mutation_order_and_controls_do_not(self):
        strike = np.array([.9, 1., 1.1])
        maturity = np.array([.25, .5, .25])

        def price(k=strike, time=maturity, nodes=256, cutoff=200.):
            return model.european_price(1., k, 1., time, parameters(), .04,
                                        nodes=nodes, upper_limit=cutoff)

        price()
        price(strike.copy(), maturity.copy())
        self.assertEqual(model._fourier_plan.cache_info().hits, 1)
        for k, time, nodes, cutoff in (
                (strike[::-1], maturity[::-1], 256, 200.),
                (strike, np.nextafter(maturity, 1.), 256, 200.),
                (strike, maturity, 512, 200.),
                (strike, maturity, 256, 201.)):
            price(k, time, nodes, cutoff)
        self.assertEqual(model._fourier_plan.cache_info().misses, 5)
        strike[0] = .95
        assert_array_equal(price(), original_price(1., strike, 1., maturity, parameters(), .04,
                                                   nodes=256, upper_limit=200.))
        self.assertEqual(model._fourier_plan.cache_info().misses, 6)

    def test_cached_arrays_cannot_be_made_writable(self):
        grid = (np.ones(3), np.array([.9, 1., 1.1]), np.array([.5, .25, .5]))
        plan = model._grid_plan(*grid, 256, 200.)
        for array in (*plan, *model._quadrature(256, 200.)):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        assert_array_equal(plan[1], [.25, .5])
        assert_array_equal(plan[2], [1, 0, 1])

    def test_oversized_grids_use_uncached_formula_and_cache_count_is_bounded(self):
        values = (1., [.9, 1., 1.1], 1., [.25, .5, .25], parameters(), .04)
        with patch.object(model, "_MAX_PLAN_BYTES", 1):
            actual = model.european_price(*values)
        assert_array_equal(actual, original_price(*values))
        self.assertEqual(model._fourier_plan.cache_info().currsize, 0)
        for maturity in (.2, .25, .3, .4, .5):
            model.european_price(1., 1., 1., maturity, parameters(), .04)
        self.assertEqual(model._fourier_plan.cache_info().currsize, 4)

    def test_broadcast_discount_and_forward_and_absent_jump_limits(self):
        for pars in (parameters(), parameters(jump_intensity=0.),
                     parameters(vol_of_vol=0.),
                     parameters(jump_intensity=0., vol_of_vol=0.)):
            for kind in ("call", "put"):
                inputs = (np.array([[100.], [101.]]), np.array([90., 100., 110.]),
                          np.array([[.98], [.97]]), np.array([[.5, 0., .5], [.25, .5, .25]]),
                          pars, .04)
                assert_array_equal(model.european_price(*inputs, kind=kind),
                                   original_price(*inputs, kind=kind))


if __name__ == "__main__":
    unittest.main()
