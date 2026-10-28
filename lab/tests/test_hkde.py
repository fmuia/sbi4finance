"""HKDE limits, explicit jump law, continuation and Fourier identities."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance import gbm, heston, hkde


def model(**updates):
    result = dict(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.6,
                  jump_intensity=0.8, jump_up_probability=0.35,
                  jump_up_rate=12.0, jump_down_rate=8.0)
    result.update(updates)
    return result


class HKDETests(unittest.TestCase):
    def test_moment_support_including_absent_upward_tail(self):
        with self.assertRaisesRegex(ValueError, "jump_up_rate>1"):
            hkde.validate_parameters(model(jump_up_rate=1.0))
        for changes in (dict(jump_intensity=0, jump_up_rate=1),
                        dict(jump_up_probability=0, jump_up_rate=0.5)):
            values = model(**changes)
            self.assertTrue(np.isfinite(hkde.jump_compensation(values)))
            assert_allclose(hkde.characteristic_function([0, -1j], 1, values, .04), [1, 1])
        for changes in (dict(jump_down_rate=0), dict(rho=1.01), dict(kappa=0),
                        dict(jump_up_probability=-.1), dict(theta=-.1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hkde.validate_parameters(model(**changes))

    def test_characteristic_identities_and_deterministic_variance_limit(self):
        u = np.array([0, .5, 2, 8])
        values = model(jump_intensity=0, vol_of_vol=0)
        expected_variance = .04*.7 + (.06-.04)*(1-np.exp(-2*.7))/2
        expected = np.exp(-.5*(u*u+1j*u)*expected_variance)
        assert_allclose(hkde.characteristic_function(u, .7, values, .06), expected, atol=1e-14)
        values = model()
        phi = hkde.characteristic_function(u, .7, values, .04)
        assert_allclose(hkde.characteristic_function(-u, .7, values, .04), phi.conjugate(), atol=1e-14)
        assert_allclose(hkde.characteristic_function([0, -1j], .7, values, .04), [1, 1], atol=0)
        assert_allclose(hkde.characteristic_function(u, 0, values, .04), np.ones(len(u)), atol=0)
        self.assertTrue(np.all(abs(phi) <= 1+1e-14))

    def test_pricing_parity_bounds_monotonicity_and_refinement(self):
        strikes = np.array([75, 90, 100, 115, 130])
        values = model()
        call = hkde.european_price(100, strikes, .97, .75, values, .04)
        put = hkde.european_price(100, strikes, .97, .75, values, .04, kind="put")
        refined = hkde.european_price(100, strikes, .97, .75, values, .04,
                                      nodes=1024, upper_limit=400)
        assert_allclose(call-put, .97*(100-strikes), atol=1e-13)
        assert_allclose(call, refined, atol=2e-7)
        self.assertTrue(np.all(np.diff(call) < 0))
        self.assertTrue(np.all(call >= .97*np.maximum(100-strikes, 0)))
        self.assertTrue(np.all(call < 97))
        exact = gbm.european_price(100, strikes, .97, .04*.75)
        assert_allclose(hkde.european_price(100, strikes, .97, .75,
                                          model(vol_of_vol=0, jump_intensity=0), .04), exact)
        assert_allclose(hkde.european_price(100, strikes, 1, 0, values, .04),
                        np.maximum(100-strikes, 0))

    def test_explicit_noise_identity_zero_jump_limit_and_restart(self):
        values = model(drift=.03)
        dt = [.1, .2, .3]
        innovations = hkde.draw_innovations(dt, 2, seed=71, example_ids=[1, 4], parameters=values)
        reordered = hkde.draw_innovations(dt, 2, seed=71, example_ids=[4, 1], parameters=values)
        for key in innovations:
            assert_array_equal(innovations[key], reordered[key][::-1])
        full = hkde.simulate(values, dt, [-.01, .04], innovations, backend="numpy")
        compiled = hkde.simulate(values, dt, [-.01, .04], innovations, backend="numba")
        for key in full:
            assert_allclose(compiled[key], full[key], atol=1e-13)
        first = hkde.simulate(values, dt[:1], [-.01, .04],
                              {k: v[:, :1] for k, v in innovations.items()})
        rest = hkde.simulate(values, dt[1:], first["variance_state"][:, -1, 0],
                             {k: v[:, 1:] for k, v in innovations.items()})
        assert_allclose(np.concatenate([first["log_returns"], rest["log_returns"]], axis=1),
                        full["log_returns"], atol=1e-13)
        assert_allclose(rest["variance_state"][:, -1], full["variance_state"][:, -1], atol=1e-13)
        zero = model(jump_intensity=0, drift=.03)
        noises = hkde.draw_innovations(dt, 2, seed=91, example_ids=[1], parameters=zero)
        actual = hkde.simulate(zero, dt, .04, noises, backend="numpy")
        plain = heston.simulate({key: zero[key] for key in ("drift", "kappa", "theta", "vol_of_vol", "rho")},
                                 dt, .04, {"brownian": noises["brownian"],
                                           "jump_counts": np.zeros((1,3), dtype=np.int64),
                                           "jump_normals": np.zeros((1,3))}, backend="numpy")
        for key in plain:
            assert_array_equal(actual[key], plain[key])

    def test_exact_interval_jump_moments_and_compensated_expectation(self):
        values = model(theta=0, vol_of_vol=0, jump_intensity=4, drift=.03)
        n, duration = 12_000, .5
        noises = hkde.draw_innovations([duration], 1, seed=199, example_ids=range(n), parameters=values)
        p, up, down, lam = .35, 12., 8., 4.
        jump = noises["jump_up_sums"][:,0]-noises["jump_down_sums"][:,0]
        expected_mean = lam*duration*(p/up-(1-p)/down)
        expected_variance = 2*lam*duration*(p/up**2+(1-p)/down**2)
        fourth_cumulant = 24*lam*duration*(p/up**4+(1-p)/down**4)
        self.assertLess(abs(jump.mean()-expected_mean), 6*np.sqrt(expected_variance/n))
        self.assertLess(abs(jump.var()-expected_variance),
                        6*np.sqrt((fourth_cumulant+2*expected_variance**2)/n))
        self.assertGreater(np.max(noises["jump_up_counts"]+noises["jump_down_counts"]), 1)
        gross = np.exp(hkde.simulate(values, [duration], 0, noises)["log_returns"][:,0,0])
        self.assertLess(abs(gross.mean()-np.exp(.03*duration)), 6*gross.std(ddof=1)/np.sqrt(n))

    def test_invalid_pricing_and_noise_fail(self):
        for kwargs in (dict(nodes=2), dict(upper_limit=0), dict(kind="american")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                hkde.european_price(1, 1, 1, .5, model(), .04, **kwargs)
        with self.assertRaises(ValueError):
            hkde.characteristic_function(1-2j, .5, model(), .04)
        with self.assertRaises(ValueError):
            hkde.characteristic_function(1, .5, model(), -.01)
        with self.assertRaises(ValueError):
            hkde.draw_innovations([.1], 1, seed=1, example_ids=[0, 0], parameters=model())


if __name__ == "__main__":
    unittest.main()
