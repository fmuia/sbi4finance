"""Financial invariants and stability of the alternate Fourier representation."""
import unittest

import numpy as np

from sbi4f_tutorial.heston_options import control_variate as cv, model
from sbi4f_tutorial.shared import gbm


def parameters(**updates):
    return dict(kappa=2., theta=.04, vol_of_vol=.4, rho=-.7,
        jump_intensity=0., jump_up_probability=.5, jump_up_rate=10., jump_down_rate=10., **updates)


class ControlVariatePrices(unittest.TestCase):
    def test_deterministic_variance_and_expiry_are_exact_black_limits(self):
        pars = parameters()
        pars["vol_of_vol"] = 0.
        t = np.array([0., .02, .25, 1.])
        f, k, d = np.full(4, 100.), np.array([90., 100., 110., 120.]), np.exp(-.03*t)
        mask = np.array([True, False, True, False])
        integrated = pars["theta"]*t + (.06-pars["theta"])*(-np.expm1(-pars["kappa"]*t))/pars["kappa"]
        expected = np.where(mask, gbm.european_price(f,k,d,integrated,option="put"),
                            gbm.european_price(f,k,d,integrated,option="call"))
        np.testing.assert_array_equal(cv.european_price(f,k,d,t,pars,.06,put=mask), expected)

    def test_regular_domain_matches_original_and_put_call_parity(self):
        pars = parameters()
        f,k,d,t = np.array([100.,103.,105.]), np.array([90.,110.,120.]), np.array([.99,.98,.96]), np.array([.1,.5,1.])
        call = cv.european_price(f,k,d,t,pars,.04,nodes=1024,upper_limit=300.)
        put = cv.european_price(f,k,d,t,pars,.04,put=True,nodes=1024,upper_limit=300.)
        old = model.european_price(f,k,d,t,pars,.04,nodes=1024,upper_limit=300.)
        np.testing.assert_allclose(call,old,rtol=0,atol=1e-7)
        np.testing.assert_allclose(call-put,d*(f-k),rtol=0,atol=1e-12)

    def test_stress_tail_refines_without_clipping(self):
        pars=parameters();pars.update(kappa=10.,theta=.01,vol_of_vol=3.,rho=-.95)
        f,k,d,t=7830.898473901978,9800.,.9577926609156969,.8903850520040588
        with self.assertRaisesRegex(ValueError,"violates bounds"):
            cv.european_price(f,k,d,t,pars,.01,nodes=1024,upper_limit=600.)
        high=cv.european_price(f,k,d,t,pars,.01,nodes=16384,upper_limit=9600.)
        self.assertGreater(float(high),0.)
        self.assertLess(float(high),1e-5)

    def test_invalid_geometry_and_quadrature_are_rejected(self):
        for nodes in [True,0,15]:
            with self.assertRaises(ValueError):cv.european_price(100.,100.,1.,.5,parameters(),.04,nodes=nodes)
        for k in [-1.,float("nan"),100+1j]:
            with self.assertRaises(ValueError):cv.european_price(100.,k,1.,.5,parameters(),.04)


if __name__=="__main__":unittest.main()
