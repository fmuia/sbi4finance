"""Exact regression against the pricer before maturity reuse.

Run the focused timing comparison with::

    python tests/test_hkde_maturity_reuse.py --benchmark

The benchmark compares the actual 63-contract calibration task, including its
base/refined quadrature and acceptance logic. It does not change the controls.
"""

import json
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4finance import gbm, hkde
from sbi4finance.calibration import tasks


ROOT = Path(__file__).resolve().parents[1]


def original_price(forward, strike, discount, maturity, parameters, initial_variance,
                   *, kind="call", upper_limit=200.0, nodes=512):
    """Frozen pre-optimization reference, with one CF row per contract."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be call or put")
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)) or nodes < 16:
        raise ValueError("nodes must be an integer >=16")
    cutoff = gbm._scalar(upper_limit, "upper_limit", positive=True)
    fwd, k, disc, time = np.broadcast_arrays(*[np.asarray(value, dtype=np.float64)
                                              for value in (forward, strike, discount, maturity)])
    if any(not value.size or not np.isfinite(value).all() or np.any(value <= 0)
           for value in (fwd, k, disc)) or not np.isfinite(time).all() or np.any(time < 0):
        raise ValueError("require finite F,K,D>0 and T>=0")
    hkde.characteristic_function(0.0, 0.0, parameters, initial_variance)
    values = hkde.validate_parameters(parameters)
    if float(values["vol_of_vol"]) == 0 and float(values["jump_intensity"]) == 0:
        kappa, theta = float(values["kappa"]), float(values["theta"])
        integrated = theta*time + (initial_variance-theta)*(-np.expm1(-kappa*time))/kappa
        return gbm.european_price(fwd, k, disc, integrated, option=kind)
    u, weights = hkde._quadrature(int(nodes), cutoff)
    phi = hkde.characteristic_function(u-0.5j, time[..., None], parameters, initial_variance)
    integrand = np.real(np.exp(1j*u*np.log(fwd/k)[..., None])*phi)/(u*u+0.25)
    call = disc*(fwd - np.sqrt(fwd*k)*np.sum(integrand*weights, axis=-1)/np.pi)
    call = np.where(time == 0, disc*np.maximum(fwd-k, 0), call)
    price = call if kind == "call" else call-disc*(fwd-k)
    lower = disc*np.maximum(fwd-k if kind == "call" else k-fwd, 0)
    upper = disc*(fwd if kind == "call" else k)
    tolerance = 1e-9*disc*np.maximum(fwd, k)
    if (not np.isfinite(price).all() or np.any(price < lower-tolerance)
            or np.any(price > upper+tolerance)):
        raise ValueError("Fourier price violates bounds; refine controls or reject this domain")
    return price


def model(**updates):
    parameters = dict(kappa=2.0, theta=.04, vol_of_vol=.3, rho=-.6,
                      jump_intensity=.8, jump_up_probability=.35,
                      jump_up_rate=12., jump_down_rate=8.)
    return {**parameters, **updates}


def calibration_task():
    return json.loads((ROOT / "configs/tasks/calibration_heston_bank.json").read_text())["task"]


class MaturityReuseTests(unittest.TestCase):
    def test_entire_surface_is_bitwise_equal_on_every_cold_and_warm_rung(self):
        task = calibration_task()
        strike = np.array([c["strike"] for c in task["context"]["contracts"]])
        maturity = np.array([c["maturity"] for c in task["context"]["contracts"]])
        models = [model(jump_intensity=0), model(), model(vol_of_vol=0),
                  model(vol_of_vol=0, jump_intensity=0)]
        for nodes, upper_limit in task["numerics"]["quadrature_ladder"]:
            hkde._quadrature.cache_clear()
            for warm in (False, True):
                for parameters in models:
                    for kind in ("call", "put"):
                        with self.subTest(nodes=nodes, warm=warm, parameters=parameters, kind=kind):
                            kwargs = dict(kind=kind, nodes=nodes, upper_limit=upper_limit)
                            if not warm:
                                hkde._quadrature.cache_clear()
                            expected = original_price(1., strike, 1., maturity, parameters, .04, **kwargs)
                            if not warm:
                                hkde._quadrature.cache_clear()
                            actual = hkde.european_price(1., strike, 1., maturity, parameters, .04, **kwargs)
                            assert_array_equal(actual, expected, strict=True)

    def test_unsorted_2d_broadcast_expiry_and_scalars_preserve_shape_and_order(self):
        for parameters in (model(), model(jump_intensity=0), model(vol_of_vol=0),
                           model(vol_of_vol=0, jump_intensity=0)):
            for kind in ("call", "put"):
                for maturity in (np.array([[.75, 0., .25], [.25, .75, .75]]),
                                 np.array([[.75, 0., .25], [.5, .6, .8]]), .75, 0.):
                    with self.subTest(parameters=parameters, kind=kind, maturity=maturity):
                        inputs = (np.array([[100.], [101.]]), np.array([90., 100., 110.]),
                                  np.array([[.98], [.97]]), maturity, parameters, .04)
                        assert_array_equal(hkde.european_price(*inputs, kind=kind),
                                           original_price(*inputs, kind=kind), strict=True)
                for maturity in (0., .75):
                    assert_array_equal(hkde.european_price(100., 95., .98, maturity, parameters, .04, kind=kind),
                                       original_price(100., 95., .98, maturity, parameters, .04, kind=kind),
                                       strict=True)

    def test_characteristic_rows_are_reused_without_rounding(self):
        task = calibration_task()
        strike = np.array([c["strike"] for c in task["context"]["contracts"]])
        maturity = np.array([c["maturity"] for c in task["context"]["contracts"]])
        original_cf = hkde.characteristic_function
        with patch.object(hkde, "characteristic_function", wraps=original_cf) as observed:
            hkde.european_price(1., strike, 1., maturity, model(), .04)
        assert_array_equal(observed.call_args_list[1].args[1], np.unique(maturity)[:, None])
        self.assertEqual(observed.call_count, 2)  # original validation plus integration
        close = np.array([.25, np.nextafter(.25, 1.), .25])
        with patch.object(hkde, "characteristic_function", wraps=original_cf) as observed:
            hkde.european_price(1., [1., 1., 1.], 1., close, model(), .04)
        assert_array_equal(observed.call_args_list[1].args[1], np.unique(close)[:, None])
        self.assertEqual(observed.call_args_list[1].args[1].shape, (2, 1))

    def test_original_failures_are_retained(self):
        cases = [dict(maturity=-.1), dict(initial_variance=-.1, maturity=0.),
                 dict(nodes=8), dict(upper_limit=0.), dict(kind="american"),
                 dict(parameters=model(rho=1.01)),
                 dict(nodes=16, upper_limit=3000., maturity=[.02, .02, .02], strike=[.8, 1., 1.2])]
        for updates in cases:
            kwargs = dict(forward=1., strike=1., discount=1., maturity=.25,
                          parameters=model(), initial_variance=.04)
            kwargs.update(updates)
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError) as expected:
                    original_price(**kwargs)
                with self.assertRaises(ValueError) as actual:
                    hkde.european_price(**kwargs)
                self.assertEqual(str(actual.exception), str(expected.exception))

    def test_refined_task_prices_and_diagnostics_are_bitwise_equal(self):
        task = calibration_task()
        candidates = [[(lo + hi) / 2 for lo, hi in task["prior_bounds"]],
                      [.015, .025, 1.3, -.9], [.15, .12, .2, -.3]]
        for parameters in candidates:
            with self.subTest(parameters=parameters):
                actual = tasks.price(task, parameters, [], 0)
                with patch.object(hkde, "european_price", original_price):
                    expected = tasks.price(task, parameters, [], 0)
                for value, reference in zip(actual, expected, strict=True):
                    assert_array_equal(value, reference, strict=True)


def benchmark(repeats=80):
    """Report medians of alternating timings; no timing-dependent test gates."""
    task = calibration_task()
    parameters = [(lo + hi) / 2 for lo, hi in task["prior_bounds"]]
    optimized = hkde.european_price
    outputs = {}
    timings = {}
    for name, implementation in (("original", original_price), ("maturity_reuse", optimized)):
        hkde._quadrature.cache_clear()
        with patch.object(hkde, "european_price", implementation):
            start = time.perf_counter()
            outputs[name] = tasks.price(task, parameters, [], 0)
            timings[name] = {"cold_wall_seconds": time.perf_counter() - start, "warm_wall_seconds": []}
    for i in range(repeats):
        order = (("original", original_price), ("maturity_reuse", optimized))
        for name, implementation in order[::1 if i % 2 == 0 else -1]:
            with patch.object(hkde, "european_price", implementation):
                start = time.perf_counter()
                tasks.price(task, parameters, [], 0)
                timings[name]["warm_wall_seconds"].append(time.perf_counter() - start)
    for values in timings.values():
        values["warm_wall_seconds_median"] = float(np.median(values.pop("warm_wall_seconds")))
    equal = all(np.array_equal(a, b) for a, b in zip(outputs["original"], outputs["maturity_reuse"], strict=True))
    report = {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
              "scope": "63-contract Heston task, existing base/refined quadrature and acceptance logic",
              "parameters": parameters, "quadrature_ladder": task["numerics"]["quadrature_ladder"],
              "warm_repeats": repeats, "timings": timings, "bitwise_equal_all_return_arrays": equal,
              "warm_speedup": timings["original"]["warm_wall_seconds_median"] /
                              timings["maturity_reuse"]["warm_wall_seconds_median"]}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--benchmark"]:
        benchmark()
    else:
        unittest.main()
