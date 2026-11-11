"""The slot 1 slides: the running example's design, pricing, error law and two routes to a range; the recorded results."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad

from sbi4f_tutorial.heston_options import model as heston
from sbi4f_tutorial.shared import surface

TUTORIAL = Path(__file__).resolve().parents[1]
FOLDER = TUTORIAL / "slides" / "slot1"


def load_illustration():
    spec = importlib.util.spec_from_file_location("slot1_illustration", FOLDER / "illustration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


il = load_illustration()
RECORDED = json.loads((FOLDER / "recorded-results.json").read_text())


def runtime_parameters(theta):
    v0, kappa, long_run, vol_of_vol, rho = (float(x) for x in theta)
    return {"kappa": kappa, "theta": long_run, "vol_of_vol": vol_of_vol, "rho": rho, "jump_intensity": 0.0,
            "jump_up_probability": 0.5, "jump_up_rate": 10.0, "jump_down_rate": 10.0}, v0


class DesignTests(unittest.TestCase):
    def test_the_thin_chain_has_seven_strikes_on_five_expiries(self):
        chain = il.thin_chain()
        self.assertEqual(chain.n, 35)
        np.testing.assert_allclose(np.unique(chain.maturity), np.array(il.MONTHS) / 12.0)
        delta = np.log(chain.strike) / (il.REFERENCE_VOLATILITY * np.sqrt(chain.maturity))
        np.testing.assert_allclose(delta.reshape(5, 7), np.tile(il.DELTAS, (5, 1)), atol=1e-12)
        np.testing.assert_array_equal(chain.put, delta <= 1e-12)  # out of the money: puts up to the forward
        self.assertTrue(np.all(chain.half_spread > 0))

    def test_its_half_spreads_are_about_ten_times_those_of_the_spx_quotes(self):
        chain, notebook = il.thin_chain(), il.notebook_case()
        _, low, high = chain.implied()
        thin = 0.5 * (high - low)
        self.assertTrue(0.0028 < thin.min() < 0.0032 and 0.012 < thin.max() < 0.02)  # 0.3 to about 1.5 points
        _, low, high = notebook.implied()
        self.assertTrue(8.0 < np.median(thin) / np.median(0.5 * (high - low)) < 14.0)

    def test_every_reading_keeps_whole_expiries_and_the_day_the_product_pays(self):
        readings, chain = il.readings(), il.thin_chain()
        self.assertEqual([reading.n for reading in readings.values()], [7, 14, 21, 28, 35])
        for name, reading in readings.items():
            months = sorted(set(np.round(12 * reading.maturity).astype(int)))
            self.assertEqual(tuple(months), il.READINGS[name])
            self.assertIn(round(12 * il.END), months)  # the forward start pays on a quoted date
            kept = np.isin(chain.observed, reading.observed)
            np.testing.assert_array_equal(chain.observed[kept], reading.observed)  # the same surface, fewer quotes
        self.assertEqual(il.READINGS["6 months and 1 year"], (round(12 * il.START), round(12 * il.END)))

    def test_the_generating_parameters_are_consistent_with_the_quotes(self):
        full = il.readings()["1 month to 2 years"]
        misses = float(full.sum_of_squares(il.TRUTH)) / il.C_TRUE**2
        self.assertLess(abs(misses - full.n) / np.sqrt(2 * full.n), 3.0)

    def test_the_notebook_case_is_the_notebooks_own_simulated_surface(self):
        stored = json.loads(il.CASE_FILE.read_text())
        observed = np.asarray(stored["observations"], dtype="<f8")  # index points, as the lab hashed them
        self.assertEqual(hashlib.sha256(observed.tobytes()).hexdigest(),
                         stored["source"]["observations"]["observations_sha256"])
        np.testing.assert_array_equal(stored["truth"], il.TRUTH)
        quotes = il.notebook_case()  # raises unless this module's prices give the notebook's clean prices to the gate
        self.assertEqual(quotes.n, 63)
        self.assertEqual(len(np.unique(quotes.maturity)), 7)


class PricingTests(unittest.TestCase):
    def test_prices_are_the_runtimes_fourier_prices(self):
        rng = np.random.default_rng(3)
        thetas = il.LOWER + (il.UPPER - il.LOWER) * rng.uniform(0.05, 0.95, (12, 5))
        strike = np.array([0.7, 0.85, 0.95, 1.0, 1.05, 1.2])
        maturity = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 2.0])
        put = strike <= 1.0
        batch = il.prices(thetas, strike, maturity, put, il.CHECK_RUNG)
        for theta, row in zip(thetas, batch, strict=True):
            parameters, v0 = runtime_parameters(theta)
            runtime = [float(heston.european_price(1.0, k, 1.0, t, parameters, v0, kind="put" if p else "call",
                                                   nodes=4096, upper_limit=2400.0))
                       for k, t, p in zip(strike, maturity, put, strict=True)]
            np.testing.assert_allclose(row, runtime, atol=1e-9)
            np.testing.assert_allclose(il.prices(theta, strike, maturity, put, il.CHECK_RUNG), row, atol=1e-14)

    def test_the_quadrature_passes_the_gate_at_the_generating_parameters(self):
        self.assertLess(il.thin_chain().gate(il.TRUTH, il.CHECK_RUNG), il.GATE)
        self.assertLess(il.notebook_case().gate(il.TRUTH, il.NOTEBOOK_CHECK_RUNG), il.GATE)

    def test_a_failed_pricing_is_a_very_large_miss(self):
        chain = il.readings()["6 months and 1 year"]
        broken = il.TRUTH.copy()
        broken[1] = np.nan
        self.assertTrue(np.isnan(chain.model(broken)).all())
        self.assertGreaterEqual(float(chain.sum_of_squares(broken)), 0.99e8)


class ForwardStartTests(unittest.TestCase):
    def test_a_start_date_of_zero_gives_the_vanilla_call(self):
        vanilla = il.prices(il.TRUTH, [1.0], [0.5], [False], il.CHECK_RUNG)[0]
        self.assertAlmostEqual(il.forward_start_call(il.TRUTH, start=1e-10, end=0.5), vanilla, places=8)

    def test_without_volatility_of_volatility_the_price_is_black_scholes(self):
        theta = il.TRUTH.copy()
        theta[3] = 1e-4
        v0, kappa, long_run = theta[:3]
        start, end = il.START, il.END
        variance = long_run * (end - start) + (v0 - long_run) * (np.exp(-kappa * start) - np.exp(-kappa * end)) / kappa
        black_scholes = float(surface.bs_call(0.0, 1.0, np.sqrt(variance)))  # at the money, total variance `variance`
        self.assertAlmostEqual(il.forward_start_call(theta), black_scholes, places=5)

    def test_products_take_one_parameter_set_or_many(self):
        thetas = np.vstack([il.TRUTH, 0.5 * (il.LOWER + il.UPPER)])
        for product in il.PRODUCTS.values():
            many = product(thetas)
            self.assertEqual(many.shape, (2,))
            self.assertAlmostEqual(float(product(il.TRUTH)), many[0], places=12)


class ErrorScaleTests(unittest.TestCase):
    def test_integrating_c_out_matches_quadrature_up_to_a_constant(self):
        n = 14
        values = (5.0, 14.0, 60.0, 400.0)
        numeric = [np.log(quad(lambda c, r2=r2: c ** (-n) * np.exp(-r2 / (2 * c * c)), il.C_LOWER, il.C_UPPER)[0])
                   for r2 in values]
        closed = il.log_c_integral(np.array(values), n)
        np.testing.assert_allclose(closed - closed[0], np.subtract(numeric, numeric[0]), atol=1e-8)

    def test_c_is_drawn_from_its_conditional(self):
        n, r2 = 14, 20.0
        draws = il.draw_c(np.full(40000, r2), n, np.random.default_rng(0))
        self.assertTrue(np.all((draws >= il.C_LOWER) & (draws <= il.C_UPPER)))
        density = lambda c: c ** (-n) * np.exp(-r2 / (2 * c * c))  # noqa: E731
        mean = quad(lambda c: c * density(c), il.C_LOWER, il.C_UPPER)[0] / quad(density, il.C_LOWER, il.C_UPPER)[0]
        self.assertAlmostEqual(draws.mean(), mean, delta=0.005)


class RangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readings = il.readings()

    def test_least_squares_with_every_quote_brackets_the_true_price(self):
        result = il.fit(self.readings["1 month to 2 years"], starts=3)
        self.assertEqual(result.on_bound, ())
        self.assertEqual(result.dof, 30)
        self.assertTrue(0.7 < result.scale < 1.4)
        point, interval = il.delta_range(result, il.forward_start_call)
        truth = float(il.forward_start_call(il.TRUTH))
        self.assertTrue(interval[0] < truth < interval[1])
        self.assertLess(interval[1] - interval[0], 0.004)  # under 0.4 per 100 of forward
        self.assertTrue(interval[0] < point < interval[1])

    def test_quotes_at_one_expiry_give_no_classical_range(self):
        result = il.fit(self.readings["1 year"], starts=4)
        self.assertNotEqual(result.on_bound, ())
        self.assertIsNone(result.covariance)
        self.assertIsNone(il.delta_range(result, il.forward_start_call)[1])

    def test_one_expiry_cannot_tell_todays_volatility_and_the_forward_start_follows_it(self):
        reading = self.readings["1 year"]
        low, high = (il.fit(reading, fixed={0: volatility**2}, starts=3) for volatility in (0.12, 0.30))
        self.assertEqual(low.theta[0], 0.12**2)
        self.assertEqual(low.dof, reading.n - 4)
        self.assertLess(abs(low.sum_of_squares - high.sum_of_squares), 0.1)  # the same fit to the quotes
        prices = [float(il.forward_start_call(result.theta)) for result in (low, high)]
        self.assertGreater(prices[0] - prices[1], 0.01)  # more than 1 per 100 of forward apart

    def test_the_posterior_sampler_agrees_with_least_squares_when_every_quote_is_read(self):
        # The sampler's tractable target: with all 35 quotes the posterior of the forward start is close to normal,
        # and the delta method gives its range (the regime in which least squares matches the notebook's exact
        # posterior).
        reading = self.readings["1 month to 2 years"]
        posterior = il.posterior(reading, adapt=(600, 600, 600), batch=1000, target=200.0, most=4000, keep=300,
                                 sweeps=3)
        draws = posterior.draws
        self.assertEqual(draws.shape, (300, 6))
        self.assertTrue(np.all((draws[:, :5] >= il.LOWER) & (draws[:, :5] <= il.UPPER)))
        self.assertTrue(np.all((draws[:, 5] >= il.C_LOWER) & (draws[:, 5] <= il.C_UPPER)))
        self.assertGreater(posterior.effective_sample_size, 20.0)
        self.assertLess(posterior.failed, 0.01)
        self.assertTrue(0.0 < posterior.acceptance < 1.0)
        point, (low, high) = il.delta_range(il.fit(reading, starts=3), il.forward_start_call)
        q05, q50, q95 = np.quantile(il.forward_start_call(draws[:, :5]), [0.05, 0.5, 0.95])
        self.assertTrue(0.75 < (q95 - q05) / (high - low) < 1.3)  # the two 90% widths agree, at this small budget
        self.assertLess(abs(q50 - point), 0.2 * (high - low))

    def test_least_squares_reproduces_the_notebooks_exact_posterior_on_its_63_quotes(self):
        result = il.fit(il.notebook_case(), starts=3)
        self.assertEqual(result.on_bound, ())
        classical = il.parameter_intervals(result)
        reference = RECORDED["reference_posteriors"]["synthetic-market-like"]
        exact = np.array([[q[0], q[2]] for q in reference["quantiles_05_50_95"][:5]])
        ratio = (classical[:, 1] - classical[:, 0]) / (exact[:, 1] - exact[:, 0])
        self.assertTrue(np.all(np.abs(ratio - 1.0) < 0.05), ratio)  # the two 90% widths agree within 5%
        shift = np.abs(result.theta - np.array(reference["mean"][:5])) / np.array(reference["sd"][:5])
        self.assertLess(shift.max(), 0.1)  # the fit sits at the posterior mean


class RecordedResultsTests(unittest.TestCase):
    def test_every_source_is_named_with_its_identity(self):
        for source in RECORDED["sources"].values():
            self.assertIn("branch", source)
            self.assertIn("path", source)
            self.assertTrue(any(len(str(value)) == 64 for key, value in source.items() if "sha256" in key))

    def test_the_coverage_table_has_six_parameters_at_four_levels(self):
        population = RECORDED["population"]
        self.assertEqual(RECORDED["parameters"], ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"])
        self.assertEqual(population["surfaces"], 512)
        for level in population["levels"]:
            coverage = population["coverage"][str(level)]
            self.assertEqual(len(coverage), 6)
            for value, (low, high) in zip(coverage, population["coverage_wilson95"][str(level)], strict=True):
                self.assertTrue(low < value < high)

    def test_the_references_are_the_three_cases_of_the_notebook(self):
        self.assertEqual(set(RECORDED["references"]), {"synthetic-broad", "synthetic-market-like", "market"})
        self.assertEqual(set(RECORDED["conditional"]), set(RECORDED["references"]))
        for run in RECORDED["references"].values():
            self.assertTrue(140000 < run["likelihood_evaluations"] < 150000)
        np.testing.assert_allclose(RECORDED["reference_posteriors"]["synthetic-market-like"]["generating_parameters"][:5],
                                   il.TRUTH, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
