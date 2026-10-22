import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sbi4finance.bergomi import option_panel
from sbi4finance.calibration import bank, correction, inference, pricing, surface, tasks


def small_rough_task(budget=10.0, ladder=(256, 512)):
    """Three maturities and three strikes, one block: fast enough for unit tests."""
    return tasks.rough_task(days=(5, 10, 21), deltas=(-1.0, 0.0, 1.0),
                            blocks=[{"horizon": 21 / 252, "n_steps": 42}], paths_ladder=ladder, budget=budget)


def config(task, splits=(3, 2, 2)):
    return {"task": task, "root_seed": 7, "splits": dict(zip(("train", "validation", "test"), splits))}


class SurfaceTests(unittest.TestCase):
    def test_black_scholes_parity_vega_and_implied_vol(self):
        k = np.array([-0.3, -0.05, 0.0, 0.1, 0.4])
        T, sigma = 0.5, 0.23
        call, put = surface.bs_call(k, T, sigma), surface.bs_put(k, T, sigma)
        assert_allclose(call - put, 1 - np.exp(k), atol=1e-15)
        h = 1e-6
        assert_allclose(surface.bs_vega(k, T, sigma), (surface.bs_call(k, T, sigma + h) - surface.bs_call(k, T, sigma - h)) / (2 * h), rtol=1e-6)
        kind = np.where(k <= 0, "put", "call")
        prices = surface.bs_price(k, T, sigma, kind)
        assert_allclose(surface.implied_vol(prices, k, T, kind), sigma, atol=1e-10)
        self.assertTrue(np.isnan(surface.implied_vol(0.0, 0.1, T, "call")))

    def test_design_grid_noise_and_smile_statistics(self):
        contracts = surface.design_grid()
        self.assertEqual(len(contracts), 63)
        k, T, kind = surface.contract_arrays(contracts)
        delta = surface.standardized_moneyness(k, T)
        assert_allclose(delta.reshape(7, 9), np.tile(surface.DESIGN_DELTAS, (7, 1)), atol=1e-12)
        self.assertTrue(np.all((kind == "put") == (delta <= 0)))
        noise = np.array(surface.quote_noise(contracts))
        self.assertTrue(np.all(noise > 0))
        atm = noise.reshape(7, 9)[:, 4] / surface.bs_vega(0.0, np.array(surface.DESIGN_DAYS) / 252, surface.SIGMA_REF)
        assert_allclose(atm, surface.NOISE_ATM, rtol=1e-12)
        flat = surface.implied_vol(surface.bs_price(k, T, 0.25, kind), k, T, kind)
        _, level, skew, curvature = surface.smile_statistics(flat, k, T)
        assert_allclose(level, 0.25, atol=1e-9)
        assert_allclose(skew, 0.0, atol=1e-7)

    def test_replication_recovers_black_scholes_total_variance(self):
        sigma, T = 0.3, 0.75
        k = np.linspace(-3.0, 3.0, 4001)
        otm = np.where(k < 0, surface.bs_put(k, T, sigma), surface.bs_call(k, T, sigma))
        self.assertAlmostEqual(surface.total_variance_by_replication(k, otm), sigma**2 * T, places=6)
        assert_allclose(surface.forward_variances([0.25, 1.0], [0.01, 0.07]), [0.04, 0.08])

    def test_parity_forward(self):
        strikes = np.array([90.0, 95, 100, 105, 110])
        F, D = 101.3, 0.97
        calls = np.array([15.0, 11, 8, 5.5, 3.5])
        puts = calls - D * (F - strikes)
        forward, discount = surface.parity_forward(strikes, calls, puts)
        self.assertAlmostEqual(forward, F, places=10)
        self.assertAlmostEqual(discount, D, places=12)


class PricingTests(unittest.TestCase):
    def test_step_curve_is_right_continuous(self):
        curve = pricing.step_curve([0, 0.1, 0.3], [0.04, 0.09])
        from sbi4finance.bergomi import curve_values
        assert_allclose(curve_values(curve, [0, 0.05, 0.1 - 1e-6, 0.1, 0.2, 0.3]), [0.04, 0.04, 0.04, 0.09, 0.09, 0.09], atol=1e-6)

    def test_rough_pricer_has_the_black_scholes_limit(self):
        # With eta -> 0 the variance is the deterministic curve: Black-Scholes with total variance int xi0.
        contracts = surface.design_grid(days=(5, 21), deltas=(-1.5, 0.0, 1.5))
        curve = pricing.step_curve([0, 5 / 252, 21 / 252], [0.04, 0.09])
        prices, errors, covariance, used = pricing.rough_surface(
            {"hurst": 0.2, "eta": 1e-8, "rho": -0.7}, curve, contracts,
            [{"horizon": 21 / 252, "n_steps": 84}], [8192], np.full(6, 1e-3), 1e6, seed=3)
        k, T, kind = surface.contract_arrays(contracts)
        total = np.where(T <= 5 / 252 + 1e-12, 0.04 * T, 0.04 * 5 / 252 + 0.09 * (T - 5 / 252))
        exact = surface.bs_price(k, T, np.sqrt(total / T), kind)
        self.assertLess(np.max(np.abs(prices - exact) / errors), 4.0)
        self.assertLess(np.max(errors / exact), 0.01)
        self.assertEqual(used.tolist(), [8192.0])
        assert_allclose(errors**2, np.diag(covariance))

    def test_control_variates_agree_with_the_plain_estimator(self):
        contracts = surface.design_grid(days=(10, 21), deltas=(-1.0, 0.0, 1.0))
        curve = {"times": [0.0, 21 / 252], "values": [0.06, 0.06]}
        parameters = {"hurst": 0.15, "eta": 1.8, "rho": -0.7}
        cv, cv_se, _, _ = pricing.rough_surface(parameters, curve, contracts, [{"horizon": 21 / 252, "n_steps": 42}],
                                                 [8192], np.full(6, 1.0), 1e6, seed=11)
        plain = option_panel("rough", parameters, curve, contracts, n_paths=16384, n_steps=42, seed=5)
        z = (cv - plain["prices"]) / np.sqrt(cv_se**2 + plain["standard_errors"] ** 2)
        self.assertLess(np.max(np.abs(z)), 4.0)
        self.assertTrue(np.all(cv_se < plain["standard_errors"]))

    def test_heston_surface_has_the_black_scholes_limit(self):
        contracts = surface.design_grid(days=(10, 63), deltas=(-1.0, 0.0, 1.0))
        task = tasks.heston_task()
        parameters, v0 = tasks._heston_split(tasks.model_parameters(task, [0.05, 0.05, 0.1, -0.5]))
        parameters["vol_of_vol"] = 0.0
        prices = pricing.heston_surface(parameters, v0, contracts, nodes=512, upper_limit=400.0)
        k, T, kind = surface.contract_arrays(contracts)
        assert_allclose(prices, surface.bs_price(k, T, np.sqrt(0.05), kind), atol=1e-9)


class TaskAndBankTests(unittest.TestCase):
    def test_builders_validate_and_reject_misdeclared_tasks(self):
        rough, heston = tasks.rough_task(), tasks.heston_task()
        self.assertEqual(len(rough["context"]["quote_noise_std"]), 63)
        self.assertEqual(tasks.context_size(rough), 7)
        self.assertEqual(tasks.context_size(heston), 0)
        bad = copy.deepcopy(rough)
        bad["numerics"]["blocks"][0]["n_steps"] = 500
        with self.assertRaisesRegex(ValueError, "grid"):
            tasks.validate_task(bad)
        bad = copy.deepcopy(heston)
        bad["fixed_parameters"]["jump_intensity"] = 1.0
        with self.assertRaisesRegex(ValueError, "jump_intensity"):
            tasks.validate_task(bad)
        bad = copy.deepcopy(rough)
        bad["context"]["quote_noise_std"] = bad["context"]["quote_noise_std"][:-1]
        with self.assertRaisesRegex(ValueError, "per contract"):
            tasks.validate_task(bad)

    def test_examples_are_reproducible_by_id_and_carry_their_context(self):
        c = config(small_rough_task())
        a = bank.generate_examples(c, "train", [0, 1, 2])
        b = bank.generate_examples(c, "train", [2, 0])
        for key in a:
            np.testing.assert_array_equal(a[key][[2, 0]], b[key])
        self.assertEqual(a["context_values"].shape, (3, 3))
        self.assertEqual(a["pricing_paths"].shape, (3, 1))
        level, slope = a["context_parameters"][0]
        knots = tasks.knots(c["task"])
        mid = 0.5 * (knots[1:] + knots[:-1])
        ratio = a["context_values"][0] / (level**2 * (1 + slope * mid))
        self.assertTrue(np.all(np.abs(np.log(ratio)) <= 0.2 + 1e-12))

    def test_bank_roundtrip_and_shards_for_both_models(self):
        for task in (tasks.heston_task(days=(10, 63), deltas=(-1.0, 0.0, 1.0)), small_rough_task()):
            c = config(task)
            with tempfile.TemporaryDirectory() as tmp:
                whole = Path(tmp) / "bank"
                manifest = bank.write_bank(whole, c)
                self.assertIn("calibration/tasks.py", manifest["source_sha256"])
                self.assertIn("context_values", manifest["roles"])
                _, arrays = bank.load_bank(whole)
                shards = []
                for job in bank.job_map(c, 2):
                    shard = Path(tmp) / f"{job['split']}-{job['start']}"
                    bank.write_shard(shard, c, job["split"], job["start"], job["stop"])
                    shards.append(shard)
                bank.assemble_bank(Path(tmp) / "assembled", c, reversed(shards))
                _, assembled = bank.load_bank(Path(tmp) / "assembled")
                with self.assertRaisesRegex(ValueError, "missing shard IDs"):
                    bank.assemble_bank(Path(tmp) / "partial", c, shards[1:])
                for split in arrays:
                    for key in arrays[split]:
                        np.testing.assert_array_equal(arrays[split][key], assembled[split][key])

    def test_heston_quadrature_ladder_rescues_low_variance_short_maturity_wings(self):
        # This surface's one-week deep call violates the price bounds at 512 nodes up to 400.
        task = tasks.heston_task()
        parameters, v0 = tasks._heston_split(tasks.model_parameters(task, [0.0149, 0.1281, 0.7626, -0.7694]))
        with self.assertRaisesRegex(ValueError, "bounds"):
            pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=512, upper_limit=400.0)
        prices, difference = tasks.heston_prices(task, parameters, v0)
        self.assertTrue(np.all(np.abs(difference) <= 0.1 * tasks.sigma(task)))
        k, T, kind = surface.contract_arrays(task["context"]["contracts"])
        self.assertTrue(np.all(np.isfinite(surface.implied_vol(prices, k, T, kind))))

    def test_precision_budget_fails_closed(self):
        c = config(small_rough_task(budget=1e-6, ladder=(256,)))
        with self.assertRaisesRegex(ValueError, "train:0.*budget failed"):
            bank.generate_examples(c, "train", [0])

    def test_archived_fixed_panel_modules_are_not_modified(self):
        # Archived evidence pins these hashes; the calibration family must reuse them by import only.
        import json
        root = Path(__file__).resolve().parents[1]
        pinned = json.loads((root / "docs/results/hkde-inference/bank-manifest.json").read_text())["source_sha256"]
        from sbi4finance.gbm_bank import _sha256
        for name in ("option_bank.py", "option_panel.py"):
            self.assertEqual(_sha256(root / "src/sbi4finance" / name), pinned[name])


class InferenceAndCorrectionTests(unittest.TestCase):
    def test_conditioning_scales_quotes_and_logs_the_curve(self):
        task = small_rough_task()
        arrays = bank.generate_examples(config(task), "train", [0, 1])
        data = inference.training_data(arrays, task)
        assert_allclose(data["observations"], arrays["observations"][..., 0] / tasks.sigma(task))
        assert_allclose(data["context"], np.log(arrays["context_values"]))
        self.assertEqual(data["eta"].shape, (2, 1))
        augmented = inference.augmented_training_data(arrays, task, augmentations=3, seed=1)
        self.assertEqual(augmented["observations"].shape, (6, 9))
        again = inference.augmented_training_data(arrays, task, augmentations=3, seed=1)
        np.testing.assert_array_equal(augmented["observations"], again["observations"])
        with self.assertRaisesRegex(ValueError, "levels"):
            inference.conditioning_data(arrays["observations"], None, task)

    def test_fresh_noise_adds_a_new_unit_normal_error_to_clean_scaled_quotes(self):
        task = small_rough_task()
        arrays = bank.generate_examples(config(task), "train", [0, 1])
        clean = inference.clean_training_data(arrays, task)
        assert_allclose(clean["observations"], arrays["reference_prices"] / tasks.sigma(task))
        assert_allclose(clean["context"], np.log(arrays["context_values"]))
        augment = inference.fresh_noise(seed=4)
        big = {"observations": np.zeros((4000, 9)), "eta": np.ones((4000, 1))}
        first, second = augment(big), augment(big)
        self.assertFalse(np.shares_memory(first["observations"], big["observations"]))
        self.assertTrue(np.all(big["observations"] == 0))  # the batch itself is not modified
        self.assertFalse(np.allclose(first["observations"], second["observations"]))  # new draw per batch
        self.assertAlmostEqual(first["observations"].std(), 1.0, delta=0.02)
        np.testing.assert_array_equal(first["eta"], big["eta"])
        np.testing.assert_array_equal(inference.fresh_noise(seed=4)(big)["observations"], first["observations"])

    def test_logit_coordinates_and_proposal_density(self):
        task = tasks.heston_task()
        bounds = correction.bounds_of(task)
        theta = np.array([[0.05, 0.08, 0.7, -0.5]])
        assert_allclose(correction.to_bounded(correction.to_unbounded(theta, bounds), bounds), theta)
        z = correction.to_unbounded(theta, bounds)
        h = 1e-6
        numeric = np.sum(np.log((correction.to_bounded(z + h, bounds) - correction.to_bounded(z - h, bounds)) / (2 * h)))
        self.assertAlmostEqual(float(correction.log_jacobian(z, bounds)[0]), numeric, places=5)
        # Proposals drawn from the prior alone weigh as likelihood ratios: weights equal for a flat likelihood.
        draws = np.random.default_rng(0).uniform(bounds[:, 0], bounds[:, 1], (500, 4))
        theta, log_q, _ = correction.propose(draws, task, 4000, seed=1)
        self.assertTrue(np.all((theta > bounds[:, 0]) & (theta < bounds[:, 1])))
        result = correction.correct(np.zeros(len(theta)), log_q, task)
        self.assertGreater(result["efficiency"], 0.2)

    def test_importance_correction_recovers_a_known_gaussian_posterior(self):
        # A Gaussian target inside the box; proposals from network-like draws that are off by half an sd.
        task = tasks.heston_task()
        rng = np.random.default_rng(3)
        target, scale = np.array([0.085, 0.07, 0.75, -0.55]), np.array([0.005, 0.006, 0.06, 0.04])
        draws = target - 0.5 * scale + rng.normal(0, scale, (2000, 4))
        theta, log_q, _ = correction.propose(draws, task, 20000, seed=2, prior_fraction=0.05)
        log_lik = -0.5 * np.sum(((theta - target) / scale) ** 2, axis=1)
        result = correction.correct(log_lik, log_q, task)
        summary = correction.weighted_summary(theta, result["weights"])
        assert_allclose(summary["mean"], target, atol=0.1 * scale.max() + 0.0)
        assert_allclose(summary["sd"], scale, rtol=0.12)
        self.assertLess(result["khat"], 0.7)
        # The same correction with a flat likelihood in three coordinates must report a heavy tail.
        flat = correction.correct(-0.5 * ((theta[:, 0] - 0.085) / 0.005) ** 2, log_q, task)
        self.assertGreater(flat["khat"], 0.7)

    def test_pareto_khat_separates_light_and_heavy_tails(self):
        rng = np.random.default_rng(4)
        self.assertLess(correction.pareto_khat(rng.normal(0, 0.3, 4000)), 0.5)
        self.assertGreater(correction.pareto_khat(np.log(rng.pareto(0.8, 4000) + 1)), 0.7)


if __name__ == "__main__":
    unittest.main()


class ProductsPricerClassicalChainTests(unittest.TestCase):
    def test_products_in_the_black_scholes_limit(self):
        from sbi4finance.calibration import products
        curve = products.extend_curve(pricing.step_curve([0, 0.25, 0.5], [0.04, 0.04]), 2.0)
        spec = dict(products.default_specification())
        prices, errors = products.price_draw("rough", {"hurst": 0.2, "eta": 1e-8, "rho": -0.5}, curve, spec, 4096, seed=1)
        expected = [0.04, 0.2, float(surface.bs_call(0.0, 0.25, 0.2)), float(surface.bs_put(np.log(0.9), 2.0, 0.2))]
        for got, se, want in zip(prices, errors, expected):
            self.assertLess(abs(got - want), 4 * se + 1e-9)
        self.assertEqual(errors[0], 0.0)
        self.assertGreater(products.reserve([1.0, 1.1, 1.2, 1.3], 1.05), 0.0)

    def test_heston_euler_paths_match_the_fourier_price(self):
        from sbi4finance.calibration import products
        task = tasks.heston_task()
        parameters, v0 = tasks._heston_split(tasks.model_parameters(task, [0.04, 0.05, 0.6, -0.6]))
        paths = products.heston_paths(parameters, v0, 0.5, 504, 2**15, seed=2)
        mc, se = products.vanilla(paths, parameters["rho"], 0.95, 0.5, "put")
        exact = pricing.heston_surface(parameters, v0, [{"strike": 0.95, "maturity": 0.5, "option_type": "put"}],
                                       nodes=1024, upper_limit=600.0)[0]
        self.assertLess(abs(mc - exact), 4 * se + 2e-5)

    def test_pricer_learns_a_small_heston_bank_and_its_chain_runs(self):
        from sbi4finance.calibration import pricer_net
        task = tasks.heston_task(days=(10, 63), deltas=(-1.0, 0.0, 1.0))
        arrays = bank.generate_examples(config(task, (400, 100, 50)), "train", range(400))
        validation_arrays = bank.generate_examples(config(task, (400, 100, 50)), "validation", range(100))
        pricer, record = pricer_net.train(task, arrays, validation_arrays, widths=(64, 64), epochs=60, patience=60)
        error = pricer_net.errors(pricer, task, validation_arrays)
        self.assertLess(np.sqrt(np.mean(error**2)), 25.0)  # tiny bank and network: only the plumbing is tested
        target = pricer_net.log_posterior(pricer, task, validation_arrays["observations"][0, :, 0])
        z0 = correction.to_unbounded(validation_arrays["parameters"][:2], correction.bounds_of(task))
        chains, acceptance = pricer_net.metropolis(target, z0, 0.01 * np.eye(4), 50, seed=0)
        self.assertEqual(chains.shape, (2, 50, 4))
        with tempfile.TemporaryDirectory() as tmp:
            pricer.save(Path(tmp) / "p")
            np.testing.assert_allclose(pricer_net.Pricer.load(Path(tmp) / "p")(np.zeros((1, 4))), pricer(np.zeros((1, 4))))

    def test_least_squares_recovers_a_noise_free_heston_surface(self):
        from sbi4finance.calibration import classical
        task = tasks.heston_task(days=(10, 21, 63, 126), deltas=(-1.5, -0.5, 0.0, 0.5, 1.5))
        truth = np.array([0.05, 0.07, 0.7, -0.6])

        def price_fn(theta):
            parameters, v0 = tasks._heston_split(tasks.model_parameters(task, theta))
            return pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=512, upper_limit=400.0)

        observation = price_fn(truth)
        fits = classical.fit(task, observation, price_fn, [[0.04, 0.04, 0.5, -0.5]])
        assert_allclose(fits[0]["theta"], truth, rtol=1e-3, atol=1e-4)
        self.assertLess(fits[0]["chi2"], 1e-6)
        interval = classical.wald_interval(fits[0])
        self.assertTrue(np.all(interval[:, 0] < truth) and np.all(truth < interval[:, 1]))
        self.assertEqual(len(classical.distinct_modes(fits)), 1)
        starts = classical.latin_starts(task, 5, seed=0)
        self.assertEqual(starts.shape, (5, 4))

    def test_chain_pipeline_recovers_a_black_scholes_chain(self):
        from datetime import date
        from sbi4finance.calibration import chain
        valuation = date(2026, 10, 16)
        spot, rate, sigma = 600.0, 0.04, 0.2
        rows = []
        for expiry in (date(2026, 10, 22), date(2026, 11, 20), date(2027, 1, 15), date(2027, 4, 16), date(2027, 12, 17)):
            T = chain.trading_days(valuation, expiry) / 252
            D, F = np.exp(-rate * T), spot * np.exp(rate * T)
            for K in np.linspace(F * np.exp(-3.5 * sigma * np.sqrt(T)), F * np.exp(3.5 * sigma * np.sqrt(T)), 61):
                k = np.log(K / F)
                for kind in ("call", "put"):
                    price = D * F * float(surface.bs_price(k, T, sigma, kind))
                    half = max(0.01, 0.002 * price)
                    if price > 2 * half:
                        rows.append({"expiration": expiry, "strike": float(K), "type": kind,
                                     "bid": price - half, "ask": price + half, "date": valuation})
        result = chain.build_surface(rows, design_days=(5, 21, 63, 126, 252))
        expirations = result["diagnostics"]["expirations"]
        for e in expirations:
            T = e["days"] / 252
            self.assertAlmostEqual(e["forward"], spot * np.exp(rate * T), delta=0.05)
            self.assertAlmostEqual(e["discount"], np.exp(-rate * T), delta=1e-3)
            self.assertAlmostEqual(e["w_replicated"] / T, sigma**2, delta=0.004)
        assert_allclose(result["implied_vol_grid"] if "implied_vol_grid" in result else result["diagnostics"]["implied_vol_grid"], sigma, atol=0.003)
        assert_allclose(result["forward_variance_levels"], sigma**2, atol=0.004)
