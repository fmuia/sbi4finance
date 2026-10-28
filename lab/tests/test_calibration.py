import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sbi4finance.bergomi import option_panel
from sbi4finance.calibration import bank, correction, inference, pricing, surface, tasks


def small_rough_task(budget=10.0, ladder=(256, 512), cap=None):
    """Three maturities and three strikes, one block: fast enough for unit tests."""
    return tasks.rough_task(days=(5, 10, 21), deltas=(-1.0, 0.0, 1.0),
                            blocks=[{"horizon": 21 / 252, "n_steps": 42}], paths_ladder=ladder, budget=budget,
                            last_rung_cap=budget if cap is None else cap)


def config(task, splits=(3, 2, 2)):
    return {"task": task, "root_seed": 7, "splits": dict(zip(("train", "validation", "test"), splits))}


def json_config(name):
    import json
    return json.loads((Path(__file__).resolve().parents[1] / "configs" / "tasks" / name).read_text())


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
        bad["numerics"]["blocks"][0]["n_steps"] = 81  # five days would fall between grid points
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

    def test_power_law_walk_curve_prior(self):
        """The full bank's prior: positive levels inside vol_bounds, reproducible, and the pilot left unchanged."""
        pilot = json_config("calibration_rough_pilot.json")["task"]
        self.assertEqual(tasks.rough_task(), pilot)  # the default builder still gives the pilot's task
        full = json_config("calibration_rough_bank.json")["task"]
        prior = full["context_prior"]["forward_variance"]
        self.assertEqual(prior["kind"], "power_law_walk")
        draws = np.array([tasks.draw_context(full, np.random.default_rng(i))[0] for i in range(2000)])
        vol = np.sqrt(draws)
        self.assertTrue(np.all(vol >= prior["vol_bounds"][0]) and np.all(vol <= prior["vol_bounds"][1]))
        again = np.array([tasks.draw_context(full, np.random.default_rng(i))[0] for i in range(2000)])
        np.testing.assert_array_equal(draws, again)
        # SPY's curve of 21 October 2026 lies inside the prior's central 90% at every level, relative to level 1
        import json
        spy = np.array(json.loads((Path(__file__).resolve().parents[1] / "tutorial_final" / "results" / "calibration"
                                   / "spy" / "surface.json").read_text())["forward_variance_levels"])
        rel = np.log(draws) - np.log(draws[:, :1])
        low, high = np.quantile(rel, [0.05, 0.95], axis=0)
        target = np.log(spy) - np.log(spy[0])
        self.assertTrue(np.all((target >= low) & (target <= high)))
        c = config(small_rough_task(), splits=(2, 1, 1))
        c["task"]["context_prior"] = {"forward_variance": dict(prior)}
        a = bank.generate_examples(c, "train", [0, 1])
        sigma0, exponent = a["context_parameters"][0]
        self.assertTrue(prior["level_bounds"][0] <= sigma0 <= prior["level_bounds"][1])
        self.assertTrue(prior["exponent_bounds"][0] <= exponent <= prior["exponent_bounds"][1])
        bad = copy.deepcopy(c["task"])
        bad["context_prior"]["forward_variance"]["vol_bounds"] = [0.1, 0.5]  # does not contain the level bounds
        with self.assertRaisesRegex(ValueError, "vol_bounds"):
            tasks.validate_task(bad)
        self.assertEqual(tasks.FULL_BANK_CURVE_PRIOR, prior)  # the builder's constant and the committed config agree

    def test_monte_carlo_likelihood_is_unbiased_with_correlated_errors(self):
        """E[N(y; P_hat, V - S)] = N(y; P, V) when P_hat = P + e, e ~ N(0, S), even with strongly correlated e."""
        rng = np.random.default_rng(3)
        n = 6
        sigma = np.linspace(1.0, 2.0, n)
        a = 0.25 * sigma
        correlation = 0.8 * np.ones((n, n)) + 0.2 * np.eye(n)
        S = correlation * np.outer(a, a)
        C = 0.05 * np.diag(sigma**2)
        P = rng.normal(size=n)
        y = P + rng.normal(size=n) * sigma
        exact = correction.gaussian_loglik_mc(y, P, sigma, np.zeros((n, n)), data_covariance=C)
        e = rng.multivariate_normal(np.zeros(n), S, size=200_000)
        values = np.array([correction.gaussian_loglik_mc(y, P + x, sigma, S, data_covariance=C) for x in e[:20_000]])
        ratio = np.exp(values - exact)
        self.assertLess(abs(ratio.mean() - 1.0), 4 * ratio.std() / np.sqrt(len(ratio)))
        with self.assertRaisesRegex(ValueError, "eigenvalue"):
            correction.gaussian_loglik_mc(y, P, sigma, 4 * S)

    def test_trading_days_with_nyse_holidays(self):
        from datetime import date
        from sbi4finance.calibration import chain
        start = date(2026, 10, 22)
        self.assertEqual(chain.trading_days(start, date(2026, 11, 20)), 21)
        self.assertEqual(chain.trading_days(start, date(2027, 4, 16)), 126)  # weekdays, as SPY's surface
        self.assertEqual(chain.trading_days(start, date(2026, 12, 18), chain.NYSE_HOLIDAYS), 40)  # Thanksgiving
        self.assertEqual(chain.trading_days(start, date(2027, 4, 16), chain.NYSE_HOLIDAYS), 120)
        with self.assertRaisesRegex(ValueError, "published"):
            chain.trading_days(start, date(2029, 1, 19), chain.NYSE_HOLIDAYS)

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

    def test_last_rung_cap_keeps_surfaces_between_the_budget_and_the_cap(self):
        task = small_rough_task(budget=1e-6, ladder=(256,), cap=1e3)
        arrays = bank.generate_examples(config(task), "train", [0])
        ratio = arrays["pricing_standard_errors"][0] / tasks.sigma(task)
        self.assertTrue(np.all(ratio > 1e-6) and np.all(ratio <= 1e3))
        np.testing.assert_array_equal(arrays["pricing_paths"][0], [256])
        with self.assertRaisesRegex(ValueError, "at least the budget"):
            small_rough_task(budget=0.3, cap=0.2)

    def test_default_rough_task_prices_one_and_two_weeks_on_their_own_short_grid(self):
        task = tasks.rough_task()
        blocks = task["numerics"]["blocks"]
        assert_allclose([b["horizon"] * surface.TRADING_DAYS for b in blocks], [10, 63, 252])
        self.assertEqual([b["n_steps"] for b in blocks], [80, 504, 504])
        self.assertEqual(task["numerics"]["paths_ladder"], [2**k for k in range(15, 22)])
        self.assertEqual((task["numerics"]["max_mc_se_over_quote_sigma"],
                          task["numerics"]["max_mc_se_over_quote_sigma_last_rung"]), (0.25, 0.5))

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

    def test_monte_carlo_likelihood_is_an_unbiased_estimate_of_the_exact_likelihood(self):
        # Two quotes; Monte Carlo prices P + s * zeta. The average of exp(log-likelihood) over many
        # Monte Carlo draws must equal the exact likelihood with sigma^2, not sigma^2 + s^2 or + 2 s^2.
        task = tasks.heston_task(days=(21,), deltas=(-1.0, 1.0))
        sigma = tasks.sigma(task)
        rng = np.random.default_rng(11)
        exact_prices = np.array([0.01, 0.004])
        y = exact_prices + sigma * rng.standard_normal(2)
        s = 0.4 * sigma
        draws = exact_prices + s * rng.standard_normal((400_000, 2))
        estimate = np.mean(np.exp(correction.gaussian_loglik(task, y, draws, mc_variance=s**2)))
        exact = float(np.exp(correction.gaussian_loglik(task, y, exact_prices[None, :])[0]))
        self.assertAlmostEqual(estimate / exact, 1.0, delta=0.01)
        plug_in = np.mean(np.exp(correction.gaussian_loglik(task, y, draws)))
        widened = float(np.exp(-0.5 * np.sum((y - exact_prices) ** 2 / (sigma**2 + s**2)
                                            + np.log(2 * np.pi * (sigma**2 + s**2)))))
        self.assertAlmostEqual(plug_in / widened, 1.0, delta=0.01)  # the plug-in targets sigma^2 + s^2
        with self.assertRaises(ValueError):
            correction.gaussian_loglik(task, y, draws[:2], mc_variance=(0.95 * sigma) ** 2)

    def test_choosing_the_path_count_does_not_bias_the_prices(self):
        # A pricer whose estimated error grows with its price error, as with heavy-tailed payoffs. Accepting the
        # first path count whose own estimate passes the budget keeps the draws with small errors, which here are
        # the low prices; correction.price_for_likelihood chooses the count on a pilot and prices independently.
        rng = np.random.default_rng(5)
        sigma = np.array([1.0])

        def price(n, purpose, attempt):
            sd = 0.6 * np.sqrt(4096 / n)
            z = rng.standard_normal()
            estimated = sd * max(1.0 + 0.5 * z, 0.05)
            return np.array([sd * z]), np.array([estimated]), np.array([[estimated**2]])

        chosen = [correction.price_for_likelihood(price, sigma, 4096, budget=0.5, max_doublings=4)
                  for _ in range(20_000)]
        prices = np.array([p[0][0] for p in chosen if p[0] is not None])
        sds = np.array([0.6 * np.sqrt(4096 / p[2]) for p in chosen if p[0] is not None])
        self.assertGreater(len(prices), 19_000)
        self.assertLess(abs(prices.mean()), 4 * np.sqrt(np.mean(sds**2) / len(prices)))
        for p in chosen[:50]:  # the covariance returned is the pilot's, scaled to the count chosen
            if p[0] is not None:
                self.assertLessEqual(p[1][0, 0], 0.5**2 + 1e-12)
        selected = []  # the old rule, for contrast: accept the first count whose own estimate passes
        for _ in range(20_000):
            for k in range(5):
                prices_k, errors_k, _ = price(4096 * 2**k, "final", k)
                if errors_k[0] <= 0.5:
                    selected.append(prices_k[0])
                    break
        self.assertLess(np.mean(selected), -0.05)

    def test_pareto_khat_separates_light_and_heavy_tails(self):
        rng = np.random.default_rng(4)
        self.assertLess(correction.pareto_khat(rng.normal(0, 0.3, 4000)), 0.5)
        self.assertGreater(correction.pareto_khat(np.log(rng.pareto(0.8, 4000) + 1)), 0.7)

    def test_posterior_correlation_is_measured_in_logit_coordinates(self):
        from sbi4finance.calibration import validation
        rng = np.random.default_rng(6)
        x = rng.standard_normal((3, 5000))
        y = 0.8 * x + 0.6 * rng.standard_normal((3, 5000))
        unit = 1 / (1 + np.exp(-np.stack([x, y], axis=-1)))
        found = validation.posterior_correlations(2 + 3 * unit, [[2.0, 5.0], [2.0, 5.0]])[(0, 1)]
        assert_allclose(found["median"], 0.8, atol=0.02)

    def test_contraction_of_rho_eta_uses_its_own_prior_variance(self):
        from sbi4finance.calibration import validation
        rng = np.random.default_rng(5)
        # A posterior equal to the prior contracts by nothing.
        assert_allclose(validation.contraction(rng.uniform(0, 1, (200, 4000, 1)), np.array([[0.0, 1.0]])), [0.0],
                        atol=0.02)
        # rho * eta is not uniform on its range; with the uniform target's variance its prior would seem to contract.
        product = (rng.uniform(-0.99, -0.2, (200, 4000)) * rng.uniform(0.5, 3.5, (200, 4000)))[..., None]
        product_bounds = [-0.99 * 3.5, -0.2 * 0.5]
        variance = validation.product_prior_variance([-0.99, -0.2], [0.5, 3.5], product_bounds)
        self.assertGreater(validation.contraction(product, np.array([product_bounds]))[0], 0.5)
        assert_allclose(validation.contraction(product, np.array([product_bounds]), [variance]), [0.0], atol=0.02)


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

    def test_cboe_valuation_is_the_underlyings_last_session(self):
        from sbi4finance.calibration import chain
        import json as _json
        payload = {"timestamp": "2026-10-22 07:46:24",
                   "data": {"last_trade_time": "2026-10-21T16:00:00",
                            "options": [{"option": "SPY261023P00770000", "bid": 1.0, "ask": 1.1}]}}
        rows = chain.parse_cboe_json(_json.dumps(payload))
        self.assertEqual(rows[0]["date"].isoformat(), "2026-10-21")
        self.assertEqual(chain.trading_days(rows[0]["date"], rows[0]["expiration"]), 2)
        self.assertEqual((rows[0]["type"], rows[0]["strike"]), ("put", 770.0))

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
        # The synthetic strips reach 2.4 to 3.1 standardized units on each side; a European chain needs no
        # rate curve, so forward and discount come from the parity regression.
        result = chain.build_surface(rows, design_days=(5, 21, 63, 126, 252), replication_deltas=(-2.3, 2.3))
        expirations = result["diagnostics"]["expirations"]
        self.assertTrue(all(e["replication_covered"] for e in expirations))
        with self.assertRaises(ValueError):
            chain.build_surface(rows, design_days=(5, 21, 63, 126, 252), replication_deltas=(-8.0, 1.8))
        with_rate = chain.build_surface(rows, design_days=(5, 21, 63, 126, 252), replication_deltas=(-2.3, 2.3),
                                        rate=chain.rate_curve([(0.25, 2 * np.expm1(rate / 2)), (2.0, 2 * np.expm1(rate / 2))]))
        for e in with_rate["diagnostics"]["expirations"]:
            T = e["days"] / 252
            self.assertAlmostEqual(e["discount"], np.exp(-rate * T), delta=1e-9)
            self.assertAlmostEqual(e["forward"], spot * np.exp(rate * T), delta=0.05)
        for e in expirations:
            T = e["days"] / 252
            self.assertAlmostEqual(e["forward"], spot * np.exp(rate * T), delta=0.05)
            self.assertAlmostEqual(e["discount"], np.exp(-rate * T), delta=1e-3)
            self.assertAlmostEqual(e["w_replicated"] / T, sigma**2, delta=0.004)
        assert_allclose(result["implied_vol_grid"] if "implied_vol_grid" in result else result["diagnostics"]["implied_vol_grid"], sigma, atol=0.003)
        assert_allclose(result["forward_variance_levels"], sigma**2, atol=0.004)
