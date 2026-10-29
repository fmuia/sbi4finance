"""Six-target rough task: analytic curve numerics, role separation and append-only identity."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from sbi4finance.bergomi import curve_values
from sbi4finance.calibration import bank, bank_collection, pricing, surface, tasks


def small_task(**kwargs):
    return tasks.joint_rough_task(days=(21, 63), deltas=(-1.0, 0.0, 1.0),
                                  blocks=[{"horizon": 0.25, "n_steps": 24}], paths_ladder=(128,),
                                  budget=0.25, last_rung_cap=0.5, **kwargs)


def cheap_price(task, parameters, context_values, seed, diagnostics=None):
    c = len(task["context"]["contracts"])
    prices = np.random.default_rng(seed).uniform(0.01, 0.02, c)
    return prices, np.zeros(c), np.zeros((c, c)), np.zeros(c), np.zeros(tasks.block_count(task))


class JointCurveTests(unittest.TestCase):
    def test_analytic_curve_integral_and_small_decay_limit(self):
        t = np.array([0, 1e-10, 0.1, 1.0, 5.0])
        for short, long, decay in ((0.09, 0.01, 2.0), (0.01, 0.09, 0.8), (0.04, 0.04, 1e-12)):
            with self.subTest(coefficients=(short, long, decay)):
                values = pricing.heston_variance_values(t, short, long, decay)
                total = pricing.heston_integrated_variance(t, short, long, decay)
                self.assertEqual(total[0], 0.0)
                self.assertTrue(np.all(values > 0))
                self.assertTrue(np.all(np.diff(total) > 0))
                for horizon, integral in zip(t[2:], total[2:]):
                    numerical, _ = quad(lambda time: pricing.heston_variance_values(time, short, long, decay),
                                        0, horizon, epsabs=1e-13, epsrel=1e-13)
                    assert_allclose(integral, numerical, rtol=1e-12, atol=1e-14)
        assert_allclose(pricing.heston_integrated_variance(t, 0.02, 0.09, 1e-12), 0.02 * t, rtol=1e-10)

    def test_invalid_curve_inputs_fail(self):
        for coefficients in ((0.0, 0.04, 2.0), (0.04, -0.01, 2.0), (0.04, 0.01, 0.0),
                             (0.04, np.nan, 2.0)):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                pricing.heston_variance_values([0, 1], *coefficients)
        for times in ((-1.0, 1.0), (0.0, np.inf)):
            with self.assertRaises(ValueError):
                pricing.heston_integrated_variance(times, 0.04, 0.01, 2.0)

    def test_joint_task_has_new_identity_six_targets_and_no_context_prior(self):
        task = tasks.joint_rough_task()
        tasks.validate_task(task)
        self.assertEqual(task["parameter_names"], ["hurst", "eta", "rho", "short_variance", "long_variance", "decay"])
        self.assertEqual(len(task["context"]["contracts"]), 63)
        self.assertEqual(task["context_prior"], {})
        self.assertTrue(tasks.is_calibration_task(task))
        self.assertTrue(tasks.is_joint_curve_task(task))
        self.assertNotEqual(task["task_id"], tasks.rough_task()["task_id"])
        row = np.array([0.1, 1.5, -0.7, 0.09, 0.01, 2.0])
        self.assertEqual(set(tasks.model_parameters(task, row)), {"hurst", "eta", "rho"})
        with self.assertRaises(ValueError):
            tasks.draw_context(task, np.random.default_rng(0))
        for mutation in (lambda t: t["prior_bounds"][3].__setitem__(0, 0.0),
                         lambda t: t["context_prior"].update({"forward_variance": {}}),
                         lambda t: t["parameter_names"].reverse()):
            changed = copy.deepcopy(task)
            mutation(changed)
            with self.assertRaises(ValueError):
                tasks.validate_task(changed)

    def test_fine_grid_values_are_analytic_and_stored_buckets_are_only_auxiliary(self):
        task = tasks.joint_rough_task()
        row = np.array([0.1, 1.5, -0.7, 0.09, 0.01, 2.0])
        curve = tasks.joint_curve(task, row)
        for block in task["numerics"]["blocks"]:
            grid = block["horizon"] / block["n_steps"] * np.arange(block["n_steps"] + 1)
            assert_allclose(curve_values(curve, grid), pricing.heston_variance_values(grid, *row[3:]),
                            rtol=1e-14, atol=1e-16)
        averages, coefficients = tasks.derived_context(task, row)
        assert_array_equal(coefficients, row[3:])
        assert_allclose(np.cumsum(averages * np.diff(tasks.knots(task))),
                        pricing.heston_integrated_variance(tasks.knots(task)[1:], *row[3:]), rtol=1e-14)
        with self.assertRaises(ValueError):
            tasks.curve(task, averages)

    def test_zero_eta_rho_matches_discrete_bs_and_converges_to_analytic_integral(self):
        coefficients = (0.09, 0.01, 2.0)
        horizon = 0.25
        contract = {"strike": 1.0, "maturity": horizon, "option_type": "put"}
        analytic = float(surface.bs_price(0.0, horizon,
                                         np.sqrt(pricing.heston_integrated_variance(horizon, *coefficients) / horizon),
                                         "put"))
        errors = []
        for steps in (8, 32, 128):
            grid = np.linspace(0.0, horizon, steps + 1)
            curve = pricing.heston_variance_curve(grid, *coefficients)
            prices, se, covariance = pricing.rough_block({"hurst": 0.1, "eta": 0.0, "rho": 0.0}, curve,
                                                        [contract], horizon=horizon, n_steps=steps,
                                                        n_pairs=32, seed=9)
            left_total = horizon / steps * pricing.heston_variance_values(grid[:-1], *coefficients).sum()
            expected = surface.bs_price(0.0, horizon, np.sqrt(left_total / horizon), "put")
            assert_allclose(prices, [expected], rtol=1e-12, atol=1e-14)
            self.assertLess(float(se.max()), 1e-14)
            self.assertLess(float(np.abs(covariance).max()), 1e-25)
            errors.append(abs(float(prices[0]) - analytic))
        self.assertLess(errors[-1], errors[0] / 10)
        self.assertTrue(all(left > right for left, right in zip(errors, errors[1:])))

    def test_joint_price_dispatch_uses_targets_not_auxiliary_curve_values(self):
        task = small_task()
        row = np.array([0.1, 0.0, 0.0, 0.09, 0.01, 2.0])
        auxiliary, _ = tasks.derived_context(task, row)
        expected = tasks.price(task, row, auxiliary, 7)
        # Derived context is storage/audit metadata; it is not the pricing input for this task.
        actual = tasks.price(task, row, np.full_like(auxiliary, 999.0), 7)
        for left, right in zip(actual, expected):
            assert_array_equal(left, right)
        self.assertEqual(len(actual[0]), len(task["context"]["contracts"]))
        self.assertEqual(actual[-1].tolist(), [128])


class JointBankTests(unittest.TestCase):
    def setUp(self):
        self.config = {"task": small_task(), "root_seed": 271026,
                       "splits": {"train": 3, "validation": 1, "test": 1}}

    @patch.object(tasks, "price", side_effect=cheap_price)
    def test_generation_derives_curve_from_targets_and_is_stable_under_growth(self, mock_price):
        config = self.config
        rows = bank.generate_examples(config, "train", [0, 1, 2])
        self.assertEqual(rows["parameters"].shape, (3, 6))
        self.assertEqual(rows["context_parameters"].shape, (3, 3))
        for i, parameters in enumerate(rows["parameters"]):
            expected_levels, expected_coefficients = tasks.derived_context(config["task"], parameters)
            assert_array_equal(rows["context_parameters"][i], expected_coefficients)
            assert_array_equal(rows["context_values"][i], expected_levels)
        reordered = bank.generate_examples(config, "train", [2, 0])
        grown = copy.deepcopy(config)
        grown["splits"]["train"] = 100
        expanded = bank.generate_examples(grown, "train", [0, 1, 2])
        for key in rows:
            assert_array_equal(rows[key][[2, 0]], reordered[key], err_msg=key)
            assert_array_equal(rows[key], expanded[key], err_msg=key)
        bank.check_arrays(rows, config, "train", 0, 3)
        corrupt = {key: value.copy() for key, value in rows.items()}
        corrupt["context_values"][0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "joint curve auxiliaries"):
            bank.check_arrays(corrupt, config, "train", 0, 3)

    @patch.object(tasks, "price", side_effect=cheap_price)
    def test_incremental_snapshot_roles_and_no_repricing(self, mock_price):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank_collection.create_collection(root / "collection", self.config)
            for job in bank_collection.plan_extension(root / "collection", 3, 2)["jobs"]:
                bank_collection.run_job(root / "collection", job)
            first = bank_collection.freeze_snapshot(root / "collection", 3, root / "three")
            self.assertIn("never conditioning", first["roles"]["context_values"])
            self.assertIn("six independent uniforms", first["prior"])
            calls = mock_price.call_count
            extension = bank_collection.plan_extension(root / "collection", 5, 2)
            self.assertEqual(sum(j["stop"] - j["start"] for j in extension["jobs"]), 2)
            for job in extension["jobs"]:
                bank_collection.run_job(root / "collection", job)
            self.assertEqual(mock_price.call_count - calls, 2)
            bank_collection.freeze_snapshot(root / "collection", 5, root / "five")
            _, before = bank_collection.load_bank(root / "three")
            _, after = bank_collection.load_bank(root / "five")
            for key in before["train"]:
                assert_array_equal(before["train"][key], after["train"][key][:3], err_msg=key)
            for split in ("validation", "test"):
                for key in before[split]:
                    assert_array_equal(before[split][key], after[split][key], err_msg=f"{split}:{key}")


if __name__ == "__main__":
    unittest.main()
