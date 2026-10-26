"""Analytical and independent-stream checks for the fixed-budget grid pilot."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import numerical_pilot as pilot, pricing, surface, tasks
from sbi4finance.gbm_bank import _json


class RoughNumericalPilotTests(unittest.TestCase):
    def setUp(self):
        self.task = tasks.joint_rough_task(days=(21, 63), deltas=(-1, 0, 1),
                    blocks=[{"horizon": .25, "n_steps": 24}], paths_ladder=(128,))
        self.case = {"name": "test", "parameters": [.1, 1.5, -.7, .09, .01, 2.]}

    def test_seed_identity_is_case_based_and_grid_independent(self):
        seed = pilot.stream_seed(self.task, self.case, 17, 1, 0, 0)
        self.assertEqual(seed, pilot.stream_seed(copy.deepcopy(self.task), dict(self.case), 17, 1, 0, 0))
        seeds = {pilot.stream_seed(self.task, self.case, 17, grid, rep, block)
                 for grid in (1, 2, 4) for rep in (0, 1) for block in (0, 1)}
        self.assertEqual(len(seeds), 12)
        changed = {**self.case, "name": "other"}
        self.assertNotEqual(seed, pilot.stream_seed(self.task, changed, 17, 1, 0, 0))

    def test_refinement_rebuilds_grid_without_changing_original(self):
        fine = pilot.refined_task(self.task, 4)
        self.assertEqual(fine["numerics"]["blocks"][0]["n_steps"], 96)
        self.assertEqual(self.task["numerics"]["blocks"][0]["n_steps"], 24)
        self.assertEqual(fine["prior_bounds"], self.task["prior_bounds"])

    def test_zero_eta_rho_has_exact_discrete_price_and_refinement(self):
        values = [.1, 0., 0., .09, .01, 2.]
        k, maturity, kind = surface.contract_arrays(self.task["context"]["contracts"])
        analytic_iv = pricing.heston_integrated_variance(maturity, *values[3:])
        analytic = surface.bs_price(k, maturity, np.sqrt(analytic_iv / maturity), kind)
        errors = []
        for multiplier in (1, 2, 4):
            prices, se = pilot.fixed_surface(self.task, values, multiplier=multiplier, paths=64, seeds=[13])
            steps = 24 * multiplier
            dt = .25 / steps
            xi = pricing.heston_variance_values(dt * np.arange(steps), *values[3:])
            indices = np.rint(maturity / dt).astype(int)
            integrated = np.cumsum(xi)[indices - 1] * dt
            exact = surface.bs_price(k, maturity, np.sqrt(integrated / maturity), kind)
            np.testing.assert_allclose(prices, exact, atol=1e-13, rtol=1e-12)
            self.assertLess(se.max(), 1e-13)
            errors.append(np.max(np.abs(prices - analytic)))
        self.assertLess(errors[-1], errors[0] / 3)

    def test_independent_mean_difference_uncertainty(self):
        rows0 = [{"prices": [1.], "standard_errors": [.2]}, {"prices": [3.], "standard_errors": [.2]}]
        rows1 = [{"prices": [2.], "standard_errors": [.4]}, {"prices": [4.], "standard_errors": [.4]}]
        result = pilot.comparison(rows0, rows1, [.5])
        np.testing.assert_allclose(result["fine_minus_coarse_over_quote_sigma"], [2.])
        np.testing.assert_allclose(result["mc_se_of_difference_over_quote_sigma"], [np.sqrt(.1) / .5])
        self.assertIn("not simultaneous", result["interpretation"])

    def test_failures_are_retained_without_replacement(self):
        c = len(self.task["context"]["contracts"])
        def price(*args, **kwargs):
            if kwargs["multiplier"] == 2:
                raise ValueError("declared numerical failure")
            return np.ones(c), np.ones(c) * tasks.sigma(self.task)
        with patch.object(pilot, "fixed_surface", side_effect=price) as mocked:
            result = pilot.run_case(self.task, self.case, paths=64)
        self.assertEqual(mocked.call_count, 6)
        self.assertEqual(len(result["runs"]), 6)
        self.assertEqual(sum(r["status"] == "failed" for r in result["runs"]), 2)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all(r["status"] == "incomplete" for r in result["comparisons"]))
        self.assertTrue(all(not r["mc_budget_met"] for r in result["runs"] if r["status"] == "priced"))

    def test_panel_is_predeclared_and_inside_candidate_prior(self):
        panel = pilot.cases(tasks.joint_rough_task())
        self.assertEqual(len(panel), 8)
        self.assertEqual(len({c["name"] for c in panel}), 8)
        changed = tasks.joint_rough_task()
        changed["prior_bounds"][3] = [.02, .03]
        with self.assertRaisesRegex(ValueError, "outside"):
            pilot.cases(changed)

    def test_components_preserve_seed_identity_and_match_serial_comparison(self):
        c = len(self.task["context"]["contracts"])
        def price(*args, **kwargs):
            # Depends on the declared slot's seed, never on call order.
            value = np.random.default_rng(kwargs["seeds"][0]).normal()
            return np.full(c, value), np.full(c, .001)
        settings = dict(paths=64, multipliers=(4, 8), replicates=2, root_seed=37)
        with patch.object(pilot, "fixed_surface", side_effect=price):
            serial = pilot.run_case(self.task, self.case, **settings)
            parts = [pilot.run_case(self.task, self.case, only_grid=grid, only_replicate=rep, **settings)
                     for grid in (4, 8) for rep in (0, 1)]
        combined = pilot.combine_components(self.task, self.case, list(reversed(parts)))
        self.assertEqual(combined["comparisons"], serial["comparisons"])
        self.assertEqual(combined["status"], "completed")
        for old, new in zip(serial["runs"], combined["runs"]):
            self.assertEqual(old["pricing_seeds"], new["pricing_seeds"])
            self.assertEqual(old["prices"], new["prices"])
        for invalid in (parts[:-1], parts + [parts[0]]):
            with self.assertRaises(ValueError):
                pilot.combine_components(self.task, self.case, invalid)
        changed = copy.deepcopy(parts)
        changed[0]["runs"][0]["pricing_seeds"][0] += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            pilot.combine_components(self.task, self.case, changed)
        changed = copy.deepcopy(parts)
        changed[0]["paths_per_grid_replicate"] *= 2
        with self.assertRaisesRegex(ValueError, "settings"):
            pilot.combine_components(self.task, self.case, changed)
        failed = copy.deepcopy(parts)
        failed[0]["runs"][0].update(status="failed", error="recorded numerical failure")
        result = pilot.combine_components(self.task, self.case, failed)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["comparisons"][0]["status"], "incomplete")
        self.assertEqual(len(result["runs"]), 4)

    def test_selected_block_allocation_and_legacy_seed_protocol(self):
        task = tasks.joint_rough_task()
        self.assertEqual(pilot.block_scales(task, 16, refine_block=0, fixed_other_grid=4), [16, 4, 4])
        fine = pilot.refined_task(task, 16, block_multipliers=[16, 4, 4])
        self.assertEqual([b["n_steps"] for b in fine["numerics"]["blocks"]], [1280, 2016, 2016])
        self.assertEqual([b["n_steps"] for b in task["numerics"]["blocks"]], [80, 504, 504])
        # Pin the prior panel's public stream protocol, excluding optional mode metadata.
        digest = hashlib.sha256(_json({"task": task, "case": self.case}).encode()).hexdigest()
        words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
        old_seed = int(np.random.SeedSequence([0x52475049, 17, *words, 4, 0, 0])
                       .generate_state(1, dtype=np.uint64)[0])
        self.assertEqual(pilot.stream_seed(task, self.case, 17, 4, 0, 0), old_seed)
        self.assertNotEqual(pilot.stream_seed(task, self.case, 17, 4, 0, 0,
                                             block_multipliers=[4, 4, 4]), old_seed)
        with self.assertRaises(ValueError):
            pilot.block_scales(task, 4, refine_block=3, fixed_other_grid=4)

    def test_selected_block_components_reproduce_serial_and_have_disjoint_streams(self):
        task = tasks.joint_rough_task()
        def price(*args, **kwargs):
            value = np.random.default_rng(kwargs["seeds"][0]).normal()
            return np.full(63, value), np.full(63, .001)
        settings = dict(paths=64, multipliers=(4, 8, 16), replicates=2, root_seed=37,
                        refine_block=0, fixed_other_grid=4)
        with patch.object(pilot, "fixed_surface", side_effect=price):
            serial = pilot.run_case(task, self.case, **settings)
            parts = [pilot.run_case(task, self.case, only_grid=g, only_replicate=r, **settings)
                     for g in (4, 8, 16) for r in (0, 1)]
        combined = pilot.combine_components(task, self.case, parts)
        self.assertEqual(combined["comparisons"], serial["comparisons"])
        old = {pilot.stream_seed(task, self.case, 37, g, r, b)
               for g in (4, 8, 16) for r in (0, 1) for b in range(3)}
        new = {seed for row in serial["runs"] for seed in row["pricing_seeds"]}
        self.assertEqual(len(new), 18)
        self.assertFalse(old & new)
        self.assertEqual([row["block_multipliers"] for row in serial["runs"]],
                         [[4, 4, 4], [4, 4, 4], [8, 4, 4], [8, 4, 4], [16, 4, 4], [16, 4, 4]])
        changed = copy.deepcopy(parts)
        changed[0]["runs"][0]["block_multipliers"] = [4, 8, 4]
        with self.assertRaisesRegex(ValueError, "slot"):
            pilot.combine_components(task, self.case, changed)

    def test_selected_block_changes_short_prices_but_not_fixed_long_grid(self):
        task = tasks.joint_rough_task(days=(21, 63), deltas=(-1, 0, 1),
                  blocks=[{"horizon": 21/252, "n_steps": 8}, {"horizon": .25, "n_steps": 24}],
                  paths_ladder=(128,))
        values = [.1, 0., 0., .09, .01, 2.]
        results = []
        for g in (4, 8):
            p, _ = pilot.fixed_surface(task, values, multiplier=g, paths=64, seeds=[13, 14],
                                       block_multipliers=pilot.block_scales(task, g, refine_block=0,
                                                                           fixed_other_grid=4))
            results.append(p)
        self.assertGreater(np.max(np.abs(results[0][:3] - results[1][:3])), 1e-8)
        np.testing.assert_allclose(results[0][3:], results[1][3:], atol=1e-13, rtol=1e-12)

    def test_short_block_array_slots_and_historical_mapping(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_core_pilot_array.py"
        spec = importlib.util.spec_from_file_location("pilot_array", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        slots, outputs = set(), set()
        for index in range(12):
            cmd = module.command("rough-short-block", index, Path("/tmp/output"))
            args = {key: cmd[i + 1] for i, key in enumerate(cmd[:-1]) if key.startswith("--")}
            slots.add((args["--case"], args["--only-grid"], args["--only-replicate"]))
            outputs.add(args["--out"])
            self.assertEqual(args["--paths"], "131072")
            self.assertEqual(args["--refine-block"], "0")
            self.assertEqual(args["--fixed-other-grid"], "4")
        self.assertEqual(len(outputs), 12)
        self.assertEqual(slots, {(case, grid, rep) for case in ("2", "6")
                                 for grid in ("4", "8", "16") for rep in ("0", "1")})
        legacy = module.command("rough-fine", 0, Path("/tmp/output"))
        self.assertEqual(legacy[legacy.index("--case") + 1], "4")
        self.assertEqual(legacy[legacy.index("--paths") + 1], "1048576")


if __name__ == "__main__":
    unittest.main()
