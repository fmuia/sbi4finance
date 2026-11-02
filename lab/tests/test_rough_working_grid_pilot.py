"""Frozen cohort/stream identities and real tiny-price execution/reduction."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import numerical_pilot, pricing, tasks
from sbi4finance.calibration import rough_working_grid_pilot as pilot
from sbi4f_tutorial.shared import pricing as core_pricing


class WorkingGridDeclarationTests(unittest.TestCase):
    def test_reviewed_counts_grids_resources_and_all_actual_streams(self):
        declaration = pilot.declare(tasks.joint_rough_task(), source_revision="frozen-test-revision")
        self.assertEqual(declaration["root_seed"], 271100)
        self.assertEqual(len(declaration["prior_points"]), 16)
        components = declaration["components"]
        self.assertEqual(len(components), 44)
        self.assertEqual([sum(c["mode"] == name for c in components)
                          for name in ("adaptive", "fixed_anchor")], [32, 12])
        self.assertEqual([[b["n_steps"] for b in p["task"]["numerics"]["blocks"]]
                          for p in declaration["policies"]], [[1280, 2016, 2016], [2560, 4032, 4032]])
        self.assertEqual([i for i, c in enumerate(components) if c["resources"]["wall_limit_seconds"] == 5400],
                         [38, 39])
        self.assertEqual(declaration["component_wall_allocation_seconds"], 67200)
        self.assertEqual(declaration["whole_pilot_cpu_ceiling_seconds"], 72000)
        self.assertEqual(declaration["seed_audit"], {"passed": True, "unique_reduced_uint64": 768,
                         "unique_initialized_pcg64_states": 724, "all_possible_block_rung_streams": 708})
        for point in declaration["prior_points"]:
            expected = np.random.default_rng(point["parameter_seed"]).uniform(
                np.asarray(declaration["task"]["prior_bounds"])[:, 0],
                np.asarray(declaration["task"]["prior_bounds"])[:, 1])
            np.testing.assert_array_equal(point["parameters"], expected)
            pair = components[2 * point["point_id"]:2 * point["point_id"] + 2]
            self.assertEqual(pair[0]["parameters"], pair[1]["parameters"])
            self.assertNotEqual(pair[0]["pricing_seed"], pair[1]["pricing_seed"])
        for component in components:
            if component["mode"] == "fixed_anchor":
                self.assertEqual(component["fixed_total_paths"],
                                 {0: 131072, 4: 1048576, 6: 524288}[component["point_id"]])
            for stream in component["streams"]:
                sequence = np.random.SeedSequence(stream["seed_sequence_entropy"])
                np.testing.assert_array_equal(sequence.generate_state(4, dtype=np.uint64),
                                              stream["seed_sequence_state_uint64"])
                self.assertEqual(pilot._rng_digest(sequence), stream["pcg64_initial_state_sha256"])
        pilot.validate_declaration(json.loads(json.dumps(declaration)), execution=True)

    def test_collisions_fail_without_replacing_seeds(self):
        with patch.object(pilot, "_seed", return_value=17), self.assertRaisesRegex(ValueError, "collision"):
            pilot.declare(tasks.joint_rough_task(), source_revision="test")
        # Distinct reduced words are insufficient if the actual generator states collide.
        with patch.object(pilot, "_rng_digest", return_value="same-state"), \
                self.assertRaisesRegex(ValueError, "collision"):
            pilot.declare(tasks.joint_rough_task(), source_revision="test")

    def test_digest_and_execution_source_runtime_must_match(self):
        declaration = pilot.declare(tasks.joint_rough_task(), source_revision="test")
        changed = copy.deepcopy(declaration)
        changed["components"][0]["pricing_seed"] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            pilot.validate_declaration(changed)
        with patch.object(pilot, "_runtime", return_value={"different": "runtime"}), \
                self.assertRaisesRegex(ValueError, "runtime"):
            pilot.validate_declaration(declaration, execution=True)
        # Reduction need not run on the generation host; it uses the exact frozen table.
        with patch.object(pilot, "_runtime", return_value={"different": "runtime"}):
            pilot.validate_declaration(declaration)


class WorkingGridExecutionTests(unittest.TestCase):
    def setUp(self):
        # Only the tests replace expensive anchor budgets; the production CLI
        # declares the exact reviewed task and module constants unchanged.
        self.enterContext(patch.dict(pilot.ANCHOR_PATHS, {0: 64, 4: 64, 6: 64}, clear=True))
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.task = tasks.joint_rough_task(days=(21, 63, 252), deltas=(-1, 0, 1),
            blocks=[{"horizon": 21/252, "n_steps": 2}, {"horizon": .25, "n_steps": 6},
                    {"horizon": 1., "n_steps": 24}], paths_ladder=(64, 128))
        self.task["context"]["quote_noise_std"] = [.1] * len(self.task["context"]["contracts"])
        self.declaration = pilot.declare(self.task, root_seed=17, source_revision="tiny-test")

    def test_actual_adaptive_price_covariance_and_seed_identity(self):
        component = self.declaration["components"][0]
        task = self.declaration["policies"][0]["task"]
        expected = tasks.price(task, component["parameters"], [], component["pricing_seed"])
        original_price = tasks.price
        original_block = pricing.rough_block
        with patch.object(tasks, "price", wraps=original_price) as called:
            report = pilot.run_component(self.declaration, 0, self.root / "component-00")
        self.assertEqual(called.call_count, 1)
        self.assertIs(pricing.rough_block, original_block)
        self.assertEqual(report["status"], "priced")
        for key, value in zip(("prices", "standard_errors", "covariance"), expected[:3], strict=True):
            np.testing.assert_array_equal(report[key], value)
        self.assertEqual(report["paths_per_block"], np.asarray(expected[-1], dtype=int).tolist())
        self.assertGreater(report["instrumented_cpu_seconds"], report["summed_block_cpu_seconds"])
        self.assertTrue(all(a["status"] == "priced" and "covariance" in a for a in report["attempts"]))
        for attempt in report["attempts"]:
            self.assertIn(attempt["stream"], component["streams"])
        with self.assertRaises(FileExistsError):
            pilot.run_component(self.declaration, 0, self.root / "component-00")

    def test_fixed_total_paths_covariance_and_actual_budget_flags(self):
        with patch.object(tasks, "price", wraps=tasks.price) as adaptive:
            report = pilot.run_component(self.declaration, 32, self.root / "component-32")
        self.assertEqual(adaptive.call_count, 0)
        self.assertEqual(report["status"], "priced")
        self.assertEqual(report["paths_per_block"], [64, 64, 64])
        self.assertTrue(all(a["stream"]["paths"] == 64 for a in report["attempts"]))
        ratios = np.asarray(report["standard_errors"]) / tasks.sigma(self.task)
        self.assertEqual(report["mc_target_met"], bool(np.all(ratios <= .25)))
        self.assertEqual(report["within_exception_cap"], bool(np.all(ratios <= .5)))
        self.assertEqual(self.declaration["task"]["numerics"]["max_mc_se_over_quote_sigma"], .25)

    def test_atomic_attempt_journal_and_failures_are_retained(self):
        original = pricing.rough_block
        output = self.root / "failure"
        calls = []

        def fail_on_second(*args, **kwargs):
            journal = json.loads((output / "report.json").read_text())
            self.assertEqual(journal["status"], "running")
            self.assertEqual(journal["attempts"][-1]["status"], "running")
            calls.append(kwargs["seed"])
            if len(calls) == 2:
                raise ValueError("predeclared test failure")
            return original(*args, **kwargs)

        with patch.object(core_pricing, "rough_block", side_effect=fail_on_second) as mocked:
            report = pilot.run_component(self.declaration, 0, output)
            self.assertIs(core_pricing.rough_block, mocked)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual([a["status"] for a in report["attempts"]], ["priced", "failed"])
        self.assertIn("covariance", report["attempts"][0])
        self.assertNotIn("prices", report)
        self.assertIn("predeclared test failure", report["error"])
        result = pilot.reduce_components(self.declaration, [report])
        self.assertEqual(result["components"][0]["status"], "failed")
        self.assertEqual(result["declared_components"], 44)
        self.assertEqual(result["comparisons"][0]["status"], "incomplete")

    def test_interrupted_missing_and_censored_are_not_mc_failures(self):
        with patch.object(core_pricing, "rough_block", side_effect=SystemExit("time limit")):
            report = pilot.run_component(self.declaration, 0, self.root / "interrupted")
        self.assertEqual(report["status"], "interrupted")
        result = pilot.reduce_components(self.declaration, [report], accounting=[
            {"component_index": 0, "state": "TIMEOUT", "elapsed_seconds": 1200., "total_cpu_seconds": 1191.},
            {"component_index": 1, "state": "TIMEOUT", "elapsed_seconds": 1200., "total_cpu_seconds": 1192.}])
        self.assertEqual([r["status"] for r in result["components"][:3]], ["censored", "censored", "incomplete"])
        self.assertEqual(result["scheduler_cpu_seconds_where_available"], 2383.)
        self.assertEqual(result["by_policy"]["working"]["adaptive"]["priced_components_above_exception_cap"], 0)

    def test_complete_real_tiny_reduction_and_corrupt_inputs(self):
        reports = [pilot.run_component(self.declaration, index, self.root / f"component-{index:02d}")
                   for index in range(44)]
        self.assertTrue(all(r["status"] == "priced" for r in reports))
        result = pilot.reduce_components(self.declaration, list(reversed(reports)))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["comparisons"]), 19)
        expected = numerical_pilot.comparison([reports[0]], [reports[1]], tasks.sigma(self.task))
        self.assertEqual(result["comparisons"][0]["fine_minus_coarse_over_quote_sigma"],
                         expected["fine_minus_coarse_over_quote_sigma"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pilot.reduce_components(self.declaration, reports + [reports[0]])
        changed = copy.deepcopy(reports)
        bad_stream = copy.deepcopy(changed[0]["attempts"][0]["stream"])
        bad_stream["seed_sequence_state_uint64"][0] += 1
        changed[0]["attempts"][0]["stream"] = bad_stream
        with self.assertRaisesRegex(ValueError, "stream"):
            pilot.reduce_components(self.declaration, changed)
        missing = pilot.reduce_components(self.declaration, reports[:-1])
        self.assertEqual(missing["status"], "incomplete")
        self.assertEqual(missing["comparisons"][-1]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
