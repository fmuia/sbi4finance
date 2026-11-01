"""The E2 prototype must preserve its control law and state its limited scope."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.path_inference.bank import teaching_task


_SPEC = importlib.util.spec_from_file_location("path_projected_euler", Path(__file__).resolve().parents[1] / "checks/path_projected_euler.py")
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


class ProjectedEulerPilotTests(unittest.TestCase):
    def test_prespecified_panel_stratifies_actual_prior_and_contains_all_corners(self):
        panel, groups = pilot.parameter_panel()
        again, _ = pilot.parameter_panel()
        assert_array_equal(panel, again)
        self.assertEqual(panel.shape, (64, 5))
        self.assertEqual(groups.count("stratified_prior"), 48)
        task = teaching_task("heston")
        columns = {"drift": 0, "kappa": 1, "long_variance": 2, "vol_of_vol": 3, "rho": 4}
        for name, (low, high), kind in zip(task["parameter_names"], task["prior_bounds"], task["prior_types"]):
            col = panel[:48, columns[name]]
            u = np.log(col / low) / np.log(high / low) if kind == "log_uniform" else (col - low) / (high - low)
            assert_array_equal(np.sort((u * 48).astype(int)), np.arange(48))
            self.assertTrue(np.all(panel[:, columns[name]] >= low))
            self.assertTrue(np.all(panel[:, columns[name]] <= high))
        self.assertEqual(len(np.unique(panel[48:, 1:], axis=0)), 16)

    def test_coupling_preserves_brownian_sum_in_both_independent_channels(self):
        fine = np.random.default_rng(7).standard_normal((3, 9, 32, 2))
        for grid in pilot.GRIDS:
            coarse = pilot.coupled_normals(fine, grid)
            assert_allclose(coarse.sum(axis=2) / np.sqrt(grid), fine.sum(axis=2) / np.sqrt(32), rtol=2e-15, atol=2e-15)
        with self.assertRaises(ValueError):
            pilot.coupled_normals(fine, 7)

    def test_signed_control_is_bitwise_production_kernel_at_every_grid(self):
        panel, _ = pilot.parameter_panel()
        parameters = panel[[0, 50, 63]]
        fine = np.random.default_rng(10).standard_normal((3, 17, 32, 2))
        dt = np.full(17, 1 / 252)
        for grid in pilot.GRIDS:
            normals = pilot.coupled_normals(fine, grid)
            result = pilot.diagnose(parameters, dt, [.04, -.02, .001], normals, projected=False)
            pilot.assert_production_parity(parameters, dt, [.04, -.02, .001], normals, result)

    def test_zero_rebound_and_price_increment_use_left_variance_and_leverage(self):
        parameters = np.array([[.05, .25, .01, .6, -.95]])
        dt = np.array([1 / 252])
        normals = np.zeros((1, 1, 2, 2))
        normals[0, 0, 1] = [1.2, -.7]
        result = pilot.diagnose(parameters, dt, 0., normals, projected=True)
        h = dt[0] / 2
        c = .25 * .01 * h
        expected_iv = c * h
        expected_r = .05 * dt[0] - .5 * expected_iv + np.sqrt(c * h) * (-.95 * 1.2 + np.sqrt(1 - .95 ** 2) * -.7)
        assert_allclose(result["integrated_variance"], expected_iv, rtol=1e-15, atol=0)
        assert_allclose(result["log_returns"], expected_r, rtol=1e-15, atol=0)
        self.assertEqual(result["zero_left_steps"][0, 0], 1)
        # One substep from zero really is deterministic; >=2 is necessary.
        one = pilot.diagnose(parameters, dt, 0., np.zeros((1, 1, 1, 2)), projected=True)
        self.assertEqual(one["integrated_variance"][0, 0], 0)

    def test_signed_daily_drift_atom_is_control_not_hidden_by_projection(self):
        parameters = np.array([[.05, .25, .01, .6, -.95]])
        normals = np.zeros((1, 3, 8, 2))
        normals[0, 0, 0, 0] = -100
        dt = np.full(3, 1 / 252)
        signed = pilot.diagnose(parameters, dt, .04, normals, projected=False)
        projected = pilot.diagnose(parameters, dt, .04, normals, projected=True)
        self.assertEqual(signed["integrated_variance"][0, 1], 0)
        assert_allclose(signed["log_returns"][0, 1, 0], .05 / 252, rtol=1e-15, atol=0)
        self.assertTrue(np.all(projected["integrated_variance"] > 0))
        self.assertTrue(np.all(projected["variance_state"] >= 0))
        self.assertTrue(np.all(projected["zero_left_steps"] <= 4))

    def test_strict_positivity_is_not_an_unconditional_uniform_bound(self):
        parameters = np.array([[.05, .25, .01, .6, -.95]])
        h = 1 / 2016
        found = []
        for epsilon in [1e-10, 1e-12]:
            # Supported finite Gaussian innovations hold a tiny positive state.
            zv = -.25 * (.01 - epsilon) * h / (.6 * np.sqrt(epsilon * h))
            normals = np.zeros((1, 1, 8, 2))
            normals[..., 0] = zv
            result = pilot.diagnose(parameters, [1 / 252], epsilon, normals, projected=True)
            found.append(result["integrated_variance"][0, 0])
            self.assertGreater(found[-1], 0)
            assert_allclose(found[-1], 8 * epsilon * h, rtol=2e-8, atol=0)
        self.assertLess(found[1], found[0] / 90)
        self.assertEqual(pilot.analytical_scope()["unconditional_uniform_lower_bound"], 0)

    def test_nonfinite_and_negative_projected_state_fail_without_redraw(self):
        parameters = np.array([[.05, .25, .01, .6, -.95]])
        normals = np.zeros((1, 1, 8, 2))
        with self.assertRaises(ValueError):
            pilot.diagnose(parameters, [1 / 252], -.01, normals, projected=True)
        normals[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            pilot.diagnose(parameters, [1 / 252], .04, normals, projected=False)

    def test_real_cli_entrypoint_executes_after_imported_kernel_tests(self):
        lab = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, str(lab / "checks/path_projected_euler.py"),
                                      "--smoke", "--days", "2", "--out", directory],
                                     cwd=lab, env={**os.environ, "PYTHONPATH": str(lab / "src")},
                                     text=True, capture_output=True, check=True)
            receipt = json.loads(process.stdout)
            self.assertFalse(receipt["complete_prespecified_panel"])
            self.assertTrue(Path(receipt["summary"]).is_file())

    def test_smoke_writes_source_bound_finite_receipt_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            report = pilot.run_panel(directory, smoke=True, days=3)
            self.assertFalse(report["complete_prespecified_panel"])
            self.assertTrue(report["signed_baseline_bitwise_production_parity"])
            self.assertTrue(report["all_recorded_values_finite"])
            self.assertFalse(report["characteristic_function_comparison_performed"])
            for events in report["aggregate_events"].values():
                self.assertIn("floating_equal_return_drift_dt_days", events)
            self.assertEqual(len(report["moment_comparisons"]), 6)
            self.assertEqual(len(report["paired_comparisons"]), 7)
            self.assertEqual(len(report["source_sha256"]), 3)
            self.assertEqual(report["payload"]["sha256"], pilot.sha256(Path(directory) / report["payload"]["path"]))
            saved = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(saved["profile_id"], pilot.PROFILE_ID)
            self.assertFalse(saved["prototype_task_published"])
            with self.assertRaises(FileExistsError):
                pilot.run_panel(directory, smoke=True, days=3)
        with self.assertRaises(ValueError):
            pilot.run_panel("unused", days=3)


if __name__ == "__main__":
    unittest.main()
