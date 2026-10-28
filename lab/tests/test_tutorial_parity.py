"""The tutorial's runtime (tutorial/src/sbi4f_tutorial) returns the lab's numbers, and its copied assets are current.

The tutorial never imports the lab. This test reads the tutorial's code from the lab's side only: a change to
either copy of a pricer, the network's conditioning or a quick check fails here before it reaches the stage.
"""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
sys.path.insert(0, str(REPO / "tutorial" / "src"))

from sbi4f_tutorial import bs_hierarchy as t_sector  # noqa: E402
from sbi4f_tutorial import heston as t_heston  # noqa: E402
from sbi4f_tutorial import inference as t_inference  # noqa: E402
from sbi4f_tutorial import long_end as t_long_end  # noqa: E402
from sbi4f_tutorial import surface as t_surface  # noqa: E402
from sbi4f_tutorial import tasks as t_tasks  # noqa: E402
from sbi4f_tutorial import teaching as t_teaching  # noqa: E402

from sbi4finance.calibration import inference, surface, tasks, teaching  # noqa: E402

HESTON_TASK = json.loads((REPO / "tutorial" / "assets" / "checkpoints" / "heston" / "task.json").read_text())
SPY = json.loads((REPO / "tutorial" / "assets" / "results" / "calibration" / "spy" / "surface.json").read_text())


def lab_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParityTests(unittest.TestCase):
    def test_design_grid_noise_and_black_scholes(self):
        self.assertEqual(t_surface.design_grid(), surface.design_grid())
        self.assertEqual(t_surface.quote_noise(t_surface.design_grid()), surface.quote_noise(surface.design_grid()))
        k, maturity, kind = surface.contract_arrays(surface.design_grid())
        for sigma in (0.08, 0.2, 0.55):
            prices = surface.bs_price(k, maturity, sigma, kind)
            assert_array_equal(t_surface.bs_price(k, maturity, sigma, kind), prices)
            assert_array_equal(t_surface.implied_vol(prices, k, maturity, kind), surface.implied_vol(prices, k, maturity, kind))

    def test_heston_smiles_including_the_prior_corners(self):
        bounds = np.asarray(HESTON_TASK["prior_bounds"])
        rows = [bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
                for u in (0.001, 0.5, 0.999, np.array([0.02, 0.9, 0.95, 0.05]), np.array([0.7, 0.1, 0.3, 0.6]))]
        for row in rows:
            with self.subTest(row=row.tolist()):
                assert_array_equal(t_teaching.heston_smile(HESTON_TASK, row), teaching.heston_smile(HESTON_TASK, row))

    def test_rough_design_is_the_rough_task_and_the_pilot_configuration(self):
        design, task = t_tasks.rough_design(), tasks.rough_task()
        self.assertEqual(design["contracts"], task["context"]["contracts"])
        assert_array_equal(design["sigma"], tasks.sigma(task))
        assert_array_equal(design["knots"], tasks.knots(task))
        self.assertEqual(design["blocks"], task["numerics"]["blocks"])
        pilot = json.loads((LAB / "configs" / "tasks" / "calibration_rough_pilot.json").read_text())["task"]
        assert_array_equal(design["sigma"], tasks.sigma(pilot))
        assert_array_equal(design["knots"], tasks.knots(pilot))

    def test_one_rough_smile_is_priced_identically(self):
        for hurst, eta, rho in ((0.1, 2.0, -0.8), (0.3, 0.7, -0.3)):
            with self.subTest(hurst=hurst, eta=eta, rho=rho):
                ours = t_teaching.price_rough_smile(hurst, eta, rho, paths=2048)
                theirs = teaching.price_rough_smile(hurst, eta, rho, paths=2048)
                assert_array_equal(ours["implied"], theirs["implied"])
                self.assertEqual(ours["max_se_over_sigma"], theirs["max_se_over_sigma"])
                self.assertEqual(ours["exponent"], theirs["exponent"])

    def test_network_conditions_and_the_sampler_grid(self):
        quotes = np.asarray(SPY["prices_forward_units"])[None]
        ours = t_inference.conditioning_data(quotes, None, HESTON_TASK)
        theirs = inference.conditioning_data(quotes, None, HESTON_TASK)
        self.assertEqual(set(ours), set(theirs))
        assert_array_equal(ours["observations"], theirs["observations"])
        assert_array_equal(t_inference.sampler_times(), inference.sampler_times())

    def test_teaching_helpers(self):
        prices = surface.bs_price(*surface.contract_arrays(surface.design_grid())[:2], 0.3,
                                  surface.contract_arrays(surface.design_grid())[2])
        assert_array_equal(t_teaching.implied_vol_grid(prices), teaching.implied_vol_grid(prices))
        assert_array_equal(t_teaching.heston_variance_swap_vol(0.04, 0.06, 2.0, [5, 63, 252]),
                           teaching.heston_variance_swap_vol(0.04, 0.06, 2.0, [5, 63, 252]))

    def test_the_long_end_posterior_and_ranges(self):
        lab = lab_module(LAB / "checks" / "long_end" / "long_end.py", "lab_long_end")
        report = json.loads((LAB / "checks" / "long_end" / "long_end.json").read_text())
        flat = np.log(np.asarray(report["xi_flat"]))
        for s_prior, vs_sd in ((0.2, 0.005), (0.4, 0.01)):
            with self.subTest(s_prior=s_prior, vs_sd=vs_sd):
                for ours, theirs in zip(t_long_end.posterior(flat, s_prior, vs_sd), lab.posterior(flat, s_prior, vs_sd)):
                    assert_array_equal(ours, theirs)
                ours, theirs = t_long_end.ranges(report, s_prior, vs_sd), lab.ranges(report, s_prior, vs_sd)
                assert_array_equal(ours[0], theirs[0])
                assert_array_equal(ours[1], theirs[1])
        self.assertEqual(t_long_end.LEAPS, lab.LEAPS)

    def test_the_sector_case(self):
        lab = lab_module(LAB / "checks" / "bs_hierarchy.py", "lab_bs_hierarchy")
        self.assertEqual(t_sector.run(2, 6, 3, 11), lab.run(2, 6, 3, 11))

    def test_the_tutorial_copies_of_lab_files_are_current(self):
        manifest = lab_module(LAB / "studies" / "manifest.py", "lab_manifest")
        self.assertEqual(manifest.stale_exports(), [], "run: pixi run manifest")


if __name__ == "__main__":
    unittest.main()
