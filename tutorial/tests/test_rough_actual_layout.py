"""Pure actual-layout unit/curve/input tests: never run the rough simulator."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from sbi4f_tutorial.rough_options import actual_layout as layout, model, simulator
from sbi4f_tutorial.shared import pricing, full_surface

ASSETS = Path(__file__).resolve().parents[1]/"assets/backbones"
TIMES = [.019266330542871638, .03844441273465246, .08228002917300863, .16447180999492644,
    .2220060565702689, .47120696981227805, .8903850520040588]


def base_task(joint=False):
    return json.loads((ASSETS/("rough-joint100k-validation0" if joint else "rough-known100k-validation0")/"task.json").read_text())


def records():
    forward = np.repeat(np.arange(100., 107.), 9); discount = np.linspace(.999, .97, 63)
    return {"forward": forward.tolist(), "discount": discount.tolist(), "strike": (forward*np.tile(np.linspace(.7, 1.2, 9), 7)).tolist(),
        "maturity": np.repeat(TIMES, 9).tolist(), "option_type": np.tile(["put"]*6+["call"]*3, 7).tolist(),
        "mid": np.linspace(-.1, 3., 63).tolist(), "half_spread": np.linspace(.02, .08, 63).tolist()}


class ActualLayoutTests(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.guards.enter_context(patch.object(pricing, "rough_block", side_effect=AssertionError("no MC in pure tests")))
        self.guards.enter_context(patch.object(simulator, "simulate_rough", side_effect=AssertionError("no MC in pure tests")))
        self.addCleanup(self.guards.close)

    def test_units_and_covariance_use_both_DF_axes(self):
        task = layout.make_task(base_task(), records(), [39, 78, 166, 332, 448, 238, 449]); layout.validate_task(task)
        data = task["design"]; factor = np.asarray(data["price_factor_DF"]); raw = np.asarray(records()["mid"])
        np.testing.assert_allclose(np.asarray(data["observations_forward"])*factor, raw, rtol=4*np.finfo(float).eps, atol=0)
        covariance = np.eye(63)+np.ones((63, 63))*.01
        normalized = layout.covariance_to_forward(task, covariance)
        self.assertAlmostEqual(normalized[0, 1], .01/(factor[0]*factor[1]))
        np.testing.assert_allclose(layout.covariance_to_raw(task, normalized), covariance, rtol=1e-15)
        self.assertLess(data["observations_forward"][0], 0)  # finite negative Gaussian quotes remain valid

    def test_actual_times_and_original_prior_knots_are_unchanged(self):
        base = base_task(); before = deepcopy(base); data = records(); old_data = deepcopy(data)
        task = layout.make_task(base, data, [39, 78, 166, 332, 448, 238, 449])
        self.assertEqual([b["horizon"] for b in task["blocks"]], TIMES)
        self.assertEqual(task["base_task"]["model_context"], base["model_context"])
        self.assertEqual(base, before); self.assertEqual(data, old_data)
        legacy = deepcopy(base); legacy["context"]["contracts"] = task["design"]["contracts"]
        with self.assertRaises(ValueError): full_surface.validate_task(legacy)
        layout.validate_task(task)

    def test_knot_crossing_retains_piecewise_law_and_discrete_control_mean(self):
        curve = layout.piecewise_node_curve([0., .11, .3], [.04, .09], [0., .1, .2, .3])
        np.testing.assert_array_equal(model.curve_values(curve, [0., .1, .2, .3]), [.04, .04, .09, .09])
        discrete = float(model.curve_values(curve, [0., .1, .2]).sum()*.1)
        self.assertAlmostEqual(discrete, .017)
        self.assertNotAlmostEqual(discrete, .04*.11+.09*.19)  # continuous control mean would bias this discretization

    def test_refined_nodes_replay_supplied_curve_without_moving_knots(self):
        task = layout.make_task(base_task(), records(), [39, 78, 166, 332, 448, 238, 449])
        levels = [.04, .05, .03, .07, .04, .06, .02]
        curve = layout.known_curve(task, levels); nodes = layout.simulation_nodes(task)
        self.assertEqual(curve["times"], nodes.tolist())
        self.assertEqual(task["base_task"]["model_context"]["forward_variance_knots"], base_task()["model_context"]["forward_variance_knots"])
        self.assertEqual(len(layout.discrete_curve_audit(task, curve, 4)), 7)

    def test_joint_curve_never_enters_inference_conditions(self):
        known = layout.make_task(base_task(), records(), [39, 78, 166, 332, 448, 238, 449])
        joint = layout.make_task(base_task(True), records(), [622, 1241, 664, 1327, 1791, 950, 1796])
        observations = np.zeros((2, 63)); levels = np.full((2, 7), .04)
        self.assertIn("curve_context", layout.conditioning_inputs(known, observations, supplied_levels=levels))
        self.assertNotIn("curve_context", layout.conditioning_inputs(joint, observations))
        with self.assertRaisesRegex(ValueError, "never enter"): layout.conditioning_inputs(joint, observations, supplied_levels=levels)
        with self.assertRaisesRegex(ValueError, "supplied"): layout.conditioning_inputs(known, observations)
        self.assertEqual(joint["parameter_names"], base_task(True)["parameter_names"])

    def test_hidden_prior_noise_and_rounding_changes_fail_closed(self):
        task = layout.make_task(base_task(), records(), [39, 78, 166, 332, 448, 238, 449])
        for change in ("prior", "time", "noise", "curve"):
            changed = deepcopy(task)
            if change == "prior": changed["base_task"]["prior_bounds"][0][0] = .03
            if change == "time": changed["blocks"][0]["horizon"] = 5/252
            if change == "noise": changed["numerical_gate"]["last_rung_exception"] = True
            if change == "curve": changed["curve_discretization"] = "move knots to expiries"
            with self.assertRaises(ValueError): layout.validate_task(changed)


if __name__ == "__main__": unittest.main()
