"""Execute exact optional-check cell bodies: idle/error paths publish real None."""

import ast
import copy
import importlib.abc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import simulation


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "trajectory_inference.py"


def control(value):
    return SimpleNamespace(value=value)


def cell_function(result_name, *, producer):
    """Compile the uniquely matching current cell, preserving its executing body."""
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    matches = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        parameters = {argument.arg for argument in node.args.args}
        assigned = {target.id for statement in ast.walk(node) if isinstance(statement, ast.Assign)
                    for target in statement.targets if isinstance(target, ast.Name)}
        if (producer and result_name in assigned) or (not producer and result_name in parameters):
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(f"expected one {'producer' if producer else 'consumer'} for {result_name}, got {len(matches)}")
    function = copy.deepcopy(matches[0])
    function.decorator_list = []
    function.name = "executing_cell"
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["executing_cell"]


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"bayesflow", "keras", "jax"}:
            raise AssertionError("optional predictive cell initialized a neural backend: " + fullname)
        return None


class PredictiveNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_producer = staticmethod(cell_function("prior_predictive_result", producer=True))
        cls.prior_consumer = staticmethod(cell_function("prior_predictive_result", producer=False))
        cls.posterior_producer = staticmethod(cell_function("posterior_predictive_result", producer=True))
        cls.posterior_consumer = staticmethod(cell_function("posterior_predictive_result", producer=False))

    def setUp(self):
        self.task = simulation.teaching_task("rough-known")
        self.history = {"returns": np.arange(1260, dtype=float)[:, None]*1e-6,
                        "truth": np.mean(self.task["prior_bounds"], axis=1), "source": "custom"}
        self.draws = np.repeat(self.history["truth"][None], 16, axis=0)
        self.posterior = {"source": "live", "draws": self.draws, "length": 137}
        self.panel = object()
        self.checks = mock.Mock()
        self.checks.compare_prior.side_effect = lambda task, panel, observed, days: {"length": days}
        self.checks.replicate_check.side_effect = lambda task, observed, draws, days, count: {
            "length": days, "metadata": {"count": count}}
        self.paths = mock.Mock()
        self.paths.generate_panel.return_value = self.panel
        self.visuals = mock.Mock()
        self.visuals.compare.return_value = {"metadata": {"gaussian_available": True}}
        self.visuals.feature_rows.return_value = []
        self.mo = mock.Mock()
        self.mo.md.side_effect = lambda text: text
        self.mo.stop.side_effect = AssertionError("optional cell must publish None instead of mo.stop")
        self.trajectory = mock.Mock()
        self.market = {"log_returns": np.full(1726, -.002),
                       "dates": np.array([f"observed-day-{i:04d}" for i in range(1726)])}
        self.trajectory.load_market_example.return_value = self.market
        self.values = {
            "active_history": self.history, "cache_task": self.task, "diagnostic_panels": {},
            "mo": self.mo, "model_checks": self.checks, "path_checks": self.paths,
            "path_simulation": simulation, "prior_check_button": control(False),
            "prior_target": control("synthetic"), "requested_days": 252, "trajectory": self.trajectory,
            "ordering_shuffle": control(False), "predictive_visuals": self.visuals,
            "prior_predictive_result": None, "posterior_predictive_result": None,
            "posterior": self.posterior, "predictive_count": control(32), "replicate_button": control(False),
        }
        self.finder = NoNeuralImports()
        sys.meta_path.insert(0, self.finder)

    def tearDown(self):
        sys.meta_path.remove(self.finder)

    def run_cell(self, function, **changes):
        values = self.values | changes
        names = function.__code__.co_varnames[:function.__code__.co_argcount]
        return function(**{name: values[name] for name in names})

    def test_prior_idle_always_returns_none_and_consumer_does_no_rendering(self):
        self.assertEqual(self.run_cell(self.prior_producer), (None,))
        self.assertIsNone(self.run_cell(self.prior_consumer))
        self.assertEqual(self.paths.mock_calls, [])
        self.assertEqual(self.checks.mock_calls, [])
        self.assertEqual(self.visuals.mock_calls, [])
        self.assertEqual(self.trajectory.mock_calls, [])
        self.mo.stop.assert_not_called()
        self.mo.ui.tabs.assert_not_called()
        self.mo.ui.table.assert_not_called()
        self.mo.vstack.assert_not_called()

    def test_prior_generation_or_comparison_failure_returns_none_and_a_warning(self):
        for stage, error in ((self.paths.generate_panel, FloatingPointError("nonfinite simulation")),
                             (self.checks.compare_prior, ValueError("invalid checked prefix"))):
            with self.subTest(stage=stage):
                previous = stage.side_effect
                stage.side_effect = error
                self.mo.callout.reset_mock()
                try:
                    self.assertEqual(self.run_cell(self.prior_producer, prior_check_button=control(True)), (None,))
                    self.mo.callout.assert_called_once()
                    self.assertEqual(self.mo.callout.call_args.kwargs["kind"], "warn")
                    self.assertIn(str(error), self.mo.callout.call_args.args[0])
                    self.run_cell(self.prior_consumer)
                    self.visuals.compare.assert_not_called()
                finally:
                    stage.side_effect = previous
                    self.values["diagnostic_panels"] = {}
        self.mo.stop.assert_not_called()

    def test_prior_compares_the_active_custom_history_and_reuses_the_full_panel(self):
        first, = self.run_cell(self.prior_producer, prior_check_button=control(True))
        self.assertEqual(first["length"], 252)
        self.assertIn("Selected synthetic history", first["comparison_observation"])
        args = self.checks.compare_prior.call_args.args
        self.assertIs(args[0], self.task)
        self.assertIs(args[1], self.panel)
        self.assertIs(args[2], self.history["returns"])
        self.assertEqual(args[3], 252)
        self.run_cell(self.prior_producer, prior_check_button=control(True), requested_days=137)
        self.paths.generate_panel.assert_called_once_with(self.task, count=200, seed=20261111)
        self.assertEqual(self.checks.compare_prior.call_args.args[-1], 137)
        # Rebuilding an idle action after a changed context clears the result.
        self.assertEqual(self.run_cell(self.prior_producer, requested_days=137), (None,))
        self.assertEqual(self.checks.compare_prior.call_count, 2)

    def test_prior_market_branch_uses_only_the_requested_dated_prefix(self):
        result, = self.run_cell(self.prior_producer, prior_check_button=control(True), prior_target=control("market"))
        args = self.checks.compare_prior.call_args.args
        np.testing.assert_array_equal(args[2], self.market["log_returns"][:252, None])
        self.assertEqual(args[2].shape, (252, 1))
        self.assertIn(self.market["dates"][251], result["comparison_observation"])
        self.trajectory.load_market_example.assert_called_once_with("voo-2020-20261113")

    def test_posterior_idle_or_absent_posterior_never_generates_replicas(self):
        for posterior, clicked in ((self.posterior, False), (None, False), (None, True)):
            with self.subTest(posterior_available=posterior is not None, clicked=clicked):
                self.assertEqual(self.run_cell(self.posterior_producer, posterior=posterior,
                                               replicate_button=control(clicked)), (None,))
                self.assertIsNone(self.run_cell(self.posterior_consumer, posterior=posterior))
        self.assertEqual(self.checks.mock_calls, [])
        self.assertEqual(self.visuals.mock_calls, [])
        self.mo.stop.assert_not_called()
        self.mo.ui.tabs.assert_not_called()
        self.mo.vstack.assert_not_called()

    def test_posterior_replication_failure_returns_none_with_notice_without_rendering(self):
        for error in (ValueError("wrong posterior axes"), RuntimeError("simulation failed"),
                      FloatingPointError("nonfinite replica")):
            with self.subTest(error=error):
                self.checks.replicate_check.side_effect = error
                self.mo.callout.reset_mock()
                self.assertEqual(self.run_cell(self.posterior_producer, replicate_button=control(True)), (None,))
                self.mo.callout.assert_called_once()
                self.assertEqual(self.mo.callout.call_args.kwargs["kind"], "warn")
                self.assertIn(str(error), self.mo.callout.call_args.args[0])
                self.run_cell(self.posterior_consumer)
                self.visuals.compare.assert_not_called()
        self.mo.stop.assert_not_called()

    def test_posterior_replication_uses_this_history_same_paired_rows_and_actual_conditioning_length(self):
        result, = self.run_cell(self.posterior_producer, replicate_button=control(True), requested_days=252)
        args = self.checks.replicate_check.call_args.args
        self.assertIs(args[0], self.task)
        self.assertIs(args[1], self.history["returns"])
        self.assertIs(args[2], self.draws)
        self.assertEqual(args[3], 137)  # Follow the posterior, not the requested 252-day display.
        self.assertEqual(self.checks.replicate_check.call_args.kwargs, {"count": 32})
        self.assertEqual(result["metadata"]["count"], 32)
        self.assertEqual(self.run_cell(self.posterior_producer), (None,))
        self.assertEqual(self.checks.replicate_check.call_count, 1)

    def test_nonempty_consumers_render_only_the_retained_result(self):
        prior = {"length": 252, "comparison_observation": "Chosen synthetic history"}
        self.run_cell(self.prior_consumer, prior_predictive_result=prior)
        self.visuals.compare.assert_called_once_with(prior)
        self.checks.plot_check.assert_called_once_with(prior)
        self.visuals.reset_mock()
        self.checks.reset_mock()
        posterior = {"length": 137, "metadata": {"count": 32}}
        self.run_cell(self.posterior_consumer, posterior_predictive_result=posterior)
        self.visuals.compare.assert_called_once_with(posterior)
        self.checks.plot_check.assert_called_once_with(posterior)
        self.mo.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
