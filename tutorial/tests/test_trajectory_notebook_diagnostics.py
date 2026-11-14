"""Exercise actual optional-diagnostic cells without fitting or sampling a model."""

import builtins
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import marimo as mo
import numpy as np

from sbi4f_tutorial.trajectories import lesson, simulation


def control(value):
    return SimpleNamespace(value=value)


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"bayesflow", "keras", "jax"}:
            raise AssertionError("optional cell initialized a neural backend: " + fullname)
        return None


class NotebookDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "trajectory_inference.py"
        spec = importlib.util.spec_from_file_location("trajectory_diagnostic_cell_tests", path)
        cls.notebook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.notebook)

    def setUp(self):
        self.task = simulation.teaching_task("rough-known")
        theta = np.mean(self.task["prior_bounds"], axis=1)
        self.panel = {"parameters": np.repeat(theta[None], 200, axis=0),
                      "observations": np.zeros((200, 1260, 1)),
                      "trajectory_id": np.array([f"diagnostic-{i}" for i in range(200)])}
        self.draws = np.broadcast_to(theta, (200, 200, 4)).copy()
        self.checks = mock.Mock()
        self.checks.generate_panel.return_value = self.panel
        self.checks.coverage_table.return_value = []
        self.checks.rank_readout.return_value = []
        self.checks.evaluate.return_value = self.draws
        self.checks.plot_diagnostics.return_value = (mo.md("Recovery"), mo.md("Ranks"))
        self.helper = mock.Mock(return_value=(self.draws, mo.md("Recovery"), mo.md("Ranks")))
        self.plt = mock.Mock()
        self.defaults = {
            "mo": mo, "np": np, "plt": self.plt, "trajectory": lesson,
            "path_checks": self.checks, "path_simulation": simulation,
            "show_independent_tools": control(False),
            "cache_task": self.task, "tutorial_task": self.task, "diagnostic_task": self.task,
            "diagnostic_source": control("tutorial"), "diagnostic_days": control(1260),
            "run_diagnostics": control(False), "diagnostic_panels": {},
            "tutorial_workflow": None, "pretrained_workflow": None,
            "check_independent_histories": self.helper, "diagnostic_result": None,
            "recovery_parameter": None, "recovery_picker": None, "recovery_points": None,
            "diagnostic_case": None, "show_diagnostic_pairs": None,
            "parameter_labels": {name: name for name in self.task["parameter_names"]},
        }
        self.finder = NoNeuralImports()
        sys.meta_path.insert(0, self.finder)

    def tearDown(self):
        sys.meta_path.remove(self.finder)

    def run_cell(self, name, **changes):
        cell = getattr(self.notebook, name)
        values = self.defaults | changes
        missing = cell.refs - values.keys() - vars(builtins).keys()
        self.assertEqual(missing, set(), "Every input is explicit; no notebook ancestors may run")
        inputs = {key: values[key] for key in cell.refs if key in values}
        _, definitions = cell.run(**inputs)
        self.assertEqual(set(definitions), cell.defs,
                         "Idle and failure paths must publish every definition")
        return definitions

    def test_controls_and_callable_exist_when_optional_tools_are_closed_or_open(self):
        for visible in (False, True):
            with self.subTest(visible=visible):
                self.defaults["show_independent_tools"] = control(visible)
                source = self.run_cell("diagnostic_network_controls")["diagnostic_source"]
                self.assertEqual(source.value, "tutorial")
                history = self.run_cell("diagnostic_history_controls", diagnostic_source=source)
                self.assertEqual(history["diagnostic_task"], self.task)
                self.assertEqual(history["diagnostic_days"].value, 1260)
                button = self.run_cell("diagnostic_run_control")["run_diagnostics"]
                self.assertFalse(button.value)
                with mock.patch.object(mo, "show_code", return_value=mo.md("Visible code")):
                    helper = self.run_cell("independent_check_code")["check_independent_histories"]
                self.assertTrue(callable(helper))
        self.assertEqual(self.checks.mock_calls, [])
        self.helper.assert_not_called()

    def test_idle_closed_and_open_paths_return_none_even_if_a_network_is_ready(self):
        for visible, clicked in ((False, False), (False, True), (True, False)):
            with self.subTest(visible=visible, clicked=clicked):
                result = self.run_cell("run_independent_check", show_independent_tools=control(visible),
                                       run_diagnostics=control(clicked), tutorial_workflow=object())
                self.assertIsNone(result["diagnostic_result"])
        self.checks.generate_panel.assert_not_called()
        self.helper.assert_not_called()

    def test_clicked_without_the_selected_workflow_publishes_none_without_generation(self):
        for source in ("tutorial", "pretrained"):
            with self.subTest(source=source):
                result = self.run_cell("run_independent_check", show_independent_tools=control(True),
                                       run_diagnostics=control(True), diagnostic_source=control(source))
                self.assertIsNone(result["diagnostic_result"])
        self.checks.generate_panel.assert_not_called()
        self.helper.assert_not_called()

    def test_success_keeps_complete_panel_paired_samples_and_actual_prefix(self):
        workflow = object()
        result = self.run_cell("run_independent_check", show_independent_tools=control(True),
                               run_diagnostics=control(True), diagnostic_source=control("pretrained"),
                               diagnostic_days=control(137), pretrained_workflow=workflow)["diagnostic_result"]
        self.assertIs(result["panel"], self.panel)
        self.assertIs(result["draws"], self.draws)
        self.assertIs(result["task"], self.task)
        self.assertEqual(result["length"], 137)
        self.assertEqual(result["source"], "pretrained")
        self.checks.generate_panel.assert_called_once_with(self.task, count=200, seed=20261111)
        self.helper.assert_called_once_with(workflow, self.task, self.panel, 137)
        self.checks.coverage_table.assert_called_once_with(self.task, self.panel, self.draws)
        self.checks.rank_readout.assert_called_once_with(self.task, self.panel, self.draws, z=.5)

    def test_changed_prefix_or_source_rebuilds_idle_button_and_clears_success(self):
        workflow = object()
        self.defaults.update(show_independent_tools=control(True), tutorial_workflow=workflow)
        first = self.run_cell("run_independent_check", run_diagnostics=control(True))["diagnostic_result"]
        self.assertIsNotNone(first)
        old = self.run_cell("diagnostic_run_control")["run_diagnostics"]
        contexts = ((137, "tutorial", self.task, workflow),
                    (252, "pretrained", self.task, workflow),
                    (1260, "pretrained", simulation.teaching_task("heston"), object()))
        for days, source, task, changed_workflow in contexts:
            with self.subTest(days=days, source=source, model=task["model"]):
                new = self.run_cell("diagnostic_run_control", diagnostic_days=control(days),
                                    diagnostic_source=control(source), diagnostic_task=task,
                                    tutorial_workflow=changed_workflow)["run_diagnostics"]
                self.assertIsNot(new, old)
                self.assertFalse(new.value)
                result = self.run_cell("run_independent_check", run_diagnostics=new,
                                       diagnostic_days=control(days), diagnostic_source=control(source),
                                       diagnostic_task=task, tutorial_workflow=changed_workflow)
                self.assertIsNone(result["diagnostic_result"])
        # Changing a prefix reuses existing histories only after another click.
        repeated = self.run_cell("run_independent_check", run_diagnostics=control(True),
                                 diagnostic_days=control(137))["diagnostic_result"]
        self.assertIs(repeated["panel"], self.panel)
        self.assertEqual(repeated["length"], 137)
        self.assertEqual(self.checks.generate_panel.call_count, 1)
        self.assertEqual(self.helper.call_count, 2)

    def test_generation_sampling_and_summary_failures_never_publish_partial_results(self):
        stages = ((self.checks.generate_panel, ValueError("simulation failed")),
                  (self.helper, RuntimeError("sampling failed")),
                  (self.checks.coverage_table, FloatingPointError("coverage failed")),
                  (self.checks.rank_readout, ValueError("rank calculation failed")))
        for stage, error in stages:
            with self.subTest(stage=stage):
                self.defaults["diagnostic_panels"] = {}
                stage.side_effect = error
                try:
                    result = self.run_cell("run_independent_check", show_independent_tools=control(True),
                                           run_diagnostics=control(True), tutorial_workflow=object())
                    self.assertIsNone(result["diagnostic_result"])
                finally:
                    stage.side_effect = None

    def test_every_downstream_cell_handles_absent_result_or_incomplete_controls(self):
        self.assertIsNone(self.run_cell("diagnostic_parameter_control")["recovery_parameter"])
        plots = self.run_cell("diagnostic_recovery_plot")
        self.assertIsNone(plots["recovery_picker"])
        self.assertIsNone(plots["recovery_points"])
        cases = self.run_cell("diagnostic_case_controls")
        self.assertIsNone(cases["diagnostic_case"])
        self.assertIsNone(cases["show_diagnostic_pairs"])
        self.run_cell("diagnostic_case_view")
        # A previous result must not make partially recreated UI controls usable.
        result = {"task": self.task, "panel": self.panel, "draws": self.draws,
                  "source": "tutorial", "length": 1260}
        self.run_cell("diagnostic_recovery_plot", diagnostic_result=result)
        self.run_cell("diagnostic_case_controls", diagnostic_result=result)
        self.run_cell("diagnostic_case_view", diagnostic_result=result)
        self.assertEqual(self.checks.mock_calls, [])
        self.assertEqual(self.plt.mock_calls, [])

    def test_visible_check_code_remains_an_explicit_callable_using_selected_days(self):
        with mock.patch.object(mo, "show_code", return_value=mo.md("Visible code")):
            helper = self.run_cell("independent_check_code", show_independent_tools=control(True))[
                "check_independent_histories"]
        self.assertEqual(self.checks.mock_calls, [])
        workflow = object()
        draws, _, _ = helper(workflow, self.task, self.panel, 137)
        self.assertIs(draws, self.draws)
        self.checks.evaluate.assert_called_once_with(workflow, self.task, self.panel, 137, draws=200, seed=20261111)
        self.checks.plot_diagnostics.assert_called_once_with(self.task, self.panel, self.draws)


if __name__ == "__main__":
    unittest.main()
