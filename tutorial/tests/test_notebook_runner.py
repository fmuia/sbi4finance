"""Intentional idle branches pass; missing dependencies and raised ancestors fail."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import run_notebooks


STOP = {"ename": "ancestor-stopped", "evalue": "This cell wasn't run because an ancestor was stopped with `mo.stop`: ",
        "traceback": [], "type": "error"}


def exported_html(outputs):
    session = {"cells": [{"outputs": values} for values in outputs]}
    return '<html><script>const exportData = {"session": ' + json.dumps(session) + '};</script></html>'


class NotebookRunnerTests(unittest.TestCase):
    def test_intentionally_stopped_descendants_are_not_execution_errors(self):
        cells, errors = run_notebooks.errors_in(exported_html([[STOP], [{"type": "data", "data": "cached content"}], [STOP]]))
        self.assertEqual(cells, 3)
        self.assertEqual(errors, [])

    def test_mixed_stop_and_missing_dependency_keep_real_and_ancestor_errors(self):
        missing = {"ename": "ModuleNotFoundError", "evalue": "No module named 'bayesflow'",
                   "traceback": ["import bayesflow"], "type": "error"}
        ancestor = {"ename": "exception", "evalue": "An ancestor raised an exception (ModuleNotFoundError): ",
                    "traceback": [], "type": "error"}
        _, errors = run_notebooks.errors_in(exported_html([[STOP], [missing], [ancestor]]))
        self.assertEqual(errors, [missing, ancestor])

    def test_only_the_specific_documented_stop_marker_is_ignored(self):
        variants = [
            {**STOP, "ename": "ValueError"},
            {**STOP, "ename": "ancestor-prevented"},
            {**STOP, "ename": "ancestor-raised"},
            {**STOP, "traceback": ["a real exception traceback"]},
            {**STOP, "evalue": "No module named 'marimo'"},
            {**STOP, "evalue": None},
            {"ename": "unknown-error", "type": "error", "evalue": "unknown schema"},
        ]
        _, errors = run_notebooks.errors_in(exported_html([[STOP], *[[value] for value in variants]]))
        self.assertEqual(errors, variants)

    def test_missing_or_invalid_session_cannot_pass_as_an_empty_success(self):
        with self.assertRaisesRegex(ValueError, "embedded session"):
            run_notebooks.errors_in("<html>export incomplete</html>")
        with self.assertRaises(json.JSONDecodeError):
            run_notebooks.errors_in('{"session": incomplete')

    def run_fixture(self, outputs, returncode=0):
        def export(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_text(exported_html(outputs))
            return SimpleNamespace(returncode=returncode, stderr="export diagnostic")

        captured = io.StringIO()
        with mock.patch.object(run_notebooks.subprocess, "run", side_effect=export):
            with contextlib.redirect_stdout(captured), self.assertRaises(SystemExit) as result:
                run_notebooks.main([Path("fixture.py").resolve()])
        return result.exception.code, captured.getvalue()

    def test_cli_passes_idle_controls_without_running_their_optional_work(self):
        status, output = self.run_fixture([[STOP], [STOP]])
        self.assertEqual(status, 0)
        self.assertIn("0 errors", output)

    def test_cli_still_fails_missing_dependency_and_nonzero_export(self):
        missing = {"type": "error", "ename": "ModuleNotFoundError",
                   "evalue": "No module named 'sbi4f_tutorial'", "traceback": []}
        status, output = self.run_fixture([[STOP], [missing]])
        self.assertEqual(status, 1)
        self.assertIn("ModuleNotFoundError", output)
        status, output = self.run_fixture([[STOP]], returncode=1)
        self.assertEqual(status, 1)
        self.assertIn("export exited 1", output)


if __name__ == "__main__":
    unittest.main()
