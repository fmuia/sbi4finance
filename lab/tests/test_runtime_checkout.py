"""A selected producer checkout must not borrow another worktree's runtime."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAB = Path(__file__).resolve().parents[1]
TUTORIAL_SOURCE = LAB.parent / "tutorial" / "src"


class RuntimeCheckoutTests(unittest.TestCase):
    def run_import(self, runtime_source):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(LAB / "src"), str(runtime_source)]))
        return subprocess.run(
            [sys.executable, "-c", "import sbi4finance; from sbi4finance.calibration import tasks; print(tasks.__name__)"],
            env=env, capture_output=True, text=True, timeout=30,
        )

    def test_same_checkout_runtime_is_accepted(self):
        run = self.run_import(TUTORIAL_SOURCE)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("sbi4finance.calibration.tasks", run.stdout)

    def test_other_checkout_runtime_is_rejected_before_numerical_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "sbi4f_tutorial").mkdir()
            (source / "sbi4f_tutorial" / "__init__.py").write_text("")
            run = self.run_import(source)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("same complete checkout", run.stderr)
        self.assertIn("pixi install --locked -e core", run.stderr)


if __name__ == "__main__":
    unittest.main()
