import os
from pathlib import Path
import subprocess
import sys
import unittest


class ContinuationDensityCliTests(unittest.TestCase):
    def test_direct_cli_help_without_pythonpath(self):
        lab = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, str(lab / "scripts/score_heston_campaign_continuation.py"), "--help"],
                                cwd=lab, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--baseline-score", result.stdout)
        self.assertIn("--cell", result.stdout)


if __name__ == "__main__":
    unittest.main()
