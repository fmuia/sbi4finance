"""Check dependency/resource accounting and the source working-directory contract."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


LAB = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("path_launcher", LAB / "cluster/submit_path_pilot.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class PathLauncherTests(unittest.TestCase):
    def test_fixed_budget_and_dependency_graph(self):
        jobs = launcher.jobs(Path("/example/run"), "/cpu/python", "/gpu/python")
        seen = set()
        for job in jobs:
            self.assertTrue(set(job["dependencies"]) <= seen)
            seen.add(job["name"])
            if job["name"].startswith("bank-"):
                self.assertEqual(job["command"][job["command"].index("--train") + 1], "100000")
        self.assertAlmostEqual(sum(j["cpus"] * j["minutes"] / 60 for j in jobs), 19 + 5 / 6)
        self.assertAlmostEqual(sum(j["gpus"] * j["minutes"] / 60 for j in jobs), 2.75)
        self.assertEqual(jobs[-1]["dependency_condition"], "afterany")

    def test_submission_sets_source_directory_and_gpu_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-run"
            arguments = ["submit_path_pilot.py", "--run-root", str(root), "--cpu-python", "/cpu/python",
                "--gpu-python", "/gpu/python", "--account", "condo", "--cpu-partition", "batch",
                "--gpu-account", "default", "--gpu-partition", "gpu", "--gpu-qos", "norm-gpu",
                "--smoke-gpu-partition", "gpu-debug", "--smoke-gpu-qos", "gpu-debug",
                "--runtime-lock-sha256", "runtime-sha", "--source-commit", "source-revision"]
            with patch("sys.argv", arguments), patch.object(launcher.subprocess, "run") as run:
                run.side_effect = [types.SimpleNamespace(stdout=f"{9000 + i}\n") for i in range(17)]
                with contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["cwd"], LAB)
                self.assertIn(f"--chdir={LAB}", call.args[0])
                self.assertTrue(any(f"SBI4FINANCE_SOURCE_LAB={LAB}" in x for x in call.args[0]))
            gpu_smoke = run.call_args_list[1].args[0]
            self.assertIn("--partition=gpu-debug", gpu_smoke)
            self.assertIn("--account=default", gpu_smoke)
            self.assertIn("--qos=gpu-debug", gpu_smoke)
            fit = next(c.args[0] for c in run.call_args_list if "--job-name=path-fit-heston" in c.args[0])
            self.assertIn("--partition=gpu", fit)
            self.assertIn("--qos=norm-gpu", fit)
            receipt = json.loads((root / "submission-receipt.json").read_text())
            self.assertTrue(receipt["submission_complete"])
            self.assertEqual(len(receipt["jobs"]), 17)


if __name__ == "__main__":
    unittest.main()
