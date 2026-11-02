"""Cluster launchers select only the new locked environments and fail closed."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_core_cluster

ROOT = Path(__file__).resolve().parents[1]


class CoreLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "commands.txt"
        self.pixi = self.root / "pixi"
        self.pixi.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CORE_TEST_LOG"\n')
        self.pixi.chmod(0o755)
        nvidia = self.root / "nvidia-smi"
        nvidia.write_text("#!/bin/bash\nexit 0\n")
        nvidia.chmod(0o755)
        self.env = {**os.environ, "SLURM_JOB_ID": "test-allocated", "SLURM_SUBMIT_DIR": str(ROOT),
                    "CORE_TEST_LOG": str(self.log), "PATH": f"{self.root}:{os.environ['PATH']}"}

    def launch(self, name, *args):
        return subprocess.run(["bash", str(ROOT / "cluster" / name), str(self.pixi), *args],
                              env=self.env, text=True, capture_output=True)

    def test_cpu_selects_explicit_locked_core_or_reference_environment(self):
        for environment in ("core", "core-reference"):
            result = self.launch("core_cpu.sbatch", environment, "python", "worker.py")
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), [
            "run --locked -e core python worker.py", "run --locked -e core-reference python worker.py"])

    def test_cpu_rejects_archived_environment_before_execution(self):
        result = self.launch("core_cpu.sbatch", "default", "python", "worker.py")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_reusable_wrappers_do_not_choose_an_owners_allocation(self):
        for name in ("core_cpu.sbatch", "core_gpu.sbatch"):
            directives = [line.strip() for line in (ROOT / "cluster" / name).read_text().splitlines()
                          if line.startswith("#SBATCH")]
            self.assertFalse(any("--account" in line or "--partition" in line or "--qos" in line
                                 for line in directives), name)
            self.assertTrue(any("--mem=" in line for line in directives), name)
            self.assertTrue(any("--time=" in line for line in directives), name)

    def test_wrappers_refuse_unallocated_execution_before_calling_the_runtime(self):
        self.env.pop("SLURM_JOB_ID")
        for name, arguments in (("core_cpu.sbatch", ("core", "python", "worker.py")),
                                ("core_gpu.sbatch", ("python", "worker.py"))):
            result = self.launch(name, *arguments)
            self.assertNotEqual(result.returncode, 0, name)
            self.assertIn("Slurm", result.stderr)
            self.assertFalse(self.log.exists())

    def test_gpu_requires_device_check_before_new_environment_command(self):
        result = self.launch("core_gpu.sbatch", "python", "worker.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), [
            "run --locked -e core-gpu core-check --require-gpu", "run --locked -e core-gpu python worker.py"])

    def test_failed_gpu_readiness_prevents_worker_execution(self):
        self.pixi.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CORE_TEST_LOG"\nexit 7\n')
        result = self.launch("core_gpu.sbatch", "python", "worker.py")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.log.read_text().splitlines(), [
            "run --locked -e core-gpu core-check --require-gpu"])

    def test_readiness_refuses_login_execution_without_creating_outputs(self):
        args = argparse.Namespace(out=self.root / "not-created", allow_local=False,
                                  require_gpu=False, seed=1)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Slurm allocation"):
                check_core_cluster.run(args)
        self.assertFalse(args.out.exists())

    def test_lock_exclusion_uses_a_separate_process(self):
        check_core_cluster.check_process_lock(self.root / "shared.lock")


if __name__ == "__main__":
    unittest.main()
