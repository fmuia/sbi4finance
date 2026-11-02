"""Tutorial-owned kernels remain shared and enter producer identities."""

from argparse import Namespace
import hashlib
import json
import subprocess
import sys
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sbi4finance.calibration import bank, bank_collection, inference, surface, tasks
from sbi4finance.path_inference.bank import teaching_task
from sbi4finance.runtime_sources import (option_runtime_hashes, runtime_hashes, trajectory_runtime_hashes,
                                        option_inference_runtime_hashes, path_inference_runtime_hashes)
from scripts import train_full_surface
from sbi4f_tutorial.shared import surface as canonical_surface


LAB = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_boundary", LAB / "studies/runtime_boundary.py")
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)


class RuntimeReuseTests(unittest.TestCase):
    def test_replacing_a_lab_alias_fails_the_boundary_check(self):
        self.assertIs(surface.bs_price, canonical_surface.bs_price)
        with patch.object(surface, "bs_price", lambda *args: None):
            failures = boundary.differences()
        self.assertEqual(len(failures), 1)
        self.assertIn("sbi4finance.calibration.surface.bs_price", failures[0])
        self.assertEqual(boundary.differences(), [])

    def test_retired_copy_command_cannot_overwrite_the_runtime(self):
        source = Path(canonical_surface.__file__)
        before = source.read_bytes()
        process = subprocess.run([sys.executable, str(LAB / "studies/copy_to_tutorial.py"),
                                  "--add", "surface.bs_price"], cwd=LAB,
                                 text=True, capture_output=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertIn("--add is retired", process.stderr)
        self.assertEqual(source.read_bytes(), before)

    def test_legacy_inference_rejects_joint_and_malformed_tasks(self):
        with self.assertRaisesRegex(ValueError, "joint-curve tasks require"):
            inference._checked_task(tasks.joint_rough_task())
        malformed = tasks.heston_task()
        malformed["measure"] = "P"
        with self.assertRaisesRegex(ValueError, "unsupported calibration task"):
            inference._checked_task(malformed)


class SharedSourceIdentityTests(unittest.TestCase):
    def test_source_identity_uses_module_bytes_with_a_stable_repository_key(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kernel.py"
            source.write_text("def kernel():\n    return 1\n")
            with patch("sbi4finance.runtime_sources.find_spec", return_value=SimpleNamespace(origin=str(source))):
                first = runtime_hashes(["sbi4f_tutorial.shared.kernel"])
                self.assertEqual(first, {"tutorial/src/sbi4f_tutorial/shared/kernel.py":
                                         hashlib.sha256(source.read_bytes()).hexdigest()})
                source.write_text("def kernel():\n    return 2\n")
                self.assertNotEqual(first, runtime_hashes(["sbi4f_tutorial.shared.kernel"]))

    def test_option_producers_record_shared_numerics_but_no_lesson_or_asset_files(self):
        for task in (tasks.heston_task(), tasks.rough_task(), tasks.joint_rough_task()):
            sources = option_runtime_hashes(task)
            for basename in ("bank_helpers", "gbm", "surface", "pricing", "tasks"):
                self.assertIn(f"tutorial/src/sbi4f_tutorial/shared/{basename}.py", sources)
            prefix = "heston_options" if task["model"] == "hkde" else "rough_options"
            self.assertIn(f"tutorial/src/sbi4f_tutorial/{prefix}/model.py", sources)
            self.assertIn(f"tutorial/src/sbi4f_tutorial/{prefix}/simulator.py", sources)
            self.assertFalse(any("lesson" in key or "assets" in key or "plots" in key for key in sources))
            self.assertLessEqual(sources.keys(), bank.source_hashes(task).keys())

    def test_trajectory_producers_record_selected_physical_simulator(self):
        for model in ("heston", "rough-known", "rough-joint"):
            sources = trajectory_runtime_hashes(teaching_task(model))
            self.assertIn("tutorial/src/sbi4f_tutorial/trajectories/simulation.py", sources)
            prefix = "heston_options" if model == "heston" else "rough_options"
            self.assertIn(f"tutorial/src/sbi4f_tutorial/{prefix}/simulator.py", sources)
            self.assertFalse(any("lesson" in key or "assets" in key or "plots" in key for key in sources))

    def test_training_contract_binds_adapter_and_sampler_runtime_sources(self):
        task = tasks.heston_task()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"task": task}))
            args = Namespace(bank=root, family="spline", encoder="vector", widths="8,8",
                             summary_dim=8, flow_steps=None, mean_steps=4, epochs=1,
                             batch_size=4, chunk_epochs=1, seed=0)
            manifest = {"configuration": {"task": task}, "task_fingerprint": "frozen-task"}
            first = train_full_surface.run_contract(args, manifest)
            expected = option_inference_runtime_hashes(task)
            self.assertEqual(first["shared_runtime_source_sha256"], expected)
            self.assertIn("tutorial/src/sbi4f_tutorial/shared/full_surface.py", expected)
            changed = {**expected, "tutorial/src/sbi4f_tutorial/shared/full_surface.py": "0" * 64}
            with patch.object(train_full_surface, "option_inference_runtime_hashes", return_value=changed):
                second = train_full_surface.run_contract(args, manifest)
            self.assertNotEqual(first, second)
        self.assertIn("tutorial/src/sbi4f_tutorial/trajectories/inference.py", path_inference_runtime_hashes())

    def test_changed_shared_source_keeps_old_collection_readable_but_blocks_extension(self):
        task = tasks.heston_task()
        config = {"task": task, "root_seed": 7, "splits": {"train": 1, "validation": 1, "test": 1}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection"
            record = bank_collection.create_collection(path, config)
            changed = dict(record["source_sha256"])
            changed["tutorial/src/sbi4f_tutorial/shared/pricing.py"] = "0" * 64
            with patch.object(bank, "source_hashes", return_value=changed):
                _, existing = bank_collection._read_collection(path)
                self.assertEqual(existing["collection_id"], record["collection_id"])
                with self.assertRaisesRegex(ValueError, "generator sources or runtime differ"):
                    bank_collection.plan_extension(path, 2, 1)


if __name__ == "__main__":
    unittest.main()
