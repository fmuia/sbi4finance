"""Training-run safety without importing BayesFlow or executing numerical experiments."""

import argparse
import ast
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"
SCRIPT = STUDIES / "train_calibration.py"


def script_functions(path, names, namespace):
    """Compile the real stdlib preflight functions, leaving optional model imports unexecuted."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("requested script functions not found")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class CalibrationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bank = self.root / "bank"
        self.bank.mkdir()
        self.metadata = {"task_fingerprint": "task-one", "collection_id": "collection-one",
                         "configuration": {"splits": {"train": 16, "validation": 4, "test": 1}}}
        (self.bank / "manifest.json").write_text(json.dumps(self.metadata))
        self.layout = self.root / "layout.json"
        self.layout.write_text('{"contracts": [0, 2]}')
        self.source = self.root / "source.py"
        self.source.write_text("# source identity\n")
        module = SimpleNamespace(__file__=str(self.source))
        self.versions = {name: "test-version" for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}
        self.helpers = script_functions(SCRIPT, {"sha256", "run_contract", "check_run_directory",
                                                  "reserve_run_directory", "commit_resume_state", "train_in_chunks",
                                                  "runtime_versions", "training_lock", "main", "run_training"},
                                       {"hashlib": hashlib, "json": json, "math": math, "Path": Path,
                                        "shutil": shutil, "tempfile": tempfile,
                                        "argparse": argparse, "sys": sys, "time": time,
                                        "contextmanager": contextmanager, "fcntl": fcntl, "os": os,
                                        "package_metadata": SimpleNamespace(version=lambda name: self.versions[name]),
                                        "platform": SimpleNamespace(python_version=lambda: "3.12.test"),
                                        "__file__": str(SCRIPT), "inference": module, "thin": module, "bank": module})
        self.args = argparse.Namespace(bank=self.bank, thin_layout=self.layout, epochs=2, batch_size=8,
                                       augmentations=4, chunk_epochs=1, seed=0, widths="16,16",
                                       time_alpha=-0.5, fresh_noise=True)
        self.contract = self.helpers["run_contract"](self.args, self.metadata)
        self.output = self.root / "run"

    def tree(self):
        return {p.relative_to(self.output).as_posix(): p.read_bytes()
                for p in self.output.rglob("*") if p.is_file()} if self.output.exists() else None

    def complete_resume(self):
        self.helpers["reserve_run_directory"](self.output, self.contract, True)
        (self.output / "model.keras").write_bytes(b"best checkpoint")
        state = {"epochs_done": 1, "best_epoch": 1, "best_val_loss": 0.5,
                 "history": {"loss": [0.6], "val_loss": [0.5]}, "resumed_at_epochs": [],
                 "planned": {"epochs": 2, "batch_size": 8, "rows": 16}, "noise_rng": {},
                 "run_contract": self.contract}
        return self.helpers["commit_resume_state"](self.output, state,
                                                     lambda path: path.write_bytes(b"last checkpoint"))

    def test_preflight_does_not_create_output_and_reservation_is_exclusive(self):
        self.assertIsNone(self.helpers["check_run_directory"](self.output, self.contract, True))
        self.assertFalse(self.output.exists())
        self.helpers["reserve_run_directory"](self.output, self.contract, True)
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "use a new --out"):
            self.helpers["reserve_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_existing_checkpoint_directory_is_preserved_without_resume(self):
        self.output.mkdir()
        (self.output / "model.keras").write_bytes(b"accepted checkpoint")
        (self.output / "notes.txt").write_text("keep this")
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "use a new --out"):
            self.helpers["reserve_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_cli_preflight_refuses_protected_output_before_loading_arrays(self):
        self.output.mkdir()
        (self.output / "model.keras").write_bytes(b"preserve accepted checkpoint")
        before = self.tree()
        load = Mock(side_effect=AssertionError("array loader must not run"))
        self.helpers["bank"].load_bank = load
        command = [str(SCRIPT), "--bank", str(self.bank), "--out", str(self.output), "--chunk-epochs", "1"]
        with patch.object(sys, "argv", command), self.assertRaisesRegex(ValueError, "nonempty training output"):
            self.helpers["main"]()
        load.assert_not_called()
        self.assertEqual(self.tree(), before)

    def test_matching_resume_is_read_only(self):
        expected = self.complete_resume()
        before = self.tree()
        self.assertEqual(self.helpers["check_run_directory"](self.output, self.contract, True), expected)
        self.helpers["reserve_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_nonchunked_run_does_not_reuse_existing_output(self):
        self.complete_resume()
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "use a new --out"):
            self.helpers["check_run_directory"](self.output, self.contract, False)
        self.assertEqual(self.tree(), before)

    def test_changed_data_or_settings_cannot_resume_same_size_run(self):
        self.complete_resume()
        before = self.tree()
        changes = [("seed", 1), ("widths", "32,32"), ("time_alpha", 0.5), ("fresh_noise", False),
                   ("augmentations", 8), ("epochs", 3), ("batch_size", 4), ("chunk_epochs", 2)]
        for name, value in changes:
            with self.subTest(setting=name):
                changed_args = copy.copy(self.args)
                setattr(changed_args, name, value)
                changed = self.helpers["run_contract"](changed_args, self.metadata)
                with self.assertRaisesRegex(ValueError, "use a new --out"):
                    self.helpers["check_run_directory"](self.output, changed, True)
                self.assertEqual(self.tree(), before)
        metadata = {**self.metadata, "collection_id": "different bank, same size"}
        (self.bank / "manifest.json").write_text(json.dumps(metadata))
        changed = self.helpers["run_contract"](self.args, metadata)
        with self.assertRaisesRegex(ValueError, "use a new --out"):
            self.helpers["check_run_directory"](self.output, changed, True)
        self.assertEqual(self.tree(), before)

    def test_changed_layout_or_source_cannot_resume(self):
        self.complete_resume()
        before = self.tree()
        for path in (self.layout, self.source):
            with self.subTest(changed=str(path)):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                changed = self.helpers["run_contract"](self.args, self.metadata)
                with self.assertRaisesRegex(ValueError, "use a new --out"):
                    self.helpers["check_run_directory"](self.output, changed, True)
                self.assertEqual(self.tree(), before)
                path.write_bytes(original)

    def test_checkpoint_must_match_saved_resume_digest(self):
        state = self.complete_resume()
        (self.output / state["checkpoint_files"]["last"]).write_bytes(b"corrupt committed checkpoint")
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "checkpoint and resume state differ"):
            self.helpers["check_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_uncommitted_working_files_do_not_invalidate_committed_resume(self):
        state = self.complete_resume()
        (self.output / "model.keras").write_bytes(b"uncommitted best from a later epoch")
        (self.output / "last.keras").write_bytes(b"uncommitted working copy")
        self.assertEqual(self.helpers["check_run_directory"](self.output, self.contract, True), state)
        best = self.output / state["checkpoint_files"]["best"]
        self.assertEqual(best.read_bytes(), b"best checkpoint")

    def test_recovery_restores_committed_best_before_training(self):
        class StopBeforeTraining(Exception):
            pass

        state = self.complete_resume()
        (self.output / "model.keras").write_bytes(b"uncommitted later best")
        optimizer = SimpleNamespace(iterations=SimpleNamespace(numpy=lambda: 2))
        approximator = SimpleNamespace(optimizer=optimizer)
        keras = SimpleNamespace(callbacks=SimpleNamespace(Callback=object),
                                saving=SimpleNamespace(load_model=lambda path: approximator))

        def fit(**kwargs):
            self.assertEqual((self.output / "model.keras").read_bytes(), b"best checkpoint")
            raise StopBeforeTraining

        workflow = SimpleNamespace(fit_offline=fit)
        self.helpers["np"] = SimpleNamespace(ceil=math.ceil, inf=math.inf)
        self.helpers["inference"].offline_optimizer = lambda steps: "unused optimizer"
        args = copy.copy(self.args)
        args.out = self.output
        with patch.dict(sys.modules, {"keras": keras}), self.assertRaises(StopBeforeTraining):
            self.helpers["train_in_chunks"](args, {}, {"observations": [None] * 16}, {}, None, (16, 16),
                                              self.output / "resume.json", lambda *a, **k: workflow, self.contract)
        self.assertEqual(self.helpers["check_run_directory"](self.output, self.contract, True), state)

    def test_incomplete_next_chunk_keeps_previous_resume_pointer_and_checkpoints(self):
        state = self.complete_resume()
        original_pointer = (self.output / "resume.json").read_bytes()
        original_checkpoints = {name: (self.output / name).read_bytes() for name in state["checkpoint_sha256"]}
        next_state = {**state, "epochs_done": 2}

        def incomplete_save(path):
            path.write_bytes(b"partial checkpoint")
            raise RuntimeError("interrupted save")

        with self.assertRaisesRegex(RuntimeError, "interrupted save"):
            self.helpers["commit_resume_state"](self.output, next_state, incomplete_save)
        self.assertEqual((self.output / "resume.json").read_bytes(), original_pointer)
        self.assertEqual({name: (self.output / name).read_bytes() for name in original_checkpoints}, original_checkpoints)
        self.assertEqual(self.helpers["check_run_directory"](self.output, self.contract, True), state)

    def test_missing_committed_checkpoint_rejects_resume(self):
        state = self.complete_resume()
        (self.output / state["checkpoint_files"]["best"]).unlink()
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "checkpoint and resume state differ"):
            self.helpers["check_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_invalid_resume_history_is_preserved_and_rejected(self):
        state = self.complete_resume()
        state["history"]["val_loss"] = []
        (self.output / "resume.json").write_text(json.dumps(state))
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "invalid training resume"):
            self.helpers["check_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_runtime_upgrade_cannot_resume_historical_training(self):
        self.complete_resume()
        before = self.tree()
        self.versions["bayesflow"] = "2.1.0"
        changed = self.helpers["run_contract"](self.args, self.metadata)
        with self.assertRaisesRegex(ValueError, "use a new --out"):
            self.helpers["check_run_directory"](self.output, changed, True)
        self.assertEqual(self.tree(), before)

    def test_only_one_training_writer_can_hold_an_output(self):
        with self.helpers["training_lock"](self.output):
            self.assertFalse(self.output.exists())
            with self.assertRaisesRegex(ValueError, "another trainer owns"):
                with self.helpers["training_lock"](self.output):
                    self.fail("concurrent training lock was accepted")
        with self.helpers["training_lock"](self.output):
            self.assertFalse(self.output.exists())

    def test_finite_but_wrong_best_loss_is_rejected(self):
        state = self.complete_resume()
        state["best_val_loss"] = 0.4
        (self.output / "resume.json").write_text(json.dumps(state))
        before = self.tree()
        with self.assertRaisesRegex(ValueError, "invalid training resume best"):
            self.helpers["check_run_directory"](self.output, self.contract, True)
        self.assertEqual(self.tree(), before)

    def test_best_epoch_must_be_the_first_minimum(self):
        state = self.complete_resume()
        state.update(epochs_done=2, best_epoch=2, history={"loss": [0.6, 0.6], "val_loss": [0.5, 0.5]})
        (self.output / "resume.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "invalid training resume best"):
            self.helpers["check_run_directory"](self.output, self.contract, True)


class EvaluationSplitTests(unittest.TestCase):
    def test_validation_selection_never_requests_test_arrays(self):
        class StopBeforeModel(Exception):
            pass

        def stop(*args):
            raise StopBeforeModel

        for script, network_loader in (("evaluate_calibration.py", "load_network"),
                                       ("evaluate_thin.py", "load_thin_network")):
            for split in ("validation", "test"):
                with self.subTest(script=script, split=split):
                    calls = []

                    def load(path, *, splits, roles):
                        calls.append((splits, roles))
                        return {}, {}

                    bank = SimpleNamespace(load_bank=load, TRAINING_ROLES=("needed",))
                    ns = script_functions(STUDIES / script, {"main"},
                                          {"argparse": argparse, "Path": Path, "bank": bank,
                                           network_loader: stop})
                    command = [script, "--bank", "unused", "--checkpoint", "unused", "--out", "unused",
                               "--split", split]
                    with patch.object(sys, "argv", command), self.assertRaises(StopBeforeModel):
                        ns["main"]()
                    self.assertEqual(calls, [((split,), bank.TRAINING_ROLES)])


if __name__ == "__main__":
    unittest.main()
