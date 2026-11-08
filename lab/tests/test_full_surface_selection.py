"""A predeclared final checkpoint must survive real chunk interruption/recovery."""

from argparse import Namespace
from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_full_surface as trainer
from sbi4finance.calibration import bank_collection, full_surface, tasks

try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE, "requires the isolated BF2.1 core environment")
class FullSurfaceSelectionTests(unittest.TestCase):
    def fixture(self, root, policy=None):
        task = tasks.rough_task()
        config = {"task": task, "root_seed": 17, "splits": {"train": 16, "validation": 8, "test": 4}}
        manifest = {"configuration": config, "task_fingerprint": bank_collection.bank.task_fingerprint(config)}
        bank = root / "bank"; bank.mkdir(); (bank / "manifest.json").write_text(json.dumps(manifest))
        rng = np.random.default_rng(91); bounds = np.asarray(task["prior_bounds"])
        def split(n):
            prices = rng.normal(0, .001, (n, 63))
            return {"parameters": rng.uniform(bounds[:, 0], bounds[:, 1], (n, 3)),
                    "reference_prices": prices, "observations": prices[..., None],
                    "context_values": rng.uniform(.02, .1, (n, 7))}
        splits = {"train": split(16), "validation": split(8)}
        args = Namespace(bank=bank, out=root / "run", family="flow_matching", encoder="vector",
                         widths="8,8", summary_dim=8, flow_steps=None, mean_steps=4,
                         epochs=2, batch_size=8, chunk_epochs=1, seed=3)
        if policy is not None: args.checkpoint_selection = policy
        return args, manifest, splits

    def patched_build(self):
        import keras
        original = full_surface.build_workflow
        class ForceBestAtEpochOne(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs["val_loss"] = float(epoch)
        def build(*args, **kwargs):
            workflow = original(*args, **kwargs)
            fit = workflow.fit_offline
            def train(*fit_args, **fit_kwargs):
                fit_kwargs["callbacks"] = [ForceBestAtEpochOne(), *fit_kwargs.get("callbacks", [])]
                return fit(*fit_args, **fit_kwargs)
            workflow.fit_offline = train
            return workflow
        return build

    def test_default_still_publishes_validation_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, manifest, splits = self.fixture(Path(temporary))
            with patch.object(trainer.bank, "load_bank", return_value=(manifest, splits)), \
                    patch.object(trainer.full_surface, "build_workflow", side_effect=self.patched_build()):
                record = trainer.run_training(args)
            state = json.loads((args.out / "resume.json").read_text())
            self.assertEqual(record["checkpoint_selection"], "validation_loss")
            self.assertEqual(record["selected_epoch"], 1)
            self.assertEqual(record["validation_best_epoch"], 1)
            self.assertEqual(record["checkpoint_sha256"], trainer.sha256(args.out / state["checkpoint_files"]["best"]))
            self.assertNotEqual(record["checkpoint_sha256"], record["final_checkpoint_sha256"])
            self.assertEqual(record["performance"]["work"]["optimizer_updates"], 4)
            self.assertEqual(record["performance"]["work"]["selected_checkpoint_optimizer_updates"], 2)

    def test_final_policy_recovers_then_publishes_last_while_retaining_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, manifest, splits = self.fixture(Path(temporary), "final")
            original_commit = trainer.recovery.commit_resume_state
            def interrupt_after_commit(*a, **kw):
                original_commit(*a, **kw)
                raise RuntimeError("synthetic interruption after protected epoch1 commit")
            with patch.object(trainer.bank, "load_bank", return_value=(manifest, splits)), \
                    patch.object(trainer.full_surface, "build_workflow", side_effect=self.patched_build()):
                with patch.object(trainer.recovery, "commit_resume_state", side_effect=interrupt_after_commit):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        trainer.run_training(args)
                first = json.loads((args.out / "resume.json").read_text())
                self.assertEqual(first["epochs_done"], 1)
                record = trainer.run_training(args)
                # Recovery after an already-completed schedule must publish the
                # same final model, rather than the restored validation best.
                repeated = trainer.run_training(args)
            state = json.loads((args.out / "resume.json").read_text())
            self.assertEqual(record["selected_epoch"], 2)
            self.assertEqual(record["validation_best_epoch"], 1)
            self.assertEqual(record["resumed_at_epochs"], [1])
            self.assertEqual(record["checkpoint_sha256"], trainer.sha256(args.out / state["checkpoint_files"]["last"]))
            self.assertEqual(record["checkpoint_sha256"], repeated["checkpoint_sha256"])
            self.assertEqual(repeated["performance"]["work"]["optimizer_updates_this_invocation"], 0)
            self.assertEqual(repeated["performance"]["work"]["training_presentations_this_invocation"], 0)
            self.assertEqual(repeated["performance"]["work"]["unique_training_rows_this_invocation"], 0)
            self.assertNotEqual(record["checkpoint_sha256"], record["validation_best_checkpoint_sha256"])
            self.assertTrue(record["same_seed_reload_exact"])
            self.assertFalse(record["scientific_validation"])


if __name__ == "__main__":
    unittest.main()
