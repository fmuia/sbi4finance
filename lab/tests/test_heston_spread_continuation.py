"""Exact toy continuation proof and rejection of changed resume targets/state."""
import copy
from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import continue_heston_spread_baseline as continuation
from scripts import train_heston_spread_baseline as trainer
from test_heston_spread_training import fixture


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE,"requires locked BF2.1 JAX environment")
class SpreadContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        design,bank,_,_ = fixture(cls.root)
        cls.design,cls.bank = design,bank
        recipe = dict(train_rows=128,updates=5,probe_rows=16,log_every=5,
                      smoke_test_only=True,checkpoint_updates=(2,5))
        trainer.run_training(design,bank,cls.root/"default",**recipe)
        trainer.run_training(design,bank,cls.root/"zero",dropout_zero_control=True,
                             control_reference=cls.root/"default",**recipe)
        cls.parent = cls.root/"zero/checkpoints/update-000002"
        cls.parent_record = json.loads((cls.parent/"checkpoint.json").read_text())
        continuation.run_continuation(cls.parent,cls.root/"resumed",smoke_test_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_resumed_two_plus_three_is_exactly_uninterrupted_five_for_every_state_array(self):
        direct = self.root/"zero/checkpoints/update-000005"
        resumed = self.root/"resumed/checkpoints/update-000005"
        with np.load(direct/"state.npz",allow_pickle=False) as left, np.load(resumed/"state.npz",allow_pickle=False) as right:
            self.assertEqual(set(left.files),set(right.files))
            for key in left.files:
                np.testing.assert_array_equal(left[key],right[key],err_msg=key)
        direct_model,direct_record = trainer.load_checkpoint(direct)
        resumed_model,resumed_record = trainer.load_checkpoint(resumed)
        continuation.restore_full_state(direct_model,direct_record,direct/"state.npz")
        continuation.restore_full_state(resumed_model,resumed_record,resumed/"state.npz")
        self.assertEqual(direct_model.estimator.optimizer.get_config(),resumed_model.estimator.optimizer.get_config())
        self.assertEqual(float(direct_model.estimator.optimizer.learning_rate),float(resumed_model.estimator.optimizer.learning_rate))
        self.assertEqual(direct_record["probe_metrics"]["batch_training_loss"],resumed_record["probe_metrics"]["batch_training_loss"])
        for key in ("train_probe_loss","validation_probe_loss"):
            self.assertEqual(direct_record["probe_metrics"][key],resumed_record["probe_metrics"][key])
        quotes = np.full((2,54),2.)
        a = trainer.sample(direct_model,direct_record["task"],quotes,16,seed=41)
        b = trainer.sample(resumed_model,resumed_record["task"],quotes,16,seed=41)
        np.testing.assert_array_equal(a,b)
        with np.load(self.root/"zero/exposure.npz",allow_pickle=False) as a, np.load(self.root/"resumed/exposure.npz",allow_pickle=False) as b:
            np.testing.assert_array_equal(a["presentations"],b["presentations"])
            np.testing.assert_array_equal(b["presentations_before_resume"]+b["continuation_presentations"],b["presentations"])
        protocol = json.loads((self.root/"resumed/resume-protocol.json").read_text())
        self.assertTrue(protocol["restoration"]["all_array_hashes_exact"])
        self.assertEqual(protocol["start_updates"],2)
        self.assertEqual(protocol["target_updates"],5)
        self.assertEqual(protocol["schedule_updates"],10000)

    def test_absolute_batch_and_noise_steps_are_not_restarted(self):
        parent = self.parent_record
        data,_ = trainer.load_clean_bank(self.bank,parent["task"],128,smoke_test_only=True)
        old = trainer.BatchPlan(128,128,5,9011)
        longer = trainer.BatchPlan(128,128,50000,9011)
        for step in range(5):
            np.testing.assert_array_equal(old.indices(step),longer.indices(step))
        indices = longer.indices(2)
        expected = trainer.noisy_quotes(data["train"]["clean"][indices],parent["task"]["records"]["half_spread"],2,9011)
        saved = json.loads((self.root/"resumed/first-batch.json").read_text())
        self.assertEqual(saved["absolute_step"],2)
        self.assertEqual(saved["indices_sha256"],trainer.array_summary(indices)["sha256"])
        self.assertEqual(saved["quotes_sha256"],trainer.array_summary(expected)["sha256"])

    def test_changed_task_schedule_or_dropout_is_rejected_before_resume(self):
        for field in ("task","schedule","dropout"):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.parent_record)
                if field == "task":
                    bad["task"]["target"]["fixed_c"] = 2.
                elif field == "schedule":
                    bad["schedule_updates"] = 50000
                else:
                    bad["dropout_zero_control"] = False
                with self.assertRaises(ValueError):
                    continuation.validate_parent_record(bad,smoke_test_only=True)
        with self.assertRaisesRegex(ValueError,"10k"):
            continuation.validate_parent_record(self.parent_record)

    def test_changed_bank_identity_or_full_state_hash_is_rejected(self):
        parent = self.parent_record
        bank = {key:copy.deepcopy(parent[key]) for key in ("bank_record_sha256","bank_payload_sha256","bank_identity")}
        for field in bank:
            bad = copy.deepcopy(bank)
            bad[field] = {} if field == "bank_identity" else "0"*64
            with self.assertRaisesRegex(ValueError,"bank/task"):
                continuation.validate_bank_binding(parent,bad)
        workflow,_ = trainer.load_checkpoint(self.parent)
        bad = copy.deepcopy(parent)
        bad["state_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"payload hash"):
            continuation.restore_full_state(workflow,bad,self.parent/"state.npz")
        bad = copy.deepcopy(parent)
        bad["state"]["non_trainable"][0]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"shape/dtype/hash"):
            continuation.restore_full_state(workflow,bad,self.parent/"state.npz")


if __name__ == "__main__":
    unittest.main()
