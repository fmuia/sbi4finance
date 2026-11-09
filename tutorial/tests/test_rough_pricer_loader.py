"""Native-pricer loading binds both the selected task and immutable payload."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4f_tutorial.rough_options import surrogate
from sbi4f_tutorial.shared import backbones, paths
from sbi4f_tutorial.shared.bank_helpers import _json


class RoughPricerLoaderTests(unittest.TestCase):
    def test_bound_numpy_pricer_loads_without_rewriting_payload_and_rejects_wrong_identity(self):
        task, _, _ = backbones.load_example("rough-joint100k-validation0")
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary)
            directory = assets / "local-checkpoints/pricer"
            p = surrogate.Pricer([(np.zeros((6, 63)), np.ones(63))], np.zeros(6), np.ones(6),
                                 np.zeros(63), np.ones(63), activation="gelu_erf")
            p.save(directory)
            original = (directory / "pricer.npz").read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            (directory / "task.json").write_text(_json(task))
            record = {"format": "full_surface_pricer_training_v1", "status": "completed",
                      "task_fingerprint": hashlib.sha256(_json(task).encode()).hexdigest(),
                      "task_id": task["task_id"], "pricer_sha256": digest,
                      "fit": {"hidden_activation": "gelu_erf"}}
            (directory / "pricer.json").write_text(json.dumps(record))
            with patch.object(paths, "ASSETS", assets):
                restored_task, restored, _ = surrogate.load_pricer(directory, expected_task=task,
                                                                  expected_pricer_sha256=digest)
                self.assertEqual(restored_task, task)
                np.testing.assert_array_equal(restored(np.zeros((1, 6))), np.ones((1, 63)))
                changed = copy.deepcopy(task)
                changed["prior_bounds"][0][0] = .03
                with self.assertRaisesRegex(ValueError, "selected lesson task"):
                    surrogate.load_pricer(directory, expected_task=changed)
                with self.assertRaisesRegex(ValueError, "published candidate"):
                    surrogate.load_pricer(directory, expected_pricer_sha256="0" * 64)
                (directory / "pricer.npz").write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "training record"):
                    surrogate.load_pricer(directory)

    def test_arbitrary_outside_paths_are_not_participant_pricer_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "local-checkpoints"):
                surrogate.load_pricer(Path(temporary))

    def test_symlinked_cache_cannot_move_pricer_reads_outside_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            assets, outside = base / "assets", base / "outside"
            assets.mkdir(); outside.mkdir()
            try:
                (assets / "local-checkpoints").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable")
            with patch.object(paths, "ASSETS", assets):
                with self.assertRaisesRegex(ValueError, "remain under tutorial/assets"):
                    surrogate.load_pricer("local-checkpoints/model")


if __name__ == "__main__":
    unittest.main()
