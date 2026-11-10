"""Checkpoint replay retains the original trajectory, schedule and full state."""
import copy
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import replay_heston_spread_whitening as replay
from scripts import train_heston_spread_whitening as frozen
from scripts import train_heston_spread_scale as scale
from test_heston_spread_training import fixture


class ReplayIdentityTests(unittest.TestCase):
    def test_curve_comparison_ignores_only_time(self):
        point = {"updates":40,"completed_bank_passes":1,"presentations":10000,
                 "seconds":1.,"batch_training_loss":-2.,"train_probe_loss":-3.,"validation_probe_loss":-4.}
        replay.assert_curve_point(point,{**point,"seconds":99.})
        for key in ("updates","presentations","batch_training_loss","train_probe_loss","validation_probe_loss"):
            with self.assertRaises(ValueError):
                replay.assert_curve_point({**point,key:point[key]+1},point)

    def test_initial_state_checks_every_group_and_array_identity(self):
        state = {group:[{"shape":[2],"dtype":"float32","sha256":"abc","path":"old"}]
                 for group in replay.fixed.GROUPS}
        renamed = copy.deepcopy(state)
        for rows in renamed.values():
            rows[0]["path"] = "new"
        replay.assert_initial_state(renamed,state)
        for group in replay.fixed.GROUPS:
            changed = copy.deepcopy(state)
            changed[group][0]["sha256"] = "wrong"
            with self.assertRaises(ValueError):
                replay.assert_initial_state(changed,state)


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE,"requires locked BF2.1 environment")
class ReplayNativeTests(unittest.TestCase):
    def test_original_prefix_native_cli_checkpoints_and_next_update(self):
        import jax
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            design,bank,_,_ = fixture(root)
            task,_ = scale.make_task(design)
            original,output = root/"original",root/"replay"
            old_protocol = root/"original-protocol.json"
            old_protocol.write_text(json.dumps({"format":"heston_spread_whitening_study_v1","arms":[{
                "mode":"zca","contract":frozen.protocol_contract(bank,task,mode="zca",smoke_test_only=True),
                "output":str(original)}]}))
            frozen.run_training(bank,task,original,mode="zca",protocol=old_protocol,smoke_test_only=True)
            protocol = root/"replay-protocol.json"
            contract = replay.protocol_contract(bank,task,original,smoke_test_only=True)
            self.assertEqual(contract["planned_updates"],4)
            self.assertEqual(contract["schedule_updates"],5)
            self.assertEqual(contract["optimizer"]["schedule"]["decay_steps"],5)
            protocol.write_text(json.dumps({"format":replay.PROTOCOL_FORMAT,"contract":contract,"output":str(output)}))
            task_path = root/"task.json"
            task_path.write_text(json.dumps(task))
            completed = subprocess.run([sys.executable,str(Path(replay.__file__).resolve()),
                "--bank",str(bank),"--task",str(task_path),"--out",str(output),"--original",str(original),
                "--protocol",str(protocol),"--smoke-test-only","--worker"],capture_output=True,text=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stdout[-4000:]+completed.stderr[-4000:])
            result = json.loads((output/"training.json").read_text())
            self.assertTrue(result["original_learning_curve_prefix_exact"])
            self.assertTrue(result["all_original_initial_state_hashes_exact"])
            self.assertTrue(result["same_seed_reload_exact"])
            self.assertEqual(result["original_curve_points_compared"],1)
            self.assertEqual((output/"preprocessing.json").read_bytes(),(original/"preprocessing.json").read_bytes())
            old_curve = json.loads((original/"learning-curve.json").read_text())
            curve = json.loads((output/"learning-curve.json").read_text())
            replay.assert_curve_point(curve[0],old_curve[0])
            # Saving and seeded sampling at update2/3 must not alter update4.
            left,_ = replay.load_checkpoint(output/"checkpoints/update-000003")
            right,_ = replay.load_checkpoint(output/"checkpoints/update-000004")
            data,_ = replay.prepare_data(bank,task,smoke_test_only=True)
            indices = replay.expansion.epoch_order(128,0,replay.SEED)[96:128]
            raw = {name:values[indices] for name,values in data["train"].items()}
            preprocessing = json.loads((output/"preprocessing.json").read_text())
            adapted = replay.adapted_data(task,raw,preprocessing)
            _,state,_,flags = replay.fixed.compiled_step(left.estimator)(replay.fixed.current_state(left.estimator),adapted)
            self.assertTrue(all(bool(value) for value in flags.values()))
            for actual,expected in zip(jax.tree.leaves(state),jax.tree.leaves(replay.fixed.current_state(right.estimator)),strict=True):
                np.testing.assert_array_equal(actual,expected)
            quotes = data["validation"]["observations"][:2]*np.asarray(task["records"]["half_spread"])
            first = replay.sample(right,task,quotes,16,77)
            again,_ = replay.load_checkpoint(output/"checkpoints/update-000004")
            np.testing.assert_array_equal(first,replay.sample(again,task,quotes,16,77))
            with self.assertRaises(ValueError):
                frozen.load_checkpoint(output/"checkpoints/update-000004")
            path = output/"checkpoints/update-000004/preprocessing.json"
            path.write_text(path.read_text()+" ")
            with self.assertRaises(ValueError):
                replay.load_checkpoint(path.parent)


if __name__ == "__main__":
    unittest.main()
