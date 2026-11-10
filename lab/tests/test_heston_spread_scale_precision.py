"""Exact-state continuation, paired examples and exposure-budget accounting."""
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import continue_heston_spread_scale_precision as continuation
from test_heston_spread_training import fixture


class ScaleExpansionDataTests(unittest.TestCase):
    def test_epoch_pairing_survives_batch_grouping_and_tail(self):
        physical=np.tile([.04,2.,.05,.4,-.7],(103,1))
        train={"clean":np.arange(103*54).reshape(103,54)/1000.,"parameters":physical}
        h=np.linspace(.01,.1,54)
        order=continuation.epoch_order(103,7)
        quotes,targets=continuation.epoch_data(train,h,7)
        seen=[]
        for batch_size in (16,64):
            grouped=[order[offset:offset+batch_size] for offset in range(0,103,batch_size)]
            self.assertEqual(len(grouped[-1]),103%batch_size)
            indices=np.concatenate(grouped)
            np.testing.assert_array_equal(np.sort(indices),np.arange(103))
            seen.append((quotes[indices],targets[indices]))
        np.testing.assert_array_equal(seen[0][0],seen[1][0])
        np.testing.assert_array_equal(seen[0][1],seen[1][1])
        self.assertFalse(np.array_equal(targets[:,-1],continuation.epoch_data(train,h,8)[1][:,-1]))

    def test_full_loader_opens_no_reserved_arrays_and_production_is_not_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            design,bank,_,payload=fixture(Path(temporary))
            task,_=continuation.scale.make_task(design)
            real_load=continuation.np.load
            accessed=[]
            class Archive:
                def __init__(self,*args,**kwargs):
                    self.archive=real_load(*args,**kwargs)
                def __enter__(self):
                    return self
                def __exit__(self,*args):
                    self.archive.close()
                def __getitem__(self,key):
                    accessed.append(key)
                    if key.startswith("test_"):
                        raise AssertionError("reserved array opened")
                    return self.archive[key]
            with patch.object(continuation.np,"load",side_effect=Archive):
                data,binding=continuation.load_bank(bank,task,smoke_test_only=True)
            np.testing.assert_array_equal(data["train"]["clean"],payload["train_clean"])
            self.assertFalse(binding["reserved_arrays_opened"])
            self.assertTrue(accessed)
            with patch.object(continuation.scale,"load_clean_bank",side_effect=RuntimeError("production guard")) as load:
                with self.assertRaisesRegex(RuntimeError,"production guard"):
                    continuation.load_bank(bank,task)
            self.assertEqual(load.call_args.kwargs,{"smoke_test_only":False})
            self.assertEqual(load.call_args.args[2],20000)


try:
    CORE=metadata.version("bayesflow")=="2.1.0"
except metadata.PackageNotFoundError:
    CORE=False


@unittest.skipUnless(CORE,"requires locked BF2.1 environment")
class ScaleExpansionWorkflowTests(unittest.TestCase):
    def test_precision_transform_keeps_true_open_prior_without_clipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            design,_,_,_=fixture(Path(temporary))
            task,_=continuation.scale.make_task(design)
            bounds=np.asarray(task["target"]["prior_bounds"],dtype=np.float64)
            values=np.tile(bounds.mean(axis=1),(4,1))
            values[0,-1]=1.9999999587495068  # Exact first production failure.
            values[1]=np.nextafter(bounds[:,0],bounds[:,1])
            values[2]=np.nextafter(bounds[:,1],bounds[:,0])
            observations=np.arange(4*54).reshape(4,54)/1000
            raw=continuation.scale.training_data(task,observations,values)
            protected={key:value.copy() for key,value in raw.items()}
            adapted=continuation.host_adapt_training(task,raw)
            self.assertTrue(np.isfinite(adapted["inference_variables"]).all())
            expected=np.log(values-bounds[:,0])-np.log(bounds[:,1]-values)
            np.testing.assert_array_equal(adapted["inference_variables"],expected.astype(np.float32))
            np.testing.assert_array_equal(adapted["inference_conditions"],raw["observations"].astype(np.float32))
            for key in raw:
                np.testing.assert_array_equal(raw[key],protected[key])
            workflow=continuation.scale.build_workflow(task)
            old=workflow.adapter(raw)
            self.assertTrue(np.isposinf(np.asarray(old["inference_variables"])[0,-1]))
            self.assertAlmostEqual(float(adapted["inference_variables"][0,-1]),17.40906784742835,places=5)
            # Ordinary observations are byte-identical to the previous input path.
            np.testing.assert_array_equal(adapted["inference_conditions"],old["inference_conditions"])
            # Exact endpoints remain invalid: this is a precision repair, not a widened prior.
            invalid={**raw,"c":raw["c"].copy()}; invalid["c"][0,0]=2.
            with self.assertRaisesRegex(ValueError,"strictly inside"):
                continuation.host_adapt_training(task,invalid)

    def test_schedule_is_paired_by_exact_examples_and_serializes(self):
        import keras
        klass=continuation.schedule_class()
        for batch in (128,512):
            batches=(100000+batch-1)//batch
            schedule=klass(100000,batch,60000)
            np.testing.assert_allclose([float(schedule(60000+epoch*batches)) for epoch in (0,100,200,400,500)],
                [5e-5,5e-6+(5e-5-5e-6)*.5*(1+np.cos(np.pi*.25)),2.75e-5,5e-6,5e-6],rtol=2e-7)
            restored=keras.optimizers.schedules.deserialize(keras.optimizers.schedules.serialize(schedule))
            self.assertEqual(float(schedule(60000+50*batches+3)),float(restored(60000+50*batches+3)))

    def test_full_parent_arrays_native_next_update_and_checkpoint_reload_are_exact(self):
        import jax
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            design,bank,_,_=fixture(root)
            continuation.scale.run_training(design,bank,root/"parent",smoke_test_only=True)
            parent=root/"parent/checkpoints/update-000005"
            protocol=continuation.protocol_contract(parent,bank,32,smoke_test_only=True)
            path=root/"protocol.json"
            path.write_text(json.dumps({"format":"heston_spread_scale_precision_recovery_study_v1",
                "arms":[{"batch_size":32,"contract":protocol,"output":str(root/"new")}]}))
            result=continuation.run_continuation(parent,bank,root/"new",protocol=path,batch_size=32,smoke_test_only=True)
            self.assertTrue(result["all_parent_array_hashes_restored"])
            self.assertEqual(result["continued_row_presentations"],256)
            self.assertEqual(result["continued_updates"],8)
            self.assertTrue(result["same_seed_reload_exact"])
            final=root/"new/checkpoints/pass-0002"
            workflow,record=continuation.load_checkpoint(final)
            self.assertEqual(record["actual_optimizer_updates"],13)
            self.assertEqual(record["completed_bank_passes"],2)
            data,_=continuation.load_bank(bank,record["task"],smoke_test_only=True)
            quotes,targets=continuation.epoch_data(data["train"],record["task"]["records"]["half_spread"],2)
            adapted=continuation.host_adapt_training(record["task"],continuation.scale.training_data(record["task"],quotes[:32],targets[:32]))
            state=continuation.fixed.current_state(workflow.estimator)
            before=continuation.fixed.compiled_step(workflow.estimator)(state,adapted)
            restored,_=continuation.load_checkpoint(final)
            after=continuation.fixed.compiled_step(restored.estimator)(continuation.fixed.current_state(restored.estimator),adapted)
            for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(after),strict=True):
                np.testing.assert_array_equal(a,b)
            self.assertTrue(all(bool(v) for v in before[-1].values()))
            # Both supported batch sizes preserve row presentations despite different update counts.
            second=continuation.protocol_contract(parent,bank,64,smoke_test_only=True)
            self.assertEqual(second["additional_presentations"],protocol["additional_presentations"])
            self.assertEqual(second["additional_optimizer_updates"]*2,protocol["additional_optimizer_updates"])
            cli_protocol=root/"cli-protocol.json"
            cli_protocol.write_text(json.dumps({"format":"heston_spread_scale_precision_recovery_study_v1",
                "arms":[{"batch_size":64,"contract":second,"output":str(root/"cli")}]}))
            completed=subprocess.run([sys.executable,str(Path(continuation.__file__).resolve()),
                "--checkpoint",str(parent),"--bank",str(bank),"--protocol",str(cli_protocol),
                "--batch-size","64","--out",str(root/"cli"),"--smoke-test-only","--worker"],
                capture_output=True,text=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stdout[-3000:]+completed.stderr[-3000:])
            cli_checkpoint=root/"cli/checkpoints/pass-0002"
            imported,cli_record=continuation.load_checkpoint(cli_checkpoint)
            self.assertEqual(cli_record["actual_optimizer_updates"],9)
            self.assertEqual(cli_record["optimizer_schedule"]["module"],"scripts.continue_heston_spread_scale")
            cli_state=continuation.fixed.current_state(imported.estimator)
            cli_before=continuation.fixed.compiled_step(imported.estimator)(cli_state,adapted)
            cli_reloaded,_=continuation.load_checkpoint(cli_checkpoint)
            cli_after=continuation.fixed.compiled_step(cli_reloaded.estimator)(continuation.fixed.current_state(cli_reloaded.estimator),adapted)
            for a,b in zip(jax.tree.leaves(cli_before),jax.tree.leaves(cli_after),strict=True):
                np.testing.assert_array_equal(a,b)
            # Payload corruption fails before a resumed update.
            with np.load(final/"state.npz",allow_pickle=False) as archive:
                arrays={key:archive[key].copy() for key in archive.files}
            arrays["trainable_0000"].flat[0]+=1
            np.savez_compressed(final/"state.npz",**arrays)
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                continuation.load_checkpoint(final)

    def test_no_output_before_exact_protocol_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            out=root/"must-not-exist"
            path=root/"wrong.json"; path.write_text(json.dumps({"format":"heston_spread_scale_precision_recovery_study_v1",
                "arms":[{"batch_size":128,"contract":{},"output":str(out)}]}))
            with patch.object(continuation,"protocol_contract",return_value={"required":1}):
                with self.assertRaisesRegex(ValueError,"persist the exact"):
                    continuation.run_continuation(root,root,out,protocol=path,batch_size=128)
            self.assertFalse(out.exists())


if __name__=="__main__":
    unittest.main()
