"""Six-target noise-scale task, independent augmentation and native checkpoints."""
from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_heston_spread_scale as scale
from test_heston_spread_training import fixture


class ScaleDataTests(unittest.TestCase):
    def test_clean_bank_reuse_is_an_explicit_unchanged_five_dimensional_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            design,bank,_,payload = fixture(Path(temporary))
            before = scale.fixed.digest(bank/"bank.npz")
            task,pilot = scale.make_task(design)
            self.assertEqual(pilot.names[-1],"c")
            self.assertEqual(len(pilot.names),6)
            np.testing.assert_array_equal(pilot.bounds[-1],[.5,2.])
            self.assertFalse(task["physical_clean_price_task"]["target"]["infer_c"])
            with patch.object(scale.HestonSpreadPilot,"prices",side_effect=AssertionError("no repricing")), patch.object(
                    scale.fixed,"load_clean_bank",wraps=scale.fixed.load_clean_bank) as load:
                data,record = scale.load_clean_bank(bank,task,128,smoke_test_only=True)
            self.assertEqual(len(load.call_args.args[1]["target"]["parameter_names"]),5)
            np.testing.assert_array_equal(data["train"]["parameters"],payload["train_parameters"])
            self.assertEqual(scale.fixed.digest(bank/"bank.npz"),before)
            self.assertFalse(record["reserved_arrays_opened"])

    def test_c_is_fresh_uniform_independent_of_gaussian_and_never_an_input(self):
        physical = np.tile([.04,2.,.05,.4,-.7],(50000,1))
        clean = np.zeros((len(physical),2)); h=np.array([.2,.5])
        y,target = scale.augmented_batch(clean,physical,h,7,9011)
        repeated = scale.augmented_batch(clean,physical,h,7,9011)
        np.testing.assert_array_equal(y,repeated[0]); np.testing.assert_array_equal(target,repeated[1])
        other = scale.augmented_batch(clean,physical,h,8,9011)
        self.assertFalse(np.array_equal(target[:,-1],other[1][:,-1]))
        c = target[:,-1]
        self.assertTrue(((c>.5)&(c<2.)).all())
        self.assertAlmostEqual(c.mean(),1.25,delta=.01)
        noise = y/(c[:,None]*h)
        np.testing.assert_allclose(noise.var(axis=0),1.,atol=.025)
        self.assertLess(abs(np.corrcoef(c,noise[:,0])[0,1]),.02)
        np.testing.assert_allclose(y.var(axis=0)/h**2,1.75,atol=.045)
        self.assertTrue((y<0).any())
        with tempfile.TemporaryDirectory() as temporary:
            design,_,_,_ = fixture(Path(temporary))
            task,_ = scale.make_task(design)
            data = scale.training_data(task,-np.ones((2,54)),target[:2])
            self.assertEqual(set(data),{"observations",*task["target"]["parameter_names"]})
            self.assertEqual(data["observations"].shape,(2,54))
            np.testing.assert_array_equal(data["c"],target[:2,-1:])
            changed = target[:2].copy(); changed[:,-1]=1.9
            np.testing.assert_array_equal(data["observations"],scale.training_data(task,-np.ones((2,54)),changed)["observations"])


try:
    CORE = metadata.version("bayesflow")=="2.1.0"
except metadata.PackageNotFoundError:
    CORE=False


@unittest.skipUnless(CORE,"requires locked BF2.1 environment")
class ScaleWorkflowTests(unittest.TestCase):
    def test_six_dimensional_native_step_target_exclusion_and_five_update_reload(self):
        import jax
        import keras
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            design,bank,_,_ = fixture(root)
            task,_ = scale.make_task(design)
            data,_ = scale.load_clean_bank(bank,task,128,smoke_test_only=True)
            keras.utils.set_random_seed(9011)
            workflow = scale.build_workflow(task)
            y,target = scale.augmented_batch(data["train"]["clean"],data["train"]["parameters"],task["records"]["half_spread"],0,9011)
            adapted = workflow.adapter(scale.training_data(task,y,target))
            self.assertEqual(adapted["inference_conditions"].shape,(128,54))
            self.assertEqual(adapted["inference_variables"].shape,(128,6))
            changed = target.copy(); changed[:,-1]=1.8
            changed_adapted = workflow.adapter(scale.training_data(task,y,changed))
            np.testing.assert_array_equal(adapted["inference_conditions"],changed_adapted["inference_conditions"])
            self.assertFalse(np.array_equal(adapted["inference_variables"][:,-1],changed_adapted["inference_variables"][:,-1]))
            model=workflow.estimator
            model.build_from_data(adapted); model.optimizer.build(model.trainable_variables)
            self.assertEqual(workflow.inference_network.depth,6)
            proof = scale.fixed.switch_dropout_zero(model)
            self.assertTrue(proof["array_state_unchanged"])
            state = scale.fixed.current_state(model)
            metrics,candidate,_,flags = scale.fixed.compiled_step(model)(state,adapted)
            native_metrics,native = jax.jit(model.stateless_train_step)(state,adapted)
            for left,right in zip(jax.tree.leaves(candidate),jax.tree.leaves(native),strict=True):
                np.testing.assert_array_equal(left,right)
            self.assertTrue(all(bool(v) for v in flags.values()))
            self.assertEqual(float(metrics["loss"]),float(native_metrics["loss"]))
            result = scale.run_training(design,bank,root/"tiny",smoke_test_only=True)
            self.assertEqual(result["completed_updates"],5)
            self.assertTrue(result["same_seed_reload_exact"])
            restored,record = scale.load_checkpoint(root/"tiny/checkpoints/update-000005")
            self.assertEqual(record["names"][-1],"c")
            self.assertEqual(scale.sample(restored,task,y[:2],8,seed=4).shape,(2,8,6))
            self.assertTrue(all(layer.rate==0 for layer in scale.fixed.dropout_layers(restored.estimator)))
            # Check built-in schedule serialization and all-array preservation.
            restored.estimator.optimizer.iterations.assign(50000)
            current = scale.fixed.current_state(restored.estimator)
            schedule_proof = scale.install_annealing(restored.estimator,current)
            self.assertTrue(schedule_proof["all_array_hashes_unchanged"])
            schedule = keras.optimizers.schedules.deserialize(schedule_proof["new_schedule"])
            np.testing.assert_allclose([float(schedule(v)) for v in (50000,54999,55000,59999)],[5e-6,5e-6,5e-7,5e-7],rtol=1e-7)
            # Save a clearly software-only synthetic50k counter to exercise the
            # phase-two native archive without a production fit.
            mock_run = {**record,"planned_updates":60000}
            scale.save_checkpoint(root/"phase-two",restored,current,mock_run,50000,{})
            again,metadata_record = scale.load_checkpoint(root/"phase-two")
            self.assertEqual(again.estimator.optimizer.get_config()["learning_rate"],schedule_proof["new_schedule"])
            self.assertEqual(metadata_record["actual_optimizer_updates"],50000)

    def test_production_requires_predeclared_protocol_before_output_or_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            design,bank,_,_ = fixture(root)
            task,_ = scale.make_task(design)
            data,record = scale.load_clean_bank(bank,task,128,smoke_test_only=True)
            with patch.object(scale,"load_clean_bank",return_value=(data,record)):
                with self.assertRaisesRegex(ValueError,"pre-run"):
                    scale.run_training(design,bank,root/"forbidden")
            self.assertFalse((root/"forbidden").exists())


if __name__=="__main__":
    unittest.main()
