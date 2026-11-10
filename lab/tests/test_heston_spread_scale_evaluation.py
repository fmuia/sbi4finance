"""Population generation and strict joint-reference matching, without fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import evaluate_heston_spread_scale as evaluation
from scripts import train_heston_spread_scale as trainer
from test_heston_spread_training import fixture


class JointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        design,_,_,_=fixture(self.root)
        self.task,_=trainer.make_task(design)
        self.target=self.task["target"]

    def test_population_has_hidden_fresh_c_and_reproducible_independent_errors(self):
        bounds=np.asarray(self.target["prior_bounds"][:5])
        physical=np.random.default_rng(91).uniform(bounds[:,0],bounds[:,1],(512,5))
        clean=np.zeros((512,54))
        y,truth=evaluation.population_observations(self.task,physical,clean)
        c=trainer.draw_c(512,0,evaluation.SEED,namespace=3)
        expected_z=np.random.default_rng(np.random.SeedSequence([0x53504556,evaluation.SEED])).standard_normal(clean.shape)
        np.testing.assert_array_equal(truth[:,:5],physical)
        np.testing.assert_array_equal(truth[:,5],c)
        np.testing.assert_array_equal(y,c[:,None]*np.asarray(self.task["records"]["half_spread"])*expected_z)
        changed=physical.copy();changed[:,0]=bounds[0].mean()
        other_y,other_truth=evaluation.population_observations(self.task,changed,clean)
        np.testing.assert_array_equal(other_y,y)
        np.testing.assert_array_equal(other_truth[:,5],c)
        conditions=trainer.condition_data(self.task,y)
        self.assertEqual(set(conditions),{"observations"})
        self.assertEqual(conditions["observations"].shape,(512,54))
        with self.assertRaisesRegex(ValueError,"shape"):
            evaluation.population_observations(self.task,physical[:511],clean[:511])

    def reference_fixture(self):
        directory=self.root/"reference";directory.mkdir()
        panel=self.root/"panel.json";panel.write_text("{}")
        y=np.arange(54,dtype=float)
        products={"control":[0.,.1],"affected":[.3,.9]}
        chain_path=self.root/"chains.npz"
        bounds=np.asarray(self.target["prior_bounds"])
        np.savez(chain_path,parameters=np.broadcast_to(bounds.mean(axis=1),(4,5,6)))
        diagnostic={"reference_ready":True,"chain_payload_sha256":trainer.fixed.digest(chain_path)}
        trainer.fixed.write_json(directory/"diagnostics.json",diagnostic)
        record={"status":"completed","chain_payload_path":str(chain_path),"task":self.target,
            "names":self.target["parameter_names"],"bounds":self.target["prior_bounds"],
            "case_id":"synthetic-broad","design_sha256":"design-sha","panel_sha256":trainer.fixed.digest(panel),
            "observation_sha256":evaluation.baseline.observations_digest(y),"observations":y.tolist(),"products":products,
            "selected_attempt":0,"attempts":[{"draws":5,"diagnostics_sha256":trainer.fixed.digest(directory/"diagnostics.json")}],
            "settings":{"chains":4}}
        trainer.fixed.write_json(directory/"run.json",record)
        protocol={"design_sha256":"design-sha","jobs":[{"case_id":"synthetic-broad","mode":"joint","evidence_out":str(directory)}]}
        return directory,panel,y,products,record,protocol

    def test_only_bound_ready_six_dimensional_reference_can_be_used(self):
        directory,panel,y,products,record,protocol=self.reference_fixture()
        selected,binding=evaluation.joint_reference(protocol,"synthetic-broad",panel,self.target,products,y)
        self.assertEqual(selected[0].shape,(4,5,6))
        self.assertTrue(binding["reference_ready"])
        for prefix in ("reference_run","reference_diagnostics","reference_chain_payload"):
            self.assertEqual(trainer.fixed.digest(binding[prefix+"_path"]),binding[prefix+"_sha256"])
        for mutation in ("fixed-target","observations","chain-payload","selected-diagnostics"):
            bad=copy.deepcopy(record)
            if mutation=="fixed-target":bad["task"]={**self.target,"infer_c":False}
            elif mutation=="observations":bad["observations"][0]+=1
            elif mutation=="chain-payload":
                bad["chain_payload_path"]=str(self.root/"other.npz")
                np.savez(bad["chain_payload_path"],parameters=np.zeros((4,5,6)))
            else:bad["attempts"][0]["diagnostics_sha256"]="wrong"
            trainer.fixed.write_json(directory/"run.json",bad)
            with self.assertRaisesRegex(ValueError,"differs"):
                evaluation.joint_reference(protocol,"synthetic-broad",panel,self.target,products,y)

    def test_missing_and_unready_reference_are_inconclusive(self):
        directory,panel,y,products,record,protocol=self.reference_fixture()
        diagnostics=json.loads((directory/"diagnostics.json").read_text())
        diagnostics["reference_ready"]=False
        trainer.fixed.write_json(directory/"diagnostics.json",diagnostics)
        record["attempts"][0]["diagnostics_sha256"]=trainer.fixed.digest(directory/"diagnostics.json")
        trainer.fixed.write_json(directory/"run.json",record)
        selected,binding=evaluation.joint_reference(protocol,"synthetic-broad",panel,self.target,products,y)
        self.assertIsNone(selected);self.assertEqual(binding["status"],"inconclusive")
        (directory/"run.json").unlink()
        selected,binding=evaluation.joint_reference(protocol,"synthetic-broad",panel,self.target,products,y)
        self.assertIsNone(selected);self.assertEqual(binding["status"],"inconclusive")

    def test_checkpoint_cannot_substitute_another_training_recipe(self):
        record={"task":self.task,"bank_record_sha256":"manifest","bank_payload_sha256":"payload",
            "train_rows":20000,"seed":9011,"batch_size":128,"planned_updates":60000}
        protocol=self.root/"training-protocol.json"
        contract=trainer.protocol_contract(self.task,record)
        trainer.fixed.write_json(protocol,contract)
        record.update(protocol_path=str(protocol),protocol_sha256=trainer.fixed.digest(protocol),recipe=contract)
        self.assertEqual(evaluation.validate_training_record(record,self.root/"study.json"),protocol)
        for field,value in (("train_rows",10000),("seed",9012),("batch_size",256),("planned_updates",100000)):
            with self.subTest(field=field):
                bad=copy.deepcopy(record);bad[field]=value
                with self.assertRaisesRegex(ValueError,"recipe/protocol"):
                    evaluation.validate_training_record(bad,self.root/"study.json")
        trainer.fixed.write_json(protocol,{**contract,"scale_prior":[.25,4.]})
        record["protocol_sha256"]=trainer.fixed.digest(protocol)
        with self.assertRaisesRegex(ValueError,"recipe/protocol"):
            evaluation.validate_training_record(record,self.root/"study.json")

    def test_declaration_cannot_claim_different_executed_comparison_settings(self):
        study={"format":"heston_spread_noise_scale_study_v1","joint_task":self.target,"joint_evaluation":{
            "seed":261112,"population_rows":512,"posterior_draws_per_population_case":512,
            "conditional_draws":4096,"family_checks":144,"mean_tolerance_reference_sd":.3,
            "width_ratio90_limits":[.75,1.25],"block_bootstrap_replicates":400,
            "population_c":"independent Uniform(.5,2) per validation surface: draw_c(512,0,261112,namespace=3)",
            "population_gaussian":"reuse PCG64 SeedSequence([0x53504556,261112]) standard errors; apply generated c*h",
            "checkpoints":[10000,50000,60000],"conditional_cases":list(evaluation.CASES)}}
        evaluation.validate_study(study,self.target)
        for key,value in (("mean_tolerance_reference_sd",.1),("width_ratio90_limits",[.8,1.2]),
                ("block_bootstrap_replicates",200),("family_checks",48),("population_c","fixed c=1")):
            bad=copy.deepcopy(study);bad["joint_evaluation"][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"protocol differs"):
                evaluation.validate_study(bad,self.target)


if __name__=="__main__":unittest.main()
