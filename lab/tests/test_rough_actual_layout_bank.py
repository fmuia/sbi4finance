"""Pure configuration and fake-estimator durability checks; never rough MC."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import rough_actual_layout_bank as bank
from rough_actual_layout_fixtures import current_portable


class ActualBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        _,self.parent=current_portable(self.root)
        self.policy={"grid_pair":[2,4],"paths_ladder":[32768],"mcse_over_strict_sd":.25,
            "refinement_over_strict_sd":None,"refinement_z":3.}

    def prepare(self, *, candidate=True):
        policy=copy.deepcopy(self.policy)
        if not candidate: policy["refinement_over_strict_sd"]=.25
        path=self.root/"plan.json"
        plan=bank.prepare(self.parent,path,root_seed=20261111,
            target_splits={"train":300000,"validation":5000,"test":7000},
            active_ranges={"supplied_known":{},"joint_unknown":{"train":[0,2]}},rows_per_shard=2,
            numerical_policy=policy,row_wall_seconds=30,shard_wall_seconds=90,
            concurrent_workers=1,total_cpu_seconds=90,candidate_only=candidate)
        review=self.root/"review.json"; review.write_text('{"measured":false}\n')
        admission=self.root/"authorization.json"
        admission.write_text(json.dumps({"format":bank.ADMISSION,"bank_plan_sha256":bank.pilot.digest(path),
            "generation_authorized":True,"mode":"candidate_only" if candidate else "qualified_rows",
            "pilot_numerical_review_passed":not candidate,"numerical_policy":policy,
            "scientific_acceptance":False,"numerical_review":bank.pilot.binding(review)}))
        return plan,path,admission

    @staticmethod
    def fake_estimator(*args,**kwargs):
        n=len(args[2]);cov=np.zeros((n,n))
        return {"controlled_prices":np.full((2,n),.1),"raw_prices":np.full((2,n),.1),
            "joint_covariance":np.zeros((4,n,4,n)),"difference":np.zeros(n),
            "difference_covariance":cov,"moments":[{"bank_covariance":cov.tolist()},{"bank_covariance":cov.tolist()}]}

    def run_fake(self,plan_path,auth,out,estimator=None):
        class Helper: pass
        helper=Helper(); helper.coupled_controlled_block=estimator or self.fake_estimator
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"1"}), patch.object(
                bank.pilot,"load_controlled_helper",return_value=helper):
            return bank.run_shard(plan_path,auth,0,out,process_started=time.monotonic())

    def test_prepare_is_metadata_only_and_budget_is_explicit(self):
        plan,path,_=self.prepare();bank.validate(plan)
        self.assertEqual(plan["target_splits"]["train"],300000)
        self.assertEqual(sum(x["stop"]-x["start"] for x in plan["jobs"]),2)
        self.assertEqual(plan["prices_generated"],0)
        self.assertEqual(plan["limits"]["active_worst_case_allocated_cpu_seconds"],90)
        changed=copy.deepcopy(plan);changed["limits"]["cpu_seconds_ceiling"]=89
        with self.assertRaisesRegex(ValueError,"allocation"):bank.validate(changed)

    def test_row_ids_are_count_and_scheduling_independent_and_dynamics_pair(self):
        plan,_,_=self.prepare();a=bank.row_spec(plan,"joint_unknown","train",1)
        changed=copy.deepcopy(plan);changed["target_splits"]["train"]=600000
        self.assertEqual(a,bank.row_spec(changed,"joint_unknown","train",1))
        b=bank.row_spec(plan,"supplied_known","train",1)
        self.assertEqual(a["parameters"][:3],b["parameters"])
        self.assertEqual(len(b["supplied_curve_levels"]),7)

    def test_no_refinement_default_can_claim_qualified_data(self):
        with self.assertRaises(ValueError):bank.checked_policy(self.policy)
        metrics=bank.quality(self.fake_estimator(None,None,[0]*9),np.ones(9),self.policy)
        self.assertTrue(metrics["mc_passed"]);self.assertIsNone(metrics["refinement_passed"]);self.assertFalse(metrics["passed"])

    def test_candidate_labels_have_false_training_flag_and_resume_without_repricing(self):
        plan,path,auth=self.prepare();out=self.root/"shard"
        result=self.run_fake(path,auth,out)
        self.assertFalse(result["training_usable"]);self.assertEqual(result["physical_rows"],2)
        with np.load(out/"data.npz",allow_pickle=False) as arrays:
            self.assertFalse(arrays["numerical_qualified"].any())
            self.assertEqual(arrays["observations_forward"].shape,(2,63))
            self.assertEqual(arrays["supplied_curve_levels"].shape,(2,0))
        def fail(*args,**kwargs):raise AssertionError("completed data was repriced")
        self.assertEqual(result,self.run_fake(path,auth,out,fail))

    def test_qualified_mode_requires_pilot_review_and_explicit_gate(self):
        plan,path,auth=self.prepare(candidate=False)
        record=json.loads(auth.read_text());record["pilot_numerical_review_passed"]=False;auth.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"authorization"):self.run_fake(path,auth,self.root/"bad")
        record["pilot_numerical_review_passed"]=True;auth.write_text(json.dumps(record))
        result=self.run_fake(path,auth,self.root/"good");self.assertTrue(result["training_usable"])

    def test_failed_row_keeps_committed_blocks_and_never_silently_retries(self):
        _,path,auth=self.prepare();out=self.root/"failed";calls=[0]
        def failure(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:raise FloatingPointError("fixture kernel failure")
            return self.fake_estimator(*args,**kwargs)
        with self.assertRaisesRegex(FloatingPointError,"fixture"):self.run_fake(path,auth,out,failure)
        with sqlite3.connect(out/"journal.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT status FROM rows WHERE id=0").fetchone()[0],"failed")
        with self.assertRaisesRegex(ValueError,"failed row retained"):self.run_fake(path,auth,out)

    def test_negative_variance_and_nonpsd_covariance_reject(self):
        value=self.fake_estimator(None,None,[0]*2);value["moments"][1]["bank_covariance"]=[[-1.,0.],[0.,1.]]
        with self.assertRaisesRegex(ValueError,"negative"):bank.quality(value,np.ones(2),self.policy)
        value["moments"][1]["bank_covariance"]=[[1.,2.],[2.,1.]]
        with self.assertRaisesRegex(ValueError,"PSD"):bank.quality(value,np.ones(2),self.policy)

    def test_changed_source_or_tranche_refuses(self):
        plan,_,_=self.prepare();bad=copy.deepcopy(plan);bad["jobs"][0]["stop"]=3
        with self.assertRaisesRegex(ValueError,"map"):bank.validate(bad)
        bad=copy.deepcopy(plan);bad["source_bindings"]["actual_bank"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"source"):bank.validate(bad)

    def test_relaxed_joint_clean_target_uses_base_h_and_independent_c_target(self):
        policy={"format":bank.RELAXED_POLICY,"grid_multiplier":4,"paths_ladder":[262144],
            "preferred_mcse_over_h":.25,"hardcap_mcse_over_h":.5,
            "refinement_requirement":"record_only_not_certified","price_target":"finite_hybrid_grid4_expectation"}
        path=self.root/"clean-plan.json"
        plan=bank.prepare(self.parent,path,root_seed=20261111,target_splits={"train":300000,"validation":5000,"test":7000},
            active_ranges={"supplied_known":{},"joint_unknown":{"train":[0,1]}},rows_per_shard=1,
            numerical_policy=policy,row_wall_seconds=30,shard_wall_seconds=60,concurrent_workers=128,
            total_cpu_seconds=60,candidate_only=True,memory_gib_per_worker=8,clean_only=True,inferred_c_blueprint=True)
        review=self.root/"pilot.json";review.write_text('{"strict_old_gate_passed":false}\n')
        auth=self.root/"clean-auth.json";record={"format":bank.ADMISSION,"bank_plan_sha256":bank.pilot.digest(path),
            "generation_authorized":True,"mode":"candidate_only","pilot_numerical_review_passed":False,
            "label_policy_authorized":True,"pilot_execution_reviewed":True,"numerical_policy":policy,
            "scientific_acceptance":False,"numerical_review":bank.pilot.binding(review)};auth.write_text(json.dumps(record))
        task=plan["tasks"]["joint_unknown"];lookup={(c["maturity"],c["strike"],c["option_type"]):h for c,h in zip(task["design"]["contracts"],task["design"]["half_spread_forward"])}
        calls=[]
        def fine(parameters,curve,contracts,**kwargs):
            calls.append(kwargs);h=np.asarray([lookup[(c["maturity"],c["strike"],c["option_type"])] for c in contracts]);cov=np.diag((.4*h)**2)
            kwargs["diagnostics"]["raw_control"]={"raw_put_mean":np.full(len(h),.1)}
            return np.full(len(h),.1),np.sqrt(np.diag(cov)),cov
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"1"}),patch.object(bank.pilot.runtime_pricing,"rough_block",side_effect=fine):
            result=bank.run_shard(path,auth,0,self.root/"clean",process_started=time.monotonic())
        self.assertEqual(len(calls),7);self.assertTrue(all(c["n_pairs"]==131072 for c in calls))
        self.assertEqual(result["numerical_qualified_rows"],1);self.assertFalse(result["training_usable"])
        self.assertFalse(result["continuous_bias_certified"]);self.assertEqual(plan["inference_blueprint"]["parameter_names"][-1],"c")
        with np.load(self.root/"clean/data.npz",allow_pickle=False) as arrays:
            self.assertEqual(arrays["physical_parameters"].shape,(1,6));self.assertEqual(arrays["inference_parameters"].shape,(1,7))
            self.assertEqual(arrays["parameters"].shape,(1,7));c=arrays["noise_scale"][:,None]
            self.assertTrue(np.all((c>.5)&(c<2)))
            np.testing.assert_allclose(arrays["observations_forward"],arrays["reference_prices_forward"]+c*np.asarray(task["design"]["half_spread_forward"])*arrays["quote_standard_normals"])
            self.assertFalse(plan["inference_blueprint"]["c_role"].startswith("conditioning"))
            self.assertFalse(arrays["refinement_evaluated"].any());self.assertTrue(np.isnan(arrays["refinement_difference_forward"]).all())
        record["label_policy_authorized"]=False;auth.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"authorization"):bank.validate_admission(path,plan,auth)


if __name__=="__main__":unittest.main()
