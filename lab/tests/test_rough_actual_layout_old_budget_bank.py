"""Fake-kernel tests for an explicitly approximate old-cost numerical budget."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from sbi4finance.calibration import rough_actual_layout_old_budget_bank as bank


class OldBudgetBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        parent=bank.ROOT/"lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable/work-order.json"
        value=bank.pilot.read(parent);old=value["tasks"]["joint_unknown"]["base_task"]
        sigma=old["context"]["quote_noise_std"]
        self.policy={"format":bank.RELAXED_POLICY,"grid_multiplier":1,"paths_ladder":old["numerics"]["paths_ladder"],
            "numerical_reference_forward":float(np.median(sigma)),"numerical_reference_scope":"historical_median_not_observation_noise",
            "historical_sigma_vector_sha256":bank.fingerprint(sigma),"preferred_mcse_over_reference_sigma":.25,
            "hardcap_mcse_over_reference_sigma":.5,"refinement_requirement":"record_only_not_certified","price_target":"finite_hybrid_base_grid_expectation"}
        self.path=self.root/"plan.json"
        self.plan=bank.prepare(parent,self.path,root_seed=20261111,target_splits={"train":300000,"validation":5000,"test":7000},
            active_ranges={"supplied_known":{},"joint_unknown":{"train":[0,1]}},rows_per_shard=1,numerical_policy=self.policy,
            row_wall_seconds=60,shard_wall_seconds=90,concurrent_workers=192,total_cpu_seconds=90,
            candidate_only=True,memory_gib_per_worker=8,clean_only=True,inferred_c_blueprint=True)
        review=self.root/"review.json";review.write_text('{"continuous_bias_certified":false}\n')
        self.auth=self.root/"auth.json";self.auth.write_text(json.dumps({"format":bank.ADMISSION,"bank_plan_sha256":bank.pilot.digest(self.path),
            "generation_authorized":True,"mode":"candidate_only","label_policy_authorized":True,"pilot_execution_reviewed":True,
            "pilot_numerical_review_passed":False,"numerical_policy":self.policy,"scientific_acceptance":False,
            "numerical_review":bank.pilot.binding(review)}))

    def test_old_budget_is_separate_from_actual_observation_halfspreads(self):
        bank.validate(self.plan)
        self.assertAlmostEqual(self.policy["numerical_reference_forward"],.00044092792100661605)
        self.assertEqual(self.policy["paths_ladder"],[32768,65536,131072,262144,524288,1048576,2097152])
        self.assertEqual(self.plan["inference_blueprint"]["parameter_names"][-1],"c")
        self.assertFalse(self.plan["SBI_training_readiness"])

    def execute(self,relative_error):
        calls=[];ref=self.policy["numerical_reference_forward"]
        def fake(parameters,curve,contracts,**kwargs):
            calls.append(kwargs);n=len(contracts);cov=np.eye(n)*(relative_error(kwargs["n_pairs"])*ref)**2
            kwargs["diagnostics"]["raw_control"]={"raw_put_mean":np.full(n,.1)}
            return np.full(n,.1),np.sqrt(np.diag(cov)),cov
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"1"}),patch.object(bank.pilot.runtime_pricing,"rough_block",side_effect=fake):
            result=bank.run_shard(self.path,self.auth,0,self.root/"out",process_started=time.monotonic())
        return result,calls

    def test_base_grid_adaptive_preferred_stop_and_joint_c_law(self):
        result,calls=self.execute(lambda pairs:.3 if pairs==16384 else .2)
        self.assertEqual(len(calls),14)
        expected=[b["n_steps"] for b in self.plan["tasks"]["joint_unknown"]["blocks"]]
        self.assertEqual([c["n_steps"] for c in calls],sum(([n,n] for n in expected),[]))
        self.assertFalse(result["training_usable"])
        with np.load(self.root/"out/data.npz",allow_pickle=False) as a:
            self.assertEqual(a["physical_parameters"].shape,(1,6));self.assertEqual(a["parameters"].shape,(1,7))
            h=np.asarray(self.plan["tasks"]["joint_unknown"]["design"]["half_spread_forward"])
            np.testing.assert_allclose(a["observations_forward"],a["reference_prices_forward"]+a["noise_scale"][:,None]*h*a["quote_standard_normals"])
            self.assertTrue(np.all(a["pricing_paths"]==65536));self.assertFalse(a["refinement_evaluated"].any())

    def test_unqualified_final_estimates_are_retained_without_prior_filtering(self):
        result,calls=self.execute(lambda pairs:1.)
        self.assertEqual(len(calls),49);self.assertEqual(result["physical_rows"],1)
        self.assertEqual(result["numerical_qualified_rows"],0);self.assertFalse(result["training_usable"])
        self.assertFalse(result["prior_filtering_permitted"])


if __name__=="__main__":unittest.main()
