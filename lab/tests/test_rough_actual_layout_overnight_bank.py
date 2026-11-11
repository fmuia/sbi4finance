"""Fake-kernel resilience, early export and grouped-child isolation; no MC."""
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from sbi4finance.calibration import rough_actual_layout_overnight_bank as bank
from sbi4finance.calibration import rough_actual_layout_old_budget_bank as previous
from rough_actual_layout_fixtures import current_portable


class OvernightBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        _,self.parent=current_portable(self.root)
        old=bank.pilot.read(bank.ROOT/"lab/evidence/pilot-results/2026-11-11/bergomi-observation-refresh/joint-legacy-budget-1000/plan.json")
        self.policy=old["numerical_policy"];self.old=old

    def prepare(self,count=3,tag="one"):
        path=self.root/(tag+".json")
        plan=bank.prepare(self.parent,path,root_seed=20261111,target_splits={"train":300000,"validation":5000,"test":7000},
            active_ranges={"supplied_known":{},"joint_unknown":{"train":[1000,1000+count]}},rows_per_shard=3,
            numerical_policy=self.policy,row_wall_seconds=10,shard_wall_seconds=60,concurrent_workers=8,
            total_cpu_seconds=((count+23)//24)*8*60,candidate_only=True,memory_gib_per_worker=8,clean_only=True,inferred_c_blueprint=True)
        review=self.root/(tag+"-review.json");review.write_text('{"legacy_budget_only":true}\n')
        auth=self.root/(tag+"-auth.json");auth.write_text(json.dumps({"format":bank.ADMISSION,"bank_plan_sha256":bank.pilot.digest(path),
            "generation_authorized":True,"mode":"candidate_only","label_policy_authorized":True,"pilot_execution_reviewed":True,
            "pilot_numerical_review_passed":False,"numerical_policy":self.policy,"scientific_acceptance":False,"numerical_review":bank.pilot.binding(review)}))
        return plan,path,auth

    @staticmethod
    def fine(parameters,curve,contracts,**kwargs):
        n=len(contracts);kwargs["diagnostics"]["raw_control"]={"raw_put_mean":np.full(n,.1)}
        return np.full(n,.1),np.zeros(n),np.zeros((n,n))

    def test_row_failure_continues_early_export_resume_and_prior_identity(self):
        plan,path,auth=self.prepare();out=self.root/"shard";calls=[];early=[]
        def failing(*args,**kwargs):
            calls.append(1)
            if len(calls)==8:
                early.append(bank.export_completed(path,out/"journal.sqlite",self.root/"early"))
                raise FloatingPointError("one deliberately failed row")
            return self.fine(*args,**kwargs)
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"1"}),patch.object(bank.pilot.runtime_pricing,"rough_block",side_effect=failing):
            result=bank.run_shard(path,auth,0,out,process_started=time.monotonic())
        self.assertEqual(len(calls),15);self.assertEqual(result["simulation_rows_complete"],2)
        self.assertEqual(result["failed_or_missing_row_ids"],[1001]);self.assertEqual(early[0]["exported_row_ids"],[1000])
        self.assertTrue(early[0]["availability_selection"]);self.assertFalse(early[0]["precision_filter_applied"])
        with np.load(out/"data.npz",allow_pickle=False) as data:
            self.assertEqual(data["valid_row_mask"].tolist(),[True,False,True]);self.assertTrue(np.isnan(data["reference_prices_forward"][1]).all())
            self.assertEqual(data["physical_parameters"].shape,(3,6));self.assertEqual(data["parameters"].shape,(3,7))
            for i,row in enumerate((1000,1001,1002)):
                spec=previous.row_spec(self.old,"joint_unknown","train",row)
                np.testing.assert_array_equal(data["physical_parameters"][i],spec["parameters"])
                self.assertEqual(float(data["noise_scale"][i]),spec["noise_scale"])
        def no_reprice(*args,**kwargs):raise AssertionError("completed or failed row was automatically retried")
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"1"}),patch.object(bank.pilot.runtime_pricing,"rough_block",side_effect=no_reprice):
            self.assertEqual(result,bank.run_shard(path,auth,0,out,process_started=time.monotonic()))
        retry=self.root/"retry";retrycalls=[]
        def repaired(*args,**kwargs):retrycalls.append(1);return self.fine(*args,**kwargs)
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture-retry","SLURM_CPUS_PER_TASK":"1"}),patch.object(bank.pilot.runtime_pricing,"rough_block",side_effect=repaired):
            fixed=bank.run_shard(path,auth,0,retry,process_started=time.monotonic(),retry_failed=True,retry_reason="explicit fixture repair",resume_from=out/"journal.sqlite")
        self.assertEqual(len(retrycalls),7);self.assertEqual(fixed["simulation_rows_complete"],3)
        with sqlite3.connect(retry/"journal.sqlite") as db:self.assertEqual(db.execute("SELECT COUNT(*) FROM retry_events").fetchone()[0],1)

    def test_group_eight_cpu_truth_and_one_dead_child_does_not_cancel_siblings(self):
        plan,path,auth=self.prepare(24,"eight");created=[]
        class Child:
            def __init__(self,command):self.command=command;self.index=len(created);self.done=False;self.terminated=False;created.append(self)
            def wait(self):self.done=True;return -9 if self.index==2 else 0
            def poll(self):return (-9 if self.index==2 else 0) if self.done else None
            def terminate(self):self.terminated=True
        with patch.dict("os.environ",{"SLURM_JOB_ID":"fixture","SLURM_CPUS_PER_TASK":"8"}),patch.object(bank.subprocess,"Popen",side_effect=Child):
            result=bank.run_allocation(path,auth,0,self.root/"group")
            bank.check_allocation(plan,7,0)
            with self.assertRaises(ValueError):bank.check_allocation(plan,7,None)
        self.assertEqual(len(created),8);self.assertTrue(all(c.done and not c.terminated for c in created))
        self.assertEqual([c["returncode"] for c in result["children"]],[0,0,-9,0,0,0,0,0])
        self.assertTrue(all("--allocation-group" in c.command for c in created))


if __name__=="__main__":unittest.main()
