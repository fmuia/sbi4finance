"""Mocked local repair of both immutable Oscar producer families; no MC."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sbi4finance.calibration import rough_actual_layout_local_repair as repair


class LocalRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.bundle=self.root/"bundle.sqlite";self.manifest=self.root/"manifest.json";self.plan=self.root/"repair-plan.json"
        metadata=self.root/"metadata";metadata.mkdir();db=sqlite3.connect(self.bundle)
        db.executescript("CREATE TABLE identity(source_key,value); CREATE TABLE draws(source_key,id,value); CREATE TABLE attempts(source_key,id,block,rung,status,summary,payload); CREATE TABLE rows(source_key,id,status,summary,payload); CREATE TABLE retry_events(source_key,event,id,previous_summary,reason,created_utc);")
        manifest={"format":repair.BUNDLE_FORMAT,"producers":{},"rows":[]};self.specs=[]
        evidence=repair.ROOT/"lab/evidence/pilot-results/2026-11-11/bergomi-observation-refresh"
        for kind,row,folder,authname,lane in (("legacy1000",440,"joint-legacy-budget-1000","admission-v2.json","initial1000"),("overnight",1000,"joint-legacy-budget-300k","admission.json","master")):
            original=repair.bank.pilot.read(evidence/folder/"plan.json")
            refs={}
            for name,source in (("plan",evidence/folder/"plan.json"),("admission",evidence/folder/"remote-execution"/authname)):
                destination=metadata/f"{kind}-{name}.json";shutil.copyfile(source,destination)
                refs[name]={"path":str(destination.relative_to(self.root)),"sha256":repair.bank.pilot.digest(destination)}
            refs["source_bindings"]=original["source_bindings"];manifest["producers"][kind]=refs
            job=next(j for j in original["jobs"] if j["start"]<=row<j["stop"])
            identity={"plan_sha256":refs["plan"]["sha256"],"admission_sha256":refs["admission"]["sha256"],"job":job,"sources":original["source_bindings"],"versions":original["versions"]}
            spec=repair.bank.row_spec(original,"joint_unknown","train",row)
            spec["parameters"][0]=float(np.nextafter(spec["parameters"][0],np.inf))
            self.specs.append(spec)  # Explicit x86/ARM one-ULP affine replay fixture.
            key=f"{lane}:{job['index']:06d}"
            db.execute("INSERT INTO identity VALUES(?,?)",(key,json.dumps(identity)))
            db.execute("INSERT INTO draws VALUES(?,?,?)",(key,row,json.dumps(spec)))
            db.execute("INSERT INTO rows VALUES(?,?,?,?,NULL)",(key,row,"failed",json.dumps({"error_type":"TimeoutError","error":"reference budget/interruption signal 14","spec":spec})))
            for block in range(7):
                n=9;cov=np.zeros((n,n));status="accepted" if block<6 else "rejected_precision"
                arrays={"controlled_prices":np.full((2,n),.01*(block+1)),"raw_prices":np.full((2,n),.01*(block+1)),"difference":np.full(n,np.nan),"difference_covariance":np.full((n,n),np.nan),"bank_covariance":np.stack([cov,cov])}
                summary={"passed":block<6,"paths":32768,"entropy":[*spec["identity_entropy"],1,1,block,0]}
                db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?)",(key,row,block,0,status,json.dumps(summary),repair.bank.pack(arrays)))
            manifest["rows"].append({"lane":lane,"bank_lane":"joint_unknown","split":"train","id":row,"producer_kind":kind,"source_key":key,"identity":identity,
                "plan_sha256":refs["plan"]["sha256"],"admission_sha256":refs["admission"]["sha256"],"bundle":{"path":"bundle.sqlite","sha256":"pending"}})
        db.commit();db.close()
        digest=repair.bank.pilot.digest(self.bundle)
        for row in manifest["rows"]:row["bundle"]["sha256"]=digest
        self.manifest.write_text(json.dumps(manifest));self.value=repair.prepare(self.manifest,self.plan,expected_rows=2)

    def test_both_producers_exact_rng_cache_reuse_and_local_identity(self):
        before={p:repair.bank.pilot.digest(p) for p in (self.manifest,self.bundle)}
        calls=[]
        def fake(parameters,curve,contracts,**kwargs):
            calls.append(kwargs);n=len(contracts);kwargs["diagnostics"]["raw_control"]={"raw_put_mean":np.full(n,.1)}
            return np.full(n,.1),np.zeros(n),np.zeros((n,n))
        environment={k:v for k,v in os.environ.items() if k!="SLURM_JOB_ID"}
        with patch.dict(os.environ,environment,clear=True),patch.object(repair.os,"getpriority",return_value=10),patch.object(repair.bank.pilot.runtime_pricing,"rough_block",side_effect=fake):
            for i in range(2):
                out=self.root/f"repair-{i}";result=repair.run_row(self.plan,i,out)
                self.assertEqual(result["status"],"complete");self.assertEqual(result["identity"]["runtime"]["execution_mode"],"local_no_slurm")
                self.assertEqual(result["identity"]["original_failure"]["error_type"],"TimeoutError")
                with np.load(out/"data.npz",allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["physical_parameters"],self.specs[i]["parameters"])
                    self.assertEqual(float(data["noise_scale"]),self.specs[i]["noise_scale"])
                    self.assertEqual(data["parameters"].shape,(7,));self.assertEqual(data["block_origin"].tolist(),["oscar_reused"]*6+["local_recomputed"])
                    for block in range(6):np.testing.assert_array_equal(data["reference_prices_forward"][block*9:(block+1)*9],np.full(9,.01*(block+1)))
                    h=np.asarray(repair.bank.pilot.read(self.root/"metadata"/f"{self.value['rows'][i]['producer_kind']}-plan.json")["tasks"]["joint_unknown"]["design"]["half_spread_forward"])
                    np.testing.assert_array_equal(data["observations_forward"],data["reference_prices_forward"]+data["noise_scale"]*h*data["quote_standard_normals"])
                self.assertEqual(repair.run_row(self.plan,i,out),result)
        self.assertEqual(len(calls),2)
        for i,call in enumerate(calls):
            self.assertEqual(call["seed"].entropy,[*self.specs[i]["identity_entropy"],1,1,6,1]);self.assertEqual(call["n_pairs"],32768)
        self.assertEqual(before,{p:repair.bank.pilot.digest(p) for p in before})
        self.assertEqual(self.value["cross_platform_prior_replay"]["max_ulp"],1.)

    def test_prior_replay_allowance_is_narrow_and_c_entropy_remain_exact(self):
        actual=self.specs[0];expected=json.loads(json.dumps(actual));expected["parameters"][0]=float(np.nextafter(actual["parameters"][0],-np.inf))
        failure={"spec":actual}
        self.assertEqual(repair.verify_stored_spec(actual,expected,failure)["parameters_max_ulp"],1.)
        invalid=json.loads(json.dumps(actual))
        for _ in range(4):invalid["parameters"][0]=float(np.nextafter(invalid["parameters"][0],np.inf))
        with self.assertRaisesRegex(ValueError,"two ULP"):repair.verify_stored_spec(invalid,expected,{"spec":invalid})
        for key in ("noise_scale","identity_entropy"):
            invalid=json.loads(json.dumps(actual))
            if key=="noise_scale":invalid[key]=float(np.nextafter(invalid[key],np.inf))
            else:invalid[key][-1]+=1
            with self.assertRaisesRegex(ValueError,"RNG identity"):repair.verify_stored_spec(invalid,expected,{"spec":invalid})
        with self.assertRaisesRegex(ValueError,"original failure"):repair.verify_stored_spec(actual,expected,{"spec":expected})

    def test_transport_source_and_duplicate_join_refused(self):
        real=repair.bank.pilot.digest
        def changed(path):return "changed" if str(path).endswith("rough_actual_layout_local_repair.py") else real(path)
        with patch.object(repair.bank.pilot,"digest",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"source changed"):repair.validate(self.value)
        value=json.loads(self.manifest.read_text());value["rows"][1]=value["rows"][0]
        other=self.root/"duplicate.json";other.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"duplicate"):repair.prepare(other,self.root/"rejected.json",expected_rows=2)
        with sqlite3.connect(self.bundle) as db:db.execute("UPDATE rows SET status='changed'");db.commit()
        with self.assertRaisesRegex(ValueError,"transport changed"):repair.validate(self.value)


if __name__=="__main__":unittest.main()
