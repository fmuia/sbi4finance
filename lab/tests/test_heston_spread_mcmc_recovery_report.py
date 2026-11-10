"""Selected-case plots preserve data matching and reference readiness."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import render_heston_spread_mcmc_recovery as report


class RecoveryReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.bounds=np.array([[.01,.16],[.1,10],[.01,.16],[.1,3],[-.95,-.1],[.5,2]])
        self.target={"parameter_names":list(report.NAMES),"prior_bounds":self.bounds.tolist(),"infer_c":True,"quote_count":54}

    def artifact(self,path):
        return {"path":str(path),"sha256":report.shared.digest(path)}

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return self.artifact(path)

    def fixture(self):
        truth=np.broadcast_to(self.bounds.mean(axis=1),(512,6)).copy()
        truth[:,-1]=.5+1.5*(np.arange(512)+.5)/512
        q=np.broadcast_to(self.bounds[:,0]+np.array([.05,.5,.95])[:,None,None]*np.diff(self.bounds,axis=1)[:,0],(3,512,6)).copy()
        networks=[]
        for batch in (128,512):
            path=self.root/f"nn{batch}"/"recovery.npz";path.parent.mkdir()
            np.savez(path,truth=truth,quantiles=q,row_id=np.array([f"validation:{i:08d}"for i in range(512)]))
            evaluation={"completed_bank_passes":500,"batch_size":batch,"task":self.target,"population":{
                "parameter_names":list(report.NAMES),"rows":512,"recovery_sha256":report.shared.digest(path),
                "truth_sha256":report.digest_values(truth),"observations_sha256":"a"*64}}
            networks.append({"batch_size":batch,"evaluation":self.write(f"nn{batch}/evaluation.json",evaluation),"recovery":self.artifact(path)})
        ranks=list(range(32,512,64));cases=[];observed=[]
        for index in ranks:
            y=np.arange(54,dtype=float)+index
            common={"id":f"case-{index}","row_id":f"validation:{index:08d}","observations_sha256":report.digest_values(y)}
            observed.append({**common,"observations":y.tolist()})
            cases.append({**common,"population_index":index,"c_rank":index,"truth":truth[index].tolist()})
        panel={"format":"heston_spread_recovery_observation_panel_v1","design_sha256":"d"*64,"joint_task":self.target,"cases":observed}
        panel_spec=self.write("panel.json",panel)
        source=self.root/"selection.py";source.write_text("# Test-only selection fixture, never executed.\n")
        selection={"format":"heston_spread_recovery_truth_selection_v1","panel":panel_spec,"parameter_names":list(report.NAMES),"prior_bounds":self.bounds.tolist(),"source":self.artifact(source),
            "networks":networks,"selection":{"population_rows":512,"replacement_of_failed_cases":False,"zero_based_c_ranks":ranks,"zero_based_population_indices":ranks},
            "population_truth_sha256":report.digest_values(truth),"population_observations_sha256":"a"*64,"cases":cases,"reserved_arrays_opened":False,"new_pricing_labels":False,"new_neural_sampling":False}
        protocol={"format":"heston_spread_recovery_reference_protocol_v1","truth_available_to_sampler":False,"panel_sha256":panel_spec["sha256"],"design_sha256":panel["design_sha256"],"settings":{"case_count":8,"chains":4,"attempts":[[2000,2000],[6000,12000]],"smoke_test_only":False,"automatic_retry_or_budget_extension":False}}
        manifest={"format":"heston_spread_mcmc_recovery_report_v1","protocol":self.write("protocol.json",protocol),"panel":panel_spec,"truth_selection":self.write("truth-selection.json",selection),"networks":networks,
            "references":[{"case_id":c["id"],"status":"failed","reason":"Deliberate unavailable-reference fixture"}for c in cases]}
        self.write("manifest.json",manifest)
        return manifest,selection,panel,protocol

    def test_all_selected_cases_remain_without_replacing_failed_references(self):
        self.fixture();data=report.load_data(self.root/"manifest.json")
        self.assertEqual(len(data["cases"]),8)
        self.assertEqual(int(data["ready"].sum()),0)
        self.assertTrue(all("Deliberate unavailable"in r["reason"]for r in data["references"]))
        self.assertEqual(data["quantiles"]["nn128"].shape,(3,8,6))

    def test_valid_hashes_cannot_hide_selection_changes_or_sampler_truth_leak(self):
        manifest,selection,panel,protocol=self.fixture()
        changed=copy.deepcopy(selection);changed["selection"]["zero_based_population_indices"][0]=33
        manifest["truth_selection"]=self.write("changed-selection.json",changed);self.write("changed.json",manifest)
        with self.assertRaisesRegex(ValueError,"rank selection"):
            report.load_data(self.root/"changed.json")
        panel["cases"][0]["truth"]=selection["cases"][0]["truth"]
        panel_spec=self.write("panel-leak.json",panel);selection["panel"]=panel_spec;protocol["panel_sha256"]=panel_spec["sha256"]
        manifest.update(panel=panel_spec,truth_selection=self.write("selection-leak.json",selection),protocol=self.write("protocol-leak.json",protocol));self.write("leak.json",manifest)
        with self.assertRaisesRegex(ValueError,"no-truth sampler boundary"):
            report.load_data(self.root/"leak.json")

    def test_neural_metrics_use_only_the_identical_ready_reference_subset(self):
        truth=np.broadcast_to(self.bounds.mean(axis=1),(8,6)).copy();truth[:,-1]=np.linspace(.6,1.9,8)
        good=np.stack([truth-.01,truth,truth+.01]);neural=good.copy();neural[:,2:,:]+=100
        data={"ready":np.array([True,True,*([False]*6)]),"truth":truth,"target":self.target,
            "quantiles":{"mcmc":good,"nn128":neural,"nn512":neural}}
        stats=report.statistics(data)
        self.assertEqual(stats["nn128"]["cases"],2)
        np.testing.assert_array_equal(stats["nn128"]["median_rmse"],np.zeros(6))
        self.assertIsNone(stats["prior"]["truth_median_correlation"][-1])

    def test_advertised_readiness_requires_passing_diagnostics(self):
        manifest,selection,_,_=self.fixture();case=selection["cases"][0]
        run={"format":"heston_spread_recovery_reference_run_v1","case_id":case["id"],"status":"completed","panel_sha256":manifest["panel"]["sha256"],"observation_sha256":case["observations_sha256"],"task":self.target,"names":list(report.NAMES),"bounds":self.bounds.tolist(),"truth_available_to_sampler":False,"settings":{"chains":4},"smoke":False,"reference_ready":True,"chain_payload_sha256":"f"*64}
        posterior={"parameter_quantiles_05_50_95":{name:[lo+.2*(hi-lo),lo+.5*(hi-lo),lo+.8*(hi-lo)]for name,(lo,hi)in zip(report.NAMES,self.bounds)}}
        rows={f"parameters[{n}]":{"r_hat":1.001,"ess_bulk":600,"ess_tail":700}for n in report.NAMES}
        diagnostics={"parameters":rows,"reference_ready":True,"parameter_mcmc_diagnostics_passed":True,"mcmc_diagnostics_passed":True,"pointwise_precision_passed":True,"chain_payload_sha256":"f"*64}
        spec={"case_id":case["id"],"run":self.write("run.json",run),"posterior":self.write("posterior.json",posterior),"diagnostics":self.write("diagnostics.json",diagnostics)}
        inputs=report.shared.Artifacts(self.root/"manifest.json")
        self.assertTrue(report.load_reference(inputs,spec,case,manifest["panel"],manifest["protocol"],self.target)["ready"])
        diagnostics["parameters"]["parameters[c]"]["ess_tail"]=399;spec["diagnostics"]=self.write("bad-diagnostics.json",diagnostics)
        with self.assertRaisesRegex(ValueError,"contradicts"):
            report.load_reference(inputs,spec,case,manifest["panel"],manifest["protocol"],self.target)
        run["reference_ready"]=False;spec["run"]=self.write("unready-run.json",run);diagnostics["reference_ready"]=False;spec["diagnostics"]=self.write("unready-diagnostics.json",diagnostics)
        result=report.load_reference(inputs,spec,case,manifest["panel"],manifest["protocol"],self.target)
        self.assertFalse(result["ready"])
        self.assertIsNone(result["quantiles"])


if __name__=="__main__":unittest.main()
