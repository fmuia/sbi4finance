"""Fixed observations, joint-target identity and bounded consumer contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_heston_spread_reference as producer
from scripts import run_heston_spread_scale_reference as consumer


class ScaleReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.panel = self.root/"panel.json"
        self.design, self.probe, self.history = (self.root/name for name in ("design.json", "probe.json", "historical.json"))
        records = {"forward": [100.]*54, "strike": np.linspace(85.,115.,54).tolist(),
            "discount": [.99]*54, "maturity": [.25]*54,
            "option_type": ["put"]*27+["call"]*27, "mid": [1.]*54, "half_spread": [.1]*54}
        truth = [.04, 2., .05, .4, -.7]
        producer.write_json(self.design, {"records": records, "products": {"control": [0.,.1], "affected": [.3,.9]}})
        producer.write_json(self.probe, {"cases": [{"validation_row": 0, "physical_parameters": truth}]})
        names = producer.HestonSpreadPilot(records).names
        producer.write_json(self.history, {"cases": {"full-wide-xi-extended": {
            "run": {"design_sha256": producer.digest(self.design)},
            "posterior": {"parameter_quantiles_05_50_95": {name: [value]*3 for name,value in zip(names,truth)}}}}})
        with patch.object(producer,"source_hashes",return_value={}), patch.object(producer,"price_audit",
                return_value=(np.full(54,.01),{"pointwise_gates_passed":True})):
            producer.prepare_panel(self.design,self.probe,self.history,self.panel,seed=19)

    def test_all_scale_targets_preserve_exact_observations_and_only_joint_adds_c(self):
        specs = [consumer.job_spec(self.panel,"synthetic-broad",mode) for mode in ("half","double","joint")]
        self.assertEqual(len({spec["observation_sha256"] for spec in specs}),1)
        self.assertTrue(all(spec["generating_c"]==1. for spec in specs))
        self.assertEqual([len(spec["names"]) for spec in specs],[5,5,6])
        self.assertEqual(specs[-1]["names"][-1],"c")
        self.assertEqual(specs[-1]["bounds"][-1],[.5,2.])
        self.assertEqual(specs[0]["task"]["fixed_c"],.5)
        self.assertEqual(specs[1]["task"]["fixed_c"],2.)
        self.assertIsNone(specs[2]["task"]["fixed_c"])
        self.assertTrue(all(spec["task"]["pricing_controls"]["c_min"]==.5 for spec in specs))
        for case,mode in (("market","half"),("market","double"),("synthetic-boundary","joint")):
            with self.assertRaisesRegex(ValueError,"bounded scale matrix"):
                consumer.job_spec(self.panel,case,mode)

    def test_target_prior_data_and_runner_flag_substitution_are_rejected(self):
        good = consumer.job_spec(self.panel,"synthetic-market-like","joint")
        consumer.validate_job(good,self.panel)
        for field in ("observation","bounds","names","runner"):
            bad=copy.deepcopy(good)
            if field=="observation": bad["observation_sha256"]="0"*64
            elif field=="bounds": bad["bounds"][-1][1]=3.
            elif field=="names": bad["names"][-1]="tau"
            else: bad["runner_kwargs"]["infer_c"]=False
            with self.assertRaisesRegex(ValueError,"declaration"):
                consumer.validate_job(bad,self.panel)

    def test_fixed_phi_oracle_uses_saved_prices_and_is_explicitly_not_full_joint(self):
        with patch.object(producer.HestonSpreadPilot,"prices",side_effect=AssertionError("no pricing during oracle preparation")):
            oracle=consumer.fixed_phi_oracle(self.panel,"synthetic-broad")
        panel=json.loads(self.panel.read_text())
        residual=np.asarray(panel["cases"][0]["standard_errors"])
        self.assertAlmostEqual(oracle["mode"],np.clip(np.sqrt(np.mean(residual**2)),.5,2.))
        self.assertAlmostEqual(oracle["density_integral"],1.,places=8)
        self.assertEqual(oracle["price_calls"],0)
        self.assertIn("Not the full joint",oracle["scope"])
        self.assertTrue(.5<oracle["quantiles_05_50_95"][0]<oracle["quantiles_05_50_95"][2]<2.)
        with self.assertRaisesRegex(ValueError,"unavailable"):
            consumer.fixed_phi_oracle(self.panel,"market")

    def test_protocol_declares_exact_seven_job_matrix_and_distinct_inner_outer_limits(self):
        out=self.root/"protocol.json"
        with patch.object(consumer,"reused_c1_reference",side_effect=lambda panel,refs,case:{"case_id":case}), patch.object(
                consumer,"source_hashes",return_value={}), patch.object(consumer,"fixed_phi_oracle",return_value={"scope":"fixed phi only"}):
            protocol=consumer.prepare_protocol(self.panel,self.root/"old-refs",self.root/"new-evidence",self.root/"new-payloads",out)
        self.assertEqual(len(protocol["jobs"]),7)
        self.assertEqual([(job["case_id"],job["mode"]) for job in protocol["jobs"]],list(consumer.CASE_MODES))
        self.assertEqual(protocol["settings"]["inner_cpu_and_wall_seconds"],900)
        self.assertEqual(protocol["settings"]["outer_total_wall_seconds"],950)
        self.assertEqual(protocol["campaign_limits"]["maximum_sum_inner_cpu_seconds"],6300)
        self.assertEqual(protocol["synthetic_observations_generated"],0)
        with self.assertRaises(FileExistsError):
            consumer.prepare_protocol(self.panel,self.root/"old-refs",self.root/"new-evidence",self.root/"new-payloads",out)

    def test_one_job_passes_exact_budget_and_scale_to_unchanged_runner_without_retry(self):
        spec=consumer.job_spec(self.panel,"synthetic-broad","half")
        spec.update(evidence_out=str(self.root/"out"/spec["id"]),payload_out=str(self.root/"payload"/spec["id"]))
        protocol={"panel_path":str(self.panel),"evidence_root":str(self.root/"out"),"source_sha256":{}}
        protocol_path=self.root/"protocol.json"
        producer.write_json(protocol_path,protocol)
        with patch.object(consumer,"validate_protocol",return_value=spec), patch.object(consumer,"source_hashes",return_value={}), patch.object(
                producer,"run_reference",return_value={}) as run, patch.object(consumer,"summarize_run",return_value={"reference_ready":False}):
            result=consumer.run_one(protocol_path,spec["id"])
            self.assertFalse(result["result"]["reference_ready"])
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.kwargs["budget_seconds"],900)
            self.assertEqual(run.call_args.kwargs["c"],.5)
            self.assertFalse(run.call_args.kwargs["infer_c"])
            with self.assertRaises(FileExistsError):
                consumer.run_one(protocol_path,spec["id"])
            self.assertEqual(run.call_count,1)


if __name__=="__main__":
    unittest.main()
