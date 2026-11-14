"""Identity, observation and bounded-execution contracts of spread references."""
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
import xarray as xr

from scripts import run_heston_spread_reference as runner


PHI = np.array([.04, 2., .05, .4, -.7])


class SpreadReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.design = self.root/"design.json"
        self.probe = self.root/"probe.json"
        self.historical = self.root/"historical.json"
        self.panel = self.root/"panel.json"
        records = {"forward": [100.]*54, "strike": np.linspace(85,115,54).tolist(),
            "discount": [.99]*54, "maturity": [.25]*54, "option_type": ["put"]*27+["call"]*27,
            "mid": [1.]*54, "half_spread": [.1]*54}
        runner.write_json(self.design, {"records": records, "products": {"control": [0.,.1], "affected": [.3,.9]}})
        runner.write_json(self.probe, {"cases": [{"validation_row":0, "physical_parameters": PHI.tolist()}]})
        names = runner.HestonSpreadPilot(records).names
        runner.write_json(self.historical, {"cases": {"full-wide-xi-extended": {
            "run": {"design_sha256": runner.digest(self.design)},
            "posterior": {"parameter_quantiles_05_50_95": {name:[float(v)]*3 for name,v in zip(names,PHI)}}}}})
        with patch.object(runner, "source_hashes", return_value={}), patch.object(runner, "price_audit",
                return_value=(np.full(54,.01), {"pointwise_gates_passed": True})):
            runner.prepare_panel(self.design,self.probe,self.historical,self.panel,seed=19)

    def read_panel(self):
        return json.loads(self.panel.read_text())

    def test_panel_preserves_declared_truths_and_same_c1_observations_across_targets(self):
        panel, case, fixed, observed = runner.load_case(self.panel,"synthetic-broad")
        _, _, joint, joint_observed = runner.load_case(self.panel,"synthetic-broad",infer_c=True)
        self.assertEqual([c["id"] for c in panel["cases"]],list(runner.CASE_IDS))
        self.assertFalse(panel["reserved_test_arrays_opened"])
        self.assertTrue(np.any(observed < 0))
        self.assertTrue(np.array_equal(observed,joint_observed))
        self.assertEqual(case["generating_c"],1.)
        self.assertEqual(len(fixed.names),5)
        self.assertEqual(joint.names[-1],"c")
        self.assertEqual(case["observations_sha256"],runner.values_digest(observed))
        _, market, _, mids = runner.load_case(self.panel,"market")
        self.assertIsNone(market["physical_truth"])
        assert_allclose(mids,1.)

    def test_changed_observations_truth_noise_or_design_are_rejected(self):
        original = self.read_panel()
        for change in ("observation","truth","noise","design"):
            with self.subTest(change=change):
                panel = json.loads(json.dumps(original))
                if change == "observation":
                    panel["cases"][0]["observations"][0] += .01
                elif change == "truth":
                    panel["cases"][0]["physical_truth"][0] += .001
                elif change == "noise":
                    panel["cases"][0]["generating_c"] = 2.
                else:
                    panel["design_sha256"] = "0"*64
                runner.write_json(self.panel,panel)
                with self.assertRaises(ValueError):
                    runner.load_case(self.panel,"synthetic-broad")

    def test_boundary_price_failure_is_retained_without_substituting_a_case(self):
        destination = self.root/"failed-panel.json"
        calls = []
        def pricing(pilot, truth):
            calls.append(np.asarray(truth).copy())
            return np.full(54,.01), {"pointwise_gates_passed": len(calls)<3}
        with patch.object(runner,"source_hashes",return_value={}), patch.object(runner,"price_audit",side_effect=pricing):
            with self.assertRaisesRegex(FloatingPointError,"synthetic-boundary"):
                runner.prepare_panel(self.design,self.probe,self.historical,destination)
        self.assertFalse(destination.exists())
        self.assertTrue(np.array_equal(calls[-1],runner.BOUNDARY_TRUTH))
        failure = json.loads(destination.with_suffix(".failure.json").read_text())
        self.assertEqual(failure["completed_cases"],list(runner.CASE_IDS[:2]))
        self.assertTrue(destination.with_suffix(".declaration.json").exists())

    def test_variance_products_keep_axes_and_match_direct_time_integrals(self):
        draws = np.broadcast_to(PHI,(2,3,5)).copy()
        draws[1,:,0] = .08
        interval = (.3,.9)
        values = runner.expected_variance(draws,interval)
        self.assertEqual(values.shape,(2,3))
        for chain in range(2):
            row = draws[chain,0]
            direct = quad(lambda t: row[2]+(row[0]-row[2])*np.exp(-row[1]*t),*interval)[0]/np.diff(interval)[0]
            self.assertAlmostEqual(values[chain,0],direct,places=14)
        products = runner.product_values(draws,{"near":(0.,.1),"forward":interval})
        self.assertEqual(products.shape,(2,3,2))
        assert_allclose(products[...,1],values)

    def test_budget_installs_resource_limit_and_restores_it_after_failure(self):
        previous = resource.getrlimit(resource.RLIMIT_CPU)
        with self.assertRaisesRegex(TimeoutError,"budget"):
            with runner.Budget(2.) as budget:
                self.assertLessEqual(resource.getrlimit(resource.RLIMIT_CPU)[0],np.ceil(time.process_time()+2.))
                budget.cpu_start = time.process_time()-3.
                budget.check()
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CPU),previous)

    @contextmanager
    def fake_sampling(self, diagnostic_passes):
        calls = []
        def sample(log_likelihood,bounds,names,**settings):
            calls.append(settings)
            draws = np.broadcast_to(PHI,(4,settings["draws"],len(names))).copy()
            return xr.DataTree.from_dict({"posterior":xr.Dataset({"parameters":(("chain","draw","parameter"),draws)},
                coords={"parameter":names}),"sample_stats":xr.Dataset()})
        def diagnostics(*args):
            passed = next(diagnostic_passes)
            return {"parameters":{},"products":{"parameters":{}},"mcmc_diagnostics_passed":passed}
        with patch.object(runner,"source_hashes",return_value={}), patch.object(runner.reference,"data_fit_initialization",return_value={}), patch.object(
                runner.reference,"sample_deterministic_reference",side_effect=sample), patch.object(runner,"diagnose",side_effect=diagnostics), patch.object(
                runner,"price_audit",return_value=(np.ones(54),{"pointwise_gates_passed":True})):
            yield calls

    def test_smoke_is_never_reference_acceptance_and_payload_binding_is_preserved(self):
        with self.fake_sampling(iter([True])) as calls:
            result = runner.run_reference(self.panel,"synthetic-broad",self.root/"evidence",self.root/"payload",smoke=True,budget_seconds=30)
        self.assertFalse(result["reference_ready"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertTrue(result["smoke"])
        self.assertEqual(calls[0]["cores"],1)
        self.assertEqual(calls[0]["chains"],4)
        diagnostics = json.loads((self.root/"evidence/diagnostics.json").read_text())
        self.assertEqual(diagnostics["chain_payload_sha256"],runner.digest(result["chain_payload_path"]))
        self.assertEqual(result["observation_sha256"],self.read_panel()["cases"][0]["observations_sha256"])

    def test_failed_convergence_gets_only_one_escalation(self):
        with self.fake_sampling(iter([False,False])) as calls:
            result = runner.run_reference(self.panel,"synthetic-broad",self.root/"evidence",self.root/"payload",budget_seconds=30)
        self.assertFalse(result["reference_ready"])
        self.assertEqual([(c["tune"],c["draws"]) for c in calls],list(runner.ATTEMPTS))
        self.assertEqual(len(result["attempts"]),2)
        self.assertTrue((self.root/"payload/attempt-0/chains.npz").exists())
        self.assertTrue((self.root/"payload/attempt-1/chains.npz").exists())


if __name__ == "__main__":
    unittest.main()
