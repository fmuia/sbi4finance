"""Observation-only recovery targets and bounded native-reference orchestration."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from scripts import run_heston_spread_recovery_reference as runner


class RecoveryReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.design, self.panel = self.root/"design.json", self.root/"panel.json"
        records = {"forward": [100.]*54, "strike": np.linspace(85, 115, 54).tolist(),
            "discount": [.99]*54, "maturity": [.25]*54, "option_type": ["put"]*27+["call"]*27,
            "mid": [1.]*54, "half_spread": [.1]*54}
        self.products = {"control": [0., .1], "affected": [.3, .9]}
        runner.write_json(self.design, {"records": records, "products": self.products})
        self.pilot = runner.HestonSpreadPilot(records, infer_c=True)
        y = np.linspace(-.05, .2, 54)
        self.case = {"id": "validation-00000017", "row_id": "validation:00000017",
            "observations": y.tolist(), "observations_sha256": runner.values_digest(y)}
        self.panel_record = {"format": "heston_spread_recovery_observation_panel_v1",
            "design_path": str(self.design), "design_sha256": runner.digest(self.design),
            "joint_task": self.pilot.task_definition(), "products": self.products, "cases": [self.case]}
        runner.write_json(self.panel, self.panel_record)
        self.out, self.payload = self.root/"evidence"/self.case["id"], self.root/"payload"/self.case["id"]
        self.protocol_path = self.root/"protocol.json"
        self.protocol = runner.prepare_protocol(self.panel, self.out.parent, self.payload.parent, self.protocol_path)

    def test_panel_has_six_joint_targets_and_rejects_truth_data_or_modified_observations(self):
        panel, case, pilot, y = runner.load_case(self.panel, self.case["id"])
        self.assertEqual(pilot.names[-1], "c")
        self.assertEqual(pilot.bounds.shape, (6, 2))
        self.assertEqual(set(case), {"id", "row_id", "observations", "observations_sha256"})
        np.testing.assert_array_equal(y, self.case["observations"])
        for mutation in ("truth", "clean", "observation", "design"):
            changed = copy.deepcopy(self.panel_record)
            if mutation == "truth": changed["cases"][0]["physical_truth"] = [.04, 2., .05, .4, -.7]
            elif mutation == "clean": changed["cases"][0]["clean_prices"] = [0.]*54
            elif mutation == "observation": changed["cases"][0]["observations"][0] += .01
            else: changed["design_sha256"] = "different"
            runner.write_json(self.panel, changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.load_case(self.panel, self.case["id"])

    def test_full_normalized_likelihood_changes_with_c_without_fixing_phi(self):
        _, _, pilot, y = runner.load_case(self.panel, self.case["id"])
        point = pilot.bounds.mean(axis=1)
        with patch.object(pilot, "prices", return_value=np.zeros(54)):
            values = []
            for c in (.6, 1.8):
                point[-1] = c
                sigma = c*pilot.records["half_spread"]
                expected = -.5*np.sum((y/sigma)**2)-np.log(sigma).sum()-.5*len(y)*np.log(2*np.pi)
                actual = pilot.log_likelihood(point, observations=y)
                self.assertAlmostEqual(actual, expected, places=11)
                values.append(actual)
            residual = np.sum((y/pilot.records["half_spread"])**2)
            expected_delta = -54*np.log(1.8/.6)-.5*residual*(1/1.8**2-1/.6**2)
            self.assertAlmostEqual(values[1]-values[0], expected_delta, places=11)

    def test_protocol_freezes_seeds_support_case_order_outputs_and_sources(self):
        runner.validate_protocol(self.protocol, self.panel, self.case["id"], self.out, self.payload)
        for mutation in ("seed", "support", "output", "sampler", "source"):
            changed = copy.deepcopy(self.protocol)
            if mutation == "seed": changed["jobs"][0]["seed"] += 1
            elif mutation == "support": changed["jobs"][0]["bounds"][-1] = [.1, 4.]
            elif mutation == "output": changed["jobs"][0]["evidence_out"] = str(self.root/"other")
            elif mutation == "sampler": changed["settings"]["attempts"] = [[100, 100]]
            else: changed["source_sha256"]["recovery_runner"]["sha256"] = "different"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.validate_protocol(changed, self.panel, self.case["id"], self.out, self.payload)

    @contextmanager
    def fake_sampling(self, passes, *, precision=True):
        calls, initializations = [], []
        def initialize(likelihood, bounds, chains, seed, **kwargs):
            self.assertEqual(bounds.shape, (6, 2))
            self.assertEqual(chains, 4)
            self.assertEqual(set(kwargs), {"fit_starts", "fit_budget"})
            initializations.append({"seed": seed, "bounds": bounds.copy()})
            return {"fixture": "data-only initialization"}
        def sample(likelihood, bounds, names, **settings):
            self.assertEqual(names, self.pilot.names)
            self.assertEqual(bounds.shape, (6, 2))
            self.assertFalse(any("truth" in key for key in settings))
            calls.append(settings)
            draws = np.broadcast_to(bounds.mean(axis=1), (4, settings["draws"], 6)).copy()
            return xr.DataTree.from_dict({"posterior": xr.Dataset({"parameters": (("chain", "draw", "parameter"), draws)},
                coords={"parameter": names}), "sample_stats": xr.Dataset()})
        def diagnostics(*args):
            return {"parameters": {}, "products": {"parameters": {}}, "mcmc_diagnostics_passed": next(passes)}
        with patch.object(runner.reference, "data_fit_initialization", side_effect=initialize), \
                patch.object(runner.reference, "sample_deterministic_reference", side_effect=sample), \
                patch.object(runner, "diagnose", side_effect=diagnostics), \
                patch.object(runner, "price_audit", return_value=(np.ones(54), {"pointwise_gates_passed": precision})):
            yield calls, initializations

    def test_one_escalation_retains_joint_chains_and_failed_convergence(self):
        with self.fake_sampling(iter([False, False])) as (calls, initializations):
            result = runner.run_reference(self.panel, self.case["id"], self.protocol_path, self.out, self.payload)
        self.assertEqual([(item["tune"], item["draws"]) for item in calls], list(runner.ATTEMPTS))
        self.assertTrue(all(item["chains"] == 4 and item["cores"] == 1 for item in calls))
        self.assertEqual(len(initializations), 2)
        self.assertFalse(result["reference_ready"])
        self.assertFalse(result["truth_available_to_sampler"])
        self.assertNotIn("physical_truth", result)
        self.assertEqual(result["task"], self.pilot.task_definition())
        with np.load(result["chain_payload_path"], allow_pickle=False) as archive:
            self.assertEqual(archive["parameters"].shape, (4, 12000, 6))
        diagnostics = runner.read(self.out/"diagnostics.json")
        self.assertEqual(diagnostics["chain_payload_sha256"], runner.digest(result["chain_payload_path"]))
        with self.assertRaises(FileExistsError):
            runner.run_reference(self.panel, self.case["id"], self.protocol_path, self.out, self.payload)

    def test_numerical_failure_never_triggers_a_sampler_escalation(self):
        with self.fake_sampling(iter([False]), precision=False) as (calls, _):
            with self.assertRaisesRegex(FloatingPointError, "precision failed"):
                runner.run_reference(self.panel, self.case["id"], self.protocol_path, self.out, self.payload)
        self.assertEqual(len(calls), 1)
        self.assertEqual(runner.read(self.out/"run.json")["status"], "failed")
        self.assertTrue((self.out/"failure.json").is_file())
        self.assertTrue((self.payload/"attempt-0/chains.npz").is_file())


if __name__ == "__main__":
    unittest.main()
