"""Scientific guardrails of the sampled preserved-bank reuse probe.

Small synthetic archives stand in for the immutable historical bank.  Pricing
is stubbed where only provenance, observations and failure reporting matter;
the HestonSpreadPilot tests exercise the actual checked Fourier engine.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from scripts import probe_heston_spread_baseline as probe


REAL_NP_LOAD = np.load
BASE_PRICES = np.array([1., 2., 3., 4.])


def toy_prices(pilot, parameters, mask=None, *, return_diagnostics=False):
    prices = (BASE_PRICES + parameters[0])[pilot._indices(mask)]
    return (prices, {"nodes": 1024, "cutoff": 600.}) if return_diagnostics else prices


class GuardedArchive:
    """Fail on reserved members or enumeration, even if the payload contains them."""

    def __init__(self, path, accesses):
        self.archive = REAL_NP_LOAD(path, allow_pickle=False)
        self.accesses = accesses

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()

    def __getitem__(self, key):
        self.accesses.append(key)
        if key not in {"validation_parameters", "validation_clean"}:
            raise AssertionError(f"reserved or undeclared bank array accessed: {key}")
        return self.archive[key]

    @property
    def files(self):
        raise AssertionError("probe must read named validation members, not enumerate the bank")


class HestonSpreadProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.design_path = self.root / "design.json"
        self.bank_path = self.root / "bank"
        self.bank_path.mkdir()
        self.out = self.root / "result"
        self.records = {
            "forward": [100., 102., 105., 105.],
            "strike": [90., 102., 110., 100.],
            "discount": [.99, .98, .96, .96],
            "maturity": [.1, .25, .75, .75],
            "option_type": ["put", "call", "call", "put"],
            "mid": [1., 2., 3., 4.], "half_spread": [.03, .02, .04, .03],
        }
        self.design_path.write_text(json.dumps({"records": self.records}))
        self.physical = np.array([[.04, 2., .05, .4, -.7],
                                  [.06, 1., .04, .6, -.5],
                                  [.05, 3., .03, .5, -.6]])
        self.clean = BASE_PRICES + self.physical[:, :1]
        payload = self.bank_path / "bank.npz"
        np.savez_compressed(payload, validation_parameters=self.physical,
                            validation_clean=self.clean,
                            test_parameters=np.full((2, 5), np.nan),
                            test_clean=np.full((2, 4), np.nan))
        fixed = probe.HestonSpreadPilot(self.records)
        task = {"format": "heston_pilot_full_campaign_task_v1", "model": "heston",
                "parameter_names": [*fixed.names, "tau"],
                "prior_bounds": [*fixed.bounds.tolist(), [0., .02]],
                "design_sha256": probe.digest(self.design_path),
                "pricing_controls": {
                    "measurement_spread_multiplier": .5,
                    "numerical_fraction": fixed.numerical_fraction,
                    "quadrature_ladder": [list(rung) for rung in fixed.quadrature_ladder]}}
        self.bank = {"format": "heston_pilot_clean_campaign_bank_v1", "identity": {"task": task},
                     "payload": "bank.npz", "payload_sha256": probe.digest(payload)}
        self.write_bank()

    def write_bank(self):
        (self.bank_path / "bank.json").write_text(json.dumps(self.bank))

    @contextmanager
    def cheap_pricing(self, *, refined_shift=.001, refined_ll_shift=0.):
        with patch.object(probe.HestonSpreadPilot, "prices", toy_prices), patch.object(
                probe.HestonSpreadPilot, "refinement_report", return_value={
                    "max_abs_price_shift_over_measurement_sigma": refined_shift,
                    "log_likelihood_shift": refined_ll_shift}):
            yield

    def run_probe(self, **kwargs):
        return probe.run(self.design_path, self.bank_path, self.out, **kwargs)

    def assert_failure_record(self, *, error_type="FloatingPointError"):
        record = json.loads((self.out / "failure.json").read_text())
        self.assertEqual(record["error_type"], error_type)
        self.assertIs(record["scientific_acceptance"], False)
        return record

    def test_design_or_payload_mismatch_fails_before_any_array_is_opened(self):
        for changed in ("design", "payload"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                # Preserve the original paths/bytes and alter only the declared identity.
                bank = json.loads(json.dumps(self.bank))
                if changed == "design":
                    bank["identity"]["task"]["design_sha256"] = "0" * 64
                else:
                    bank["payload_sha256"] = "0" * 64
                (self.bank_path / "bank.json").write_text(json.dumps(bank))
                with patch.object(probe.np, "load", side_effect=AssertionError("archive opened")) as load:
                    with self.assertRaisesRegex(ValueError, "exact design|payload hash"):
                        probe.run(self.design_path, self.bank_path, Path(tmp) / "result")
                load.assert_not_called()

    def test_changed_physical_prior_or_precision_gate_is_not_bank_reuse(self):
        for changed in ("prior", "precision"):
            with self.subTest(changed=changed):
                bank = json.loads(json.dumps(self.bank))
                if changed == "prior":
                    bank["identity"]["task"]["prior_bounds"][3][1] = 1.5
                else:
                    bank["identity"]["task"]["pricing_controls"]["numerical_fraction"] = .1
                (self.bank_path / "bank.json").write_text(json.dumps(bank))
                with patch.object(probe.np, "load", side_effect=AssertionError("archive opened")) as load:
                    with self.assertRaisesRegex(ValueError, "physical prior or pricing gate"):
                        self.run_probe()
                load.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_fixed_c1_observations_are_identical_for_both_targets_and_reserved_members_unread(self):
        calls, accesses = [], []
        original = probe.HestonSpreadPilot.log_likelihood

        def likelihood(pilot, theta, mask=None, *, observations=None):
            calls.append((pilot.infer_c, np.asarray(theta).copy(), np.asarray(observations).copy()))
            return original(pilot, theta, mask, observations=observations)

        def guarded_load(path, **kwargs):
            self.assertIs(kwargs["allow_pickle"], False)
            return GuardedArchive(path, accesses)

        with self.cheap_pricing(), patch.object(probe.np, "load", side_effect=guarded_load), patch.object(
                probe.HestonSpreadPilot, "log_likelihood", likelihood):
            summary = self.run_probe(rows=(2, 0), seed=19)
        self.assertIs(summary["checks_passed"], True)
        self.assertIs(summary["scientific_acceptance"], False)
        self.assertEqual(set(accesses), {"validation_parameters", "validation_clean"})
        cases = json.loads((self.out / "cases.json").read_text())["cases"]
        for j, row in enumerate((2, 0)):
            expected = self.clean[row] + np.random.default_rng(19+j).standard_normal(4) * self.records["half_spread"]
            fixed, joint = calls[2*j:2*j+2]
            self.assertFalse(fixed[0]); self.assertTrue(joint[0])
            self.assertEqual(joint[1][-1], 1.)
            self.assertTrue(np.array_equal(fixed[2], joint[2]))
            self.assertTrue(np.array_equal(fixed[2], expected))
            assert_allclose(cases[j]["observations"], expected, rtol=0, atol=0)
            self.assertEqual(cases[j]["generating_c"], 1.)
            self.assertAlmostEqual(summary["cases"][j]["conditional_c_density_grid_integral"], 1., places=5)
        self.assertEqual(summary["case_file_sha256"], probe.digest(self.out / "cases.json"))

    def test_finite_refinement_failure_is_preserved_as_failed_probe(self):
        with self.cheap_pricing(refined_shift=.03):
            with self.assertRaisesRegex(FloatingPointError, "probe failed"):
                self.run_probe(rows=(0,))
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertIs(summary["checks_passed"], False)
        self.assertIs(summary["scientific_acceptance"], False)
        self.assertIs(summary["cases"][0]["checks_passed"], False)
        self.assert_failure_record()

    def test_incorrect_but_finite_conditional_density_is_not_accepted(self):
        with self.cheap_pricing(), patch.object(probe, "conditional_c_log_density",
                side_effect=lambda grid, _: np.zeros_like(grid)):
            with self.assertRaisesRegex(FloatingPointError, "probe failed"):
                self.run_probe(rows=(0,))
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertIs(summary["checks_passed"], False)
        self.assertAlmostEqual(summary["cases"][0]["conditional_c_density_grid_integral"], 1.5)
        self.assert_failure_record()

    def test_nonfinite_metrics_fail_explicitly_and_leave_serializable_failure_record(self):
        for metric in ("likelihood", "density", "refined_likelihood"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / "result"
                if metric == "likelihood":
                    corrupt = patch.object(probe.HestonSpreadPilot, "log_likelihood", return_value=-np.inf)
                elif metric == "density":
                    corrupt = patch.object(probe, "conditional_c_log_density",
                        side_effect=lambda grid, _: np.full_like(grid, np.nan))
                else:
                    corrupt = patch.object(probe.HestonSpreadPilot, "refinement_report", return_value={
                        "max_abs_price_shift_over_measurement_sigma": .001,
                        "log_likelihood_shift": np.nan})
                with self.cheap_pricing(), corrupt:
                    with self.assertRaisesRegex(FloatingPointError, "nonfinite engineering probe metrics"):
                        probe.run(self.design_path, self.bank_path, destination, rows=(0,))
                record = json.loads((destination / "failure.json").read_text())
                self.assertEqual(record["error_type"], "FloatingPointError")
                self.assertEqual(record["completed_cases"], [])
                self.assertIs(record["scientific_acceptance"], False)
                self.assertFalse((destination / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
