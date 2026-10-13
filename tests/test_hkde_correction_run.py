"""Frozen-input, untouched-holdout and numerical-reference boundaries."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "hkde_correction_run_tested", ROOT / "scripts/evaluate_hkde_correction.py")
run = importlib.util.module_from_spec(_SPEC)
# The runner imports two existing sibling scripts; mirror script invocation
# without requiring installation of scripts as a new package.
sys.path.insert(0, str(ROOT / "scripts"))
try:
    _SPEC.loader.exec_module(run)
finally:
    sys.path.pop(0)


class HKDECorrectionRunTests(unittest.TestCase):
    def test_test_generation_cannot_precede_matching_validation(self):
        protocol = json.loads((ROOT / "configs/hkde_correction_protocol.json").read_text())
        bindings = {"frozen-input": "original-sha"}
        valid = {
            "passed": True,
            "split": "validation",
            "experiment_id": protocol["experiment_id"],
            "input_sha256": bindings,
            "source_sha256": run.source_hashes(),
            "calibration": {"rank_diagnostic_passed": True},
            "gates": {**dict.fromkeys(protocol["gates"], True),
                      "proposal_ess": True, "proposal_max_weight": True,
                      "sir_sbc": True, "calibration_ess": True,
                      "calibration_max_weight": True},
            "precision": {"gates": {**dict.fromkeys(protocol["gates"], True),
                                    "proposal_ess": True, "proposal_max_weight": True}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "validation.json"
            path.write_text(json.dumps(valid))
            run.require_validation(path, protocol, bindings)
            variants = []
            for field, value in (("passed", False), ("split", "test"),
                                 ("experiment_id", "another-candidate"),
                                 ("input_sha256", {"frozen-input": "changed-sha"}),
                                 ("source_sha256", {})):
                changed = deepcopy(valid)
                changed[field] = value
                variants.append((field, changed))
            for field in ("gates", "precision"):
                changed = deepcopy(valid)
                gates = changed[field] if field == "gates" else changed[field]["gates"]
                gates["max_abs_target_quantile_error_currency"] = False
                variants.append((f"{field}-price-failure", changed))
            for gate in ("proposal_ess", "calibration_ess"):
                changed = deepcopy(valid)
                changed["gates"][gate] = False
                variants.append((gate, changed))
            changed = deepcopy(valid)
            changed["calibration"]["rank_diagnostic_passed"] = False
            variants.append(("sbc", changed))
            variants.append(("missing-report", None))
            for name, report in variants:
                with self.subTest(name=name):
                    args = ["evaluate_hkde_correction.py", "--split", "test",
                            "--out", str(folder / "fresh-test")]
                    if report is not None:
                        path.write_text(json.dumps(report))
                        args += ["--validation-report", str(path)]
                    inputs = (protocol, None, None, None, None, None, bindings)
                    with patch.object(sys, "argv", args), \
                         patch.object(run, "load_inputs", return_value=inputs), \
                         patch.object(run.option_bank, "generate_examples") as generate, \
                         patch.object(run.inference, "build_workflow") as workflow:
                        with self.assertRaisesRegex(ValueError, "validation"):
                            run.main()
                        generate.assert_not_called()
                        workflow.assert_not_called()
                    self.assertFalse((folder / "fresh-test").exists())

    def test_loading_binds_the_original_task_seed_checkpoint_and_price_model(self):
        protocol_path = ROOT / "configs/hkde_correction_protocol.json"
        protocol, _, _, _, _, _, bindings = run.load_inputs(protocol_path)
        original_hashes = {name: run.sha(ROOT / name) for name in bindings}
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary)
            for name in bindings:
                target = copied / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            copied_protocol = copied / "configs/hkde_correction_protocol.json"
            self.assertEqual(run.load_inputs(copied_protocol)[-1], bindings)
            config_path = copied / protocol["task_config"]
            config = json.loads(config_path.read_text())
            changed = deepcopy(config)
            changed["root_seed"] += 1
            config_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "matching frozen"):
                run.load_inputs(copied_protocol)
            shutil.copyfile(ROOT / protocol["task_config"], config_path)
            changed = deepcopy(config)
            changed["task"]["context"]["quote_noise_std"] *= 2
            config_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "matching frozen"):
                run.load_inputs(copied_protocol)
            shutil.copyfile(ROOT / protocol["task_config"], config_path)
            for relative in (f"{protocol['checkpoint']}/model.keras",
                             f"{protocol['baseline_results']}/reference-grid.npz",
                             f"{protocol['surrogate_results']}/model.npz"):
                with self.subTest(artifact=relative):
                    with (copied / relative).open("ab") as file:
                        file.write(b"changed")
                    with self.assertRaisesRegex(ValueError, "matching frozen"):
                        run.load_inputs(copied_protocol)
                    shutil.copyfile(ROOT / relative, copied / relative)
            changed = deepcopy(protocol)
            changed["gates"]["max_abs_target_quantile_error_currency"] *= 2
            copied_protocol.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "preserve baseline"):
                run.load_inputs(copied_protocol)
        self.assertEqual({name: run.sha(ROOT / name) for name in bindings}, original_hashes)

    def test_thinned_grid_reuses_exact_nodes_but_reintegrates_the_prior(self):
        # Nonuniform axes catch accidental uniform masses. Distinct functions
        # in each price column catch swapped contract or tensor axes.
        axis0 = 0.025 + 0.04 * np.linspace(0, 1, 129)**2
        axis1 = 0.25 + 0.75 * np.linspace(0, 1, 129)**1.5
        parameters = np.stack(np.meshgrid(axis0, axis1, indexing="ij"), axis=-1).reshape(-1, 2)
        x, y = parameters.T
        fine = {"axes": (axis0, axis1), "shape": (129, 129),
                "parameters": parameters,
                "prices": np.column_stack((x + 10*y, 100*x - y, x*y)),
                "target_prices": np.column_stack((2*x - y, 3*y + x)),
                "quadrature_weights": np.full(len(parameters), 1 / len(parameters))}
        coarse = run.coarse_grid(fine)
        selected = np.array([129*i + j for i in range(0, 129, 2) for j in range(0, 129, 2)])
        self.assertEqual(coarse["shape"], (65, 65))
        for field in ("parameters", "prices", "target_prices"):
            assert_array_equal(coarse[field], fine[field][selected])
        weights = coarse["quadrature_weights"]
        self.assertGreater(weights.min(), 0)
        assert_allclose(weights.sum(), 1, rtol=0, atol=1e-14)
        # Tensor trapezoids integrate constants, each linear coordinate and
        # their product exactly, including with nonuniform nodes.
        means = np.array([(axis0[0] + axis0[-1]) / 2,
                          (axis1[0] + axis1[-1]) / 2])
        assert_allclose(weights @ coarse["parameters"], means, rtol=0, atol=1e-14)
        assert_allclose(weights @ np.prod(coarse["parameters"], axis=1), means.prod(), rtol=0, atol=1e-14)
        self.assertFalse(np.allclose(weights, fine["quadrature_weights"][selected]))
        with self.assertRaisesRegex(ValueError, "129x129"):
            run.coarse_grid(dict(fine, shape=(65, 65)))

    def test_price_and_reliability_failures_do_not_cancel_across_cases(self):
        protocol = json.loads((ROOT / "configs/hkde_correction_protocol.json").read_text())
        base = {"mean_error_in_ref_sd": [0., 0.], "relative_std_error": [0., 0.],
                "correlation_error": 0., "reference_mass_of_90_interval": [.9, .9],
                "target_quantile_error_currency": np.zeros((2, 3)).tolist()}
        cases = [deepcopy(base), deepcopy(base)]
        cases[0]["target_quantile_error_currency"][0][0] = .031
        cases[1]["target_quantile_error_currency"][0][0] = -.031
        efficiency = [{"ess_fraction": .8, "max_weight": .002},
                      {"ess_fraction": .49, "max_weight": .011}]
        result = run.summarize_cases(cases, efficiency, protocol["conditional"], protocol["gates"])
        self.assertAlmostEqual(result["metrics"]["max_abs_target_quantile_error_currency"], .031)
        self.assertFalse(result["gates"]["max_abs_target_quantile_error_currency"])
        self.assertFalse(result["gates"]["proposal_ess"])
        self.assertFalse(result["gates"]["proposal_max_weight"])
        self.assertTrue(result["gates"]["mean_rmse_in_ref_sd"])


if __name__ == "__main__":
    unittest.main()
