"""Readiness uses development data only and does not itself admit training."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import check_heston_budget_readiness as readiness
from sbi4finance.calibration import tasks


class ReadinessTests(unittest.TestCase):
    def test_support_requires_finite_physical_draws_without_clipping(self):
        bounds = np.asarray([[0., 1.], [-1., 0.]])
        self.assertTrue(readiness.checked_support(np.array([[.5, -.5]]), bounds))
        self.assertFalse(readiness.checked_support(np.array([[1.01, -.5]]), bounds))
        self.assertFalse(readiness.checked_support(np.array([[np.nan, -.5]]), bounds))

    def test_control_reads_validation_only_and_never_admits_matrix(self):
        from sbi4finance.calibration import bank_collection, full_surface, heston_budget_diagnostics

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, checkpoint, reference = root / "bank", root / "checkpoint", root / "reference"
            for path in (bank, checkpoint, reference):
                path.mkdir()
            task = tasks.heston_task()
            (bank / "manifest.json").write_text("{}")
            (checkpoint / "model.keras").write_bytes(b"test model identity")
            names = task["parameter_names"]
            diagnostics = {"mcmc_diagnostics_passed": True, "parameters": {
                f"parameters[{name}]": {"r_hat": 1., "ess_bulk": 1000, "ess_tail": 1000, "mcse_mean": .0001}
                for name in names}}
            row = np.asarray(task["prior_bounds"]).mean(1)
            ref = {"task": task, "settings": {"chains": 2, "draws": 16}, "observed_prices": [.02] * 63}
            (reference / "reference.json").write_text(json.dumps(ref))
            (reference / "diagnostics.json").write_text(json.dumps(diagnostics))
            np.savez(reference / "chains.npz", parameters=np.broadcast_to(row, (2, 16, 4)))
            protocol = {"format": "heston_training_budget_protocol_v1",
                "bank": {"manifest_sha256": readiness.digest(bank / "manifest.json"), "task_fingerprint": "test"},
                "control": {"checkpoint_sha256": readiness.digest(checkpoint / "model.keras")},
                "conditional": {"case_indices": [0], "source_template": str(reference), "draws": 8,
                    "seed_base": 11, "mean_error_reference_sd_max": .3, "central90_width_ratio": [.75, 1.25]},
                "data_roles": {"development": {"split": "validation", "start": 0, "stop": 4, "draws": 8,
                    "seed": 12, "batch_size": 2}, "posterior_predictive": {"start": 0, "stop": 2,
                    "paired_draws": 2, "seed": 13}}}
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            args = Namespace(protocol=protocol_path, bank=bank, checkpoint=checkpoint, out=root / "readiness")
            split = {"observations": np.full((4, 63, 1), .02) + np.arange(4)[:, None, None] * .00001,
                     "parameters": row + np.linspace(-.0001, .0001, 4)[:, None],
                     "example_id": np.asarray([f"validation:{i}" for i in range(4)])}
            record = {"checkpoint_sha256": protocol["control"]["checkpoint_sha256"],
                      "task_fingerprint": "test", "network": {"encoder": "vector"}}
            def sample(workflow, task, observations, count, **kwargs):
                result = np.broadcast_to(row, (len(observations), count, 4)).copy()
                result += np.linspace(-.0001, .0001, count)[None, :, None]
                result += (np.asarray(observations).reshape(len(observations), -1)[:, 0] - .02)[:, None, None]
                return result
            with patch.object(bank_collection, "load_bank", return_value=(
                    {"configuration": {"task": task}}, {"validation": split})) as loader, \
                    patch.object(full_surface, "load_checkpoint", return_value=(task, object(), record)), \
                    patch.object(full_surface, "sample", side_effect=sample), \
                    patch.object(heston_budget_diagnostics, "conditional_agreement", return_value={"status": "pass"}), \
                    patch.object(tasks, "heston_prices", return_value=(np.full(63, .02), np.zeros(63))):
                report = readiness.run(args)
            self.assertEqual(loader.call_args.kwargs["splits"], ("validation",))
            self.assertEqual(loader.call_args.kwargs["roles"], bank_collection.TRAINING_ROLES)
            self.assertFalse(report["reserved_test_evaluated"])
            self.assertFalse(report["matrix_launch_allowed"])
            self.assertFalse(report["training_performed"])
            self.assertEqual(len(report["development"]["example_ids"]), 4)
            self.assertEqual(json.loads((args.out / "readiness.json").read_text())["protocol_sha256"], readiness.digest(protocol_path))


if __name__ == "__main__":
    unittest.main()
