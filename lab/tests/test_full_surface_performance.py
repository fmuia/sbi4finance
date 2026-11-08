"""Performance records separate elapsed clocks, resource allocations and actual work."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sbi4finance.calibration.performance import PerformanceRecorder, synchronize_model


class PerformanceTests(unittest.TestCase):
    def test_elapsed_and_allocated_clocks_are_distinct(self):
        with patch.object(PerformanceRecorder, "now", side_effect=[(10., 1.), (11., 1.5), (14., 2.), (15., 3.)]), \
                patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4", "SLURM_GPUS": "1"}):
            recorder = PerformanceRecorder()
            synchronize = Mock()
            with recorder.phase("fit", synchronize=synchronize):
                pass
            report = recorder.report(work={"optimizer_updates": 7})
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(report["wall_seconds"], 5.)
        self.assertEqual(report["process_cpu_seconds"], 2.)
        self.assertEqual(report["phases"]["fit"], {"wall_seconds": 3., "process_cpu_seconds": .5, "calls": 1})
        self.assertEqual(report["allocation"]["cpu_seconds"], 20.)
        self.assertEqual(report["allocation"]["gpu_seconds"], 5.)
        self.assertEqual(report["work"]["optimizer_updates"], 7)
        self.assertFalse(report["phases_are_additive"])
        json.dumps(report, allow_nan=False)

    def test_phase_failure_is_still_timed_without_masking_the_exception(self):
        with patch.object(PerformanceRecorder, "now", side_effect=[(0., 0.), (1., .1), (3., .2)]):
            recorder = PerformanceRecorder()
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                with recorder.phase("failed"):
                    raise ValueError("synthetic failure")
        self.assertEqual(recorder.phases["failed"]["wall_seconds"], 2.)

    def test_synchronization_waits_on_model_and_optimizer_arrays(self):
        backend = SimpleNamespace(block_until_ready=Mock())
        model = SimpleNamespace(variables=[SimpleNamespace(value="weights")],
                                optimizer=SimpleNamespace(variables=[SimpleNamespace(value="iteration")]))
        with patch.dict("sys.modules", {"jax": backend}):
            synchronize_model(model)
        backend.block_until_ready.assert_called_once_with(["weights", "iteration"])


if __name__ == "__main__":
    unittest.main()
