"""A complete cached synthetic workflow is usable without a neural runtime."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sbi4finance.datasets.local import LocalArtifact
from sbi4finance.svj import model, teaching
from sbi4finance.svj.adapters import prepared_returns
from sbi4finance.svj.bank import default_task

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/synthetic/svj-workflow"


class SVJTeachingTests(unittest.TestCase):
    def test_synthetic_window_has_complete_task_and_truth_is_never_conditioning(self):
        artifact, simulation = teaching.synthetic_case()
        self.assertEqual(artifact.metadata["scientific_task"], default_task())
        self.assertEqual(artifact.metadata["data_origin"], "synthetic")
        self.assertEqual(artifact.metadata["provenance"]["selection"], "contributed_illustration_not_calibration_sample")
        np.testing.assert_allclose(prepared_returns(artifact), simulation["returns"][0], rtol=1e-14, atol=1e-14)
        poisoned = deepcopy(artifact.metadata)
        poisoned["truth"]["targets"] = [1000.] * model.N_TARGETS
        poisoned["future_outcomes"] = [999.] * 10
        changed = LocalArtifact(poisoned, artifact.rows)
        np.testing.assert_array_equal(prepared_returns(changed), prepared_returns(artifact))

    def test_cache_binds_sources_checkpoint_task_observation_and_output_status(self):
        manifest, window, arrays, report = teaching.load_bundle(BUNDLE)
        self.assertEqual(manifest["source_sha256"], teaching.source_hashes())
        self.assertEqual(set(arrays), set(teaching.ROLES))
        self.assertEqual(arrays["posterior"].shape, (128, model.N_TARGETS))
        self.assertEqual(len(window.rows), model.WINDOW)
        workflow = report["workflow"]
        self.assertEqual(workflow["scientific_acceptance"], "not_assessed_by_this_run")
        self.assertEqual(workflow["reliability"], report["risk"]["reliability"])
        self.assertEqual(workflow["provenance"]["diagnostic_reference"]["reference_sha256"],
                         manifest["checkpoint"]["files"]["misspecification.npz"])
        self.assertEqual(report["risk"]["unit"], "percent_log_loss")
        self.assertEqual([stage["stage"] for stage in workflow["stages"]], ["amortized", "correction"])
        self.assertFalse(manifest["scientific_acceptance"])

    def test_cached_loading_does_not_import_neural_backends(self):
        code = (
            "import sys; from sbi4finance.svj.teaching import load_bundle; "
            "load_bundle(sys.argv[1]); "
            "assert not {'jax','keras','bayesflow'}.intersection(sys.modules), 'neural backend loaded'"
        )
        result = subprocess.run([sys.executable, "-c", code, str(BUNDLE)], cwd=ROOT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cache_corruption_and_stale_identity_fail_closed(self):
        cases = ("payload", "source", "checkpoint", "roles", "status")
        for kind in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bundle"
                shutil.copytree(BUNDLE, path)
                manifest = json.loads((path / "bundle.json").read_text())
                if kind == "payload":
                    with (path / "illustration.npz").open("ab") as stream:
                        stream.write(b"changed")
                elif kind == "source":
                    manifest["source_sha256"]["svj/pipeline.py"] = "0" * 64
                elif kind == "checkpoint":
                    manifest["checkpoint"]["files"]["model.keras"] = "0" * 64
                elif kind == "roles":
                    manifest["roles"]["truth"] = "observation"
                else:
                    report = json.loads((path / "report.json").read_text())
                    report["risk"]["reliability"] = "accepted"
                    (path / "report.json").write_text(json.dumps(report))
                    manifest["files"]["report.json"] = hashlib.sha256((path / "report.json").read_bytes()).hexdigest()
                (path / "bundle.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    teaching.load_bundle(path)


if __name__ == "__main__":
    unittest.main()
