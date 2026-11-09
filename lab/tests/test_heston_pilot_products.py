"""Downstream variance-swap strikes agree with integrated Heston mean variance."""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.integrate import quad

from scripts.run_heston_observation_pilot import expected_variance, finalize_reference


class VarianceProductTests(unittest.TestCase):
    def test_analysis_rejects_wrong_design_before_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "run.json").write_text(json.dumps({"design_sha256": "a" * 64}))
            sentinel = out / "diagnostics.json"
            sentinel.write_text("preserved")
            with self.assertRaisesRegex(ValueError, "design SHA-256"):
                finalize_reference(out, {}, design_sha256="b" * 64)
            self.assertEqual(sentinel.read_text(), "preserved")

    def test_forward_and_spot_strikes_against_independent_integration(self):
        rows = np.array([[.025, 2., .06, .7, -.6], [.1, 1e-10, .01, .4, -.4],
                         [.04, 10., .04, .3, -.5]])
        for interval in ([0., .05], [.22, .89]):
            actual = expected_variance(rows, interval)
            expected = [quad(lambda t: theta + (v0-theta)*np.exp(-kappa*t), *interval)[0]
                        / (interval[1]-interval[0]) for v0,kappa,theta,*_ in rows]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
