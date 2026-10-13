"""Numerical failures must not become apparently successful forecasts."""
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.svj import audit, model, risk


class FailureTests(unittest.TestCase):
    def test_no_supported_importance_mass_is_an_explicit_failure(self):
        draws = model.simulate(12, seed=43)["targets"]
        with patch.object(audit, "loglik_sd", return_value=1.0), patch.object(
            audit, "loglik_batch", return_value=(np.full(8, -np.inf), np.ones(8))
        ):
            with self.assertRaisesRegex(ValueError, "all-zero"):
                audit.audit(draws, np.zeros(1000), n_proposals=8, n_particles=8)

    def test_invalid_proposal_input_is_rejected(self):
        for draws in (np.zeros((1, 9)), np.zeros((5, 8)), np.full((5, 9), np.nan)):
            with self.assertRaises(ValueError):
                audit.fit_proposal(draws)

    def test_log_loss_definition_and_empty_or_invalid_samples(self):
        self.assertEqual(risk.RISK_UNIT, "percent_log_loss")
        sample = np.array([-50., -10., 0., 10., 50.])
        var, es = risk.var_es(sample, .8, .8)
        self.assertAlmostEqual(var, 18.)
        self.assertEqual(es, 50.)
        for bad in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                risk.var_es(bad)


if __name__ == "__main__":
    unittest.main()
