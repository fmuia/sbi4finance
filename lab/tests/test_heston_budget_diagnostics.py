"""Uncertainty can turn a pointwise agreement into an inconclusive comparison."""

import json
import unittest

import numpy as np

from sbi4finance.calibration.heston_budget_diagnostics import conditional_agreement


class ConditionalAgreementTests(unittest.TestCase):
    def fixture(self, *, dims=4, draws=512):
        rng = np.random.default_rng(193)
        chains = rng.normal(size=(4, draws, dims))
        neural = rng.normal(size=(2048, dims))
        names = [f"parameter_{index}" for index in range(dims)]
        diagnostics = {"mcmc_diagnostics_passed": True, "parameters": {
            f"parameters[{name}]": {"ess_bulk": 4 * draws, "ess_tail": 4 * draws, "mcse_mean": .01}
            for name in names}}
        return neural, chains, diagnostics, names

    def compare(self, neural, chains, diagnostics, names, **kwargs):
        return conditional_agreement(neural, chains, diagnostics, names, seed=17, replicates=100, **kwargs)

    def test_matching_samples_pass_and_preserve_shapes_and_determinism(self):
        inputs = self.fixture()
        first = self.compare(*inputs)
        second = self.compare(*inputs)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(first["reference_chain_shape"], [4, 512, 4])
        self.assertEqual(first["block_lengths"], [8, 16])
        self.assertEqual(len(first["block_length_sensitivity"]), 2)
        for parameter in first["parameters"].values():
            self.assertGreater(parameter["reference_mean_mcse"], 0)
            self.assertGreater(parameter["reference_quantile_mcse"][0], 0)
            self.assertGreater(parameter["neural_quantile_mcse_iid"][1], 0)
            self.assertLessEqual(parameter["signed_standardized_mean_interval"][0],
                                 parameter["signed_standardized_mean_difference"])
        json.dumps(first, allow_nan=False)

    def test_mean_threshold_crossing_is_inconclusive_and_clear_miss_fails(self):
        neural, chains, diagnostics, names = self.fixture()
        reference = chains.reshape(-1, 4)
        # Match the point mean to 0.29 reference SD, then let the prescribed
        # Monte Carlo interval determine whether the 0.3 margin is established.
        neural[:, 0] += reference.mean(0)[0] - neural.mean(0)[0] + .29 * reference.std(0, ddof=1)[0]
        result = self.compare(neural, chains, diagnostics, names)
        self.assertEqual(result["parameters"][names[0]]["mean_status"], "inconclusive")
        self.assertEqual(result["status"], "inconclusive")
        neural[:, 0] += 1
        result = self.compare(neural, chains, diagnostics, names)
        self.assertEqual(result["parameters"][names[0]]["mean_status"], "fail")
        self.assertEqual(result["status"], "fail")

    def test_width_threshold_crossing_is_inconclusive(self):
        neural, chains, diagnostics, names = self.fixture()
        ref_width = np.diff(np.quantile(chains.reshape(-1, 4), [.05, .95], axis=0), axis=0)[0]
        net_width = np.diff(np.quantile(neural, [.05, .95], axis=0), axis=0)[0]
        neural[:, 0] *= 1.24 * ref_width[0] / net_width[0]
        result = self.compare(neural, chains, diagnostics, names)
        self.assertEqual(result["parameters"][names[0]]["width_status"], "inconclusive")

    def test_saved_mean_mcse_is_a_floor(self):
        neural, chains, diagnostics, names = self.fixture()
        diagnostics["parameters"][f"parameters[{names[0]}]"]["mcse_mean"] = .2
        result = self.compare(neural, chains, diagnostics, names)
        parameter = result["parameters"][names[0]]
        self.assertGreaterEqual(parameter["reference_mean_mcse"], .2)
        self.assertGreaterEqual(parameter["standardized_mean_mcse"], .2 / parameter["reference_sd"])
        self.assertEqual(parameter["mean_status"], "inconclusive")

    def test_correlated_chains_have_more_uncertainty_than_permuted_iid_rows(self):
        neural, chains, diagnostics, names = self.fixture(dims=1, draws=2048)
        rng = np.random.default_rng(682)
        innovations = rng.normal(size=chains.shape)
        for draw in range(1, chains.shape[1]):
            chains[:, draw] = .95 * chains[:, draw - 1] + np.sqrt(1 - .95 ** 2) * innovations[:, draw]
        diagnostic = diagnostics["parameters"][f"parameters[{names[0]}]"]
        diagnostic.update(ess_bulk=210, ess_tail=210, mcse_mean=0.)
        correlated = self.compare(neural, chains, diagnostics, names)
        independent = self.compare(neural, np.stack([rng.permutation(chain) for chain in chains]), diagnostics, names)
        a = correlated["parameters"][names[0]]["reference_mean_mcse"]
        b = independent["parameters"][names[0]]["reference_mean_mcse"]
        self.assertGreater(a, 2 * b)

    def test_low_effective_sample_size_cannot_be_silently_capped(self):
        neural, chains, diagnostics, names = self.fixture()
        diagnostics["parameters"][f"parameters[{names[0]}]"]["ess_tail"] = 2
        result = self.compare(neural, chains, diagnostics, names)
        self.assertEqual(result["status"], "inconclusive")
        self.assertGreaterEqual(result["block_lengths"][-1], chains.shape[1])
        self.assertIn("insufficient effective blocks", result["reason"])

    def test_failed_reference_diagnostics_cannot_admit_a_fit(self):
        neural, chains, diagnostics, names = self.fixture()
        diagnostics["mcmc_diagnostics_passed"] = False
        result = self.compare(neural, chains, diagnostics, names)
        self.assertEqual(result["status"], "inconclusive")

    def test_malformed_or_degenerate_inputs_fail_explicitly(self):
        neural, chains, diagnostics, names = self.fixture()
        invalid = [
            (neural[:4], chains, diagnostics, names),
            (neural, chains.reshape(-1, 4), diagnostics, names),
            (neural.astype(complex), chains, diagnostics, names),
            (neural, chains, {}, names),
            (np.zeros_like(neural), chains, diagnostics, names),
            (neural, np.zeros_like(chains), diagnostics, names),
            (neural, chains, diagnostics, [*names[:-1], names[0]]),
        ]
        for inputs in invalid:
            with self.subTest(shapes=[np.shape(value) for value in inputs[:2]]), self.assertRaises(ValueError):
                self.compare(*inputs)
        for options in ({"family_checks": 7}, {"mean_limit": 0}, {"width_limits": (1., 2.)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.compare(neural, chains, diagnostics, names, **options)


if __name__ == "__main__":
    unittest.main()
