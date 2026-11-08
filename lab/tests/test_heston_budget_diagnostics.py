"""Uncertainty can turn a pointwise agreement into an inconclusive comparison."""

import json
import unittest

import numpy as np

from sbi4finance.calibration.heston_budget_diagnostics import conditional_agreement, paired_final_comparison, _simultaneous_bounds


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


class PairedFinalComparisonTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(67)
        truth = rng.uniform(-.3, .3, (96, 2))
        draws = truth[:, None, :] + rng.normal(0, .1, (96, 1, 2)) + rng.uniform(-.5, .5, (96, 64, 2))
        identifiers = [f"test:{index}" for index in range(96)]
        record = {"draws": draws, "example_ids": identifiers}
        return {"recipe-a-seed-0": record}, record, truth, [[-5., 5.], [-5., 5.]], identifiers, ["a", "b"]

    def compare(self, *args, **kwargs):
        return paired_final_comparison(*args, seed=109, replicates=200, **kwargs)

    def test_identical_estimators_have_exact_paired_zero_error_and_zero_se(self):
        inputs = self.fixture()
        result = self.compare(*inputs)
        self.assertEqual(result, self.compare(*inputs))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["family_checks"], 10)
        self.assertEqual(result["zero_se_checks"], 10)
        for parameter in result["candidates"]["recipe-a-seed-0"]["parameters"].values():
            self.assertEqual(parameter["recovery"]["mse_ratio"], 1.)
            self.assertEqual(parameter["recovery"]["upper_bound"], 1.)
            for value in parameter["coverage"].values():
                self.assertEqual(value["absolute_error_increase"], 0.)
                self.assertEqual(value["upper_bound"], 0.)
        json.dumps(result, allow_nan=False)

    def test_all_candidate_seeds_and_parameters_enter_one_family(self):
        candidates, control, truth, bounds, identifiers, names = self.fixture()
        candidates["recipe-a-seed-1"] = control
        result = self.compare(candidates, control, truth, bounds, identifiers, names)
        self.assertEqual(result["family_checks"], 20)
        self.assertEqual(result["zero_se_checks"], 20)

    def test_coverage_is_absolute_error_from_nominal_and_catches_undercoverage(self):
        n, samples = 100, 64
        truth = np.zeros((n, 1)); identifiers = [f"test:{index}" for index in range(n)]
        ramp = np.linspace(-.5, .5, samples)[None, :, None]
        def record(covered):
            means = np.where(np.arange(n) < covered, .1, 2.)[:, None, None]
            return {"draws": means + ramp, "example_ids": identifiers}
        result = self.compare({"undercovered": record(60)}, record(80), truth, [[-5., 5.]], identifiers, ["a"], levels=(.9,))
        metric = result["candidates"]["undercovered"]["parameters"]["a"]["coverage"]["0.9"]
        self.assertAlmostEqual(metric["absolute_error_increase"], .2)
        self.assertEqual(metric["status"], "fail")

    def test_recovery_is_per_parameter_and_does_not_hide_a_regression(self):
        n, samples = 100, 64
        truth = np.zeros((n, 2)); identifiers = [f"test:{index}" for index in range(n)]
        ramp = np.linspace(-1.25, 1.25, samples)[None, :, None]
        def record(offset):
            return {"draws": np.broadcast_to(np.asarray(offset)[None, None, :] + ramp, (n, samples, 2)),
                    "example_ids": identifiers}
        result = self.compare({"mixed": record([.2, .1])}, record([.1, .9]), truth,
                              [[-5., 5.], [-5., 5.]], identifiers, ["a", "b"])
        parameters = result["candidates"]["mixed"]["parameters"]
        self.assertEqual(parameters["a"]["recovery"]["status"], "fail")
        self.assertEqual(parameters["b"]["recovery"]["status"], "pass")
        self.assertAlmostEqual(parameters["a"]["recovery"]["mse_ratio"], 4.)
        self.assertAlmostEqual(parameters["b"]["recovery"]["mse_ratio"], 1 / 81)

    def test_upper_bound_uses_negative_centered_error_and_handles_constant_metrics(self):
        # A skew empirical error distribution distinguishes the two signs.
        bootstrap = np.column_stack([np.r_[np.full(60, 11.), np.full(20, 7.)], np.ones(80)])
        low, high, se, _, _, zero_se = _simultaneous_bounds(np.array([10., 1.]), bootstrap)
        np.testing.assert_allclose(high, [13., 1.])
        np.testing.assert_allclose(low, [9., 1.])
        self.assertEqual(zero_se, 1)
        self.assertEqual(se[1], 0.)
        with self.assertRaisesRegex(ValueError, "zero bootstrap SE"):
            _simultaneous_bounds(np.array([2.]), np.ones((80, 1)))

    def test_identity_order_duplicates_and_out_of_support_cannot_be_ignored(self):
        candidates, control, truth, bounds, identifiers, names = self.fixture()
        for altered in (list(reversed(identifiers)), [*identifiers[:-1], identifiers[0]],
                        [identifier.replace("test:", "validation:") for identifier in identifiers]):
            bad = {"draws": control["draws"], "example_ids": altered}
            with self.subTest(ids=altered[:2]), self.assertRaises(ValueError):
                self.compare({"bad": bad}, control, truth, bounds, identifiers, names)
        for value in (-5., 5., np.nan):
            draws = np.copy(control["draws"]); draws[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare({"bad": {"draws": draws, "example_ids": identifiers}}, control,
                             truth, bounds, identifiers, names)
        with self.assertRaisesRegex(ValueError, "explicit ordered example_ids"):
            self.compare({"unidentified": control["draws"]}, control, truth, bounds, identifiers, names)


if __name__ == "__main__":
    unittest.main()
