import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import norm, truncnorm

from sbi4finance import heston
from sbi4finance.path_inference import bank, teaching_reference as ref
from sbi4finance.rough_bergomi import rough_driver


class TeachingReferenceTests(unittest.TestCase):
    def test_gaussian_drift_integral_and_moments_against_quadrature(self):
        y = np.array([-.018, .007, .003])
        integrated = np.array([[.0001, .0002, .00015], [.000003, .000002, .000004]])
        leverage = np.array([[.004, -.002, .001], [-.0005, .001, .0002]])
        rho = np.array([-.6, -.9])
        dt, bounds = 1 / 252, [-.1, .2]
        log_likelihood, location, scale, mean, second = ref.integrate_uniform_drift(y, integrated, leverage, rho, dt, bounds)
        for row in range(2):
            variance = (1 - rho[row] ** 2) * integrated[row]
            offset = -.5 * integrated[row] + rho[row] * leverage[row]
            def density(mu):
                return np.exp(np.sum(norm.logpdf(y, loc=mu * dt + offset, scale=np.sqrt(variance)))) / .3
            mass = quad(density, *bounds, epsabs=1e-100, epsrel=1e-11)[0]
            first = quad(lambda mu: mu * density(mu), *bounds, epsabs=1e-100, epsrel=1e-11)[0] / mass
            second_q = quad(lambda mu: mu ** 2 * density(mu), *bounds, epsabs=1e-100, epsrel=1e-11)[0] / mass
            assert_allclose(log_likelihood[row], np.log(mass), rtol=1e-11, atol=1e-10)
            assert_allclose(mean[row], first, rtol=1e-9, atol=1e-11)
            assert_allclose(second[row], second_q, rtol=1e-8, atol=1e-11)
            a, b = (bounds[0] - location[row]) / scale[row], (bounds[1] - location[row]) / scale[row]
            self.assertTrue(0 <= truncnorm.cdf(mean[row], a, b, loc=location[row], scale=scale[row]) <= 1)

    def test_tail_interval_mass_and_almost_flat_drift_moments(self):
        for a, b in [(-2, 3), (10, 10.01), (-40, -39), (-1e-5, 2e-5)]:
            answer = ref._log_interval_probability(np.array([a]), np.array([b]))[0]
            mode = min(max(0, a), b)
            scaled = quad(lambda x: np.exp(-.5 * (x ** 2 - mode ** 2)), a, b, epsabs=1e-15)[0]
            expected = np.log(scaled) - .5 * mode ** 2 - .5 * np.log(2 * np.pi)
            assert_allclose(answer, expected, atol=2e-11)
        # Almost no information about drift: its mean/variance must reproduce
        # a finite uniform prior without cancellation of large Normal moments.
        answer = ref.integrate_uniform_drift(np.array([0.0]), np.array([[1e6]]),
                                            np.array([[-1e6]]), np.array([-.5]), 1 / 252, [-.1, .2])
        self.assertTrue(np.isfinite(answer[0]).all())
        # Very large variance still includes the model's -I/2 location term;
        # compare moments to smooth direct integration rather than assume flat.
        loc, scale = answer[1][0], answer[2][0]
        f = lambda mu: np.exp(-.5 * ((mu - loc) / scale) ** 2)
        mass = quad(f, -.1, .2)[0]
        assert_allclose(answer[3][0], quad(lambda mu: mu * f(mu), -.1, .2)[0] / mass, atol=2e-10)

    def test_heston_prescribed_innovations_match_bank_simulator(self):
        task = bank.teaching_task("heston", observations=21)
        rng = np.random.default_rng(18)
        p = ref.sample_nondrift_prior(task, 3, rng)
        z = rng.normal(size=(3, 21, 8))
        integrated, leverage = ref.conditional_statistics(task, p, z)
        normals = np.zeros((3, 21, 8, 2))
        normals[..., 0] = z
        drift = np.array([-.08, .02, .17])
        result = heston.simulate(dict(kappa=p[:, 0], theta=p[:, 1], vol_of_vol=p[:, 2], rho=p[:, 3], drift=drift),
                                 np.full(21, 1 / 252), .04,
                                 dict(brownian=normals, jump_counts=np.zeros((3, 21), dtype=np.int64),
                                      jump_normals=np.zeros((3, 21))))
        expected = drift[:, None] / 252 - .5 * integrated + p[:, 3:4] * leverage
        assert_allclose(result["log_returns"][..., 0], expected, rtol=2e-13, atol=2e-17)
        self.assertTrue(np.all(integrated[:, 0] > 0))

    def test_rough_statistics_match_native_driver_and_daily_aggregation(self):
        rng = np.random.default_rng(319)
        for model in ("rough-known", "rough-joint"):
            task = bank.teaching_task(model, observations=21)
            p = ref.sample_nondrift_prior(task, 3, rng)
            z = rng.normal(size=(3, 21, 4))
            bridge = rng.normal(size=z.shape)
            integrated, leverage = ref.conditional_statistics(task, p, z, bridge)
            for row in range(3):
                innovation = np.zeros((1, 84, 3))
                innovation[0, :, 0] = z[row].reshape(-1)
                innovation[0, :, 1] = bridge[row].reshape(-1)
                driver, variance = rough_driver(p[row, 0], 1 / 1008, innovation)
                left_driver, left_comp = np.r_[0, driver[0, :-1]], np.r_[0, variance[:-1]]
                curve = (.04 if model == "rough-known" else
                         p[row, 4] + (p[row, 3] - p[row, 4]) * np.exp(-p[row, 5] * np.arange(84) / 1008))
                left = curve * np.exp(p[row, 1] * left_driver - .5 * p[row, 1] ** 2 * left_comp)
                assert_allclose(integrated[row], (left / 1008).reshape(21, 4).sum(axis=1), rtol=2e-13, atol=1e-18)
                assert_allclose(leverage[row], (np.sqrt(left / 1008) * z[row].reshape(-1)).reshape(21, 4).sum(axis=1), rtol=2e-13, atol=1e-17)
                fine_returns = (.07 - .5 * left) / 1008 + p[row, 2] * np.sqrt(left / 1008) * z[row].reshape(-1)
                expected = .07 / 252 - .5 * integrated[row] + p[row, 2] * leverage[row]
                assert_allclose(fine_returns.reshape(21, 4).sum(axis=1), expected, atol=2e-17)

    def test_heston_multiday_atoms_are_never_floored_or_omitted(self):
        task = bank.teaching_task("heston", observations=21)
        p = np.array([[.25, .01, .6, -.6]])
        z = np.zeros((1, 21, 8))
        z[0, 0, 0] = -100
        integrated, leverage = ref.conditional_statistics(task, p, z)
        self.assertGreater(integrated[0, 0], 0)
        self.assertEqual(integrated[0, 1], 0)
        with self.assertRaisesRegex(ValueError, "atoms"):
            ref.integrate_uniform_drift(np.zeros(21), integrated, leverage, p[:, 3], 1 / 252, [-.1, .2])
        with self.assertRaisesRegex(ValueError, "atom-aware"):
            ref.importance_reference(task, np.zeros(21), samples=8, seed=11)
        answer = ref.atom_support_diagnostic(task, np.zeros(21), samples=128, seed=19)
        self.assertEqual(answer["status"], "posterior_reference_inconclusive")
        self.assertNotIn("posterior_mean", answer)

    def test_short_reference_is_chunk_invariant_and_joint_draws_in_support(self):
        for model in bank.MODELS:
            task = bank.teaching_task(model, observations=21)
            first, arrays = ref.importance_reference(task, np.array([.006]), samples=192, seed=79, chunk=19, draws=128)
            second, second_arrays = ref.importance_reference(task, np.array([.006]), samples=192, seed=79, chunk=128, draws=128)
            for name in arrays:
                assert_allclose(arrays[name], second_arrays[name], rtol=3e-12, atol=1e-13)
            assert_allclose(first["posterior_mean"], second["posterior_mean"], atol=1e-13)
            bounds = np.asarray(task["prior_bounds"])
            self.assertTrue(np.all(arrays["posterior_draws"] >= bounds[:, 0]))
            self.assertTrue(np.all(arrays["posterior_draws"] <= bounds[:, 1]))
            self.assertAlmostEqual(arrays["weights"].sum(), 1)
            self.assertEqual(first["observations_sha256"], ref.observations_sha256(np.array([.006])))

    def test_legacy_tasks_invalid_and_precision_gates_not_relaxed(self):
        with self.assertRaisesRegex(ValueError, "inferred-drift"):
            ref.importance_reference(bank.default_task("heston"), [.01], samples=8, seed=8)
        task = bank.teaching_task("rough-known", observations=21)
        runs = [ref.importance_reference(task, [.002], samples=128, seed=i, draws=8)[0] for i in [1, 2]]
        result = ref.compare_runs(runs)
        self.assertEqual(result["status"], "inconclusive_monte_carlo_precision")
        self.assertFalse(result["scientific_acceptance"])
        self.assertEqual(result["gates"]["minimum_ess"], 1000)

    def test_actual_cli_bank_loading_limits_and_bounded_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_path_teaching_reference.py"
        spec = importlib.util.spec_from_file_location("check_path_teaching_reference", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = bank.teaching_task("rough-joint", observations=21)
            bank.write_bank(task, root / "bank", train=0, validation=2, test=0, chunk=1, seed=31)
            loaded, source, rows, _ = cli.load_inputs(bank_path=root / "bank", cases=[0, 1])
            self.assertEqual(loaded, task)
            self.assertEqual(source["split"], "validation")
            self.assertEqual(set(rows), {0, 1})
            with self.assertRaisesRegex(ValueError, "inspected"):
                cli.load_inputs(bank_path=root / "bank", cases=[1024])
            args = SimpleNamespace(bank=root / "bank", replay=None, cases=[0, 1], lengths=[1, 21],
                                   samples=[32, 64], replicates=2, chunk=16, draws=16,
                                   seed=900, max_seconds=60, out=root / "output")
            result = cli.run(args)
            self.assertEqual(result["status"], "bounded_diagnostic_complete")
            self.assertEqual(len(result["cases"]), 2)
            self.assertFalse(result["scientific_acceptance"])
            for item in result["cases"].values():
                for length in item["lengths"].values():
                    self.assertEqual(len(length["runs"]), 4)
                    self.assertEqual(length["comparison"]["status"], "inconclusive_monte_carlo_precision")
            saved = json.loads((root / "output/reference-summary.json").read_text())
            self.assertEqual(saved["task_sha256"], bank.task_fingerprint(task))


if __name__ == "__main__":
    unittest.main()
