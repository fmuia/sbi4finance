"""Scientific checks for signed-state Heston and Gaussian price jumps."""

from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.heston import draw_innovations, simulate


def parameters(**changes):
    result = {
        "drift": 0.08,
        "kappa": 1.0,
        "theta": 0.04,
        "vol_of_vol": 0.5,
        "rho": -0.7,
    }
    result.update(changes)
    return result


def noise(brownian):
    brownian = np.asarray(brownian, dtype=np.float64)
    shape = brownian.shape[:2]
    return {
        "brownian": brownian,
        "jump_counts": np.zeros(shape, dtype=np.int64),
        "jump_normals": np.zeros(shape, dtype=np.float64),
    }


class HestonTests(unittest.TestCase):
    def test_full_truncation_retains_negative_state_and_handles_rho_edges(self):
        innovations = noise(np.tile([-3.0, 2.0], (3, 2, 1, 1)))
        result = simulate(
            parameters(rho=[1.0, -1.0, 0.0]),
            [0.5, 0.5], [0.01, 0.01, -0.03], innovations, backend="numpy",
        )
        # First two paths cross zero. Their negative auxiliary remains stored;
        # on the next step its effective variance is zero in both coefficients.
        expected_state = [
            [0.01, -0.08106601717798213, -0.06106601717798213],
            [0.01, -0.08106601717798213, -0.06106601717798213],
            [-0.03, -0.01, 0.01],
        ]
        assert_allclose(result["variance_state"][:, :, 0], expected_state, atol=1e-14)
        assert_allclose(
            result["log_returns"][:, :, 0],
            [[-0.17463203435596425, 0.04], [0.24963203435596425, 0.04], [0.04, 0.04]],
            atol=1e-14,
        )
        self.assertEqual(result["log_returns"].shape, (3, 2, 1))
        self.assertEqual(result["variance_state"].shape, (3, 3, 1))

    def test_constant_variance_matches_exact_gbm_with_the_same_noise(self):
        dt = np.array([0.1, 0.2, 0.4])
        innovations = draw_innovations(dt, 3, seed=41, example_ids=[8, 9])
        result = simulate(
            parameters(drift=[0.03, 0.08], vol_of_vol=0.0, rho=-0.6),
            dt, 0.04, innovations, backend="numpy",
        )
        z = innovations["brownian"]
        gbm_noise = (-0.6 * z[..., 0] + 0.8 * z[..., 1]).sum(axis=2) / np.sqrt(3)
        expected = (
            (np.array([0.03, 0.08])[:, None] - 0.02) * dt
            + 0.2 * np.sqrt(dt) * gbm_noise
        )
        assert_allclose(result["log_returns"][:, :, 0], expected, rtol=0, atol=1e-13)
        assert_allclose(result["variance_state"], 0.04, rtol=0, atol=1e-14)

    def test_numpy_numba_and_segmented_restart_preserve_the_same_transition(self):
        dt = np.array([0.1, 0.2, 0.3])
        innovations = draw_innovations(
            dt, 2, seed=31, example_ids=[0, 1, 2], jump_intensity=2.0
        )
        innovations["brownian"][0, 0, 0, 0] = -5.0
        model = parameters(
            kappa=1.2, vol_of_vol=0.8,
            jump_intensity=2.0, jump_mean=-0.08, jump_std=0.03,
        )
        initial = [0.0001, -0.05, 0.04]
        whole = simulate(model, dt, initial, innovations, backend="numpy")
        self.assertLess(whole["variance_state"][0, 1, 0], 0.0)
        for backend in ("numpy", "numba"):
            with self.subTest(backend=backend):
                full = simulate(model, dt, initial, innovations, backend=backend)
                first = simulate(
                    model, dt[:1], initial,
                    {key: value[:, :1] for key, value in innovations.items()},
                    backend=backend,
                )
                rest = simulate(
                    model, dt[1:], first["variance_state"][:, -1, 0],
                    {key: value[:, 1:] for key, value in innovations.items()},
                    backend=backend,
                )
                for key in whole:
                    assert_allclose(full[key], whole[key], rtol=0, atol=1e-13)
                assert_allclose(
                    np.concatenate([first["log_returns"], rest["log_returns"]], axis=1),
                    whole["log_returns"], rtol=0, atol=1e-13,
                )
                assert_allclose(
                    np.concatenate([first["variance_state"], rest["variance_state"][:, 1:]], axis=1),
                    whole["variance_state"], rtol=0, atol=1e-13,
                )

    def test_one_step_moments_and_leverage_match_the_gaussian_transition(self):
        count = 50_000
        h = 1e-4
        innovations = noise(np.random.default_rng(182).standard_normal((count, 1, 1, 2)))
        result = simulate(
            parameters(kappa=1.5, theta=0.05), [h], 0.04,
            innovations, backend="numpy",
        )
        sample = np.column_stack((
            result["log_returns"][:, 0, 0],
            result["variance_state"][:, 1, 0] - 0.04,
        ))
        expected_mean = np.array([6e-6, 1.5e-6])
        expected_covariance = np.array([[4e-6, -1.4e-6], [-1.4e-6, 1e-6]])
        variances = np.diag(expected_covariance)
        self.assertTrue(np.all(
            np.abs(sample.mean(axis=0) - expected_mean) < 5.5 * np.sqrt(variances / count)
        ))
        covariance_error = 5.5 * np.sqrt(
            (np.outer(variances, variances) + expected_covariance**2) / (count - 1)
        )
        self.assertTrue(np.all(
            np.abs(np.cov(sample, rowvar=False) - expected_covariance) < covariance_error
        ))

    def test_poisson_multiple_jumps_and_compensated_price_expectation(self):
        count = 30_000
        innovations = draw_innovations(
            [0.5], 1, seed=916, example_ids=range(count), jump_intensity=4.0
        )
        counts = innovations["jump_counts"][:, 0]
        self.assertEqual(counts.dtype, np.dtype("int64"))
        self.assertGreater(counts.max(), 1)
        self.assertLess(abs(counts.mean() - 2.0), 5.5 * np.sqrt(2.0 / count))
        self.assertLess(abs(counts.var() - 2.0), 5.5 * np.sqrt(10.0 / count))
        result = simulate(
            parameters(
                drift=0.03, theta=0.0, vol_of_vol=0.0,
                jump_intensity=4.0, jump_mean=0.04, jump_std=0.08,
            ),
            [0.5], 0.0, innovations, backend="numpy",
        )
        log_returns = result["log_returns"][:, 0, 0]
        expected_log_mean = 0.015 + 2 * (0.04 - np.expm1(0.04 + 0.08**2 / 2))
        expected_log_variance = 2 * (0.04**2 + 0.08**2)
        self.assertLess(
            abs(log_returns.mean() - expected_log_mean),
            5.5 * np.sqrt(expected_log_variance / count),
        )
        fourth_cumulant = 2 * (0.04**4 + 6 * 0.04**2 * 0.08**2 + 3 * 0.08**4)
        self.assertLess(
            abs(log_returns.var() - expected_log_variance),
            5.5 * np.sqrt((fourth_cumulant + 2 * expected_log_variance**2) / count),
        )
        gross_returns = np.exp(log_returns)
        self.assertLess(
            abs(gross_returns.mean() - np.exp(0.015)),
            5.5 * gross_returns.std(ddof=1) / np.sqrt(count),
        )
        assert_array_equal(result["variance_state"], np.zeros((count, 2, 1)))

    def test_innovation_identity_does_not_depend_on_batch_order(self):
        kwargs = {"seed": 209, "jump_intensity": 3.0}
        original = draw_innovations([0.2, 0.3], 2, example_ids=[4, 8, 9], **kwargs)
        reordered = draw_innovations([0.2, 0.3], 2, example_ids=[9, 4, 8], **kwargs)
        separate = draw_innovations([0.2, 0.3], 2, example_ids=[8], **kwargs)
        other_stream = draw_innovations(
            [0.2, 0.3], 2, example_ids=[4, 8, 9], stream_id=1, **kwargs
        )
        for key, value in original.items():
            assert_array_equal(reordered[key], value[[2, 0, 1]])
            assert_array_equal(separate[key], value[1:2])
        self.assertFalse(np.array_equal(original["brownian"], other_stream["brownian"]))
        self.assertFalse(np.array_equal(original["jump_normals"], other_stream["jump_normals"]))

    def test_invalid_inputs_and_unsupported_backend_fail_explicitly(self):
        valid_noise = noise(np.zeros((1, 1, 1, 2)))
        wrong_shape = deepcopy(valid_noise)
        wrong_shape["brownian"] = np.zeros((1, 1, 1, 1))
        invalid_calls = {
            "rho outside correlation range": lambda: simulate(parameters(rho=1.01), [0.1], 0.04, valid_noise, backend="numpy"),
            "nonpositive interval": lambda: simulate(parameters(), [0.0], 0.04, valid_noise, backend="numpy"),
            "wrong innovation shape": lambda: simulate(parameters(), [0.1], 0.04, wrong_shape, backend="numpy"),
            "unknown backend": lambda: simulate(parameters(), [0.1], 0.04, valid_noise, backend="gpu"),
            "zero substeps": lambda: draw_innovations([0.1], 0, seed=1, example_ids=[0]),
            "duplicate IDs": lambda: draw_innovations([0.1], 1, seed=1, example_ids=[0, 0]),
            "negative intensity": lambda: draw_innovations([0.1], 1, seed=1, example_ids=[0], jump_intensity=-1.0),
        }
        for name, call in invalid_calls.items():
            with self.subTest(name=name):
                with self.assertRaises((ValueError, TypeError)):
                    call()


if __name__ == "__main__":
    unittest.main()
