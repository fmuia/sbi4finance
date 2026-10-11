"""Regression split, paired-estimand and immutable-baseline boundaries."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("hkde_surrogate_run_tested", ROOT / "scripts/run_hkde_surrogate.py")
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


class HKDESurrogateRunTests(unittest.TestCase):
    def test_named_splits_are_stable_separate_parameter_bases(self):
        bounds = [[0.025, 0.065], [0.25, 1.0]]
        train = run.parameters_for_split(bounds, 9321, "train", 32)
        validation = run.parameters_for_split(bounds, 9321, "validation", 32)
        test = run.parameters_for_split(bounds, 9321, "test", 32)
        assert_array_equal(train, run.parameters_for_split(bounds, 9321, "train", 32))
        # Extending one split must not shift other splits or earlier row identities.
        assert_array_equal(train, run.parameters_for_split(bounds, 9321, "train", 64)[:32])
        assert_array_equal(validation, run.parameters_for_split(bounds, 9321, "validation", 32))
        all_rows = np.concatenate([train, validation, test])
        self.assertEqual(len(np.unique(all_rows, axis=0)), 96)
        self.assertTrue(np.all(all_rows >= np.asarray(bounds)[:, 0]))
        self.assertTrue(np.all(all_rows < np.asarray(bounds)[:, 1]))
        self.assertFalse(np.array_equal(train, run.parameters_for_split(bounds, 9322, "train", 32)))
        with self.assertRaises(ValueError):
            run.parameters_for_split(bounds, 9321, "old_posterior_test", 32)

    def test_same_draw_quantiles_isolate_known_forward_price_shift(self):
        rng = np.random.default_rng(9323)
        direct = rng.normal(size=(3, 31, 2))
        offset = np.array([0.002, -0.001])
        predicted_q, direct_q, mean_shift = run.paired_propagation(direct + offset, direct, [.05, .5, .95])
        self.assertEqual(predicted_q.shape, (3, 2, 3))
        assert_allclose(predicted_q - direct_q, np.broadcast_to(offset[None, :, None], (3, 2, 3)), atol=1e-15)
        assert_allclose(mean_shift, np.broadcast_to(offset, (3, 2)), atol=1e-15)
        # Arbitrary bounded pointwise substitution has no larger empirical
        # quantile displacement, despite potentially changing price ordering.
        predicted = direct + rng.uniform(-0.003, 0.003, size=direct.shape)
        predicted_q, direct_q, _ = run.paired_propagation(predicted, direct, [.05, .5, .95])
        self.assertLessEqual(np.max(np.abs(predicted_q - direct_q)), 0.003 + 1e-15)

    def test_weighted_pushforward_preserves_joint_rows_and_case_weights(self):
        direct = np.array([[1., 100.], [3., 30.], [2., 60.], [4., 10.]])
        weights = np.array([[.6, .1, .2, .1], [.1, .2, .1, .6]])
        # The two price columns have reversed ordering. Independently sorting
        # parameters/weights or mixing case axes would fail this fixed example.
        offset = np.array([0.002, -0.001])
        predicted_q, direct_q, mean_shift = run.paired_propagation(
            direct + offset, direct, [.05, .5, .95], weights)
        assert_array_equal(direct_q, [[[1., 1., 4.], [10., 100., 100.]],
                                     [[1., 4., 4.], [10., 10., 100.]]])
        assert_allclose(predicted_q - direct_q, np.broadcast_to(offset[None, :, None], (2, 2, 3)), atol=1e-14)
        assert_allclose(mean_shift, np.broadcast_to(offset, (2, 2)), atol=1e-14)

    def test_baseline_load_preserves_failed_inference_and_rejects_tampering(self):
        protocol = json.loads((ROOT / "configs/hkde_surrogate_protocol.json").read_text())
        protocol["baseline_protocol"] = str(ROOT / protocol["baseline_protocol"])
        protocol["task_config"] = str(ROOT / protocol["task_config"])
        task = json.loads(Path(protocol["task_config"]).read_text())["task"]
        source = ROOT / protocol["baseline_results"]
        names = ("reference.json", "reference-grid.npz", "neural-validation.json", "validation-checks.npz")
        original_hashes = {name: run.sha(source / name) for name in names}
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary)
            for name in names:
                shutil.copyfile(source / name, copied / name)
            protocol["baseline_results"] = str(copied)
            _, checks, historical, hashes = run.load_baseline(protocol, task)
            self.assertFalse(historical["passed"])
            self.assertFalse(historical["reserved_test_evaluated"])
            self.assertEqual(hashes, original_hashes)
            self.assertEqual(checks["conditional_draws"].shape, (20, 4096, 2))
            # A stale/changed artifact must not be silently treated as the same
            # historical posterior experiment. Hash validation precedes loading.
            for name in ("reference-grid.npz", "validation-checks.npz"):
                with self.subTest(name=name):
                    with (copied / name).open("ab") as file:
                        file.write(b"changed")
                    with self.assertRaisesRegex(ValueError, "validation artifacts"):
                        run.load_baseline(protocol, task)
                    shutil.copyfile(source / name, copied / name)
            changed = deepcopy(historical)
            changed["reserved_test_evaluated"] = True
            (copied / "neural-validation.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "validation artifacts"):
                run.load_baseline(protocol, task)
        # Inspection did not rewrite the prior experiment or its final-test flag.
        self.assertEqual({name: run.sha(source / name) for name in names}, original_hashes)


if __name__ == "__main__":
    unittest.main()
