"""Train-only whitening, paired controls and portable native checkpoints."""

import copy
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_heston_spread_scale as scale
from scripts import train_heston_spread_whitening as training
from test_heston_spread_training import fixture


def _whitening_state(value):
    if isinstance(value, dict):
        if value.get("format") == "full_covariance_zca_whitening_v1":
            return value
        for child in value.values():
            found = _whitening_state(child)
            if found is not None:
                return found
    return None


class WhiteningDataTests(unittest.TestCase):
    def test_preparation_reads_only_named_development_arrays_and_is_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            design, bank, _, payload = fixture(Path(temporary))
            task, _ = scale.make_task(design)
            original_load, accesses = np.load, []

            class Archive:
                def __init__(self, *args, **kwargs):
                    self.archive = original_load(*args, **kwargs)
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.archive.close()
                def __getitem__(self, key):
                    accesses.append(key)
                    if key.startswith("test_"):
                        raise AssertionError("reserved arrays must remain unopened")
                    return self.archive[key]

            with patch.object(np, "load", side_effect=Archive):
                data, binding = training.prepare_data(bank, task, smoke_test_only=True)
            self.assertTrue(accesses)
            self.assertEqual(set(accesses), {f"{split}_{key}" for split in ("train", "validation")
                                             for key in ("parameters", "clean", "row_id")})
            self.assertFalse(binding["reserved_arrays_opened"])
            repeated, _ = training.prepare_data(bank, task, smoke_test_only=True)
            for split, count in (("train", 128), ("validation", 32)):
                self.assertEqual(set(data[split]), {"observations", *task["target"]["parameter_names"]})
                self.assertEqual(data[split]["observations"].shape, (count, 54))
                for key, value in data[split].items():
                    np.testing.assert_array_equal(value, repeated[split][key])
                for index, name in enumerate(task["target"]["parameter_names"][:5]):
                    np.testing.assert_array_equal(data[split][name][:, 0], payload[f"{split}_parameters"][:, index])
                self.assertTrue(((data[split]["c"] > .5) & (data[split]["c"] < 2.)).all())

    def test_preprocessing_uses_training_moments_retains_all_features_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            design, bank, _, _ = fixture(Path(temporary))
            task, _ = scale.make_task(design)
            data, binding = training.prepare_data(bank, task, smoke_test_only=True)
            train = data["train"]["observations"]
            for mode in ("diagonal", "zca"):
                changed = {key: value.copy() for key, value in data["train"].items()}
                changed["observations"][0, 0] += 1.
                with self.assertRaises(ValueError):
                    training.fit_preprocessing(task, changed, binding, mode)
                record = training.fit_preprocessing(task, data["train"], binding, mode)
                fitted = training.preprocessing_transform(record)
                transformed = fitted.transform(train)
                self.assertEqual(transformed.shape, train.shape)
                self.assertEqual(transformed.dtype, np.float64)
                np.testing.assert_allclose(transformed.mean(axis=0), 0., atol=1e-10)
                np.testing.assert_allclose(fitted.inverse(transformed), train, rtol=2e-10, atol=2e-10)
                if mode == "zca":
                    state = _whitening_state(record)
                    self.assertIsNotNone(state)
                    self.assertEqual(state["training_rows"], 128)
                    np.testing.assert_array_equal(state["mean"], train.mean(axis=0))
                    if np.all(np.asarray(state["eigenvalues"]) > max(state["eigenvalues"]) * 1e-8):
                        np.testing.assert_allclose(np.cov(transformed, rowvar=False), np.eye(54), atol=2e-8)
                    bad = copy.deepcopy(record)
                    matrix = _whitening_state(bad)["whitening_matrix"]
                    matrix[0][0] = -abs(matrix[0][0])
                    with self.assertRaises(ValueError):
                        training.preprocessing_transform(bad)
                else:
                    np.testing.assert_allclose(np.diag(np.cov(transformed, rowvar=False)), 1., atol=1e-10)
                # Validation can be arbitrarily different; fitted training moments stay fixed.
                before = json.dumps(record, sort_keys=True)
                data["validation"]["observations"][:] = 1e9
                self.assertEqual(json.dumps(training.fit_preprocessing(task, data["train"], binding, mode), sort_keys=True), before)


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE, "requires locked BF2.1 environment")
class WhiteningWorkflowTests(unittest.TestCase):
    def test_matched_initialization_cli_checkpoint_sampling_and_next_update(self):
        import jax
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            design, bank, _, _ = fixture(root)
            task, _ = scale.make_task(design)
            task_path = root / "task.json"
            task_path.write_text(json.dumps(task))
            arms = [{"mode": mode, "contract": training.protocol_contract(bank, task, mode=mode, smoke_test_only=True),
                     "output": str(root / mode)} for mode in ("diagonal", "zca")]
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"format": "heston_spread_whitening_study_v1", "arms": arms}))
            for mode in ("diagonal", "zca"):
                training.run_training(bank, task, root / mode, mode=mode, protocol=protocol, smoke_test_only=True)
            runs = [json.loads((root / mode / "run.json").read_text()) for mode in ("diagonal", "zca")]
            self.assertEqual(runs[0]["initial_trainable_hashes"], runs[1]["initial_trainable_hashes"])
            self.assertTrue(runs[0]["initial_trainable_hashes"])
            checkpoint = root / "zca/checkpoints/update-000005"
            wrapper, record = training.load_checkpoint(checkpoint)
            self.assertEqual(record["actual_optimizer_updates"], 5)
            self.assertIn(wrapper.estimator.standardizer.standardize, ("inference_variables", ["inference_variables"]))
            data, _ = training.prepare_data(bank, task, smoke_test_only=True)
            raw_conditions = {"observations": data["validation"]["observations"][:2]}
            preprocessing = json.loads((checkpoint / "preprocessing.json").read_text())
            fitted = training.preprocessing_transform(preprocessing)
            automatic = wrapper.sample(conditions=raw_conditions, num_samples=16, sample_shape=(), seed=77)
            manual = wrapper.workflow.sample(conditions={"observations": fitted.transform(raw_conditions["observations"])},
                                             num_samples=16, sample_shape=(), seed=77)
            for name in task["target"]["parameter_names"]:
                np.testing.assert_array_equal(automatic[name], manual[name])

            cli_protocol = root / "cli-protocol.json"
            cli_protocol.write_text(json.dumps({"format": "heston_spread_whitening_study_v1", "arms": [
                {"mode": "zca", "contract": arms[1]["contract"], "output": str(root / "cli")}] }))
            completed = subprocess.run([sys.executable, str(Path(training.__file__).resolve()), "--bank", str(bank),
                "--task", str(task_path), "--out", str(root / "cli"), "--mode", "zca", "--protocol", str(cli_protocol),
                "--smoke-test-only", "--worker"], capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
            cli_checkpoint = root / "cli/checkpoints/update-000005"
            imported, _ = training.load_checkpoint(cli_checkpoint)
            repeated = imported.sample(conditions=raw_conditions, num_samples=16, sample_shape=(), seed=77)
            for name in task["target"]["parameter_names"]:
                np.testing.assert_array_equal(automatic[name], repeated[name])
            # Restore both full states afresh after sampling before comparing the next update.
            left, _ = training.load_checkpoint(checkpoint)
            right, _ = training.load_checkpoint(cli_checkpoint)
            raw_batch = {name: values[:32] for name, values in data["train"].items()}
            adapted = training.adapted_data(task, raw_batch, preprocessing)
            results = [scale.fixed.compiled_step(item.estimator)(scale.fixed.current_state(item.estimator), adapted)
                       for item in (left, right)]
            self.assertTrue(all(bool(value) for value in results[0][-1].values()))
            for a, b in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
                np.testing.assert_array_equal(a, b)
            # The fitted preprocessor is part of the checkpoint identity.
            path = cli_checkpoint / "preprocessing.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaises(ValueError):
                training.load_checkpoint(cli_checkpoint)


if __name__ == "__main__":
    unittest.main()
