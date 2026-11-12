"""Quote units, verified model loading and concurrent experiment progress."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import heston_flexible_experiment as experiment


class FrozenWhitening:
    def __init__(self):
        self.inputs = []

    def transform(self, values):
        self.inputs.append(np.asarray(values).copy())
        return np.asarray(values) * 2 + 3


class ExperimentTests(unittest.TestCase):
    def test_raw_quotes_divide_by_half_spread_once_before_whitening(self):
        half_spread = np.arange(1., 64.)
        quotes = np.arange(126., dtype=float).reshape(2, 63)
        whitening = FrozenWhitening()
        task = {"records": {"half_spread": half_spread}}
        actual = experiment.condition_quotes(quotes, task, whitening)
        np.testing.assert_array_equal(whitening.inputs[0], quotes / half_spread)
        np.testing.assert_array_equal(actual, (quotes / half_spread) * 2 + 3)
        with self.assertRaisesRegex(ValueError, "masks"):
            experiment.condition_quotes(np.ma.array(quotes, mask=False), task, whitening)

    def test_training_uses_stored_y_over_h_without_dividing_again(self):
        task = {"target": {"parameter_names": list(experiment.NAMES), "quote_count": 63,
                           "geometry_and_spreads_sha256": "fixed-layout"},
                "records": {"half_spread": [100.] * 63}}
        arrays = {}
        for split, count in (("train", 32), ("validation", 100)):
            arrays.update({f"{split}_{name}": np.ones((count, 1)) for name in experiment.NAMES})
            arrays[f"{split}_observations"] = np.full((count, 63), 2.)
        whitening = FrozenWhitening()

        class BeforeFit(Exception):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bundle/bank").mkdir(parents=True)
            (root / "bundle/bank/whitening.json").write_text("{}")
            with (patch.object(experiment.delivery, "load_bank", return_value=(task, arrays, {"data_payload_sha256": "bank"})),
                  patch.object(experiment.delivery, "_whitening", return_value=whitening),
                  patch.object(experiment.inference, "build_workflow", side_effect=BeforeFit),
                  patch.object(experiment.metadata, "version", return_value="test"),
                  patch.dict("sys.modules", {"keras": SimpleNamespace(), "jax": SimpleNamespace()})):
                with self.assertRaises(BeforeFit):
                    experiment.train(root / "bundle", root / "fit", rows=32, epochs=1)
            self.assertEqual(len(whitening.inputs), 2)
            np.testing.assert_array_equal(whitening.inputs[0], np.full((32, 63), 2.))
            np.testing.assert_array_equal(whitening.inputs[1], np.full((100, 63), 2.))

    def test_corrupt_checkpoint_fails_before_workflow_build_or_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = {"target": {"parameter_names": list(experiment.NAMES), "quote_count": 63}}
            preprocessing = {"mask_support": False, "geometry_variation_support": False}
            experiment.write(root / "task.json", task)
            experiment.write(root / "preprocessing.json", preprocessing)
            model = root / "model.keras"
            model.write_bytes(b"original serialized estimator")
            run = {"format": experiment.FORMAT, "status": "complete", "task": task,
                   "configuration": experiment.inference.configuration("mlp", 8), "seed": 9011,
                   "task_sha256": experiment.digest(root / "task.json"),
                   "preprocessing_sha256": experiment.digest(root / "preprocessing.json"),
                   "source": {"inference_sha256": experiment.digest(experiment.inference.__file__)},
                   "versions": {name: experiment.metadata.version(name)
                                for name in ("bayesflow", "keras", "jax", "jaxlib", "numpy")},
                   "model_sha256": experiment.digest(model)}
            experiment.write(root / "run.json", run)
            model.write_bytes(b"damaged estimator")
            with patch.object(experiment.inference, "build_workflow") as build:
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    experiment.load_run(root)
                build.assert_not_called()

    def test_concurrent_study_updates_retain_all_runs_and_later_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def update(index, status):
                run = {"status": status, "configuration": {"representation": "mlp", "summary_dim": 8},
                       "train_rows": 32, "seed": index, "completed_epochs": int(status == "complete")}
                experiment.update_study(root, root / f"run-{index:03d}", run)

            with ThreadPoolExecutor(max_workers=6) as workers:
                list(workers.map(lambda index: update(index, "training"), range(24)))
                list(workers.map(lambda index: update(index, "complete"), range(8)))
            study = experiment.read(root / "study.json")
            self.assertEqual(len(study["runs"]), 24)
            by_seed = {run["seed"]: run for run in study["runs"]}
            self.assertEqual(set(by_seed), set(range(24)))
            for index, run in by_seed.items():
                self.assertEqual(run["status"], "complete" if index < 8 else "training")

    def test_invalid_allocations_and_representations_fail_before_reading_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arguments in ({"rows": True}, {"rows": 300001}, {"epochs": 0}, {"seconds": 0},
                              {"batch_size": 0}, {"representation": "set"}, {"summary_dim": 0}):
                with self.subTest(arguments=arguments), patch.object(experiment.delivery, "load_bank") as load:
                    with self.assertRaises(ValueError):
                        experiment.train(root / "bundle", root / "fit", **arguments)
                    load.assert_not_called()
                    self.assertFalse((root / "fit").exists())


if __name__ == "__main__":
    unittest.main()
