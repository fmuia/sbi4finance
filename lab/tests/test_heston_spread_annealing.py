"""Schedule boundary, state-preserving reassignment and negative identity checks."""
import copy
from importlib import metadata
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import anneal_heston_spread_baseline as annealing
from scripts import train_heston_spread_baseline as trainer
from test_heston_spread_training import fixture


try:
    CORE = metadata.version("bayesflow") == "2.1.0"
except metadata.PackageNotFoundError:
    CORE = False


@unittest.skipUnless(CORE, "requires locked BF2.1 JAX environment")
class AnnealingTests(unittest.TestCase):
    def test_boundary_uses_first_rate_through_54999_then_second_at_55000(self):
        schedule = annealing.make_schedule()
        np.testing.assert_allclose([float(schedule(step)) for step in (50000, 54999, 55000, 59999, 60000)],
                                   [5e-6, 5e-6, 5e-7, 5e-7, 5e-7], rtol=1e-7)
        import keras
        serialized = keras.optimizers.schedules.serialize(schedule)
        restored = keras.optimizers.schedules.deserialize(serialized)
        self.assertEqual(float(schedule(55000)), float(restored(55000)))

    def test_public_schedule_assignment_preserves_every_state_array_and_adam_moment(self):
        import keras
        # A native optimizer with populated moments exercises the public setter
        # without running the scientific model or opening any bank/reference.
        variable = keras.Variable(np.array([.2, -.1], dtype=np.float32))
        optimizer = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(5e-4, 10000, alpha=.1))
        optimizer.apply_gradients([(np.array([.3, -.2], dtype=np.float32), variable)])
        optimizer.iterations.assign(50000)
        class Model:
            trainable_variables = [variable]
            non_trainable_variables = []
            metrics_variables = []
        model = Model()
        model.optimizer = optimizer
        class Workflow:
            estimator = model
        parent = {"optimizer_schedule": {"serialized": optimizer.get_config()["learning_rate"]}}
        before = [[np.array(value, copy=True) for value in group] for group in trainer.current_state(model)]
        proof = annealing.install_schedule(Workflow(), parent)
        self.assertTrue(proof["all_saved_array_hashes_unchanged_at_assignment"])
        for old, new in zip(before, trainer.current_state(model), strict=True):
            for a, b in zip(old, new, strict=True):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(int(optimizer.iterations), 50000)
        self.assertAlmostEqual(float(optimizer.learning_rate), 5e-6, places=11)

    def test_changed_parent_task_schedule_or_iteration_is_rejected(self):
        # A lightweight software fixture, with no training or ignored payloads.
        import keras
        with tempfile.TemporaryDirectory() as temporary:
            _, bank, task, _ = fixture(Path(temporary))
            _, bank_binding = trainer.load_clean_bank(bank, task, 128, smoke_test_only=True)
        source_sha = {name: trainer.digest(path) for name, path in annealing.source_paths().items()}
        parent = {"format": "heston_spread_baseline_checkpoint_v1", "task": task,
            "actual_optimizer_updates": 50000, "planned_updates": 50000,
            "dropout_zero_control": True, "train_rows": 20000, "batch_size": 128,
            "seed": 9011, "schedule_updates": 10000,
            "architecture": {"family": "spline", "widths": [128, 128], "subnet_dropout": 0.},
            "probe_rows": {"train": 512, "validation": 512}, "task_fingerprint": trainer.fingerprint(task),
            "names": task["target"]["parameter_names"], "bounds": task["target"]["prior_bounds"],
            "design_sha256": task["design_sha256"], "source_sha256": source_sha,
            "paired_default_reference": {"initial_deterministic_probes_exact": True},
            "optimizer_schedule": {"serialized": keras.optimizers.schedules.serialize(
                keras.optimizers.schedules.CosineDecay(5e-4, 10000, alpha=.1))}, **bank_binding}
        annealing.validate_parent(parent)
        for field in ("task", "schedule", "iteration", "bank"):
            bad = copy.deepcopy(parent)
            if field == "task":
                bad["task"]["target"]["fixed_c"] = 2.
            elif field == "schedule":
                bad["optimizer_schedule"]["serialized"]["config"]["decay_steps"] = 60000
            elif field == "iteration":
                bad["actual_optimizer_updates"] = 49999
            else:
                bad["bank_payload_sha256"] = "0"*64
                with self.assertRaises(ValueError):
                    annealing.continuation.validate_bank_binding(parent, bad)
                continue
            with self.assertRaises(ValueError):
                annealing.validate_parent(bad)


if __name__ == "__main__":
    unittest.main()
