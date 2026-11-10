"""Saved joint-c samples remain immutable when later references are compared."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import compare_heston_spread_scale as comparison
from scripts import train_heston_spread_scale as trainer
from test_heston_spread_training import fixture


class SavedScaleComparisonTests(unittest.TestCase):
    """Software fixtures exercise binding guards without fitting or pricing."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        design, _, _, _ = fixture(self.root)
        self.task, _ = trainer.make_task(design)
        self.target = self.task["target"]
        self.bounds = np.asarray(self.target["prior_bounds"])
        self.fixed = comparison.fixed
        self.exported = comparison.exported
        self.panel_path = self.root / "panel.json"
        self.panel = {
            "joint_task": self.target,
            "products": {"control": [0., .1], "affected": [.3, .9]},
            "cases": [{"id": case, "observations": (np.arange(54) + index).tolist()}
                      for index, case in enumerate(self.exported.CASES)],
        }
        self.write(self.panel_path, self.panel)
        self.study_path = self.root / "study.json"
        self.study = {
            "format": "heston_spread_noise_scale_study_v1", "joint_task": self.target,
            "panel": self.artifact(self.panel_path),
            "joint_evaluation": {
                "seed": 261112, "population_rows": 512,
                "posterior_draws_per_population_case": 512, "conditional_draws": 4096,
                "family_checks": 144, "mean_tolerance_reference_sd": .3,
                "width_ratio90_limits": [.75, 1.25], "block_bootstrap_replicates": 400,
                "population_c": "independent Uniform(.5,2) per validation surface: draw_c(512,0,261112,namespace=3)",
                "population_gaussian": "reuse PCG64 SeedSequence([0x53504556,261112]) standard errors; apply generated c*h",
                "checkpoints": [10000, 50000, 60000],
                "conditional_cases": list(self.exported.CASES),
            },
        }
        self.write(self.study_path, self.study)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "model.keras").write_bytes(b"opaque model: never loaded")
        (self.checkpoint / "state.npz").write_bytes(b"opaque state: never loaded")
        metadata = {
            "task": self.task, "bank_record_sha256": "manifest", "bank_payload_sha256": "payload",
            "train_rows": 20000, "seed": 9011, "batch_size": 128, "planned_updates": 60000,
        }
        contract = trainer.protocol_contract(self.task, metadata)
        training_protocol = self.root / "training-protocol.json"
        self.write(training_protocol, contract)
        metadata.update(protocol_path=str(training_protocol), protocol_sha256=self.fixed.digest(training_protocol),
                        recipe=contract)
        self.write(self.checkpoint / "checkpoint.json", metadata)

        self.rng = np.random.default_rng(48)
        self.neural = {case: self.draws((4096, 6)).astype(np.float32)
                       for case in self.exported.CASES}
        self.payload = self.root / "conditional-draws.npz"
        np.savez(self.payload, **self.neural)
        self.export_dir = self.root / "export"
        self.export_dir.mkdir()
        self.quantiles = self.export_dir / "recovery.npz"
        np.savez(self.quantiles, truth=np.ones((512, 6)), quantiles=np.ones((3, 512, 6)))
        source = self.root / "export-source.py"
        source.write_text("# frozen software fixture\n")
        self.source = source
        self.record = {
            "format": "heston_spread_joint_scale_evaluation_v1", "status": "complete",
            "task": self.target, "study_protocol_sha256": self.fixed.digest(self.study_path),
            "seed": 261112, "family_checks": 144, "actual_optimizer_updates": 60000,
            "conditional_draws": {**self.artifact(self.payload), "draws_per_case": 4096},
            "population": {"recovery_sha256": self.fixed.digest(self.quantiles),
                           "ranks": [[.25, .75]], "coverage": {"90": [.9] * 6},
                           "do_not_recompute": {"sentinel": "original population"}},
            "checkpoint": str(self.checkpoint),
            "checkpoint_record_sha256": self.fixed.digest(self.checkpoint / "checkpoint.json"),
            "model_sha256": self.fixed.digest(self.checkpoint / "model.keras"),
            "state_sha256": self.fixed.digest(self.checkpoint / "state.npz"),
            "source_sha256": {"exporter": self.artifact(source)},
            "cases": {case: {"status": "inconclusive", "reason": "reference not yet ready",
                             "observation_sha256": comparison.baseline.observations_digest(entry["observations"]),
                             "neural_quantiles_05_50_95": np.quantile(self.neural[case], [.05, .5, .95], axis=0).tolist()}
                      for case, entry in zip(self.exported.CASES, self.panel["cases"], strict=True)},
        }
        self.neural_path = self.export_dir / "evaluation.json"
        self.write(self.neural_path, self.record)

        self.chains, self.reference_directories = {}, {}
        jobs = []
        for case, entry in zip(self.exported.CASES, self.panel["cases"], strict=True):
            directory = self.root / case
            directory.mkdir()
            self.reference_directories[case] = directory
            chains = self.draws((2, 64, 6))
            self.chains[case] = chains
            chain_path = directory / "chains.npz"
            np.savez(chain_path, parameters=chains)
            diagnostics = {"reference_ready": True, "chain_payload_sha256": self.fixed.digest(chain_path)}
            self.write(directory / "diagnostics.json", diagnostics)
            self.write(directory / "run.json", {
                "status": "completed", "chain_payload_path": str(chain_path), "task": self.target,
                "names": self.target["parameter_names"], "bounds": self.target["prior_bounds"],
                "case_id": case, "design_sha256": "design-sha", "panel_sha256": self.fixed.digest(self.panel_path),
                "observation_sha256": comparison.baseline.observations_digest(entry["observations"]),
                "observations": entry["observations"], "products": self.panel["products"],
                "selected_attempt": 0,
                "attempts": [{"draws": 64, "diagnostics_sha256": self.fixed.digest(directory / "diagnostics.json")}],
                "settings": {"chains": 2},
            })
            jobs.append({"id": case + "-joint", "case_id": case, "mode": "joint", "evidence_out": str(directory)})
        self.reference_protocol = self.root / "reference-protocol.json"
        self.write(self.reference_protocol, {"design_sha256": "design-sha", "jobs": jobs})
        self.progress_path = self.root / "progress.json"
        self.write(self.progress_path, {"status": "completed"})
        self.recovery_path = self.root / "reference-recovery.json"
        self.write(self.recovery_path, {"progress_path": str(self.progress_path),
                                       "recovery_reference_protocol": self.artifact(self.reference_protocol)})

    def write(self, path, value):
        self.fixed.write_json(path, value)

    def artifact(self, path):
        return {"path": str(path), "sha256": self.fixed.digest(path)}

    def draws(self, shape):
        return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], shape)

    def read(self, path):
        return json.loads(path.read_text())

    def run_comparison(self, name="comparison", compare_draws=None):
        # Recovery construction and its startup-only audit are tested separately.
        # Keep the frozen target/observation/chain binding and training guards real.
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.exported, "selected_reference_protocol",
                return_value=(self.reference_protocol, {})))
            stack.enter_context(patch.object(self.exported.scale_reference, "validate_protocol"))
            for owner, method in ((trainer, "sample"), (trainer, "load_checkpoint"),
                                  (self.exported.validation, "sbc"), (self.exported.validation, "coverage")):
                stack.enter_context(patch.object(owner, method,
                    side_effect=AssertionError("saved-draw comparison must not call " + method)))
            metric = stack.enter_context(patch.object(comparison.baseline, "compare_draws",
                side_effect=compare_draws or (lambda *args, **kwargs: {"status": "pass"})))
            result = comparison.compare(self.neural_path, self.study_path, self.recovery_path, self.root / name)
        return result, metric

    def test_exact_saved_draws_compared_and_population_copied_without_recomputation(self):
        before = self.fixed.digest(self.neural_path)
        received = []

        def inspect(neural, chains, diagnostics, names, products, **kwargs):
            index = len(received)
            case = self.exported.CASES[index]
            np.testing.assert_array_equal(neural, self.neural[case])
            np.testing.assert_array_equal(chains, self.chains[case])
            self.assertEqual(names, self.target["parameter_names"])
            self.assertEqual(products, self.panel["products"])
            self.assertEqual(kwargs, {"seed": 261112 + 100 + index, "replicates": 400, "family_checks": 144})
            received.append(case)
            return {"status": "pass", "fixture": "same saved samples"}

        result, metric = self.run_comparison(compare_draws=inspect)
        self.assertEqual(received, list(self.exported.CASES))
        self.assertEqual(metric.call_count, 3)
        self.assertEqual(result["population"], self.record["population"])
        self.assertEqual(self.fixed.digest(self.root / "comparison" / "recovery.npz"), self.fixed.digest(self.quantiles))
        self.assertEqual(self.fixed.digest(self.neural_path), before)
        self.assertFalse(result["posterior_resampling"])
        self.assertFalse(result["population_recomputed"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertTrue(result["protected_comparison_inputs_unchanged"])
        self.assertEqual(result["conditional_status"], "pass")
        for case in self.exported.CASES:
            self.assertAlmostEqual(result["cases"][case]["reference_c_marginal"]["mean"], self.chains[case][..., -1].mean())

    def test_declared_artifact_source_model_state_and_population_hash_mismatches_rejected(self):
        for name, path in (("source", self.source), ("payload", self.payload),
                           ("model", self.checkpoint / "model.keras"),
                           ("state", self.checkpoint / "state.npz"), ("population", self.quantiles)):
            with self.subTest(name=name):
                content = path.read_bytes()
                path.write_bytes(content + b"tampered")
                try:
                    with self.assertRaisesRegex(ValueError, "differ"):
                        self.run_comparison(name)
                finally:
                    path.write_bytes(content)

    def test_case_observation_shape_and_joint_reference_target_mismatches_rejected(self):
        original = copy.deepcopy(self.record)
        self.record["cases"][self.exported.CASES[0]]["observation_sha256"] = "different-case"
        self.write(self.neural_path, self.record)
        with self.assertRaisesRegex(ValueError, "other observations"):
            self.run_comparison("wrong-case")
        self.record = original
        changed = {**self.neural, self.exported.CASES[0]: self.neural[self.exported.CASES[0]][:4095]}
        np.savez(self.payload, **changed)
        self.record["conditional_draws"]["sha256"] = self.fixed.digest(self.payload)
        self.write(self.neural_path, self.record)
        with self.assertRaisesRegex(ValueError, "dimensions/support"):
            self.run_comparison("wrong-shape")
        np.savez(self.payload, **self.neural)
        self.record["conditional_draws"]["sha256"] = self.fixed.digest(self.payload)
        self.write(self.neural_path, self.record)
        directory = self.reference_directories[self.exported.CASES[0]]
        reference = self.read(directory / "run.json")
        reference["task"] = self.task["physical_clean_price_task"]["target"]
        self.write(directory / "run.json", reference)
        with self.assertRaisesRegex(ValueError, "reference target"):
            self.run_comparison("fixed-target")

    def test_missing_and_unready_references_remain_inconclusive(self):
        missing, unready = self.exported.CASES[:2]
        (self.reference_directories[missing] / "run.json").unlink()
        directory = self.reference_directories[unready]
        diagnostic = self.read(directory / "diagnostics.json")
        diagnostic["reference_ready"] = False
        self.write(directory / "diagnostics.json", diagnostic)
        reference = self.read(directory / "run.json")
        reference["attempts"][0]["diagnostics_sha256"] = self.fixed.digest(directory / "diagnostics.json")
        self.write(directory / "run.json", reference)
        result, metric = self.run_comparison()
        self.assertEqual(metric.call_count, 1)
        for case in (missing, unready):
            self.assertEqual(result["cases"][case]["status"], "inconclusive")
            self.assertNotIn("comparison", result["cases"][case])
        self.assertEqual(result["conditional_status"], "inconclusive")
        self.assertEqual(result["synthetic_conditional_status"], "inconclusive")

    def test_reference_campaign_must_settle_and_output_identity_cannot_be_reused(self):
        self.write(self.progress_path, {"status": "running"})
        with self.assertRaisesRegex(ValueError, "must settle"):
            self.run_comparison()
        self.write(self.progress_path, {"status": "completed"})
        self.run_comparison()
        with self.assertRaisesRegex(FileExistsError, "must be new"):
            self.run_comparison()

    def test_undefined_descriptive_correlations_are_explicitly_unavailable(self):
        correlations = comparison.c_correlations(np.ones((12, 6)), self.target["parameter_names"])
        self.assertEqual(correlations, dict.fromkeys(self.target["parameter_names"][:5]))


if __name__ == "__main__":
    unittest.main()
