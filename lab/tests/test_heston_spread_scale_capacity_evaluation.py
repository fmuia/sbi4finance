"""Continuation diagnostics preserve population, posterior and budget identities."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import scripts

from scripts import evaluate_heston_spread_scale_continuation as evaluation
from scripts import train_heston_spread_scale as scale
from test_heston_spread_training import fixture


class CapacityEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        design, _, _, _ = fixture(self.root)
        self.task, _ = scale.make_task(design)
        self.target = self.task["target"]
        self.bounds = np.asarray(self.target["prior_bounds"])
        self.rng = np.random.default_rng(812)

    def write(self, path, value):
        evaluation.fixed.write_json(path, value)

    def artifact(self, path):
        return {"path": str(path), "sha256": evaluation.fixed.digest(path)}

    def study_fixture(self):
        for filename in ("old-study.json", "recovery-addendum.json", "old-evaluation.json"):
            self.write(self.root/filename, {})
        return {"format": "heston_spread_scale_capacity_study_v1", "joint_task": self.target,
            "evaluation": evaluation.evaluation_contract(self.target),
            "original_study": self.artifact(self.root/"old-study.json"),
            "reference_recovery": self.artifact(self.root/"recovery-addendum.json"),
            "baseline_evaluation": self.artifact(self.root/"old-evaluation.json")}

    def test_recovery_distinguishes_prior_return_from_informative_medians(self):
        truth = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (512, 6))
        midpoint = self.bounds.mean(axis=1)
        prior = np.broadcast_to(self.bounds[:, 0]+np.array([.05, .5, .95])[:, None, None]
            *(self.bounds[:, 1]-self.bounds[:, 0]), (3, 512, 6)).copy()
        summary = evaluation.recovery_summary(truth, prior, self.bounds)
        np.testing.assert_allclose(summary["median_width90_over_prior"], 1)
        np.testing.assert_allclose(summary["median_rmse"], summary["prior_midpoint_rmse"])
        np.testing.assert_allclose(summary["median_rmse"], np.sqrt(((truth-midpoint)**2).mean(axis=0)))
        tight = np.stack([truth-.001, truth, truth+.001])
        recovered = evaluation.recovery_summary(truth, tight, self.bounds)
        np.testing.assert_array_equal(recovered["median_rmse"], np.zeros(6))
        np.testing.assert_allclose(recovered["truth_posterior_median_correlation"], 1)
        self.assertLess(recovered["median_width90_over_prior"][-1], .002)
        for malformed in (tight[:, :511], tight[::-1], np.full_like(tight, np.nan)):
            with self.assertRaisesRegex(ValueError, "finite ordered"):
                evaluation.recovery_summary(truth, malformed, self.bounds)

    def test_declaration_cannot_change_family_population_or_artifact_bindings(self):
        study = self.study_fixture()
        evaluation.validate_study(study, self.target)
        for key, value in (("family_checks", 144), ("checkpoint_bank_passes", [500]),
                ("batch_sizes", [128]), ("seed", 9), ("population_rows", 128),
                ("scientific_acceptance", True)):
            changed = copy.deepcopy(study)
            changed["evaluation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "declaration differs"):
                evaluation.validate_study(changed, self.target)
        (self.root/"old-evaluation.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binding differs"):
            evaluation.validate_study(study, self.target)

    def test_only_larger_bank_declared_arm_and_endpoint_supported(self):
        metadata = {"batch_size": 128, "completed_bank_passes": 25, "train_rows": 100000,
            "architecture": {"effective_dropout": 0.}}
        evaluation.validate_endpoint(metadata)
        for key, value in (("batch_size", 256), ("completed_bank_passes", 26),
                ("train_rows", 20000), ("architecture", {"effective_dropout": .05})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "endpoint"):
                evaluation.validate_endpoint({**metadata, key: value})

    def campaign_fixture(self):
        study = self.study_fixture()
        panel_path, ref_path = self.root/"panel.json", self.root/"reference-protocol.json"
        panel = {"joint_task": self.target, "products": {"control": [0., .1], "affected": [.3, .9]},
            "cases": [{"id": case, "observations": (np.arange(54)+i).tolist()}
                for i, case in enumerate(evaluation.previous.CASES)]}
        self.write(panel_path, panel)
        self.write(ref_path, {"panel_sha256": evaluation.fixed.digest(panel_path), "jobs": [{"id": "fixture"}]})
        self.write(self.root/"old-study.json", {"panel": self.artifact(panel_path)})
        progress = self.root/"progress.json"
        self.write(progress, {"status": "completed"})
        self.write(self.root/"recovery-addendum.json", {"progress_path": str(progress)})
        physical = self.rng.uniform(self.bounds[:5, 0], self.bounds[:5, 1], (512, 5))
        clean = self.rng.uniform(1., 10., (512, 54))
        y, truth = evaluation.previous.population_observations(self.task, physical, clean)
        old_recovery = self.root/"recovery.npz"
        quantiles = np.broadcast_to(self.bounds[:, 0]+np.array([.05, .5, .95])[:, None, None]
            *(self.bounds[:, 1]-self.bounds[:, 0]), (3, 512, 6)).copy()
        np.savez(old_recovery, truth=truth, quantiles=quantiles)
        self.write(self.root/"old-evaluation.json", {"status": "complete", "task": self.target,
            "actual_optimizer_updates": 60000, "seed": evaluation.previous.SEED,
            "population": {"recovery_sha256": evaluation.fixed.digest(old_recovery),
                "observations_sha256": evaluation.baseline.observations_digest(y),
                "truth_sha256": evaluation.baseline.observations_digest(truth)}})
        study.update({key: self.artifact(self.root/filename) for key, filename in
            (("original_study", "old-study.json"), ("reference_recovery", "recovery-addendum.json"),
             ("baseline_evaluation", "old-evaluation.json"))})
        study_path = self.root/"study.json"
        self.write(study_path, study)
        bank = self.root/"capacity-bank"
        bank.mkdir()
        payload = bank/"payload.npz"
        np.savez(payload, validation_parameters=physical, validation_clean=clean,
            reserved_test_parameters=np.full((2, 5), np.nan), reserved_test_clean=np.full((2, 54), np.nan))
        bank_record = {"identity": "bank-fixture", "payload": payload.name,
            "payload_sha256": evaluation.fixed.digest(payload)}
        self.write(bank/"bank.json", bank_record)
        checkpoint = self.root/"checkpoint"
        checkpoint.mkdir()
        for filename in ("model.keras", "state.npz"):
            (checkpoint/filename).write_bytes(b"opaque loader fixture")
        training_protocol = self.root/"training.json"
        self.write(training_protocol, {})
        metadata = {"task": self.task, "batch_size": 128, "completed_bank_passes": 25,
            "train_rows": 100000, "architecture": {"effective_dropout": 0.},
            "protocol_path": str(training_protocol), "protocol_sha256": evaluation.fixed.digest(training_protocol),
            "bank_path": str(bank), "bank_record_sha256": evaluation.fixed.digest(bank/"bank.json"),
            "bank_identity": bank_record["identity"], "bank_payload_sha256": bank_record["payload_sha256"],
            "model_sha256": evaluation.fixed.digest(checkpoint/"model.keras"),
            "state_sha256": evaluation.fixed.digest(checkpoint/"state.npz"), "actual_optimizer_updates": 79550}
        self.write(checkpoint/"checkpoint.json", metadata)
        return study_path, checkpoint, metadata, ref_path, panel_path, payload, y

    def test_population_reused_references_compared_and_reserved_arrays_never_read(self):
        study, checkpoint, metadata, ref_path, panel_path, bank_payload, population_y = self.campaign_fixture()
        calls, read_keys = [], []
        original_load = np.load

        class GuardedArchive:
            def __enter__(self):
                self.arrays = original_load(bank_payload, allow_pickle=False)
                return self
            def __exit__(self, *args):
                self.arrays.close()
            def __getitem__(self, key):
                read_keys.append(key)
                if key not in ("validation_parameters", "validation_clean"):
                    raise AssertionError("reserved/training data decompressed")
                return self.arrays[key]

        def guarded_load(path, *args, **kwargs):
            return GuardedArchive() if Path(path) == bank_payload else original_load(path, *args, **kwargs)

        def sample(workflow, task, quotes, count, seed):
            self.assertEqual(set(scale.condition_data(task, quotes)), {"observations"})
            calls.append((np.asarray(quotes).copy(), count, seed))
            return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (len(quotes), count, 6))

        dummy = types.ModuleType("scripts.continue_heston_spread_scale")
        dummy.load_checkpoint = lambda directory: (object(), metadata)
        dummy.__file__ = str(self.root/"dummy.py")
        Path(dummy.__file__).write_text("# immutable mock loader\n")
        source = {"fixture": self.artifact(Path(dummy.__file__))}
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"scripts.continue_heston_spread_scale": dummy}))
            stack.enter_context(patch.object(scripts, "continue_heston_spread_scale", dummy, create=True))
            stack.enter_context(patch.object(evaluation.previous, "validate_study"))
            stack.enter_context(patch.object(evaluation.previous, "selected_reference_protocol", return_value=(ref_path, {})))
            stack.enter_context(patch.object(evaluation.previous.scale_reference, "validate_protocol"))
            stack.enter_context(patch.object(evaluation, "source_bindings", return_value=source))
            stack.enter_context(patch.object(np, "load", side_effect=guarded_load))
            stack.enter_context(patch.object(evaluation.previous.reference, "load_case", side_effect=lambda path, case, **kw:
                (None, None, None, np.asarray(evaluation.read(panel_path)["cases"][evaluation.previous.CASES.index(case)]["observations"]))))
            stack.enter_context(patch.object(scale, "sample", side_effect=sample))
            reference = stack.enter_context(patch.object(evaluation.previous, "joint_reference", return_value=
                ((np.broadcast_to(self.bounds.mean(axis=1), (2, 64, 6)), {}), {"reference_ready": True})))
            compare = stack.enter_context(patch.object(evaluation.baseline, "compare_draws", return_value={"status": "pass"}))
            result = evaluation.evaluate(checkpoint, study, self.root/"output", self.root/"large-payload")
        self.assertEqual(read_keys, ["validation_parameters", "validation_clean"])
        self.assertEqual(reference.call_count, 3)
        self.assertEqual(compare.call_count, 3)
        self.assertTrue(all(call.kwargs["family_checks"] == 384 for call in compare.call_args_list))
        self.assertEqual([call[2] for call in calls[:3]], [261112, 261113, 261114])
        np.testing.assert_array_equal(np.concatenate([call[0] for call in calls[3:]]), population_y)
        self.assertEqual([call[2] for call in calls[3:]], [261112+1000+i for i in range(0, 512, 16)])
        self.assertEqual(result["conditional_status"], "pass")
        self.assertTrue(result["protected_inputs_unchanged"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertTrue(result["baseline_population"]["outside_new_conditional_family"])
        with self.assertRaisesRegex(FileExistsError, "must be new"):
            evaluation.evaluate(checkpoint, study, self.root/"output", self.root/"large-payload")


if __name__ == "__main__":
    unittest.main()
