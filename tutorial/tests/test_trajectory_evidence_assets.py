"""Dense example frames must match every bound model, history and original snapshot."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import evidence, evidence_assets, simulation


BASE_LENGTHS = [1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260]


class DenseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.assets = Path(self.temporary.name)
        self.folder = self.assets / "trajectories" / evidence_assets.ASSET_FOLDER
        self.folder.mkdir(parents=True)
        self.task = simulation.teaching_task("rough-known")
        self.base_hash = "bound-original-replay"
        self.checkpoint = "unchanged-native-checkpoint"
        bounds = np.asarray(self.task["prior_bounds"])
        theta = bounds[:, 0] + np.arange(1, 5)[:, None]*.15*(bounds[:, 1]-bounds[:, 0])
        self.arrays = {"observations": np.arange(4*1260).reshape(4, 1260, 1)*1e-6,
                       "trajectory_id": np.array([f"{simulation.task_fingerprint(self.task)[:16]}:73:validation:{i}"
                                                   for i in range(4)])}
        for day in BASE_LENGTHS:
            self.arrays[f"draws_{day}"] = np.repeat(theta[:, None, :], 8, axis=1)
        lengths = np.array(sorted(set(evidence_assets.schedule())-set(BASE_LENGTHS)), dtype=np.int64)
        fractions = .2+np.arange(4)[:, None, None, None]*.1+np.arange(len(lengths))[None, :, None, None]*.001
        draws = bounds[:, 0]+fractions*(bounds[:, 1]-bounds[:, 0])
        draws = np.broadcast_to(draws, (4, len(lengths), 128, 4)).copy()
        self.payload = {"lengths": lengths, "draws": draws, "trajectory_id": self.arrays["trajectory_id"]}
        histories = []
        for case in range(4):
            manager = self.manager(case)
            histories.append({"case": case, "trajectory_id": manager["metadata"]["history_id"],
                              "history_sha256": manager["metadata"]["history_sha256"],
                              "history_length": 1260,
                              "original_frame_sha256": {str(day): evidence_assets._draws_sha(value["draws"])
                                                          for day, value in manager["frames"].items()}})
        self.bundle = {"format": "trajectory_dense_evidence_v1", "model": "rough-known", "measure": "P",
                       "task": self.task, "task_sha256": simulation.task_fingerprint(self.task),
                       "checkpoint_sha256": self.checkpoint, "source_replay_bundle_sha256": self.base_hash,
                       "scientific_acceptance": False, "interpolated": False, "parameter_prior_updated": False,
                       "source_split": "validation", "source_case_offset": 0, "posterior_draws": 128,
                       "base_lengths": BASE_LENGTHS, "all_lengths": evidence_assets.schedule(),
                       "histories": histories}
        self.write_payload()
        self.write_bundle()
        self.assets_patch = mock.patch.object(evidence_assets, "ASSETS", self.assets)
        self.catalogue_patch = mock.patch.object(evidence_assets.lesson, "catalogue", return_value={
            "artifacts": {"rough-known": {"bundle_sha256": self.base_hash}}})
        self.assets_patch.start()
        self.catalogue_patch.start()

    def tearDown(self):
        self.catalogue_patch.stop()
        self.assets_patch.stop()
        self.temporary.cleanup()

    def manager(self, case=0):
        return evidence.saved_manager(self.task, self.arrays, case, self.checkpoint)

    def write_payload(self):
        np.savez_compressed(self.folder / "frames.npz", **self.payload)
        self.bundle["files"] = {"frames.npz": evidence_assets._sha(self.folder / "frames.npz")}

    def write_bundle(self):
        (self.folder / "bundle.json").write_text(json.dumps(self.bundle))

    def load(self, manager=None):
        # Updating the test pin permits independent validation of inner contracts.
        # The real loader has a source-controlled immutable bundle SHA instead.
        pin = evidence_assets._sha(self.folder / "bundle.json")
        with mock.patch.object(evidence_assets, "BUNDLE_SHA256", pin):
            return evidence_assets.load_dense(self.manager() if manager is None else manager)

    def test_all_four_cases_gain_actual_early_daily_frames_without_changing_original_samples(self):
        for case in range(4):
            with self.subTest(case=case):
                manager = self.manager(case)
                before = {day: value["draws"].copy() for day, value in manager["frames"].items()}
                self.assertIs(self.load(manager), manager)
                self.assertEqual(sorted(manager["frames"]), evidence_assets.schedule())
                self.assertEqual(len(manager["frames"]), 65)
                self.assertEqual([f["length"] for f in evidence.frame_list(manager, 21)], list(range(1, 22)))
                for day, draws in before.items():
                    np.testing.assert_array_equal(manager["frames"][day]["draws"], draws)
                index = self.payload["lengths"].tolist().index(3)
                np.testing.assert_array_equal(manager["frames"][3]["draws"], self.payload["draws"][case, index])
                self.assertEqual(manager["frames"][3]["draws"].shape, (128, 4))
                self.assertIs(self.load(manager), manager)  # Idempotent, no posterior replacement.

    def test_bundle_pin_and_payload_bytes_are_independent_integrity_checks(self):
        with mock.patch.object(evidence_assets, "BUNDLE_SHA256", "0"*64):
            with self.assertRaisesRegex(ValueError, "pinned"):
                evidence_assets.load_dense(self.manager())
        with (self.folder / "frames.npz").open("ab") as stream:
            stream.write(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "file hash"):
            self.load()

    def test_wrong_task_checkpoint_base_history_or_original_frame_cannot_be_relabelled(self):
        original = copy.deepcopy(self.bundle)
        for key, value in (("checkpoint_sha256", "other-model"), ("task_sha256", "other-task"),
                           ("source_replay_bundle_sha256", "other-replay"), ("scientific_acceptance", True),
                           ("interpolated", True), ("parameter_prior_updated", True)):
            with self.subTest(key=key):
                self.bundle = copy.deepcopy(original)
                self.bundle[key] = value
                self.write_bundle()
                with self.assertRaises(ValueError):
                    self.load()
        self.bundle = copy.deepcopy(original)
        self.bundle["histories"][0]["history_sha256"] = "other-returns"
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "history bytes"):
            self.load()
        self.bundle = copy.deepcopy(original)
        self.write_bundle()
        manager = self.manager()
        manager["frames"][4]["draws"] = manager["frames"][4]["draws"].copy()+.00001
        with self.assertRaisesRegex(ValueError, "original saved posterior"):
            self.load(manager)

    def test_invalid_rows_in_another_case_or_fractional_schedule_reject_entire_bundle(self):
        original = {key: value.copy() for key, value in self.payload.items()}
        for bad in (np.nan, 1.):
            with self.subTest(bad=bad):
                self.payload = {key: value.copy() for key, value in original.items()}
                self.payload["draws"][3, 0, 0, 0] = bad  # Selected history zero must not hide a bad case three.
                self.write_payload()
                self.write_bundle()
                manager = self.manager()
                with self.assertRaisesRegex(ValueError, "finite|support"):
                    self.load(manager)
                self.assertEqual(sorted(manager["frames"]), BASE_LENGTHS)
        self.payload = {key: value.copy() for key, value in original.items()}
        self.payload["lengths"] = self.payload["lengths"].astype(float)
        self.write_payload()
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "schedule"):
            self.load()

    def test_other_models_and_native_contexts_need_no_dense_asset(self):
        native = evidence.native_manager(self.task, self.arrays["observations"][0], self.checkpoint, "history-A")
        other = simulation.teaching_task("heston")
        arrays = {"observations": np.zeros((1, 1260, 1)), "trajectory_id": np.array(["heston-case"]),
                  "draws_1": np.broadcast_to(np.mean(other["prior_bounds"], axis=1), (1, 8, 5))}
        saved = evidence.saved_manager(other, arrays, 0, "heston-checkpoint")
        with mock.patch.object(evidence_assets, "_sha", side_effect=AssertionError("must not read dense assets")):
            self.assertIs(evidence_assets.load_dense(native), native)
            self.assertIs(evidence_assets.load_dense(saved), saved)

    def test_symlinked_child_files_cannot_escape_the_asset_root(self):
        with tempfile.TemporaryDirectory() as outside:
            for name in ("bundle.json", "frames.npz"):
                with self.subTest(name=name):
                    local = self.folder / name
                    external = Path(outside) / name
                    local.replace(external)
                    local.symlink_to(external)
                    try:
                        pin = hashlib.sha256((self.folder / "bundle.json").read_bytes()).hexdigest()
                        with mock.patch.object(evidence_assets, "BUNDLE_SHA256", pin):
                            with self.assertRaisesRegex(ValueError, "remain under"):
                                evidence_assets.load_dense(self.manager())
                    finally:
                        local.unlink()
                        external.replace(local)


if __name__ == "__main__":
    unittest.main()
