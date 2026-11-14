"""CRLF checkouts preserve frozen trajectory inputs and still reject corruption."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import (
    evidence, evidence_assets, latent_assets, lesson, simulation, variance_aging,
)


class TrajectoryAssetPortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = lesson.ASSETS
        cls.replays = {name: lesson.load_example(name) for name in simulation.MODELS}
        cls.latents = {(name, case): latent_assets.load_history(task, arrays, case)
                       for name, (task, _, arrays) in cls.replays.items() for case in range(4)}
        cls.markets = {name: lesson.load_market_example(name)
                       for name in ("voo-covid-256", "voo-2020-20261113")}
        cls.aging = variance_aging.load(cls.replays["rough-known"][0])["report"]
        cls.dense = {}
        task, report, arrays = cls.replays["rough-known"]
        for case in range(4):
            manager = evidence.saved_manager(task, arrays, case, report["checkpoint_sha256"])
            cls.dense[case] = evidence_assets.load_dense(manager)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.assets = Path(self.temporary.name) / "assets"
        for name in ("trajectories", "market-examples", "diagnostics/" + variance_aging.ASSET_FOLDER,
                     "results/trajectories/closeout-20261031"):
            shutil.copytree(self.source / name, self.assets / name)
        self.original = {path.relative_to(self.assets): path.read_bytes()
                         for path in self.assets.rglob("*") if path.is_file()}
        for relative, raw in self.original.items():
            if relative.suffix == ".json":
                (self.assets / relative).write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for module in (lesson, latent_assets, evidence_assets, variance_aging):
            self.patches.enter_context(mock.patch.object(module, "ASSETS", self.assets))
        self.patches.enter_context(mock.patch.object(lesson, "CATALOGUE", self.assets / "trajectories/artifacts.json"))

    def assert_arrays_identical(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def new_dense_manager(self):
        task, report, arrays = lesson.load_example("rough-known")
        return evidence.saved_manager(task, arrays, 0, report["checkpoint_sha256"])

    def test_crlf_copies_load_all_saved_routes_without_rebinding_identity_or_data(self):
        for name, (expected_task, expected_report, expected_arrays) in self.replays.items():
            with self.subTest(model=name):
                task, report, arrays = lesson.load_example(name)
                self.assertEqual(task, expected_task)
                self.assertEqual(report, expected_report)
                self.assertEqual(set(arrays), set(expected_arrays))
                for key in arrays:
                    self.assert_arrays_identical(arrays[key], expected_arrays[key])
                for case in range(4):
                    actual = latent_assets.load_history(task, arrays, case)
                    expected = self.latents[name, case]
                    self.assertEqual(actual["metadata"], expected["metadata"])
                    for key, value in expected.items():
                        if isinstance(value, np.ndarray):
                            self.assert_arrays_identical(actual[key], value)
        task, report, arrays = lesson.load_example("rough-known")
        for case, expected in self.dense.items():
            manager = evidence.saved_manager(task, arrays, case, report["checkpoint_sha256"])
            actual = evidence_assets.load_dense(manager)
            self.assertEqual(actual["context_id"], expected["context_id"])
            self.assertEqual(actual["metadata"], expected["metadata"])
            self.assertEqual(set(actual["frames"]), set(expected["frames"]))
            for day, frame in actual["frames"].items():
                self.assertEqual(frame["frame_id"], expected["frames"][day]["frame_id"])
                self.assert_arrays_identical(frame["draws"], expected["frames"][day]["draws"])
        self.assertEqual(variance_aging.load(task)["report"], self.aging)
        for name, expected in self.markets.items():
            actual = lesson.load_market_example(name)
            self.assertEqual(actual["metadata"], expected["metadata"])
            for key, values in expected.items():
                if isinstance(values, np.ndarray):
                    self.assert_arrays_identical(actual[key], values)
        closeout = self.assets / "results/trajectories/closeout-20261031"
        receipt = json.loads((closeout / "ACCEPTED.json").read_text())
        for name in ("closeout-summary.json", "figure-metadata.json", "drift-learning.png", "marginal-coverage.png"):
            self.assertEqual(lesson.asset_digest(closeout / name), receipt["source_sha256"][name])
        # Loading must not rewrite accepted inputs or alter numerical payloads.
        for relative, original in self.original.items():
            copied = (self.assets / relative).read_bytes()
            if relative.suffix == ".json":
                self.assertEqual(copied.replace(b"\r\n", b"\n"), original.replace(b"\r\n", b"\n"))
            else:
                self.assertEqual(copied, original)

    def test_changed_json_content_is_rejected_for_every_hashed_route(self):
        task, _, arrays = lesson.load_example("rough-known")
        checks = [
            ("trajectories/rough-known/bundle.json", lambda: lesson.load_example("rough-known")),
            ("trajectories/rough-known/task.json", lambda: lesson.load_example("rough-known")),
            ("trajectories/latent-history-v1/bundle.json", lambda: latent_assets.load_history(task, arrays, 0)),
            ("trajectories/evidence-rough-known-v1/bundle.json", lambda: evidence_assets.load_dense(self.new_dense_manager())),
            ("diagnostics/rough-variance-aging-v1/report.json", lambda: variance_aging.load(task)),
            ("market-examples/voo-covid-256/market.json", lambda: lesson.load_market_example()),
        ]
        for name, load in checks:
            with self.subTest(asset=name):
                path = self.assets / name
                before = path.read_bytes()
                changed = json.loads(before)
                changed["content_changed_for_portability_test"] = True
                path.write_bytes((json.dumps(changed, indent=2) + "\n").replace("\n", "\r\n").encode())
                try:
                    with self.assertRaisesRegex(ValueError, "hash|checksum|pinned"):
                        load()
                finally:
                    path.write_bytes(before)

    def test_binary_bytes_remain_exact_even_when_the_only_change_is_crlf(self):
        task, _, arrays = lesson.load_example("rough-known")
        checks = [
            ("trajectories/rough-known/posterior.npz", lambda: lesson.load_example("rough-known")),
            ("trajectories/latent-history-v1/rough-known.npz", lambda: latent_assets.load_history(task, arrays, 0)),
            ("trajectories/evidence-rough-known-v1/frames.npz", lambda: evidence_assets.load_dense(self.new_dense_manager())),
            ("diagnostics/rough-variance-aging-v1/variance-aging.png", lambda: variance_aging.load(task)),
            ("market-examples/voo-covid-256/returns.npz", lambda: lesson.load_market_example()),
        ]
        for name, load in checks:
            with self.subTest(asset=name):
                path = self.assets / name
                before = path.read_bytes()
                path.write_bytes(before + b"\r\n")
                try:
                    with self.assertRaisesRegex(ValueError, "hash|checksum"):
                        load()
                finally:
                    path.write_bytes(before)
        # A binary extension must never acquire JSON's newline exception.
        forged = self.assets / "forged.npz"
        lf = b"binary payload\nwith two lines\n"
        forged.write_bytes(lf.replace(b"\n", b"\r\n"))
        self.assertEqual(lesson.asset_digest(forged), hashlib.sha256(forged.read_bytes()).hexdigest())
        self.assertNotEqual(lesson.asset_digest(forged), hashlib.sha256(lf).hexdigest())




if __name__ == "__main__":
    unittest.main()
