"""Compact report provenance and finite-precision counter verification."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import summarize_heston_spread_scale_large_batch as summary
from scripts import render_heston_spread_scale_large_batch_report as report


class LargeBatchReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixed = summary.fixed

    def counter_fixture(self, value=57631768.):
        checkpoint = self.root/"checkpoint"
        checkpoint.mkdir()
        values = [np.asarray(value, dtype=np.float32), np.asarray(value, dtype=np.float32)]
        arrays = {"non_trainable_0000": np.ones(30, dtype=np.float32),
            "non_trainable_0001": values[0], "non_trainable_0002": values[1]}
        np.savez(checkpoint/"state.npz", **arrays)
        rows = [{"path": "learned-looking-array", **self.fixed.array_summary(arrays["non_trainable_0000"])}]
        rows.extend({"path": f"basic_estimator/standardization_1/standardize_{i}/counter", **self.fixed.array_summary(v)}
            for i, v in enumerate(values))
        metadata = {"state_sha256": self.fixed.digest(checkpoint/"state.npz"), "state": {"non_trainable": rows}}
        self.fixed.write_json(checkpoint/"checkpoint.json", metadata)
        return checkpoint, metadata

    def test_summary_reads_only_two_hash_verified_counter_scalars(self):
        checkpoint, _ = self.counter_fixture()
        original_load, keys = np.load, []
        class GuardedArchive:
            def __enter__(self):
                self.archive = original_load(checkpoint/"state.npz", allow_pickle=False)
                return self
            def __exit__(self, *args): self.archive.close()
            def __getitem__(self, key):
                keys.append(key)
                if key not in ("non_trainable_0001", "non_trainable_0002"):
                    raise AssertionError("learned arrays must not be decompressed")
                return self.archive[key]
        with patch.object(np, "load", return_value=GuardedArchive()):
            counters = summary.normalizer_counts(checkpoint)
        self.assertEqual(keys, ["non_trainable_0001", "non_trainable_0002"])
        self.assertEqual(counters["value"], 57631768.)
        self.assertEqual([c["dtype"] for c in counters["counters"]], ["float32", "float32"])
        with (checkpoint/"state.npz").open("ab") as stream: stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "payload differs"):
            summary.normalizer_counts(checkpoint)

    def test_report_checks_encoded_counter_hash_without_opening_state(self):
        checkpoint, _ = self.counter_fixture()
        counter = summary.normalizer_counts(checkpoint)
        parent = 7643328.
        logical = parent+50000000
        counter.update(logical_parent_plus_examples=logical,
            difference_from_logical=counter["value"]-logical,
            relative_difference=(counter["value"]-logical)/logical)
        record = {"bank_passes": 500, "record": summary.binding(checkpoint/"checkpoint.json"),
            "standardizer_count": counter}
        inputs = report.shared.Artifacts(self.root/"manifest.json")
        with patch.object(np, "load", side_effect=AssertionError("presentation must not open state")):
            result = report.verify_compact_counter(inputs, record, parent)
        self.assertEqual(result["difference_from_logical"], -11560.)
        self.assertAlmostEqual(100*result["relative_difference"], -.020054359109869576, places=12)
        counter["counters"][0]["value"] += 4
        with self.assertRaisesRegex(ValueError, "recorded array hash"):
            report.verify_compact_counter(inputs, record, parent)

    def test_summary_rejects_a_different_compatible_historical_manifest_before_export(self):
        study_path, old_manifest, mcmc_manifest = self.root/"study.json", self.root/"old.json", self.root/"mcmc.json"
        old_protocol = self.root/"old-protocol.json"
        self.fixed.write_json(old_protocol, {})
        self.fixed.write_json(old_manifest, {"format": "heston_spread_scale_capacity_report_v1", "protocol": summary.binding(old_protocol)})
        self.fixed.write_json(mcmc_manifest, {"format": "heston_spread_mcmc_recovery_report_v1"})
        self.fixed.write_json(study_path, {"format": "heston_spread_scale_large_batch_study_v1",
            "arms": [{"batch_size": 1024}, {"batch_size": 4098}],
            "previous_training_study": {"path": "wrong-compatible-study", "sha256": "wrong"},
            "matched_mcmc_report": summary.binding(mcmc_manifest)})
        with self.assertRaisesRegex(ValueError, "predeclared comparison"):
            summary.build(study_path, old_manifest, mcmc_manifest, self.root/"out.json")
        self.assertFalse((self.root/"training").exists())
        self.assertFalse((self.root/"out.json").exists())


if __name__ == "__main__":
    unittest.main()
