import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance import option_bank, option_bank_shards


ROOT = Path(__file__).resolve().parents[1]


def config():
    result = json.loads((ROOT / "configs/tasks/rough_bergomi_option_bank.json").read_text())
    result["splits"] = dict(train=4, validation=2, test=1)
    result["task"]["numerics"].update(n_paths=512, n_steps=8, max_mc_se_over_quote_sigma=10.0)
    return result


def chunks(path, configuration):
    paths = []
    for split, count in configuration["splits"].items():
        for start in range(0, count, 2):
            destination = path / f"{split}-{start}"
            option_bank_shards.write_shard(destination, configuration, split, start, min(count, start + 2))
            paths.append(destination)
    return paths


class OptionBankShardTests(unittest.TestCase):
    def test_array_wrapper_and_assembly_cli(self):
        c = config()
        c["splits"] = dict(train=1, validation=1, test=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(c))
            records = [{"config": "config.json", "split": split, "start": 0, "stop": 1, "group": "pilot"}
                       for split in c["splits"]]
            (root / "jobs.json").write_text(json.dumps(records))
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), SLURM_SUBMIT_DIR=str(ROOT))
            for index in range(len(records)):
                subprocess.run(["bash", str(ROOT / "cluster/option_bank_array.sbatch"),
                                sys.executable, str(root / "jobs.json"), str(root / "chunks")],
                               env=dict(env, SLURM_ARRAY_TASK_ID=str(index)), check=True,
                               capture_output=True, text=True)
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/assemble_option_bank.py"),
                                        "--config", str(root / "config.json"),
                                        "--shards", str(root / "chunks/pilot"), "--out", str(root / "bank")],
                                       env=env, check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(completed.stdout)["integrity_verified"])
            _, arrays = option_bank.load_bank(root / "bank")
            self.assertEqual(arrays["train"]["example_id"].tolist(), ["train:0"])

    def test_shuffled_chunks_equal_monolithic_bank_and_preserve_roles(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = chunks(root / "chunks", c)
            assembled = root / "assembled"
            option_bank_shards.assemble_bank(assembled, c, reversed(paths))
            option_bank.write_bank(root / "monolithic", c)
            actual_manifest, actual = option_bank.load_bank(assembled)
            expected_manifest, expected = option_bank.load_bank(root / "monolithic")
            for split in expected:
                for key in expected[split]:
                    np.testing.assert_array_equal(actual[split][key], expected[split][key])
            self.assertEqual({k: v for k, v in actual_manifest.items() if k not in {"splits", "shard_assembly"}},
                             {k: v for k, v in expected_manifest.items() if k != "splits"})
            self.assertEqual(len(actual_manifest["shard_assembly"]["chunks"]), len(paths))
            with self.assertRaises(FileExistsError):
                option_bank_shards.assemble_bank(assembled, c, paths)
            with self.assertRaises(FileExistsError):
                option_bank_shards.write_shard(paths[0], c, "train", 0, 2)

    def test_missing_duplicate_and_overlapping_ranges_are_rejected(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = chunks(root / "chunks", c)
            overlap = root / "overlapping-chunk"
            option_bank_shards.write_shard(overlap, c, "train", 1, 3)
            for name, selected in (("missing", paths[1:]), ("duplicate", paths + [paths[0]]),
                                   ("overlap", paths + [overlap])):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "shard IDs"):
                    option_bank_shards.assemble_bank(root / name, c, selected)
                self.assertFalse((root / name).exists())

    def test_checksum_source_configuration_and_versions_are_checked(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = chunks(root / "chunks", c)
            manifest_path = paths[0] / "manifest.json"
            original = json.loads(manifest_path.read_text())
            mutations = [("configuration_sha256", "bad"), ("task_fingerprint", "bad"),
                         ("source_sha256", {}), ("shard_writer_sha256", "bad"),
                         ("versions", {"numpy": "changed", "scipy": "changed", "python": "changed"}),
                         ("sha256", "bad")]
            for key, value in mutations:
                modified = copy.deepcopy(original)
                modified[key] = value
                option_bank._write_json(manifest_path, modified)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    option_bank_shards.assemble_bank(root / "out", c, paths)
                self.assertFalse((root / "out").exists())
            option_bank._write_json(manifest_path, original)
            changed = copy.deepcopy(c)
            changed["root_seed"] += 1
            with self.assertRaisesRegex(ValueError, "configuration"):
                option_bank_shards.assemble_bank(root / "out", changed, paths)

    def test_rehashed_invalid_arrays_still_fail_before_success_manifest(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = chunks(root / "chunks", c)
            destination = paths[0] / "arrays.npz"
            with np.load(destination, allow_pickle=False) as saved:
                arrays = dict(saved)
            arrays["pricing_covariance"][0, 0, 0] = -1
            np.savez(destination, **arrays)
            manifest_path = paths[0] / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sha256"] = option_bank._sha256(destination)
            option_bank._write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "covariance"):
                option_bank_shards.assemble_bank(root / "out", c, paths)
            self.assertFalse((root / "out").exists())

    def test_failed_chunk_records_exact_id_without_redrawing(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed"
            with patch.object(option_bank, "price_panel", side_effect=ValueError("numerical failure")) as pricing:
                with self.assertRaisesRegex(ValueError, "train:2.*numerical failure"):
                    option_bank_shards.write_shard(path, c, "train", 2, 4)
            self.assertEqual(pricing.call_count, 1)
            self.assertFalse((path / "manifest.json").exists())
            self.assertIn("train:2", json.loads((path / "failure.json").read_text())["error"])

    def test_invalid_ranges_are_rejected_before_creating_directories(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            for split, start, stop in (("bad", 0, 1), ("train", 1, 1), ("train", 0, 5),
                                       ("train", True, 2), ("train", -1, 2)):
                with self.subTest(split=split, start=start, stop=stop), self.assertRaises(ValueError):
                    option_bank_shards.write_shard(out, c, split, start, stop)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
