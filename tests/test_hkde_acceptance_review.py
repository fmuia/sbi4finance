"""Integration may replace teaching entry points without rebinding science."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "hkde_acceptance_review",
    Path(__file__).resolve().parents[1] / "scripts/check_hkde_acceptance_review.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

NOTEBOOK = "notebooks/10_hkde_corrected_workflow.py"
SOURCE = "src/sbi4finance/hkde_correction.py"
CONFIG = '''[project]
name = "test"
[tool.pixi.dependencies]
python = "3.12.*"
[tool.pixi.tasks]
review = "marimo edit notebooks/10_hkde_corrected_workflow.py"
'''


class AcceptanceReviewProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = {
            NOTEBOOK: b"# Historical selected-case lesson\n",
            "pyproject.toml": CONFIG.encode(),
            SOURCE: b"# Frozen scientific implementation\n",
            "pixi.lock": b"frozen dependency resolution\n",
        }
        for name, content in self.original.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        self.git("init", "--quiet")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Frozen inputs")
        self.manifest = {
            "base_commit": self.git("rev-parse", "HEAD").decode().strip(),
            "files": {name: review.sha(content) for name, content in self.original.items()},
        }

    def git(self, *arguments):
        return subprocess.run(["git", "-C", str(self.root), *arguments],
                              check=True, capture_output=True).stdout

    def replace_teaching(self):
        (self.root / NOTEBOOK).write_text("# Integrated cached/default and opt-in live lesson\n")
        (self.root / "pyproject.toml").write_text(CONFIG + 'data-review = "python data_review.py"\n')

    def test_exact_inputs_pass_but_replacement_requires_explicit_flag(self):
        result = review.verify_manifest(self.root, self.manifest)
        self.assertTrue(result["all_frozen_files_match_current_tree"])
        self.assertEqual(result["manifest_current_files_verified"], 4)
        self.replace_teaching()
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            review.verify_manifest(self.root, self.manifest)

    def test_integrated_teaching_verifies_history_and_records_both_identities(self):
        self.replace_teaching()
        original_manifest = deepcopy(self.manifest)
        result = review.verify_manifest(self.root, self.manifest, allow_integrated_teaching=True)
        self.assertEqual(self.manifest, original_manifest)
        self.assertFalse(result["all_frozen_files_match_current_tree"])
        self.assertTrue(result["scientific_inputs_match_current_tree"])
        self.assertEqual(result["manifest_current_files_verified"], 2)
        self.assertEqual(result["manifest_historical_teaching_files_verified"], 2)
        self.assertEqual(set(result["historical_teaching_bindings"]), {NOTEBOOK, "pyproject.toml"})
        for name, binding in result["historical_teaching_bindings"].items():
            self.assertEqual(binding["historical_commit"], self.manifest["base_commit"])
            self.assertEqual(binding["historical_sha256"], review.sha(self.original[name]))
            self.assertEqual(binding["current_sha256"], review.sha((self.root / name).read_bytes()))

    def test_flag_cannot_rebind_scientific_sources_or_lock(self):
        self.replace_teaching()
        for name in (SOURCE, "pixi.lock"):
            with self.subTest(name=name):
                (self.root / name).write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "hashes differ"):
                    review.verify_manifest(self.root, self.manifest, allow_integrated_teaching=True)
                (self.root / name).write_bytes(self.original[name])

    def test_flag_cannot_change_dependency_or_environment_configuration(self):
        for config in (
            CONFIG.replace('python = "3.12.*"', 'python = "3.13.*"'),
            CONFIG + '\n[tool.pixi.activation.env]\nKERAS_BACKEND = "torch"\n',
        ):
            with self.subTest(config=config):
                (self.root / "pyproject.toml").write_text(config)
                with self.assertRaisesRegex(ValueError, "only the Pixi task table"):
                    review.verify_manifest(self.root, self.manifest, allow_integrated_teaching=True)

    def test_missing_history_and_wrong_historical_hash_fail_closed(self):
        self.replace_teaching()
        unknown_commit = deepcopy(self.manifest)
        unknown_commit["base_commit"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "Cannot verify historical teaching input"):
            review.verify_manifest(self.root, unknown_commit, allow_integrated_teaching=True)
        wrong_hash = deepcopy(self.manifest)
        wrong_hash["files"][NOTEBOOK] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Historical teaching input differs"):
            review.verify_manifest(self.root, wrong_hash, allow_integrated_teaching=True)


if __name__ == "__main__":
    unittest.main()
