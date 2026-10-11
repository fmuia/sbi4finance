"""Transfer integrity and numerical reproduction are separate guarantees."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance import option_bank

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generate_option_bank", ROOT / "scripts/generate_option_bank.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class OptionBankVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_dir.name) / "fixture"
        config = json.loads((ROOT / "configs/tasks/hkde_option_bank.json").read_text())
        config["splits"] = {"train": 1, "validation": 1, "test": 1}
        config["task"]["context"]["contracts"] = [
            {"strike": 100.0, "maturity": 1.0, "option_type": "put"}]
        option_bank.write_bank(cls.fixture, config)
        option_bank._write_json(cls.fixture / "verification.json", verifier.verify_bank(cls.fixture))

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bank = Path(self.tmp.name) / "bank"
        shutil.copytree(self.fixture, self.bank)

    def update_manifest(self, change):
        path = self.bank / "manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        option_bank._write_json(path, manifest)

    def change_train(self, change):
        path = self.bank / "train.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data)
        change(arrays)
        np.savez(path, **arrays)
        # Refresh transfer checksum deliberately: numeric regeneration must
        # detect changed values even when the file/manifest pair is consistent.
        self.update_manifest(lambda m: m["splits"]["train"].update(sha256=option_bank._sha256(path)))

    def test_exact_and_portable_report_distinct_guarantees(self):
        exact = verifier.verify_bank(self.bank)
        portable = verifier.verify_bank(self.bank, portable=True)
        self.assertTrue(exact["passed"])
        self.assertEqual(exact["verification_mode"], "exact")
        self.assertEqual(portable["tolerances"]["rtol"], 1e-12)
        self.assertEqual(portable["tolerances"]["base_atol"], 1e-14)
        self.assertEqual(portable["scientific_source_identity"]["differences"], {})
        self.assertTrue(portable["transfer_integrity"]["matched_manifest_split_hashes"])
        self.assertEqual(portable["array_comparisons"]["train"]["observations"]["max_absolute_difference"], 0)
        self.assertEqual(portable["array_comparisons"]["train"]["example_id"]["comparison"], "exact IDs")
        self.assertFalse(portable["numerical_bias_certified"])

    def test_versions_must_match_exact_mode_but_portable_discloses_changes(self):
        self.update_manifest(lambda m: m["versions"].update(numpy="different-numpy"))
        with self.assertRaisesRegex(ValueError, "recorded environment"):
            verifier.verify_bank(self.bank)
        report = verifier.verify_bank(self.bank, portable=True)
        self.assertEqual(report["environment_comparison"]["version_differences"]["numpy"]["saved"],
                         "different-numpy")

    def test_platform_change_or_absence_is_disclosed(self):
        current = verifier._environment()
        changed = copy.deepcopy(current)
        changed["platform"]["machine"] = "another-architecture"
        with patch.object(verifier, "_environment", return_value=changed):
            with self.assertRaisesRegex(ValueError, "recorded environment"):
                verifier.verify_bank(self.bank)
            report = verifier.verify_bank(self.bank, portable=True)
        self.assertEqual(report["environment_comparison"]["platform_differences"]["machine"]["current"],
                         "another-architecture")
        (self.bank / "verification.json").unlink()
        old_report = verifier.verify_bank(self.bank, portable=True)
        self.assertFalse(old_report["environment_comparison"]["platform_available"])
        self.assertIsNone(old_report["environment_comparison"]["platform_differences"])

    def test_portable_never_accepts_changed_or_missing_scientific_source(self):
        self.update_manifest(lambda m: m["source_sha256"].update({"hkde.py": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "generator source changed.*hkde.py"):
            verifier.verify_bank(self.bank, portable=True)
        self.update_manifest(lambda m: m["source_sha256"].pop("hkde.py"))
        with self.assertRaisesRegex(ValueError, "source list"):
            verifier.verify_bank(self.bank, portable=True)

    def test_transfer_corruption_fails_before_regeneration(self):
        with (self.bank / "train.npz").open("ab") as file:
            file.write(b"corrupt transfer")
        with patch.object(option_bank, "generate_examples") as generate:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verifier.verify_bank(self.bank, portable=True)
            generate.assert_not_called()

    def test_changed_observations_fail_despite_matching_transfer_hash(self):
        self.change_train(lambda a: a["observations"].__iadd__(0.01))
        for portable in (False, True):
            with self.assertRaisesRegex(ValueError, "train/observations: regeneration mismatch"):
                verifier.verify_bank(self.bank, portable=portable)

    def test_small_float_change_requires_declared_portable_tolerance(self):
        self.change_train(lambda a: a["observations"].__iadd__(5e-14))
        with self.assertRaisesRegex(ValueError, "regeneration mismatch"):
            verifier.verify_bank(self.bank)
        report = verifier.verify_bank(self.bank, portable=True)
        self.assertGreater(report["array_comparisons"]["train"]["observations"]["max_absolute_difference"], 0)
        # Currency noise-scaled absolute tolerance remains when base atol is zero.
        self.assertTrue(verifier.verify_bank(self.bank, portable=True, rtol=0, atol=0)["passed"])
        with self.assertRaisesRegex(ValueError, "tolerances"):
            verifier.verify_bank(self.bank, portable=True, rtol=float("nan"))

    def test_currency_roundoff_uses_explicit_quote_noise_fraction(self):
        self.change_train(lambda a: [a[key].__iadd__(2e-9) for key in verifier.CURRENCY_ARRAYS])
        with self.assertRaisesRegex(ValueError, "regeneration mismatch"):
            verifier.verify_bank(self.bank)
        report = verifier.verify_bank(self.bank, portable=True)
        self.assertEqual(report["tolerances"]["price_noise_fraction"], 1e-6)
        self.assertEqual(report["tolerances"]["currency_atol"], 2.5e-7)
        for key in verifier.CURRENCY_ARRAYS:
            comparison = report["array_comparisons"]["train"][key]
            self.assertEqual(comparison["units"], "currency")
            self.assertEqual(comparison["tolerances"]["atol"], 2.5e-7)
            self.assertGreater(comparison["max_absolute_difference_over_quote_sigma"], 0)
        self.assertEqual(report["array_comparisons"]["train"]["parameters"]["tolerances"]["atol"], 1e-14)
        self.assertEqual(report["array_comparisons"]["train"]["pricing_covariance"]["units"], "currency squared")

    def test_currency_changes_above_noise_scaled_budget_fail(self):
        self.change_train(lambda a: a["observations"].__iadd__(5e-7))
        with self.assertRaisesRegex(ValueError, "train/observations: regeneration mismatch"):
            verifier.verify_bank(self.bank, portable=True)

    def test_parameters_keep_the_tight_base_tolerance(self):
        self.change_train(lambda a: a["parameters"].__iadd__(2e-9))
        with self.assertRaisesRegex(ValueError, "train/parameters: regeneration mismatch"):
            verifier.verify_bank(self.bank, portable=True)

    def test_covariance_keeps_the_tight_currency_squared_tolerance(self):
        def change(arrays):
            arrays["pricing_covariance"][:] = 2e-14
            arrays["pricing_standard_errors"][:] = np.sqrt(2e-14)
        self.change_train(change)
        # Coherent covariance/SE passes storage checks; currency-valued SE
        # passes its portable budget but covariance must still fail at 1e-14.
        with self.assertRaisesRegex(ValueError, "train/pricing_covariance: regeneration mismatch"):
            verifier.verify_bank(self.bank, portable=True)

    def test_invalid_quote_sigma_cannot_define_a_portable_budget(self):
        manifest, splits = option_bank.load_bank(self.bank)
        for sigma in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(sigma=sigma):
                manifest["configuration"]["task"]["context"]["quote_noise_std"] = sigma
                with patch.object(option_bank, "load_bank", return_value=(manifest, splits)):
                    with self.assertRaisesRegex(ValueError, "positive scalar quote_noise_std"):
                        verifier.verify_bank(self.bank, portable=True)

    def test_ids_are_exact_even_in_portable_mode(self):
        self.change_train(lambda a: a["example_id"].__setitem__(0, "train:9"))
        with self.assertRaisesRegex(ValueError, "base-panel identity"):
            verifier.verify_bank(self.bank, portable=True)

    def test_verify_existing_cli_is_read_only_and_can_write_external_report(self):
        before = {p.name: p.read_bytes() for p in self.bank.iterdir()}
        report_path = Path(self.tmp.name) / "portable.json"
        argv = ["generate_option_bank.py", "--verify-existing", str(self.bank),
                "--portable", "--report", str(report_path)]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
            verifier.main()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.bank.iterdir()})
        self.assertEqual(json.loads(output.getvalue()), json.loads(report_path.read_text()))
        with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                verifier.main()  # Existing reports are not overwritten either.
        argv[-1] = str(self.bank / "new-report.json")
        with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                verifier.main()
        self.assertFalse((self.bank / "new-report.json").exists())


if __name__ == "__main__":
    unittest.main()
