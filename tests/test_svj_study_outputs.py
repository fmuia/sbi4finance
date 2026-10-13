"""Study CLIs preserve historical outputs and make replay/resume explicit."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sbi4finance.svj.artifacts import sha256

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("svj_agreement", "svj_backtest", "svj_backtest_check", "svj_classical", "svj_costs", "svj_rolling")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSVJStudyOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scripts = {name: load_script(name) for name in NAMES}

    def test_all_studies_require_explicit_outputs_and_reject_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, script in self.scripts.items():
                with self.subTest(name=name), patch("sys.argv", [name]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        script.main()
                    self.assertEqual(error.exception.code, 2)
                output = root / name
                if name == "svj_backtest_check":
                    output.write_text("historical report")
                    sentinel = output
                else:
                    output.mkdir()
                    sentinel = output / "sentinel.json"
                    sentinel.write_text("historical report")
                with self.subTest(existing=name), patch("sys.argv", [name, "--out", str(output)]), \
                     patch.object(script.model, "simulate", side_effect=AssertionError("no expensive study")):
                    with self.assertRaises(FileExistsError):
                        script.main()
                self.assertEqual(sentinel.read_text(), "historical report")

    def test_backtest_saved_replay_only_writes_new_directory(self):
        script = self.scripts["svj_backtest"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "replay"
            source.mkdir()
            shutil.copyfile(ROOT / "docs/results/svj-backtest/backtest_series.npz", source / "backtest_series.npz")
            before = sha256(source / "backtest_series.npz")
            class Figure:
                def savefig(self, path, **kwargs):
                    Path(path).write_text("render fixture")
            with patch("sys.argv", ["svj_backtest", "--from-saved", str(source), "--out", str(output)]), \
                 patch.object(script.plots, "method_panels", return_value=Figure()), \
                 patch.object(script.amortized.AmortizedPosterior, "load", side_effect=AssertionError("replay is read-only")), \
                 redirect_stdout(io.StringIO()):
                script.main()
            self.assertEqual(sha256(source / "backtest_series.npz"), before)
            self.assertEqual(set(path.name for path in source.iterdir()), {"backtest_series.npz"})
            report = json.loads((output / "backtest.json").read_text())
            self.assertEqual(report["risk_unit"], "percent_log_loss")
            self.assertEqual(report["provenance"]["source_sha256"], before)
            self.assertEqual(report["provenance"]["mode"], "saved_replay")

    def test_classical_resume_requires_matching_new_run_manifest(self):
        script = self.scripts["svj_classical"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shard0"
            command = ["svj_classical", "--n-series", "2", "--shards", "2", "--out", str(output)]
            with patch("sys.argv", command), patch.object(script, "fit_series") as fit:
                script.main()
                fit.assert_called_once()
            before = (output / "run.json").read_bytes()
            with patch("sys.argv", [*command, "--resume"]), patch.object(script, "fit_series") as fit:
                script.main()
                fit.assert_called_once()
            with patch("sys.argv", [*command, "--resume", "--seed", "999"]), patch.object(script, "fit_series") as fit:
                with self.assertRaisesRegex(ValueError, "matching run.json"):
                    script.main()
                fit.assert_not_called()
            self.assertEqual((output / "run.json").read_bytes(), before)
            legacy = Path(temporary) / "legacy"
            legacy.mkdir()
            with patch("sys.argv", ["svj_classical", "--resume", "--out", str(legacy)]):
                with self.assertRaisesRegex(ValueError, "historical"):
                    script.main()

    def test_classical_merge_passes_read_only_inputs_to_new_output(self):
        script = self.scripts["svj_classical"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / "shard0", root / "shard1"]
            before = {}
            for index, path in enumerate(inputs):
                path.mkdir()
                name = f"classical_series_{index}_of_2.json"
                shutil.copyfile(ROOT / "docs/results/svj-classical" / name, path / name)
                before[path / name] = sha256(path / name)
            output = root / "summary"
            def inexpensive_paired_references(args, records):
                for record in records:
                    record["check"] = record["plugin_at_mle"]
                    record["predictive"] = record["plugin_at_mle"]
            class Figure:
                def savefig(self, path, **kwargs):
                    Path(path).write_text("render fixture")
            with patch("sys.argv", ["svj_classical", "--merge", "--inputs", *map(str, inputs), "--out", str(output)]), \
                 patch.object(script, "paired_references", side_effect=inexpensive_paired_references), \
                 patch.object(script.plots, "method_panels", return_value=Figure()), \
                 patch.object(script, "fit_series") as fit, redirect_stdout(io.StringIO()):
                script.main()
                fit.assert_not_called()
            self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))
            report = json.loads((output / "classical.json").read_text())
            self.assertEqual(report["n_series"], 40)
            self.assertEqual(report["source_directories"], list(map(str, inputs)))
            self.assertEqual(report["risk_unit"], "percent_log_loss")
            with patch("sys.argv", ["svj_classical", "--merge", "--inputs", str(inputs[0]), str(inputs[0]),
                                     "--out", str(root / "duplicates")]):
                with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                    script.main()


if __name__ == "__main__":
    unittest.main()
