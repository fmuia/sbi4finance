"""Portable aggregate reports must preserve pairing, endpoint roles and privacy."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import heston_flexible_report as report


def population():
    return {"parameter_names": list(report.NAMES), "rows": 512, "draws_per_case": 512,
            "truth_sha256": "shared-truth", "observations_sha256": "shared-observations",
            "rank_dkw": {"passed": [True] * 6},
            "support": {"outside_prior": {"sample_fraction": .02}},
            "recovery": {"median_rmse": [.1] * 6, "median_posterior_width90": [.3] * 6},
            "coverage": {"0.9": {"coverage": [.9] * 6}},
            "fractional_ranks": [[.123456] * 6],
            "raw_draw_payload": {"path": "/private/not-for-export/draws.npz"}}


def evaluation():
    comparison = {"status": "defined", "quantities": list(report.NAMES),
                  "standardized_mean_error": [.2] * 6, "width_ratio": [1.1] * 6,
                  "neural_mean": [12345.6789] * 6, "reference_quantiles_05_50_95": [[98765.4321] * 6]}
    return {"status": "complete", "endpoint": "final", "population": population(),
            "cases": {name: {"reference_ready": True, "comparison": deepcopy(comparison),
                             "support": {"samples": 4096}, "reference_chain_sha256": "chain-" + name,
                             "reference_chain_payload": {"path": "/private/chains.npz", "sha256": "chain-" + name}}
                      for name in report.CASES}}


def put(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, dimension=32, best_epoch=258):
    baseline_dir = root / "baseline"
    old_run = {"data_payload_sha256": "shared-bank", "elapsed_seconds": 1800,
               "actual_optimizer_updates": 351600}
    put(baseline_dir / "run.json", old_run)
    put(baseline_dir / "bank.json", {"data_payload_sha256": "shared-bank"})
    binding = lambda path: {"path": str(path), "sha256": report.digest(path)}
    baseline = evaluation()
    baseline.update(format="heston_pr87_grid_evaluation_v1", training_rows=300000, native_epoch=300,
                    endpoint="primary_final", scientific_acceptance=False,
                    task={"parameter_names": list(report.NAMES), "quote_count": 63},
                    training_presentations=90000000, training_run=binding(baseline_dir / "run.json"),
                    common_bank=binding(baseline_dir / "bank.json"))
    put(baseline_dir / "evaluation.json", baseline)
    run_dir = root / f"mlp{dimension}-seed9011"
    run_dir.mkdir()
    for name in ("model.keras", "best.keras", "draws.npz", "best-draws.npz"):
        (run_dir / name).write_bytes(name.encode())
    task = {"target": baseline["task"], "records": {"mid": [12345.6789] * 63}}
    put(run_dir / "task.json", task)
    put(run_dir / "preprocessing.json", {"kind": "frozen_zca"})
    run = {"status": "complete", "smoke_test_only": False, "train_rows": 300000,
           "epochs": 300, "completed_epochs": 300, "best_epoch": best_epoch,
           "task": task, "data_payload_sha256": "shared-bank", "elapsed_seconds": 2200,
           "configuration": {"representation": "mlp", "summary_dim": dimension},
           "optimizer_updates": 351600, "presentations": 90000000,
           "task_sha256": report.digest(run_dir / "task.json"),
           "preprocessing_sha256": report.digest(run_dir / "preprocessing.json"),
           "model_sha256": report.digest(run_dir / "model.keras"),
           "best_model_sha256": report.digest(run_dir / "best.keras"),
           "bundle": str(root / "private-bank"),
           "source": {"commit": "source-commit", "runner_sha256": "runner-hash", "path": str(root / "private-source")}}
    put(run_dir / "run.json", run)
    put(run_dir / "history.json", {"epoch": list(range(1, 301)), "loss": [2.] * 300, "val_loss": [3.] * 300})
    for role, filename, draws in (("final", "evaluation.json", "draws.npz"), ("best", "best-evaluation.json", "best-draws.npz")):
        record = evaluation()
        record.update(endpoint=role, draws_sha256=report.digest(run_dir / draws))
        put(run_dir / filename, record)
    return baseline_dir / "evaluation.json", run_dir


class FlexibleReportTests(unittest.TestCase):
    def test_compact_records_exclude_raw_draws_ranks_parameters_and_owner_paths(self):
        compact = report.compact_evaluation(evaluation(), endpoint="primary_final", epoch=300, source_sha256="source")
        self.assertNotIn("fractional_ranks", compact["population"])
        self.assertNotIn("raw_draw_payload", compact["population"])
        self.assertNotIn("comparison", compact["cases"][report.CASES[0]])
        self.assertEqual(compact["parameters"]["c"]["median_width90"], .3)
        self.assertNotIn("mean_width90", compact["parameters"]["c"])
        self.assertFalse(compact["scientific_acceptance"])
        with self.assertRaisesRegex(ValueError, "absolute paths"):
            report.portable({"nested": [{"accidental": "/Users/person/private"}]})

    def test_bound_metadata_requires_the_recorded_hash_even_after_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "relocated.json"
            put(path, {"aggregate": 1})
            binding = {"path": "/unavailable/original.json", "sha256": report.digest(path)}
            self.assertEqual(report.bound_json(binding, override=path), {"aggregate": 1})
            put(path, {"aggregate": 2})
            with self.assertRaisesRegex(ValueError, "checksum"):
                report.bound_json(binding, override=path)

    def test_export_preserves_endpoint_roles_is_portable_and_can_append_another_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, first = fixture(root)
            out = root / "export"
            study = report.export_study([first], baseline, out)
            self.assertEqual(len(study["runs"]), 3)
            roles = [entry["endpoint"] for entry in study["runs"]]
            self.assertEqual(roles.count("primary_final"), 2)
            self.assertEqual(roles.count("secondary_validation_minimum"), 1)
            second_root = root / "another"
            second_root.mkdir()
            _, second = fixture(second_root, dimension=16)
            study = report.export_study([second], baseline, out)
            self.assertEqual(len(study["runs"]), 5)
            self.assertTrue(all(not Path(entry["directory"]).is_absolute() for entry in study["runs"]))
            for path in out.rglob("*"):
                if path.is_file():
                    self.assertIn(path.suffix, (".json", ".html"))
                    text = path.read_text()
                    self.assertNotIn(str(root), text)
                    self.assertNotIn("12345.6789", text)
                    self.assertNotIn("98765.4321", text)
            html = (out / "report.html").read_text()
            self.assertIn("Development evidence only", html)
            self.assertIn("validation minimum 258 (secondary)", html)
            self.assertIn("<svg", html)
            self.assertNotIn("<script", html)
            self.assertNotIn('src="http', html)

    def test_intermediate_exports_append_paired_endpoints_without_changing_existing_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, first = fixture(root)
            out = root / "export"
            report.export_study([first], baseline, out)
            protected = {path: path.read_bytes() for path in out.rglob("*.json")
                         if path.name not in ("study.json", "report-summary.json")}
            additions = []
            for dimension in (7, 6, 5):
                source_root = root / f"source{dimension}"
                source_root.mkdir()
                _, source = fixture(source_root, dimension=dimension)
                additions.append(source)
            study = report.export_study(additions, baseline, out)
            self.assertEqual(len(study["runs"]), 9)
            for dimension in (7, 6, 5):
                entries = [item for item in study["runs"] if item["summary_dim"] == dimension]
                self.assertEqual({entry["endpoint"] for entry in entries},
                                 {"primary_final", "secondary_validation_minimum"})
            for path, original in protected.items():
                self.assertEqual(path.read_bytes(), original)
            html = (out / "report.html").read_text()
            self.assertIn("32, 7, 6, 5", html)
            self.assertIn("one seed per size cannot establish a universal information cutoff", html)

    def test_changed_pairing_or_missing_secondary_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, run = fixture(root)
            secondary = run / "best-evaluation.json"
            secondary.unlink()
            with self.assertRaises(FileNotFoundError):
                report.export_study([run], baseline, root / "export")
            self.assertFalse((root / "export").exists())
            record = evaluation()
            record.update(endpoint="best", draws_sha256=report.digest(run / "best-draws.npz"))
            record["population"]["truth_sha256"] = "different-truth"
            put(secondary, record)
            with self.assertRaisesRegex(ValueError, "pairing"):
                report.export_study([run], baseline, root / "export")
            self.assertFalse((root / "export").exists())


if __name__ == "__main__":
    unittest.main()
