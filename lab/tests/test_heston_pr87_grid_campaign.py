"""Exercise shared-bank scope, reference concurrency and worker failure controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_heston_pr87_grid as controller


class GridCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(controller.subprocess, "check_output", return_value="fixture-revision\n")
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.input = self.root / "input.json"
        self.dump(self.input, {})
        self.bound = controller.identity(self.input)
        self.sources = {"controller": controller.identity(controller.__file__)}
        for key in ("producer", "evaluator", "grid_reference", "black_formula"):
            path = self.root / (key + ".py")
            path.write_text("# supervised fixture; never executed\n")
            self.sources[key] = controller.identity(path)
        preserved = self.root / "preserved.json"
        self.dump(preserved, {"files": [{**self.bound, "bytes": self.input.stat().st_size}]})
        self.bank_contract = {"format": "heston_pr87_grid_bank_contract_v1", "train_rows": 300000,
            "validation_rows": 100, "historical_rows": 512, "quote_count": 63, "whitening_rows": 10000,
            "source": {}}
        self.bank_path = self.root / "bank-protocol.json"
        self.dump(self.bank_path, {"contract": self.bank_contract, "task": {}, "output": str(self.root / "bank")})
        self.reference = {"format": "heston_pr87_grid_reference_protocol_v1", "panel_path": str(self.input),
            "settings": {"smoke_test_only": False, "inner_cpu_and_wall_seconds": 900, "outer_total_wall_seconds": 950},
            "campaign_limits": {"maximum_concurrent_reference_processes": 2}, "jobs": []}
        for case in ("synthetic-broad", "synthetic-market-like", "market"):
            self.reference["jobs"].append({"id": case, "evidence_out": str(self.root / ("reference-" + case)),
                "payload_out": str(self.root / ("reference-payload-" + case))})
        reference_path = self.root / "reference-protocol.json"
        self.dump(reference_path, self.reference)
        self.campaign = {"format": controller.FORMAT, "predeclared_before_bank": True,
            "scientific_acceptance": False, "limits": controller.LIMITS, "sources": self.sources,
            "readiness": {"task": self.bound}, "historical_inputs": self.bound, "old54_control": self.bound,
            "native_smoke": self.bound, "parent_training": str(self.root / "old-parent"),
            "bank_protocol": controller.identity(self.bank_path), "bank_log": str(self.root / "bank.log"),
            "reference_protocol": controller.identity(reference_path),
            "reference_logs": {job["id"]: str(self.root / (job["id"] + ".log")) for job in self.reference["jobs"]},
            "reference_cache_root": str(self.root / "cache"), "preserved_artifacts": controller.identity(preserved), "runs": {}}
        self.studies = {}
        for name, rows in controller.RUNS:
            folder = self.root / name
            folder.mkdir()
            study = {"contract": {"format": "heston_pr87_grid_training_contract_v1", "train_rows": rows,
                    "epochs": 300, "batch_size": 256, "bank_contract": self.bank_contract},
                "task": {}, "sources": self.sources, "bank_protocol": self.campaign["bank_protocol"],
                "reference_protocol": self.campaign["reference_protocol"], "endpoint_policy": controller.ENDPOINTS,
                "predeclared_before_bank": True, "output": str(folder / "training"), "training_log": str(folder / "training.log"),
                "evaluations": {}}
            for role in ("primary", "secondary"):
                study["evaluations"][role] = {"checkpoint": "epoch-000300" if role == "primary" else "best-validation",
                    "output": str(folder / role), "payload_output": str(folder / (role + "-payload")),
                    "log": str(folder / (role + ".log"))}
            self.studies[name] = study
            self.persist_study(name, rows)
        self.protocol = self.root / "campaign-protocol.json"
        self.dump(self.protocol, self.campaign)

    @staticmethod
    def dump(path, value):
        Path(path).write_text(json.dumps(value))

    def persist_study(self, name, rows):
        path = self.root / name / "protocol.json"
        self.dump(path, self.studies[name])
        self.campaign["runs"][name] = {"train_rows": rows, "protocol": controller.identity(path)}

    def fake_workers(self, *, best=250, failure=None, hang=False):
        calls, events, workers = [], [], []
        class Worker:
            def __init__(worker, command, **kwargs):
                calls.append(command)
                worker.pid = 9000 + len(calls)
                worker.command = command
                worker.kind = "reference" if "--case-id" in command else "evaluation" if "--checkpoint" in command else command[2]
                worker.label = command[command.index("--case-id") + 1] if worker.kind == "reference" else worker.kind
                worker.polls, worker.code = 0, None
                worker.env = kwargs["env"]
                events.append(("start", worker.label, tuple(item.kind for item in workers if item.code is None)))
                workers.append(worker)

            def poll(worker):
                if worker.code is not None: return worker.code
                worker.polls += 1
                if hang or worker.polls < (12 if worker.kind == "reference" else 2): return None
                worker.code = int(worker.label == failure)
                events.append(("end", worker.label, worker.code))
                if worker.code: return worker.code
                out = Path(worker.command[worker.command.index("--out") + 1]); out.mkdir()
                if worker.kind == "reference":
                    filename, record = "run.json", {"status": "completed", "case_id": worker.label,
                        "reference_ready": worker.label != "synthetic-market-like"}
                elif worker.kind == "generate": filename, record = "bank.json", {"status": "complete"}
                elif worker.kind == "train":
                    filename, record = "training.json", {"status": "complete", "best_validation_archive": {"epoch": best}}
                else: filename, record = "evaluation.json", {"status": "complete"}
                (out / filename).write_text(json.dumps(record))
                return worker.code
        return Worker, calls, events, workers

    def launch(self, **kwargs):
        worker, calls, events, workers = self.fake_workers(**kwargs)
        with patch.object(controller.subprocess, "Popen", worker), patch.object(controller.time, "sleep"):
            result = controller.run(self.protocol, self.root / "campaign.json")
        return result, calls, events, workers

    def test_shared_bank_two_fits_bounded_parallel_references_then_evaluations(self):
        hashes = [spec["protocol"]["sha256"] for spec in self.campaign["runs"].values()]
        result, calls, events, workers = self.launch()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(calls), 10)  # one bank, three refs, two fits, four endpoints
        self.assertEqual(list(result["runs"]), ["200k-300", "300k-300"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertFalse(result["references"]["synthetic-market-like"]["reference_ready"])
        for event in events:
            if event[0] == "start":
                self.assertLessEqual(event[2].count("reference") + int(event[1] in ("synthetic-broad", "synthetic-market-like", "market")), 2)
                self.assertLessEqual(len(event[2]), 2)
                if event[1] == "evaluation":
                    self.assertFalse("reference" in event[2])
        first_fit = next(event for event in events if event[:2] == ("start", "train"))
        self.assertIn("reference", first_fit[2])
        first_eval = next(index for index, event in enumerate(events) if event[:2] == ("start", "evaluation"))
        self.assertTrue(all(next(index for index, event in enumerate(events) if event[:2] == ("end", case)) < first_eval
                            for case in ("synthetic-broad", "synthetic-market-like", "market")))
        self.assertEqual(hashes, [controller.identity(spec["protocol"]["path"])["sha256"] for spec in self.campaign["runs"].values()])
        ref_workers = [worker for worker in workers if worker.kind == "reference"]
        self.assertEqual(len({worker.env["PYTENSOR_FLAGS"] for worker in ref_workers}), 3)

    def test_same_final_minimum_skips_both_secondary_queries(self):
        result, calls, _, _ = self.launch(best=300)
        self.assertEqual(len(calls), 8)
        self.assertTrue(all("secondary_skipped" in row for row in result["runs"].values()))

    def test_reference_failure_stops_active_peer_and_forbids_retry(self):
        worker, calls, _, _ = self.fake_workers(failure="synthetic-broad")
        with patch.object(controller.subprocess, "Popen", worker), patch.object(controller.time, "sleep"), patch.object(controller.shared, "stop") as stop:
            with self.assertRaisesRegex(RuntimeError, "reference_synthetic-broad"):
                controller.run(self.protocol, self.root / "campaign.json")
            self.assertTrue(stop.called)
            with self.assertRaisesRegex(FileExistsError, "no retries"):
                controller.run(self.protocol, self.root / "campaign.json")
        self.assertFalse(any("--checkpoint" in call for call in calls))

    def test_reference_outer_timeout_stops_every_worker(self):
        worker, _, _, _ = self.fake_workers(hang=True)
        tick = iter([0, 0, 0, 0, 0, 0, 0, 951, 951, 951, 951, 951])
        with patch.object(controller.subprocess, "Popen", worker), patch.object(controller.time, "sleep"), patch.object(
                controller.time, "monotonic", side_effect=lambda: next(tick, 951)), patch.object(controller.shared, "stop") as stop:
            with self.assertRaisesRegex(TimeoutError, "reference_"):
                controller.run(self.protocol, self.root / "campaign.json")
            self.assertGreaterEqual(stop.call_count, 2)

    def test_changed_preserved_input_refuses_before_any_worker_or_claim(self):
        self.input.write_text('{"changed":true}')
        with patch.object(controller.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "changed"):
                controller.run(self.protocol, self.root / "campaign.json")
            launch.assert_not_called()
        self.assertFalse((self.root / "campaign.claim.json").exists())

    def test_duplicate_outputs_or_existing_secondary_refuse(self):
        original = copy.deepcopy(self.studies["300k-300"])
        self.studies["300k-300"]["evaluations"]["secondary"]["output"] = self.studies["200k-300"]["evaluations"]["primary"]["output"]
        self.persist_study("300k-300", 300000)
        with self.assertRaisesRegex(ValueError, "distinct"): controller.validate(self.campaign)
        self.studies["300k-300"] = original
        self.persist_study("300k-300", 300000); self.dump(self.protocol, self.campaign)
        Path(original["evaluations"]["secondary"]["payload_output"]).mkdir()
        with self.assertRaisesRegex(FileExistsError, "no retries"):
            controller.run(self.protocol, self.root / "campaign.json")

    def readiness_fixture(self):
        prep = self.root / "preparation"; prep.mkdir(); (prep / "numerical-pilot").mkdir()
        target = {"quote_count": 63, "pricing_controls": {"fourier_representation": "black_control_variate_v1"},
            "prior_bounds": [[.01,.16],[.1,10.],[.01,.16],[.1,3.],[-.95,-.1],[.5,2.]]}
        self.dump(prep / "task.json", {"target": target}); self.dump(prep / "design.json", {})
        protocol = prep / "preparation-protocol.json"
        self.dump(protocol, {"task": controller.identity(prep / "task.json"), "design": controller.identity(prep / "design.json"),
            "sources": {"pricer": self.sources["black_formula"]}})
        panel = prep / "panel.json"
        self.dump(panel, {"task": target, "design": controller.identity(prep / "design.json")})
        self.dump(prep / "numerical-pilot" / "run.json", {"status": "complete", "reserved_arrays_opened": False,
            "points": [{"index": i, "pointwise_gates_passed": True} for i in range(98)],
            "protocol": controller.identity(protocol), "panel": controller.identity(panel)})
        smoke = prep / "smoke.json"
        self.dump(smoke, {"format": "heston_pr87_grid_reference_run_v1", "status": "completed", "smoke": True,
            "attempts": [{}], "task": target, "truth_available_to_sampler": False, "protocol": controller.identity(protocol)})
        return prep, smoke

    def test_readiness_requires_all_98_points_and_exact_frozen_task(self):
        prep, smoke = self.readiness_fixture()
        controller.readiness(prep, smoke)
        task = json.loads((prep / "task.json").read_text())
        task["target"]["new_pricing_control"] = 1
        self.dump(prep / "task.json", task)
        with self.assertRaisesRegex(ValueError, "bind the current"): controller.readiness(prep, smoke)

    def test_readiness_rejects_changed_pricer_and_failed_numerical_point(self):
        prep, smoke = self.readiness_fixture()
        Path(self.sources["black_formula"]["path"]).write_text("# changed pricer\n")
        with self.assertRaisesRegex(ValueError, "changed"): controller.readiness(prep, smoke)
        pilot_path = prep / "numerical-pilot" / "run.json"
        pilot = json.loads(pilot_path.read_text()); pilot["points"][74]["pointwise_gates_passed"] = False
        self.dump(pilot_path, pilot)
        with self.assertRaisesRegex(ValueError, "complete98-point"): controller.readiness(prep, smoke)

    def test_reference_caches_replace_inherited_cache_only(self):
        env = controller.reference_environment({"PYTENSOR_FLAGS": "floatX=float64,compiledir=/old,base_compiledir=/older,optimizer=fast_run"}, "/fresh")
        self.assertEqual(env["PYTENSOR_FLAGS"], "floatX=float64,optimizer=fast_run,base_compiledir=/fresh")


if __name__ == "__main__":
    unittest.main()
