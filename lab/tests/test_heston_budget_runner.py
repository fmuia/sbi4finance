"""Admission requires finite resources, same-bank smoke and explicit readiness."""
import json
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_heston_budget_study as runner


class BudgetRunnerTests(unittest.TestCase):
    def fixture(self, root):
        protocol_path = runner.DEFAULT_PROTOCOL
        protocol = runner.read_json(protocol_path)
        readiness = root / "readiness.json"
        readiness.write_text(json.dumps({"format": "heston_training_budget_readiness_v1",
            "protocol_sha256": runner.digest(protocol_path),
            "control_checkpoint_sha256": protocol["control"]["checkpoint_sha256"],
            "bank_manifest_sha256": protocol["bank"]["manifest_sha256"],
            "status": "pass", "reserved_test_evaluated": False,
            "training_performed": False, "new_mcmc_performed": False,
            "gates": {key: True for key in ("finite_in_support_without_clipping", "development_rank_screen",
                                            "conditional_all_cases", "predictive_pricing_completed")}}))
        smoke = root / "smoke.json"
        smoke.write_text(json.dumps({"format": "full_surface_training_v1", "checkpoint_selection": "endpoint",
            "same_seed_reload_exact": True, "scientific_validation": False, "reserved_test_evaluated": False,
            "task_fingerprint": protocol["bank"]["task_fingerprint"],
            "run_contract": {"format": "full_surface_update_budget_run_v1", "settings": {
                "updates": 3, "batch_size": 256, "steps_per_execution": 1, "checkpoint_selection": "endpoint"},
                "dataset": {"manifest_sha256": protocol["bank"]["manifest_sha256"],
                            "task_fingerprint": protocol["bank"]["task_fingerprint"]}},
            "selected_update": 3, "performance": {"work": {"optimizer_updates": 3,
                "available_training_rows": 100000, "example_presentations": 768, "unique_rows_seen": 768,
                "reserved_test_evaluated": False}, "runtime": runner.runtime_identity(),
                "hardware": {"devices": ["fixture:cpu"]}, "peak_process_rss_bytes": 2**30}}))
        admission = {"format": "heston_training_budget_admission_v1",
            "protocol_sha256": runner.digest(protocol_path), "readiness_sha256": runner.digest(readiness),
            "smoke_training_sha256": runner.digest(smoke), "source_revision": "frozen-revision",
            "host_cpus": 4, "gpu_count": 0, "cleanup_seconds_per_lane": 5,
            "lane_wall_seconds": {"small-u00030-s0": 20, "full-u00030-s0": 30},
            "already_used": {"allocated_cpu_seconds": 40, "gpu_seconds": 10},
            "reserved_confirmation": {"allocated_cpu_seconds": 2000, "gpu_seconds": 500},
            "source_manifest_sha256": runner.executing_sources_digest(),
            "runtime": runner.runtime_identity(), "execution_devices": ["fixture:cpu"],
            "memory": {"host_allocation_bytes": 24 * 2**30, "forecast_peak_host_bytes": 2 * 2**30,
                       "forecast_peak_device_bytes": 0, "available_device_bytes": 0},
            "approved_for_execution": True}
        path = root / "admission.json"; path.write_text(json.dumps(admission))
        return protocol_path, readiness, smoke, path, admission

    def inconclusive_fixture(self, root):
        p, r, s, a, admission = self.fixture(root)
        protocol, readiness = runner.read_json(p), runner.read_json(r)
        readiness["status"] = "inconclusive"
        readiness["gates"]["conditional_all_cases"] = False
        readiness["conditional"] = {}
        for index in protocol["conditional"]["case_indices"]:
            parameters = {name: {"status": "pass", "mean_status": "pass", "width_status": "pass",
                                 "absolute_standardized_mean_difference": .1, "width_ratio_90": 1.}
                          for name in runner.HESTON_TARGETS}
            if index == 2:
                parameters["initial_variance"].update(status="inconclusive", mean_status="inconclusive",
                                                      absolute_standardized_mean_difference=.24)
            readiness["conditional"][str(index)] = {
                "format": "heston_budget_conditional_agreement_v1",
                "status": "inconclusive" if index == 2 else "pass", "parameters": parameters,
                "reference_diagnostics_passed": True, "finite_in_support_without_clipping": True,
                "input_sha256": {name: "a" * 64 for name in ("reference.json", "diagnostics.json", "chains.npz")}}
        r.write_text(json.dumps(readiness))
        admission["readiness_sha256"] = runner.digest(r)
        admission["user_authorized_exploratory_control_override"] = {
            "authorized": True, "authorization_source": "User: run the training matrix",
            "rationale": "Explore the fixed seed-0 screen while retaining the unresolved reference interval.",
            "readiness_sha256": runner.digest(r), "scope": "seed0_development_screen_only",
            "scientific_acceptance": False, "model_publication_allowed": False,
            "reserved_test_evaluation_allowed": False, "confirmation_allowed": False}
        a.write_text(json.dumps(admission))
        return p, r, s, a, admission

    def test_explicit_exploration_retains_the_original_control_verdict(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, r, s, a, admission = self.inconclusive_fixture(Path(temporary))
            original = r.read_bytes()
            protocol, admitted = runner.validate_admission(p, r, s, a, "frozen-revision")
            policy = runner.readiness_admission(protocol, runner.read_json(r), r, admitted)
            self.assertEqual(policy["mode"], "user_authorized_exploratory_screen")
            self.assertEqual(policy["original_control_status"], "inconclusive")
            self.assertFalse(policy["original_control_gates"]["conditional_all_cases"])
            self.assertFalse(policy["scientific_acceptance"])
            self.assertFalse(policy["model_publication_allowed"])
            self.assertFalse(policy["reserved_test_evaluation_allowed"])
            self.assertFalse(policy["confirmation_allowed"])
            self.assertEqual(r.read_bytes(), original)
            without = {key: value for key, value in admission.items()
                       if key != "user_authorized_exploratory_control_override"}
            a.write_text(json.dumps(without))
            with self.assertRaisesRegex(ValueError, "every gate"):
                runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_exploration_cannot_bypass_failed_or_missing_reference_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, r, s, a, admission = self.inconclusive_fixture(Path(temporary))
            original = runner.read_json(r)
            mutations = [lambda report: report.update(status="fail"),
                lambda report: report["gates"].update(development_rank_screen=False),
                lambda report: report["gates"].update(finite_in_support_without_clipping=False),
                lambda report: report["gates"].update(predictive_pricing_completed=False),
                lambda report: report["conditional"]["2"].update(reference_diagnostics_passed=False),
                lambda report: report["conditional"]["2"].update(finite_in_support_without_clipping=False),
                lambda report: report["conditional"]["2"].update(status="fail"),
                lambda report: report["conditional"].pop("0"),
                lambda report: report["conditional"]["2"].pop("input_sha256"),
                lambda report: report["conditional"]["2"]["parameters"].pop("rho"),
                lambda report: report["conditional"]["2"]["parameters"]["initial_variance"].update(mean_status="fail"),
                lambda report: report["conditional"]["2"]["parameters"]["initial_variance"].update(
                    absolute_standardized_mean_difference=.31)]
            for mutate in mutations:
                report = copy.deepcopy(original)
                mutate(report)
                r.write_text(json.dumps(report))
                changed = copy.deepcopy(admission)
                changed["readiness_sha256"] = runner.digest(r)
                changed["user_authorized_exploratory_control_override"]["readiness_sha256"] = runner.digest(r)
                a.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_exploration_permission_is_hashed_and_cannot_authorize_a_later_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, r, s, a, admission = self.inconclusive_fixture(Path(temporary))
            changes = [{"authorized": False}, {"readiness_sha256": "another report"}, {"rationale": ""},
                       {"scope": "final_confirmation"}, {"scientific_acceptance": True},
                       {"model_publication_allowed": True}, {"reserved_test_evaluation_allowed": True},
                       {"confirmation_allowed": True}, {"authorization_source": None}]
            for change in changes:
                altered = copy.deepcopy(admission)
                altered["user_authorized_exploratory_control_override"].update(change)
                a.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_omitting_confirmation_reserve_requires_an_explicit_scope_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, r, s, a, admission = self.inconclusive_fixture(Path(temporary))
            admission["reserved_confirmation"] = {"allocated_cpu_seconds": 0, "gpu_seconds": 0}
            a.write_text(json.dumps(admission))
            with self.assertRaisesRegex(ValueError, "final CPU reserve"):
                runner.validate_admission(p, r, s, a, "frozen-revision")
            admission["user_authorized_exploratory_control_override"]["confirmation_not_in_current_scope"] = True
            a.write_text(json.dumps(admission))
            runner.validate_admission(p, r, s, a, "frozen-revision")
            admission["lane_wall_seconds"] = {"small-u00030-s0": 10000}
            a.write_text(json.dumps(admission))
            with self.assertRaisesRegex(ValueError, "resource ceiling"):
                runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_admission_binds_evidence_and_caps_total_including_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); p, r, s, a, admission = self.fixture(root)
            runner.validate_admission(p, r, s, a, "frozen-revision")
            cpu_only = {**admission, "reserved_confirmation": {"allocated_cpu_seconds": 2000, "gpu_seconds": 0}}
            a.write_text(json.dumps(cpu_only))
            runner.validate_admission(p, r, s, a, "frozen-revision")
            for changes, error in (({"source_revision": "changed"}, "identity changed"),
                ({"approved_for_execution": False}, "explicitly admitted"),
                ({"reserved_confirmation": {"allocated_cpu_seconds": 21600, "gpu_seconds": 1}}, "resource ceiling")):
                a.write_text(json.dumps({**admission, **changes}))
                with self.assertRaisesRegex(ValueError, error):
                    runner.validate_admission(p, r, s, a, "frozen-revision")
            a.write_text(json.dumps(admission))
            r.write_text(r.read_text().replace('"pass"', '"inconclusive"'))
            with self.assertRaisesRegex(ValueError, "every gate"):
                runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_nonfinite_boolean_and_missing_resource_proofs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); p, r, s, a, admission = self.fixture(root)
            changes = [{"lane_wall_seconds": {"small-u00030-s0": value}}
                       for value in (float("nan"), float("inf"), True)]
            changes += [{"gpu_count": True}, {"cleanup_seconds_per_lane": float("nan")},
                {"already_used": {"allocated_cpu_seconds": float("nan"), "gpu_seconds": 0}},
                {"reserved_confirmation": {"allocated_cpu_seconds": 0, "gpu_seconds": 0}},
                {"memory": {**admission["memory"], "forecast_peak_host_bytes": 1}},
                {"memory": {**admission["memory"], "host_allocation_bytes": 25 * 2**30}},
                {"runtime": {}}, {"execution_devices": ["another device"]},
                {"source_manifest_sha256": "changed executing source"}]
            for change in changes:
                with self.subTest(change=change):
                    a.write_text(json.dumps({**admission, **change}))
                    with self.assertRaises(ValueError):
                        runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_status_only_readiness_and_missing_or_wrong_bank_smoke_cannot_admit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); p, r, s, a, admission = self.fixture(root)
            original_readiness, original_smoke = runner.read_json(r), runner.read_json(s)
            for gates in ({}, {**original_readiness["gates"], "conditional_all_cases": False}):
                r.write_text(json.dumps({**original_readiness, "gates": gates}))
                admission["readiness_sha256"] = runner.digest(r); a.write_text(json.dumps(admission))
                with self.assertRaisesRegex(ValueError, "every gate"):
                    runner.validate_admission(p, r, s, a, "frozen-revision")
            r.write_text(json.dumps(original_readiness)); admission["readiness_sha256"] = runner.digest(r)
            for change in ({"selected_update": None}, {"selected_update": 0},
                           {"task_fingerprint": "another task"}, {"performance": {}},
                           {"run_contract": {"format": "full_surface_update_budget_run_v1"}}):
                s.write_text(json.dumps({**original_smoke, **change}))
                admission["smoke_training_sha256"] = runner.digest(s); a.write_text(json.dumps(admission))
                with self.assertRaisesRegex(ValueError, "smoke evidence"):
                    runner.validate_admission(p, r, s, a, "frozen-revision")

    def test_execution_requires_real_cpu_binding_and_matched_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, _, admission = self.fixture(Path(temporary))
            device = type("Device", (), {"platform": "cpu", "__str__": lambda self: "fixture:cpu"})()
            with patch.dict(runner.os.environ, {}, clear=True), \
                    patch.object(runner.os, "sched_getaffinity", return_value=set(range(8)), create=True):
                with self.assertRaisesRegex(ValueError, "thread variables alone"):
                    runner.verify_execution_resources(admission)
            with patch.dict(runner.os.environ, {"SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "24576"}, clear=True), \
                    patch.dict(runner.sys.modules, {"jax": SimpleNamespace(devices=lambda: [device])}):
                result = runner.verify_execution_resources(admission)
                self.assertTrue(result["blas_thread_values_are_not_cpu_binding"])
                self.assertFalse(result["python_enforces_memory_limit"])
            with patch.object(runner.subprocess, "check_output", return_value="?? src/new_runtime.py\n"):
                self.assertFalse(runner.clean_source())

    def test_reservations_are_independent_nonretryable_and_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, _, _, a, admission = self.fixture(root)
            out = root / "lanes"
            runner.reserve_lane(out, "small-u00030-s0", a, admission)
            with self.assertRaisesRegex(ValueError, "already reserved"):
                runner.reserve_lane(out, "small-u00030-s0", a, admission)
            runner.reserve_lane(out, "full-u00030-s0", a, admission)
            self.assertEqual(len(runner.read_json(out / "ledger.json")["lanes"]), 2)
            with runner.ledger_lock(out):
                with self.assertRaisesRegex(ValueError, "ledger is locked"):
                    runner.reserve_lane(out, "small-u00030-s0", a, admission)

    def test_preparation_is_charged_and_cannot_start_an_expired_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p, r, s, a, admission = self.fixture(root)
            bank = root / "bank"
            bank.mkdir()
            (bank / "manifest.json").write_text("{}")
            protocol = runner.read_json(p)
            protocol["bank"]["manifest_sha256"] = runner.digest(bank / "manifest.json")
            readiness = runner.read_json(r)
            readiness["bank_manifest_sha256"] = protocol["bank"]["manifest_sha256"]
            r.write_text(json.dumps(readiness))
            args = SimpleNamespace(protocol=p, readiness=r, smoke=s, admission=a, bank=bank,
                                   out=root / "lanes", candidate="small-u00030-s0")
            with patch.object(runner, "clean_source", return_value=True), \
                    patch.object(runner, "source_revision", return_value="frozen-revision"), \
                    patch.object(runner, "executing_sources_digest", return_value="frozen-sources"), \
                    patch.object(runner, "validate_admission", return_value=(protocol, admission)), \
                    patch.object(runner, "verify_execution_resources", return_value={}), \
                    patch.object(runner.time, "perf_counter", side_effect=[0., 25., 25.]), \
                    patch.object(runner.subprocess, "run") as launch:
                with self.assertRaisesRegex(TimeoutError, "preparation exhausted"):
                    runner.run(args)
            launch.assert_not_called()
            journal = runner.read_json(args.out / args.candidate / "lane.json")
            self.assertEqual(journal["status"], "failed")
            self.assertEqual(journal["elapsed_wall_seconds"], 25.)
            self.assertEqual(journal["allocated_cpu_seconds"], 100.)

    def test_candidates_and_commands_keep_task_recipe(self):
        protocol = runner.read_json(runner.DEFAULT_PROTOCOL)
        candidates = runner.candidates(protocol)
        self.assertEqual(len(candidates), 14)
        self.assertEqual(sum(row["updates"] for row in candidates.values()), 88860)
        command = runner.command(candidates["full-u30000-s0"], "/caller/bank", "/caller/out")
        self.assertEqual(command[command.index("--updates") + 1], "30000")
        self.assertEqual(command[command.index("--widths") + 1], "256,256,256,256")
        self.assertEqual(command[command.index("--checkpoint-selection") + 1], "final")
        self.assertFalse(any("oscar" in item for item in command))
        candidate = candidates["full-u30000-s0"]
        actual = runner.lane_command(candidate, "/caller/bank", Path("/caller/lane"), Path("/caller/protocol"),
                                     Path("/caller/readiness"), 123.5,
                                     {"execution_scope": "training_and_development_evaluation"})
        self.assertEqual(actual[actual.index("--candidate") + 1], candidate["id"])
        self.assertEqual(actual[actual.index("--wall-seconds") + 1], "123.5")
        self.assertEqual(actual[actual.index("--out") + 1], "/caller/lane")
        self.assertIn("--control-readiness", actual)
