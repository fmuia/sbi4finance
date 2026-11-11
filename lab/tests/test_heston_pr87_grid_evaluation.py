"""New63 diagnostics preserve raw policies and reject old54 reference targets."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from scripts import run_heston_pr87_grid_reference as reference
from scripts import evaluate_heston_pr87_grid as evaluation
from scripts import render_heston_pr87_grid as report
from scripts import export_heston_pr87_grid_figures as figures
from scripts import evaluate_heston_pr87_followup as parent_evaluation
from scripts import render_heston_pr87_followup as parent_report
from sbi4finance.calibration.heston_grid import HestonGridPilot
from tests.test_heston_spread import records as small_records
from tests.test_heston_pr87_followup_evaluation import fixture as old_fixture


def task_fixture(root):
    records = {key: np.resize(value, 63).tolist() for key, value in small_records().items()}
    pilot = HestonGridPilot(records, infer_c=True)
    design = root/"design.json"
    evaluation.fixed.write_json(design, {"records": records, "products": {"control": [0., .1], "affected": [.1, .89]}})
    return {"format": "heston_pr87_grid_task_v1", "records": records, "target": pilot.task_definition(), "design_sha256": evaluation.fixed.digest(design)}, design


def reference_fixture(root, *, smoke=False):
    task, design = task_fixture(root); mid = np.asarray(task["records"]["mid"])
    panel = {"format": "heston_pr87_grid_reference_panel_v1", "design": reference.binding(design), "task": task["target"],
        "products": {"control": [0., .1], "affected": [.1, .89]}, "cases": []}
    for i, name in enumerate(reference.CASES):
        y = mid if name == "market" else mid+.01+i*.001
        panel["cases"].append({"id": name, "observations": y.tolist(), "observations_sha256": reference.values_digest(y)})
    panel_path = root/"panel.json"; evaluation.fixed.write_json(panel_path, panel); protocol_path = root/"reference-protocol.json"
    protocol = reference.prepare_protocol(panel_path, root/"references", root/"reference-payload", protocol_path, smoke_test_only=smoke)
    return task, panel_path, protocol_path, protocol


def fixture(root, rows=200000, best_epoch=300):
    protocol_path, primary_path, secondary_path, record, run = old_fixture(root, rows=200000, best_epoch=best_epoch)
    task, _ = task_fixture(root); control = deepcopy(record); control["task"] = deepcopy(task["target"]); control["task"]["quote_count"] = 54
    control_path = root/"old54-control.json"; evaluation.fixed.write_json(control_path, control)
    arrays = {}
    for panel, prefix in (("native_validation", "validation"), ("population", "historical")):
        with np.load(record[panel]["recovery_payload"]["path"], allow_pickle=False) as archive: truth = archive["truth"].copy()
        quotes = np.zeros((len(truth), 63)); arrays[prefix+"_quotes"] = quotes; arrays[prefix+"_observations"] = quotes
        for j, name in enumerate(task["target"]["parameter_names"]): arrays[prefix+"_"+name] = truth[:, j]
        record[panel]["observations_sha256"] = evaluation.baseline.observations_digest(quotes)
    arrays["historical_row_id"] = np.array([f"validation:{i:08d}" for i in range(512)])
    arrays["validation_row_id"] = np.array([f"native-validation:{i:06d}" for i in range(100)])
    bank = root/"bank"; bank.mkdir(); evaluation.fixed.write_npz(bank/"data.npz", arrays)
    bank_contract = {"unit_fixture": True}; bank_record = {"format": "heston_pr87_grid_bank_v1", "status": "complete", "task": task,
        "protocol": bank_contract, "data_payload": "data.npz", "whitening_sha256": "w"*64}
    evaluation.fixed.write_json(bank/"bank.json", bank_record)
    steps = (rows+255)//256
    run.update(format="heston_pr87_grid_run_v1", actual_optimizer_updates=steps*300, bank_path=str(bank),
        bank_record_sha256=evaluation.fixed.digest(bank/"bank.json"), data_payload=str(bank/"data.npz"),
        data_payload_sha256=evaluation.fixed.digest(bank/"data.npz"), whitening_sha256="w"*64)
    run["protocol"].update(train_rows=rows, batches_per_epoch=steps, expected_optimizer_updates=steps*300, bank_contract=bank_contract)
    run_path = Path(record["training_run"]["path"]); evaluation.fixed.write_json(run_path, run)
    for name in ("reference-protocol.json", "conditional-panel.json"): evaluation.fixed.write_json(root/name, {})
    protocol = {"task": task, "evaluation": evaluation.evaluation_contract(task["target"]), "old54_control": evaluation.binding(control_path),
        "reference_protocol": evaluation.binding(root/"reference-protocol.json"), "conditional_panel": evaluation.binding(root/"conditional-panel.json"),
        "sources": {"renderer": evaluation.binding(report.__file__), "parent_renderer": evaluation.binding(parent_report.__file__)}}
    evaluation.fixed.write_json(protocol_path, protocol)
    raw = np.tile(np.asarray(task["target"]["prior_bounds"]).mean(axis=1), (4096, 1))
    record.update(format="heston_pr87_grid_evaluation_v1", task=task["target"], protocol=evaluation.binding(protocol_path),
        training_rows=rows, training_presentations=rows*300, actual_optimizer_updates=steps*300, training_accounting=evaluation.training_accounting(run),
        training_run=evaluation.binding(run_path), common_bank=evaluation.binding(bank/"bank.json"), old54_control=protocol["old54_control"],
        reference_protocol=protocol["reference_protocol"], conditional_panel=protocol["conditional_panel"],
        cases={case: {"reference_ready": False, "reason": "new63 reference missing in fixture", "support": evaluation.raw_support(raw, task["target"]["prior_bounds"], task["target"]["parameter_names"])} for case in reference.CASES})
    evaluation.fixed.write_json(primary_path, record)
    secondary = deepcopy(record); secondary.update(native_epoch=best_epoch, actual_optimizer_updates=steps*best_epoch,
        training_presentations=rows*best_epoch, endpoint="secondary_best_validation", checkpoint=evaluation.binding(Path(run["best_validation_archive"]["path"])/"checkpoint.json"))
    evaluation.fixed.write_json(secondary_path, secondary)
    return task, run, control, protocol_path, primary_path, secondary_path, record


class StubBudget:
    def __init__(self, seconds): self.seconds = seconds
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def usage(self): return {"wall_seconds": .01, "cpu_seconds": .01, "limit_seconds": self.seconds}


@contextmanager
def mocked_sampling(midpoint, *, precision=True, mcmc=True):
    def sampler(*args, **kwargs):
        values = np.tile(midpoint, (4, kwargs["draws"], 1))
        parameter = SimpleNamespace(transpose=lambda *args: SimpleNamespace(values=values))
        return {"posterior": {"parameters": parameter}, "sample_stats": SimpleNamespace(data_vars={})}
    with ExitStack() as stack:
        stack.enter_context(patch.object(reference.shared, "Budget", StubBudget))
        stack.enter_context(patch.object(reference.shared.reference, "data_fit_initialization", return_value={}))
        sample = stack.enter_context(patch.object(reference.shared.reference, "sample_deterministic_reference", side_effect=sampler))
        stack.enter_context(patch.object(reference.shared, "diagnose", return_value={"mcmc_diagnostics_passed": mcmc}))
        stack.enter_context(patch.object(reference.shared, "price_audit", return_value=(None, {"pointwise_gates_passed": precision})))
        yield sample


class ReferenceTests(unittest.TestCase):
    def test_new63_panel_production_limits_and_truth_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, panel, _, protocol = reference_fixture(Path(temporary)); _, _, pilot, y = reference.load_case(panel, "market")
            self.assertEqual((len(y), len(pilot.names)), (63, 6)); self.assertFalse(protocol["truth_available_to_sampler"])
            self.assertEqual(protocol["settings"]["inner_cpu_and_wall_seconds"], 900); self.assertEqual(protocol["settings"]["outer_total_wall_seconds"], 950)
            self.assertEqual(reference.validate_protocol(protocol, "market")["task"], task["target"])
            self.assertEqual(protocol["campaign_limits"]["maximum_concurrent_reference_processes"], 2)

    def test_optional_preparation_binding_is_hash_checked_not_truth_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, panel_path, _, _ = reference_fixture(root)
            preparation = root/"preparation-protocol.json"
            evaluation.fixed.write_json(preparation, {"opaque_preparation_metadata": True})
            panel = evaluation.read(panel_path); panel["preparation_protocol"] = reference.binding(preparation)
            evaluation.fixed.write_json(panel_path, panel)
            _, _, _, y = reference.load_case(panel_path, "market"); self.assertEqual(y.shape, (63,))
            evaluation.fixed.write_json(preparation, {"changed": True})
            with self.assertRaisesRegex(ValueError, "identity differs"): reference.load_case(panel_path, "market")

    def test_truth_leak_and_old54_observations_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, panel, _, _ = reference_fixture(Path(temporary)); original = evaluation.read(panel); changed = deepcopy(original)
            changed["cases"][0]["physical_truth"] = [1]*5; evaluation.fixed.write_json(panel, changed)
            with self.assertRaisesRegex(ValueError, "truths"): reference.load_case(panel, "synthetic-broad")
            changed = deepcopy(original); changed["cases"][0]["observations"] = changed["cases"][0]["observations"][:54]
            evaluation.fixed.write_json(panel, changed)
            with self.assertRaisesRegex(ValueError, "63-quote"): reference.load_case(panel, "synthetic-broad")

    def test_old54_target_rejected_before_any_chain_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, panel, _, protocol = reference_fixture(Path(temporary)); target = deepcopy(task["target"]); target["quote_count"] = 54
            _, _, _, y = reference.load_case(panel, "market")
            with patch.object(reference.np, "load", side_effect=AssertionError("must not open chains")):
                with self.assertRaisesRegex(ValueError, "exact posterior"): reference.matched_reference(protocol, "market", target, y, {})

    def test_missing_reference_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, panel, _, protocol = reference_fixture(Path(temporary)); _, _, _, y = reference.load_case(panel, "market")
            chains, result = reference.matched_reference(protocol, "market", task["target"], y, {})
            self.assertIsNone(chains); self.assertFalse(result["reference_ready"]); self.assertEqual(result["status"], "inconclusive")

    def test_canonical63_gaussian_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, _ = task_fixture(Path(temporary)); pilot = HestonGridPilot(task["records"], infer_c=True)
            point = np.array([.04, 2., .05, .4, -.7, 1.3]); clean = np.zeros(63); y = np.ones(63)*.01
            sigma = 1.3*np.asarray(task["records"]["half_spread"])
            expected = -.5*np.sum(((y-clean)/sigma)**2)-np.log(sigma).sum()-63*.5*np.log(2*np.pi)
            with patch.object(pilot, "prices", return_value=clean): self.assertAlmostEqual(pilot.log_likelihood(point, observations=y), expected, places=10)

    def test_smoke_outputs_never_ready_even_when_mock_diagnostics_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, _, path, protocol = reference_fixture(Path(temporary), smoke=True)
            midpoint = np.asarray(task["target"]["prior_bounds"]).mean(axis=1)
            with mocked_sampling(midpoint) as sampler:
                result = reference.run_reference(path, "market", smoke_test_only=True)
            self.assertEqual(result["status"], "completed"); self.assertFalse(result["reference_ready"])
            self.assertEqual(sampler.call_count, 1); self.assertEqual(sampler.call_args.kwargs["draws"], 20)
            self.assertEqual(sampler.call_args.kwargs["chains"], 4); self.assertEqual(sampler.call_args.kwargs["cores"], 1)
            diagnostic = evaluation.read(Path(protocol["jobs"][2]["evidence_out"])/"diagnostics.json")
            self.assertFalse(diagnostic["reference_ready"])

    def test_numerical_failure_preserves_attempt_and_never_escalates(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, _, path, protocol = reference_fixture(Path(temporary))
            midpoint = np.asarray(task["target"]["prior_bounds"]).mean(axis=1)
            with mocked_sampling(midpoint, precision=False, mcmc=False) as sampler:
                with self.assertRaisesRegex(FloatingPointError, "no escalation"): reference.run_reference(path, "market")
            self.assertEqual(sampler.call_count, 1)
            failure = evaluation.read(Path(protocol["jobs"][2]["evidence_out"])/"failure.json")
            self.assertEqual(failure["status"], "failed"); self.assertFalse(failure["reference_ready"])
            self.assertEqual(len(failure["attempts"]), 1)
            self.assertTrue((Path(protocol["jobs"][2]["payload_out"])/"attempt-0/chains.npz").is_file())

    def test_smoke_cannot_be_used_as_production_protocol(self):
        self.assertEqual(reference.settings_contract(smoke_test_only=True)["attempts"], [[20, 20]])
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, protocol = reference_fixture(Path(temporary), smoke=True)
            with self.assertRaisesRegex(ValueError, "limits"): reference.validate_protocol(protocol, "market")


class ConsumerTests(unittest.TestCase):
    def test_dynamic63_contract_and_frozen_raw_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, _ = task_fixture(Path(temporary)); contract = evaluation.evaluation_contract(task["target"])
            self.assertEqual(contract["supported_train_rows"], [200000, 300000]); self.assertEqual(contract["conditional_draws"], 4096)
            self.assertIs(evaluation.population_summary, parent_evaluation.population_summary); self.assertIs(report.readiness, parent_report.readiness)
            old = deepcopy(task["target"]); old["quote_count"] = 54
            with self.assertRaisesRegex(ValueError, "63-quote"): evaluation.evaluation_contract(old)
            classic = deepcopy(task["target"]); classic["pricing_controls"].pop("fourier_representation")
            with self.assertRaisesRegex(ValueError, "63-quote"): evaluation.evaluation_contract(classic)

    def test_both_sizes_tail_batches_and_only_declared_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            for rows, steps, tail in ((200000, 782, 64), (300000, 1172, 224)):
                _, run, _, _, _, _, record = fixture(Path(temporary)/str(rows), rows, 287); accounting = evaluation.training_accounting(run)
                self.assertEqual(accounting["primary_optimizer_updates"], steps*300); self.assertEqual(accounting["tail_batch_rows"], tail)
                self.assertEqual(evaluation.checkpoint_endpoint(Path(record["checkpoint"]["path"]).parent, record, run), "primary_final")
                with self.assertRaisesRegex(ValueError, "endpoints"):
                    evaluation.checkpoint_endpoint(Path(run["primary_checkpoint"]).parent/"epoch-000200", {"native_epoch": 200, "actual_optimizer_updates": steps*200}, run)

    def test_truth_pairing_with_new63_quote_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, run, control, *_ = fixture(Path(temporary)); inputs, _ = evaluation.paired_bank_inputs(run, task, control, {})
            self.assertEqual(inputs["population"][0].shape, (512, 63)); control["population"]["truth_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "truth pairing"): evaluation.paired_bank_inputs(run, task, control, {})

    def test_compact_full300k_report_keeps_unready_cases_and_no_old_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, _, _, protocol, primary, secondary, _ = fixture(root, 300000, 287)
            result = report.build(protocol, primary, root/"report.html", secondary_path=secondary); text = (root/"report.html").read_text()
            self.assertLess(result["report_bytes"], 1048576); self.assertEqual(result["inline_svg_count"], 9)
            for phrase in ("90 million", "same 300,000 noisy datasets", "Old54 MCMC chains cannot validate", "new63 reference missing in fixture"):
                self.assertIn(phrase, text)
            self.assertNotIn("Frozen eight-case", text); self.assertFalse(result["promotion"]); self.assertNotIn("<script", text)

    def test_secondary_required_and_all_three_cases_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, _, _, protocol, primary, _, record = fixture(root, best_epoch=287)
            with self.assertRaisesRegex(ValueError, "secondary required"): report.build(protocol, primary, root/"missing.html")
            del record["cases"]["market"]; evaluation.fixed.write_json(primary, record)
            with self.assertRaisesRegex(ValueError, "immutable"): report.build(protocol, primary, root/"omitted.html")

    def test_standalone_figures_show300k_new63_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, _, _, protocol, primary, _, _ = fixture(root, 300000)
            result = figures.build(protocol, primary, root/"figures")
            self.assertEqual(result["format"], "heston_pr87_grid_scientific_figures_v1"); self.assertEqual(len(result["exports"]), 5)
            self.assertIn("300k new63 primary", (root/"figures/primary-historical-recovery.svg").read_text()); self.assertFalse(result["new_neural_sampling"])


if __name__ == "__main__": unittest.main()
