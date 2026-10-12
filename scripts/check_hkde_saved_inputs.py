"""Read-only replay of inputs to the saved HKDE correction test.

Run with the repository's pinned Python. Replays saved random inputs and
reprices selected saved parameter rows; generates no new observation or
posterior proposal. Stdout is a JSON audit record, not an acceptance result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
root = args.repo.resolve()
sys.path.insert(0, str(root / "src"))

import numpy as np
from sbi4finance import option_bank, hkde_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


report_path = root / "docs/results/hkde-correction/test.json"
panel_path = root / "docs/results/hkde-correction/new-test-panels.npz"
grid_path = root / "docs/results/hkde-inference/reference-grid.npz"
report = json.loads(report_path.read_text())
with np.load(panel_path, allow_pickle=False) as saved:
    panels = dict(saved)
with np.load(grid_path, allow_pickle=False) as saved:
    grid = dict(saved)

config = report["panel_configuration"]
task = config["task"]
bounds = np.asarray(task["prior_bounds"])
digest = option_bank.task_fingerprint(config)
words = [int(digest[i:i+8], 16) for i in range(0, 64, 8)]
max_parameter_delta = max_observation_delta = 0.0
errors = []
for i in range(256):
    identity = [0x4F5054, config["root_seed"], *words,
                option_bank.SPLITS["test"], i]
    prior_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 0])))
    quote_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 2])))
    parameters = prior_rng.uniform(bounds[:, 0], bounds[:, 1])
    noise = quote_rng.standard_normal(9)
    observation = panels["reference_prices"][i] + task["context"]["quote_noise_std"] * noise
    max_parameter_delta = max(max_parameter_delta, float(np.max(np.abs(parameters-panels["parameters"][i]))))
    max_observation_delta = max(max_observation_delta, float(np.max(np.abs(observation-panels["observations"][i, :, 0]))))
    errors.append(noise)

truth_rows = [0, 17, 64, 128, 255]
grid_rows = [0, 128, 8320, 16512, 16640]
truth_price_delta = float(np.max(np.abs(
    hkde_reference.price_parameters(task, panels["parameters"][truth_rows])
    - panels["reference_prices"][truth_rows])))
grid_price_delta = float(np.max(np.abs(
    hkde_reference.price_parameters(task, grid["parameters"][grid_rows])
    - grid["prices"][grid_rows])))
expected_parameters = np.stack(np.meshgrid(grid["axis0"], grid["axis1"], indexing="ij"), axis=-1).reshape(-1, 2)
axis_weights = []
for axis in (grid["axis0"], grid["axis1"]):
    gaps = np.diff(axis)
    axis_weights.append(np.r_[gaps[0], gaps[:-1]+gaps[1:], gaps[-1]]/(2*(axis[-1]-axis[0])))
quadrature_delta = float(np.max(np.abs(np.outer(*axis_weights).ravel()-grid["quadrature_weights"])))

protocol_path = root / "configs/hkde_correction_boundary_protocol.json"
protocol = json.loads(protocol_path.read_text())
seeds, seed_count_rows = [], []
for split, name, cases, conditional_cases in [(0, "validation", 512, 20), (1, "test", 256, 16)]:
    phase_counts = {"0_conditional_proposal": conditional_cases,
                    "1_calibration_proposal": cases,
                    "2_calibration_resampling": cases,
                    "3_joint_rank_randomization": 1}
    for phase, count in enumerate(phase_counts.values()):
        for i in range(count):
            seeds.append(int(np.random.SeedSequence(
                [protocol["root_seed"], split, phase, i]).generate_state(1)[0]))
    seed_count_rows.append({"split": name, "split_index": split,
                            "phase_counts": phase_counts, "total": sum(phase_counts.values())})

source_checks = {}
for name, expected in report["source_sha256"].items():
    path = root / "scripts" / name
    if not path.exists():
        path = root / "src/sbi4finance" / name
    actual = sha(path)
    source_checks[str(path.relative_to(root))] = {
        "saved_sha256": expected, "current_sha256": actual, "match": expected == actual}
input_checks = {}
for name, expected in report["input_sha256"].items():
    actual = sha(root / name)
    input_checks[name] = {"saved_sha256": expected, "current_sha256": actual,
                          "match": expected == actual}

checks = {
    "task_equals_report_task": task == report["task"],
    "parameter_order_theta_jump_intensity": task["parameter_names"] == ["theta", "jump_intensity"],
    "saved_ids_exactly_test_0_through_255": bool(np.array_equal(
        panels["example_id"], np.array([f"test:{i}" for i in range(256)]))),
    "prior_replay_exact": max_parameter_delta == 0,
    "quote_replay_exact": max_observation_delta == 0,
    "selected_saved_truth_prices_exact": truth_price_delta == 0,
    "selected_saved_grid_prices_exact": grid_price_delta == 0,
    "grid_tensor_order_exact": bool(np.array_equal(grid["parameters"], expected_parameters)),
    "uniform_prior_quadrature_exact": quadrature_delta == 0,
    "case_seeds_unique": len(seeds) == len(set(seeds)),
    "recorded_source_hashes_match": all(v["match"] for v in source_checks.values()),
    "recorded_input_hashes_match": all(v["match"] for v in input_checks.values()),
}
errors = np.asarray(errors)
output = {
    "scope": "Independent saved-input consistency audit; no new acceptance decision",
    "scientific_acceptance_changed": False,
    "historical_test_passed": report["passed"],
    "new_model_panels": 0, "new_estimator_proposals": 0, "training_runs": 0,
    "audit_script_sha256": sha(__file__),
    "versions": {"python": platform.python_version(), "numpy": np.__version__},
    "artifact_sha256": {str(p.relative_to(root)): sha(p) for p in
                        (report_path, panel_path, grid_path, protocol_path)},
    "checks": checks,
    "consistency_checks_passed": all(checks.values()),
    "model_scope": {"measure": task["measure"], "parameter_names": task["parameter_names"],
        "prior": "independent uniforms", "prior_bounds": task["prior_bounds"],
        "quote_noise_std_currency": task["context"]["quote_noise_std"],
        "contract_order": task["context"]["contracts"],
        "pricing_controls": task["numerics"]},
    "replay": {"test_root_seed": config["root_seed"], "split_code": option_bank.SPLITS["test"],
        "cases": 256, "prior_purpose": 0, "quote_purpose": 2,
        "parameter_max_abs_delta": max_parameter_delta,
        "observation_max_abs_delta": max_observation_delta},
    "reprice": {"saved_truth_row_ids": truth_rows, "truth_price_max_abs_delta": truth_price_delta,
        "saved_grid_row_ids": grid_rows, "grid_price_max_abs_delta": grid_price_delta,
        "uniform_quadrature_max_abs_delta": quadrature_delta},
    "case_seed_audit": {"root": protocol["root_seed"], "namespace": "root,split,phase,index",
        "counts": seed_count_rows, "total": len(seeds), "distinct": len(set(seeds))},
    "cohort_noise_description": {"standardized_means_by_contract": errors.mean(axis=0).tolist(),
        "standardized_sd_by_contract": errors.std(axis=0, ddof=1).tolist(),
        "standardized_overall_mean": float(errors.mean()),
        "all_nine_means_negative": bool(np.all(errors.mean(axis=0) < 0)),
        "scope": "Post-hoc descriptive cohort property, not a selected test or causal attribution"},
    "source_checks": source_checks, "input_checks": input_checks,
    "limits": [
        "Repricing reuses the existing numerical engine; checks consistency, not independent continuum pricing accuracy.",
        "Distinct named pseudorandom streams do not prove mathematical independence.",
        "Raw calibration proposal clouds and SIR draws were not saved; this audit does not regenerate them.",
        "Consistency checks do not override the historical SBC failure or establish population calibration."]}
print(json.dumps(output, indent=2, allow_nan=False))
if not all(checks.values()):
    raise SystemExit(1)
