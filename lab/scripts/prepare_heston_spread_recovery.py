"""Freeze eight matched recovery observations without selecting on inference quality.

The sampler panel contains no generating parameters. Truths and the selection
audit are written separately and are consumed only by the recovery report.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_scale as population

fixed, baseline = population.fixed, population.baseline
RANKS = (32, 96, 160, 224, 288, 352, 416, 480)


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": fixed.digest(path)}


def selected_rows(truth):
    truth = np.asarray(truth, dtype=float)
    if truth.shape != (512, 6) or not np.isfinite(truth).all():
        raise ValueError("the unchanged 512-row six-parameter population is required")
    return np.argsort(truth[:, -1], kind="stable")[list(RANKS)]


def prepare(capacity_study, out):
    capacity_study, out = Path(capacity_study).resolve(), Path(out).resolve()
    study = read(capacity_study)
    if study["format"] != "heston_spread_scale_precision_recovery_study_v1" or out.exists():
        raise ValueError("unchanged capacity study and a new recovery output are required")
    parent = read(Path(study["parent_checkpoint"]["path"])/"checkpoint.json")
    if fixed.digest(Path(study["parent_checkpoint"]["path"])/"checkpoint.json") != study["parent_checkpoint"]["record"]["sha256"]:
        raise ValueError("original task checkpoint changed")
    task = parent["task"]
    if task["target"] != study["joint_task"]:
        raise ValueError("joint posterior task differs")
    bank = Path(study["bank"]["path"])
    record = read(bank/"bank.json")
    payload = bank/record["payload"]
    if (fixed.digest(bank/"bank.json") != study["bank"]["record"]["sha256"]
            or fixed.digest(payload) != study["bank"]["payload"]["sha256"]
            or payload.resolve().parent != bank.resolve()):
        raise ValueError("preserved population bank identity differs")
    with np.load(payload, allow_pickle=False) as archive:
        physical = archive["validation_parameters"][:512].copy()
        clean = archive["validation_clean"][:512].copy()
        ids = archive["validation_row_id"][:512].copy()
    if not np.array_equal(ids, np.array([f"validation:{i:08d}" for i in range(512)])):
        raise ValueError("validation population row identities differ")
    quotes, truth = population.population_observations(task, physical, clean)
    networks = []
    for batch in (128, 512):
        evaluation = capacity_study.parent/"evaluations"/f"batch-{batch}"/"pass-0500"/"evaluation.json"
        native = read(evaluation)
        recovery = evaluation.parent/"recovery.npz"
        if (native["status"] != "complete" or native["task"] != study["joint_task"]
                or native["population"]["observations_sha256"] != baseline.observations_digest(quotes)
                or native["population"]["truth_sha256"] != baseline.observations_digest(truth)
                or fixed.digest(recovery) != native["population"]["recovery_sha256"]):
            raise ValueError("frozen final neural population differs")
        with np.load(recovery, allow_pickle=False) as saved:
            if not np.array_equal(saved["truth"], truth) or not np.array_equal(saved["row_id"], ids):
                raise ValueError("saved neural recovery truth/row identities differ")
        networks.append({"batch_size": batch, "evaluation": binding(evaluation), "recovery": binding(recovery)})
    chosen = selected_rows(truth)
    original_panel = read(Path(study["original_study"]["path"]))["panel"]
    if fixed.digest(original_panel["path"]) != original_panel["sha256"]:
        raise ValueError("original products/reference panel changed")
    products = read(original_panel["path"])["products"]
    cases = [{"id": f"validation-{i:08d}", "row_id": str(ids[i]),
        "observations": quotes[i].tolist(),
        "observations_sha256": baseline.observations_digest(quotes[i])} for i in chosen]
    panel = {"format": "heston_spread_recovery_observation_panel_v1",
        "design_path": parent["design_path"], "design_sha256": parent["design_sha256"],
        "joint_task": study["joint_task"], "products": products, "cases": cases,
        "source": binding(__file__), "population_observations_sha256": baseline.observations_digest(quotes),
        "scope": "Only these quotes, geometry and prior enter the sampler; no generating parameters or neural starts."}
    selection = {"format": "heston_spread_recovery_truth_selection_v1",
        "source": binding(__file__), "capacity_study": binding(capacity_study),
        "selection": {"rule": "Stable-sort generating c; choose midpoint ranks of eight equal rank strata, independent of physical parameters, neural error or sampler results.",
            "population_rows": 512, "zero_based_c_ranks": list(RANKS),
            "zero_based_population_indices": chosen.tolist(), "replacement_of_failed_cases": False},
        "parameter_names": study["joint_task"]["parameter_names"],
        "prior_bounds": study["joint_task"]["prior_bounds"],
        "networks": networks, "bank": study["bank"],
        "population_truth_sha256": baseline.observations_digest(truth),
        "population_observations_sha256": baseline.observations_digest(quotes),
        "cases": [{"id": case["id"], "row_id": case["row_id"], "population_index": int(i),
            "c_rank": rank, "truth": truth[i].tolist(), "observations_sha256": case["observations_sha256"]}
            for case, i, rank in zip(cases, chosen, RANKS, strict=True)],
        "reserved_arrays_opened": False, "new_pricing_labels": False,
        "new_neural_sampling": False,
        "scope": "Report-only selected-case recovery illustration, not SBC or a population coverage estimate. The sampler never opens this file."}
    out.mkdir(parents=True)
    fixed.write_json(out/"panel.json", panel)
    selection["panel"] = binding(out/"panel.json")
    fixed.write_json(out/"truth-selection.json", selection)
    return {"panel": binding(out/"panel.json"), "truth_selection": binding(out/"truth-selection.json"),
        "selected_indices": chosen.tolist(), "generating_c": truth[chosen, -1].tolist(), "networks": networks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.capacity_study, args.out), indent=2))
