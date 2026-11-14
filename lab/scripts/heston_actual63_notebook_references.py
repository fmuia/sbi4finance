"""Regenerate the three actual63 PyMC references used by the development notebook.

The original reference chains (``chain_payload_path`` in each recorded
``run.json``) were never exported from the producer's machine. This driver
reruns the unchanged canonical worker,
``run_heston_pr87_grid_reference_recovery.run_reference``: same seed, sampler,
data-only initialization, 2k/2k then 6k/12k allocation, likelihood, pricer and
numerical/convergence gates. Only the panel's owner-specific absolute input
paths are relocated to this checkout; every other panel value and every bound
file hash is checked identical. Outputs are new identities. The original
evidence is read, never modified.

    prepare    relocate the panel and freeze a new protocol (no sampling)
    summarize  verify the new chains, compare them with the recorded
               quantiles and write a small thinned display cache
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
sys.path.insert(0, str(LAB))
from scripts import run_heston_pr87_grid_reference_recovery as grid_reference  # noqa: E402
from scripts import run_heston_spread_reference as shared  # noqa: E402

ORIGINAL = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid"
ORIGINAL_PANEL = ORIGINAL / "cv-preparation/conditional-panel.json"
ORIGINAL_PANEL_SHA256 = "824b765095b7549a65c2f120701c6eaa2a78a88bd34a40965f1972584651258b"
ORIGINAL_TRUTHS = ORIGINAL / "cv-preparation/conditional-truth-audit.json"
ORIGINAL_REFERENCES = ORIGINAL / "recovery-bank-independent/references"
PRODUCER_ROOT = "/sbi4finance-heston-spread-baseline-20261109/"
CASES = grid_reference.CASES
PATH_FIELDS = ("design", "source", "preparation_protocol")
DISPLAY_DRAWS_PER_CHAIN = 500
read = lambda path: json.loads(Path(path).read_text())


def shown(path):
    """Repository-relative when inside the checkout, otherwise absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def relocate(spec):
    """Map one producer-absolute binding onto this checkout; hashes must match."""
    if set(spec) != {"path", "sha256"} or PRODUCER_ROOT not in spec["path"]:
        raise ValueError("unexpected producer binding: " + str(spec))
    local = (REPO / spec["path"].split(PRODUCER_ROOT, 1)[1]).resolve()
    if shared.digest(local) != spec["sha256"]:
        raise ValueError("relocated input differs from the frozen identity: " + str(local))
    return {"path": str(local), "sha256": spec["sha256"]}


def prepare(out, payload_root, *, smoke_test_only=False):
    out, payload_root = Path(out).resolve(), Path(payload_root).resolve()
    if out.exists() or payload_root.exists():
        raise FileExistsError("new reference identities required")
    if shared.digest(ORIGINAL_PANEL) != ORIGINAL_PANEL_SHA256:
        raise ValueError("original conditional panel differs")
    original = read(ORIGINAL_PANEL)
    relocated = dict(original)
    changes = {}
    for field in PATH_FIELDS:
        relocated[field] = relocate(original[field])
        changes[field] = {"original": original[field]["path"], "relocated": relocated[field]["path"],
                          "sha256": original[field]["sha256"]}
    unchanged = {k: v for k, v in original.items() if k not in PATH_FIELDS}
    if {k: v for k, v in relocated.items() if k not in PATH_FIELDS} != unchanged:
        raise AssertionError("only path fields may change")
    out.mkdir(parents=True)
    panel_path = out / "reference-panel.json"
    shared.write_json(panel_path, relocated)
    protocol = grid_reference.prepare_protocol(panel_path, out / "references", payload_root,
                                               out / "reference-protocol.json", smoke_test_only=smoke_test_only)
    shared.write_json(out / "reference-relocation.json", {
        "format": "heston_actual63_reference_relocation_v1",
        "original_panel": {"path": shown(ORIGINAL_PANEL), "sha256": ORIGINAL_PANEL_SHA256},
        "relocated_panel": {"path": shown(panel_path), "sha256": shared.digest(panel_path)},
        "changed_fields": changes, "all_other_panel_values_identical": True,
        "protocol_sha256": shared.digest(out / "reference-protocol.json"),
        "settings": protocol["settings"], "smoke_test_only": smoke_test_only,
        "scope": "Re-execution of the unchanged canonical worker with new output identities; "
                 "original chains/evidence untouched; no acceptance."})
    return protocol


def _thin(values, per_chain):
    step = values.shape[1] // per_chain
    return values[:, ::step][:, :per_chain]


def summarize(out):
    out = Path(out).resolve()
    protocol = read(out / "reference-protocol.json")
    truths = {case["id"]: case for case in read(ORIGINAL_TRUTHS)["cases"]}
    names = protocol["jobs"][0]["names"]
    display, cases = {}, {}
    for spec in protocol["jobs"]:
        case = spec["case_id"]
        run = read(Path(spec["evidence_out"]) / "run.json")
        if run.get("status") != "completed" or run.get("reference_ready") is not True:
            raise ValueError(f"{case}: regenerated reference is not ready")
        chains_path = Path(run["chain_payload_path"])
        if shared.digest(chains_path) != run["chain_payload_sha256"]:
            raise ValueError(f"{case}: chain payload identity differs")
        with np.load(chains_path, allow_pickle=False) as archive:
            draws, products = archive["parameters"].copy(), archive["products"].copy()
        flat = draws.reshape(-1, len(names))
        original_run = read(ORIGINAL_REFERENCES / case / "run.json")
        original = read(ORIGINAL_REFERENCES / case / "posterior.json")
        rows = {}
        for j, name in enumerate(names):
            q = np.quantile(flat[:, j], [.05, .5, .95])
            q0 = np.asarray(original["parameter_quantiles_05_50_95"][name])
            sd = float(flat[:, j].std(ddof=1))
            rows[name] = {"regenerated_q05_q50_q95": q.tolist(), "regenerated_mean": float(flat[:, j].mean()),
                          "regenerated_sd": sd, "recorded_q05_q50_q95": q0.tolist(),
                          "recorded_mean": original["parameter_means"][name],
                          "mean_difference_in_regenerated_sd": (float(flat[:, j].mean()) - original["parameter_means"][name]) / sd,
                          "width90_ratio_regenerated_over_recorded": float((q[2] - q[0]) / (q0[2] - q0[0]))}
        truth = truths.get(case, {})  # the market case has no generating truth
        cases[case] = {
            "parameters": rows,
            "selected_attempt": run["selected_attempt"], "attempts": run["attempts"], "usage": run["usage"],
            "chain_payload_sha256": run["chain_payload_sha256"], "chain_shape": list(draws.shape),
            "recorded_chain_payload_sha256": original_run["chain_payload_sha256"],
            "recorded_selected_attempt": original_run["selected_attempt"],
            "observation_sha256": run["observation_sha256"],
            "physical_truth": truth.get("physical_truth"), "generating_c": truth.get("generating_c")}
        display[f"{case}__parameters"] = _thin(draws, DISPLAY_DRAWS_PER_CHAIN)
        display[f"{case}__products"] = _thin(products, DISPLAY_DRAWS_PER_CHAIN)
    display_path = out / "reference-display-draws.npz"
    np.savez_compressed(display_path, names=np.asarray(names), product_names=np.asarray(list(protocol["products"])),
                        **display)
    summary = {"format": "heston_actual63_reference_regeneration_v1", "names": names,
               "products": protocol["products"], "cases": cases,
               "display_cache": {"path": shown(display_path), "sha256": shared.digest(display_path),
                                 "draws_per_chain": DISPLAY_DRAWS_PER_CHAIN,
                                 "thinning": "every k-th retained draw per chain, k = retained // 500"},
               "protocol_sha256": shared.digest(out / "reference-protocol.json"),
               "scientific_acceptance": False,
               "scope": "Same-protocol re-execution on a different platform. Chains are not expected to be "
                        "bit-identical to the producer's; agreement is judged against recorded quantiles."}
    shared.write_json(out / "reference-comparison.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="role", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--payload-root", type=Path, required=True)
    make.add_argument("--smoke-test-only", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "prepare":
        result = prepare(args.out, args.payload_root, smoke_test_only=args.smoke_test_only)
        print(json.dumps({"jobs": [job["case_id"] for job in result["jobs"]], "settings": result["settings"]}))
    else:
        result = summarize(args.out)
        print(json.dumps({case: {name: round(row["mean_difference_in_regenerated_sd"], 3)
                                 for name, row in value["parameters"].items()}
                          for case, value in result["cases"].items()}, indent=1))
