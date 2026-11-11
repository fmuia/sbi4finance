"""Collect the compact data behind the figures of the option-application documentation.

The sources are evidence files of the Heston study (PR #85, branch
`feat/heston-spread-baseline-20261109`) and the regenerated reference posteriors of the
Heston notebook (`lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/`, branch
`claude/trusting-dijkstra-hmgvcm`). Pass each file's path; the script copies only the
fields the figures need into `data.json`, with the SHA-256 of every source.

    python extract_data.py \
        --design   .../revised-grid/cv-preparation/design.json \
        --evaluation .../revised-grid/recovery-bank-independent/300k-300/evaluation/evaluation.json \
        --recovery .../revised-grid/recovery-bank-independent/300k-300/evaluation/recovery.npz \
        --whitening-diagonal .../whitening-pilot/evaluations/diagonal/recovery.npz \
        --whitening-zca .../whitening-pilot/evaluations/zca/recovery.npz \
        --reference-comparison .../heston-actual63-notebook/reference-comparison.json \
        --reference-draws .../heston-actual63-notebook/reference-display-draws.npz \
        --notebook-commit <commit of the notebook branch> \
        --commit 1c276bb72c1a8d43d1055327934f359708ba1e3f
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMETERS = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"]
EVIDENCE = "lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/"
REPO_PATHS = {
    "design": EVIDENCE + "revised-grid/cv-preparation/design.json",
    "evaluation": EVIDENCE + "revised-grid/recovery-bank-independent/300k-300/evaluation/evaluation.json",
    "recovery": EVIDENCE + "revised-grid/recovery-bank-independent/300k-300/evaluation/recovery.npz",
    "whitening-diagonal": EVIDENCE + "noise-scale/capacity-study/whitening-pilot/evaluations/diagonal/recovery.npz",
    "whitening-zca": EVIDENCE + "noise-scale/capacity-study/whitening-pilot/evaluations/zca/recovery.npz",
}
NOTEBOOK_EVIDENCE = "lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/"
NOTEBOOK_PATHS = {
    "reference-comparison": NOTEBOOK_EVIDENCE + "reference-comparison.json",
    "reference-draws": NOTEBOOK_EVIDENCE + "reference-display-draws.npz",
}
DISPLAY_DRAWS = 1000  # reference draws kept per case for the figures (every second display draw)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(values, digits=7):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in (*REPO_PATHS, *NOTEBOOK_PATHS):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--notebook-commit", required=True)
    parser.add_argument("--out", type=Path, default=HERE / "data.json")
    args = parser.parse_args()

    design = json.loads(args.design.read_text())
    records = design["records"]
    quotes = design["selected_quotes"]
    evaluation = json.loads(args.evaluation.read_text())
    population = evaluation["population"]
    recovery = np.load(args.recovery)
    diagonal = np.load(args.whitening_diagonal)
    zca = np.load(args.whitening_zca)
    comparison_file = json.loads(args.reference_comparison.read_text())
    draws_file = np.load(args.reference_draws)

    references = {}
    for name, case in comparison_file["cases"].items():
        parameters = case["parameters"]
        accepted = case["attempts"][case["selected_attempt"]]
        stride = draws_file[f"{name}__parameters"].reshape(-1, 6).shape[0] // DISPLAY_DRAWS
        references[name] = {
            "mean": rounded([parameters[p]["regenerated_mean"] for p in PARAMETERS], 9),
            "sd": rounded([parameters[p]["regenerated_sd"] for p in PARAMETERS], 9),
            "quantiles_05_50_95": rounded([parameters[p]["regenerated_q05_q50_q95"] for p in PARAMETERS], 9),
            "truth": None if case["physical_truth"] is None else rounded([*case["physical_truth"], case["generating_c"]], 9),
            "accepted_run": {"tune": accepted["tune"], "draws": accepted["draws"],
                             "likelihood_evaluations": accepted["likelihood_evaluations"],
                             "cpu_seconds": round(accepted["cpu_seconds"], 1)},
            "first_run": {"tune": case["attempts"][0]["tune"], "draws": case["attempts"][0]["draws"],
                          "likelihood_evaluations": case["attempts"][0]["likelihood_evaluations"],
                          "cpu_seconds": round(case["attempts"][0]["cpu_seconds"], 1)},
            "draws": [[float(f"{v:.6g}") for v in row]
                      for row in draws_file[f"{name}__parameters"].reshape(-1, 6)[::stride]],
            "window_variance_draws": [[float(f"{v:.6g}") for v in row]
                                      for row in draws_file[f"{name}__products"].reshape(-1, 2)[::stride]],
        }

    cases = {}
    for name, case in evaluation["cases"].items():
        comparison = case["comparison"]
        cases[name] = {
            "neural_quantiles_05_50_95": rounded(comparison["neural_quantiles_05_50_95"]),
            "reference_quantiles_05_50_95": rounded(comparison["reference_quantiles_05_50_95"]),
            "neural_mean": rounded(comparison["neural_mean"]),
            "reference_mean": rounded(comparison["reference_mean"]),
            "reference_sd": rounded(comparison["reference_sd"], 9),
            "standardized_mean_error": rounded(comparison["standardized_mean_error"], 4),
            "width_ratio": rounded(comparison["width_ratio"], 4),
            "products": {
                "standardized_mean_error": rounded(comparison["derived_products"]["standardized_mean_error"], 4),
                "width_ratio": rounded(comparison["derived_products"]["width_ratio"], 4),
            },
            "outside_prior_fraction": case["support"]["outside_prior"]["sample_fraction"],
        }

    data = {
        "parameters": PARAMETERS,
        "prior_bounds": evaluation["task"]["prior_bounds"],
        "design": {
            "valuation_time_utc": design["valuation_time_utc"],
            "time_convention": design["time_convention"],
            "reference_volatility": design["reference_volatility"],
            "targets_z": design["targets_z"],
            "products": design["products"],
            "expiration": [q["expiration"] for q in quotes],
            "forward": rounded(records["forward"], 6),
            "strike": records["strike"],
            "discount": rounded(records["discount"], 9),
            "maturity": rounded(records["maturity"], 9),
            "option_type": records["option_type"],
            "mid": records["mid"],
            "half_spread": rounded(records["half_spread"], 6),
            "target_z": [q["target_z"] for q in quotes],
        },
        "heston_300k": {
            "training_rows": evaluation["training_rows"],
            "epochs": evaluation["native_epoch"],
            "optimizer_updates": evaluation["actual_optimizer_updates"],
            "population_rows": population["rows"],
            "draws_per_row": population["draws_per_case"],
            "coverage": {level: rounded(entry["coverage"], 6)
                         for level, entry in population["coverage"].items()},
            "coverage_wilson95": {level: rounded(entry["wilson"], 6)
                                  for level, entry in population["coverage"].items()},
            "mean_posterior_sd": rounded(population["mean_posterior_sd"], 9),
            "outside_prior_per_parameter": [count / population["support"]["samples"] for count in
                                            population["support"]["outside_prior"]["per_dimension_count"]],
            "sbc_ks": rounded(population["rank_dkw"]["rank_ks"], 6),
            "sbc_bound": population["rank_dkw"]["dkw_bound"],
            "sbc_passed": population["rank_dkw"]["passed"],
            "fractional_ranks": rounded(population["fractional_ranks"], 5),
            "truth_median_correlation": rounded(population["recovery"]["truth_posterior_median_correlation"], 6),
            "median_width90_over_prior": rounded(population["recovery"]["median_width90_over_prior"], 6),
            "outside_prior_fraction": population["support"]["outside_prior"]["sample_fraction"],
            "truth": rounded(recovery["truth"]),
            "quantiles_05_50_95": rounded(recovery["quantiles"]),
            "cases": cases,
        },
        "whitening": {
            "truth_c": rounded(diagonal["truth"][:, 5]),
            "diagonal_c_quantiles": rounded(diagonal["quantiles"][:, :, 5]),
            "zca_c_quantiles": rounded(zca["quantiles"][:, :, 5]),
            "condition_number": {"raw": 1.09e6, "per_feature": 2.20e6, "zca": 1.0},
        },
        "references_regenerated": references,
        "sources": {
            "repository": "fmuia/sbi4finance",
            "branch": "feat/heston-spread-baseline-20261109",
            "commit": args.commit,
            "files": {name: {"path": path, "sha256": sha256(getattr(args, name.replace("-", "_")))}
                      for name, path in REPO_PATHS.items()},
            "condition_numbers": "whitening-pilot/README.md of the same commit",
            "notebook_branch": "claude/trusting-dijkstra-hmgvcm",
            "notebook_commit": args.notebook_commit,
            "notebook_files": {name: {"path": path, "sha256": sha256(getattr(args, name.replace("-", "_")))}
                               for name, path in NOTEBOOK_PATHS.items()},
        },
    }
    args.out.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
