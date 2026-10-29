"""Exploratory adjacent-pair cross fitting of existing rough diagnostic summaries.

First use --freeze-method, then run without it. This script draws no paths,
calls no model and fits no coefficients. Plain report.json and report.json.gz
are interchangeable when their uncompressed bytes match the frozen hashes.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

METHOD = {
    "status": "EXPLORATORY: specified after inspection of the first diagnostic, frozen before this calculation",
    "anchors": [0, 4, 6],
    "batch_pairs": [[i, i + 1] for i in range(0, 16, 2)],
    "preidentified_contract_indices": [8, 49],
    "cross_fit": "For each adjacent pair i,j, use beta_j with raw/control means from i and beta_i with means from j; average the two prices. No coefficient refit.",
    "contrast": "same_sample_pair_mean - swapped_coefficient_pair_mean; equals -(Xbar_i-Xbar_j)@(beta_i-beta_j)/2, with parity cancelling",
    "uncertainty": "Sample SD of the eight independent paired contrasts divided by sqrt(8); heuristic descriptive t7 interval. No nominal within-batch SE is used.",
    "reconstruction_absolute_price_tolerance": 1e-12,
    "scope": "Diagnoses the same-sample coefficient-fitting effect. It cannot validate control centering, F-control L2/integrability, time-grid bias, or unbiased continuous-model prices. No verified pointwise/global coverage or formal acceptance.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_input(root, logical):
    path = root / logical
    compressed = path.with_suffix(path.suffix + ".gz")
    if path.exists():
        data = path.read_bytes()
        if compressed.exists() and gzip.decompress(compressed.read_bytes()) != data:
            raise ValueError("plain and gzip input bytes differ")
    else:
        path = compressed
        data = gzip.decompress(path.read_bytes())
    return json.loads(data), {"logical_path": logical, "uncompressed_sha256": digest(data)}, {
        "path": str(path), "stored_sha256": digest(path.read_bytes()), "uncompressed_sha256": digest(data)}


def inputs(root):
    logical = ["declaration.json"] + [f"groups/group-{i}/report.json" for i in range(4)]
    loaded = [read_input(root, name) for name in logical]
    declaration = loaded[0][0]
    for i, (report, _, _) in enumerate(loaded[1:]):
        if (report["group"] != declaration["groups"][i] or report["status"] != "completed"
                or any(report[k] != declaration[k] for k in ("declaration_sha256", "runtime", "source_sha256"))):
            raise ValueError("input group is incomplete or differs from its declaration")
    return declaration, [x[0] for x in loaded[1:]], [x[1] for x in loaded], [x[2] for x in loaded]


def summarize(values, sigma):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    half = student_t.ppf(.975, len(values) - 1) * se
    return {"pair_means_price_units": values.tolist(), "mean_price_units": mean.tolist(),
            "mean_over_quote_sigma": (mean / sigma).tolist(),
            "between_pair_se_price_units": se.tolist(), "between_pair_se_over_quote_sigma": (se / sigma).tolist(),
            "heuristic_t95_lower_over_quote_sigma": ((mean - half) / sigma).tolist(),
            "heuristic_t95_upper_over_quote_sigma": ((mean + half) / sigma).tolist()}


def analyze(declaration, reports):
    task = declaration["reference"]["task"]
    sigma = np.asarray(task["context"]["quote_noise_std"])
    contracts = task["context"]["contracts"]
    tolerance = METHOD["reconstruction_absolute_price_tolerance"]
    result = []
    for group, case in enumerate(METHOD["anchors"]):
        rows = sorted(reports[group]["runs"], key=lambda r: r["call"]["batch"])
        if [r["call"]["batch"] for r in rows] != list(range(16)) or any(r["status"] != "priced" for r in rows):
            raise ValueError("exact sixteen recorded batches required")
        maximum_reconstruction_error = 0.
        for row in rows:
            for block in row["blocks"]:
                q = block["diagnostics"]
                indices = block["contract_indices"]
                parity = np.array([1. - contracts[i]["strike"] if contracts[i]["option_type"] == "call" else 0.
                                   for i in indices])
                reconstructed = (np.asarray(q["raw_put_mean"]) - np.asarray(q["control_mean"])
                                 @ np.asarray(q["fitted_coefficients"]) + parity)
                error = float(np.max(np.abs(reconstructed - block["prices"])))
                maximum_reconstruction_error = max(maximum_reconstruction_error, error)
                if error > tolerance:
                    raise ValueError("original controlled-price reconstruction exceeds frozen tolerance")
        same, crossed, differences = [], [], []
        for i, j in METHOD["batch_pairs"]:
            self_pair = (np.asarray(rows[i]["controlled_prices"]) + rows[j]["controlled_prices"]) / 2
            cross_pair = np.empty(len(contracts))
            direct_contrast = np.empty(len(contracts))
            for left, right in zip(rows[i]["blocks"], rows[j]["blocks"], strict=True):
                indices = left["contract_indices"]
                if indices != right["contract_indices"]:
                    raise ValueError("paired block contract order differs")
                a, b = left["diagnostics"], right["diagnostics"]
                xa, xb = np.asarray(a["control_mean"]), np.asarray(b["control_mean"])
                ba, bb = np.asarray(a["fitted_coefficients"]), np.asarray(b["fitted_coefficients"])
                parity = np.array([1. - contracts[k]["strike"] if contracts[k]["option_type"] == "call" else 0.
                                   for k in indices])
                cross_pair[indices] = (np.asarray(a["raw_put_mean"]) - xa @ bb
                                       + np.asarray(b["raw_put_mean"]) - xb @ ba) / 2 + parity
                direct_contrast[indices] = -(xa - xb) @ (ba - bb) / 2
            difference = self_pair - cross_pair
            if np.max(np.abs(difference - direct_contrast)) > tolerance:
                raise ValueError("paired contrast does not reconstruct the frozen bilinear identity")
            same.append(self_pair); crossed.append(cross_pair); differences.append(difference)
        statistics = {"same_sample": summarize(same, sigma), "cross_fitted": summarize(crossed, sigma),
                      "same_minus_cross": summarize(differences, sigma)}
        examples = []
        for index in METHOD["preidentified_contract_indices"]:
            c = statistics["same_minus_cross"]
            examples.append({"contract_index": index, "contract": contracts[index],
                "same_mean_over_quote_sigma": statistics["same_sample"]["mean_over_quote_sigma"][index],
                "cross_mean_over_quote_sigma": statistics["cross_fitted"]["mean_over_quote_sigma"][index],
                "same_minus_cross_over_quote_sigma": c["mean_over_quote_sigma"][index],
                "between_pair_se_over_quote_sigma": c["between_pair_se_over_quote_sigma"][index],
                "heuristic_t95_interval_over_quote_sigma": [c["heuristic_t95_lower_over_quote_sigma"][index],
                                                             c["heuristic_t95_upper_over_quote_sigma"][index]]})
        result.append({"case_index": case, "independent_pairs": 8,
                       "maximum_reconstruction_error_price_units": maximum_reconstruction_error,
                       "all_63_contract_statistics": statistics, "preidentified_examples": examples})
    return result


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--freeze-method", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    declaration, reports, identities, actual = inputs(args.root)
    source_hash = digest(Path(__file__).read_bytes())
    if args.freeze_method:
        write_new(args.method, {"method": METHOD, "inputs": identities, "script_sha256": source_hash,
                               "frozen_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"frozen_method": str(args.method), "method_sha256": digest(args.method.read_bytes())}))
        return
    frozen = json.loads(args.method.read_text())
    if frozen["method"] != METHOD or frozen["inputs"] != identities or frozen["script_sha256"] != source_hash:
        raise ValueError("frozen method, script or input bytes changed")
    if args.out is None:
        parser.error("--out is required for analysis")
    results = analyze(declaration, reports)
    write_new(args.out, {"format": "saved_rough_cross_batch_exploratory_v1", "frozen_method": frozen,
                        "frozen_method_sha256": digest(args.method.read_bytes()), "actual_input_files": actual,
                        "reference_declaration_sha256": declaration["declaration_sha256"], "results": results,
                        "scientific_acceptance": "not_assessed"})
    print(json.dumps({"out": str(args.out), "anchors": len(results)}))


if __name__ == "__main__":
    main()
