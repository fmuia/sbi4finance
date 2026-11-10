"""Measure clean-bank spacing at fixed c=1 without pricing or model fitting.

Only train_clean and validation_clean NPZ members are decompressed. The first
512 already declared validation means and three fixed synthetic panel means
query the nested 20k/100k training cohorts in Euclidean P/halfspread units.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration.heston_spread import HestonSpreadPilot


VALIDATION_ROWS = 512
PREFIXES = (20000, 100000)
CASE_IDS = ("synthetic-broad", "synthetic-market-like", "synthetic-boundary", "market")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def values_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_clean_arrays(payload, train_rows, validation_rows, dimension, query_rows=VALIDATION_ROWS):
    """Access only named clean-price members; never list/decompress test arrays."""
    with np.load(payload, allow_pickle=False) as archive:
        train = np.asarray(archive["train_clean"], dtype=np.float64)
        validation = np.asarray(archive["validation_clean"], dtype=np.float64)
    if (train.shape != (train_rows, dimension) or validation.shape != (validation_rows, dimension)
            or not 1 <= query_rows <= validation_rows
            or not np.isfinite(train).all() or not np.isfinite(validation).all()):
        raise ValueError("clean price shapes, finiteness or query-row count differ")
    # An NPZ member is decompressed as a whole; only this preserved prefix is analyzed.
    return train, validation[:query_rows].copy()


def exact_neighbors(train, queries, halfspreads, prefixes=PREFIXES):
    train, queries, h = map(lambda value: np.asarray(value, dtype=np.float64), (train, queries, halfspreads))
    if (train.ndim != 2 or queries.ndim != 2 or train.shape[1] != queries.shape[1]
            or h.shape != (train.shape[1],) or not len(queries) or not len(train)
            or not np.isfinite(train).all() or not np.isfinite(queries).all()
            or not np.isfinite(h).all() or np.any(h <= 0)
            or not prefixes or list(prefixes) != sorted(set(prefixes))
            or any(type(n) is not int or not 1 <= n <= len(train) for n in prefixes)):
        raise ValueError("finite matching clean arrays, positive spreads and nested prefixes required")
    scaled_train, scaled_queries = train / h, queries / h
    if not np.isfinite(scaled_train).all() or not np.isfinite(scaled_queries).all():
        raise ValueError("whitened clean prices must remain finite")
    result, previous = {}, None
    for n in prefixes:
        tree = cKDTree(scaled_train[:n], leafsize=16, compact_nodes=True,
                       balanced_tree=True, copy_data=False)
        distances, indices = tree.query(scaled_queries, k=1, eps=0., p=2., workers=1)
        distances, indices = np.asarray(distances), np.asarray(indices)
        if (not np.isfinite(distances).all() or np.any(indices < 0) or np.any(indices >= n)
                or previous is not None and np.any(distances > previous + 1e-12 * np.maximum(1., previous))):
            raise ValueError("nearest-neighbor finiteness, indices or nested distance monotonicity failed")
        result[str(n)] = {"distance": distances, "nearest_training_index": indices}
        previous = distances
        del tree
    return result


def distribution(distances, dimension):
    values = np.asarray(distances)
    probabilities = [0., .05, .25, .5, .75, .95, 1.]
    return {"count": len(values), "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "quantiles": dict(zip(map(str, probabilities), np.quantile(values, probabilities).tolist())),
        "fractions_within_euclidean_radius": {
            "1": float(np.mean(values <= 1.)), "2": float(np.mean(values <= 2.)),
            "sqrt_dimension": float(np.mean(values <= np.sqrt(dimension)))},
        "median_contract_rms_distance": float(np.median(values) / np.sqrt(dimension))}


def panel_clean_queries(panel_path, design_path, pilot):
    panel = json.loads(Path(panel_path).read_text())
    if (panel["format"] != "heston_spread_reference_panel_v1"
            or [case["id"] for case in panel["cases"]] != list(CASE_IDS)
            or panel["design_sha256"] != digest(design_path)
            or panel["fixed_task"] != pilot.task_definition()
            or digest(panel["declaration_path"]) != panel["declaration_sha256"]):
        raise ValueError("panel case order, exact target/design or declaration binding differs")
    declaration = json.loads(Path(panel["declaration_path"]).read_text())
    for key in ("design_sha256", "fixed_task", "joint_task", "products", "seed", "truth_sources"):
        if panel[key] != declaration[key]:
            raise ValueError("panel declaration fields differ")
    clean = []
    for index, case in enumerate(panel["cases"][:3]):
        prices, y, errors = (np.asarray(case[key], dtype=float) for key in
                             ("clean_prices", "observations", "standard_errors"))
        expected_shape = pilot.records["mid"].shape
        if (case["kind"] != "synthetic" or case["generating_c"] != 1.
                or case["seed"] != panel["seed"] + index
                or prices.shape != expected_shape or y.shape != expected_shape or errors.shape != expected_shape
                or not np.isfinite(prices).all() or not np.isfinite(y).all()
                or values_digest(prices) != case["clean_prices_sha256"]
                or values_digest(y) != case["observations_sha256"]
                or not np.array_equal(case["physical_truth"], declaration["truths"][case["id"]])
                or not np.array_equal(errors, np.random.default_rng(case["seed"]).standard_normal(len(prices)))
                or not np.array_equal(y, prices + pilot.records["half_spread"] * errors)):
            raise ValueError("declared synthetic clean prices, truth or c=1 observations differ")
        clean.append(prices)
    return np.asarray(clean), panel


def run(design_path, bank_path, panel_path, binding_path, out):
    started = time.perf_counter()
    design_path, bank_path, panel_path, binding_path, out = map(
        Path, (design_path, bank_path, panel_path, binding_path, out))
    if out.exists():
        raise FileExistsError(out)
    design, binding = json.loads(design_path.read_text()), json.loads(binding_path.read_text())
    pilot = HestonSpreadPilot(design["records"], c=1., infer_c=False)
    record = json.loads((bank_path / "bank.json").read_text())
    identity, target = record["identity"], pilot.task_definition()
    original = identity["task"]
    controls, original_controls = target["pricing_controls"], original["pricing_controls"]
    payload = bank_path / record["payload"]
    if (binding["format"] != "heston_spread_engineering_probe_v1"
            or binding["design_sha256"] != digest(design_path) or binding["fixed_task"] != target
            or binding["bank_manifest_sha256"] != digest(bank_path / "bank.json")
            or record["format"] != "heston_pilot_campaign_clean_bank_v1"
            or original["design_sha256"] != digest(design_path)
            or identity["task_fingerprint"] != fingerprint(original)
            or original["parameter_names"][:5] != target["parameter_names"]
            or original["prior_bounds"][:5] != target["prior_bounds"]
            or original_controls["measurement_spread_multiplier"] != controls["c_min"]
            or original_controls["numerical_fraction"] != controls["numerical_fraction"]
            or original_controls["quadrature_ladder"] != controls["quadrature_ladder"]
            or payload.resolve().parent != bank_path.resolve()
            or record["payload_sha256"] != binding["bank_payload_sha256"]
            or digest(payload) != record["payload_sha256"]
            or identity["train_rows"] != 100000 or identity["validation_rows"] != 5000
            or target["quote_count"] != 54):
        raise ValueError("preserved design/bank/prior/numerical or engineering binding differs")
    train, validation = read_clean_arrays(payload, 100000, 5000, 54)
    synthetic, panel = panel_clean_queries(panel_path, design_path, pilot)
    queries = np.concatenate([validation, synthetic])
    neighbors = exact_neighbors(train, queries, pilot.records["half_spread"])
    cohorts = {}
    for key, values in neighbors.items():
        distances, indices = values["distance"], values["nearest_training_index"]
        cohorts[key] = {"validation": {"summary": distribution(distances[:VALIDATION_ROWS], 54),
                "distances": distances[:VALIDATION_ROWS].tolist(),
                "nearest_training_indices": indices[:VALIDATION_ROWS].tolist(),
                "distinct_nearest_training_rows": int(len(set(indices[:VALIDATION_ROWS].tolist())))},
            "synthetic_panel": {case_id: {"distance": float(distances[VALIDATION_ROWS+i]),
                "nearest_training_index": int(indices[VALIDATION_ROWS+i])}
                for i, case_id in enumerate(CASE_IDS[:3])}}
    d20, d100 = [neighbors[str(n)]["distance"][:VALIDATION_ROWS] for n in PREFIXES]
    ratios = np.divide(d100, d20, out=np.ones_like(d20), where=d20 > 0)
    result = {"format": "heston_spread_bank_resolution_audit_v1", "scientific_acceptance": False,
        "design_sha256": digest(design_path), "bank_manifest_sha256": digest(bank_path / "bank.json"),
        "bank_payload_sha256": record["payload_sha256"], "binding_record_sha256": digest(binding_path),
        "panel_sha256": digest(panel_path), "panel_declaration_sha256": panel["declaration_sha256"],
        "producer_source_sha256": digest(__file__),
        "geometry_validation_source_sha256": digest(sys.modules[HestonSpreadPilot.__module__].__file__),
        "inputs": {"design": str(design_path.resolve()), "bank": str(bank_path.resolve()),
            "panel": str(panel_path.resolve()), "binding_record": str(binding_path.resolve())},
        "protocol": {"c": 1., "scale": "frozen observed price half-spreads",
            "metric": "Euclidean norm of (clean_query - clean_training_surface) / half_spread",
            "query_validation_rows": list(range(VALIDATION_ROWS)), "synthetic_case_ids": list(CASE_IDS[:3]),
            "cohort_training_prefix_rows": list(PREFIXES), "nearest_training_index_base": 0,
            "algorithm": "scipy.spatial.cKDTree exact k=1 eps=0 p=2 workers=1; leafsize16,compact_nodes,balanced_tree",
            "ties": "SciPy exact-query index among equal-distance ties; no unique-index claim",
            "arrays_decompressed": ["train_clean", "validation_clean"],
            "validation_member_decompressed_rows": 5000, "validation_rows_analyzed": VALIDATION_ROWS,
            "reserved_test_arrays_opened": False, "bank_physical_parameter_arrays_opened": False,
            "model_pricing_evaluations": 0, "neural_fits": 0,
            "array_values_mutated": False,
            "measured_process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.**2 if sys.platform == "darwin" else 1024.)},
        "gaussian_noise_norm_context": {"dimension": 54, "sqrt_expected_squared_norm": float(np.sqrt(54)),
            "mean_norm": float(chi.mean(54)), "quantiles_05_50_95": chi.ppf([.05,.5,.95], 54).tolist(),
            "scope": "Norm of one 54-dimensional independent standard-Normal error vector; a descriptive reference, not a bank adequacy or identifiability bound."},
        "cohorts": cohorts,
        "paired_100k_over_20k_validation_distance_ratio": {
            "quantiles_05_50_95": np.quantile(ratios, [.05,.5,.95]).tolist(),
            "zero_20k_distances": int(np.sum(d20 == 0)),
            "fraction_strictly_improved": float(np.mean(d100 < d20)),
            "scope": "Same query means, nested physical training cohorts; hypothetical spacing comparison, no 100k fit performed by this audit."},
        "elapsed_wall_seconds": time.perf_counter() - started,
        "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("numpy", "scipy")}},
        "interpretation": [
            "Finite empirical training-prior spacing is different from the intended continuous physical prior.",
            "Noise augmentation varies observations around these centers; it does not create new physical parameter locations.",
            "A learned continuous conditional density can interpolate between training centers; success depends on inductive bias and optimization.",
            "Nearest clean-mean distances alone do not prove posterior unidentifiability, statistical impossibility, or inability of a neural estimator to learn.",
            "The 100k cohort is a data-spacing counterfactual, not evidence that increasing data or optimization will resolve current conditional failures.",
            "The market case has no known clean mean and is intentionally excluded; no extra scientific cases or reserved outcomes were inspected."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "bank", "panel", "binding-record", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.design, args.bank, args.panel, args.binding_record, args.out)
    print(json.dumps({"out": str(args.out), "elapsed_wall_seconds": result["elapsed_wall_seconds"],
        "median_validation_distances": {n: c["validation"]["summary"]["quantiles"]["0.5"]
                                        for n, c in result["cohorts"].items()},
        "reserved_test_arrays_opened": False, "model_pricing_evaluations": 0}))


if __name__ == "__main__":
    main()
