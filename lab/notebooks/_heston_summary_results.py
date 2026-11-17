"""Read and validate the shipped Heston summary-dimension lesson records.

This leaf utility imports only standard Python and NumPy. It reads compact JSON
and verifies recovery-image hashes; it performs no pricing, training, model
loading, posterior sampling, network access or file writes.

``load_results(root)`` returns ``(all_runs, recovery_figures, record)`` for both
marimo lessons. The numerical experiment remains in its archived source checkout.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np


names = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")
bounds = {"initial_variance": (.01, .16), "kappa": (.1, 10.), "theta": (.01, .16),
              "vol_of_vol": (.1, 3.), "rho": (-.95, -.1), "c": (.5, 2.)}
dimensions = (64, 32, 16, 8, 7, 6, 5, 4, 2)
case_labels = {"synthetic-broad": "Broad-prior synthetic case",
                   "synthetic-market-like": "Market-like synthetic case", "market": "Observed SPX quotes"}


def load_results(root):
    """Validate the compact, read-only lesson data and rebuild the plotting view."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate result key: " + key)
            result[key] = value
        return result

    def hash_value(value):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return sha256(payload).hexdigest()

    def check_hash(value):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("a source SHA-256 identity is malformed")

    def vector(value, label, *, low=None, high=None):
        array = np.asarray(value, dtype=float)
        if array.shape != (6,) or not np.isfinite(array).all():
            raise ValueError("six finite values required for " + label)
        if low is not None and np.any(array < low) or high is not None and np.any(array > high):
            raise ValueError("out-of-range values for " + label)
        return value

    root = Path(root).resolve()
    record = json.loads((root / "results.json").read_text(), object_pairs_hook=unique_keys)
    protocol, identities = record["protocol"], record["identities"]
    budget = {"rows": 300000, "validation_rows": 100, "epochs": 300, "batch_size": 256,
              "optimizer_updates": 351600, "presentations": 90000000, "seed": 9011}
    expected_summary = {"input_features": 63, "widths": [128, 128], "activation": "silu", "output_activation": "linear"}
    expected_posterior = {"family": "FlowMatching", "widths": [128, 128, 128], "dropout": .1,
                          "target_transform": "raw_coordinates_native_inference_variable_standardization"}
    panel = protocol["population"]
    if (record.get("format") != "heston_summary_dimensions_v1" or record.get("scientific_acceptance") is not False
            or tuple(protocol["parameter_names"]) != names or tuple(protocol["summary_dimensions"]) != dimensions
            or protocol["prior_bounds"] != [list(bounds[name]) for name in names]
            or protocol["training"] != budget or protocol["summary"] != expected_summary
            or protocol["posterior"] != expected_posterior or tuple(protocol["conditional_cases"]) != tuple(case_labels)
            or panel["rows"] != 512 or panel["draws_per_case"] != 512):
        raise ValueError("the complete matched nine-size development study is required")
    vector(panel["prior_midpoint_rmse"], "prior midpoint RMSE", low=0)
    if panel["prior_width90"] != [.9 * (bounds[name][1] - bounds[name][0]) for name in names]:
        raise ValueError("prior interval widths differ")
    if not 0 < panel["rank_dkw_bound"] < 1:
        raise ValueError("invalid SBC screen boundary")
    for key, value in identities.items():
        for digest in value.values() if key == "reference_chains" else (value,):
            check_hash(digest)
    if set(identities["reference_chains"]) != set(case_labels):
        raise ValueError("three matched reference identities required")
    for component in ("runner", "inference"):
        check_hash(record["producer"][component]["sha256"])
    check_hash(record["archive"]["sha256"])
    check_hash(record["archive"]["manifest_sha256"])
    contract = hash_value({"protocol": protocol, "identities": identities})
    runs, figures, seen = [], {}, set()
    for fit in record["fits"]:
        dimension, training = fit["dimension"], fit["training"]
        if (dimension is not None and (type(dimension) is not int or dimension not in dimensions)) or dimension in seen:
            raise ValueError("each of the nine summary fits and direct baseline must occur once")
        seen.add(dimension)
        if fit["contract_sha256"] != contract:
            raise ValueError("fit and shared protocol/panel identities differ")
        if not np.isfinite(training["elapsed_seconds"]) or training["elapsed_seconds"] <= 0:
            raise ValueError("invalid recorded training time")
        for key, value in fit["source"].items():
            if key.endswith("sha256"):
                check_hash(value)
        history = None
        roles = {"primary_final"}
        if dimension is not None:
            history = {"epoch": list(range(1, 301)), **fit["history"]}
            if any(len(history[key]) != 300 or not np.isfinite(history[key]).all() for key in ("loss", "val_loss")):
                raise ValueError("one complete finite 300-epoch history is required per summary fit")
            payload = json.dumps(history, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if sha256(payload.encode()).hexdigest() != fit["source"]["compact_history_sha256"]:
                raise ValueError("learning-history checksum differs")
            if (training["best_epoch"] != int(np.argmin(history["val_loss"])) + 1
                    or training["best_validation_loss"] != min(history["val_loss"])):
                raise ValueError("validation checkpoint and history differ")
            roles.add("secondary_validation_minimum")
        if set(fit["endpoints"]) != roles:
            raise ValueError("final and distinct validation-minimum endpoint roles are required")
        figure = fit["recovery_figure"]
        relative = Path(figure["file"])
        image_path = (root / relative).resolve()
        if relative.is_absolute() or root not in image_path.parents:
            raise ValueError("recovery images must stay inside the lesson data directory")
        check_hash(figure["sha256"])
        if sha256(image_path.read_bytes()).hexdigest() != figure["sha256"]:
            raise ValueError("recovery image checksum differs")
        figures[dimension] = image_path
        for role, endpoint in fit["endpoints"].items():
            expected_epoch = 300 if role == "primary_final" else training["best_epoch"]
            if type(endpoint["epoch"]) is not int or endpoint["epoch"] != expected_epoch:
                raise ValueError("checkpoint role and epoch differ")
            for value in endpoint["source"].values():
                check_hash(value)
            if dimension is not None and "model_sha256" not in endpoint["source"]:
                raise ValueError("summary model identity required")
            if role == "primary_final" and endpoint["source"]["draws_sha256"] != figure["source_draws_sha256"]:
                raise ValueError("recovery image and primary draw identities differ")
            metrics = endpoint["population"]
            for key in ("rmse", "median_width90"):
                vector(metrics[key], key, low=0)
            for key in ("coverage90", "rank_ks"):
                vector(metrics[key], key, low=0, high=1)
            if (len(metrics["rank_passed"]) != 6 or any(type(v) is not bool for v in metrics["rank_passed"])
                    or metrics["rank_passed"] != [value <= panel["rank_dkw_bound"] for value in metrics["rank_ks"]]):
                raise ValueError("SBC screen flags and rank statistics differ")
            if set(endpoint["cases"]) != set(case_labels):
                raise ValueError("all matched conditional cases are required")
            cases = {}
            for case, values in endpoint["cases"].items():
                for key in ("mean_error_in_reference_sd", "width90_ratio"):
                    vector(values[key], key, low=0)
                cases[case] = {"reference_ready": True, "reference_chain_sha256": identities["reference_chains"][case],
                               "parameters": {name: {key: values[key][i] for key in values} for i, name in enumerate(names)}}
            run = {**training, "status": "complete", "smoke_test_only": False, "train_rows": budget["rows"],
                   "completed_epochs": budget["epochs"], "optimizer_updates": budget["optimizer_updates"],
                   "presentations": budget["presentations"], "seed": budget["seed"], "evaluated_epoch": endpoint["epoch"],
                   "source_hashes": endpoint["source"], "data_payload_sha256": identities["bank_sha256"]}
            evaluation = {"status": "complete", "scientific_acceptance": False, "endpoint": role, "epoch": endpoint["epoch"],
                "source_evaluation_sha256": endpoint["source"]["evaluation_sha256"],
                "parameters": {name: {key: metrics[key][i] for key in ("rmse", "coverage90", "median_width90")} for i, name in enumerate(names)},
                "population": {"parameter_names": names, "rows": 512, "draws_per_case": 512,
                    "truth_sha256": identities["truth_sha256"], "observations_sha256": identities["observations_sha256"],
                    "recovery": {key: panel[key] for key in ("prior_midpoint_rmse", "prior_width90")},
                    "rank_dkw": {"rank_ks": metrics["rank_ks"], "passed": metrics["rank_passed"], "dkw_bound": panel["rank_dkw_bound"]}},
                "cases": cases}
            label = "Direct" if dimension is None else f"MLP {dimension}"
            runs.append({"dimension": dimension, "endpoint": role, "label": label, "run": run, "evaluation": evaluation, "history": history})
    if seen != {None, *dimensions} or len(runs) != 19:
        raise ValueError("all nine fits, nineteen endpoints and ten recovery figures are required")
    return runs, figures, record
