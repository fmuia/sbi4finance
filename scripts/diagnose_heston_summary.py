"""Inspect a saved Heston GRU on validation examples 0..15 only.

This bounded representation diagnostic does not train, evaluate a posterior,
load a bank, or access reserved test examples. It reconstructs the pinned
Keras GRU recurrence and checks that reconstruction against the saved network.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("KERAS_BACKEND", "jax")

import bayesflow as bf
import jax
import keras
import numpy as np
from scipy.special import expit

from sbi4finance import heston, heston_bank, heston_inference


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standardized_summary_input(approximator, conditions):
    adapted = approximator.adapter(conditions, strict=False)
    return approximator.standardizer.maybe_standardize(
        keras.ops.convert_to_tensor(adapted["summary_variables"]),
        key="summary_variables", stage="inference",
    )


def diagnose(config, checkpoint_dir):
    heston_bank.validate_config(config)
    if config["splits"]["validation"] < 16:
        raise ValueError("this diagnostic requires at least 16 validation examples")
    saved = json.loads((checkpoint_dir / "task.json").read_text())
    if set(saved) != {"task", "adapter_id"} or saved["task"] != config["task"]:
        raise ValueError("saved checkpoint task does not match the supplied configuration")
    task, adapter_id = config["task"], saved["adapter_id"]
    # build_workflow validates the saved adapter ID before restoring its model.
    workflow = heston_inference.build_workflow(task, adapter_id=adapter_id)
    workflow.load_approximator(checkpoint_dir / "model.keras")
    approximator = workflow.approximator
    summary = approximator.summary_network
    expected_time_axis = 1 if adapter_id == heston_inference.TIME_ADAPTER_ID else None
    if summary.time_axis != expected_time_axis or len(summary.recurrent_layers) != 1:
        raise ValueError("expected one recurrent layer and the declared adapter time axis")

    # Generate only these validation inputs; never call load_bank.
    validation = heston_bank.generate_examples(config, "validation", range(16))
    conditions = heston_inference.conditioning_data(
        validation["observations"], validation["dt_years"], task, adapter_id=adapter_id,
    )
    x = standardized_summary_input(approximator, conditions)
    # Follow the pinned BayesFlow call path exactly for implicit or explicit time.
    observed, time_vector = summary._split_time_series(x)
    embedded = np.asarray(summary.time_embedding(observed, t=time_vector)).astype(np.float64)
    trend_index = observed.shape[-1]
    gru = summary.recurrent_layers[0]
    cell = gru.cell
    if not (cell.reset_after and cell.activation.__name__ == "tanh"
            and cell.recurrent_activation.__name__ == "sigmoid"
            and not gru.go_backwards and not gru.return_sequences):
        raise ValueError("manual reconstruction requires a forward reset-after tanh/sigmoid GRU")
    kernel, recurrent_kernel, bias = [np.asarray(v).astype(np.float64) for v in gru.weights]
    state = np.zeros((len(embedded), cell.units))
    records = []
    selected_times = {0, min(31, embedded.shape[1] - 1),
                      min(127, embedded.shape[1] - 1), embedded.shape[1] - 1}
    for t in range(embedded.shape[1]):
        # einsum avoids platform BLAS warnings seen with tiny float32 matmuls.
        input_values = np.einsum("bi,ij->bj", embedded[:, t], kernel) + bias[0]
        recurrent_values = np.einsum("bi,ij->bj", state, recurrent_kernel) + bias[1]
        xz, xr, xh = np.split(input_values, 3, axis=1)
        rz, rr, rh = np.split(recurrent_values, 3, axis=1)
        z_pre, r_pre = xz + rz, xr + rr
        z, r = expit(z_pre), expit(r_pre)
        h_pre = xh + r * rh
        state = z * state + (1 - z) * np.tanh(h_pre)
        if t in selected_times:
            record = {"time_index": t, "embedded_time_trend": float(embedded[0, t, trend_index])}
            for name, pre in (("update", z_pre), ("reset", r_pre), ("candidate", h_pre)):
                record[name] = {
                    "median_abs_preactivation": float(np.median(np.abs(pre))),
                    "fraction_abs_preactivation_above_5": float(np.mean(np.abs(pre) > 5)),
                    "max_abs_preactivation": float(np.max(np.abs(pre))),
                }
            records.append(record)
    actual = np.asarray(gru(keras.ops.convert_to_tensor(embedded.astype(np.float32))))
    manual_error = float(np.max(np.abs(actual - state)))
    if not np.isfinite(manual_error) or manual_error >= 2e-5:
        raise RuntimeError(f"manual GRU reconstruction differs from Keras: {manual_error}")

    base_summary = np.asarray(summary(x))
    # Reverse only the last min(8,T) returns. All supplied trailing moments stay
    # equal apart from summation roundoff; known observation times stay in order.
    reversed_length = min(8, task["n_observations"])
    permuted = validation["observations"].copy()
    permuted[:, -reversed_length:, 0] = permuted[:, -reversed_length:, 0][:, ::-1]
    permuted_conditions = heston_inference.conditioning_data(
        permuted, validation["dt_years"], task, adapter_id=adapter_id,
    )
    feature_error = float(np.max(np.abs(
        conditions["history_features"] - permuted_conditions["history_features"])))
    permuted_x = standardized_summary_input(approximator, permuted_conditions)
    summary_difference = np.linalg.norm(base_summary - np.asarray(summary(permuted_x)), axis=1)
    return {
        "purpose": "saved-checkpoint representation diagnosis; not posterior acceptance",
        "split": "validation", "example_ids": validation["example_id"].tolist(),
        "root_seed": config["root_seed"], "task": task, "adapter_id": adapter_id,
        "reserved_test_accessed": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "bayesflow": bf.__version__, "keras": keras.__version__, "jax": jax.__version__},
        "time_embedding_weights": {v.name: np.asarray(v).tolist() for v in summary.time_embedding.weights},
        "final_embedded_time_trend": float(embedded[0, -1, trend_index]),
        "manual_gru_hidden_state_max_abs_error": manual_error,
        "gate_diagnostics": records,
        "reversed_trailing_return_count": reversed_length,
        "reversal_moment_max_abs_difference": feature_error,
        "reversal_summary_l2_difference": summary_difference.tolist(),
        "summary_std_across_16_cases": base_summary.std(axis=0).tolist(),
        "limits": "Gate magnitudes and order sensitivity describe this saved representation. "
                  "They do not establish a cause of posterior error or replace conditional inference checks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="model.keras and task.json directory")
    parser.add_argument("--out", type=Path, required=True, help="new JSON report path; never overwritten")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new report path")
    report = diagnose(json.loads(args.config.read_text()), args.checkpoint)
    package_dir = Path(bf.__file__).parent
    sources = {
        "config.json": args.config,
        "checkpoint/model.keras": args.checkpoint / "model.keras",
        "checkpoint/task.json": args.checkpoint / "task.json",
        "scripts/diagnose_heston_summary.py": Path(__file__),
        "src/sbi4finance/heston_inference.py": Path(heston_inference.__file__),
        "src/sbi4finance/heston_bank.py": Path(heston_bank.__file__),
        "src/sbi4finance/heston.py": Path(heston.__file__),
        "bayesflow/networks/summary/recurrent/recurrent_network.py":
            package_dir / "networks/summary/recurrent/recurrent_network.py",
        "bayesflow/networks/helpers/embeddings/time2vec.py":
            package_dir / "networks/helpers/embeddings/time2vec.py",
    }
    report["source_sha256"] = {key: file_hash(path) for key, path in sources.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "adapter_id": report["adapter_id"],
                      "final_embedded_time_trend": report["final_embedded_time_trend"],
                      "final_step": report["gate_diagnostics"][-1]}, indent=2))


if __name__ == "__main__":
    main()
