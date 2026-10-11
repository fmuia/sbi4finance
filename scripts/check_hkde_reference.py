"""Build the frozen HKDE conditional references before neural interpretation."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import logsumexp

from sbi4finance import hkde_reference as ref, option_bank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fixed_cases(config, protocol):
    ids = protocol["conditional_audit_ids"]
    bank_cases = option_bank.generate_examples(config, "validation", ids)
    task = config["task"]
    bounds = np.asarray(task["prior_bounds"])
    stress = bounds[:, 0] + np.asarray(protocol["stress_prior_fractions"]) * np.diff(bounds)[:, 0]
    prices = ref.price_parameters(task, stress)
    observed = prices + np.random.default_rng(protocol["stress_quote_seed"]).normal(
        0, task["context"]["quote_noise_std"], prices.shape)
    return (np.concatenate([bank_cases["observations"], observed[..., None]]),
            np.concatenate([bank_cases["parameters"], stress]),
            np.concatenate([bank_cases["example_id"], [f"stress:{i}" for i in range(len(stress))]]))


def save_grid(path, grid, observations, truth, case_ids, weights):
    np.savez_compressed(path, axis0=grid["axes"][0], axis1=grid["axes"][1],
                        parameters=grid["parameters"], prices=grid["prices"],
                        target_prices=grid["target_prices"], quadrature_weights=grid["quadrature_weights"],
                        observations=observations, truth=truth, case_ids=case_ids, weights=weights)


def load_grid(path, task, targets):
    with np.load(path, allow_pickle=False) as saved:
        data = dict(saved)
    grid = {key: data[key] for key in ("parameters", "prices", "target_prices", "quadrature_weights")}
    grid.update(axes=(data["axis0"], data["axis1"]), shape=(len(data["axis0"]), len(data["axis1"])),
                parameter_names=task["parameter_names"], target_contracts=targets)
    return grid, data


def evidence(grid, observation, sigma):
    log_density = -0.5 * np.sum(((grid["prices"] - observation[:, 0]) / sigma) ** 2, axis=1)
    return float(logsumexp(np.log(grid["quadrature_weights"]) + log_density)
                 - grid["prices"].shape[1] * np.log(sigma * np.sqrt(2 * np.pi)))


def compare(coarse, fine, observations, sigma, probabilities):
    rows, summaries, all_weights = [], [], []
    for observation in observations:
        weights_a, weights_b = [ref.posterior(g, observation, sigma) for g in (coarse, fine)]
        a, b = [ref.summarize(g, w, probabilities) for g, w in ((coarse, weights_a), (fine, weights_b))]
        rows.append({
            "max_abs_mean_difference_in_ref_sd": np.max(np.abs(a["mean"] - b["mean"]) / b["std"]),
            "max_abs_relative_std_difference": np.max(np.abs(a["std"] / b["std"] - 1)),
            "max_abs_correlation_difference": abs(a["correlation"] - b["correlation"]),
            "max_abs_target_quantile_difference": np.max(np.abs(a["target_quantiles"] - b["target_quantiles"])),
            "max_abs_log_evidence_difference": abs(evidence(coarse, observation, sigma) - evidence(fine, observation, sigma)),
        })
        summaries.append({key: b[key] for key in ("mean", "std", "correlation", "marginal_quantiles", "target_mean", "target_std", "target_quantiles")})
        all_weights.append(weights_b)
    return rows, summaries, np.asarray(all_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/hkde_inference_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    config_path = Path(protocol["bank_config"])
    config = json.loads(config_path.read_text())
    task, settings = config["task"], protocol["reference"]
    # A reference must use exactly the bank's numerical label map.
    if (settings["price_nodes"] != task["numerics"]["refined_nodes"]
            or settings["price_upper_limit"] != task["numerics"]["refined_upper_limit"]):
        raise ValueError("reference price controls differ from the bank's labels")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observations, truth, case_ids = fixed_cases(config, protocol)
    bounds = np.asarray(task["prior_bounds"])
    corners = np.stack(np.meshgrid(*bounds, indexing="ij"), axis=-1).reshape(-1, 2)
    price_cases = np.vstack([corners, bounds.mean(axis=1)])
    contracts = task["context"]["contracts"] + protocol["targets"]
    ordinary = ref.price_parameters(task, price_cases, contracts)
    tighter = ref.price_parameters(task, price_cases, contracts,
                                  nodes=settings["check_price_nodes"], upper_limit=settings["check_price_upper_limit"])
    sigma = task["context"]["quote_noise_std"]
    price_difference = np.abs(tighter - ordinary)
    price_metric = float(price_difference.max() / sigma)
    price_passed = price_metric <= settings["gates"]["max_price_difference_over_quote_sigma"]
    grids, timings, comparisons = {}, {}, []
    sizes = [*settings["grid_points"], settings["fallback_grid_points"]]
    for size in sizes:
        before = time.perf_counter()
        grid = ref.build_grid(task, size, protocol["targets"])
        timings[str(size)] = time.perf_counter() - before
        grids[size] = grid
        print(f"Grid {size} built in {timings[str(size)]:.2f}s", flush=True)
        if len(grids) < 2:
            continue
        previous = list(grids)[-2]
        rows, summaries, weights = compare(grids[previous], grid, observations, sigma, protocol["evaluation"]["quantiles"])
        metrics = {key: max(row[key] for row in rows) for key in rows[0]}
        gates = {key: value <= settings["gates"][key] for key, value in metrics.items()}
        comparisons.append({"coarse": previous, "fine": size, "metrics": metrics, "gates": gates, "case_metrics": rows})
        print(json.dumps(jsonable(comparisons[-1]["metrics"])), flush=True)
        if all(gates.values()):
            break
    grid_path = args.out / "reference-grid.npz"
    save_grid(grid_path, grid, observations, truth, case_ids, weights)
    report = {
        "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q",
        "protocol_sha256": sha(args.protocol), "bank_config_sha256": sha(config_path),
        "task": task, "case_ids": case_ids, "truth": truth, "target_contracts": protocol["targets"],
        "pricing_refinement": {"parameters": price_cases, "absolute_differences": price_difference,
                               "max_price_difference_over_quote_sigma": price_metric, "passed": price_passed,
                               "controls": [[settings["price_nodes"], settings["price_upper_limit"]],
                                            [settings["check_price_nodes"], settings["check_price_upper_limit"]]]},
        "grid_comparisons": comparisons, "selected_grid_points": size, "case_summaries": summaries,
        "passed": bool(price_passed and all(gates.values())), "thresholds": settings["gates"],
        "reference_grid_sha256": sha(grid_path),
        "source_sha256": {"check_hkde_reference.py": sha(__file__), "hkde_reference.py": sha(ref.__file__),
                          **{name: sha(Path(option_bank.__file__).with_name(name)) for name in option_bank.source_files("hkde")}},
        "timing_seconds": {"grids": timings, "total": time.perf_counter() - started},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()},
        "limits": ["Numerical posterior for the declared Fourier/Gaussian experiment, not analytic truth.",
                   "Selected pricing corners/center do not prove uniform continuum accuracy.",
                   "Reserved test has not been evaluated; no neural posterior accepted by this report."],
    }
    (args.out / "reference.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    print(f"Reference passed: {report['passed']}", flush=True)
    if not report["passed"]:
        raise SystemExit("Reference gates unresolved at the bounded refinement limit.")


if __name__ == "__main__":
    main()
