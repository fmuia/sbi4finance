#!/usr/bin/env python3
"""Curate aggregate-only evidence for the Heston notebook's Bergomi capstone.

Run from ``lab/`` with NumPy installed. ``--artifacts`` is the preserved private
noise-experiment directory; ``--frozen-prices`` is a private NPZ produced with
the original research checkout's ``Actual63Emulator.prices``. Its sibling JSON
manifest binds that canonical producer, emulator, posterior draws and selection.
This script never trains, reprices with Monte Carlo, or implements a financial
kernel. It validates the inputs, converts forward-price units exactly once,
and publishes only summaries. The raw inputs are intentionally not distributed.

Example (paths supplied by the operator)::

    python scripts/prepare_bergomi_capstone_results.py \
        --artifacts "$PRIVATE_EXPERIMENTS" \
        --frozen-prices "$PRIVATE_FROZEN_PRICES"

The private frozen NPZ has ``mcmc_forward`` and ``amortizer_forward`` (1024,63),
``mcmc_chain_indices`` (256,) and ``amortizer_indices`` (1024,). The exact
selection is 256 evenly spaced retained draws from EACH of four MCMC chains,
and 1024 evenly spaced raw neural draws. Physical coefficients and c remain
paired. Its manifest records source/payload SHA-256 values, never copied paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "bergomi_capstone_results_v1"
NAMES = ["hurst", "eta", "rho", "short_variance", "long_variance", "decay", "c"]
BOUNDS = [[.02, .35], [.5, 3.5], [-.99, -.2], [.01, .16], [.01, .16], [.25, 8.], [.5, 50.]]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_digest(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def require(condition, message: str):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label: str):
    require(np.array_equal(np.asarray(actual), np.asarray(expected)), f"{label} differs")


def quantiles(value):
    return np.quantile(value, [.05, .5, .95], axis=0).tolist()


def curate(artifacts: Path, frozen_path: Path, design_path: Path):
    control = artifacts / "exact-emulator-control-v1"
    reference = artifacts / "reference-gaussian-c50-v2"
    comparison = artifacts / "fit-comparison-20261113-v3"
    protocol_path = control / "protocol.json"
    protocol = read(protocol_path)
    task = protocol["task"]
    fixed = task["fixed_price_function"]
    run = read(reference / "run.json")
    diagnostics = read(reference / "diagnostics.json")
    evaluation = read(control / "evaluation/summary.json")
    fit = read(comparison / "summary.json")
    fit_protocol = read(comparison / "protocol.json")
    checkpoint_path = control / "checkpoints/epoch-000300/checkpoint.json"
    checkpoint = read(checkpoint_path)
    design = read(design_path)
    frozen_record_path = frozen_path.with_suffix(".json")
    frozen = read(frozen_record_path)
    chain_path = reference / f"attempt-{run['selected_attempt']}" / "chains.npz"
    neural_path = control / "evaluation/market-posterior.npz"

    require(task["parameter_names"] == NAMES and task["prior_bounds"] == BOUNDS,
            "exact-control seven-target priors required")
    require(task["price_function_kind"] == "deterministic_fixed_emulator",
            "MC-label and fixed-emulator amortizers must not be mixed")
    require(task["noise"] == {"family": "gaussian", "sd": "c*h", "independent_per_quote": True},
            "only independent Gaussian c-times-half-spread evidence belongs here")
    require(run["family"] == "gaussian" and run["bounds"] == BOUNDS and run["names"] == NAMES,
            "reference noise/priors differ")
    require(fixed == run["emulator"] and evaluation["task"] == task,
            "reference and amortizer must have the same frozen mean function")
    require(run["chain_payload_sha256"] == digest(chain_path), "reference payload changed")
    require(evaluation["source_checkpoint_sha256"] == digest(checkpoint_path), "evaluation checkpoint changed")
    require(checkpoint["protocol_sha256"] == digest(protocol_path), "training protocol changed")
    require(checkpoint["epoch"] == 300 and checkpoint["actual_optimizer_updates"] == 234600,
            "final primary checkpoint required")
    require(checkpoint["model_sha256"] == digest(checkpoint_path.parent / "model.keras"),
            "amortizer model changed")
    require(checkpoint["task_fingerprint"] == fingerprint(task), "control task identity changed")
    require(frozen["format"] == "bergomi_capstone_private_frozen_prices_v1"
            and frozen["payload_sha256"] == digest(frozen_path)
            and frozen["same_mean_function_for_both"] is True
            and frozen["new_financial_pricing_calls"] == 0,
            "validated private fixed-emulator price arrays required")
    for key, expected in {
        "emulator_record_sha256": fixed["record_sha256"],
        "emulator_pricer_sha256": fixed["pricer_sha256"],
        "emulator_task_fingerprint": fixed["task_fingerprint"],
        "reference_chain_payload_sha256": digest(chain_path),
        "neural_posterior_sha256": digest(neural_path),
    }.items():
        require(frozen[key] == expected, f"frozen prices: {key} changed")
    require(run["gradient_checks_passed"] and diagnostics["mcmc_diagnostics_passed"],
            "fixed-emulator MCMC diagnostics must pass")
    require(run["bergomi_numerical_qualification_passed"] is False,
            "do not silently promote emulator convergence to numerical Bergomi qualification")

    raw = task["design"]["raw_records"]
    require(raw == design["records"], "current host and preserved actual63 records differ")
    same(fit["observed_raw"], raw["mid"], "fit comparison observations")
    same(fit["half_spread_raw"], raw["half_spread"], "fit comparison half-spreads")
    require(fit["all_financial_record_equal"] is True, "Heston/Bergomi record mismatch")
    require("c in [.5,50]" in fit_protocol["noise_SD_convention"],
            "both fit-comparison references must use common wider c support")
    df = np.asarray(raw["discount"]) * np.asarray(raw["forward"])
    same(df, task["design"]["price_factor_DF"], "discount-times-forward conversion")
    observed, h = (np.asarray(raw[key], dtype=float) for key in ("mid", "half_spread"))
    same(observed / df, task["design"]["observations_forward"], "forward-normalized observations")
    same(h / df, task["design"]["half_spread_forward"], "forward-normalized half-spreads")
    same(run["observed_forward_prices"], observed / df, "MCMC observations")
    require(run["contract_order_sha256"] == fingerprint(task["design"]["contracts"]),
            "MCMC contract order changed")
    require(run["observations_sha256"] == fingerprint((observed / df).tolist()),
            "MCMC observed payload identity changed")
    with np.load(chain_path, allow_pickle=False) as source:
        chains = source["parameters"].copy()
        same(source["names"].tolist(), NAMES, "reference target order")
    with np.load(neural_path, allow_pickle=False) as source:
        neural = source["draws"].copy()[0]
        same(source["observed"], observed / df, "neural observations")
        same(source["half_spread"], h / df, "neural half-spreads")
    require(chains.shape == (4, 2000, 7) and neural.shape == (4096, 7), "unexpected raw draw budgets")
    with np.load(frozen_path, allow_pickle=False) as source:
        prices_forward = {name: source[f"{name}_forward"].copy() for name in ("mcmc", "amortizer")}
        m_indices = source["mcmc_chain_indices"].copy()
        n_indices = source["amortizer_indices"].copy()
    same(m_indices, np.linspace(0, 1999, 256, dtype=int), "balanced MCMC selection")
    same(n_indices, np.linspace(0, 4095, 1024, dtype=int), "neural selection")
    selected = {"mcmc": chains[:, m_indices, :].reshape(-1, 7), "amortizer": neural[n_indices]}
    for method, values in selected.items():
        require(array_digest(values) == frozen[f"{method}_selected_parameter_sha256"],
                f"{method} frozen prices must preserve original paired coefficients and c")
        require(prices_forward[method].shape == (1024, 63)
                and np.isfinite(prices_forward[method]).all(), "invalid frozen price matrix")
    raw_draws = {"mcmc": chains.reshape(-1, 7), "amortizer": neural}
    bounds = np.asarray(BOUNDS)
    for values in raw_draws.values():
        require(np.isfinite(values).all(), "nonfinite raw samples")
        require(not np.any((values < bounds[:, 0]) | (values > bounds[:, 1])),
                "support violations must be reported, never silently filtered for prediction")

    with np.load(comparison / "comparison-prices.npz", allow_pickle=False) as source:
        fit_prices = {name.lower(): source[f"gaussian_{name}_prices_raw"].copy()
                      for name in ("Heston", "Bergomi")}
    require(fit["raw_prices"]["sha256"] == digest(comparison / "comparison-prices.npz"),
            "fit price payload changed")
    same(prices_forward["mcmc"] * df, fit_prices["bergomi"], "fixed-emulator reference prices")
    fit_comparison = {}
    for method, prices in fit_prices.items():
        mean = prices.mean(axis=0)
        rms = float(np.sqrt(np.mean(((mean - observed) / h)**2)))
        source_summary = fit["families"]["gaussian"][method.capitalize()]
        same(mean, source_summary["mean_clean_price_raw"], f"{method} posterior average")
        require(np.isclose(rms, source_summary["rms_residual_half_spreads"], rtol=1e-14),
                "fit RMS changed")
        fit_comparison[method] = {
            "price_mean_raw": mean.tolist(), "price_q05_q50_q95_raw": quantiles(prices),
            "residual_mean_minus_observed_over_h": ((mean - observed) / h).tolist(),
            "rms_over_h": rms, "draw_count": len(prices),
            "c_q05_q50_q95": fit["c_posteriors"]["gaussian"][method.capitalize()],
        }
    fit_comparison["rms_reduction_fraction"] = 1 - fit_comparison["bergomi"]["rms_over_h"] / fit_comparison["heston"]["rms_over_h"]
    fit_comparison["scope"] = "Posterior-average clean-price fit on the same 63 calibration quotes; not a prior predictive or general model ranking."

    histogram_edges = {}
    for name in ("c", "hurst"):
        index = NAMES.index(name)
        low = min(values[:, index].min() for values in raw_draws.values())
        high = max(values[:, index].max() for values in raw_draws.values())
        pad = .02 * (high - low)
        histogram_edges[name] = np.linspace(max(bounds[index, 0], low - pad),
                                           min(bounds[index, 1], high + pad), 65)
    posteriors, predictives = {}, {}
    innovations = np.random.default_rng(20261114).standard_normal((1024, 63))
    for method, values in raw_draws.items():
        posteriors[method] = {
            "q05_q50_q95": quantiles(values), "mean": values.mean(axis=0).tolist(),
            "sd": values.std(axis=0).tolist(), "correlation": np.corrcoef(values.T).tolist(),
            "summaries_use_all_raw_draws": True, "raw_draw_count": len(values),
            "support_clipped": False, "out_of_prior_count": 0, "histograms": {},
        }
        for name, edges in histogram_edges.items():
            density, _ = np.histogram(values[:, NAMES.index(name)], bins=edges, density=True)
            posteriors[method]["histograms"][name] = {"edges": edges.tolist(), "density": density.tolist()}
        clean = prices_forward[method] * df
        # c*h remains the raw-price SD. No conditional c redraw and no MCSE term.
        noisy = clean + selected[method][:, -1, None] * h * innovations
        noisy_q = np.quantile(noisy, [.05, .5, .95], axis=0)
        predictives[method] = {
            "draw_count": 1024, "raw_draw_count": len(values), "support_exclusions": 0,
            "paired_parameter_and_c_draws": True, "conditional_c_redraw": False,
            "clean_q05_q50_q95_raw": quantiles(clean),
            "noisy_q05_q50_q95_raw": noisy_q.tolist(), "clean_mean_raw": clean.mean(axis=0).tolist(),
            "noisy_in_band_count": int(((observed >= noisy_q[0]) & (observed <= noisy_q[2])).sum()),
            "viewed_quote_count": 63,
        }
    geometry = {
        "observed_raw": raw["mid"], "half_spread_raw": raw["half_spread"],
        "maturity_years": raw["maturity"], "strike_raw": raw["strike"],
        "forward_raw": raw["forward"], "discount": raw["discount"],
        "option_type": raw["option_type"],
        "log_moneyness": np.log(np.asarray(raw["strike"]) / raw["forward"]).tolist(),
        "standardized_log_moneyness": [quote["z_fixed16"] for quote in design["selected_quotes"]],
        "expiry_index": np.repeat(np.arange(7), 9).tolist(), "expiry_labels": fit["expiries"],
    }
    requirements = diagnostics["parameters"]["requirements"]
    parameter_diag = list(diagnostics["parameters"]["parameters"].values())
    source_bindings = {
        "same_emulator_pricer": fixed["pricer_sha256"], "emulator_record": fixed["record_sha256"],
        "emulator_task": fixed["task_fingerprint"], "control_task": fingerprint(task),
        "control_protocol": digest(protocol_path), "control_checkpoint": digest(checkpoint_path),
        "control_model": checkpoint["model_sha256"], "reference_protocol": digest(reference / "protocol.json"),
        "reference_run": digest(reference / "run.json"), "reference_diagnostics": digest(reference / "diagnostics.json"),
        "reference_chains": digest(chain_path), "neural_market_draws": digest(neural_path),
        "neural_evaluation": digest(control / "evaluation/summary.json"),
        "fit_comparison_summary": digest(comparison / "summary.json"),
        "fit_comparison_protocol": digest(comparison / "protocol.json"),
        "fit_comparison_prices": digest(comparison / "comparison-prices.npz"),
        "public_host_design": digest(design_path), "contracts": fingerprint(task["design"]["contracts"]),
        "observations": fingerprint(raw["mid"]), "half_spreads": fingerprint(raw["half_spread"]),
        "geometry": fingerprint(geometry), "frozen_prices": digest(frozen_path),
        "frozen_price_manifest": digest(frozen_record_path),
        "canonical_price_producer": frozen["producer_source_sha256"],
        "aggregation_source": digest(Path(__file__)),
        "training_whitening": protocol["whitening_sha256"],
        "target_affine": protocol["target_affine_sha256"],
        "training_bank_manifest": protocol["bank"]["manifest_sha256"],
        "training_bank_payload": protocol["bank"]["payload_sha256"],
    }
    return {
        "format": FORMAT, "schema_version": 1, "status": "development",
        "observation_model": {"family": "gaussian", "formula": "Y_i=P_i(phi)+c*h_i*Z_i",
                              "c_bounds": [.5, 50.], "h_units": "raw price half-spread", "independent": True},
        "host_heston_c_bounds": [.5, 2.], "comparison_heston_c_bounds": [.5, 50.],
        "parameter_names": NAMES, "parameter_bounds": BOUNDS,
        "task": {"id": task["format"], "fingerprint": fingerprint(task),
                 "mean_function": "fixed_deterministic_price_emulator", "conditioning": "whitened_Y_over_h",
                 "target_names": NAMES},
        "geometry": geometry, "geometry_sha256": fingerprint(geometry),
        "fit_comparison": fit_comparison, "posteriors": posteriors, "predictives": predictives,
        "predictive_protocol": {"seed": 20261114, "shared_gaussian_innovations": True,
                                "quantile_levels": [.05, .5, .95], "raw_price_conversion": "P_raw=P_forward*D*F exactly once",
                                "scope": "Descriptive posterior predictive checks on viewed calibration quotes; not held-out coverage."},
        "training": {"rows": protocol["train_rows"], "epochs": protocol["epochs"],
                     "mean_function_sha256": fixed["pricer_sha256"],
                     "batch_size": protocol["batch_size"], "optimizer_updates": checkpoint["actual_optimizer_updates"],
                     "checkpoint_epoch": checkpoint["epoch"], "checkpoint_role": "primary_final", "seed": protocol["seed"],
                     "preprocessing_fit": "training rows only", "noise_refresh": "fixed offline noise realization"},
        "reference": {"fixed_emulator": True, "exact_bergomi": False, "sampler": "PyMC NUTS", "chains": 4,
                      "mean_function_sha256": fixed["pricer_sha256"],
                      "draws_per_chain": 2000, "balanced_draws_per_chain": 256,
                      "mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"],
                      "gradient_checks_passed": run["gradient_checks_passed"], "divergences": diagnostics["divergences"],
                      "max_r_hat": max(p["r_hat"] for p in parameter_diag),
                      "min_ess_bulk": min(p["ess_bulk"] for p in parameter_diag),
                      "min_ess_tail": min(p["ess_tail"] for p in parameter_diag), "requirements": requirements},
        "source_bindings": source_bindings, "numerical_qualification_passed": False, "scientific_acceptance": False,
        "scope": "Gaussian c-times-half-spread working model. The reference and exact-control amortizer use the same fixed deterministic Bergomi mean emulator and the same seven priors; this is not exact Bergomi inference.",
        "caveats": [
            "The common c support [.5,50] was informed by viewed market data, rather than predeclared for untouched market validation.",
            "The host Heston lesson uses c support [.5,2]; the Heston-versus-Bergomi fit panel explicitly uses separate references with common support [.5,50].",
            "The mean emulator approximates finite-grid Monte Carlo price labels. Monte Carlo, discretization and emulator uncertainty are excluded from these bands and remain numerically unqualified, especially near low c.",
            "Converged fixed-emulator MCMC establishes a reference for this working posterior, not empirical coverage or financial-model adequacy.",
            "Posterior summaries retain every raw draw without clipping. Predictives use a deterministic balanced 1024-draw subset with original physical/curve/c pairing.",
            "The displayed control is the final epoch-300 checkpoint trained on 200000 rows.",
            "The inherited 200000-row physical training subset was availability-selected. These market summaries do not establish prior-wide acceptance.",
            "A misspecified working model still has a posterior the amortizer should approximate. Neural disagreement alone does not establish why the network fails or that more data cannot help.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--frozen-prices", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/cv-preparation/design.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/pilot-results/2026-11-14/bergomi-capstone/results.json"))
    args = parser.parse_args()
    record = curate(args.artifacts, args.frozen_prices, args.design)
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    require(not any(token in text for token in ("/Users/", "ssh.", "oscar", "row_id", "bank.npz")),
            "public aggregates must not expose private paths, hosts or row-level payloads")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(text)
    temporary.replace(args.output)
    print(json.dumps({"format": FORMAT, "bytes": args.output.stat().st_size,
                      "sha256": digest(args.output),
                      "noisy_in_band_counts": {method: result["noisy_in_band_count"]
                                               for method, result in record["predictives"].items()}}))


if __name__ == "__main__":
    main()
