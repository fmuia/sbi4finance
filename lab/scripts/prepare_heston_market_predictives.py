"""Export compact actual63 market prediction bands from frozen posterior draws.

No model loading, training or MCMC. Raw posterior parameters remain private.
Run from lab/ with --conditional-draws pointing to the preserved final300 file.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from sbi4finance.calibration.heston_grid import HestonGridPilot

GRID = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid"
REFERENCES = LAB / "evidence/pilot-results/2026-11-11/heston-actual63-notebook"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def predict(draws, bounds, pilot, *, seed, method_index, count=256, chains=None):
    """Keep paired physical/c rows; c support violations do not change price eligibility."""
    draws, bounds = np.asarray(draws), np.asarray(bounds)
    physical_ok = np.all(np.isfinite(draws[:, :5]) & (draws[:, :5] >= bounds[:5, 0])
                         & (draws[:, :5] <= bounds[:5, 1]), axis=1)
    usable = physical_ok & np.isfinite(draws[:, -1]) & (draws[:, -1] > 0)
    rng = np.random.default_rng([seed, method_index, 0])
    if chains is None:
        selected = rng.choice(np.flatnonzero(usable), count, replace=False)
    else:
        if len(draws) % chains or count % chains:
            raise ValueError("balanced chain selection requires divisible counts")
        per_chain = len(draws) // chains
        selected = np.concatenate([rng.choice(np.flatnonzero(usable[i * per_chain:(i + 1) * per_chain])
                                             + i * per_chain, count // chains, replace=False)
                                   for i in range(chains)])
    paired = draws[selected]
    clean = np.array([pilot.prices(row) for row in paired])
    noise = np.random.default_rng([seed, method_index, 1]).standard_normal(clean.shape)
    noisy = clean + paired[:, -1:] * pilot.records["half_spread"] * noise
    outside = (draws < bounds[:, 0]) | (draws > bounds[:, 1])
    return {"clean_q05_q50_q95_raw": np.quantile(clean, [.05, .5, .95], axis=0).tolist(),
            "noisy_q05_q50_q95_raw": np.quantile(noisy, [.05, .5, .95], axis=0).tolist(),
            "c_q05_q50_q95": np.quantile(draws[:, -1], [.05, .5, .95]).tolist(),
            "c_mean": float(draws[:, -1].mean()), "raw_draw_count": len(draws),
            "priced_draw_count": count, "outside_prior_count": int(np.any(outside, axis=1).sum()),
            "outside_prior_per_parameter": outside.sum(axis=0).tolist(),
            "outside_pricing_bounds_count": int((~physical_ok).sum()),
            "invalid_noise_scale_count": int((~(np.isfinite(draws[:, -1]) & (draws[:, -1] > 0))).sum()),
            "selected_c_outside_prior_count": int(np.sum((paired[:, -1] < bounds[-1, 0]) | (paired[:, -1] > bounds[-1, 1]))),
            "selection_sha256": sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest(),
            "pairing": "Each clean price and noise SD uses the same selected physical/c posterior row; no c projection, clipping, rejection or reshuffling.",
            "reference_chain_balanced": chains is not None}


def prepare(conditional_draws, output):
    evaluation_path = GRID / "recovery-bank-independent/300k-300/evaluation/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    if digest(conditional_draws) != evaluation["conditional_draws"]["sha256"]:
        raise ValueError("final300 conditional posterior payload differs")
    task_path = GRID / "cv-preparation/task.json"
    task = json.loads(task_path.read_text())
    if task["target"] != evaluation["task"]:
        raise ValueError("the actual63 task differs from the saved final300 evaluation")
    panel_path = GRID / "cv-preparation/conditional-panel.json"
    if digest(panel_path) != evaluation["conditional_panel"]["sha256"]:
        raise ValueError("conditional observations differ")
    panel = json.loads(panel_path.read_text())
    observed = np.asarray(next(c for c in panel["cases"] if c["id"] == "market")["observations"])
    if array_digest(observed) != evaluation["cases"]["market"]["observation_sha256"]:
        raise ValueError("market observations differ")
    pilot = HestonGridPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    pilot.validate_task(task["target"])
    reference_summary_path = REFERENCES / "reference-comparison.json"
    reference_summary = json.loads(reference_summary_path.read_text())
    reference_path = REFERENCES / "reference-display-draws.npz"
    if digest(reference_path) != reference_summary["display_cache"]["sha256"]:
        raise ValueError("regenerated PyMC display cache differs")
    if reference_summary["cases"]["market"]["observation_sha256"] != array_digest(observed):
        raise ValueError("reference and amortizer observations differ")
    with np.load(conditional_draws, allow_pickle=False) as cache:
        amortizer = cache["market"].copy()
    with np.load(reference_path, allow_pickle=False) as cache:
        chains = cache["market__parameters"].copy()
    reference = chains.reshape(-1, 6)
    expected = evaluation["cases"]["market"]
    outside = (amortizer < pilot.bounds[:, 0]) | (amortizer > pilot.bounds[:, 1])
    if (amortizer.shape != (4096, 6) or not np.isfinite(amortizer).all()
            or outside.sum(axis=0).tolist() != expected["support"]["outside_prior"]["per_dimension_count"]
            or not np.array_equal(np.quantile(amortizer, [.05, .5, .95], axis=0), expected["comparison"]["neural_quantiles_05_50_95"])):
        raise ValueError("raw neural support or quantiles differ from the saved evaluation")
    methods = {"pymc": predict(reference, pilot.bounds, pilot, seed=20261113, method_index=0, chains=len(chains)),
               "amortizer": predict(amortizer, pilot.bounds, pilot, seed=20261113, method_index=1)}
    for name, values in (("pymc", reference), ("amortizer", amortizer)):
        edges = np.linspace(values[:, -1].min(), values[:, -1].max(), 41)
        density, _ = np.histogram(values[:, -1], bins=edges, density=True)
        methods[name]["c_histogram"] = {"density": density.tolist(), "edges": edges.tolist()}
    record = {"format": "heston_actual63_market_predictives_v1", "checkpoint": "300k-300-final300",
              "seed": 20261113, "draws_per_method": 256,
              "parameter_names": evaluation["task"]["parameter_names"], "prior_bounds": pilot.bounds.tolist(),
              "quote_count": len(observed), "methods": methods,
              "sources": {"evaluation_file_sha256": digest(evaluation_path),
                          "evaluation_fingerprint": fingerprint(evaluation), "task_fingerprint": fingerprint(task["target"]),
                          "conditional_draws_sha256": digest(conditional_draws),
                          "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
                          "training_run_sha256": evaluation["training_run"]["sha256"],
                          "common_bank_record_sha256": evaluation["common_bank"]["sha256"],
                          "observation_sha256": array_digest(observed),
                          "half_spread_sha256": array_digest(pilot.records["half_spread"]),
                          "reference_cache_sha256": digest(reference_path),
                          "reference_summary_sha256": digest(reference_summary_path),
                          "reference_chain_sha256": reference_summary["cases"]["market"]["chain_payload_sha256"],
                          "generator_sha256": digest(__file__)},
              "protocol": {"seed": 20261113, "draws_priced_per_method": 256,
                           "quantiles": [.05, .5, .95], "units": "raw option prices, original actual63 contract ordering",
                           "selection": "Independent deterministic selection per method from rows with physical parameters inside pricing prior and positive finite raw c; PyMC selection is balanced across its four stored chains.",
                           "noise": "Independent standard Gaussian errors with SD equal to each selected row's raw c times the frozen half-spread vector; positive c outside its prior remains included.",
                           "scope": "Descriptive pointwise bands on viewed calibration quotes. No held-out coverage or acceptance claim. Clean bands contain parameter uncertainty; noisy bands also contain replicated quote noise."}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional-draws", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=LAB / "notebooks/data/heston_market_predictives")
    args = parser.parse_args()
    result = prepare(args.conditional_draws, args.output)
    print(json.dumps({name: {k: result["methods"][name][k] for k in ("raw_draw_count", "priced_draw_count", "outside_prior_count")}
                      for name in result["methods"]}))
