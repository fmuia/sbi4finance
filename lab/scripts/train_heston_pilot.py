"""Bounded BayesFlow check for one frozen European Heston observation design.

This development tool supports two explicit fixed layouts only: all records or
the first three expiries. It creates no tutorial assets and makes no generic
missing-data claim. A bank fixes clean prices and latent discrepancy scales;
independent quote noise and a common spread multiplier are augmented per batch.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def records_from_design(design):
    """Accept a records mapping, directly or under a design's records key."""
    records = design.get("records", design)
    required = ("forward", "strike", "discount", "maturity", "option_type", "mid", "half_spread")
    if not isinstance(records, dict) or not all(key in records for key in required):
        raise ValueError("design must contain a mapping of raw-contract record arrays")
    return records


def layout_mask(records, layout):
    time = np.asarray(records["maturity"], dtype=float)
    expiries = np.unique(time)
    if layout == "full":
        return np.ones(len(time), dtype=bool)
    if layout != "short" or len(expiries) < 4:
        raise ValueError("short layout requires at least four expiries and retains the first three")
    return time <= expiries[2]


def noise_batch(clean, parameters, half_spread, market_vega, step, seed, *, multiplier=None):
    """Paired full-layout noise, invariant to the later fixed-layout selection."""
    clean = np.asarray(clean, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([0x4850494C, seed, step]))
    count = len(clean)
    c = rng.uniform(.5, 2., size=count) if multiplier is None else np.full(count, multiplier)
    sd = np.sqrt((c[:, None] * half_spread)**2 + (parameters[:, -1, None] * market_vega)**2)
    return clean + rng.standard_normal(clean.shape) * sd, c


def condition_data(quotes, c, records, mask):
    """Explicit price and quote-SD inputs; all design coordinates stay frozen."""
    quotes = np.asarray(quotes, dtype=float)
    c = np.asarray(c, dtype=float)
    h = np.asarray(records["half_spread"], dtype=float)
    df = np.asarray(records["discount"], dtype=float) * np.asarray(records["forward"], dtype=float)
    if (quotes.ndim != 2 or quotes.shape[1] != len(h) or c.shape != (len(quotes),)
            or not np.isfinite(quotes).all() or not np.isfinite(c).all() or np.any(c < .5) or np.any(c > 2.)):
        raise ValueError("finite full-design quotes and supported spread multipliers [.5,2] required")
    return {"observations": quotes[:, mask] / h[mask],
            "log_noise": np.log(c[:, None] * h[mask] / df[mask])}


def training_data(quotes, c, parameters, records, mask, names):
    return {**condition_data(quotes, c, records, mask),
            **{name: parameters[:, i:i+1] for i, name in enumerate(names)}}


def build_workflow(names, bounds, updates):
    import bayesflow as bf
    import keras
    if bf.__version__ != "2.1.0":
        raise RuntimeError("the pilot requires the locked BayesFlow 2.1.0 environment")
    adapter = bf.Adapter()
    for name, (lo, hi) in zip(names, bounds, strict=True):
        adapter = adapter.constrain(name, lower=float(lo), upper=float(hi), inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(names, into="inference_variables")
               .concatenate(["observations", "log_noise"], into="inference_conditions"))
    optimizer = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-4, decay_steps=updates, alpha=.1))
    flow = bf.networks.CouplingFlow(transform="spline", subnet_kwargs={"widths": [128, 128]})
    workflow = bf.BasicWorkflow(adapter=adapter, inference_network=flow,
                               standardize=["inference_variables", "inference_conditions"], optimizer=optimizer)
    workflow.estimator.compile(optimizer=optimizer, steps_per_execution=1)
    return workflow


def sample(workflow, names, quotes, c, records, mask, draws, seed):
    values = workflow.sample(num_samples=draws, conditions=condition_data(quotes, c, records, mask),
                             sample_shape=(), seed=seed)
    result = np.concatenate([np.asarray(values[name]) for name in names], axis=-1)
    if result.shape != (len(quotes), draws, len(names)) or not np.isfinite(result).all():
        raise FloatingPointError("invalid posterior draw shape or values")
    return result


def bank_identity(design, pilot, train, validation, test, seed):
    from sbi4f_tutorial.heston_options import model
    lab = Path(__file__).resolve().parents[1]
    return {"format": "heston_pilot_clean_bank_v1", "design_sha256": digest(design),
            "names": list(pilot.names), "bounds": np.asarray(pilot.bounds).tolist(),
            "splits": {"train": train, "validation": validation, "test": test}, "seed": seed,
            "heston_pilot_sha256": digest(sys.modules[type(pilot).__module__].__file__),
            "heston_model_sha256": digest(model.__file__),
            "environment_sha256": {name: digest(lab/name) for name in ("pyproject.toml", "pixi.lock")},
            "trainer_sha256": digest(__file__),
            "observation_model": "independent Gaussian; sd=sqrt((c*halfspread)^2+(tau*frozen_market_vega)^2)",
            "spread_multiplier_training_prior": {"kind": "Uniform", "bounds": [.5, 2.]},
            "scope": "fixed design/vega/halfspreads only; price labels checked against half the observed halfspread"}


def load_or_generate_bank(path, identity, pilot):
    """Keep immutable source-bound payloads outside Git, never relabel old banks."""
    path = Path(path)
    if path.exists():
        record = json.loads((path / "bank.json").read_text())
        if record["identity"] != identity or digest(path / "bank.npz") != record["payload_sha256"]:
            raise ValueError("existing bank identity/payload differs; select a fresh bank path")
        with np.load(path / "bank.npz") as values:
            return {key: values[key].copy() for key in values.files}, record
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "pending.json", identity)
    started = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([0x4842414E, identity["seed"]]))
    bounds = np.asarray(pilot.bounds)
    count = sum(identity["splits"].values())
    theta = rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, len(bounds)))
    clean = []
    for i, row in enumerate(theta):
        clean.append(pilot.prices(row))
        if (i+1) % 1000 == 0:
            print(json.dumps({"bank_priced": i+1, "total": count, "seconds": time.perf_counter()-started}), flush=True)
    clean = np.asarray(clean)
    payload = {}
    offset = 0
    for split, size in identity["splits"].items():
        payload[split+"_parameters"] = theta[offset:offset+size]
        payload[split+"_clean"] = clean[offset:offset+size]
        offset += size
    np.savez_compressed(path / "bank.npz", **payload)
    record = {"identity": identity, "payload_sha256": digest(path / "bank.npz"),
              "pricing_seconds": time.perf_counter()-started, "raw_surface_count": count}
    write_json(path / "bank.json", record)
    (path / "pending.json").unlink()
    return payload, record


def calibration_summary(draws, truth, names):
    lo, hi = np.quantile(draws, [.05, .95], axis=1)
    ranks = (draws < truth[:, None, :]).sum(axis=1)
    report = {}
    for i, name in enumerate(names):
        covered = (truth[:, i] >= lo[:, i]) & (truth[:, i] <= hi[:, i])
        report[name] = {"coverage_90": float(covered.mean()), "cases": len(truth),
                        "posterior_mean_rmse": float(np.sqrt(np.mean((draws[:, :, i].mean(axis=1)-truth[:, i])**2))),
                        "rank_histogram_10bins": np.histogram(ranks[:, i], bins=np.linspace(0, draws.shape[1]+1, 11))[0].tolist()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=("full", "short"), default="full")
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9011)
    args = parser.parse_args()
    for key in ("train", "validation", "test", "updates", "batch_size", "draws"):
        if getattr(args, key) < 1:
            parser.error(key+" must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from sbi4finance.calibration.heston_pilot import HestonPilot
    from train_full_surface_budget import BatchPlan, update_guard
    design = json.loads(args.design.read_text())
    records = records_from_design(design)
    mask = layout_mask(records, args.layout)
    # Both layouts use the complete paired bank and complete noise vectors.
    # The strictest supported multiplier .5 sets the bank precision gate.
    pricing_pilot = HestonPilot(records, infer_discrepancy=True, spread_multiplier=.5)
    pilot = HestonPilot(records, infer_discrepancy=True)
    identity = bank_identity(args.design, pricing_pilot, args.train, args.validation, args.test, args.seed)
    payload, bank_record = load_or_generate_bank(args.bank, identity, pricing_pilot)
    import bayesflow as bf
    import keras
    keras.utils.set_random_seed(args.seed)
    half_spread = np.asarray(records["half_spread"], dtype=float)
    market_vega = np.asarray(pilot.market_vega, dtype=float)
    plan = BatchPlan(args.train, args.batch_size, args.updates, args.seed)
    workflow = build_workflow(list(pilot.names), pilot.bounds, args.updates)

    class Dataset(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)

        def __len__(self):
            return plan.updates

        def __getitem__(self, step):
            index = plan.indices(step)
            theta = payload["train_parameters"][index]
            quotes, c = noise_batch(payload["train_clean"][index], theta, half_spread, market_vega, step, args.seed)
            return workflow.adapter(training_data(quotes, c, theta, records, mask, pilot.names))

    val_quotes, val_c = noise_batch(payload["validation_clean"], payload["validation_parameters"],
                                   half_spread, market_vega, 0, args.seed+1)
    from bayesflow.datasets import OfflineDataset
    validation = OfflineDataset(training_data(val_quotes, val_c, payload["validation_parameters"], records, mask, pilot.names),
                                args.batch_size, workflow.adapter, shuffle=False)
    guard = update_guard(plan)
    started = time.perf_counter()
    history = workflow.estimator.fit(dataset=Dataset(), epochs=1, shuffle=False, verbose=0,
                                     validation_data=validation, callbacks=[guard])
    training_seconds = time.perf_counter()-started
    workflow.estimator.save(args.out / "model.keras")
    restored = build_workflow(list(pilot.names), pilot.bounds, args.updates)
    restored.load_estimator(args.out / "model.keras")
    before = sample(workflow, pilot.names, val_quotes[:2], val_c[:2], records, mask, 16, args.seed+11)
    after = sample(restored, pilot.names, val_quotes[:2], val_c[:2], records, mask, 16, args.seed+11)
    if not np.array_equal(before, after):
        raise RuntimeError("same-seed checkpoint reload changed posterior samples")
    test_quotes, test_c = noise_batch(payload["test_clean"], payload["test_parameters"],
                                     half_spread, market_vega, 0, args.seed+2, multiplier=1.)
    # Bounded sampling batches avoid a single large JAX allocation.
    test_draws = np.concatenate([sample(restored, pilot.names, test_quotes[i:i+16], test_c[i:i+16],
                                        records, mask, args.draws, args.seed+21+i)
                                 for i in range(0, args.test, 16)], axis=0)
    market = np.asarray(records["mid"], dtype=float)[None, :]
    market_draws = {str(c): sample(restored, pilot.names, market, np.array([c]), records, mask,
                                   args.draws, args.seed+31)[0] for c in (.5, 1., 2.)}
    np.savez_compressed(args.out / "posterior-draws.npz", test=test_draws,
                        test_truth=payload["test_parameters"], test_quotes=test_quotes,
                        **{"market_c"+key: value for key, value in market_draws.items()})
    report = {"format": "heston_pilot_neural_development_v1", "scientific_acceptance": False,
              "scope": "bounded fixed-design development fit; calibration and reference agreement must be assessed",
              "design_sha256": digest(args.design), "bank": bank_record, "layout": args.layout,
              "selected_indices": np.flatnonzero(mask).tolist(), "names": list(pilot.names),
              "bounds": np.asarray(pilot.bounds).tolist(), "architecture": {"family": "spline", "widths": [128, 128]},
              "seed": args.seed, "updates": guard.updates, "batch_size": args.batch_size,
              "endpoint_selection": "predeclared exact-update endpoint; validation not used for selection",
              "history": {key: [float(v) for v in values] for key, values in history.history.items()},
              "training_seconds": training_seconds, "same_seed_reload_exact": True,
              "checkpoint_sha256": digest(args.out / "model.keras"),
              "posterior_sha256": digest(args.out / "posterior-draws.npz"),
              "synthetic_calibration": calibration_summary(test_draws, payload["test_parameters"], pilot.names),
              "market_summary": {key: {name: np.quantile(draws[:, i], [.05, .5, .95]).tolist()
                                       for i, name in enumerate(pilot.names)} for key, draws in market_draws.items()},
              "versions": {"python": platform.python_version(), **{name: metadata.version(name)
                            for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}},
              "source_sha256": {"trainer": digest(__file__), "pilot": digest(sys.modules[type(pilot).__module__].__file__)}}
    write_json(args.out / "training.json", report)
    print(json.dumps({"out": str(args.out), "training_seconds": training_seconds,
                      "updates": guard.updates, "scientific_acceptance": False}), flush=True)


if __name__ == "__main__":
    main()
