"""Controlled full-panel Heston training comparison with fresh nuisance draws.

The engineering target is fixed within a campaign: all 54 frozen quotes,
five Heston parameters with xi <= 3, and independent tau ~ Uniform(0,.02).
The clean bank stores only the five physical parameters and numerical prices.
Tau, spread multiplier and observation noise are resampled for every batch.
This tool neither selects a scientific market model nor accepts a release.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_heston_pilot import (build_workflow, digest, records_from_design, sample,
                                       training_data)
from scripts.train_full_surface_budget import BatchPlan, schedule_record, update_guard
from sbi4finance.calibration.heston_pilot import DEFAULT_BOUNDS, DEFAULT_LADDER, HestonPilot


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_task(design_path, *, kappa_max=10., xi_max=3.):
    """Distinct inference/numerical identity, independent of neural architecture."""
    design_path = Path(design_path)
    design = json.loads(design_path.read_text())
    records = records_from_design(design)
    if len(records["mid"]) != 54:
        raise ValueError("this campaign requires the fixed complete 54-quote design")
    if kappa_max not in (10., 20.) or xi_max != 3.:
        raise ValueError("the campaign supports only kappa ceiling 10 or 20 and xi ceiling 3")
    bounds = np.asarray(DEFAULT_BOUNDS, dtype=float).copy()
    bounds[1, 1], bounds[3, 1] = kappa_max, xi_max
    pilot = HestonPilot(records, bounds=bounds, infer_discrepancy=True,
                        spread_multiplier=.5, quadrature_ladder=(*DEFAULT_LADDER, (16384, 9600.)))
    task = {"format": "heston_pilot_full_campaign_task_v1", "model": "heston", "measure": "Q",
            "parameter_names": pilot.names, "prior_bounds": pilot.bounds.tolist(),
            "prior": "independent Uniform in physical coordinates",
            "design_sha256": digest(design_path), "quote_count": 54, "layout": "full",
            "c_training_bounds": [.5, 2.], "tau_training_bounds": [0., .02],
            "observation_model": {"distribution": "independent Gaussian",
                "mean": "deterministic refined Heston price",
                "variance": "(c * observed halfspread)^2 + (tau * frozen observed market vega)^2",
                "vega_units": "quote price per unit decimal implied volatility",
                "conditioning": "fixed quote design, carry, market vegas and spread metadata"},
            "pricing_controls": {"quadrature_ladder": [list(rung) for rung in pilot.quadrature_ladder],
                "numerical_fraction": pilot.numerical_fraction, "measurement_spread_multiplier": .5,
                "precision_scope": "gate relative to smallest supported measurement SD; never relaxed by tau"},
            "scope": "fixed engineering benchmark; no market-model or release acceptance"}
    return task, records, pilot


def physical_rows(bounds, count, seed, split):
    """Separate split streams; increasing count preserves the exact row prefix."""
    rng = np.random.default_rng(np.random.SeedSequence([0x43414D50, seed, split]))
    bounds = np.asarray(bounds)[:5]
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, 5))


def bank_identity(task, bank_size, validation_rows, test_rows, seed):
    from sbi4f_tutorial.heston_options import model
    lab = Path(__file__).resolve().parents[1]
    return {"task_fingerprint": fingerprint(task), "task": task,
            "train_rows": bank_size, "validation_rows": validation_rows, "test_rows": test_rows, "seed": seed,
            "producer_source_sha256": {"campaign_trainer": digest(__file__),
                 "heston_pilot": digest(sys.modules[HestonPilot.__module__].__file__),
                 "heston_model": digest(model.__file__)},
            "environment_sha256": {name: digest(lab / name) for name in ("pyproject.toml", "pixi.lock")},
            "random_protocol": "PCG64 SeedSequence([0x43414D50, seed, split]); split0 train, split1 validation, split2 reserved test; prefix-stable physical Uniform draws",
            "stored_latents": task["parameter_names"][:5],
            "excluded_latents": "tau does not affect clean prices and is sampled fresh during training"}


def load_or_generate_bank(directory, identity, pilot):
    """Immutable complete bank; numerical failures preserve pending state."""
    directory = Path(directory)
    if directory.exists():
        record = json.loads((directory / "bank.json").read_text())
        if record["identity"] != identity or digest(directory / "bank.npz") != record["payload_sha256"]:
            raise ValueError("bank target, producer or payload differs; choose a fresh bank path")
        with np.load(directory / "bank.npz", allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files
                       if key.startswith(("train_", "validation_"))}
        return payload, record
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "pending.json", identity)
    started = time.perf_counter()
    payload = {}
    for split, count, namespace in (("train", identity["train_rows"], 0),
                                     ("validation", identity["validation_rows"], 1),
                                     ("test", identity["test_rows"], 2)):
        theta = physical_rows(pilot.bounds, count, identity["seed"], namespace)
        clean = np.empty((count, len(pilot.records["mid"])))
        for index, row in enumerate(theta):
            clean[index] = pilot.prices(np.r_[row, .01])
            if (index + 1) % 1000 == 0:
                print(json.dumps({"split": split, "priced": index + 1, "total": count,
                                  "seconds": time.perf_counter() - started}), flush=True)
        payload[split + "_parameters"] = theta
        payload[split + "_clean"] = clean
        payload[split + "_row_id"] = np.array([f"{split}:{i:08d}" for i in range(count)])
    np.savez_compressed(directory / "bank.npz", **payload)
    record = {"format": "heston_pilot_campaign_clean_bank_v1", "identity": identity,
              "payload": "bank.npz", "payload_sha256": digest(directory / "bank.npz"),
              "pricing_seconds": time.perf_counter() - started,
              "split_shapes": {key: list(value.shape) for key, value in payload.items()}}
    write_json(directory / "bank.json", record)
    (directory / "pending.json").unlink()
    return {key: value for key, value in payload.items()
            if key.startswith(("train_", "validation_"))}, record


def paired_subset(payload, rows):
    """The small bank arm is an exact prefix of the same large clean bank."""
    if type(rows) is not int or not 1 <= rows <= len(payload["train_parameters"]):
        raise ValueError("train_rows must select a positive prefix of the stored bank")
    return {key.removeprefix("train_"): value[:rows] for key, value in payload.items()
            if key.startswith("train_")}


def augmented_batch(clean, physical_parameters, halfspread, market_vega, step, seed,
                    *, namespace=0, multiplier=None):
    """Fresh tau/c/noise, deterministic on repeated probes of the same step."""
    clean, physical = np.asarray(clean), np.asarray(physical_parameters)
    h, v = np.asarray(halfspread), np.asarray(market_vega)
    if (clean.ndim != 2 or physical.shape != (len(clean), 5)
            or h.shape != (clean.shape[1],) or v.shape != h.shape):
        raise ValueError("matching clean prices, five physical latents and quote scales required")
    rng = np.random.default_rng(np.random.SeedSequence([0x4155474D, seed, namespace, step]))
    tau = rng.uniform(0., .02, size=len(clean))
    c = rng.uniform(.5, 2., size=len(clean)) if multiplier is None else np.full(len(clean), multiplier)
    if not np.isfinite(c).all() or np.any(c < .5) or np.any(c > 2.):
        raise ValueError("spread multiplier must lie in [.5,2]")
    parameters = np.column_stack((physical, tau))
    sd = np.hypot(c[:, None] * h, tau[:, None] * v)
    quotes = clean + rng.standard_normal(clean.shape) * sd
    return quotes, c, parameters


def _checkpoint_record(run, directory, updates, probe_metrics, elapsed):
    return {"format": "heston_pilot_campaign_checkpoint_v1", "scientific_acceptance": False,
            "task": run["task"], "task_fingerprint": run["task_fingerprint"],
            "design_sha256": run["design_sha256"], "names": run["names"], "bounds": run["bounds"],
            "architecture": run["architecture"], "bank_path": run["bank_path"],
            "bank_record_sha256": run["bank_record_sha256"], "bank_payload_sha256": run["bank_payload_sha256"],
            "train_rows": run["train_rows"], "seed": run["seed"],
            "actual_optimizer_updates": updates, "planned_updates": run["planned_updates"],
            "schedule_updates": run["schedule_updates"],
            "model_sha256": digest(directory / "model.keras"), "curve_probe_metrics": probe_metrics,
            "seconds_since_fit_started": elapsed, "source_sha256": run["source_sha256"],
            "versions": run["versions"], "checkpoint_selection": "predeclared update endpoint; no validation selection"}


def load_checkpoint(directory):
    """Load the public BayesFlow workflow after strict task/file binding checks."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    directory = Path(directory)
    record = json.loads((directory / "checkpoint.json").read_text())
    task = record["task"]
    if (record["format"] != "heston_pilot_campaign_checkpoint_v1"
            or fingerprint(task) != record["task_fingerprint"]
            or task["design_sha256"] != record["design_sha256"]
            or task["parameter_names"] != record["names"] or task["prior_bounds"] != record["bounds"]
            or digest(directory / "model.keras") != record["model_sha256"]
            or not 0 < record["actual_optimizer_updates"] <= record["planned_updates"]):
        raise ValueError("checkpoint task, dimensions, update counts or model hash mismatch")
    workflow = build_workflow(record["names"], np.asarray(record["bounds"]), record["schedule_updates"])
    workflow.load_estimator(directory / "model.keras")
    if int(workflow.estimator.optimizer.iterations.numpy()) != record["actual_optimizer_updates"]:
        raise ValueError("checkpoint optimizer iterations differ from recorded endpoint")
    return workflow, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, default=100000)
    parser.add_argument("--train-rows", type=int, default=20000)
    parser.add_argument("--validation-rows", type=int, default=5000)
    parser.add_argument("--test-rows", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--schedule-updates", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--probe-rows", type=int, default=512)
    parser.add_argument("--diagnostic-draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=9011)
    parser.add_argument("--kappa-max", type=float, choices=(10., 20.), default=10.)
    parser.add_argument("--xi-max", type=float, choices=(3.,), default=3.)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    for key in ("bank_size", "train_rows", "validation_rows", "test_rows", "updates", "schedule_updates", "batch_size", "log_every", "probe_rows", "diagnostic_draws"):
        if getattr(args, key) < 1:
            parser.error(key + " must be positive")
    if args.train_rows > args.bank_size or args.seed < 0:
        parser.error("train_rows must not exceed bank_size, and seed must be nonnegative")
    if args.schedule_updates < args.updates:
        parser.error("schedule_updates must cover the full requested update trajectory")
    task, records, pilot = make_task(args.design, kappa_max=args.kappa_max, xi_max=args.xi_max)
    identity = bank_identity(task, args.bank_size, args.validation_rows, args.test_rows, args.seed)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    run = {"format": "heston_pilot_full_training_campaign_v1", "scientific_acceptance": False,
           "task": task, "task_fingerprint": fingerprint(task), "names": pilot.names, "bounds": pilot.bounds.tolist(),
           "design_sha256": digest(args.design), "design_path": str(args.design.resolve()),
           "train_rows": args.train_rows, "planned_updates": args.updates,
           "schedule_updates": args.schedule_updates, "seed": args.seed,
           "batch_size": args.batch_size, "bank_path": str(args.bank.resolve()),
           "architecture": {"family": "spline", "widths": [128, 128]},
           "augmentation": {"tau": "fresh Uniform(0,.02) per row and step", "c": "fresh Uniform(.5,2) per row and step",
                            "noise": "fresh independent standard normals per quote, row and step",
                            "random_protocol": "PCG64 SeedSequence([0x4155474D, seed, namespace, step]); namespace0 training,1 validation curve,2 training curve"},
           "source_sha256": {"campaign_trainer": digest(__file__),
                  "frozen_pilot_trainer": digest(Path(__file__).with_name("train_heston_pilot.py")),
                  "batch_plan_and_guard": digest(Path(__file__).with_name("train_full_surface_budget.py")),
                  "heston_pilot": digest(sys.modules[HestonPilot.__module__].__file__)}}
    write_json(args.out / "run.json", run)
    payload, bank_record = load_or_generate_bank(args.bank, identity, pilot)
    if args.generate_only:
        write_json(args.out / "generation.json", bank_record)
        print(json.dumps({"out": str(args.out), "generated_only": True, "bank": str(args.bank)}))
        return
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import keras
    keras.utils.set_random_seed(args.seed)
    run.update(bank_record_sha256=digest(args.bank / "bank.json"), bank_payload_sha256=bank_record["payload_sha256"],
               versions={"python": platform.python_version(), **{name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}})
    train = paired_subset(payload, args.train_rows)
    plan = BatchPlan(args.train_rows, args.batch_size, args.updates, args.seed)
    workflow = build_workflow(pilot.names, pilot.bounds, args.schedule_updates)
    mask = np.ones(len(records["mid"]), dtype=bool)
    h, vega = np.asarray(records["half_spread"]), pilot.market_vega

    class Dataset(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)

        def __len__(self):
            return args.updates

        def __getitem__(self, step):
            indices = plan.indices(step)
            quotes, c, theta = augmented_batch(train["clean"][indices], train["parameters"][indices], h, vega, step, args.seed)
            return workflow.adapter(training_data(quotes, c, theta, records, mask, pilot.names))

    def fixed_probe(clean, theta, namespace):
        count = min(args.probe_rows, len(clean))
        quotes, c, parameters = augmented_batch(clean[:count], theta[:count], h, vega, 0, args.seed, namespace=namespace)
        return workflow.adapter(training_data(quotes, c, parameters, records, mask, pilot.names))

    probes = {"train": fixed_probe(train["clean"], train["parameters"], 2),
              "validation": fixed_probe(payload["validation_clean"], payload["validation_parameters"], 1)}
    run["curve_probe"] = {"rows": {key: len(value["inference_variables"]) for key, value in probes.items()},
                           "scope": "fixed independent-noise transformed-target NLL probes; development only; no checkpoint selection"}
    run["learning_rate"] = schedule_record(workflow.estimator.optimizer, args.schedule_updates)
    write_json(args.out / "run.json", run)
    guard = update_guard(plan)
    curve = []
    checkpoint_updates = {args.updates} | ({10000} if args.updates >= 10000 else set())
    started = time.perf_counter()

    class CampaignMonitor(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            updates = int(self.model.optimizer.iterations.numpy())
            if updates % args.log_every and updates not in checkpoint_updates:
                return
            if hasattr(self.model, "jax_state_sync"):
                self.model.jax_state_sync()
            if updates != guard.updates:
                raise RuntimeError("curve callback and exact optimizer guard disagree")
            point = {"updates": updates, "seconds": time.perf_counter() - started,
                     "cumulative_training_loss": float(logs["loss"])}
            for key, value in probes.items():
                metrics = self.model.compute_metrics(**value, stage="validation")
                point[key + "_probe_loss"] = float(keras.ops.convert_to_numpy(metrics["loss"]))
            if any(not np.isfinite(value) for value in point.values()):
                raise FloatingPointError("nonfinite campaign learning curve")
            curve.append(point)
            write_json(args.out / "curve.json", {"points": curve, "selection": "predeclared update endpoints"})
            print(json.dumps(point), flush=True)
            if updates in checkpoint_updates:
                destination = args.out if updates == args.updates else args.out / "checkpoints" / f"update-{updates:06d}"
                if destination != args.out:
                    destination.mkdir(parents=True, exist_ok=False)
                self.model.save(destination / "model.keras")
                record = _checkpoint_record(run, destination, updates, point, time.perf_counter() - started)
                write_json(destination / "checkpoint.json", record)
                restored, _ = load_checkpoint(destination)
                market = np.asarray(records["mid"])[None, :]
                before = sample(workflow, pilot.names, market, np.ones(1), records, mask, 16, args.seed + 31)
                after = sample(restored, pilot.names, market, np.ones(1), records, mask, 16, args.seed + 31)
                if not np.array_equal(before, after):
                    raise RuntimeError("checkpoint reload changed same-seed samples")
                market_draws = sample(restored, pilot.names, market, np.ones(1), records, mask, args.diagnostic_draws, args.seed + 32)[0]
                np.savez_compressed(destination / "condition-diagnostics.npz", market_c1=market_draws)
                record.update(same_seed_reload_exact=True, diagnostic_payload_sha256=digest(destination / "condition-diagnostics.npz"),
                    market_c1_parameter_quantiles_05_50_95={name: np.quantile(market_draws[:, i], [.05, .5, .95]).tolist() for i, name in enumerate(pilot.names)})
                write_json(destination / "checkpoint.json", record)

    history = workflow.estimator.fit(dataset=Dataset(), epochs=1, shuffle=False, verbose=0,
                                      callbacks=[guard, CampaignMonitor()])
    record = json.loads((args.out / "checkpoint.json").read_text())
    record.update(format="heston_pilot_campaign_training_v1", updates=guard.updates,
                  training_seconds=time.perf_counter() - started,
                  history={key: [float(x) for x in value] for key, value in history.history.items()},
                  exposure={"row_visits_min": int(guard.visits.min()), "row_visits_max": int(guard.visits.max()),
                            "batch_order_sha256": guard.order.hexdigest(), "batch_sizes": guard.batch_sizes},
                  curve_sha256=digest(args.out / "curve.json"))
    write_json(args.out / "training.json", record)
    print(json.dumps({"out": str(args.out), "actual_updates": guard.updates,
                      "seconds": record["training_seconds"], "scientific_acceptance": False}), flush=True)


if __name__ == "__main__":
    main()
