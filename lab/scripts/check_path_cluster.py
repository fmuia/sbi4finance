"""Real recurrent training, arbitrary masks and native reload in the locked runtime."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import resource

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank, inference  # noqa: E402


def check(args):
    started = time.monotonic()
    import bayesflow as bf
    import jax
    import keras
    devices = [str(device) for device in jax.devices()]
    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU allocation required, but JAX has no GPU device")
    result = {"format": "path_cluster_readiness_v1", "bayesflow": bf.__version__,
              "jax_devices": devices, "require_gpu": args.require_gpu,
              "production_shape": args.production_shape,
              "teaching": getattr(args, "teaching", False),
              "reserved_test_evaluated": False, "scientific_validation": False, "tasks": {}}
    network = (dict(inference.DEFAULT_NETWORK) if args.production_shape else
               {"hidden_dim": 8, "summary_dim": 8, "depth": 2, "bins": 4, "widths": [8, 8]})
    names = ([args.model] if getattr(args, "model", None) else list(bank.MODELS))
    for number, name in enumerate(names):
        task = bank.teaching_task(name) if getattr(args, "teaching", False) else bank.default_task(name)
        if args.quick and not args.production_shape:
            task = dict(task, n_observations=29, prefix_lengths=[1, 7, 13, 21, 29],
                        history_length_range=[1, 29])
        maximum = task["n_observations"]
        rng = np.random.default_rng(831 + number)
        rows = 128 if args.production_shape else 8
        split = {"observations": rng.normal(size=(rows, maximum, 1)) * np.sqrt(.04 * task["dt_years"]),
                 "parameters": inference.prior_samples(task, rows, 831 + number)}
        data = inference.training_data(split, task)
        keras.utils.set_random_seed(831 + number)
        workflow = inference.build_workflow(task, network=network,
                                            optimizer=keras.optimizers.Adam(5e-4))
        augmentation = inference.random_prefixes(task, 9)
        stream_statistics = None
        if getattr(args, "teaching", False):
            from sbi4finance.path_inference import dataset as streamed
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "readiness-bank"
                bank.write_bank(task, source, rows, 4, 0, chunk=128, seed=831 + number)
                split = streamed.load_subset(source, "train", rows)
                data = inference.training_data(split, task)
                offline = streamed.ShardedPathDataset(source, adapter=workflow.adapter,
                    batch_size=128 if args.production_shape else 4, seed=9,
                    prefix_policy="uniform_log_mixture", cache_shards=2)
                history = workflow._fit(dataset=offline, epochs=1, strategy="offline",
                    validation_data=data, keep_optimizer=True, verbose=0, shuffle=False).history
                stream_statistics = offline.statistics()
        else:
            history = workflow.fit_offline(data=data, epochs=1, batch_size=128 if args.production_shape else 4,
                                            validation_data=data, augmentations=augmentation,
                                            keep_optimizer=True, verbose=0).history
        if not np.isfinite(history["loss"] + history["val_loss"]).all():
            raise RuntimeError("nonfinite readiness loss")
        checked = sorted(set([1, min(17, maximum), maximum - 1, maximum, *task["prefix_lengths"]]))
        for length in checked:
            samples = inference.sample(workflow, task, split["observations"][:2, :length], 4, seed=13)
            if samples.shape != (2, 4, len(task["parameter_names"])):
                raise RuntimeError("arbitrary prefix sample shape mismatch")
        length = min(17, maximum)
        conditions = inference.conditioning_data(split["observations"][:2], task,
                                                  lengths=np.full(2, length))
        original = np.asarray(workflow.summary_network(conditions["observations"].astype("float32"),
                                                       mask=conditions["summary_mask"], training=False))
        altered = conditions["observations"].copy()
        altered[:, length:] = 1000.0
        changed = np.asarray(workflow.summary_network(altered.astype("float32"),
                                                      mask=conditions["summary_mask"], training=False))
        if not np.array_equal(original, changed):
            raise RuntimeError("masked future tokens changed the GRU summary")
        original_draws = workflow.sample(num_samples=4, conditions=conditions, sample_shape=(), seed=13)
        changed_draws = workflow.sample(num_samples=4, conditions={**conditions, "observations": altered},
                                        sample_shape=(), seed=13)
        if not np.array_equal(original_draws["parameters"], changed_draws["parameters"]):
            raise RuntimeError("BayesFlow sampling failed to forward the summary mask")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.keras"
            workflow.estimator.save(checkpoint)
            restored = inference.build_workflow(task, network=network)
            restored.load_estimator(checkpoint)
            before = inference.sample(workflow, task, split["observations"][:2, :length], 8, seed=91)
            after = inference.sample(restored, task, split["observations"][:2, :length], 8, seed=91)
            if not np.array_equal(before, after):
                raise RuntimeError("seeded native model reload changed draws")
        result["tasks"][name] = {"maximum_length": maximum, "checked_lengths": checked,
                                 "finite_training_loss": True, "mask_future_invariance": True,
                                 "seeded_native_reload_exact": True,
                                 "optimizer_steps": int(workflow.optimizer.iterations.numpy())}
        result["tasks"][name]["network"] = network
        result["tasks"][name]["batch_size"] = 128 if args.production_shape else 4
        result["tasks"][name]["streamed_offline_readiness"] = stream_statistics
        print(f"readiness passed: {name}", flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss"] = {"raw": raw_rss, "raw_unit": "bytes" if sys.platform == "darwin" else "KiB",
                          "bytes": int(raw_rss if sys.platform == "darwin" else raw_rss * 1024)}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "readiness.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--teaching", action="store_true", help="Check harmonized daily tasks with inferred drift")
    parser.add_argument("--model", choices=bank.MODELS, help="Check one independent model lane")
    parser.add_argument("--quick", action="store_true", help="Use length 29 for a local software check")
    parser.add_argument("--production-shape", action="store_true",
                        help="Check full history, batch 128 and the production GRU64/spline6")
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(check(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
