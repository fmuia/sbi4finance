"""Evaluate the pinned MMAR checkpoint and prepare a compact, reproducible lesson bundle.

Run with the participant Python, PYTHONPATH pointing at tutorial/src. The upstream
checkout is used only for an authoring-time parity check; never by the lesson.
The frozen rule is written before reserved-test inference. This script never
chooses seeds based on results or publishes assets; use studies/accept.py later.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from sbi4f_tutorial.mmar_takehome import diagnostics, inference, model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def parity(upstream):
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if rev != model.DESIGN["upstream_revision"]:
        raise ValueError("upstream checkout is not the pinned design revision")
    sys.path.insert(0, str(upstream))
    from mmar import sensitivity as reference
    rows = []
    for d in model.SUPPORTED_D:
        left = model.simulate(64, d, seed=170061112 + d)
        right = reference.simulate(64, d=d, seed=170061112 + d)
        errors = {name: float(np.max(np.abs(left[name] - right[name])))
                  for name in (*model.PARAMETERS, "returns", "trading_time_increments", "log_d")}
        if any(errors.values()):
            raise AssertionError(f"source parity failed at d={d}: {errors}")
        rows.append({"d": d, "paths": 64, "maximum_absolute_error": errors})
    # An out-of-prior stress case exercises whole-window rejection without
    # changing the parameter draw itself. It is not a training/test cohort.
    stress = np.tile([0., .8, .89, 2.06], (24, 1))
    a = model.simulate_parameters(stress, 8, seed=170061112)
    b = reference.simulate_from_parameters(stress, 8, np.random.default_rng(170061112))
    if not np.array_equal(a["returns"], b["returns"]):
        raise AssertionError("whole-path rejection parity failed")
    return {"upstream_revision": rev, "all_six_design_values": rows,
            "stress_rejection_parity": True, "kind": "bit-identical outputs at fixed seeds, not historical training provenance"}


def mdd(returns):
    # Import the workshop's shared log-return statistic when available.
    from sbi4f_tutorial.shared.path_statistics import maximum_drawdown
    return maximum_drawdown(np.log1p(returns))


def summaries(returns):
    from sbi4f_tutorial.shared.path_statistics import sample_acf
    log_returns = np.log1p(returns)
    acf = sample_acf(log_returns, absolute=True)
    return np.column_stack((100*np.std(returns, axis=-1),
                            100*np.quantile(np.abs(returns), .95, axis=-1),
                            100*mdd(returns))), acf["values"]


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rule = inference.freeze_rule()
    write_json(out / "frozen-rule.json", rule)
    parity_record = parity(args.upstream.resolve())
    write_json(out / "source-parity.json", parity_record)
    print("Parity passed at all six d values and whole-path redraw stress case", flush=True)
    null = diagnostics.null_statistics(rule)
    # Authoring may read external model payloads; participant restoration is
    # separately restricted to its verified assets/local-checkpoints cache.
    workflow = inference.build_workflow()
    workflow.load_estimator(args.checkpoint)
    arrays, records, full = {}, [], {}
    for d in model.CAPSTONE_D:
        started = time.perf_counter()
        panel = model.simulate(rule["histories_per_resolution"], d, seed=rule["test_seeds"][str(d)])
        truth = model.parameter_array(panel)
        draws = inference.sample(workflow, panel["returns"], d, count=512,
                                 seed=rule["sampling_seeds"][str(d)])
        result = diagnostics.evaluate(draws, truth, null=null)
        records.extend([dict(row, d=d) for row in result["records"]])
        for name in ("ranks", "median", "lower90", "upper90", "truth", "rank_deviations"):
            arrays[f"diagnostic_{d}_{name}"] = result[name]
        arrays[f"diagnostic_{d}_examples"] = draws[:6]
        arrays[f"diagnostic_{d}_histories"] = panel["returns"][:6]
        full[f"posterior_{d}"] = draws
        print(f"d={d}: evaluated {len(truth)} histories x {draws.shape[1]} draws in {time.perf_counter()-started:.1f}s", flush=True)
    adjusted = diagnostics.holm(records)
    np.savez_compressed(out / "reserved-full-draws.npz", **full)
    # Fixed observed data, never used in reserved-cohort checking or selection.
    if args.market_npz:
        with np.load(args.market_npz, allow_pickle=False) as market:
            observed = np.asarray(market["simple_returns"]).reshape(-1)
            dates = np.asarray(market["dates"]).astype(str)
    else:
        import pandas as pd
        market = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        window = market.loc[:"2021-01-07", "VOO"].tail(256)
        observed = window.to_numpy()
        dates = window.index.astype(str).to_numpy()
    if observed.shape != (256,):
        raise ValueError("the fixed market must have exactly 256 simple returns")
    prior = model.simulate(200, 8, seed=190061112)
    arrays["prior_summaries"], arrays["prior_abs_acf"] = summaries(prior["returns"])
    arrays["observed_summaries"], arrays["observed_abs_acf"] = summaries(observed[None])
    for d in model.CAPSTONE_D:
        posterior = inference.sample(workflow, observed, d, count=1024, seed=180061112+d)[0]
        scenarios = model.simulate_parameters(posterior[:512], d, seed=180071112+d)["returns"]
        arrays[f"posterior_{d}"] = posterior
        arrays[f"mdd_{d}"] = mdd(scenarios)
        arrays[f"wealth_{d}"] = np.concatenate((np.ones((12, 1)), np.cumprod(1+scenarios[:12], axis=1)), axis=1)
        arrays[f"predictive_{d}_summaries"], arrays[f"predictive_{d}_abs_acf"] = summaries(scenarios)
        if d == 8:
            point = np.repeat(posterior.mean(axis=0)[None], 512, axis=0)
            point_scenarios = model.simulate_parameters(point, d, seed=180081112)["returns"]
            arrays["point_mdd"] = mdd(point_scenarios)
            arrays["observed_wealth"] = np.r_[1., np.cumprod(1+observed)]
    bundle = out / "bundle"
    bundle.mkdir()
    np.savez_compressed(bundle / "saved.npz", **arrays)
    runtime = Path(model.__file__).resolve().parent
    report = {"format": "mmar_takehome_bundle_v1", "created": datetime.now(timezone.utc).isoformat(),
              "model_design": model.DESIGN, "checkpoint_sha256": digest(args.checkpoint),
              "checkpoint_origin": "amortized-fractals/checkpoints/sensitivity/model.keras at pinned revision",
              "checkpoint_historical_provenance": "source configuration reconstructed and numerical parity verified; original training seed/bank not independently established",
              "source_parity": parity_record, "frozen_rule": rule, "calibration": adjusted,
              "calibration_rejections": sum(row["rank_holm_reject"] for row in adjusted),
              "scope": "approximate amortized posterior; checks do not establish exact posterior or state-conditioned forecasts",
              "saved_sha256": digest(bundle / "saved.npz"),
              "software": {name: importlib.metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")},
              "runtime_sha256": {p.name: digest(p) for p in runtime.glob("*.py")},
              "producer_script_sha256": digest(__file__),
              "shared_statistics_sha256": digest(runtime.parent / "shared" / "path_statistics.py"),
              "task_sha256": model.fingerprint(),
              "checkpoint_url": "https://github.com/stefanradev93/amortized-fractals/raw/069d5dc5943dbe85505849f2fa8bfb563617ff97/checkpoints/sensitivity/model.keras",
              "scenario_count_per_group": 512, "market_draws_per_resolution": 1024,
              "market": {"ticker": "VOO", "n": 256, "return_type": "daily_simple",
                         "start_date": str(dates[0]), "end_date": str(dates[-1]),
                         "simple_returns_sha256": hashlib.sha256(np.asarray(observed, dtype="<f8").tobytes()).hexdigest(),
                         "source_file_sha256": digest(args.market_npz or args.csv)},
              "seeds": {"market_sampling": "180061112+d", "posterior_scenarios": "180071112+d",
                        "point_scenarios": 180081112, "prior": 190061112}}
    if args.verify_repeat:
        previous = json.loads((args.verify_repeat / "report.json").read_text())
        if previous["frozen_rule"] != rule or previous["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise AssertionError("repeat must use the unchanged frozen cohort and checkpoint")
        with np.load(args.verify_repeat / "saved.npz", allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[key], arrays[key], equal_nan=True) for key in old.files):
                raise AssertionError("source-hardening repeat changed a saved result; do not publish as equivalent")
        if json.dumps(previous["calibration"], sort_keys=True) != json.dumps(adjusted, sort_keys=True):
            raise AssertionError("source-hardening repeat changed the reserved calibration result")
        report["repeat_verification"] = {
            "previous_report_sha256": digest(args.verify_repeat / "report.json"),
            "all_saved_arrays_identical": True, "calibration_identical": True,
            "scope": "same frozen cohort/checkpoint, no tuning or seed selection; repeat binds final hardened source"}
    write_json(bundle / "report.json", report)
    write_json(out / "evaluation.json", report)
    print(json.dumps({"bundle": str(bundle), "rank_rejections": report["calibration_rejections"],
                      "bytes": sum(p.stat().st_size for p in bundle.iterdir())}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-repeat", type=Path, help="prior bundle: assert unchanged cohort, model and every saved array")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--csv", type=Path)
    data.add_argument("--market-npz", type=Path)
    prepare(parser.parse_args())
