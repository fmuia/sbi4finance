"""The crash-day study: how likely is a crash day, answered by every method (seed of slot 6, on priors).

    pixi run crash-risk-study --out docs/results/crash-risk  # about an hour on 4 cores
    pixi run crash-risk-study --series 8 --fixed-series 4 --out /tmp/crash-risk-smoke

Each series is 1,000 daily returns simulated from a known truth. Every method then forecasts the
probability of at least one day with a loss of 5% or 7% or more over the next year, with an
interval, and the jump model's methods also give an interval for the jump frequency. The truth
is known, so each forecast is scored against the true probability.

Two designs, for each of the two tail models in `sbi4finance.crash_risk`:

* **prior**: every series has its own truth, drawn from the prior. This is the design of slot 5's
  40-series study, and the one under which a calibrated method must be calibrated on average.
* **fixed**: a few named truths, many series each. This is frequentist risk at a given truth:
  the jump model's truths sit at the two ends of the ridge, in its middle and near no jumps.

The output directory must be new. It receives config.json, records.jsonl (one series per line,
with the truth and every method's answers) and summary.json (the scores of `crash_risk.score`).
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread per worker: the workers are the parallelism

import numpy as np

from sbi4finance import crash_risk as cr

THRESHOLDS = (5.0, 7.0)
DRIFT = 8.0 / cr.YEAR


def _daily_sd(vol):
    return 100 * vol / np.sqrt(cr.YEAR)


FIXED = {
    "jump": {
        "typical": [np.log(2.5), np.log(3.0), -1.0, np.log(_daily_sd(0.16))],
        "rare_and_large": [np.log(0.5), np.log(6.0), -3.0, np.log(_daily_sd(0.16))],
        "frequent_and_small": [np.log(12.0), np.log(1.5), -0.5, np.log(_daily_sd(0.16))],
        "almost_none": [np.log(0.1), np.log(3.0), -1.0, np.log(_daily_sd(0.16))],
    },
    "student": {
        "nu_4": [np.log(4.0), np.log(_daily_sd(0.16) * np.sqrt(2 / 4))],
        "nu_8": [np.log(8.0), np.log(_daily_sd(0.16) * np.sqrt(6 / 8))],
    },
}


def tasks(n_prior, n_fixed, seed):
    for model in ("jump", "student"):
        for i in range(n_prior):
            yield {"model": model, "design": "prior", "truth": None, "index": i, "seed": seed}
        for name in FIXED[model]:
            for i in range(n_fixed):
                yield {"model": model, "design": "fixed", "truth": name, "index": i, "seed": seed}


def run_one(task):
    model = cr.MODELS[task["model"]]
    key = [task["seed"], ("jump", "student").index(task["model"]), 0 if task["design"] == "prior" else 1,
           0 if task["truth"] is None else 1 + list(FIXED[task["model"]]).index(task["truth"]), task["index"]]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    if task["truth"] is None:
        x, mu = model.draw_truth(rng)
    else:
        x, mu = np.array(FIXED[task["model"]][task["truth"]]), DRIFT
    record = {**task, "truth_x": [float(v) for v in x], "truth_mu": float(mu)}
    if task["model"] == "jump":
        y, jumps = model.simulate(x, mu, 1000, rng, with_jumps=True)
        record["jumps_in_window"] = int(jumps.sum())
    else:
        y = model.simulate(x, mu, 1000, rng)
    record["p_true"] = {f"{t:g}": float(model.crash_probability(x, t, mu)) for t in THRESHOLDS}
    start = time.process_time()
    try:
        record["result"] = cr.analyse(model, y, THRESHOLDS, n_boot=100, rng=rng)
    except Exception as error:  # recorded and counted in the summary, never silently dropped
        record["error"] = f"{type(error).__name__}: {error}"
    record["cpu_s"] = time.process_time() - start
    return record


def summarize(records):
    summary = {}
    for model in ("jump", "student"):
        for design, truths in (("prior", [None]), ("fixed", list(FIXED[model]))):
            for truth in truths:
                group = [r for r in records if r["model"] == model and r["design"] == design and r["truth"] == truth]
                done = [r for r in group if "error" not in r]
                if done:
                    name = f"{model}/{design}/{truth}" if truth else f"{model}/{design}"
                    summary[name] = {t: cr.score(done, t) for t in (f"{v:g}" for v in THRESHOLDS)}
                    summary[name]["failed_series"] = len(group) - len(done)
                    summary[name]["profile_off_grid"] = float(np.mean([r["result"]["profile_off_grid"] for r in done]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="a new directory")
    parser.add_argument("--series", type=int, default=300, help="series per model in the prior design")
    parser.add_argument("--fixed-series", type=int, default=100, help="series per named truth")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = {"series": args.series, "fixed_series": args.fixed_series, "seed": args.seed, "thresholds": THRESHOLDS,
              "window_days": 1000, "bootstrap_resamples": 100, "fixed_truths": FIXED,
              "levels": {"crash_probability": cr.LEVEL_INTERVAL, "jump_frequency": cr.LEVEL_PARAMETER}}
    (args.out / "config.json").write_text(json.dumps(config, indent=1))
    records, start = [], time.time()
    with Pool(args.workers) as pool, (args.out / "records.jsonl").open("w") as sink:
        for k, record in enumerate(pool.imap_unordered(run_one, list(tasks(args.series, args.fixed_series, args.seed)))):
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if (k + 1) % 25 == 0:
                print(f"{k + 1} series, {time.time() - start:.0f} s", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summarize(records), indent=1))
    print(f"done: {len(records)} series in {time.time() - start:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
