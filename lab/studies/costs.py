"""Costs of every route on one machine (study S9), in CPU-seconds that count every thread.

    python lab/studies/costs.py --out NEW_DIR [--rough-paths 32768 --repeats 3]
        [--rough-checkpoint RUN --rough-bank BANK] [--heston-checkpoint RUN --heston-bank BANK]

Measured here, in CPU-seconds and in wall-clock seconds: one rough-Bergomi
likelihood evaluation (one priced design surface at the ladder's first rung,
with antithetic paths and control variates), one Heston likelihood evaluation
(the Fourier surface), and, when checkpoints are given, loading the network
with its first call (which compiles the sampler) and one amortized posterior
of 1,000 draws after that. The exact posteriors are not run here: their cost is
extrapolated as evaluations x cost per evaluation, with the number of
evaluations a random-walk chain needs for 400 effective draws taken from the
Heston chains of study S3 when their report is given (--heston-check), else
stated as an assumption. The costs of the audit, the pricer route and the
least-squares fits are recorded per surface by studies S5, S4 and S6.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import pricing, tasks  # noqa: E402


def cpu(fn, repeats):
    """CPU-seconds (every thread) and wall-clock seconds of repeated calls."""
    times, walls = [], []
    for _ in range(repeats):
        started, clock = time.process_time(), time.perf_counter()
        fn()
        times.append(time.process_time() - started)
        walls.append(time.perf_counter() - clock)
    return {"median": float(np.median(times)), "runs": times,
            "wall_median": float(np.median(walls)), "wall_runs": walls}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rough-paths", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rough-checkpoint", type=Path)
    parser.add_argument("--rough-bank", type=Path)
    parser.add_argument("--heston-checkpoint", type=Path)
    parser.add_argument("--heston-bank", type=Path)
    parser.add_argument("--heston-check", type=Path, help="S3 heston-check.json, for chain lengths")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rough = tasks.rough_task()
    curve = tasks.curve(rough, np.full(7, 0.04))
    median = {"hurst": 0.185, "eta": 2.0, "rho": -0.595}

    def rough_eval():
        pricing.rough_surface(median, curve, rough["context"]["contracts"], rough["numerics"]["blocks"],
                              [args.rough_paths], tasks.sigma(rough), np.inf, seed=1)

    heston = tasks.heston_task()
    parameters, v0 = tasks._heston_split(tasks.model_parameters(heston, [0.04, 0.05, 0.6, -0.6]))
    report = {"study": "S9", "cpu_count": __import__("os").cpu_count(),
              "rough_likelihood_evaluation_cpu_s": cpu(rough_eval, args.repeats),
              "rough_paths": args.rough_paths,
              "heston_likelihood_evaluation_cpu_s": cpu(
                  lambda: pricing.heston_surface(parameters, v0, heston["context"]["contracts"], nodes=1024,
                                                 upper_limit=600.0), max(args.repeats, 20))}
    evaluations = 20000
    source = "assumed: 20,000 random-walk steps for 400 effective draws in three dimensions"
    if args.heston_check:
        check = json.loads(args.heston_check.read_text())
        chains = [s["chains"] for s in check["surfaces"] if "chains" in s]
        steps = check["settings"]["steps"] * check["settings"]["chains"]
        per_ess = [steps / min(c["ess"]) for c in chains if c["ess"]]
        evaluations = float(np.median(per_ess) * 400)
        source = "median over the S3 Heston chains of evaluations per effective draw, times 400"
    unit = report["rough_likelihood_evaluation_cpu_s"]["median"]
    report["exact_rough_posterior"] = {"evaluations_for_400_effective_draws": evaluations, "source": source,
                                       "cpu_hours": evaluations * unit / 3600}
    report["exact_heston_posterior"] = {
        "evaluations_for_400_effective_draws": evaluations, "source": source,
        "cpu_seconds": evaluations * report["heston_likelihood_evaluation_cpu_s"]["median"]}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for name, checkpoint, bank_path in (("rough", args.rough_checkpoint, args.rough_bank),
                                        ("heston", args.heston_checkpoint, args.heston_bank)):
        if checkpoint is None:
            continue
        from evaluate_calibration import load_network
        from sbi4finance.calibration import bank, inference
        _, splits = bank.load_bank(bank_path)
        test = splits["test"]
        started, clock = time.process_time(), time.perf_counter()
        task, workflow, _ = load_network(checkpoint)
        inference.sample(workflow, task, test["observations"][:1], test["context_values"][:1], 1000, seed=0)
        report[f"{name}_network_load_and_first_call"] = {"cpu_s": time.process_time() - started,
                                                         "wall_s": time.perf_counter() - clock}

        def one():
            inference.sample(workflow, task, test["observations"][1:2], test["context_values"][1:2], 1000, seed=1)

        report[f"{name}_amortized_posterior_1000_draws_cpu_s"] = cpu(one, args.repeats)
    (args.out / "costs.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
