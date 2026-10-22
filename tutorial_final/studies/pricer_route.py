"""The neural-likelihood route against the amortized posterior (study S4).

    python tutorial_final/studies/pricer_route.py --bank BANK --checkpoint RUN --out NEW_DIR \
        [--first 2050 --surfaces 50 --chains 4 --steps 20000 --burn 4000]

1. Train the neural pricer (deep calibration's forward map) on the bank's clean
   training prices, select by validation, and measure its error on test surfaces
   in quote-noise units.
2. On --surfaces test surfaces from --first, run random-walk Metropolis chains
   on the pricer's likelihood, started from dispersed network draws, and compare
   the pooled chain draws with the amortized posterior: means within 0.3
   posterior sd and 90% widths within 25%, per parameter.

The two routes share the bank and nothing else, so their agreement bounds the
systematic error of either one by the pricer's error.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank, correction, plots, pricer_net, validation  # noqa: E402
from sbi4finance.svj.pmmh import rhat_ess  # noqa: E402
from evaluate_calibration import LABELS, load_network, posterior_draws  # noqa: E402
from heston_check import jsonable, passes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=2050)
    parser.add_argument("--surfaces", type=int, default=50)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--burn", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank)
    task, workflow, _ = load_network(args.checkpoint)
    if task != manifest["configuration"]["task"]:
        raise ValueError("checkpoint and bank tasks differ")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    pricer, record = pricer_net.train(task, splits["train"], splits["validation"], epochs=args.epochs, seed=args.seed)
    train_cpu = time.process_time() - started
    pricer.save(args.out / "pricer")
    test = splits["test"]
    held_out = pricer_net.errors(pricer, task, {k: v[: args.first] for k, v in test.items()})
    rows = np.arange(args.first, args.first + args.surfaces)
    network = posterior_draws(workflow, task, test, rows, args.draws, args.seed)
    bounds = correction.bounds_of(task)
    names = task["parameter_names"]
    surfaces, panels = [], []
    for i, row in enumerate(rows):
        target = pricer_net.log_posterior(pricer, task, test["observations"][row, :, 0],
                                          test["context_values"][row] if test["context_values"].shape[1] else None)
        z = correction.to_unbounded(network[i], bounds)
        starts = z[np.random.default_rng(args.seed + i).choice(len(z), args.chains, replace=False)]
        scale = np.cov(z.T) * 2.38**2 / z.shape[1]
        begun = time.process_time()
        chains, acceptance = pricer_net.metropolis(target, starts, scale, args.steps, seed=args.seed + 1000 + i)
        chain_cpu = time.process_time() - begun
        kept = correction.to_bounded(chains[:, args.burn:], bounds)
        diagnostics = [rhat_ess(kept[:, :, d]) for d in range(kept.shape[2])]
        pooled = kept.reshape(-1, kept.shape[2])
        agree = validation.agreement(network[i], pooled)
        agree["passed"] = passes(agree)
        surfaces.append({"test_row": int(row), "truth": test["parameters"][row].tolist(), "agreement": agree,
                         "rhat": [d[0] for d in diagnostics], "ess": [d[1] for d in diagnostics],
                         "acceptance": acceptance.tolist(), "chain_cpu_seconds": chain_cpu})
        if len(panels) < 4:
            panels.append((f"surface {int(row)}", pooled, network[i], None, test["parameters"][row]))
    rms = np.sqrt(np.mean(held_out**2, axis=0))
    summary = {
        "pricer": {**record, "train_cpu_seconds": train_cpu,
                   "held_out_rms_error_in_noise_sd": {"median_over_contracts": float(np.median(rms)), "max": float(rms.max())},
                   "held_out_abs_error_q99_in_noise_sd": float(np.quantile(np.abs(held_out), 0.99))},
        "pass_rate": float(np.mean([s["agreement"]["passed"] for s in surfaces])),
        "max_standardized_mean_error": np.max([s["agreement"]["standardized_mean_error"] for s in surfaces], axis=0).tolist(),
        "median_standardized_mean_error": np.median([s["agreement"]["standardized_mean_error"] for s in surfaces], axis=0).tolist(),
        "width_ratio_median": np.median([s["agreement"]["width_ratio"] for s in surfaces], axis=0).tolist(),
        "rhat_max": float(np.max([s["rhat"] for s in surfaces])), "ess_min": float(np.min([s["ess"] for s in surfaces])),
        "chain_cpu_seconds_median": float(np.median([s["chain_cpu_seconds"] for s in surfaces])),
    }
    report = {"study": "S4", "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
              "checkpoint_sha256": hashlib.sha256((args.checkpoint / "model.keras").read_bytes()).hexdigest(),
              "settings": {k: v for k, v in vars(args).items() if k not in ("bank", "checkpoint", "out")},
              "parameters": names, "summary": summary, "surfaces": surfaces}
    (args.out / "pricer-route.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    labels = [LABELS[n] for n in names]
    plots.agreement(panels, labels, "neural pricer + MCMC").savefig(args.out / "agreement.png", dpi=160)
    print(json.dumps(jsonable(summary), indent=1))


if __name__ == "__main__":
    main()
