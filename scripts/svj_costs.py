"""Cost per name of each route, from the recorded runs (feasibility result 4).

    pixi run svj-costs

The network pass is timed afresh: 1,000 draws for one simulated series, after one warm-up call that compiles
the sampler. The other routes come from the runs already recorded in docs/results: the audit from the four
agreement series and the two daily windows, particle MCMC per 400 effective draws from the four agreement
series and VOO's window, and approximate ML from the 40 fits of the classical baseline. All costs are
process CPU seconds, which count every thread.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sbi4finance.svj import amortized, model, plots

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
DESIGN_DOCUMENT_MEDIANS = {"network": 1.4, "audit": 35.0, "pmmh": 367.0, "mle": 118.0}  # CPU-s per name, 2-core VM
NAMES = 2_000


def network_runs(checkpoint, repeats, seed):
    posterior = amortized.AmortizedPosterior.load(checkpoint)
    y = model.simulate(1, seed=seed)["returns"]
    posterior.sample(y, num_samples=1_000, seed=0)  # compiles the sampler
    runs = []
    for k in range(repeats):
        start = time.process_time()
        posterior.sample(y, num_samples=1_000, seed=k + 1)
        runs.append(time.process_time() - start)
    return runs


def recorded_runs():
    agreement = json.loads((RESULTS / "svj-agreement" / "agreement.json").read_text())["series"]
    daily = {name: json.loads((RESULTS / "svj-daily" / f"{name}.json").read_text())
             for name in ("VOO_2026-09-03", "SPMO_2026-09-03", "VOO_2026-09-03_pmmh")}
    voo_pmmh = daily["VOO_2026-09-03_pmmh"]
    classical = []
    for path in sorted((RESULTS / "svj-classical").glob("classical_series_*_of_*.json")):
        classical += [record["cpu_s"] for record in json.loads(path.read_text())]
    return {
        "audit": [s["importance_sampling"]["cpu_s"] for s in agreement]
        + [daily[name]["timings"]["audit_cpu_s"] for name in ("VOO_2026-09-03", "SPMO_2026-09-03")],
        "pmmh": [s["pmmh"]["cpu_s_per_400_ess"] for s in agreement]
        + [voo_pmmh["timings"]["pmmh_cpu_s"] * 400 / voo_pmmh["pmmh"]["min_ess_bulk"]],
        "mle": classical,
    }


def per_portfolio(seconds):
    hours = seconds * NAMES / 3600
    return f"{hours:,.1f} CPU-hours" if hours < 10 else f"{hours:,.0f} CPU-hours"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=str(RESULTS / "svj-costs"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {"network": network_runs(args.checkpoint, args.repeats, args.seed), **recorded_runs()}
    summary = {
        route: {
            "runs": [round(float(v), 2) for v in values],
            "median_cpu_s": float(np.median(values)),
            "range_cpu_s": [float(np.min(values)), float(np.max(values))],
            "cpu_hours_for_2000_names_at_median": float(np.median(values)) * NAMES / 3600,
            "design_document_median_cpu_s": DESIGN_DOCUMENT_MEDIANS[route],
        }
        for route, values in runs.items()
    }
    medians = {route: s["median_cpu_s"] for route, s in summary.items()}
    summary["ratios_at_the_median"] = {
        "pmmh_over_network": medians["pmmh"] / medians["network"],
        "pmmh_over_network_plus_audit": medians["pmmh"] / (medians["network"] + medians["audit"]),
    }
    (out / "costs.json").write_text(json.dumps(summary, indent=1))

    labels = {
        "network": "amortized posterior, 1,000 draws\n(one series, compiled sampler)",
        "audit": "audit: importance sampling with the filter\n(3,000 proposals)",
        "mle": "approximate ML + Hessian\n(point estimate only, two starts)",
        "pmmh": "particle MCMC to 400 effective draws\n(4 chains)",
    }
    colors = {"network": plots.SERIES[0], "audit": plots.SERIES[2], "mle": plots.INK_2, "pmmh": plots.SERIES[1]}
    rows = []
    for route in ("network", "audit", "mle", "pmmh"):
        values = runs[route]
        median = np.median(values)
        note = f"median {median:,.1f} s" if median < 10 else f"median {median:,.0f} s"
        note += f" over {len(values)} runs\n{per_portfolio(median)} for {NAMES:,} names"
        rows.append((labels[route], values, colors[route], note, DESIGN_DOCUMENT_MEDIANS[route]))
    minutes = json.loads((Path(args.checkpoint) / "training.json").read_text())["seconds"] / 60
    fig = plots.cost_dots(rows, "CPU seconds per name (log scale)",
                          f"Cost per name of each route; the network itself trains once, in {minutes:.0f} minutes",
                          reference_label="design document, median (2-core VM)")
    fig.savefig(out / "cost_per_name.png", dpi=150)
    for route, s in summary.items():
        print(route, s)


if __name__ == "__main__":
    main()
