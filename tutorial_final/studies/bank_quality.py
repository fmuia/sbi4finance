"""Cost and numerical quality of a calibration bank (study S1, figure S1-1).

    python tutorial_final/studies/bank_quality.py --bank BANK --out NEW_DIR [--shards SHARD_DIR]

For rough Bergomi: the Monte Carlo standard error of every clean price over its
quote-noise standard deviation, by maturity and delta, against the bank's
budget; the largest eigenvalue of each surface's Monte Carlo covariance in units
of the quote noise (quotes priced on the same paths have correlated errors, so the
per-quote ratio understates their joint effect); the share of maturity blocks that
needed each rung of the paths ladder; and, given the shard directory, the
generation cost in CPU-hours per 1,000 surfaces. For Heston the numerical error is the refined-minus-base quadrature
difference instead, against its own budget.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sbi4finance.calibration import bank, surface, tasks  # noqa: E402

TENORS = {5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m", 126: "6m", 252: "1y"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=Path)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank)
    task = manifest["configuration"]["task"]
    sigma = tasks.sigma(task)
    rough = task["model"] == "rough"
    if rough:
        error = np.concatenate([s["pricing_standard_errors"] for s in splits.values()]) / sigma
        budget = task["numerics"]["max_mc_se_over_quote_sigma"]
        cap = task["numerics"].get("max_mc_se_over_quote_sigma_last_rung", budget)
        label = "Monte Carlo standard error / quote-noise sd"
    else:
        error = np.abs(np.concatenate([s["refinement_difference"] for s in splits.values()])) / sigma
        budget = task["numerics"]["max_refinement_over_quote_sigma"]
        cap = budget
        label = "|refined − base quadrature price| / quote-noise sd"
    count = len(error)
    k, maturity, _ = surface.contract_arrays(task["context"]["contracts"])
    delta = surface.standardized_moneyness(k, maturity)
    report = {"study": "S1", "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
              "surfaces": count, "error_measure": label, "budget": budget, "last_rung_cap": cap,
              "share_of_surfaces_with_a_quote_above_budget": float(np.mean(np.any(error > budget, axis=1))),
              "error_quantiles_all_quotes": dict(zip(("q50", "q90", "q99", "max"),
                                                     np.quantile(error, [0.5, 0.9, 0.99, 1.0]).tolist())),
              "share_of_quotes_above_0_1": float(np.mean(error > 0.1)),
              "noise_variance_added_by_numerics": {"median": float(np.median(error**2)), "max": float(np.max(error**2))}}
    if rough:
        paths = np.concatenate([s["pricing_paths"] for s in splits.values()])
        ladder = task["numerics"]["paths_ladder"]
        report["paths_ladder"] = ladder
        report["share_of_blocks_by_rung"] = {
            f"block {b}": {str(int(n)): float(np.mean(paths[:, b] == n)) for n in ladder} for b in range(paths.shape[1])}
        scale = 1.0 / np.outer(sigma, sigma)
        eigen = np.concatenate([np.linalg.eigvalsh(s["pricing_covariance"] * scale)[:, -1] for s in splits.values()])
        report["mc_covariance_largest_eigenvalue_over_noise_variance"] = {
            "q50": float(np.median(eigen)), "q90": float(np.quantile(eigen, 0.9)), "q99": float(np.quantile(eigen, 0.99)),
            "max": float(eigen.max()), "share_above_0_25": float(np.mean(eigen > 0.25)),
            "note": "of diag(1/sigma) C diag(1/sigma); the per-quote (SE/sigma)^2 is its diagonal"}
    if args.shards:
        records = [json.loads(p.read_text()) for p in sorted(args.shards.glob("*/manifest.json"))]
        cpu = sum(r["generation_cpu_seconds"] for r in records)
        examples = sum(r["examples"] for r in records)
        report["generation"] = {"shards": len(records), "surfaces": examples, "cpu_seconds": cpu,
                                "cpu_seconds_per_surface": cpu / examples,
                                "cpu_hours_per_1000_surfaces": cpu / examples * 1000 / 3600}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "bank-quality.json").write_text(json.dumps(report, indent=2) + "\n")

    from sbi4finance.calibration.plots import INK, SERIES, plt, style

    style()
    days = sorted({int(round(t * surface.TRADING_DAYS)) for t in maturity})
    fig, axes = plt.subplots(1, len(days), figsize=(13, 2.9), sharey=True, constrained_layout=True)
    deltas = sorted(set(np.round(delta, 2)))
    for ax, d in zip(axes, days):
        columns = [np.flatnonzero(np.isclose(maturity * surface.TRADING_DAYS, d) & np.isclose(delta, x))[0]
                   for x in deltas]
        ax.boxplot([error[:, c] for c in columns], positions=range(len(deltas)), widths=0.6, showfliers=False,
                   patch_artist=True, boxprops={"facecolor": SERIES[0], "alpha": 0.6}, medianprops={"color": INK})
        ax.axhline(budget, color=INK, linestyle="--", linewidth=1)
        if cap > budget:
            ax.axhline(cap, color=INK, linestyle=":", linewidth=1)
        ax.set_xticks(range(len(deltas))[::2], [f"{x:g}" for x in deltas[::2]])
        ax.set_title(TENORS.get(d, f"{d}d"), loc="left", fontsize=9)
        ax.set_xlabel("δ")
    axes[0].set_ylabel(label, fontsize=8)
    model = "rough Bergomi" if rough else "Heston"
    fig.suptitle(f"{model} bank: numerical error per quote over {count:,} surfaces (boxes: quartiles and 1.5 IQR; "
                 f"dashed: the budget, {budget:g}" + (f"; dotted: the last rung's cap, {cap:g})" if cap > budget else ")"),
                 x=0.01, ha="left", fontsize=9.5)
    fig.savefig(args.out / "numerical-error.png", dpi=160)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
