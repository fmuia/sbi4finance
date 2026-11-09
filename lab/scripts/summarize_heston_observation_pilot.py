"""Collect compact, source-bound Heston pilot evidence and a scientific figure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ["full-v2", "short-v2", "full-spread-only", "short-spread-only",
         "full-spread-half", "full-spread-double", "full-wide-tau",
         "symmetric-full", "symmetric-short", "full-wide-xi-extended", "short-wide-xi-extended"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases, missing = {}, []
    for name in CASES:
        folder = args.inputs / ("reference-" + name)
        paths = {key: folder / (key + ".json") for key in ("run", "diagnostics", "posterior", "initialization")}
        if not all(p.exists() for p in paths.values()):
            missing.append(name)
            continue
        records = {key: json.loads(path.read_text()) for key, path in paths.items()}
        d = records["diagnostics"]
        per_variable = [*d["parameters"].values(), *d["products"]["parameters"].values()]
        cases[name] = {"run": records["run"], "diagnostics": d, "posterior": records["posterior"],
                       "diagnostic_overview": {"r_hat_max": max(row["r_hat"] for row in per_variable),
                                               "ess_bulk_min": min(row["ess_bulk"] for row in per_variable),
                                               "ess_tail_min": min(row["ess_tail"] for row in per_variable)},
                       "input_sha256": {key: digest(path) for key, path in paths.items()},
                       "payload_sha256": digest(folder / "chains.npz"),
                       "initialization_scope": records["initialization"]["scope"],
                       "fit_terminations": [{key: row[key] for key in ("fit", "optimizer_success", "message", "function_evaluations")} for row in records["initialization"]["fits"]]}
    contrasts = {}
    for label, full, short in (("baseline_prior", "full-v2", "short-v2"),
                                ("wider_vol_of_vol_prior", "full-wide-xi-extended", "short-wide-xi-extended"),
                                ("symmetric_candidate", "symmetric-full", "symmetric-short")):
        if full in cases and short in cases:
            contrasts[label] = {}
            for product in ("control", "affected"):
                qfull, qshort = [np.array(cases[name]["posterior"]["products"][product]["expected_variance_strike_quantiles_05_50_95"]) for name in (full, short)]
                contrasts[label][product] = {"full_quantiles": qfull.tolist(), "short_quantiles": qshort.tolist(),
                    "short_to_full_90_width_ratio": float((qshort[2]-qshort[0])/(qfull[2]-qfull[0]))}
    record = {"format": "heston_european_observations_reflection_v1", "scientific_acceptance": False,
              "scope": "Heston-only pilot: full panel of 54 and short panel of 27 actual SPXW smile quotes, conditional on common carry anchors, ATM-based quote selection and frozen market vegas. No Bergomi work or asset promotion.",
              "cases": cases, "missing_cases": missing, "contrasts": contrasts,
              "limitations": ["No synchronized per-quote bid/ask timestamps or empirical measurement-noise calibration.",
                   "Full means full selected panel, not full market chain.",
                   "Scalar independent discrepancy does not represent systematic residual structure.",
                   "Carry sensitivity uses optimization; carry posterior uncertainty is not integrated.",
                   "Reference convergence does not establish financial-model adequacy.",
                   "Neural checkpoints fail conditional reference agreement; training budgets are development evidence.",
                   "Wider prior requires the extended Fourier ladder; the failed shorter-control audit is retained."],
              "source_sha256": digest(Path(__file__))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if all(name in cases for name in ("full-v2", "short-v2", "full-wide-xi-extended", "short-wide-xi-extended")):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), constrained_layout=True)
        names = ["full-v2", "short-v2", "full-wide-xi-extended", "short-wide-xi-extended"]
        labels = ["54 quotes, ξ ≤ 1.5", "27 short quotes, ξ ≤ 1.5", "54 quotes, ξ ≤ 3", "27 short quotes, ξ ≤ 3"]
        colors = ["#2176ae", "#db7c26", "#2176ae", "#db7c26"]
        for ax, product, title in zip(axes[0], ("control", "affected"),
                                      ("Near-term variance control", "Later forward-variance strike")):
            for i, name in enumerate(names):
                q = 10000 * np.asarray(cases[name]["posterior"]["products"][product]["expected_variance_strike_quantiles_05_50_95"])
                ax.errorbar(q[1], 3-i, xerr=[[q[1]-q[0]], [q[2]-q[1]]], fmt="o", color=colors[i], capsize=4, linewidth=2)
            ax.set(yticks=np.arange(4), yticklabels=labels[::-1], xlabel="Annualized variance × 10⁴", title=title)
            ax.grid(axis="x", alpha=.25)
        for ax, name, title in zip(axes[1], ("full-v2", "short-v2"),
                                   ("Full selected panel: residuals", "Short panel: later rows withheld")):
            residual = np.asarray(cases[name]["posterior"]["residual_over_halfspread"]).reshape(6, 9)
            picture = ax.imshow(residual, cmap="RdBu_r", vmin=-40, vmax=40, aspect="auto")
            ax.set(xticks=np.arange(9), xticklabels=[-4,-3,-2,-1.5,-1,-.5,0,.5,1],
                   yticks=np.arange(6), yticklabels=["Nov30", "Dec31", "Jan29", "Mar31", "Jun30", "Sep30"],
                   xlabel="Target standardized log-moneyness", title=title)
            if name == "short-v2":
                ax.axhline(2.5, color="black", linewidth=1.5)
            figure.colorbar(picture, ax=ax, label="(mid − median fair price) / half-spread", extend="both")
        figure.suptitle("Heston SPXW pilot: information, prior sensitivity and residual structure\n90% fair-value intervals; conditional on frozen carry and quote design", fontsize=13)
        figure.savefig(args.out / "heston-reflection.png", dpi=170)
        plt.close(figure)
    print(json.dumps({"cases": len(cases), "missing": missing, "all_mcmc_diagnostics_passed": all(c["diagnostics"]["mcmc_diagnostics_passed"] for c in cases.values()), "out": str(args.out)}))


if __name__ == "__main__":
    main()
