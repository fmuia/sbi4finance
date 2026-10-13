"""Validate a trained SVJ network on held-out simulations and build its misspecification reference.

    pixi run svj-validate --out /tmp/svj-validation-new           # 1,000 held-out series, 500 draws each
    pixi run svj-validate --n-series 200 --out /tmp/svj-check     # quick look

Writes validation.json, figures and a bound misspecification.npz under a NEW explicit output
directory. The checkpoint and its original reference remain untouched. Checks: SBC ranks,
interval coverage near nominal, and contraction (what four years of returns identify).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sbi4finance import training_workflow
from sbi4finance.svj import amortized, artifacts, checks, model, plots

ROOT = Path(__file__).resolve().parents[1]
# Contraction reported by the design document's reference run (feasibility result 2), for comparison.
DESIGN_CONTRACTION = (0.52, 0.94, 0.61, 0.82, 0.81, 0.33, 0.56, 0.42, 0.75)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--n-series", type=int, default=1_000)
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out", required=True, help="new output directory; checkpoint/reference are immutable")
    args = parser.parse_args()
    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    out = training_workflow.reserve_output_dir(args.out)
    start = time.time()
    test = checks.MisspecificationTest.build(posterior, seed=args.seed)
    test.save(out / "misspecification.npz")
    reference_binding = artifacts.bind_reference(out / "misspecification.npz", checkpoint=args.checkpoint, seed=args.seed)
    fresh = model.simulate(500, seed=args.seed + 1)["returns"]
    false_alarm = float(np.mean(test.pvalue(posterior.summaries(fresh)) < 0.05))
    misspec_seconds = time.time() - start

    sims = model.simulate(args.n_series, seed=args.seed + 2)
    start = time.time()
    draws = posterior.sample(sims["returns"], num_samples=args.n_draws, seed=args.seed + 3)
    sample_seconds = time.time() - start
    truths = sims["targets"]

    sbc = checks.sbc(draws, truths)
    cover = checks.coverage(draws, truths)
    contraction = checks.contraction(draws, truths)
    labels = model.QUANTITY_LABELS

    plots.sbc_panels(sbc, labels, f"Simulation-based calibration on {args.n_series:,} held-out series").savefig(
        out / "sbc.png", dpi=150
    )
    plots.recovery_panels(
        model.quantities(truths), model.quantities(draws), labels, contraction, "Recovery on held-out series"
    ).savefig(out / "recovery.png", dpi=150)
    plots.contraction_bars(
        contraction, labels, "What four years of returns identify", DESIGN_CONTRACTION, "design document"
    ).savefig(out / "contraction.png", dpi=150)

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "n_series": args.n_series,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "checkpoint_sha256": artifacts.sha256(Path(args.checkpoint) / "model.keras"),
        "reference_binding": reference_binding,
        "scientific_acceptance": False,
        "sampling_seconds_per_series": sample_seconds / args.n_series,
        "sbc_band_half_width_at_centre": sbc["band_half_width_at_centre"],
        "sbc_all_inside": bool(sbc["inside"].all()),
        "misspecification": {
            "n_reference": len(test.reference),
            "n_null": len(test.null),
            "false_alarm_rate_at_5pct_on_500_fresh_simulations": false_alarm,
            "seconds": misspec_seconds,
        },
        "targets": {
            name: {
                "sbc_inside_band": bool(sbc["inside"][j]),
                "sbc_max_abs_deviation": float(sbc["max_abs_deviation"][j]),
                "coverage": {f"{lvl:.0%}": float(cover[i, j]) for i, lvl in enumerate(checks.COVERAGE_LEVELS)},
                "contraction": float(contraction[j]),
                "design_document_contraction": DESIGN_CONTRACTION[j],
            }
            for j, name in enumerate(model.TARGET_NAMES)
        },
    }
    (out / "validation.json").write_text(json.dumps(summary, indent=1))

    print(f"SBC: {int(sbc['inside'].sum())} of {len(labels)} targets inside the 95% simultaneous band "
          f"(half-width {sbc['band_half_width_at_centre']:.3f} at the centre)")
    print(f"{'target':<18}{'inside':>7}{'max dev':>9}{'cov50':>7}{'cov90':>7}{'cov95':>7}{'contr':>7}{'doc':>6}")
    for j, label in enumerate(labels):
        print(f"{label:<18}{'yes' if sbc['inside'][j] else 'NO':>7}{sbc['max_abs_deviation'][j]:>9.3f}"
              f"{cover[0, j]:>7.2f}{cover[2, j]:>7.2f}{cover[3, j]:>7.2f}{contraction[j]:>7.2f}{DESIGN_CONTRACTION[j]:>6.2f}")
    print(f"misspecification false-alarm rate on fresh simulations: {false_alarm:.1%} (nominal 5%)")
    print(f"sampling: {sample_seconds / args.n_series:.2f} s per series for {args.n_draws} draws; outputs in {out}/")


if __name__ == "__main__":
    main()
