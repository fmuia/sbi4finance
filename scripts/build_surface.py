"""A real end-of-day chain to the design-grid observation (study S8).

    python scripts/build_surface.py --chain CHAIN_FILE --format alpha-vantage|cboe \
        --source "vendor, endpoint, retrieval time" --out NEW_DIR \
        [--rate-curve 0.25:0.0426,2:0.0479 --rate-source "Treasury yields, date"]

American options (SPY) need --rate-curve: the discount then comes from Treasury
yields (bond-equivalent, in decimals, at maturities in years) and the forward from
put-call parity on the pairs nearest the money.

The chain file stays where it is; only derived quantities are written:
design-grid prices in forward units, their quote-noise sds, the seven
forward-variance levels, the implied-volatility grid, forwards, discounts and
replicated total variances per expiration, and the cleaning counts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import chain, surface, tasks  # noqa: E402


def figure(artifact, k, T, path):
    """The design-grid smiles by maturity, and the ATM skew's term structure with its power law."""
    import numpy as np

    from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style

    style()
    days = artifact["design"]["days"]
    deltas = np.array(artifact["design"]["deltas"])
    implied = np.array(artifact["diagnostics"]["implied_vol_grid"]).reshape(len(days), len(deltas))
    outside = np.array(artifact["grid_points_outside_quoted_strikes"], dtype=bool).reshape(implied.shape)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(days)))
    for i, (day, colour) in enumerate(zip(days, colours)):
        left.plot(deltas, 100 * implied[i], color=colour, marker="o", markersize=3.5, linewidth=1.3, label=f"{day} d")
        left.scatter(deltas[outside[i]], 100 * implied[i][outside[i]], s=40, facecolor="none", edgecolor=INK,
                     linewidth=0.8, zorder=4)
    left.set_xlabel("δ = k / (0.2 √T)")
    left.set_ylabel("implied volatility (%)")
    left.legend(fontsize=7.5, ncol=2, title="maturity, trading days", title_fontsize=7.5)
    left.set_title("design-grid smiles (open circles: outside the quoted strikes)", loc="left", fontsize=9)
    maturities = np.array(days) / 252
    skew = np.abs(np.array(artifact["atm_skew"]))
    fit = artifact["skew_power_law"]
    right.loglog(maturities, skew, "o", color=SERIES[0], label="ATM skew |∂σ/∂k|")
    grid = np.geomspace(maturities[0], maturities[-1], 50)
    right.loglog(grid, fit["scale"] * grid ** fit["exponent"], color=SERIES[1],
                 label=f"power law, exponent {fit['exponent']:.2f}")
    right.loglog(grid, fit["scale"] * grid ** -0.4 * (maturities[2] ** fit["exponent"] / maturities[2] ** -0.4),
                 color=BASELINE, linestyle="--", label="exponent −0.4, for reference")
    right.set_xlabel("maturity (years)")
    right.legend(fontsize=7.5)
    right.set_title("the ATM skew's term structure", loc="left", fontsize=9)
    fig.suptitle(f"SPY, {artifact['diagnostics']['valuation']}: the smile on the design grid", x=0.01, ha="left",
                 color=INK, fontsize=10.5, fontweight="bold")
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--format", choices=("alpha-vantage", "cboe"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate-curve", help="T:yield pairs, e.g. 0.25:0.0426,2:0.0479 (bond-equivalent yields)")
    parser.add_argument("--rate-source", default="")
    args = parser.parse_args()
    text = args.chain.read_text()
    rows = chain.parse_alpha_vantage_csv(text) if args.format == "alpha-vantage" else chain.parse_cboe_json(text)
    points = [tuple(float(x) for x in pair.split(":")) for pair in args.rate_curve.split(",")] if args.rate_curve else None
    result = chain.build_surface(rows, rate=chain.rate_curve(points) if points else None)
    k, T, _ = surface.contract_arrays(tasks.rough_task()["context"]["contracts"])
    implied = result["diagnostics"]["implied_vol_grid"]
    maturities, atm, skew, curvature = surface.smile_statistics(implied, k, T)
    exponent, scale = surface.skew_power_law(maturities, skew)
    args.out.mkdir(parents=True, exist_ok=False)
    artifact = {
        "study": "S8", "source": args.source, "chain_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "rate_curve": {"points_bond_equivalent": points, "source": args.rate_source} if points else None,
        "raw_quotes_committed": False, "design": {"days": list(surface.DESIGN_DAYS), "deltas": list(surface.DESIGN_DELTAS)},
        "prices_forward_units": result["prices"].tolist(), "noise_sd": result["noise_sd"].tolist(),
        "forward_variance_levels": result["forward_variance_levels"].tolist(),
        "grid_points_outside_quoted_strikes": result["missing"].tolist(),
        "atm_implied_vol": atm.tolist(), "atm_skew": skew.tolist(), "atm_curvature": curvature.tolist(),
        "skew_power_law": {"exponent": exponent, "scale": scale,
                           "note": "fitted from one week to one year; under rough Bergomi it is steeper than "
                                   "H - 1/2 when the vol-of-vol is large, so exponent + 1/2 tends to understate H"},
        "diagnostics": result["diagnostics"],
    }
    (args.out / "surface.json").write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    figure(artifact, k, T, args.out / "surface.png")
    print(json.dumps({"valuation": result["diagnostics"]["valuation"], "skew_exponent": exponent,
                      "expirations_used": len(result["diagnostics"]["expirations"]),
                      "grid_points_outside_quotes": int(result["missing"].sum())}, indent=1))


if __name__ == "__main__":
    main()
