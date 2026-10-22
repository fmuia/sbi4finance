"""A real end-of-day chain to the design-grid observation (study S8).

    python scripts/build_surface.py --chain CHAIN_FILE --format alpha-vantage|cboe \
        --source "vendor, endpoint, retrieval time" --out NEW_DIR

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--format", choices=("alpha-vantage", "cboe"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = args.chain.read_text()
    rows = chain.parse_alpha_vantage_csv(text) if args.format == "alpha-vantage" else chain.parse_cboe_json(text)
    result = chain.build_surface(rows)
    k, T, _ = surface.contract_arrays(tasks.rough_task()["context"]["contracts"])
    implied = result["diagnostics"]["implied_vol_grid"]
    maturities, atm, skew, curvature = surface.smile_statistics(implied, k, T)
    exponent, scale = surface.skew_power_law(maturities, skew)
    args.out.mkdir(parents=True, exist_ok=False)
    artifact = {
        "study": "S8", "source": args.source, "chain_sha256": hashlib.sha256(text.encode()).hexdigest(),
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
    print(json.dumps({"valuation": result["diagnostics"]["valuation"], "skew_exponent": exponent,
                      "expirations_used": len(result["diagnostics"]["expirations"]),
                      "grid_points_outside_quotes": int(result["missing"].sum())}, indent=1))


if __name__ == "__main__":
    main()
