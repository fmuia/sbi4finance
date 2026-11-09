"""Resolve the symmetric-layout audit failure without discarding failed rungs.

Run with the isolated QuantLib directory on PYTHONPATH, as documented by
audit_heston_pilot_quantlib.py. The fixed parameter vector is the exact prior
corner where doubling the accepted rung violates price bounds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_heston_pilot_precision import read_design
from audit_heston_pilot_quantlib import quantlib_prices


PARAMETERS = [.01, 10., .01, 1.5, -.95]


def main():
    import QuantLib as ql

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, pilot = read_design(args.design)
    indices = np.arange(len(pilot.records["mid"]))
    base, controls = pilot.prices(PARAMETERS, return_diagnostics=True)
    reference = quantlib_prices(pilot, PARAMETERS, indices, adaptive=True)
    sigma = pilot.measurement_sigma
    report = {"schema": "raw_heston_symmetric_corner_diagnostic_v1", "parameters": PARAMETERS,
              "design_sha256": hashlib.sha256(args.design.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "quantlib_version": ql.__version__, "number_of_quotes": len(indices),
              "production_controls": {key: value for key, value in controls.items() if key != "difference"},
              "production_prices": base.tolist(), "quantlib_prices": reference.tolist(),
              "production_max_abs_difference_over_measurement_sigma": float(np.max(np.abs(base - reference) / sigma)),
              "production_worst_contract_index": int(np.argmax(np.abs(base - reference) / sigma)),
              "production_price_gate_passed": bool(np.all(np.abs(base - reference) <= pilot.numerical_fraction * sigma)),
              "numerical_fraction": pilot.numerical_fraction,
              "scope": "exact failed prior corner; actual production prices versus independent QuantLib and explicit integration variants",
              "control_variants": []}
    n, c = controls["nodes"], controls["cutoff"]
    for name, nodes, cutoff in (("double_nodes_and_cutoff", 2*n, 2*c),
                                ("double_nodes_same_cutoff", 2*n, c),
                                ("quadruple_nodes_same_cutoff", 4*n, c),
                                ("quadruple_nodes_and_cutoff", 4*n, 4*c)):
        variant = {"name": name, "nodes": nodes, "cutoff": cutoff}
        try:
            prices = pilot._rung_prices(PARAMETERS, indices, nodes, cutoff)
            ratio = np.abs(prices - reference) / sigma
            variant.update(priced=True, prices=prices.tolist(),
                           max_abs_difference_over_measurement_sigma=float(np.max(ratio)),
                           max_abs_shift_from_production_over_measurement_sigma=float(np.max(np.abs(prices - base) / sigma)),
                           price_gate_passed=bool(np.all(ratio <= pilot.numerical_fraction)))
        except Exception as error:
            variant.update(priced=False, error_type=type(error).__name__, error=str(error), price_gate_passed=False)
        report["control_variants"].append(variant)
        print(json.dumps({key: value for key, value in variant.items() if key != "prices"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
