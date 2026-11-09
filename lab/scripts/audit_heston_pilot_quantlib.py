"""Independent QuantLib Heston price comparison for the raw-quote pilot.

QuantLib is optional, isolated from project dependencies. Example after a
temporary install with ``uv pip install --target /private/tmp/sbi4f-quantlib
QuantLib==1.44``:

  PYTHONPATH=/private/tmp/sbi4f-quantlib:src:../tutorial/src \
    .pixi/envs/core-reference/bin/python scripts/audit_heston_pilot_quantlib.py \
    --design evidence/.../designs/put_heavy.json --out evidence/.../quantlib.json

QuantLib's independently implemented Gatheral integrals supply the model
benchmark. Production uses a separate Lewis inversion and characteristic
function implementation. This remains a pointwise comparison.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from audit_heston_pilot_precision import read_design


QUANTLIB_SOURCE = "https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/vanilla/analytichestonengine.cpp"


def quantlib_prices(pilot, parameters, indices, *, adaptive):
    """Exact time scaling removes QuantLib's date rounding for actual maturities.

    Let s=t/T, U_s=T*V_(T*s). Then dU=κT(θT-U)ds+ξT√U dB_s;
    d log(S/F)=-U/2 ds+√U dB_s. Pricing to exactly one year with
    (v0T,κT,θT,ξT,ρ) therefore preserves any original real-valued T.
    Normalizing spot/forward to one removes deterministic carry; multiplying
    by the original D*F restores actual quote price units.
    """
    import QuantLib as ql

    v0, kappa, theta, xi, rho = parameters
    today = ql.Date(9, 11, 2026)
    expiry = today + 365
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    if day_count.yearFraction(today, expiry) != 1.:
        raise RuntimeError("QuantLib comparison requires exactly one ACT/365F year")
    zero = ql.YieldTermStructureHandle(ql.FlatForward(today, 0., day_count))
    spot = ql.QuoteHandle(ql.SimpleQuote(1.))
    prices = []
    for i in indices:
        r = pilot.records
        T = r["maturity"][i]
        process = ql.HestonProcess(zero, zero, spot, v0*T, kappa*T, theta*T, xi*T, rho)
        heston = ql.HestonModel(process)
        engine = (ql.AnalyticHestonEngine(heston, 1e-11, 5000) if adaptive
                  else ql.AnalyticHestonEngine(heston, 192))
        kind = ql.Option.Put if r["option_type"][i] == "put" else ql.Option.Call
        payoff = ql.PlainVanillaPayoff(kind, r["strike"][i] / r["forward"][i])
        option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
        option.setPricingEngine(engine)
        price = option.NPV() * r["discount"][i] * r["forward"][i]
        if not np.isfinite(price):
            raise FloatingPointError("QuantLib returned a nonfinite price")
        prices.append(price)
    return np.asarray(prices)


def audit(pilot):
    import QuantLib as ql

    n = len(pilot.records["mid"])
    indices = np.unique(np.linspace(0, n - 1, min(8, n), dtype=int))
    low, high = pilot.bounds.T
    rows = [
        dict(id="center", parameters=pilot.bounds.mean(axis=1).tolist()),
        dict(id="reference", parameters=[.04, 2., .04, .6, -.7]),
        dict(id="low_variance_high_xi", parameters=[low[0], low[1], low[2], high[3], low[4]]),
        dict(id="high_variance_high_xi", parameters=[high[0], low[1], high[2], high[3], low[4]]),
        dict(id="low_variance_high_kappa_weak_correlation", parameters=[low[0], high[1], high[2], high[3], high[4]]),
    ]
    result = {
        "schema": "raw_heston_quantlib_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantlib_version": ql.__version__, "quantlib_module": ql.__file__,
        "quantlib_source": QUANTLIB_SOURCE,
        "independence": "QuantLib AnalyticHestonEngine has separate CF and Gatheral inversion; production uses Lewis half-shift integral",
        "time_transform": "s=t/T,U_s=T*V_(Ts): v0'=T*v0,theta'=T*theta,kappa'=T*kappa,xi'=T*xi,rho'=rho; ACT/365F horizon exactly1year",
        "price_transform": "F'=1,K'=K/F,r'=q'=0; actual quote price=D*F*normalized QuantLib price",
        "quantlib_numerics": {"adaptive_relative_tolerance": 1e-11, "adaptive_max_evaluations": 5000,
                              "independent_laguerre_order": 192},
        "contract_indices": indices.tolist(), "number_of_quotes": n,
        "parameter_names": pilot.parameter_names, "bounds": pilot.bounds.tolist(),
        "numerical_fraction": pilot.numerical_fraction,
        "python": platform.python_version(), "points": [],
        "scope": "40 fixed parameter/contract price comparisons, not exhaustive model validation or posterior stability",
        "posterior_stability_assessed": False,
    }
    started = time.perf_counter()
    for row in rows:
        point = dict(row)
        try:
            production = pilot.prices(row["parameters"], indices)
            adaptive = quantlib_prices(pilot, row["parameters"], indices, adaptive=True)
            laguerre = quantlib_prices(pilot, row["parameters"], indices, adaptive=False)
            sigma = pilot.measurement_sigma[indices]
            ratio = np.abs(production - adaptive) / sigma
            refinement = np.abs(laguerre - adaptive) / sigma
            point.update(priced=True, production_prices=production.tolist(),
                         quantlib_adaptive_prices=adaptive.tolist(), quantlib_laguerre_prices=laguerre.tolist(),
                         max_abs_difference_over_measurement_sigma=float(ratio.max()),
                         max_abs_price_difference=float(np.max(np.abs(production - adaptive))),
                         quantlib_refinement_over_measurement_sigma=float(refinement.max()),
                         passed=bool(np.all(ratio <= pilot.numerical_fraction)
                                     and np.all(refinement <= pilot.numerical_fraction)))
        except Exception as error:
            point.update(priced=False, error_type=type(error).__name__, error=str(error), passed=False)
        result["points"].append(point)
        print(json.dumps(point), flush=True)
    result.update(all_points_priced=all(row["priced"] for row in result["points"]),
                  pointwise_price_gates_passed=all(row["passed"] for row in result["points"]),
                  max_abs_difference_over_measurement_sigma=max(
                      (row["max_abs_difference_over_measurement_sigma"] for row in result["points"] if row["priced"]), default=None),
                  wall_seconds=time.perf_counter() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, pilot = read_design(args.design)
    report = audit(pilot)
    report["design_path"] = str(args.design.resolve())
    report["design_sha256"] = hashlib.sha256(args.design.read_bytes()).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["pointwise_price_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
