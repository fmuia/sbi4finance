"""Bounded full-panel sensitivity to a Uniform kappa upper bound of 20 vs 10.

Reference generation uses run_heston_observation_pilot.py unchanged, with the
explicitly derived put_heavy_kappa20_xi3.json design, --vol-of-vol-max 3 and
--extended-quadrature. Four chains retain 12000 draws after 6000 tuning steps.
The primary target and all historical identities remain unchanged.

This script compares completed references with chain-aware Monte Carlo errors.
It can also resolve the exact new-domain quadrature corner independently:
put the isolated QuantLib install on PYTHONPATH and use --corner-only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from sbi4finance.calibration.heston_pilot import HestonPilot
if __package__:
    from .run_heston_observation_pilot import expected_variance
else:
    from run_heston_observation_pilot import expected_variance


Z95 = 1.959963984540054
PRACTICAL_MEAN_FRACTION_SD = .15
PRACTICAL_QUANTILE_FRACTION_WIDTH = .15


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def corner_check(design_path):
    """Retain failed controls, compare accepted prices to independent QuantLib."""
    import QuantLib as ql
    if __package__:
        from .audit_heston_pilot_precision import read_design
        from .audit_heston_pilot_quantlib import quantlib_prices
    else:
        from audit_heston_pilot_precision import read_design
        from audit_heston_pilot_quantlib import quantlib_prices

    _, pilot = read_design(design_path)
    parameters = [.01, 20., .01, 3., -.95]
    indices = np.arange(len(pilot.records["mid"]))
    production, controls = pilot.prices(parameters, return_diagnostics=True)
    independent = quantlib_prices(pilot, parameters, indices, adaptive=True)
    sigma = pilot.measurement_sigma
    result = {"parameters": parameters, "design_sha256": digest(design_path),
              "quantlib_version": ql.__version__, "number_of_quotes": len(indices),
              "production_controls": {k: v for k, v in controls.items() if k != "difference"},
              "production_prices": production.tolist(), "quantlib_prices": independent.tolist(),
              "production_max_difference_measurement_sigma": float(np.max(np.abs(production - independent) / sigma)),
              "production_gate_passed": bool(np.all(np.abs(production - independent) <= pilot.numerical_fraction * sigma)),
              "variants": []}
    nodes, cutoff = controls["nodes"], controls["cutoff"]
    for name, n, c in (("double_both", 2*nodes, 2*cutoff),
                       ("double_nodes_same_cutoff", 2*nodes, cutoff),
                       ("quadruple_nodes_same_cutoff", 4*nodes, cutoff),
                       ("quadruple_both", 4*nodes, 4*cutoff)):
        point = {"name": name, "nodes": n, "cutoff": c}
        try:
            price = pilot._rung_prices(parameters, indices, n, c)
            ratio = float(np.max(np.abs(price - independent) / sigma))
            point.update(priced=True, max_difference_measurement_sigma=ratio,
                         price_gate_passed=ratio <= pilot.numerical_fraction)
        except Exception as error:
            point.update(priced=False, error_type=type(error).__name__, error=str(error), price_gate_passed=False)
        result["variants"].append(point)
        print(json.dumps(point), flush=True)
    result["scope"] = "exact new-prior corner; failed finer controls remain visible; independent production-price comparison is separate"
    return result


def load_reference(directory, design_path):
    directory = Path(directory)
    config = json.loads((directory / "run.json").read_text())
    design = json.loads(Path(design_path).read_text())
    if config["design_sha256"] != digest(design_path):
        raise ValueError("reference run differs from its exact declared design bytes")
    with np.load(directory / "chains.npz", allow_pickle=False) as archive:
        draws = archive["parameters"]
    expected = (config["settings"]["chains"], config["settings"]["draws"], len(config["names"]))
    if draws.shape != expected or not np.isfinite(draws).all():
        raise ValueError("reference chain/draw/parameter shape or values disagree with run record")
    diagnostic = json.loads((directory / "diagnostics.json").read_text())
    return config, design, draws, diagnostic


def summaries(values):
    import arviz_stats as az

    return {"mean": values.mean(axis=(0, 1)), "sd": values.reshape(-1, values.shape[-1]).std(axis=0, ddof=1),
            "quantiles": np.quantile(values, [.05, .5, .95], axis=(0, 1)),
            "mean_mcse": np.asarray(az.mcse(values, method="mean", chain_axis=0, draw_axis=1)),
            "quantile_mcse": np.stack([np.asarray(az.mcse(values, method="quantile", prob=q,
                                                          chain_axis=0, draw_axis=1)) for q in (.05, .5, .95)])}


def compare(base, wider, names):
    b, w = summaries(base), summaries(wider)
    rows = {}
    for i, name in enumerate(names):
        delta = float(w["mean"][i] - b["mean"][i])
        mcse = float(np.hypot(b["mean_mcse"][i], w["mean_mcse"][i]))
        sd = float(b["sd"][i])
        width = float(b["quantiles"][2, i] - b["quantiles"][0, i])
        quantiles = []
        for j, q in enumerate((.05, .5, .95)):
            shift = float(w["quantiles"][j, i] - b["quantiles"][j, i])
            error = float(np.hypot(b["quantile_mcse"][j, i], w["quantile_mcse"][j, i]))
            quantiles.append({"probability": q, "baseline": float(b["quantiles"][j, i]),
                              "wider": float(w["quantiles"][j, i]), "difference": shift,
                              "combined_mcse": error, "difference_mc95": [shift-Z95*error, shift+Z95*error],
                              "absolute_difference_upper_mc95_over_baseline_width90": (abs(shift)+Z95*error)/width})
        rows[name] = {"baseline_mean": float(b["mean"][i]), "wider_mean": float(w["mean"][i]),
                      "baseline_sd": sd, "wider_sd": float(w["sd"][i]),
                      "mean_difference": delta, "combined_mean_mcse": mcse,
                      "mean_difference_mc95": [delta-Z95*mcse, delta+Z95*mcse],
                      "mean_difference_over_baseline_sd": delta/sd,
                      "absolute_mean_difference_upper_mc95_over_baseline_sd": (abs(delta)+Z95*mcse)/sd,
                      "width90_ratio": float((w["quantiles"][2, i]-w["quantiles"][0, i])/width),
                      "quantiles": quantiles,
                      "practical_mean_gate_passed": (abs(delta)+Z95*mcse)/sd <= PRACTICAL_MEAN_FRACTION_SD,
                      "practical_quantile_gates_passed": all(
                          q["absolute_difference_upper_mc95_over_baseline_width90"] <= PRACTICAL_QUANTILE_FRACTION_WIDTH
                          for q in quantiles)}
    return rows


def study(args):
    if __package__:
        from .audit_heston_pilot_posterior_precision import audit_reference
    else:
        from audit_heston_pilot_posterior_precision import audit_reference

    bcfg, bdesign, bdraws, bdiag = load_reference(args.baseline, args.design)
    wcfg, wdesign, wdraws, wdiag = load_reference(args.variant, args.variant_design)
    for key in ("records", "products", "short_mask"):
        if canonical_digest(bdesign[key]) != canonical_digest(wdesign[key]):
            raise ValueError(f"kappa sensitivity changed {key}")
    if bcfg["arm"] != "full" or wcfg["arm"] != "full":
        raise ValueError("this sensitivity is restricted to the full 54-quote panel")
    for key in ("names", "infer_discrepancy", "spread_multiplier", "tau_max", "quadrature_ladder"):
        if bcfg[key] != wcfg[key]:
            raise ValueError(f"non-kappa reference setting changed: {key}")
    expected = np.asarray(bcfg["bounds"]).copy()
    if expected[1, 1] != 10. or expected[3, 1] != 3.:
        raise ValueError("baseline must have kappa<=10 and vol_of_vol<=3")
    expected[1, 1] = 20.
    if not np.array_equal(expected, wcfg["bounds"]):
        raise ValueError("only kappa upper support may change in this study")
    product_names = list(bdesign["products"])
    bproducts = np.stack([expected_variance(bdraws, bdesign["products"][name]) for name in product_names], axis=-1)
    wproducts = np.stack([expected_variance(wdraws, wdesign["products"][name]) for name in product_names], axis=-1)
    comparison = compare(np.concatenate([bdraws, bproducts], axis=-1),
                         np.concatenate([wdraws, wproducts], axis=-1), bcfg["names"] + product_names)
    kappa = wdraws[..., wcfg["names"].index("kappa")]
    indicator = kappa > 10.
    kdiag = wdiag["parameters"]["parameters[kappa]"]
    tail = {"threshold": 10., "observed_probability": float(indicator.mean()),
            "exceedance_count": int(indicator.sum()), "correlated_draws": int(indicator.size),
            "chain_exceedance_counts": indicator.sum(axis=1).tolist(),
            "kappa_quantiles_50_95_99_999": np.quantile(kappa, [.5, .95, .99, .999]).tolist(),
            "maximum_sampled_kappa": float(kappa.max()),
            "kappa_tail_ess": kdiag["ess_tail"],
            "tail_probability_mcse": None,
            "uncertainty_scope": "constant exceedance indicator cannot estimate its own Markov-chain MCSE; zero draws do not prove zero tail mass"}
    if indicator.any() and not indicator.all():
        import arviz_stats as az
        tail["tail_probability_mcse"] = float(az.mcse(indicator.astype(float), method="mean", chain_axis=0, draw_axis=1))
        tail["uncertainty_scope"] = "chain-aware indicator-mean MCSE; not a guarantee of remote-mode exploration"
    panel = json.loads(args.numerical_panel.read_text())
    corner = json.loads(args.corner_json.read_text())
    local = audit_reference(args.variant, wdesign)
    all_comparisons_pass = all(row["practical_mean_gate_passed"] and row["practical_quantile_gates_passed"]
                               for row in comparison.values())
    diagnostics_pass = bool(bdiag["mcmc_diagnostics_passed"] and wdiag["mcmc_diagnostics_passed"])
    production_corner_pass = bool(corner["production_gate_passed"])
    report = {"schema": "heston_kappa_upper_prior_sensitivity_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
              "baseline": {"path": str(args.baseline.resolve()), "design_sha256": digest(args.design),
                           "chains_sha256": digest(args.baseline/"chains.npz"), "bounds": bcfg["bounds"],
                           "diagnostics": bdiag},
              "wider": {"path": str(args.variant.resolve()), "design_sha256": digest(args.variant_design),
                        "chains_sha256": digest(args.variant/"chains.npz"), "bounds": wcfg["bounds"],
                        "diagnostics": wdiag},
              "identical_quote_records_sha256": canonical_digest(bdesign["records"]),
              "comparison": comparison, "kappa_above_old_bound": tail,
              "numerical_panel": {"sha256": digest(args.numerical_panel),
                  "all_points_priced": panel["all_points_priced"], "failed_point_ids": panel["failed_point_ids"],
                  "max_price_shift_measurement_sigma": panel["max_abs_price_shift_over_measurement_sigma"],
                  "failures": [point for point in panel["points"] if not point["priced"]]},
              "failed_corner_followup": corner, "posterior_pointwise_precision": local,
              "comparison_gates": {"absolute_mean_difference_upper_mc95_baseline_sd": PRACTICAL_MEAN_FRACTION_SD,
                  "absolute_quantile_difference_upper_mc95_baseline_width90": PRACTICAL_QUANTILE_FRACTION_WIDTH,
                  "scope": "predeclared practical sensitivity thresholds, including independent-run Monte Carlo uncertainty"},
              "all_parameter_product_comparisons_pass": all_comparisons_pass,
              "both_mcmc_diagnostics_pass": diagnostics_pass,
              "conditional_identity": "With every other target component fixed, posterior(kappa<=10) under U(.1,20) equals the U(.1,10) posterior exactly; total-variation distance equals the wider posterior probability kappa>10.",
              "scope": "bounded full-panel prior sensitivity and pointwise numerical checks; does not validate unseen markets or establish global mode coverage"}
    report["recommendation"] = ("retain kappa<=10 for the new full-panel bank; no material sensitivity detected"
        if all_comparisons_pass and diagnostics_pass and production_corner_pass
            and local["pointwise_gates_passed"] and not indicator.any()
        else "review sensitivity evidence before freezing the new bank prior")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--variant-design", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--variant", type=Path)
    parser.add_argument("--numerical-panel", type=Path)
    parser.add_argument("--corner-json", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--corner-only", action="store_true")
    args = parser.parse_args()
    if args.corner_only:
        result = corner_check(args.variant_design)
        args.corner_json.parent.mkdir(parents=True, exist_ok=True)
        args.corner_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("production_max_difference_measurement_sigma", "production_gate_passed")}))
    else:
        if any(value is None for value in (args.baseline, args.variant, args.numerical_panel, args.report)):
            parser.error("comparison requires --baseline, --variant, --numerical-panel and --report")
        result = study(args)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in (
            "both_mcmc_diagnostics_pass", "all_parameter_product_comparisons_pass", "recommendation")}))


if __name__ == "__main__":
    main()
