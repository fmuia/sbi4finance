"""Post-pilot comparison on common actual quotes excluded from both layouts.

This is descriptive development evidence chosen after the pilot, conditional
on common point-estimated carry, primary ATM-based selection and held-out
market-vega scales. It is not a population calibration or untouched acceptance
test, and does not select a winning layout.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from sbi4finance.calibration.european_quotes import parse_spxw_snapshot
from sbi4finance.calibration.heston_pilot import DEFAULT_LADDER, HestonPilot
from scripts.run_heston_observation_pilot import black_iv, write_json


TARGETS = (-3.5, -2.5, -.75, .25, .75)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_common_quotes(snapshot, primary, symmetric):
    """Select distinct nearest actual quotes, excluding both fitting panels."""
    if snapshot["raw_sha256"] != primary["source_sha256"] or symmetric["source_sha256"] != primary["source_sha256"]:
        raise ValueError("raw snapshot and both layouts must have the same source identity")
    if primary["carry"] != symmetric["carry"] or primary["expiries"] != symmetric["expiries"]:
        raise ValueError("both layouts must condition on identical carry and expiries")
    if snapshot["valuation_time_utc"] != primary["snapshot_metadata"]["valuation_time_utc"]:
        raise ValueError("held-out raw quote clock differs from the fitting design")
    excluded = {q["symbol"] for d in (primary, symmetric) for q in d["selected_quotes"]}
    excluded.update(symbol for d in (primary, symmetric) for carry in d["carry"].values() for symbol in carry["anchor_symbols"])
    selected = []
    for expiry in primary["expiries"]:
        carry = primary["carry"][expiry]
        f, d, t = (carry[key] for key in ("forward", "discount", "maturity"))
        atm_values = {q["atm_iv"] for q in primary["selected_quotes"] if q["expiration"] == expiry}
        if len(atm_values) != 1:
            raise ValueError("exactly one frozen primary ATM IV required per expiry")
        atm_iv = next(iter(atm_values))
        candidates = []
        for quote in snapshot["quotes"]:
            if quote["expiration"] != expiry or quote["symbol"] in excluded:
                continue
            if quote["option_type"] != ("put" if quote["strike"] <= f else "call") or quote["half_spread"] / quote["mid"] > .25:
                continue
            try:
                iv = black_iv(quote["mid"], f, quote["strike"], d, t, quote["option_type"])
            except ValueError:
                continue
            z = np.log(quote["strike"] / f) / (atm_iv * np.sqrt(t))
            candidates.append(dict(quote, forward=f, discount=d, market_iv=iv, atm_iv=atm_iv, z=float(z)))
        used = set()
        for target in TARGETS:
            pool = [q for q in candidates if q["symbol"] not in used]
            if not pool:
                raise ValueError(f"no disjoint actual quote for {expiry}, target {target}")
            quote = min(pool, key=lambda q: (abs(q["z"] - target), q["symbol"]))
            used.add(quote["symbol"])
            selected.append(dict(quote, target_z=target, target_supported=bool(abs(quote["z"] - target) <= .35)))
    keys = ("forward", "strike", "discount", "maturity", "option_type", "mid", "half_spread", "market_iv")
    records = {key: [q[key] for q in selected] for key in keys}
    records["market_vega"] = HestonPilot(records).market_vega.tolist()
    return {"schema": "heston_common_disjoint_heldout_quotes_v1", "source_sha256": snapshot["raw_sha256"],
            "valuation_time_utc": snapshot["valuation_time_utc"], "targets_z": TARGETS,
            "expiries": primary["expiries"], "selected_quotes": selected, "records": records,
            "excluded_fit_and_anchor_symbol_count": len(excluded),
            "all_targets_supported": all(q["target_supported"] for q in selected),
            "selection": "Nearest distinct actual positive noncrossed OTM quote, halfspread/mid<=.25; primary frozen ATM-IV coordinates; exclude union of all carry anchors and both selected fitting layouts. No interpolation."}


def mixture_interval(means, sigmas, probabilities=(.05, .95)):
    """Gaussian-mixture quantiles conditional on the selected posterior draws."""
    means, sigmas = np.asarray(means), np.asarray(sigmas)
    if means.shape != sigmas.shape or means.ndim != 2 or not np.isfinite(means).all() or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError("finite draw-by-quote means and positive matching sigmas required")
    output = np.empty((len(probabilities), means.shape[1]))
    for column in range(means.shape[1]):
        mu, sd = means[:, column], sigmas[:, column]
        lower, upper = float(np.min(mu - 12 * sd)), float(np.max(mu + 12 * sd))
        for row, p in enumerate(probabilities):
            output[row, column] = brentq(lambda x: float(np.mean(ndtr((x - mu) / sd)) - p), lower, upper)
    return output


def score_reference(directory, expected_design_hash, heldout, sample_count=256):
    root = Path(directory)
    if not (root / "chains.npz").exists() or not (root / "diagnostics.json").exists():
        return {"status": "incomplete_reference", "reference_path": str(root.resolve())}
    config = json.loads((root / "run.json").read_text())
    if config["design_sha256"] != expected_design_hash:
        raise ValueError("reference layout hash differs from comparison input")
    payload_hash = digest(root / "chains.npz")
    with np.load(root / "chains.npz", allow_pickle=False) as archive:
        draws = archive["parameters"]
    bounds = np.asarray(config["bounds"])
    if (draws.shape != (config["settings"]["chains"], config["settings"]["draws"], len(config["names"]))
            or not np.isfinite(draws).all() or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
        raise ValueError("reference draw shape, finiteness or support mismatch")
    if sample_count < draws.shape[0] or sample_count % draws.shape[0]:
        raise ValueError("requested posterior sample count must divide equally among chains")
    positions = np.unique(np.linspace(0, draws.shape[1] - 1, min(draws.shape[1], sample_count // draws.shape[0]), dtype=int))
    chosen = draws[:, positions].reshape(-1, draws.shape[-1])
    pilot = HestonPilot(heldout["records"], bounds=bounds[:5], infer_discrepancy=config["infer_discrepancy"],
                        tau_max=config["tau_max"], spread_multiplier=config["spread_multiplier"],
                        quadrature_ladder=config.get("quadrature_ladder", DEFAULT_LADDER))
    if pilot.names != config["names"]:
        raise ValueError("reference parameter names differ from prediction model")
    prices = np.array([pilot.prices(theta) for theta in chosen])
    sigmas = np.array([pilot.observation_sigma(theta) for theta in chosen])
    fair = np.quantile(prices, [.05, .5, .95], axis=0)
    predictive = mixture_interval(prices, sigmas)
    observed, spread = (np.asarray(heldout["records"][key]) for key in ("mid", "half_spread"))
    residual = observed - fair[1]
    in_predictive = (observed >= predictive[0]) & (observed <= predictive[1])
    early = np.array([q["expiration"] in heldout["expiries"][:3] for q in heldout["selected_quotes"]])
    metrics = {}
    masks = {"all": np.ones(len(observed), dtype=bool), "early": early, "later": ~early}
    masks.update({expiry: np.array([q["expiration"] == expiry for q in heldout["selected_quotes"]]) for expiry in heldout["expiries"]})
    for name, mask in masks.items():
        metrics[name] = {"quote_count": int(np.sum(mask)), "rms_residual_in_raw_halfspreads": float(np.sqrt(np.mean((residual[mask] / spread[mask])**2))),
                         "mae_residual_in_raw_halfspreads": float(np.mean(abs(residual[mask] / spread[mask]))),
                         "rms_residual_in_index_points": float(np.sqrt(np.mean(residual[mask]**2))),
                         "mae_residual_in_index_points": float(np.mean(abs(residual[mask]))),
                         "conditional_predictive_90_interval_fraction": float(np.mean(in_predictive[mask]))}
    if digest(root / "chains.npz") != payload_hash:
        raise RuntimeError("reference payload changed during held-out prediction")
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    return {"status": "scored", "reference_path": str(root.resolve()), "arm": config["arm"],
            "reference_mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"],
            "run_sha256": digest(root / "run.json"), "chain_sha256": payload_hash,
            "names": pilot.names, "bounds": pilot.bounds.tolist(), "spread_multiplier": config["spread_multiplier"],
            "tau_max": config["tau_max"], "posterior_draws_priced": len(chosen),
            "posterior_draw_indices_per_chain": positions.tolist(), "metrics": metrics,
            "fair_price_quantiles_05_50_95": fair.tolist(), "conditional_predictive_quantiles_05_95": predictive.tolist(),
            "residual_in_raw_halfspreads": (residual / spread).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--primary-design", type=Path, required=True)
    parser.add_argument("--symmetric-design", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True, help="Parent directory of the four completed reference runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    primary, symmetric = (json.loads(p.read_text()) for p in (args.primary_design, args.symmetric_design))
    snapshot = parse_spxw_snapshot(args.raw.read_bytes(), timestamp_zone="UTC",
                                  valuation_time=datetime.fromisoformat(primary["snapshot_metadata"]["valuation_time_utc"]))
    heldout = select_common_quotes(snapshot, primary, symmetric)
    args.out.mkdir(parents=True, exist_ok=True)
    design_path = args.out / "heldout_design.json"
    write_json(design_path, heldout)
    report = {"schema": "heston_common_heldout_development_comparison_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": __doc__, "scientific_acceptance": False, "winner_selected": False,
              "selection_predeclared_before_pilot": False,
              "predictive_scope": "Pointwise Gaussian-mixture intervals conditional on selected posterior draws, fixed carry, fixed primary ATM design and fixed vegas obtained from held-out midpoint IVs. Fractions over these selected quotes are descriptive; not population coverage.",
              "heldout_design_sha256": digest(design_path), "raw_sha256": snapshot["raw_sha256"],
              "primary_design_sha256": digest(args.primary_design), "symmetric_design_sha256": digest(args.symmetric_design),
              "script_sha256": digest(__file__), "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("numpy", "scipy")}},
              "quote_count": len(heldout["selected_quotes"]), "all_targets_supported": heldout["all_targets_supported"],
              "cases": {}}
    started = time.perf_counter()
    cases = [("put_heavy_full", "reference-full-v2", args.primary_design),
             ("put_heavy_short", "reference-short-v2", args.primary_design),
             ("symmetric_full", "reference-symmetric-full", args.symmetric_design),
             ("symmetric_short", "reference-symmetric-short", args.symmetric_design)]
    for name, directory, fitting_design in cases:
        report["cases"][name] = score_reference(args.references / directory, digest(fitting_design), heldout)
        print(json.dumps({"case": name, **{key: value for key, value in report["cases"][name].items() if key in ("status", "metrics", "reference_mcmc_diagnostics_passed")}}), flush=True)
    report["wall_seconds"] = time.perf_counter() - started
    write_json(args.out / "heldout.json", report)


if __name__ == "__main__":
    main()
