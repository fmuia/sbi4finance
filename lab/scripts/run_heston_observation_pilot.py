"""Frozen European quote-design and deterministic-reference Heston pilot.

Run from lab/. This study never publishes participant assets or reuses the
63-contract task identity. Carry anchors remain available in both arms;
withholding later *smile* quotes is conditional on that common carry panel.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration.european_quotes import parse_spxw_snapshot
from sbi4finance.calibration.heston_pilot import DEFAULT_LADDER, HestonPilot

EXPIRIES = ["2026-11-30", "2026-12-31", "2027-01-29", "2027-03-31", "2027-06-30", "2027-09-30"]
TARGETS = {"symmetric": [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2],
           "put_heavy": [-4, -3, -2, -1.5, -1, -.5, 0, .5, 1]}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def black_iv(price, forward, strike, discount, maturity, kind):
    def residual(iv):
        std = iv * np.sqrt(maturity)
        d1 = np.log(forward / strike) / std + .5 * std
        call = discount * (forward * ndtr(d1) - strike * ndtr(d1 - std))
        return call - (discount * (forward - strike) if kind == "put" else 0) - price
    return brentq(residual, .001, 4.)


def carry_anchors(quotes, spot):
    """Disjoint deterministic every-third-strike parity anchors, with WLS SE.

    Quote half-spreads are assumed Gaussian SDs only for this conditional WLS
    covariance. They are not independent empirical sampling-error estimates.
    """
    pairs = {}
    for quote in quotes:
        if abs(quote["strike"] / spot - 1) <= .05:
            pairs.setdefault(quote["strike"], {})[quote["option_type"]] = quote
    strikes = sorted(k for k, pair in pairs.items() if len(pair) == 2)[::3]
    if len(strikes) < 15:
        raise ValueError("at least 15 disjoint parity anchor pairs required")
    calls, puts = [[pairs[k][kind] for k in strikes] for kind in ("call", "put")]
    y = np.array([c["mid"] - p["mid"] for c, p in zip(calls, puts)])
    sd = np.array([np.hypot(c["half_spread"], p["half_spread"]) for c, p in zip(calls, puts)])
    x = np.column_stack((np.ones(len(strikes)), spot - np.array(strikes)))
    coef = np.linalg.lstsq(x / sd[:, None], y / sd, rcond=None)[0]
    cov = np.linalg.inv((x / sd[:, None]).T @ (x / sd[:, None]))
    a, discount = coef
    forward = spot + a / discount
    maturity = calls[0]["maturity"]
    if not np.exp(-.1 * maturity) < discount < np.exp(.02 * maturity):
        raise ValueError("parity discount outside predeclared plausible rate interval [-2%,10%]")
    gradient = np.array([1 / discount, -a / discount**2])
    hsum = np.array([c["half_spread"] + p["half_spread"] for c, p in zip(calls, puts)])
    return {"forward": float(forward), "discount": float(discount),
            "rate": float(-np.log(discount) / maturity), "maturity": maturity,
            "forward_se_conditional": float(np.sqrt(gradient @ cov @ gradient)),
            "discount_se_conditional": float(np.sqrt(cov[1, 1])),
            "intercept_discount_covariance": cov.tolist(), "parity_intercept": float(a),
            "anchor_symbols": [q["symbol"] for q in calls + puts],
            "anchor_quotes": calls + puts,
            "anchor_pairs": len(strikes),
            "parity_max_residual_over_combined_halfspread": float(np.max(abs(y - x @ coef) / hsum)),
            "parity_rms_residual_over_combined_halfspread": float(np.sqrt(np.mean(((y - x @ coef) / hsum)**2)))}


def build_design(snapshot, layout):
    spot = float(snapshot["underlying_price"])
    selected, carry = [], {}
    for expiry in EXPIRIES:
        quotes = [q for q in snapshot["quotes"] if q["expiration"] == expiry]
        anchor = carry_anchors(quotes, spot)
        carry[expiry] = anchor
        forward, discount, maturity = [anchor[k] for k in ("forward", "discount", "maturity")]
        reserved = set(anchor["anchor_symbols"])
        candidates = []
        for quote in quotes:
            if quote["symbol"] in reserved:
                continue
            if quote["option_type"] != ("put" if quote["strike"] <= forward else "call"):
                continue
            if quote["half_spread"] / quote["mid"] > .25:
                continue
            try:
                iv = black_iv(quote["mid"], forward, quote["strike"], discount, maturity, quote["option_type"])
            except ValueError:
                continue
            candidates.append(dict(quote, forward=forward, discount=discount, market_iv=iv))
        atm = min(candidates, key=lambda q: abs(np.log(q["strike"] / forward)))
        atm_iv = atm["market_iv"]
        used = set()
        for target in TARGETS[layout]:
            pool = [q for q in candidates if q["symbol"] not in used]
            quote = min(pool, key=lambda q: abs(np.log(q["strike"] / forward) / (atm_iv * np.sqrt(maturity)) - target))
            z = np.log(quote["strike"] / forward) / (atm_iv * np.sqrt(maturity))
            if abs(z - target) > .35 and layout == "put_heavy":
                raise ValueError(f"unsupported {layout} target {target} at {expiry}: nearest z={z}")
            used.add(quote["symbol"])
            selected.append(dict(quote, z=float(z), target_z=target, atm_iv=atm_iv,
                                 target_supported=bool(abs(z - target) <= .35)))
    keys = ("forward", "strike", "discount", "maturity", "option_type", "mid", "half_spread", "market_iv")
    records = {key: [q[key] for q in selected] for key in keys}
    # Frozen observed-vega convention must be identical in reference and neural simulation.
    pilot = HestonPilot(records, infer_discrepancy=True)
    records["market_vega"] = pilot.market_vega.tolist()
    return {"format": "heston_spxw_conditional_design_v1", "layout": layout,
            "source_sha256": snapshot["raw_sha256"], "snapshot_metadata": {k: v for k, v in snapshot.items() if k != "quotes"},
            "expiries": EXPIRIES, "targets_z": TARGETS[layout], "records": records,
            "selected_quotes": selected, "carry": carry,
            "short_mask": [q["expiration"] in EXPIRIES[:3] for q in selected],
            "scope": "Actual disjoint OTM smile quotes; both arms condition on common parity anchors at every expiry and frozen observed vegas. Later smile prices, not all later option information, are withheld.",
            "observation_model": "independent Normal(mid | Heston_price, sqrt((c*halfspread)^2+(tau*frozen_market_vega)^2)); c a convention, tau uniform IV decimal scale; not an estimated market error law",
            "products": {"control": [0., carry[EXPIRIES[0]]["maturity"]],
                         "affected": [carry[EXPIRIES[2]]["maturity"], carry[EXPIRIES[-1]]["maturity"]]},
            "priors": {"names": pilot.names, "bounds": pilot.bounds.tolist()},
            "design_filters": "SPXW only, future regular PM expiry, positive noncrossed bid/ask, OTM, halfspread/mid<=25%, nearest distinct actual quote; target_supported flags distance<=0.35 standardized log-moneyness, required for put-heavy candidate. Symmetric unavailable targets remain visibly flagged and use nearest actual quotes, without interpolation. Every third paired near-spot strike reserved for carry"}


def expected_variance(parameters, interval):
    values = np.asarray(parameters)
    v0, kappa, theta = (values[..., i] for i in range(3))
    start, end = interval
    weight = np.exp(-kappa * start) * (-np.expm1(-kappa * (end - start))) / (kappa * (end - start))
    return theta + (v0 - theta) * weight


def posterior_report(pilot, draws, design):
    flat = draws.reshape(-1, draws.shape[-1])
    chosen = flat[np.linspace(0, len(flat) - 1, min(1000, len(flat))).astype(int)]
    prices = np.array([pilot.prices(row) for row in chosen])
    mid, halfspread = (np.asarray(design["records"][key]) for key in ("mid", "half_spread"))
    median = np.median(prices, axis=0)
    short = np.asarray(design["short_mask"])
    price_interval = np.quantile(prices, [.05, .5, .95], axis=0)
    rng = np.random.default_rng(0x50524544)
    predictive = prices + np.array([pilot.observation_sigma(row) for row in chosen]) * rng.standard_normal(prices.shape)
    pred_interval = np.quantile(predictive, [.05, .95], axis=0)
    result = {"parameter_quantiles_05_50_95": {name: np.quantile(flat[:, i], [.05, .5, .95]).tolist() for i, name in enumerate(pilot.names)},
              "fair_price_quantiles_05_50_95": price_interval.tolist(), "predictive_quantiles_05_95": pred_interval.tolist(),
              "residual_over_halfspread": ((mid - median) / halfspread).tolist(),
              "products": {key: {"interval": interval, "expected_variance_strike_quantiles_05_50_95": np.quantile(expected_variance(flat, interval), [.05, .5, .95]).tolist()} for key, interval in design["products"].items()}}
    for label, mask in (("short", short), ("later", ~short), ("all", np.ones(len(short), bool))):
        result[label] = {"n": int(mask.sum()), "rms_residual_halfspreads": float(np.sqrt(np.mean(((mid - median) / halfspread)[mask]**2))),
                         "max_absolute_residual_halfspreads": float(np.max(abs((mid - median) / halfspread)[mask])),
                         "predictive_90_interval_fraction": float(np.mean(((mid >= pred_interval[0]) & (mid <= pred_interval[1]))[mask]))}
    return result


def finalize_reference(out, design, *, design_sha256):
    """Analyze a completed retained trace without rerunning the sampler."""
    import xarray as xr
    from sbi4finance.calibration import reference
    config = json.loads((out / "run.json").read_text())
    if design_sha256 != config["design_sha256"]:
        raise ValueError("design SHA-256 differs from the sampled reference target")
    pilot = HestonPilot(design["records"], infer_discrepancy=config["infer_discrepancy"],
                        spread_multiplier=config["spread_multiplier"], tau_max=config["tau_max"],
                        bounds=np.asarray(config["bounds"])[:5],
                        quadrature_ladder=config.get("quadrature_ladder", DEFAULT_LADDER))
    with np.load(out / "chains.npz") as values:
        draws = values["parameters"]
        stats = {key.removeprefix("sample_stats_"): (("chain", "draw"), values[key])
                 for key in values.files if key.startswith("sample_stats_")}
    expected = (config["settings"]["chains"], config["settings"]["draws"], len(pilot.names))
    if (draws.shape != expected or config["names"] != pilot.names
            or not np.isfinite(draws).all() or np.any(draws < pilot.bounds[:, 0])
            or np.any(draws > pilot.bounds[:, 1])
            or any(value[1].shape != expected[:2] for value in stats.values())):
        raise ValueError("reference draw dimensions, names, support or sample statistics differ from target")
    if not (out / "trace.json").exists():
        write_json(out / "trace.json", {"dimensions": {"parameters": ["chain", "draw", "parameter"],
                   **{"sample_stats_" + key: ["chain", "draw"] for key in stats}},
                   "coordinates": {"chain": list(range(expected[0])), "draw": list(range(expected[1])), "parameter": pilot.names},
                   "payload": "chains.npz", "warmup_discarded": True, "sampler_resumable": False})
    posterior = xr.Dataset({"parameters": (("chain", "draw", "parameter"), draws)},
                          coords={"parameter": pilot.names})
    idata = xr.DataTree.from_dict({"posterior": posterior, "sample_stats": xr.Dataset(stats)})
    diagnostics = reference.chain_diagnostics(idata)
    product_names = list(design["products"])
    product_draws = np.stack([expected_variance(draws, design["products"][key]) for key in product_names], axis=-1)
    product_tree = xr.DataTree.from_dict({"posterior": xr.Dataset(
        {"parameters": (("chain", "draw", "parameter"), product_draws)}, coords={"parameter": product_names})})
    diagnostics["products"] = reference.chain_diagnostics(product_tree)
    diagnostics["parameter_mcmc_diagnostics_passed"] = diagnostics["mcmc_diagnostics_passed"]
    diagnostics["mcmc_diagnostics_passed"] = bool(diagnostics["mcmc_diagnostics_passed"] and diagnostics["products"]["mcmc_diagnostics_passed"])
    diagnostics["chain_payload_sha256"] = hashlib.sha256((out / "chains.npz").read_bytes()).hexdigest()
    diagnostics["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out / "diagnostics.json", diagnostics)
    write_json(out / "posterior.json", posterior_report(pilot, draws, design))
    print(json.dumps({"out": str(out), **diagnostics}))


def run_reference(args):
    from sbi4finance.calibration import reference
    design = json.loads(args.design.read_text())
    bounds = np.asarray(design["priors"]["bounds"], dtype=float)[:5].copy()
    if args.vol_of_vol_max is not None:
        bounds[3, 1] = args.vol_of_vol_max
    ladder = DEFAULT_LADDER + (((16384, 9600.),) if args.extended_quadrature else ())
    pilot = HestonPilot(design["records"], infer_discrepancy=not args.spread_only,
                        spread_multiplier=args.spread_multiplier, tau_max=args.tau_max, bounds=bounds,
                        quadrature_ladder=ladder)
    mask = np.asarray(design["short_mask"]) if args.arm == "short" else None
    likelihood = lambda point: pilot.log_likelihood(point, mask=mask)
    args.out.mkdir(parents=True, exist_ok=False)
    config = {"format": "heston_spxw_reference_v1", "design_sha256": hashlib.sha256(args.design.read_bytes()).hexdigest(),
              "arm": args.arm, "names": pilot.names, "bounds": pilot.bounds.tolist(),
              "spread_multiplier": args.spread_multiplier, "infer_discrepancy": not args.spread_only, "tau_max": args.tau_max,
              "quadrature_ladder": ladder,
              "settings": {k: getattr(args, k) for k in ("seed", "draws", "tune", "chains", "cores", "fit_budget")},
              "versions": {"python": platform.python_version(), **{k: metadata.version(k) for k in ("pymc", "pytensor", "arviz-stats", "numpy", "scipy")}},
              "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), Path(sys.modules[HestonPilot.__module__].__file__).resolve())},
              "sampler": "PyMC DEMetropolisZ, deterministic checked Fourier likelihood; physical independent uniform priors; no gradients",
              "scientific_acceptance": False, "scope": design["scope"]}
    write_json(args.out / "run.json", config)
    started = time.perf_counter()
    initialization = reference.data_fit_initialization(likelihood, pilot.bounds, args.chains, args.seed, fit_starts=4, fit_budget=args.fit_budget)
    write_json(args.out / "initialization.json", initialization)
    idata = reference.sample_deterministic_reference(likelihood, pilot.bounds, pilot.names,
                draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
                seed=args.seed, progressbar=False, initialization=initialization)
    draws = idata["posterior"]["parameters"].transpose("chain", "draw", "parameter").values
    arrays = {"parameters": draws}
    for name, value in idata["sample_stats"].data_vars.items():
        if set(value.dims) == {"chain", "draw"}:
            arrays["sample_stats_" + name] = value.transpose("chain", "draw").values
    np.savez_compressed(args.out / "chains.npz", **arrays)
    write_json(args.out / "trace.json", {"dimensions": {"parameters": ["chain", "draw", "parameter"],
                **{key: ["chain", "draw"] for key in arrays if key != "parameters"}},
                "coordinates": {"chain": list(range(args.chains)), "draw": list(range(args.draws)), "parameter": pilot.names},
                "payload": "chains.npz", "seconds_including_initialization": time.perf_counter() - started,
                "warmup_discarded": True, "sampler_resumable": False})
    finalize_reference(args.out, design, design_sha256=hashlib.sha256(args.design.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    design = sub.add_parser("design")
    design.add_argument("--raw", type=Path, required=True)
    design.add_argument("--out", type=Path, required=True)
    design.add_argument("--lag-minutes", type=float, default=15.)
    ref = sub.add_parser("reference")
    ref.add_argument("--design", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("--arm", choices=("full", "short"), default="full")
    ref.add_argument("--spread-only", action="store_true")
    ref.add_argument("--spread-multiplier", type=float, default=1.)
    ref.add_argument("--tau-max", type=float, default=.02)
    ref.add_argument("--vol-of-vol-max", type=float)
    ref.add_argument("--extended-quadrature", action="store_true")
    ref.add_argument("--draws", type=int, default=12000)
    ref.add_argument("--tune", type=int, default=6000)
    ref.add_argument("--chains", type=int, default=4)
    ref.add_argument("--cores", type=int, default=4)
    ref.add_argument("--seed", type=int, default=261109)
    ref.add_argument("--fit-budget", type=int, default=8000)
    summary = sub.add_parser("summarize")
    summary.add_argument("--design", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        finalize_reference(args.out, json.loads(args.design.read_text()),
                           design_sha256=hashlib.sha256(args.design.read_bytes()).hexdigest())
    elif args.command == "reference":
        run_reference(args)
    else:
        raw = args.raw.read_bytes()
        initial = parse_spxw_snapshot(raw, timestamp_zone="UTC")
        valuation = datetime.fromisoformat(initial["snapshot_timestamp_utc"]) - timedelta(minutes=args.lag_minutes)
        snapshot = parse_spxw_snapshot(raw, timestamp_zone="UTC", valuation_time=valuation)
        for layout in TARGETS:
            path = args.out / (layout + ".json")
            if path.exists():
                raise FileExistsError(path)
            write_json(path, build_design(snapshot, layout))
            print(path)


if __name__ == "__main__":
    main()
