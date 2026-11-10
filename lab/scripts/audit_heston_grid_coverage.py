"""Audit a fixed actual-quote E layout on the frozen SPXW snapshot.

Run from lab/. This reads the historical snapshot and its source/design
records, selects geometry and estimates parity carry only. It never prices a
model option, reads a bank, trains or samples a posterior. No download occurs.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np


TARGET_DAYS = (5, 10, 21, 42, 63, 126, 252)
TARGET_Z = (-5., -3.5, -2.25, -1.25, -.5, 0., .5, 1., 2.)
SIGMA_REF = .16
MAX_Z_DISTANCE = .35
MAX_RELATIVE_HALFSPREAD = .25
VALUATION_TIME = "2026-11-09T20:13:37+00:00"
EVIDENCE = Path("lab/evidence/pilot-results/2026-11-09/heston-observations")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_bound_raw(raw_path, provenance_path):
    raw = Path(raw_path).read_bytes()
    provenance = json.loads(Path(provenance_path).read_text())
    if (hashlib.sha256(raw).hexdigest() != provenance["raw_sha256"]
            or len(raw) != provenance["raw_size_bytes"]):
        raise ValueError("raw snapshot hash or byte count differs from provenance")
    return raw, provenance


def load_reader(path):
    """Reuse the standalone historical quote reader without importing pricing."""
    spec = importlib.util.spec_from_file_location("frozen_spxw_coverage_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_spxw_snapshot


def standardized_z(quote, forward):
    return float(np.log(quote["strike"] / forward) / (SIGMA_REF * np.sqrt(quote["maturity"])))


def nearest_expiries(quotes, target_days=TARGET_DAYS):
    """Nearest actual ACT/365F maturity to each nominal day/252 target.

    The nominal labels are not a change of clock. A duplicate nearest expiry
    remains a visible failure rather than being replaced by a further date.
    """
    expiries = {}
    for quote in quotes:
        old = expiries.setdefault(quote["expiration"], quote["maturity"])
        if old != quote["maturity"]:
            raise ValueError("one expiry has inconsistent exact maturities")
    if not expiries:
        raise ValueError("no eligible expiries")
    used, result = set(), []
    for day in target_days:
        target = day / 252.
        expiry = min(expiries, key=lambda e: (abs(expiries[e] - target), expiries[e], e))
        maturity = expiries[expiry]
        result.append({"nominal_business_days": day, "target_maturity": target,
            "expiration": expiry, "maturity": maturity,
            "calendar_days": maturity * 365.,
            "maturity_offset_calendar_days": (maturity - target) * 365.,
            "duplicate_nearest_expiry": expiry in used})
        used.add(expiry)
    return result


def parity_carry(quotes, spot, forbidden_symbols=()):
    """Same near-spot every-third-pair WLS convention as the historical pilot.

    These SDs/covariances condition on the working half-spread convention;
    they do not measure quote freshness or empirical carry sampling error.
    """
    forbidden = set(forbidden_symbols)
    pairs = {}
    for quote in quotes:
        if abs(quote["strike"] / spot - 1.) <= .05 and quote["symbol"] not in forbidden:
            pairs.setdefault(quote["strike"], {})[quote["option_type"]] = quote
    strikes = sorted(k for k, pair in pairs.items() if set(pair) == {"call", "put"})[::3]
    if len(strikes) < 15:
        raise ValueError(f"only {len(strikes)} disjoint every-third near-spot parity pairs; require 15")
    calls, puts = [[pairs[k][kind] for k in strikes] for kind in ("call", "put")]
    y = np.array([c["mid"] - p["mid"] for c, p in zip(calls, puts)])
    sd = np.array([np.hypot(c["half_spread"], p["half_spread"]) for c, p in zip(calls, puts)])
    x = np.column_stack((np.ones(len(strikes)), spot - np.asarray(strikes)))
    coef = np.linalg.lstsq(x / sd[:, None], y / sd, rcond=None)[0]
    covariance = np.linalg.inv((x / sd[:, None]).T @ (x / sd[:, None]))
    intercept, discount = coef
    forward = spot + intercept / discount
    maturity = calls[0]["maturity"]
    if not np.exp(-.1 * maturity) < discount < np.exp(.02 * maturity):
        raise ValueError("parity discount outside historical rate interval [-2%,10%]")
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("parity forward is not positive finite")
    gradient = np.array([1. / discount, -intercept / discount**2])
    combined_half = np.array([c["half_spread"] + p["half_spread"] for c, p in zip(calls, puts)])
    residual = (y - x @ coef) / combined_half
    return {"forward": float(forward), "discount": float(discount), "maturity": maturity,
        "rate": float(-np.log(discount) / maturity), "parity_intercept": float(intercept),
        "intercept_discount_covariance": covariance.tolist(),
        "forward_se_conditional": float(np.sqrt(gradient @ covariance @ gradient)),
        "discount_se_conditional": float(np.sqrt(covariance[1, 1])),
        "anchor_pairs": len(strikes), "anchor_symbols": [q["symbol"] for q in calls + puts],
        "parity_max_residual_over_combined_halfspread": float(np.max(abs(residual))),
        "parity_rms_residual_over_combined_halfspread": float(np.sqrt(np.mean(residual**2)))}


def compact_carry(record):
    return {key: value for key, value in record.items() if key != "anchor_quotes"}


def validate_preserved(snapshot, baseline, comparison, scored):
    """Bind old observations/anchors to the exact source before preserving carry."""
    by_symbol = {q["symbol"]: q for q in snapshot["quotes"]}
    for design in (baseline, comparison, scored):
        if design["source_sha256"] != snapshot["raw_sha256"]:
            raise ValueError("historical design belongs to another raw snapshot")
        for quote in design["selected_quotes"]:
            actual = by_symbol.get(quote["symbol"])
            if actual is None or any(quote[key] != actual[key] for key in
                                    ("strike", "maturity", "bid", "ask", "mid", "half_spread", "option_type")):
                raise ValueError("historical selected quote differs from frozen raw snapshot")
    if set(baseline["carry"]) != set(comparison["carry"]):
        raise ValueError("historical comparison has different common carry expiries")
    fitted = {q["symbol"] for d in (baseline, comparison) for q in d["selected_quotes"]}
    scored_symbols = {q["symbol"] for q in scored["selected_quotes"]}
    for expiry, carry in baseline["carry"].items():
        if carry != comparison["carry"][expiry]:
            raise ValueError("historical comparison has different carry estimates/anchors")
        if len(set(carry["anchor_symbols"])) != len(carry["anchor_symbols"]):
            raise ValueError("historical carry repeats an anchor symbol")
        if set(carry["anchor_symbols"]) & (fitted | scored_symbols):
            raise ValueError("historical carry overlaps fitted/scored quotes")
        for quote in carry["anchor_quotes"]:
            actual = by_symbol.get(quote["symbol"])
            if actual is None or any(quote[key] != actual[key] for key in
                                    ("expiration", "maturity", "strike", "mid", "half_spread")):
                raise ValueError("historical carry anchor differs from frozen raw snapshot")
        if set(carry["anchor_symbols"]) != {q["symbol"] for q in carry["anchor_quotes"]}:
            raise ValueError("historical anchor symbols/quotes differ")
        if not all(q["expiration"] == expiry for q in carry["anchor_quotes"]):
            raise ValueError("historical carry anchor expiry differs")
    return fitted, scored_symbols


def select_bucket(quotes, carry, reserved, scored_symbols, used_symbols, targets=TARGET_Z):
    """Choose nearest distinct actual OTM quotes; retain unsupported targets."""
    forward, discount = carry["forward"], carry["discount"]
    candidates, raw_otm, rejected = [], [], Counter()
    for quote in quotes:
        if quote["symbol"] in reserved:
            rejected["carry_anchor"] += 1
            continue
        if quote["symbol"] in scored_symbols:
            rejected["preserved_scoring_quote"] += 1
            continue
        kind = "put" if quote["strike"] <= forward else "call"
        if quote["option_type"] != kind:
            rejected["not_otm"] += 1
            continue
        candidate = dict(quote, z_fixed16=standardized_z(quote, forward))
        raw_otm.append(candidate)
        if quote["half_spread"] / quote["mid"] > MAX_RELATIVE_HALFSPREAD:
            rejected["halfspread_over_mid_exceeds_25pct"] += 1
            continue
        upper = discount * (quote["strike"] if kind == "put" else forward)
        if not 0 < quote["mid"] < upper:
            rejected["mid_outside_european_price_bounds"] += 1
            continue
        candidates.append(candidate)
    selected, target_reports = [], []
    for target in targets:
        target_kind = "put" if target <= 0 else "call"
        pool = [q for q in candidates if q["symbol"] not in used_symbols and q["option_type"] == target_kind]
        raw_kind = [q for q in raw_otm if q["option_type"] == target_kind]
        nearest_raw = (min(raw_kind, key=lambda q: (abs(q["z_fixed16"] - target), q["symbol"]))
                       if raw_kind else None)
        if not pool:
            target_reports.append({"target_z": target, "target_option_type": target_kind, "target_supported": False,
                                   "failure": "no_unused_quality_eligible_actual_quote"})
            continue
        quote = min(pool, key=lambda q: (abs(q["z_fixed16"] - target), q["symbol"]))
        distance = abs(quote["z_fixed16"] - target)
        supported = distance <= MAX_Z_DISTANCE
        used_symbols.add(quote["symbol"])
        selected.append(dict(quote, forward=forward, discount=discount,
                             target_z=target, target_distance=distance, target_supported=supported))
        target_reports.append({"target_z": target, "target_option_type": target_kind, "symbol": quote["symbol"],
            "realized_z": quote["z_fixed16"], "distance": distance,
            "half_spread": quote["half_spread"], "halfspread_over_mid": quote["half_spread"] / quote["mid"],
            "target_supported": supported,
            "nearest_before_spread_quality_filter": (None if nearest_raw is None else
                {"symbol": nearest_raw["symbol"], "z": nearest_raw["z_fixed16"],
                 "halfspread_over_mid": nearest_raw["half_spread"] / nearest_raw["mid"]})})
    return selected, {"parser_eligible_quote_count": len(quotes),
        "quality_eligible_otm_count": len(candidates), "selection_rejections": dict(rejected),
        "targets": target_reports}


def old_coverage(design):
    result = []
    for expiry in design["expiries"]:
        rows = [q for q in design["selected_quotes"] if q["expiration"] == expiry]
        carry = design["carry"][expiry]
        z = [standardized_z(q, carry["forward"]) for q in rows]
        result.append({"expiration": expiry, "maturity": rows[0]["maturity"],
            "calendar_days": rows[0]["maturity"] * 365., "n": len(rows),
            "minimum_z_fixed16": min(z), "maximum_z_fixed16": max(z),
            "nearest_atm_abs_z_fixed16": min(abs(x) for x in z)})
    return result


def build_candidate(snapshot, baseline, comparison, scored, *, last_call_target=None):
    if last_call_target is not None and last_call_target != 1.5:
        raise ValueError("the only declared revision is last_call_target=1.5")
    fitted, scored_symbols = validate_preserved(snapshot, baseline, comparison, scored)
    union = {expiry: compact_carry(record) for expiry, record in baseline["carry"].items()}
    origins = {expiry: {"kind": "preserved_historical_estimate_and_symbols",
                       "original_full_record_sha256": fingerprint(record)}
               for expiry, record in baseline["carry"].items()}
    buckets = nearest_expiries(snapshot["quotes"])
    selected, used = [], set()
    for index, bucket in enumerate(buckets):
        expiry = bucket["expiration"]
        quotes = [q for q in snapshot["quotes"] if q["expiration"] == expiry]
        if bucket["duplicate_nearest_expiry"]:
            bucket.update(targets=[], failure="duplicate_nearest_actual_expiry")
            continue
        if expiry not in union:
            try:
                union[expiry] = parity_carry(quotes, float(snapshot["underlying_price"]), fitted | scored_symbols)
                origins[expiry] = {"kind": "new_disjoint_every_third_near_spot_pair_wls"}
            except ValueError as error:
                bucket.update(targets=[], failure=str(error))
                continue
        reserved = {s for carry in union.values() for s in carry["anchor_symbols"]}
        targets = list(TARGET_Z)
        if index == len(buckets) - 1 and last_call_target is not None:
            targets[-1] = last_call_target
        rows, report = select_bucket(quotes, union[expiry], reserved, scored_symbols, used, targets=targets)
        if index == len(buckets) - 1 and last_call_target is not None:
            for quote in rows:
                if quote["target_z"] == last_call_target:
                    quote["original_E_target_z"] = TARGET_Z[-1]
                    quote["distance_to_original_E_target"] = abs(quote["z_fixed16"] - TARGET_Z[-1])
        selected.extend(rows)
        bucket.update(report)
    reserved = {s for carry in union.values() for s in carry["anchor_symbols"]}
    if reserved & ({q["symbol"] for q in selected} | fitted | scored_symbols):
        raise ValueError("union carry overlaps candidate/historical fitted/scored quotes")
    records = {key: [q[key] for q in selected] for key in
               ("forward", "strike", "discount", "maturity", "option_type", "mid", "half_spread")}
    candidate = {"format": "heston_spxw_actual_e_grid_coverage_v1",
        "source_sha256": snapshot["raw_sha256"],
        "layout": ("literal_E_targets_nearest_actual_spxw" if last_call_target is None
                   else "explicit_revision_last_expiry_call_target_1p5"),
        "valuation_time_utc": snapshot["valuation_time_utc"], "time_convention": snapshot["time_convention"],
        "reference_volatility": SIGMA_REF, "targets_z": list(TARGET_Z),
        "target_overrides": ([] if last_call_target is None else
            [{"expiration": buckets[-1]["expiration"], "original_target_z": TARGET_Z[-1],
              "revised_target_z": last_call_target,
              "reason": "Separately declared supported revision; original +2 target remains unsupported. No endpoint substitution."}]),
        "nominal_business_day_labels": list(TARGET_DAYS),
        "selected_quotes": selected, "records": records,
        "carry": {b["expiration"]: union[b["expiration"]] for b in buckets if b["expiration"] in union},
        "union_carry": union, "carry_origin": origins,
        "common_conditioning_scope": "Both old/new comparisons would condition on all historical plus new carry anchors. Old carry estimates/symbols are preserved exactly. New anchor symbols exclude old fitted/scored quotes; all anchors are excluded from candidate fitting/scoring.",
        "scope": "Coverage geometry only. Unsupported nearest actual quotes remain flagged. No pricing labels, model fit, training, posterior acceptance or final grid selection."}
    targets = [t for b in buckets for t in b["targets"]]
    desired_count = len(TARGET_DAYS) * len(TARGET_Z)
    existing_symbols = {q["symbol"] for q in baseline["selected_quotes"]}
    all_maturities = [q["maturity"] for q in snapshot["quotes"]]
    plus2_rows = [q for q in selected if q["target_z"] == 2. or q.get("original_E_target_z") == 2.]
    original_plus2_supported = (len(plus2_rows) == len(TARGET_DAYS)
                               and all(abs(q["z_fixed16"] - 2.) <= MAX_Z_DISTANCE for q in plus2_rows))
    summary = {"format": "heston_spxw_grid_coverage_summary_v1", "buckets": buckets,
        "desired_contract_count": desired_count, "selected_contract_count": len(selected),
        "supported_quote_target_count": sum(t["target_supported"] for t in targets),
        "unsupported_quote_target_count": sum(not t["target_supported"] for t in targets),
        "missing_quote_target_count": desired_count - len(selected),
        "unassessed_quote_target_count": desired_count - len(targets),
        "duplicate_selected_symbol_count": len(selected) - len({q["symbol"] for q in selected}),
        "complete_supported_quote_layout": len(targets) == desired_count and all(t["target_supported"] for t in targets),
        "snapshot_maximum_spxw_maturity": max(all_maturities),
        "one_year_pm_expiry_available": max(all_maturities) >= 1.,
        "original_one_year_and_plus2_coverage_criteria_satisfied": bool(
            selected and max(q["maturity"] for q in selected) >= 1. and original_plus2_supported),
        "one_year_endpoint_shortfall_calendar_days": max(0., (1. - max(all_maturities)) * 365.),
        "old54_coverage": old_coverage(baseline),
        "overlap_with_old54_contract_symbols": len({q["symbol"] for q in selected} & existing_symbols),
        "historical_carry_expiries_preserved": sorted(baseline["carry"]),
        "new_carry_expiries": sorted(set(union) - set(baseline["carry"])),
        "union_carry_expiries": sorted(union),
        "carry_disjoint_from_candidate_old_fitted_and_scored": True,
        "candidate_disjoint_from_preserved_scoring_quotes": True,
        "parser_exclusion_counts": snapshot["exclusion_counts"],
        "scientific_acceptance": False, "model_pricing_evaluations": 0,
        "target_overrides": candidate["target_overrides"],
        "scope": candidate["scope"]}
    return candidate, summary


def run(historical_root, out, raw_path=None, *, last_call_target=None):
    historical_root, out = Path(historical_root).resolve(), Path(out)
    source = historical_root / EVIDENCE / "source.json"
    provenance = json.loads(source.read_text())
    raw_path = Path(raw_path) if raw_path is not None else historical_root / provenance["raw_path_from_repository_root"]
    raw, provenance = load_bound_raw(raw_path, source)
    reader = historical_root / "lab/src/sbi4finance/calibration/european_quotes.py"
    parse = load_reader(reader)
    snapshot = parse(raw, timestamp_zone=provenance["timestamp_zone_assumption"],
                     trade_time_zone=provenance["trade_time_zone_assumption"],
                     valuation_time=datetime.fromisoformat(VALUATION_TIME))
    paths = {"old54": historical_root / EVIDENCE / "designs/put_heavy.json",
             "symmetric": historical_root / EVIDENCE / "designs/symmetric.json",
             "scoring": historical_root / EVIDENCE / "heldout_design.json"}
    baseline, comparison, scored = [json.loads(paths[k].read_text()) for k in ("old54", "symmetric", "scoring")]
    candidate, summary = build_candidate(snapshot, baseline, comparison, scored)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "candidate.json", candidate)
    write_json(out / "summary.json", summary)
    output_names = ["candidate.json", "summary.json"]
    if last_call_target is not None:
        revision, revision_summary = build_candidate(snapshot, baseline, comparison, scored,
                                                     last_call_target=last_call_target)
        revision_summary["revised_vs_original_records_identical"] = revision["records"] == candidate["records"]
        write_json(out / "candidate-supported-revision.json", revision)
        write_json(out / "summary-supported-revision.json", revision_summary)
        output_names += ["candidate-supported-revision.json", "summary-supported-revision.json"]
    write_json(out / "run.json", {"format": "heston_spxw_grid_coverage_run_v1",
        "historical_root": str(historical_root), "raw_path": str(raw_path),
        "raw_sha256": provenance["raw_sha256"], "raw_size_bytes": len(raw),
        "source_record_sha256": digest(source), "producer_source_sha256": digest(__file__),
        "historical_reader_source_sha256": digest(reader),
        "input_design_sha256": {key: digest(path) for key, path in paths.items()},
        "output_sha256": {name: digest(out / name) for name in output_names},
        "valuation_time_utc": VALUATION_TIME,
        "snapshot_timestamp_utc": snapshot["snapshot_timestamp_utc"],
        "source_record_default_valuation_time_utc": provenance["valuation_time_utc"],
        "valuation_assumption": "Original pilot: source snapshot timestamp minus assumed 15-minute delay; no bid/ask update timestamps or new source fetch.",
        "selection_protocol": {"expiry_rule": "nearest absolute ACT/365F maturity to nominal business-day label /252; ties earlier; duplicate nearest expiries flagged, not replaced",
            "coordinate": "log(K/F)/(.16*sqrt(exact_ACT365F_T))", "maximum_z_distance": MAX_Z_DISTANCE,
            "maximum_halfspread_over_mid": MAX_RELATIVE_HALFSPREAD,
            "target_order": list(TARGET_Z), "quote_tie_break": "lexicographic raw symbol",
            "target_option_types": "puts for target_z<=0, calls for target_z>0; actual quotes must also be OTM",
            "carry": "old exact estimates/symbols retained; new every-third paired strike within5%spot, at least15pairs, WLS halfspread SD convention",
            "separately_declared_last_call_target_revision": last_call_target,
            "excluded_scoring_design": "preserved common raw held-out quotes remain unavailable to new fitting/carry"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "raw_downloaded": False, "banks_opened": False, "model_pricing_evaluations": 0,
        "training_performed": False, "scientific_acceptance": False})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="optional preserved raw-file location, still hash checked")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--last-call-target", type=float, choices=(1.5,),
                        help="also write an explicitly separate supported-target revision; preserve original E audit")
    args = parser.parse_args()
    summary = run(args.historical_root, args.out, args.raw, last_call_target=args.last_call_target)
    print(json.dumps({key: summary[key] for key in
        ("selected_contract_count", "supported_quote_target_count", "missing_quote_target_count",
         "complete_supported_quote_layout", "one_year_pm_expiry_available", "model_pricing_evaluations")}))


if __name__ == "__main__":
    main()
