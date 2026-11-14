"""Validated saved examples and small session calculations for the take-home."""

import hashlib
import json

import numpy as np

from sbi4f_tutorial.shared.paths import ASSETS
from sbi4f_tutorial.shared.path_statistics import maximum_drawdown, sample_acf
from . import model


def load_saved():
    folder = ASSETS / "takehomes" / "mmar-v2"
    report = json.loads((folder / "report.json").read_text())
    source = folder / "saved.npz"
    if report["format"] != "mmar_takehome_bundle_v1" or hashlib.sha256(source.read_bytes()).hexdigest() != report["saved_sha256"]:
        raise ValueError("MMAR saved results differ from their bound report")
    if (json.dumps(report["model_design"], sort_keys=True) != json.dumps(model.DESIGN, sort_keys=True)
            or report["frozen_rule"]["resolutions"] != list(model.CAPSTONE_D)
            or report["frozen_rule"]["parameter_order"] != list(model.PARAMETERS)):
        raise ValueError("saved results describe a different simulator, prior, target order or diagnostic design")
    with np.load(source, allow_pickle=False) as content:
        data = {key: content[key] for key in content.files}
    for d in model.CAPSTONE_D:
        theta = model.parameter_array(data[f"posterior_{d}"])
        if theta.ndim != 2 or theta.shape[1] != 4 or data[f"mdd_{d}"].shape != (512,):
            raise ValueError("saved posterior/scenario shapes differ from lesson contract")
    return report, data


def load_market():
    folder = ASSETS / "market-examples" / "voo-covid-256"
    metadata = json.loads((folder / "market.json").read_text())
    with np.load(folder / "returns.npz", allow_pickle=False) as content:
        dates = content["dates"].copy()
        returns = np.asarray(content["simple_returns"], dtype=float).reshape(-1)
    if returns.shape != (256,) or len(dates) != 256 or not np.isfinite(returns).all() or np.any(returns <= -1):
        raise ValueError("fixed market example must contain 256 finite accepted simple returns")
    report = json.loads((ASSETS / "takehomes" / "mmar-v2" / "report.json").read_text())
    identity = report["market"]
    if (hashlib.sha256(returns.astype("<f8").tobytes()).hexdigest() != identity["simple_returns_sha256"]
            or str(dates[0]) != identity["start_date"] or str(dates[-1]) != identity["end_date"]):
        raise ValueError("observed market differs from the saved posterior's bound inputs")
    return metadata, dates, returns


def scenario_result(draws, d=8, *, count=512, seed=270061112):
    theta = model.parameter_array(draws)
    count = min(count, len(theta))
    samples = model.simulate_parameters(theta[:count], d, seed=seed)["returns"]
    point = np.repeat(theta.mean(axis=0)[None], count, axis=0)
    point_samples = model.simulate_parameters(point, d, seed=seed+1)["returns"]
    return {"draws": theta, "mdd": maximum_drawdown(np.log1p(samples)),
            "point_mdd": maximum_drawdown(np.log1p(point_samples)),
            "wealth": np.column_stack((np.ones(min(count, 12)), np.cumprod(1+samples[:12], axis=1))),
            "summaries": summary(samples), "abs_acf": sample_acf(np.log1p(samples), absolute=True)["values"]}


def summary(returns):
    values = np.asarray(returns)
    return np.column_stack((100*np.std(values, axis=-1),
                            100*np.quantile(np.abs(values), .95, axis=-1),
                            100*maximum_drawdown(np.log1p(values))))


def exceedance(drawdowns, threshold):
    from .diagnostics import wilson
    values = np.asarray(drawdowns)
    hits = int(np.sum(values > threshold))
    low, high = wilson(hits, len(values))
    return {"scenarios": len(values), "above_threshold": hits,
            "probability_pct": 100*hits/len(values), "simulation_interval95_pct": [100*low, 100*high]}


def sensitivity_table(data, threshold=.2):
    rows = []
    for d in model.CAPSTONE_D:
        q = data[f"posterior_{d}"][:, 2]
        mdd = data[f"mdd_{d}"]
        lo, med, hi = np.quantile(q, [.05, .5, .95])
        rows.append({"d (known blocks)": d, "days per block": 256//d,
                     "q median": float(med), "q central 90%": f"{lo:.3f}–{hi:.3f}",
                     "median drawdown (%)": float(100*np.median(mdd)),
                     f"P(drawdown > {100*threshold:g}%)": float(100*np.mean(mdd > threshold))})
    return rows
