"""Declared synthetic conditional references for the full-panel training study.

Observations follow the untruncated Gaussian law, including negative prices.
Contract/scaling metadata remain the valid frozen market design. No synthetic
observation is used to reconstruct an IV, vega, forward or discount.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration.heston_pilot import HestonPilot, DEFAULT_LADDER
from scripts.run_heston_observation_pilot import expected_variance, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


class SyntheticLikelihood:
    """Physical normalized likelihood, independent of the source midpoint sign."""
    def __init__(self, pilot, observations):
        self.pilot = pilot
        self.observations = np.asarray(observations, dtype=float).copy()
        if self.observations.shape != pilot.records["mid"].shape or not np.isfinite(self.observations).all():
            raise ValueError("one arbitrary finite observation per frozen contract required")
        self.observations.setflags(write=False)

    def __call__(self, parameters):
        point = np.asarray(parameters, dtype=float)
        if point.shape != (len(self.pilot.names),) or not np.isfinite(point).all():
            raise ValueError("one finite physical parameter vector required")
        if np.any(point < self.pilot.bounds[:, 0]) or np.any(point > self.pilot.bounds[:, 1]):
            return -np.inf
        prices = self.pilot.prices(point)
        sigma = self.pilot.observation_sigma(point)
        residual = (self.observations - prices) / sigma
        return float(-.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2)))


def load_case(design_path, panel_path, index):
    design = json.loads(Path(design_path).read_text())
    panel = json.loads(Path(panel_path).read_text())
    if panel["design_sha256"] != digest(design_path):
        raise ValueError("declared reference panel differs from frozen quote design")
    bounds = np.asarray(panel["task_bounds"], dtype=float)
    if bounds.shape != (6, 2) or bounds[-1, 0] != 0:
        raise ValueError("six physical prior intervals including uniform tau from zero required")
    matches = [case for case in panel["cases"] if case["index"] == index]
    if len(matches) != 1:
        raise ValueError("index must select exactly one declared case")
    case = matches[0]
    pilot = HestonPilot(design["records"], bounds=bounds[:5], infer_discrepancy=True,
                        tau_max=bounds[-1, 1], spread_multiplier=1.,
                        quadrature_ladder=DEFAULT_LADDER + ((16384, 9600.),))
    truth = np.asarray(case["parameters"], dtype=float)
    standard = np.asarray(case["standard_normal_errors"], dtype=float)
    if (truth.shape != (6,) or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])
            or not np.isfinite(truth).all() or standard.shape != (len(pilot.records["mid"]),)
            or not np.isfinite(standard).all()):
        raise ValueError("declared truth or standard-normal errors disagree with the task")
    clean = pilot.prices(truth)
    observations = clean + standard * pilot.observation_sigma(truth)
    return design, pilot, {**case, "clean_prices": clean.tolist(), "observations": observations.tolist(),
                           "observation_sha256": values_digest(observations),
                           "design_sha256": digest(design_path), "panel_sha256": digest(panel_path),
                           "scope": "Fixed development observation from untruncated Gaussian law; frozen market design/vegas/carry, no selection or empirical-noise claim."}


def finalize(directory, design, pilot):
    import xarray as xr
    from sbi4finance.calibration import reference
    config = json.loads((directory / "run.json").read_text())
    case = json.loads((directory / "case.json").read_text())
    if config["case_json_sha256"] != digest(directory / "case.json"):
        raise ValueError("synthetic reference observation binding changed")
    with np.load(directory / "chains.npz") as archive:
        draws = archive["parameters"]
        stats = {key.removeprefix("sample_stats_"): (("chain", "draw"), archive[key])
                 for key in archive.files if key.startswith("sample_stats_")}
    expected = (config["settings"]["chains"], config["settings"]["draws"], len(pilot.names))
    if (draws.shape != expected or config["names"] != pilot.names or not np.isfinite(draws).all()
            or np.any(draws < pilot.bounds[:, 0]) or np.any(draws > pilot.bounds[:, 1])):
        raise ValueError("completed trace dimensions, names or physical support differ")
    tree = xr.DataTree.from_dict({"posterior": xr.Dataset(
        {"parameters": (("chain", "draw", "parameter"), draws)}, coords={"parameter": pilot.names}),
        "sample_stats": xr.Dataset(stats)})
    diagnostics = reference.chain_diagnostics(tree)
    products = np.stack([expected_variance(draws, interval) for interval in design["products"].values()], axis=-1)
    product_tree = xr.DataTree.from_dict({"posterior": xr.Dataset(
        {"parameters": (("chain", "draw", "parameter"), products)}, coords={"parameter": list(design["products"])})})
    diagnostics["products"] = reference.chain_diagnostics(product_tree)
    diagnostics["parameter_mcmc_diagnostics_passed"] = diagnostics["mcmc_diagnostics_passed"]
    diagnostics["mcmc_diagnostics_passed"] = bool(diagnostics["mcmc_diagnostics_passed"] and diagnostics["products"]["mcmc_diagnostics_passed"])
    diagnostics["chain_payload_sha256"] = digest(directory / "chains.npz")
    diagnostics["case_json_sha256"] = digest(directory / "case.json")
    diagnostics["scientific_acceptance"] = False
    diagnostics["analysis_source_sha256"] = digest(__file__)
    write_json(directory / "diagnostics.json", diagnostics)
    truth = np.asarray(case["parameters"])
    flat = draws.reshape(-1, draws.shape[-1])
    summary = {"truth": truth.tolist(), "parameter_quantiles_05_50_95": np.quantile(flat, [.05, .5, .95], axis=0).tolist(),
               "products": {name: {"truth": float(expected_variance(truth, interval)),
                                   "quantiles_05_50_95": np.quantile(expected_variance(flat, interval), [.05, .5, .95]).tolist()}
                            for name, interval in design["products"].items()},
               "scientific_acceptance": False, "scope": "Conditional reference at a declared development observation; not population calibration."}
    write_json(directory / "posterior-summary.json", summary)
    return diagnostics


def run_case(args, index):
    from sbi4finance.calibration import reference
    from scripts.calibration_reference import PartialDrawWriter
    design, pilot, case = load_case(args.design, args.panel, index)
    directory = args.out / f"case-{index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "case.json", case)
    likelihood = SyntheticLikelihood(pilot, case["observations"])
    seed = args.seed + index
    config = {"format": "heston_training_conditional_reference_v1", "arm": "full",
              "design_sha256": digest(args.design), "panel_sha256": digest(args.panel),
              "case_json_sha256": digest(directory / "case.json"), "index": index,
              "observations": case["observations"], "observation_sha256": case["observation_sha256"],
              "names": pilot.names, "bounds": pilot.bounds.tolist(), "infer_discrepancy": True,
              "tau_max": float(pilot.bounds[-1, 1]), "spread_multiplier": 1.,
              "quadrature_ladder": pilot.quadrature_ladder,
              "settings": {"seed": seed, **{key: getattr(args, key) for key in ("draws", "tune", "chains", "cores", "fit_budget")}},
              "sampler": "Native PyMC DEMetropolisZ with deterministic normalized likelihood and physical Uniform priors",
              "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("pymc", "pytensor", "arviz-stats", "numpy", "scipy")}},
              "source_sha256": {"runner": digest(__file__), "pilot": digest(sys.modules[HestonPilot.__module__].__file__),
                                "reference": digest(reference.__file__)},
              "scientific_acceptance": False, "scope": case["scope"]}
    write_json(directory / "run.json", config)
    started = time.perf_counter()
    try:
        initialization = reference.data_fit_initialization(likelihood, pilot.bounds, args.chains, seed,
                                                           fit_starts=min(4, args.chains), fit_budget=args.fit_budget)
        write_json(directory / "initialization.json", initialization)
        partial = PartialDrawWriter(directory / "partial", chains=args.chains, tune=args.tune, draws=args.draws, every=1000)
        tree = reference.sample_deterministic_reference(likelihood, pilot.bounds, pilot.names,
                    draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores, seed=seed,
                    progressbar=False, initialization=initialization, callback=partial)
        draws = tree["posterior"]["parameters"].transpose("chain", "draw", "parameter").values
        arrays = {"parameters": draws}
        for name, value in tree["sample_stats"].data_vars.items():
            if set(value.dims) == {"chain", "draw"}:
                arrays["sample_stats_" + name] = value.transpose("chain", "draw").values
        np.savez_compressed(directory / "chains.npz", **arrays)
        write_json(directory / "trace.json", {"dimensions": {"parameters": ["chain", "draw", "parameter"],
                **{key: ["chain", "draw"] for key in arrays if key != "parameters"}},
                "coordinates": {"chain": list(range(args.chains)), "draw": list(range(args.draws)), "parameter": pilot.names},
                "warmup_discarded": True, "sampler_resumable": False, "seconds": time.perf_counter() - started})
        diagnostics = finalize(directory, design, pilot)
        print(json.dumps({"index": index, "out": str(directory), "mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"],
                          "seconds": time.perf_counter() - started}), flush=True)
    except Exception as error:
        write_json(directory / "failure.json", {"error_type": type(error).__name__, "error": str(error), "seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=int, nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=12000)
    parser.add_argument("--tune", type=int, default=6000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--fit-budget", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=261129)
    args = parser.parse_args()
    for index in args.index:
        run_case(args, index)


if __name__ == "__main__":
    main()
