"""A fixed-budget, independently replicated rough time-grid development pilot.

Different grids use independent streams. Giving differently shaped normal
arrays the same integer seed would not create a valid coupled refinement.
This pilot reports approximate pointwise Monte Carlo uncertainty; it does not
turn a collection of price comparisons into posterior or global acceptance.
"""

import copy
import hashlib
import os
import platform
import resource
import time

import numpy as np

from ..gbm_bank import _integer, _json
from . import pricing, tasks

FORMAT = "rough_grid_development_pilot_v1"
DEFAULT_CASES = (
    ("typical_flat", (0.10, 1.50, -0.70, 0.040, 0.040, 2.0)),
    ("rising_curve", (0.12, 1.60, -0.65, 0.020, 0.090, 3.0)),
    ("falling_curve", (0.12, 1.60, -0.65, 0.090, 0.020, 3.0)),
    ("nearly_flat_fast_decay", (0.15, 1.80, -0.60, 0.040, 0.0401, 7.5)),
    ("low_h_high_eta", (0.025, 3.40, -0.95, 0.040, 0.090, 7.5)),
    ("high_h_low_eta", (0.34, 0.55, -0.25, 0.090, 0.040, 0.3)),
    ("high_variance", (0.08, 2.50, -0.90, 0.150, 0.150, 4.0)),
    ("low_short_steep_rise", (0.05, 2.80, -0.95, 0.012, 0.140, 7.5)),
)


def cases(task):
    """Predeclared points; they are not a random prior sample or selected by results."""
    tasks.validate_task(task)
    if not tasks.is_joint_curve_task(task):
        raise ValueError("a joint rough task is required")
    bounds = np.asarray(task["prior_bounds"])
    result = [{"name": name, "parameters": list(values)} for name, values in DEFAULT_CASES]
    for case in result:
        values = np.asarray(case["parameters"])
        if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
            raise ValueError(f"predeclared case {case['name']} is outside the candidate prior")
    return result


def refined_task(task, multiplier):
    """Refine each block and rebuild its analytic input curve when pricing."""
    multiplier = _integer(multiplier, "grid multiplier", minimum=1)
    refined = copy.deepcopy(task)
    for block in refined["numerics"]["blocks"]:
        block["n_steps"] *= multiplier
    tasks.validate_task(refined)
    return refined


def stream_seed(task, case, root_seed, multiplier, replicate, block):
    """Stable under case selection/order, job arrays and worker counts."""
    root_seed = _integer(root_seed, "root seed")
    multiplier = _integer(multiplier, "grid multiplier", minimum=1)
    replicate = _integer(replicate, "replicate")
    block = _integer(block, "block")
    digest = hashlib.sha256(_json({"task": task, "case": case}).encode()).hexdigest()
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    return int(np.random.SeedSequence([0x52475049, root_seed, *words, multiplier,
                                      replicate, block]).generate_state(1, dtype=np.uint64)[0])


def fixed_surface(task, values, *, multiplier, paths, seeds):
    """Price a complete surface once at fixed paths; no precision-based redraw.

    Unlike production generation, an over-budget MCSE is returned and labelled
    unresolved by the caller, not silently replaced or treated as accurate.
    """
    paths = _integer(paths, "paths", minimum=2)
    if paths % 2:
        raise ValueError("an even number of paths is required for antithetic pairs")
    refined = refined_task(task, multiplier)
    contracts = refined["context"]["contracts"]
    blocks = refined["numerics"]["blocks"]
    if len(seeds) != len(blocks):
        raise ValueError("one independent seed per block is required")
    curve = tasks.joint_curve(refined, values)
    parameters = tasks.model_parameters(refined, values)
    maturities = np.asarray([c["maturity"] for c in contracts])
    assignment = np.full(len(contracts), -1)
    for b, block in reversed(list(enumerate(blocks))):
        assignment[maturities <= block["horizon"] + 1e-12] = b
    if np.any(assignment < 0):
        raise ValueError("all contracts require a pricing block")
    prices, se = np.empty(len(contracts)), np.empty(len(contracts))
    for b, block in enumerate(blocks):
        chosen = np.flatnonzero(assignment == b)
        if not len(chosen):
            continue
        # Regression degrees of freedom require more pairs than controls.
        if paths // 2 <= 2 * len(np.unique(maturities[chosen])) + 1:
            raise ValueError("too few antithetic pairs for the block's control regression")
        p, s, covariance = pricing.rough_block(
            parameters, curve, [contracts[i] for i in chosen], horizon=block["horizon"],
            n_steps=block["n_steps"], n_pairs=paths // 2, seed=seeds[b])
        if (not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(s < 0)
                or not np.isfinite(covariance).all()):
            raise FloatingPointError("nonfinite price/error covariance in numerical pilot")
        prices[chosen], se[chosen] = p, s
    return prices, se


def comparison(coarse, fine, sigma):
    """Independent-replicate pooled means and approximate pointwise 95% intervals."""
    sigma = np.asarray(sigma, dtype=float)
    def pooled(rows):
        p = np.asarray([row["prices"] for row in rows])
        s = np.asarray([row["standard_errors"] for row in rows])
        if p.shape != s.shape or p.ndim != 2 or p.shape[1:] != sigma.shape or not len(p):
            raise ValueError("matching nonempty price/error rows required")
        # Each row is independent and uses the same fixed path count.
        mean = p.mean(axis=0)
        se = np.sqrt(np.sum(s**2, axis=0)) / len(p)
        between = p.std(axis=0, ddof=1) / np.sqrt(len(p)) if len(p) > 1 else None
        return mean, se, between
    p0, s0, b0 = pooled(coarse)
    p1, s1, b1 = pooled(fine)
    delta = (p1 - p0) / sigma
    error = np.sqrt(s0**2 + s1**2) / sigma
    half = 1.96 * error
    return {"fine_minus_coarse_over_quote_sigma": delta.tolist(),
            "mc_se_of_difference_over_quote_sigma": error.tolist(),
            "pointwise_mc95_lower_over_quote_sigma": (delta - half).tolist(),
            "pointwise_mc95_upper_over_quote_sigma": (delta + half).tolist(),
            "max_abs_shift_over_quote_sigma": float(np.max(np.abs(delta))),
            "max_pointwise_mc95_halfwidth_over_quote_sigma": float(np.max(half)),
            "coarse_between_rep_se_over_quote_sigma": None if b0 is None else (b0 / sigma).tolist(),
            "fine_between_rep_se_over_quote_sigma": None if b1 is None else (b1 / sigma).tolist(),
            "interpretation": "Approximate pointwise MC intervals, not simultaneous/global bounds; "
                              "price refinement alone does not establish posterior/product accuracy."}


def run_case(task, case, *, paths=32768, replicates=2, multipliers=(1, 2, 4),
             root_seed=271026, only_grid=None, only_replicate=None, progress=None):
    """Run every predeclared grid/replicate, retaining failures without replacements."""
    tasks.validate_task(task)
    if not tasks.is_joint_curve_task(task):
        raise ValueError("joint rough task required")
    paths = _integer(paths, "paths", minimum=2)
    replicates = _integer(replicates, "replicates", minimum=1)
    multipliers = tuple(_integer(x, "multiplier", minimum=1) for x in multipliers)
    if paths % 2 or multipliers != tuple(sorted(set(multipliers))) or len(multipliers) < 2:
        raise ValueError("even paths and at least two increasing unique grid multipliers required")
    if only_grid is not None and (type(only_grid) is not int or only_grid not in multipliers):
        raise ValueError("selected grid must belong to the declared multiplier list")
    if only_replicate is not None and (type(only_replicate) is not int or not 0 <= only_replicate < replicates):
        raise ValueError("selected replicate must belong to the declared replicate range")
    values = np.asarray(case["parameters"], dtype=float)
    bounds = np.asarray(task["prior_bounds"])
    if (not case.get("name") or values.shape != (6,) or not np.isfinite(values).all()
            or np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1])):
        raise ValueError("named six-target case inside the declared prior required")
    sigma = tasks.sigma(task)
    report = {"format": FORMAT, "case": case, "paths_per_grid_replicate": paths,
              "replicates": replicates, "grid_multipliers": list(multipliers),
              "task_sha256": hashlib.sha256(_json(task).encode()).hexdigest(),
              "component_selection": {"grid_multiplier": only_grid, "replicate": only_replicate},
              "root_seed": root_seed, "streams": "independent across cases, grids, replicates and blocks",
              "runs": [], "comparisons": [], "scientific_acceptance": "not_assessed",
              "mc_target_over_quote_sigma": task["numerics"]["max_mc_se_over_quote_sigma"],
              "existing_generation_exception_cap_over_quote_sigma": task["numerics"].get(
                  "max_mc_se_over_quote_sigma_last_rung", task["numerics"]["max_mc_se_over_quote_sigma"]),
              "thread_environment": {name: os.environ.get(name) for name in
                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    all_seeds = []
    for multiplier in multipliers:
        if only_grid is not None and multiplier != only_grid:
            continue
        for replicate in range(replicates):
            if only_replicate is not None and replicate != only_replicate:
                continue
            seeds = [stream_seed(task, case, root_seed, multiplier, replicate, b)
                     for b in range(len(task["numerics"]["blocks"]))]
            if any(seed in all_seeds for seed in seeds):
                raise ValueError("actual reduced pricing-seed collision")
            all_seeds.extend(seeds)
            row = {"grid_multiplier": multiplier, "replicate": replicate, "pricing_seeds": seeds}
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                prices, errors = fixed_surface(task, values, multiplier=multiplier, paths=paths, seeds=seeds)
                ratios = errors / sigma
                row.update(status="priced", prices=prices.tolist(), standard_errors=errors.tolist(),
                           max_mc_se_over_quote_sigma=float(np.max(ratios)),
                           mc_budget_met=bool(np.all(ratios <= task["numerics"]["max_mc_se_over_quote_sigma"])),
                           within_existing_generation_exception_cap=bool(np.all(ratios <= report[
                               "existing_generation_exception_cap_over_quote_sigma"])))
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__, error=str(error))
            row.update(wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu)
            row["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                                 * (1 if platform.system() == "Darwin" else 1024))
            report["runs"].append(row)
            if progress is not None:
                progress(report)
    return _summarize(report, sigma)


def _summarize(report, sigma):
    """Shared serial/component-reduction summaries; no re-pricing or dropped runs."""
    multipliers, replicates = report["grid_multipliers"], report["replicates"]
    report["comparisons"] = []
    for coarse, fine in zip(multipliers, multipliers[1:]):
        rows0 = [r for r in report["runs"] if r["grid_multiplier"] == coarse and r["status"] == "priced"]
        rows1 = [r for r in report["runs"] if r["grid_multiplier"] == fine and r["status"] == "priced"]
        item = {"coarse_multiplier": coarse, "fine_multiplier": fine}
        if len(rows0) != replicates or len(rows1) != replicates:
            item.update(status="incomplete", reason="a predeclared grid replicate failed; no replacement seed")
        else:
            item.update(status="reported", **comparison(rows0, rows1, sigma))
        report["comparisons"].append(item)
    selected = any(value is not None for value in report["component_selection"].values())
    report["status"] = (("component_completed" if selected else "completed")
                        if all(r["status"] == "priced" for r in report["runs"]) else "incomplete")
    report["runs_exceeding_mc_target"] = sum(r["status"] == "priced" and not r["mc_budget_met"]
                                             for r in report["runs"])
    report["runs_exceeding_existing_exception_cap"] = sum(
        r["status"] == "priced" and not r["within_existing_generation_exception_cap"] for r in report["runs"])
    report["total_cpu_seconds"] = sum(r["cpu_seconds"] for r in report["runs"])
    report["total_wall_seconds"] = sum(r["wall_seconds"] for r in report["runs"])
    report["memory_scope"] = "Process high-water RSS, not incremental allocation per price call."
    return report


def combine_components(task, case, components):
    """Check exact component coverage and seeds, then reuse the serial comparisons.

    Every component must be one independently executed declared (grid,replicate)
    slot. Failures remain rows; missing or duplicated slots are rejected.
    """
    tasks.validate_task(task)
    if not tasks.is_joint_curve_task(task) or not components:
        raise ValueError("joint rough task and nonempty component reports required")
    first = components[0]
    keys = ("format", "case", "paths_per_grid_replicate", "replicates", "grid_multipliers",
            "root_seed", "task_sha256", "mc_target_over_quote_sigma",
            "existing_generation_exception_cap_over_quote_sigma")
    expected_task = hashlib.sha256(_json(task).encode()).hexdigest()
    if (first.get("format") != FORMAT or first.get("case") != case
            or first.get("task_sha256") != expected_task):
        raise ValueError("component task/case identity differs from the declaration")
    grids, repetitions = first["grid_multipliers"], first["replicates"]
    expected = {(grid, rep) for grid in grids for rep in range(repetitions)}
    found, seeds, rows = set(), set(), []
    for component in components:
        if any(component.get(key) != first.get(key) for key in keys):
            raise ValueError("component task/case/path/settings identities differ")
        selected = component.get("component_selection", {})
        slot = (selected.get("grid_multiplier"), selected.get("replicate"))
        if slot not in expected or slot in found or len(component.get("runs", [])) != 1:
            raise ValueError("each component must cover one unique declared grid/replicate slot")
        row = component["runs"][0]
        actual = [stream_seed(task, case, first["root_seed"], slot[0], slot[1], b)
                  for b in range(len(task["numerics"]["blocks"]))]
        if ((row.get("grid_multiplier"), row.get("replicate")) != slot
                or row.get("pricing_seeds") != actual or any(seed in seeds for seed in actual)):
            raise ValueError("component pricing seed or slot changed/collided")
        if row.get("status") not in ("priced", "failed"):
            raise ValueError("invalid component outcome")
        if row["status"] == "priced":
            c = len(task["context"]["contracts"])
            prices, errors = np.asarray(row.get("prices")), np.asarray(row.get("standard_errors"))
            if (prices.shape != (c,) or errors.shape != (c,) or not np.isfinite(prices).all()
                    or not np.isfinite(errors).all() or np.any(errors < 0)):
                raise ValueError("invalid component price/error arrays")
        found.add(slot)
        seeds.update(actual)
        rows.append(copy.deepcopy(row))
    if found != expected:
        raise ValueError("component coverage is incomplete; no missing slot may be silently omitted")
    result = copy.deepcopy(first)
    result["component_selection"] = {"grid_multiplier": None, "replicate": None}
    result["runs"] = sorted(rows, key=lambda row: (row["grid_multiplier"], row["replicate"]))
    result.pop("declaration_sha256", None)
    result["combined_component_count"] = len(components)
    result["thread_environment"] = [component["thread_environment"] for component in components]
    return _summarize(result, tasks.sigma(task))
