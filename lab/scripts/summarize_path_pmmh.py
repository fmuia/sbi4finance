"""Reduce four completed PMMH chains with explicit convergence gates.

Use the existing core-reference Python, which supplies ArviZ. A runtime without
ArviZ still writes an unavailable-diagnostics report and never passes gates.
This reducer does not establish particle precision or SBI conditional accuracy.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np


GATES = {"rank_rhat_max": 1.01, "bulk_ess_min": 400,
         "tail_ess_min": 400, "tail_quantiles": [0.05, 0.95]}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_digest(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _longest_rejection(accepted):
    longest = current = 0
    for value in accepted:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _load_chains(directory):
    directories = sorted(Path(directory).glob("chain-*"))
    if len(directories) != 4:
        raise ValueError(f"require four chain-* directories; found {len(directories)}")
    rows, identity, observed = [], None, None
    for path in directories:
        summary_path = path / "pmmh-summary.json"
        summary = json.loads(summary_path.read_text())
        observation = summary["observation"]
        task_digest = _task_digest(summary["task"])
        if task_digest != observation["task_sha256"]:
            raise ValueError(f"{path}: task fingerprint differs from observation identity")
        current = {"task_sha256": task_digest,
                   "parameter_names": summary["task"]["parameter_names"],
                   "trajectory_id": observation["trajectory_id"], "length": observation["length"],
                   "source_shard_sha256": observation["source_shard_sha256"], "split": observation["split"],
                   "particles": summary["particles"], "tune": summary["tune"], "draws": summary["draws"],
                   "phase_seed": summary.get("phase_seed"), "root_seed": summary.get("root_seed"),
                   "execution_mode": summary.get("execution_mode", "legacy_prior_initialized_phase"),
                   "proposal_covariance": summary.get("proposal_covariance"),
                   "particle_filter": summary.get("particle_filter"),
                   "particle_rng": summary.get("particle_rng")}
        if identity is not None and identity != current:
            raise ValueError("chains differ in task, observations, particle count, phase or iteration counts")
        identity = current
        if (summary.get("status") != "completed_feasibility_only_not_accepted_reference"
                or summary.get("completed_iterations") != summary["tune"] + summary["draws"]):
            raise ValueError(f"{path}: chain is not complete; no convergence reduction performed")
        file_hashes = {item["file"]: item["sha256"] for item in summary.get("files", [])}
        for name in ("observations.npz", "draws.npz"):
            if name not in file_hashes or _sha256(path / name) != file_hashes[name]:
                raise ValueError(f"{path}: missing or mismatched completed-chain checksum for {name}")
        with np.load(path / "observations.npz", allow_pickle=False) as saved:
            values = saved["log_returns"]
            steps = saved["dt_years"]
            if observed is None:
                observed = (values.copy(), steps.copy())
            elif not np.array_equal(values, observed[0]) or not np.array_equal(steps, observed[1]):
                raise ValueError("chains contain different actual observations/time intervals")
        with np.load(path / "draws.npz", allow_pickle=False) as saved:
            samples = saved["parameters"]
            expected = (1, summary["draws"], len(identity["parameter_names"]))
            if (samples.shape != expected or not np.isfinite(samples).all()
                    or saved["chain_id"].shape != (1,)
                    or int(saved["chain_id"][0]) != summary["chain_id"]):
                raise ValueError(f"{path}: retained sample shape/chain identity/values differ")
            accepted = saved["accepted"][0].astype(bool)
            collapse = saved["proposal_collapsed"][0].astype(bool)
            seeds = saved["likelihood_seed"][0]
            ll = saved["log_likelihood_estimate"][0]
            if accepted.shape != (summary["draws"],) or not np.isfinite(ll).all():
                raise ValueError(f"{path}: invalid retained diagnostics")
            rows.append({"samples": samples[0].copy(), "chain_id": summary["chain_id"],
                         "report": {"chain_id": summary["chain_id"], "seconds": summary["seconds"],
                                    "acceptance_draws": float(accepted.mean()),
                                    "acceptance_tune": summary.get("acceptance_tune"),
                                    "collapsed_draw_proposals": int(collapse.sum()),
                                    "longest_consecutive_rejections": _longest_rejection(accepted),
                                    "unique_accepted_particle_states": int(len(np.unique(seeds))),
                                    "physical_mean": samples[0].mean(axis=0).tolist(),
                                    "likelihood_noise": summary.get("likelihood_noise", []),
                                    "directory": str(path), "summary_sha256": _sha256(summary_path),
                                    "draws_sha256": file_hashes["draws.npz"]}})
    if {row["chain_id"] for row in rows} != {0, 1, 2, 3}:
        raise ValueError("require unique chain IDs 0, 1, 2 and 3")
    rows.sort(key=lambda item: item["chain_id"])
    return np.stack([row["samples"] for row in rows]), identity, [row["report"] for row in rows]


def _diagnostics(samples, names, arviz):
    result = {}
    for index, name in enumerate(names):
        values = samples[:, :, index]
        # Direct ndarray APIs work with ArviZ 1.3.0; specify the tail quantiles
        # explicitly rather than relying on summary's default interval width.
        rhat = float(arviz.rhat(values, method="rank"))
        bulk = float(arviz.ess(values, method="bulk"))
        tail = float(arviz.ess(values, method="tail", prob=(0.05, 0.95)))
        finite = bool(np.isfinite([rhat, bulk, tail]).all())
        result[name] = {"rank_rhat": rhat if np.isfinite(rhat) else None,
                        "bulk_ess": bulk if np.isfinite(bulk) else None,
                        "tail_ess": tail if np.isfinite(tail) else None,
                        "bulk_ess_per_retained_draw": bulk / values.size if np.isfinite(bulk) else None,
                        "physical_mean": float(values.mean()), "physical_sd": float(values.std(ddof=1)),
                        "passes_gates": bool(finite and rhat <= GATES["rank_rhat_max"]
                                             and bulk >= GATES["bulk_ess_min"] and tail >= GATES["tail_ess_min"])}
    return result


def summarize(directory, arviz, unavailable_reason=None):
    report = {"chains_directory": str(directory), "input_valid": False,
              "convergence_gates": {**GATES, "evaluated": False, "passed": False},
              "reference_accepted": False, "sbi_accuracy_accepted": False}
    try:
        samples, identity, chains = _load_chains(directory)
    except Exception as error:
        return {**report, "status": "invalid_or_incomplete_chains_no_acceptance", "error": str(error)}
    report.update(input_valid=True, identity=identity, shape=list(samples.shape), chains=chains)
    if arviz is None:
        return {**report, "status": "diagnostics_unavailable_no_acceptance", "diagnostics_error": unavailable_reason}
    try:
        diagnostics = _diagnostics(samples, identity["parameter_names"], arviz)
    except Exception as error:
        return {**report, "status": "diagnostics_unavailable_no_acceptance", "diagnostics_error": repr(error)}
    passed = all(item["passes_gates"] for item in diagnostics.values())
    report.update(diagnostics=diagnostics,
                  convergence_gates={**GATES, "evaluated": True, "passed": passed},
                  status="convergence_pass_particle_precision_not_certified" if passed else "convergence_failed_no_reference_acceptance",
                  interpretation="Convergence gates concern these four chains for one history/horizon. Particle-number precision and a matched SBI comparison remain separate requirements; no SBI accuracy is accepted by this reduction.")
    return report


def run(chains, out, baseline_chains=None):
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("arviz", "arviz-stats", "arviz-base"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    unavailable = None
    try:
        import arviz
    except Exception as error:
        arviz, unavailable = None, repr(error)
    report = {"schema": "path-pmmh-four-chain-reduction-v1", "versions": versions,
              "current": summarize(chains, arviz, unavailable),
              "source_sha256": _sha256(Path(__file__))}
    if baseline_chains is not None:
        baseline = summarize(baseline_chains, arviz, unavailable)
        report["baseline"] = baseline
        current = report["current"]
        if "diagnostics" in current and "diagnostics" in baseline:
            for field in ("task_sha256", "trajectory_id", "length", "source_shard_sha256", "particles"):
                if current["identity"][field] != baseline["identity"][field]:
                    report["comparison_error"] = f"baseline differs in {field}; no before/after comparison"
                    break
            else:
                report["before_after"] = {name: {"before": baseline["diagnostics"][name],
                                                "after": current["diagnostics"][name]}
                                          for name in current["identity"]["parameter_names"]}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-chains", type=Path)
    args = parser.parse_args()
    report = run(args.chains, args.out, args.baseline_chains)
    print(json.dumps({"out": str(args.out), "status": report["current"]["status"],
                      "convergence_passed": report["current"]["convergence_gates"]["passed"],
                      "reference_accepted": False, "sbi_accuracy_accepted": False}))


if __name__ == "__main__":
    main()
