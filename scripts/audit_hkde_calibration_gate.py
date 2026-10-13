"""Explain the saved HKDE rank alarm without changing an estimator or gate.

Post-hoc diagnostics only: no new market/model observations, proposals, training
or acceptance decision. Randomness below belongs to explicit diagnostic controls.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.stats import binom, kstwo


SETTINGS = {"seed": 96012, "conditional_replicates": 20000,
            "uniform_replicates": 100000, "chunk_size": 500}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def ks(values):
    """One-sample distance from uniform; last axis indexes cases."""
    ordered = np.sort(values, axis=-1)
    n = ordered.shape[-1]
    return np.maximum(np.max(np.arange(1, n + 1) / n - ordered, axis=-1),
                      np.max(ordered - np.arange(n) / n, axis=-1))


def distribution(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "quantiles_025_50_975": np.quantile(values, [.025, .5, .975]).tolist()}


def pit_from_prices(axes, prices, observations, truth, sigma):
    """Independent likelihood -> trapezoidal marginal integration -> PIT.

    Integrate likelihood values directly, rather than calling the original
    reference's mass-to-density conversion. Uniform prior constants cancel.
    """
    result = np.empty_like(truth)
    for i, observation in enumerate(observations):
        log_likelihood = -.5 * np.sum(((prices - observation[:, 0]) / sigma) ** 2, axis=1)
        joint = np.exp(log_likelihood - log_likelihood.max()).reshape(tuple(map(len, axes)))
        for d, axis in enumerate(axes):
            density = np.trapezoid(joint, x=axes[1-d], axis=1-d)
            cdf = np.r_[0., np.cumsum(.5 * (density[:-1] + density[1:]) * np.diff(axis))]
            cdf /= cdf[-1]
            result[i, d] = np.interp(truth[i, d], axis, cdf)
    return result


def oracle_controls(pit, observed, samples, bound):
    """Coordinatewise IID-reference rank controls conditional on this cohort.

    Independent marginal binomials do NOT reproduce joint rank dependence.
    These are not new cohorts, a bootstrap, or a null for finite-pool SIR.
    """
    arrays, rows = {}, []
    for d in range(2):
        u = pit[:, d]
        rng = np.random.default_rng(np.random.SeedSequence([SETTINGS["seed"], 1, d]))
        pieces = []
        for first in range(0, SETTINGS["conditional_replicates"], SETTINGS["chunk_size"]):
            count = min(SETTINGS["chunk_size"], SETTINGS["conditional_replicates"] - first)
            ranks = rng.binomial(samples, u, size=(count, len(u)))
            z = (ranks + rng.random(ranks.shape)) / (samples + 1)
            delta = z - u
            pieces.append(np.column_stack([ks(z), delta.mean(axis=1),
                                          np.sqrt(np.mean(delta**2, axis=1))]))
        metrics = np.concatenate(pieces)
        arrays[f"oracle_metrics_{d}"] = metrics
        actual = observed[:, d] - u
        bias = (.5-u) / (samples+1)
        variance = (samples*u*(1-u) + 1/12) / (samples+1)**2
        fraction = float(np.mean(metrics[:, 0] > bound))
        rows.append({
            "coordinate": d, "reference_ks": float(ks(u)),
            "sir_ks": float(ks(observed[:, d])),
            "sir_minus_reference_mean": float(actual.mean()),
            "sir_minus_reference_rmse": float(np.sqrt(np.mean(actual**2))),
            "sir_reference_correlation": float(np.corrcoef(observed[:, d], u)[0, 1]),
            "iid_reference_expected_mean_difference": float(bias.mean()),
            "iid_reference_mean_difference_sd": float(np.sqrt(variance.sum()) / len(u)),
            "iid_reference_root_expected_mse": float(np.sqrt(np.mean(variance+bias**2))),
            "conditional_iid_reference_ks": distribution(metrics[:, 0]),
            "conditional_iid_reference_mean_difference": distribution(metrics[:, 1]),
            "conditional_iid_reference_rmse": distribution(metrics[:, 2]),
            "conditional_iid_reference_gate_failure_fraction": fraction,
            "failure_fraction_mcse": float(np.sqrt(fraction*(1-fraction)/len(metrics))),
            "sir_ks_percentile_in_control": float(np.mean(metrics[:, 0] <= ks(observed[:, d]))),
            "sir_rmse_percentile_in_control": float(np.mean(metrics[:, 2] <= np.sqrt(np.mean(actual**2)))),
        })
    return rows, arrays


def conditional_pool_checks(checks, fine_pit):
    """CDF at truth from saved conditional pools; no resampling or new draws."""
    result, arrays = [], {}
    for count in (2048, 8192):
        parameters = checks["parameters"][:, :count]
        weights = checks["weights"][:, :count]
        weights = weights / weights.sum(axis=1, keepdims=True)
        indicators = parameters < checks["truth"][:, None, :]
        cdf = np.sum(weights[..., None] * indicators, axis=1)
        se = np.sqrt(count/(count-1) * np.sum(
            weights[..., None]**2 * (indicators-cdf[:, None, :])**2, axis=1))
        delta = cdf - fine_pit[:len(cdf)]
        arrays.update({f"pool_cdf_{count}": cdf, f"pool_cdf_se_{count}": se,
                       f"pool_cdf_error_{count}": delta})
        result.append({"proposals": count, "cases": len(cdf),
                       "mean_cdf_error": delta.mean(axis=0).tolist(),
                       "rmse_cdf_error": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                       "max_abs_cdf_error": np.max(np.abs(delta), axis=0).tolist(),
                       "rms_plugin_snis_cdf_se": np.sqrt(np.mean(se**2, axis=0)).tolist(),
                       "max_abs_error_over_plugin_se": np.max(np.abs(delta)/se, axis=0).tolist()})
    return result, arrays


def figures(out, report, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Long-run variance theta", "Jump intensity lambda"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for d, ax in enumerate(axes):
        row = report["conditional_oracle"][d]
        ax.hist(arrays[f"oracle_metrics_{d}"][:, 0], bins=50, density=True,
                color="#ccdce8", label="Ideal IID reference draws on saved cohort")
        ax.axvline(report["gate"]["dkw_bound"], color="#b53a38", ls=":", label="Frozen gate")
        ax.axvline(row["reference_ks"], color="#343a40", lw=2, label="Reference PIT")
        ax.axvline(row["sir_ks"], color="#007d83", ls="--", lw=2, label="Saved corrected SIR")
        ax.set(title=names[d], xlabel="Rank KS distance from uniform", ylabel="Control density")
        ax.legend(fontsize=8)
    fig.suptitle("The same cohort can trigger the gate with ideal reference sampling")
    fig.savefig(out / "oracle-rank-control.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    samples = report["gate"]["posterior_draws"]
    for d, ax in enumerate(axes):
        u = arrays["reference_pit_129"][:, d]
        order = np.argsort(u)
        low = binom.ppf(.025, samples, u)/(samples+1) - u
        high = (binom.ppf(.975, samples, u)+1)/(samples+1) - u
        ax.fill_between(u[order], low[order], high[order], color="#e1e6eb",
                        label="Pointwise IID-reference rank envelope")
        ax.scatter(u, arrays["sir_randomized_ranks"][:, d]-u, s=12, alpha=.65,
                   color="#007d83", label="Saved SIR minus reference PIT")
        ax.axhline(0, color="#343a40", lw=1)
        ax.set(title=names[d], xlabel="Numerical reference CDF at truth", ylabel="Rank minus reference percentile")
        ax.legend(fontsize=8)
    fig.suptitle("Paired differences include rank noise AND finite importance-pool error")
    fig.savefig(out / "paired-rank-errors.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for d, ax in enumerate(axes):
        for count, offset, color in [(2048, -.12, "#007d83"), (8192, .12, "#b36d16")]:
            error, se = arrays[f"pool_cdf_error_{count}"][:, d], arrays[f"pool_cdf_se_{count}"][:, d]
            ax.errorbar(np.arange(len(error))+offset, error, yerr=2*se, fmt="o", ms=3,
                        capsize=2, color=color, label=f"{count:,} proposals; +/-2 plug-in MCSE")
        ax.axhline(0, color="#343a40", lw=1)
        ax.set(title=names[d], xlabel="Saved conditional test case", ylabel="Weighted CDF minus reference PIT")
        ax.legend(fontsize=8)
    fig.suptitle("Direct CDF checks: 16 cases, nested pools, no SIR resampling")
    fig.savefig(out / "conditional-cdf-errors.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    before = time.perf_counter()
    directory = root / "docs/results/hkde-correction"
    test = json.loads((directory / "test.json").read_text())
    audit = json.loads((directory / "test-independent-audit.json").read_text())
    protocol = json.loads((root / "configs/hkde_correction_boundary_protocol.json").read_text())
    bindings = dict(test["input_sha256"])
    for name, expected in test["source_sha256"].items():
        path = root / "scripts" / name
        if not path.exists():
            path = root / "src/sbi4finance" / name
        bindings[str(path.relative_to(root))] = expected
    bindings.update({
        "docs/results/hkde-correction/test-checks.npz": test["checks_sha256"],
        "docs/results/hkde-correction/new-test-panels.npz": test["test_panels_sha256"],
        "docs/results/hkde-correction/test-reference-pit.npz": audit["reference_pit_checks_sha256"],
        "docs/results/hkde-correction/validation.json": test["validation_report_sha256"],
    })
    for name, expected in bindings.items():
        if sha(root / name) != expected:
            raise ValueError(f"Changed historical input/source: {name}")
    for name in ("test.json", "test-independent-audit.json"):
        bindings[str((directory / name).relative_to(root))] = sha(directory / name)
    checks = load_npz(directory / "test-checks.npz")
    panels = load_npz(directory / "new-test-panels.npz")
    pit_cache = load_npz(directory / "test-reference-pit.npz")
    grid = load_npz(root / "docs/results/hkde-inference/reference-grid.npz")
    n, dimensions = checks["calibration_truth"].shape
    assert (n, dimensions) == (256, 2)
    assert np.array_equal(checks["calibration_ids"], panels["example_id"])
    assert np.array_equal(checks["calibration_truth"], panels["parameters"])
    assert np.array_equal(checks["case_ids"], panels["example_id"][:16])
    assert np.array_equal(checks["observations"], panels["observations"][:16])
    assert np.array_equal(checks["truth"], panels["parameters"][:16])
    assert np.array_equal(pit_cache["case_ids"], panels["example_id"])
    assert np.array_equal(pit_cache["truth"], panels["parameters"])
    samples = test["calibration"]["draws_per_case"]
    alpha = test["calibration"]["sbc_family_alpha"]
    bound = np.sqrt(np.log(2*dimensions/alpha)/(2*n))
    assert bound == test["calibration"]["rank_dkw_bound"]
    jitter_seed = int(np.random.SeedSequence([protocol["root_seed"], 1, 3, 0]).generate_state(1)[0])
    # Replay the continuous/no-tie case; original individual draws are absent.
    # Matching jitter supports this assumption but cannot inspect missing draws.
    # integers with high==1 returns zeros, matching the diagnostic RNG call.
    rng = np.random.default_rng(jitter_seed)
    rng.integers(0, np.ones((n, dimensions), dtype=int), size=(n, dimensions))
    randomized = (checks["calibration_ranks"] + rng.random((n, dimensions))) / (samples+1)
    np.testing.assert_array_equal(randomized, checks["calibration_randomized_ranks"])
    np.testing.assert_array_equal(randomized, pit_cache["sir_randomized_ranks"])
    np.testing.assert_array_equal(ks(randomized.T), test["calibration"]["rank_ks"])
    axes = (grid["axis0"], grid["axis1"])
    expected_rows = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    np.testing.assert_array_equal(grid["parameters"], expected_rows)
    sigma = test["task"]["context"]["quote_noise_std"]
    pits = {}
    for stride in (1, 2):
        use_axes = tuple(axis[::stride] for axis in axes)
        prices = grid["prices"].reshape(129, 129, 9)[::stride, ::stride].reshape(-1, 9)
        size = len(use_axes[0])
        pits[f"reference_pit_{size}"] = pit_from_prices(use_axes, prices, panels["observations"], panels["parameters"], sigma)
        np.testing.assert_allclose(pits[f"reference_pit_{size}"], pit_cache[f"reference_pit_{size}"], rtol=0, atol=2e-14)

    oracle, arrays = oracle_controls(pits["reference_pit_129"], randomized, samples, bound)
    uniform_rng = np.random.default_rng(np.random.SeedSequence([SETTINGS["seed"], 2]))
    null_ks = np.concatenate([ks(uniform_rng.random((min(SETTINGS["chunk_size"], SETTINGS["uniform_replicates"]-i), n)))
                             for i in range(0, SETTINGS["uniform_replicates"], SETTINGS["chunk_size"])])
    null_fraction = float(np.mean(null_ks > bound))
    pool_rows, pool_arrays = conditional_pool_checks(checks, pits["reference_pit_129"])
    noise = (panels["observations"][..., 0] - panels["reference_prices"]) / sigma
    arrays.update(pits, **pool_arrays, sir_randomized_ranks=randomized, uniform_null_ks=null_ks,
                  case_ids=panels["example_id"], truth=panels["parameters"])
    report = {
        "scope": "Post-hoc saved-cohort gate/reference audit; no new acceptance test or override",
        "settings": SETTINGS, "input_sha256": bindings,
        "source_sha256": {str(Path(__file__).relative_to(root)): sha(__file__)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "historical_test_passed": test["passed"], "scientific_acceptance_changed": False,
        "new_model_panels": 0, "new_estimator_proposals": 0, "training_runs": 0,
        "gate": {"cases": n, "coordinates": dimensions, "posterior_draws": samples,
                 "family_alpha": alpha, "dkw_bound": float(bound),
                 "ideal_uniform_coordinate_exact_rejection_probability": float(kstwo.sf(bound, n)),
                 "ideal_two_coordinate_rejection_upper_bound": float(min(1, dimensions*kstwo.sf(bound, n))),
                 "uniform_control_rejection_fraction": null_fraction,
                 "uniform_control_fraction_mcse": float(np.sqrt(null_fraction*(1-null_fraction)/len(null_ks))),
                 "observed_sir_ks_ideal_uniform_tail_probabilities": kstwo.sf(ks(randomized.T), n).tolist(),
                 "tail_probability_scope": "Ideal uniform null only; post-hoc, not chance algorithm is wrong"},
        "reconstruction": {"historical_hashes_match": True, "ranks_jitter_and_ks_match": True,
                           "independent_marginal_integration_matches_saved_pit": True,
                           "pit_65_to_129_max_change": np.max(np.abs(pits["reference_pit_129"]-pits["reference_pit_65"]), axis=0).tolist()},
        "conditional_oracle": oracle, "saved_conditional_pool_cdf": pool_rows,
        "cohort_noise_description": {"standardized_quote_error_means_by_contract": noise.mean(axis=0).tolist(),
                                     "overall_mean": float(noise.mean()),
                                     "scope": "Post-hoc description of existing cohort, not a selected statistical test or causal attribution"},
        "limits": ["Numerical-reference PIT is not analytic truth and may share model/pricing errors.",
                   "Conditional binomial controls reproduce one marginal at a time, not cross-parameter rank dependence.",
                   "Controls condition on the saved cohort; they do not establish population calibration or fresh-cohort reproducibility.",
                   "Oracle rank controls omit finite importance-pool error; excess SIR dispersion need not imply systematic bias.",
                   "Saved 16 conditional clouds use different seeds from the 256 calibration clouds; 2048/8192 pools are nested.",
                   "SNIS CDF standard errors are asymptotic plug-in diagnostics, not simultaneous confidence guarantees.",
                   "All256 calibration clouds and individual SIR draws were not saved; aggregate ranks cannot fully separate pool error and resampling.",
                   "Original thresholds, test outcome, reserved tests, neural checkpoint and participant gate remain unchanged."],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "checks.npz", **arrays)
    figures(args.out, report, arrays)
    report["artifact_sha256"] = {path.name: sha(path) for path in sorted(args.out.iterdir())}
    report["elapsed_seconds"] = time.perf_counter() - before
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("gate", "conditional_oracle", "saved_conditional_pool_cdf", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
