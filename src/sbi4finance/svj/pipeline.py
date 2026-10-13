"""Daily SVJ application of the shared conditional-inference lifecycle.

The neural posterior, particle importance correction, PMMH and conditional risk
algorithms remain application components. Their execution, diagnostic decisions
and evidence reporting follow the same core as the HKDE option-panel example.
The compatibility entry point still accepts 1,000 percent log returns and
returns ``(report, equal_weight_joint_draws)``.
"""

import hashlib
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..inference_workflow import (
    Diagnostic, Estimate, JointDraws, PreparedObservation, WorkflowHooks,
    WorkflowPolicy, run_workflow,
)
from . import model, risk
from .audit import DEFAULT_THRESHOLD, audit
from .pmmh import run_pmmh


@dataclass
class PipelineConfig:
    n_draws: int = 1_000
    run_audit: bool = True
    n_proposals: int = 3_000
    audit_threshold: float = DEFAULT_THRESHOLD
    audit_particles: int | None = None
    loglik_sd_threshold: float = 1.5
    misspecification_threshold: float = .05  # retained historical summary-space screen
    pmmh_on_flag: bool = False
    pmmh_iterations: int = 12_500
    pmmh_burn_in: int = 2_000
    pmmh_max_rhat: float = 1.01
    pmmh_min_ess: float = 400.
    on_flag: str | None = None
    horizons: tuple = risk.HORIZONS
    n_paths: int = 20
    variance_split: bool = True
    seed: int = 0

    def __post_init__(self):
        for name in ("n_draws", "n_proposals", "n_paths", "seed", "pmmh_iterations", "pmmh_burn_in"):
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "pmmh_burn_in"} else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.audit_particles is not None and (isinstance(self.audit_particles, bool) or not isinstance(self.audit_particles, (int, np.integer)) or self.audit_particles < 1):
            raise ValueError("audit_particles must be a positive integer or None")
        for name in ("audit_threshold", "misspecification_threshold"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in (0,1)")
        if (not np.isfinite([self.loglik_sd_threshold, self.pmmh_max_rhat, self.pmmh_min_ess]).all()
                or self.loglik_sd_threshold <= 0 or self.pmmh_max_rhat <= 1 or self.pmmh_min_ess <= 0
                or self.pmmh_burn_in >= self.pmmh_iterations):
            raise ValueError("invalid likelihood-noise or PMMH diagnostic settings")
        WorkflowPolicy(self.on_flag or ("fallback" if self.pmmh_on_flag else "report"))


def describe(targets):
    """Posterior medians and 90% intervals in the units of the prior table."""
    q = model.quantities(targets)
    return {
        name: {"median": float(np.median(q[:, j])), "q05": float(np.quantile(q[:, j], 0.05)), "q95": float(np.quantile(q[:, j], 0.95))}
        for j, name in enumerate(model.QUANTITY_LABELS[: q.shape[1]])
    }


def _number(value):
    """Failed numerical diagnostics serialize as null, accompanied by a flag."""
    return float(value) if np.isfinite(value) else None


def run_window(returns, posterior, misspecification=None, config=None, *, provenance=None):
    """Run percent log returns through prepare/infer/check/respond/propagate.

    Reports distinguish unaudited, locally checked and flagged estimates. PMMH
    execution is never itself evidence of convergence. A flagged result may be
    returned for analysis under the default ``report`` policy, with the same
    status attached to its risk output; choose ``on_flag='raise'`` to decline
    downstream calculation. Historical scientific acceptance is not changed.
    """
    config = PipelineConfig() if config is None else config
    if misspecification is not None:
        from .checks import MisspecificationTest
        if isinstance(misspecification, MisspecificationTest):
            misspecification.assert_compatible(posterior)
    seed = config.seed
    timings, legacy, state = {}, {"config": asdict(config)}, {}

    def prepare(value):
        from ..datasets.local import LocalArtifact
        artifact_provenance = {}
        if isinstance(value, LocalArtifact):
            from .adapters import prepared_returns
            artifact_provenance = {"local_artifact": value.metadata}
            value = prepared_returns(value, checkpoint_task=getattr(posterior, "task", None))
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (model.WINDOW,) or not np.isfinite(value).all():
            raise ValueError(f"finite percent log returns for exactly {model.WINDOW} days required, got {value.shape}")
        return PreparedObservation(value.copy(), {
            "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
            "shape": [model.WINDOW], "measure": "P", "units": "percent_log_return",
            "calendar": "252_trading_days_per_year", **artifact_provenance, **(provenance or {}),
        })

    def infer(prepared):
        start = time.process_time()
        values = np.asarray(posterior.sample(prepared.values, num_samples=config.n_draws, seed=seed))
        if values.shape != (1, config.n_draws, model.N_TARGETS):
            raise ValueError("SVJ posterior must return matched joint draws [1,N,9]")
        draws = JointDraws(values[0], model.TARGET_NAMES)
        state["network"] = draws.values
        timings["network_cpu_s"] = time.process_time() - start
        legacy["network_posterior"] = describe(draws.values)
        diagnostics = []
        if misspecification is not None:
            start = time.process_time()
            pvalue = float(misspecification.pvalue(posterior.summaries(prepared.values))[0])
            legacy["misspecification_pvalue"] = _number(pvalue)
            timings["misspecification_cpu_s"] = time.process_time() - start
            diagnostics.append(Diagnostic("summary_space_misspecification", "passed" if np.isfinite(pvalue) and pvalue >= config.misspecification_threshold else "flagged",
                metrics={"pvalue": _number(pvalue)}, thresholds={"min_pvalue": config.misspecification_threshold}, scope="observation",
                explanation="Summary-space compatibility only; changing the posterior estimator cannot cure model misspecification."))
        return Estimate(draws, "amortized_flow", tuple(diagnostics))

    def correct(prepared, estimate):
        try:
            result = audit(estimate.draws.values, prepared.values, seed,
                           n_proposals=config.n_proposals, n_out=config.n_draws,
                           n_particles=config.audit_particles)
            # Validate weighted rows before retaining any correction output.
            draws = JointDraws(result.proposals, model.TARGET_NAMES, np.exp(result.log_weights))
            equal = JointDraws(result.posterior, model.TARGET_NAMES)
        except ValueError as error:
            legacy["audit"] = {"flagged": True, "threshold": config.audit_threshold, "error": str(error)}
            return Estimate(estimate.draws, "amortized_flow_after_failed_audit", (
                Diagnostic("particle_importance_audit_failed", "flagged", explanation=str(error)),))
        timings["audit_cpu_s"] = result.seconds
        finite = np.isfinite([result.efficiency, result.ess, result.loglik_sd]).all()
        flagged = not finite or result.flagged(config.audit_threshold) or result.loglik_sd > config.loglik_sd_threshold
        details = {"efficiency": _number(result.efficiency), "ess": _number(result.ess),
                   "n_particles": result.n_particles, "loglik_sd": _number(result.loglik_sd),
                   "flagged": bool(flagged), "threshold": config.audit_threshold,
                   "loglik_sd_threshold": config.loglik_sd_threshold}
        legacy["audit"] = details
        state["importance_equal"] = equal.values
        return Estimate(draws, "particle_importance_sampling", details=details)

    def fallback(prepared, estimate):
        try:
            chains = run_pmmh(prepared.values, state["network"], seed,
                              n_iter=config.pmmh_iterations, burn_in=config.pmmh_burn_in)
            pooled = JointDraws(chains.pooled(), model.TARGET_NAMES)
        except ValueError as error:
            legacy["pmmh"] = {"flagged": True, "error": str(error)}
            return Estimate(estimate.draws, estimate.method, (
                Diagnostic("particle_mcmc_failed", "flagged", explanation=str(error)),), estimate.details)
        timings["pmmh_cpu_s"] = chains.seconds
        rhat, ess = float(np.max(chains.rhat)), float(np.min(chains.ess_bulk))
        finite = np.isfinite([rhat, ess, chains.loglik_sd]).all()
        flagged = (not finite or rhat > config.pmmh_max_rhat or ess < config.pmmh_min_ess
                   or chains.loglik_sd > config.loglik_sd_threshold or len(pooled.values) < config.n_draws)
        details = {"accept_rate": _number(chains.accept_rate), "max_rhat": _number(rhat),
                   "min_ess_bulk": _number(ess), "n_particles": chains.n_particles,
                   "loglik_sd": _number(chains.loglik_sd), "pooled_draw_count": len(pooled.values), "flagged": bool(flagged)}
        legacy["pmmh"] = details
        count = min(len(pooled.values), config.n_draws)
        state["pmmh_equal"] = pooled.values[np.random.default_rng(seed).choice(len(pooled.values), count, replace=False)]
        return Estimate(pooled, "particle_mcmc", details=details)

    def diagnose(prepared, estimate):
        if any(item.status == "flagged" for item in estimate.diagnostics):
            return ()
        if estimate.method == "particle_importance_sampling":
            return (Diagnostic("particle_importance_concentration_and_noise", "flagged" if estimate.details["flagged"] else "passed",
                metrics={key: estimate.details[key] for key in ("efficiency", "ess", "loglik_sd")},
                thresholds={"min_efficiency": config.audit_threshold, "max_loglik_sd": config.loglik_sd_threshold},
                explanation="Finite weighted-pool concentration and estimated likelihood noise; does not establish posterior accuracy or calibration."),)
        if estimate.method == "particle_mcmc":
            return (Diagnostic("particle_mcmc_mixing_and_noise", "flagged" if estimate.details["flagged"] else "passed",
                metrics={key: estimate.details[key] for key in ("max_rhat", "min_ess_bulk", "loglik_sd", "pooled_draw_count")},
                thresholds={"max_rhat": config.pmmh_max_rhat, "min_ess_bulk": config.pmmh_min_ess,
                            "max_loglik_sd": config.loglik_sd_threshold, "min_pooled_draws": config.n_draws},
                explanation="Declared local mixing/noise checks; successful execution alone is not convergence or scientific acceptance."),)
        return (Diagnostic("amortized_accuracy", "not_run", explanation="No particle likelihood audit or independent conditional check has established local accuracy."),)

    def downstream(prepared, estimate):
        legacy["network_risk"] = risk.risk_numbers(state["network"], config.horizons, config.n_paths, seed)
        if estimate.method == "particle_importance_sampling":
            final, route = state["importance_equal"], "network + importance sampling"
        elif estimate.method == "particle_mcmc":
            final, route = state["pmmh_equal"], "particle MCMC"
        else:
            final, route = estimate.draws.values, "network"
        state["final"] = final
        legacy["route"], legacy["posterior"] = route, describe(final)
        start = time.process_time()
        forecast = risk.forecast(final, prepared.values, seed, config.horizons, config.n_paths, config.variance_split)
        timings["forecast_cpu_s"] = time.process_time() - start
        return {"estimand": "P_conditional_return_risk", "risk": forecast,
                "weight_handling": "joint_resampling_before_path_simulation" if estimate.draws.weights is not None else "equal_weight_joint_rows",
                "propagated_draw_count": len(final)}

    policy = WorkflowPolicy(config.on_flag or ("fallback" if config.pmmh_on_flag else "report"))
    result = run_workflow(returns, WorkflowHooks(prepare, infer, diagnose, downstream,
                          correct=correct if config.run_audit else None,
                          fallback=fallback if config.pmmh_on_flag or policy.on_flag == "fallback" else None),
                          application="svj_daily_returns_v1", policy=policy,
                          provenance={"config": asdict(config),
                                      "checkpoint": getattr(posterior, "artifact", None),
                                      "diagnostic_reference": getattr(misspecification, "binding", None)})
    legacy["workflow"] = result.report
    legacy["status"] = result.report["reliability"]
    legacy["diagnostic_reasons"] = result.report["diagnostic_reasons"]
    legacy["scientific_acceptance"] = result.report["scientific_acceptance"]
    observation_flags = [item["name"] for item in result.report["stages"][0]["diagnostics"]
                         if item["scope"] == "observation" and item["status"] == "flagged"]
    legacy["network_risk_status"] = {"reliability": "flagged" if observation_flags else "unassessed",
                                     "diagnostic_reasons": observation_flags, "method": "amortized_flow",
                                     "scientific_acceptance": "not_assessed_by_this_run",
                                     "definition": "negative_cumulative_percent_log_return_v1",
                                     "unit": "percent_log_loss"}
    legacy["risk"] = {**result.report["downstream"]["risk"], "reliability": legacy["status"],
                      "diagnostic_reasons": legacy["diagnostic_reasons"],
                      "scientific_acceptance": legacy["scientific_acceptance"]}
    legacy["timings"] = {key: round(value, 2) for key, value in timings.items()}
    return legacy, state["final"]


def run_prepared_window(artifact, posterior, misspecification=None, config=None):
    """Use the repository's validated local return artifact as observation input."""
    return run_window(artifact, posterior, misspecification, config)
