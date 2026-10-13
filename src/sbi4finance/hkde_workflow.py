"""HKDE application of the common conditional-inference lifecycle.

The existing flow and exact declared Fourier/Gaussian correction are unchanged.
SNIS weights travel with matched (theta, jump_intensity) rows into target prices.
Local weight diagnostics and historical scientific acceptance remain distinct.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np

from . import hkde_correction as correction, hkde_inference as inference, hkde_reference as reference
from .inference_workflow import (
    Diagnostic, Estimate, JointDraws, PreparedObservation, WorkflowHooks,
    WorkflowPolicy, run_workflow,
)


@dataclass(frozen=True)
class HKDEWorkflowConfig:
    n_draws: int = 2048
    n_proposals: int = 2048
    seed: int = 0
    run_correction: bool = True
    prior_fraction: float = .1
    boundary_fraction: float = .1
    min_ess_fraction: float = .5
    max_weight: float = .01
    on_flag: str = "report"

    def __post_init__(self):
        for name in ("n_draws", "n_proposals", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        for name in ("min_ess_fraction", "max_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if (not np.isfinite([self.prior_fraction, self.boundary_fraction]).all()
                or not 0 < self.prior_fraction < 1 or self.boundary_fraction < 0
                or self.prior_fraction + self.boundary_fraction >= 1):
            raise ValueError("invalid defensive-mixture fractions")
        WorkflowPolicy(self.on_flag)


def run_panel(observation, posterior, task, *, checkpoint_task, config=None,
              target_contracts=None, target_price_fn=None, diagnostic_fn=None,
              provenance=None):
    """Run a complete noisy-price panel through the shared workflow.

    ``checkpoint_task`` must be the full task read from the selected checkpoint;
    matching only shape/adapter ID is insufficient. ``target_price_fn(rows)``
    may replace downstream Fourier prices with an independently accepted target
    surrogate. It never replaces the likelihood. ``diagnostic_fn`` can attach
    independent reference checks as Diagnostic objects; no reference is implied
    by a high ESS. No generating parameters enter the conditioning dictionary.
    """
    config = HKDEWorkflowConfig() if config is None else config
    if checkpoint_task != task:
        raise ValueError("complete application task differs from checkpoint task")
    contracts = list(reference.DEFAULT_TARGET_CONTRACTS if target_contracts is None else target_contracts)
    if not contracts:
        raise ValueError("at least one downstream target contract required")
    task_sha = hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def prepare(value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (9, 1):
            raise ValueError("one complete HKDE observation [9,1] required")
        inference.conditioning_data(value[None], task)
        return PreparedObservation(value.copy(), {
            "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
            "shape": [9, 1], "units": "currency", "measure": "Q",
            "adapter_id": inference.ADAPTER_ID, "task_sha256": task_sha,
        })

    def infer(prepared):
        sampled = posterior.sample(num_samples=config.n_draws,
                                   conditions=inference.conditioning_data(prepared.values[None], task),
                                   seed=config.seed)
        pieces = [np.asarray(sampled[name], dtype=float) for name in inference.PARAMETER_NAMES]
        if any(piece.shape != (1, config.n_draws, 1) for piece in pieces):
            raise ValueError("flow must return matched named draws [1,N,1]")
        draws = JointDraws(np.concatenate(pieces, axis=-1)[0], inference.PARAMETER_NAMES)
        bounds = np.asarray(task["prior_bounds"])
        if np.any(draws.values <= bounds[:, 0]) or np.any(draws.values >= bounds[:, 1]):
            raise ValueError("network draws must be strictly inside the declared prior")
        return Estimate(draws, "amortized_flow")

    def correct(prepared, estimate):
        proposals = correction.propose(posterior, task, prepared.values,
                                       config.n_proposals, config.seed,
                                       config.prior_fraction, config.boundary_fraction)
        likelihood_start = time.perf_counter()
        weighted = correction.correct(proposals["parameters"], proposals["log_proposal"],
                                      prepared.values, task)
        return Estimate(JointDraws(proposals["parameters"], inference.PARAMETER_NAMES, weighted["weights"]),
                        "defensive_importance_sampling", details={
                            "prior_rows": int(proposals["is_prior"].sum()),
                            "boundary_rows": int(proposals["is_boundary"].sum()),
                            "timing_seconds": {**proposals["timing_seconds"], "likelihood": time.perf_counter() - likelihood_start},
                            "ess": weighted["ess"], "ess_fraction": weighted["ess_fraction"],
                            "max_weight": weighted["max_weight"],
                            "proposal_fractions": {"prior": config.prior_fraction, "boundary": config.boundary_fraction},
                            "likelihood": "declared_refined_fourier_independent_gaussian",
                        })

    def diagnose(prepared, estimate):
        diagnostics = []
        if estimate.method == "defensive_importance_sampling":
            details = estimate.details
            passed = details["ess_fraction"] >= config.min_ess_fraction and details["max_weight"] <= config.max_weight
            diagnostics.append(Diagnostic("importance_weight_concentration", "passed" if passed else "flagged",
                metrics={key: details[key] for key in ("ess", "ess_fraction", "max_weight")},
                thresholds={"min_ess_fraction": config.min_ess_fraction, "max_weight": config.max_weight},
                explanation="Finite-pool weight concentration only; independent reference checks, numerical refinement and population calibration remain separate."))
        if diagnostic_fn is not None:
            diagnostics.extend(diagnostic_fn(prepared, estimate))
        if not any(item.scope == "estimator" for item in diagnostics):
            diagnostics.append(Diagnostic("amortized_accuracy", "not_run",
                explanation="No case-specific accuracy certificate follows from loading a trained network."))
        return diagnostics

    def downstream(prepared, estimate):
        draws = estimate.draws
        targets = np.asarray(reference.price_parameters(task, draws.values, contracts)
                             if target_price_fn is None else target_price_fn(draws.values), dtype=float)
        if targets.shape != (len(draws.values), len(contracts)) or not np.isfinite(targets).all():
            raise ValueError("one finite ordered target-price row per joint parameter row required")
        weights = draws.normalized_weights
        return {"estimand": "Q_posterior_distribution_of_model_prices", "units": "currency",
                "target_contracts": contracts, "probabilities": [.05, .5, .95],
                "parameter_names": list(draws.names),
                "parameter_quantiles": reference.weighted_quantiles(draws.values, weights).tolist(),
                "parameter_mean": (weights @ draws.values).tolist(),
                "target_mean": (weights @ targets).tolist(),
                "target_quantiles": reference.weighted_quantiles(targets, weights).tolist(),
                "weight_handling": "weighted_joint_rows_without_resampling",
                "pricing_method": "refined_fourier" if target_price_fn is None else "caller_supplied_target_price_function"}

    return run_workflow(observation, WorkflowHooks(prepare, infer, diagnose, downstream,
                        correct=correct if config.run_correction else None),
                        application="hkde_option_panel_v1", policy=WorkflowPolicy(config.on_flag),
                        provenance={**(provenance or {}), "config": asdict(config), "task_sha256": task_sha})
