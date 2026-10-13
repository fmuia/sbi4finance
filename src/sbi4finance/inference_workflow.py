"""Application-independent lifecycle for one conditional inference problem.

Architectures, likelihoods, diagnostics and downstream estimands belong to the
application hooks. This module owns the order, joint-row/weight contract,
explicit correction/fallback decisions, and the report. Passing local numerical
diagnostics never establishes population calibration or scientific acceptance.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class JointDraws:
    """Matched joint rows; optional SNIS weights belong to entire rows.

    Values are in the declared application coordinates. Converting a weighted
    posterior into equal-weight draws is an explicit resampling operation;
    consumers must not silently discard weights or shuffle columns.
    """

    values: np.ndarray
    names: tuple[str, ...]
    weights: np.ndarray | None = None

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float64).copy()
        names = tuple(self.names)
        if (values.ndim != 2 or not len(values) or values.shape[1] != len(names)
                or not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
                or not np.isfinite(values).all()):
            raise ValueError("finite nonempty joint rows [N,D] and D unique names required")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", names)
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64).copy()
            if (weights.shape != (len(values),) or not np.isfinite(weights).all()
                    or np.any(weights < 0) or not np.any(weights > 0)):
                raise ValueError("one finite nonnegative weight per joint row, with positive total, required")
            weights /= weights.max()
            weights /= weights.sum()
            weights.setflags(write=False)
            object.__setattr__(self, "weights", weights)

    @property
    def normalized_weights(self):
        return np.full(len(self.values), 1 / len(self.values)) if self.weights is None else self.weights

    def resample(self, count, seed):
        """Sample whole rows with replacement, including their continuation state."""
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("positive integer resampling count required")
        indices = np.random.default_rng(seed).choice(len(self.values), count, p=self.normalized_weights)
        return JointDraws(self.values[indices], self.names)

    def summary(self):
        return {"names": list(self.names), "count": len(self.values),
                "representation": "weighted_joint_rows" if self.weights is not None else "equal_weight_joint_rows",
                "mean": (self.normalized_weights @ self.values).tolist()}


@dataclass(frozen=True)
class Diagnostic:
    name: str
    status: str  # passed / flagged / not_run
    metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    scope: str = "estimator"  # observation flags cannot be cured by changing an estimator
    explanation: str = ""

    def __post_init__(self):
        if self.status not in {"passed", "flagged", "not_run"}:
            raise ValueError("diagnostic status must be passed, flagged or not_run")
        if self.scope not in {"observation", "estimator"}:
            raise ValueError("diagnostic scope must be observation or estimator")
        if not self.name:
            raise ValueError("diagnostic name required")

    def report(self):
        return {"name": self.name, "status": self.status, "scope": self.scope,
                "metrics": self.metrics, "thresholds": self.thresholds,
                "explanation": self.explanation}


@dataclass(frozen=True)
class PreparedObservation:
    values: Any
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Estimate:
    draws: JointDraws
    method: str
    diagnostics: tuple[Diagnostic, ...] = ()
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowHooks:
    prepare: Callable
    infer: Callable
    diagnose: Callable
    downstream: Callable
    correct: Callable | None = None
    fallback: Callable | None = None


@dataclass(frozen=True)
class WorkflowPolicy:
    on_flag: str = "report"

    def __post_init__(self):
        if self.on_flag not in {"report", "fallback", "raise"}:
            raise ValueError("on_flag must be report, fallback or raise")


@dataclass(frozen=True)
class WorkflowResult:
    report: dict
    posterior: JointDraws


class FlaggedInferenceError(RuntimeError):
    """A declared fail-closed policy stopped before downstream propagation."""

    def __init__(self, report):
        super().__init__("inference diagnostics flagged this case; downstream calculation declined")
        self.report = report


def run_workflow(observation, hooks, *, application, policy=None, provenance=None):
    """Execute prepare -> infer -> diagnose -> correct/fallback -> propagate.

    ``diagnose(prepared, estimate)`` returns application-specific diagnostics.
    ``correct`` and ``fallback`` return a new Estimate, then run the same
    diagnostic stage. Historical stage failures stay in the report; selected
    estimator diagnostics determine current numerical reliability. Observation
    failures persist across estimators. A missing/unrun check is never a pass.
    """
    policy = WorkflowPolicy() if policy is None else policy
    prepared = hooks.prepare(observation)
    if not isinstance(prepared, PreparedObservation):
        raise TypeError("prepare must return PreparedObservation")
    stages = []
    observation_diagnostics = []

    def inspect(stage, estimate):
        if not isinstance(estimate, Estimate):
            raise TypeError(f"{stage} must return Estimate")
        if stages and estimate.draws.names != tuple(stages[0]["posterior"]["names"]):
            raise ValueError("correction/fallback must preserve joint coordinate names and order")
        diagnostics = tuple(estimate.diagnostics) + tuple(hooks.diagnose(prepared, estimate))
        if any(not isinstance(item, Diagnostic) for item in diagnostics):
            raise TypeError("diagnose must return Diagnostic objects")
        observation_diagnostics.extend(item for item in diagnostics if item.scope == "observation")
        stages.append({"stage": stage, "method": estimate.method,
                       "posterior": estimate.draws.summary(),
                       "diagnostics": [item.report() for item in diagnostics],
                       "details": estimate.details})
        return estimate, diagnostics

    selected, diagnostics = inspect("amortized", hooks.infer(prepared))
    if hooks.correct is not None:
        selected, diagnostics = inspect("correction", hooks.correct(prepared, selected))
    selected_stage = stages[-1]["stage"]
    estimator_flagged = any(item.status == "flagged" and item.scope == "estimator" for item in diagnostics)
    decision = "retained"
    if estimator_flagged and policy.on_flag == "fallback":
        if hooks.fallback is None:
            decision = "fallback_unavailable"
        else:
            selected, diagnostics = inspect("fallback", hooks.fallback(prepared, selected))
            selected_stage = "fallback"
            decision = "fallback_run"
    relevant = observation_diagnostics + [item for item in diagnostics if item.scope == "estimator"]
    flagged = [item.name for item in relevant if item.status == "flagged"]
    checked = any(item.status == "passed" and item.scope == "estimator" for item in diagnostics)
    # A skipped required estimator check prevents a locally_checked label.
    incomplete = any(item.status == "not_run" and item.scope == "estimator" for item in diagnostics)
    reliability = "flagged" if flagged else "locally_checked" if checked and not incomplete else "unassessed"
    if flagged and decision == "retained":
        decision = "flagged_report_only"
    report = {
        "schema_version": "inference_workflow_v1", "application": application,
        "provenance": {**(provenance or {}), "observation": prepared.provenance},
        "policy": {"on_flag": policy.on_flag}, "stages": stages,
        "selected_stage": selected_stage, "method": selected.method,
        "reliability": reliability, "diagnostic_reasons": list(dict.fromkeys(flagged)),
        "decision": decision, "scientific_acceptance": "not_assessed_by_this_run",
        "scope": "Local numerical diagnostics are not a calibration, model-adequacy or scientific-acceptance certificate.",
        "posterior": selected.draws.summary(),
    }
    if flagged and policy.on_flag == "raise":
        report["decision"] = "downstream_declined"
        raise FlaggedInferenceError(report)
    downstream = hooks.downstream(prepared, selected)
    if not isinstance(downstream, dict):
        raise TypeError("downstream must return a report dictionary")
    report["downstream"] = {**downstream, "reliability": reliability,
                            "diagnostic_reasons": report["diagnostic_reasons"],
                            "scientific_acceptance": report["scientific_acceptance"]}
    return WorkflowResult(report, selected.draws)
