"""Explicit BayesFlow routes. Imports are lazy; saved exploration needs no backend."""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import model


def build_workflow(*, small=False):
    import bayesflow as bf

    # Bounds are transformed by the adapter and reversed when sampling.
    adapter = (bf.Adapter().convert_dtype("float64", "float32")
               .constrain("sigma_bar", lower=.005, upper=.025)
               .constrain("q", lower=.52, upper=.90)
               .constrain("nu", lower=2.05, upper=12.)
               .concatenate(model.PARAMETERS, into="inference_variables")
               .rename("log_d", "inference_conditions")
               .as_time_series("returns").rename("returns", "summary_variables"))
    # A full-window encoder. This model is not trained on masked prefixes.
    summary = (bf.networks.FusionTransformer(summary_dim=16, embed_dims=(16, 16),
                                            num_heads=(2, 2), template_dim=32, dropout=.1)
               if small else bf.networks.FusionTransformer(summary_dim=16, dropout=.1))
    flow = bf.networks.CouplingFlow(transform="spline", depth=2 if small else 4)
    return bf.BasicWorkflow(adapter=adapter, summary_network=summary,
                            inference_network=flow, standardize="all")


def sample(estimator, returns, d=8, *, count=512, seed=20261112, batch_size=40):
    count = model._count(count, "posterior draw count")
    named = estimator.sample(conditions=model.conditions(returns, d), num_samples=count,
                             seed=seed, batch_size=batch_size)
    rows = model.parameter_array(named)
    if rows.shape != (len(model.conditions(returns, d)["returns"]), count, 4):
        raise ValueError("posterior has an unexpected shape or target order")
    for j, (lo, hi) in enumerate(model.BOUNDS):
        if lo is not None and (np.any(rows[..., j] < lo) or np.any(rows[..., j] > hi)):
            raise ValueError("posterior is outside the matching training support")
    return rows


def load_checkpoint(path, expected_sha256):
    """Load only a checkpoint whose bytes match the accepted lesson's identity."""
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "model.keras"
    from sbi4f_tutorial.shared.paths import ASSETS
    if not source.is_relative_to((ASSETS / "local-checkpoints").resolve()):
        raise ValueError("place this optional checkpoint inside tutorial/assets/local-checkpoints/")
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("checkpoint missing or SHA-256 differs from this lesson's supplied estimator")
    workflow = build_workflow()
    # Full-model restoration replaces the estimator, including its fitted adapter
    # and standardization. No dummy training or weights-only reconstruction.
    workflow.load_estimator(source)
    return workflow


def freeze_rule(n=200, draws=512):
    """Record the decision before any reserved-cohort inference takes place."""
    return {"format": "mmar_calibration_rule_v1", "histories_per_resolution": n,
            "posterior_draws": draws, "resolutions": list(model.CAPSTONE_D),
            "parameter_order": list(model.PARAMETERS), "rank_null": f"discrete uniform integers 0 through {draws}",
            "rank_statistic": "maximum absolute ECDF deviation on all integer rank support points",
            "null_replications": 10000, "null_seed": 720061112,
            "p_value": "(1 + number of null statistics >= observed)/(10000 + 1)",
            "decision": "Holm familywise alpha .05 across four parameters x three resolutions",
            "coverage": "central 90% empirical coverage; Wilson 95% descriptive interval, no second pass/fail rule",
            "test_seeds": {str(d): 63061112 + d for d in model.CAPSTONE_D},
            "sampling_seeds": {str(d): 64061112 + d for d in model.CAPSTONE_D},
            "cohort_use": "reserved evaluation, not used to tune or select seeds"}


def rule_json():
    return json.dumps(freeze_rule(), sort_keys=True, indent=2)
