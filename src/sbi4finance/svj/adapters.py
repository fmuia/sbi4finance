"""Strict observation and target boundaries for the fixed daily SVJ task."""

from datetime import date

import numpy as np

from ..datasets.local import ArtifactError, LocalArtifact, _check_metadata
from . import model

ADAPTER_ID = "svj_decimal_log_returns_to_native_features_v1"


def _task(task=None):
    from .bank import default_task, validate_task
    expected = default_task() if task is None else task
    validate_task(expected)
    return expected


def _daily_grid(dt_years):
    steps = np.asarray(dt_years, dtype=float)
    if (steps.shape != (model.WINDOW,) or not np.isfinite(steps).all()
            or not np.allclose(steps, 1 / model.DAYS_PER_YEAR, rtol=0, atol=1e-15)):
        raise ValueError("SVJ needs exactly 1,000 daily increments of 1/252 years")


def observations_to_features(observations, dt_years=None):
    """Convert decimal log returns to percent once, then preserve native features."""
    values = np.asarray(observations, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] == 0 or values.shape[1:] != (model.WINDOW, 1)
            or not np.isfinite(values).all()):
        raise ValueError("observations must be finite nonempty decimal log returns [B,1000,1]")
    if dt_years is not None:
        _daily_grid(dt_years)
    features = model.network_input(100 * values[..., 0])
    if not np.isfinite(features).all():
        raise ValueError("decimal observations overflow the native SVJ feature representation")
    return features


def native_features(data):
    """Validate native percent returns or already prepared two-channel features.

Three-dimensional one-channel bank observations are deliberately rejected;
their decimal units must pass through observations_to_features instead.
"""
    data = np.asarray(data)
    if data.ndim in (1, 2):
        data = np.atleast_2d(data)
        if data.shape[1] != model.WINDOW:
            raise ValueError("native SVJ returns require exactly 1,000 observations")
        data = model.network_input(data)
    if (data.ndim != 3 or data.shape[0] == 0 or data.shape[1:] != (model.WINDOW, 2)
            or not np.isfinite(data).all()):
        raise ValueError("expected percent returns [T]/[B,T] or prepared features [B,1000,2]; use the decimal bank adapter for [B,T,1]")
    data = data.astype("float32")
    if not np.isfinite(data).all():
        raise ValueError("features must be representable in float32")
    # The first channel was already rounded to float32 by network_input. Allow
    # that rounding when checking the saved feature contract, not arbitrary channels.
    squared_log = np.log(data[..., 0].astype(float) ** 2 + 0.01)
    if not np.allclose(data[..., 1], squared_log, rtol=2e-6, atol=2e-6):
        raise ValueError("prepared feature channel must equal log(percent_return squared + 0.01)")
    return data


def bank_batch(split, task=None):
    """Select observations as features and parameter/state truth as targets only."""
    _task(task)
    features = observations_to_features(split["observations"], split["dt_years"])
    parameters = np.asarray(split["parameters"], dtype=np.float64)
    state = np.asarray(split["terminal_state"], dtype=np.float64)
    if (parameters.shape != (len(features), model.N_PARAMS) or not np.isfinite(parameters).all()
            or state.shape != (len(features), 1) or not np.isfinite(state).all()
            or np.any(state < model.V_MIN)):
        raise ValueError("finite native [B,8] coordinates and floored percent-squared/day [B,1] state required")
    targets = np.concatenate([parameters, model.log_spot_vol(state)], axis=1)
    return {"targets": targets, "features": features}


def prepared_returns(artifact: LocalArtifact, *, checkpoint_task=None):
    """Validate a prepared LocalArtifact and return native percent log returns.

The explicit daily schedule is authoritative; calendar days are labels and no
holidays, missing values or observation intervals are inferred or repaired.
"""
    if not isinstance(artifact, LocalArtifact):
        raise ArtifactError("expected a prepared LocalArtifact")
    meta = artifact.metadata
    _check_metadata(meta)
    task = _task(checkpoint_task)
    for key, value in (("artifact_type", "return_window"), ("representation", "log_returns"),
                       ("measure", "P"), ("task_id", task["task_id"]), ("scientific_task", task)):
        if meta.get(key) != value:
            raise ArtifactError(f"incompatible {key} for the complete daily SVJ task")
    conventions = meta["conventions"]
    for key, value in (("return_kind", "log"), ("return_unit", "decimal"),
                       ("normalization_version", 1), ("n_returns", model.WINDOW)):
        if conventions.get(key) != value:
            raise ArtifactError(f"incompatible {key}: expected {value}")
    columns = {column["name"]: column for column in meta["columns"]}
    required = {
        "row_id": ("str", "id", None), "interval_start": ("str", "auxiliary", None),
        "interval_end": ("str", "auxiliary", None), "dt_years": ("float", "context", "years"),
        "return": ("float", "observation", "decimal"), "observed": ("bool", "auxiliary", None),
    }
    for name, (dtype, role, unit) in required.items():
        col = columns.get(name, {})
        if (col.get("dtype") != dtype or col.get("role") != role or col.get("nullable", False)
                or (unit is not None and col.get("unit") != unit)):
            raise ArtifactError(f"incompatible column: {name}")
    if any(col["role"] != "auxiliary" for name, col in columns.items() if name not in required):
        raise ArtifactError("extra columns may be auxiliary only")
    if len(artifact.rows) != model.WINDOW:
        raise ArtifactError("SVJ requires a complete 1,000-return window")
    values, steps, ids = [], [], set()
    previous = conventions.get("window_start")
    try:
        for row in artifact.rows:
            if not isinstance(row, dict) or set(row) != set(columns):
                raise ArtifactError("row keys must match declared columns")
            start, end = date.fromisoformat(row["interval_start"]), date.fromisoformat(row["interval_end"])
            if (start.isoformat() != previous or start >= end or end.isoformat() != row["interval_end"]
                    or row["observed"] is not True or not isinstance(row["row_id"], str)
                    or not row["row_id"] or row["row_id"] in ids):
                raise ArtifactError("window must have complete, ordered, contiguous daily intervals and unique identities")
            ids.add(row["row_id"])
            previous = end.isoformat()
            if isinstance(row["return"], bool) or isinstance(row["dt_years"], bool):
                raise ArtifactError("returns and year fractions must be finite numbers")
            values.append(float(row["return"]))
            steps.append(float(row["dt_years"]))
        _daily_grid(steps)
    except (TypeError, KeyError, ValueError) as exc:
        raise ArtifactError(f"invalid prepared SVJ window: {exc}") from exc
    if previous != conventions.get("conditioning_cutoff") or not np.isfinite(values).all():
        raise ArtifactError("finite returns ending at the declared cutoff are required")
    returns = 100 * np.asarray(values, dtype=np.float64)
    if not np.isfinite(returns).all():
        raise ArtifactError("prepared observations overflow percent log returns")
    return returns


def prepared_features(artifact, *, checkpoint_task=None):
    return model.network_input(prepared_returns(artifact, checkpoint_task=checkpoint_task)[None])
