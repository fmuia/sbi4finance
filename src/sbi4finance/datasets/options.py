"""Local panels for the restricted HKDE premium example; no pricing or adapter."""

import hashlib
import json
import math

from .local import ArtifactError, LocalArtifact, _check_metadata


_CONVENTIONS = {
    "quote_form": "single_premium",
    "price_unit": "currency",
    "exercise": "European",
    "settlement": "cash",
    "premium_basis": "per_underlying_unit",
}
_COLUMNS = [
    {"name": "contract_id", "dtype": "str", "role": "id"},
    {"name": "strike", "dtype": "float", "role": "context", "unit": "currency"},
    {"name": "maturity", "dtype": "float", "role": "context", "unit": "years"},
    {"name": "option_type", "dtype": "str", "role": "context"},
    {"name": "price", "dtype": "float", "role": "observation", "unit": "currency"},
    {"name": "quote_time", "dtype": "str", "role": "auxiliary", "nullable": True},
]


def _copy_json(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ArtifactError("metadata/task must contain finite JSON values") from exc


def _finite(value, name):
    if value is None or isinstance(value, bool):
        raise ArtifactError(f"{name} must be a complete finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError(f"{name} must be a complete finite number") from exc
    if not math.isfinite(result):
        raise ArtifactError(f"{name} must be a complete finite number")
    return result


def _contracts(task):
    # Scientific parameter/pricing validation remains in the owning simulator
    # adapter. Here we restrict the local interface and check its coordinates.
    if (not isinstance(task, dict) or task.get("task_id") != "hkde_option_panel_v1"
            or task.get("model") != "hkde" or task.get("measure") != "Q"
            or task.get("parameter_names") != ["theta", "jump_intensity"]):
        raise ArtifactError("expected the restricted HKDE Q theta/jump_intensity task")
    required = ("context", "model_context", "fixed_parameters", "numerics", "prior_bounds")
    if any(name not in task for name in required) or not isinstance(task["context"], dict):
        raise ArtifactError("the complete scientific task is required")
    contracts = task["context"].get("contracts")
    if not isinstance(contracts, list) or len(contracts) != 9:
        raise ArtifactError("HKDE requires nine ordered contracts")
    seen = set()
    for contract in contracts:
        if not isinstance(contract, dict) or set(contract) != {"strike", "maturity", "option_type"}:
            raise ArtifactError("contracts need strike, maturity and option_type")
        strike, maturity = (_finite(contract[name], name) for name in ("strike", "maturity"))
        if strike <= 0 or maturity <= 0 or contract["option_type"] not in ("call", "put"):
            raise ArtifactError("positive strikes/maturities and European calls/puts required")
        key = (strike, maturity, contract["option_type"])
        if key in seen:
            raise ArtifactError("duplicate contract coordinates")
        seen.add(key)
    return contracts


def hkde_panel_prices(artifact: LocalArtifact, *, checkpoint_task: dict) -> tuple[float, ...]:
    """Validate one local panel against the FULL saved task, then select prices.

    This returns raw prices, not neural conditions. Call the simulator's own
    ``hkde_inference.conditioning_data`` on a [batch,9,1] array afterwards.
    Matching this boundary does not accept the checkpoint's statistical quality.
    ``checkpoint_task`` must come from its saved metadata, not inferred quotes.
    """
    meta = artifact.metadata
    _check_metadata(meta)
    expected_task = _copy_json(checkpoint_task)
    contracts = _contracts(expected_task)
    if meta.get("scientific_task") != expected_task:
        raise ArtifactError("scientific_task differs from the complete checkpoint task")
    for key, expected in (("artifact_type", "option_panel"), ("representation", "raw_quote"),
                          ("measure", "Q"), ("task_id", expected_task["task_id"])):
        if meta.get(key) != expected:
            raise ArtifactError(f"incompatible {key}; expected {expected}")
    for key, expected in _CONVENTIONS.items():
        if meta["conventions"].get(key) != expected:
            raise ArtifactError(f"incompatible {key}; expected {expected}")
    columns = {col["name"]: col for col in meta["columns"]}
    for expected in _COLUMNS:
        if expected["name"] == "quote_time" and "quote_time" not in columns:
            continue  # Optional acquisition metadata is not a model input.
        actual = columns.get(expected["name"], {})
        if (any(actual.get(key) != value for key, value in expected.items())
                or actual.get("nullable", False) != expected.get("nullable", False)):
            raise ArtifactError(f"incompatible column declaration: {expected['name']}")
    required_names = {col["name"] for col in _COLUMNS}
    if any(col["role"] != "auxiliary" for name, col in columns.items() if name not in required_names):
        raise ArtifactError("extra columns must be auxiliary, not conditioning")
    if len(artifact.rows) != len(contracts):
        raise ArtifactError("panel needs exactly nine rows, without missing or extra contracts")
    prices, ids = [], set()
    for row, contract in zip(artifact.rows, contracts):
        if not isinstance(row, dict) or set(row) != set(columns):
            raise ArtifactError("row keys must match declared columns")
        identity = row["contract_id"]
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ArtifactError("contract IDs must be nonempty and unique")
        ids.add(identity)
        strike, maturity = (_finite(row[name], name) for name in ("strike", "maturity"))
        if {"strike": strike, "maturity": maturity, "option_type": row["option_type"]} != contract:
            raise ArtifactError("contract coordinates/order differ from the checkpoint task")
        timestamp = row.get("quote_time")
        if timestamp is not None and (not isinstance(timestamp, str) or not timestamp):
            raise ArtifactError("quote_time must be a nonempty string or null")
        prices.append(_finite(row["price"], "price"))
    return tuple(prices)


def prepare_hkde_panel(records, *, metadata: dict, task: dict) -> LocalArtifact:
    """Preserve nine ordered single-premium records under their declared task.

    The task describes the SOURCE experiment, not a desired destination to
    force arbitrary market quotes into. Only named contract/price fields and
    optional quote_time are copied. No sorting, interpolation, IV conversion,
    bid/ask invention, curve estimation or clipping is performed. In particular,
    negative noisy premiums are retained under the explicit Gaussian model.
    """
    meta, scientific_task = _copy_json(metadata), _copy_json(task)
    _contracts(scientific_task)
    if not isinstance(meta, dict):
        raise ArtifactError("metadata must be an object")
    for key, expected in (("artifact_type", "option_panel"), ("representation", "raw_quote"),
                          ("measure", "Q"), ("task_id", scientific_task["task_id"]),
                          ("scientific_task", scientific_task)):
        if key in meta and meta[key] != expected:
            raise ArtifactError(f"metadata {key} conflicts with the source task")
        meta[key] = expected
    if "columns" in meta and meta["columns"] != _COLUMNS:
        raise ArtifactError("source column declarations conflict with single-premium preparation")
    meta.setdefault("schema_version", 1)
    meta.pop("table", None)
    meta["columns"] = _copy_json(_COLUMNS)
    rows = []
    for record in records:
        if not isinstance(record, dict):
            raise ArtifactError("each premium record must be an object")
        row = {col["name"]: record.get(col["name"]) for col in _COLUMNS}
        for name in ("strike", "maturity", "price"):
            row[name] = _finite(row[name], name)
        rows.append(row)
    artifact = LocalArtifact(meta, tuple(rows))
    prices = hkde_panel_prices(artifact, checkpoint_task=scientific_task)
    meta["preparation"] = {
        "n_contracts": len(rows), "negative_premiums_retained": sum(value < 0 for value in prices),
        "unknown_quote_times": sum(row["quote_time"] is None for row in rows),
        "task_sha256": hashlib.sha256(json.dumps(
            scientific_task, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
    }
    return artifact
