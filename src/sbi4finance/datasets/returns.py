"""Prepare one complete daily return window without guessing a calendar."""

from datetime import date
import hashlib
import json
import math

from .local import ArtifactError, LocalArtifact, _check_metadata


def _day(value):
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("dates must use YYYY-MM-DD; intraday data are not supported") from exc
    if parsed.isoformat() != value:
        raise ArtifactError("dates must use YYYY-MM-DD")
    return parsed


def _number(value, name):
    if value is None or isinstance(value, bool):
        raise ArtifactError(f"{name} must be a finite number; missing values are unsupported")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ArtifactError(f"{name} must be a finite number")
    return value


def prepare_return_window(
    records, *, metadata, source_kind, source_unit, return_kind,
    boundaries, dt_years, cutoff,
) -> LocalArtifact:
    """Select N complete intervals ending at cutoff and return decimal returns.

    ``boundaries`` contains N+1 explicit daily ISO dates, including the initial
    boundary. ``dt_years`` supplies N positive year fractions; dates are labels,
    not an inferred day-count convention. The final boundary must equal cutoff.

    Price rows have ``timestamp`` and ``value``; supplied-return rows have
    ``interval_start``, ``interval_end`` and ``value``. Input endpoint dates must
    be strictly ordered. Selected endpoints must match the requested schedule.
    Extra columns and values outside the window are never conditioning inputs.

    Source kinds are prices/log_returns/simple_returns. Source units are currency
    for prices and decimal/percent for returns. Output kind is log/simple and
    units are always decimal. This positive-price scope rejects simple returns
    <= -1 as well as nonpositive prices. No filling or standardization occurs.

    Metadata describes the intended artifact and source policies. Any synthetic
    truth must be supplied for this particular window, not copied from a later
    state. This helper does not assert compatibility with a model or checkpoint.
    """
    if source_kind not in ("prices", "log_returns", "simple_returns"):
        raise ArtifactError("source_kind must be prices, log_returns or simple_returns")
    if return_kind not in ("log", "simple"):
        raise ArtifactError("return_kind must be log or simple")
    allowed_units = ("currency",) if source_kind == "prices" else ("decimal", "percent")
    if source_unit not in allowed_units:
        raise ArtifactError(f"source_unit for {source_kind} must be one of {allowed_units}")

    days = [_day(value) for value in boundaries]
    if len(days) < 2 or any(a >= b for a, b in zip(days, days[1:])):
        raise ArtifactError("boundaries need at least two strictly increasing dates")
    if days[-1] != _day(cutoff):
        raise ArtifactError("the window's final boundary must equal cutoff")
    steps = [_number(value, "dt_years") for value in dt_years]
    if len(steps) != len(days) - 1 or any(value <= 0 for value in steps):
        raise ArtifactError("dt_years needs one positive year fraction per interval")

    # Copy metadata; the input record collection and caller's declarations stay intact.
    meta = json.loads(json.dumps(metadata, allow_nan=False))
    if not isinstance(meta, dict):
        raise ArtifactError("metadata must be an object")
    conventions = meta.get("conventions")
    provenance = meta.get("provenance")
    for obj, keys in (
        (conventions, ("currency", "adjustment_policy", "calendar", "time_basis")),
        (provenance, ("source", "series_id")),
    ):
        if not isinstance(obj, dict):
            raise ArtifactError("conventions and provenance must be objects")
        for key in keys:
            if not isinstance(obj.get(key), str) or not obj[key].strip():
                raise ArtifactError(f"explicit {key} is required")
    representation = f"{return_kind}_returns"
    for key, expected in (("artifact_type", "return_window"), ("representation", representation)):
        if key in meta and meta[key] != expected:
            raise ArtifactError(f"metadata {key} conflicts with requested return window")
        meta[key] = expected
    if "return_kind" in conventions and conventions["return_kind"] != return_kind:
        raise ArtifactError("declared return_kind conflicts with requested output")
    if "return_unit" in conventions and conventions["return_unit"] != "decimal":
        raise ArtifactError("prepared return_unit must be decimal")
    conventions.update(
        return_kind=return_kind, return_unit="decimal", normalization_version=1,
        conditioning_cutoff=cutoff, window_start=days[0].isoformat(),
        n_returns=len(steps),
    )
    meta.setdefault("schema_version", 1)
    # A prepared in-memory artifact has no storage checksum until written.
    meta.pop("table", None)

    endpoint = "timestamp" if source_kind == "prices" else "interval_end"
    selected, input_count, before, after = [], 0, 0, 0
    previous = None
    for record in records:
        input_count += 1
        if not isinstance(record, dict) or endpoint not in record:
            raise ArtifactError(f"each source row needs {endpoint}")
        end = _day(record[endpoint])
        if previous is not None and end <= previous:
            raise ArtifactError("source dates must be strictly ordered with no duplicates")
        previous = end
        if end > days[-1]:
            after += 1
        elif end < days[0] or (source_kind != "prices" and end == days[0]):
            before += 1
        else:
            selected.append(record)
    expected_ends = days if source_kind == "prices" else days[1:]
    if [_day(row[endpoint]) for row in selected] != expected_ends:
        raise ArtifactError("selected dates do not match boundaries; missing or extra observations")

    values = [_number(row.get("value"), "selected value") for row in selected]
    if source_kind == "prices" and any(value <= 0 for value in values):
        raise ArtifactError("prices must be strictly positive")
    if source_kind != "prices":
        if any(_day(row.get("interval_start")) != day for row, day in zip(selected, days)):
            raise ArtifactError("supplied return intervals do not match boundaries")
        scaled = [value / 100 if source_unit == "percent" else value for value in values]
        if source_kind == "simple_returns" and any(value <= -1 for value in scaled):
            raise ArtifactError("simple returns must exceed -1 in this positive-price scope")

    columns = [
        {"name": "row_id", "dtype": "str", "role": "id"},
        {"name": "interval_start", "dtype": "str", "role": "auxiliary"},
        {"name": "interval_end", "dtype": "str", "role": "auxiliary"},
        {"name": "dt_years", "dtype": "float", "role": "context", "unit": "years"},
        {"name": "return", "dtype": "float", "role": "observation", "unit": "decimal"},
        {"name": "observed", "dtype": "bool", "role": "auxiliary"},
    ]
    source_columns = ("source_start", "source_end") if source_kind == "prices" else ("source_value",)
    columns.extend(
        {"name": name, "dtype": "float", "role": "auxiliary", "unit": source_unit}
        for name in source_columns
    )
    rows = []
    for i, (start, end, step) in enumerate(zip(days, days[1:], steps)):
        try:
            if source_kind == "prices":
                value = (math.log(values[i + 1]) - math.log(values[i])
                         if return_kind == "log" else values[i + 1] / values[i] - 1)
                original = {"source_start": values[i], "source_end": values[i + 1]}
            else:
                value = scaled[i]
                if source_kind == "simple_returns" and return_kind == "log":
                    value = math.log1p(value)
                elif source_kind == "log_returns" and return_kind == "simple":
                    value = math.expm1(value)
                original = {"source_value": values[i]}
        except OverflowError as exc:
            raise ArtifactError("return conversion overflowed") from exc
        value = _number(value, "converted return")
        if return_kind == "simple" and value <= -1:
            raise ArtifactError("return conversion lost positive-price precision")
        rows.append({
            "row_id": end.isoformat(), "interval_start": start.isoformat(),
            "interval_end": end.isoformat(), "dt_years": step,
            "return": value, "observed": True, **original,
        })

    meta["columns"] = columns
    # Only selected source values enter this lineage hash, never future outcomes.
    source_payload = {"kind": source_kind, "unit": source_unit,
                      "boundaries": [day.isoformat() for day in days], "values": values}
    meta["preparation"] = {
        "source_kind": source_kind, "source_unit": source_unit,
        "input_rows": input_count, "selected_rows": len(selected),
        "excluded_before_window": before, "excluded_after_cutoff": after,
        "selected_source_sha256": hashlib.sha256(json.dumps(
            source_payload, sort_keys=True, allow_nan=False, separators=(",", ":"),
        ).encode()).hexdigest(),
    }
    _check_metadata(meta)
    return LocalArtifact(meta, tuple(rows))
