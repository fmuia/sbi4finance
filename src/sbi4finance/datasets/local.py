"""CSV + JSON IO for small, explicitly described observation fixtures.

This validates storage and column roles, not financial conventions or checkpoint
compatibility. It intentionally has no model, neural-network or provider imports.
"""

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path


class ArtifactError(ValueError):
    """An artifact is incomplete, inconsistent, or outside the local format."""


@dataclass
class LocalArtifact:
    metadata: dict
    rows: tuple[dict, ...]

    def columns(self, role: str) -> dict[str, tuple]:
        """Select declared columns only; do not infer roles from numeric dtype."""
        if role not in {"id", "observation", "context", "auxiliary"}:
            raise ValueError(f"Unknown column role: {role!r}")
        return {
            column["name"]: tuple(row[column["name"]] for row in self.rows)
            for column in self.metadata["columns"]
            if column["role"] == role
        }

    @property
    def observations(self) -> dict[str, tuple]:
        return self.columns("observation")

    @property
    def context(self) -> dict[str, tuple]:
        return self.columns("context")

    @property
    def truth(self) -> dict | None:
        """Optional synthetic labels, never included in observed conditioning."""
        return self.metadata.get("truth")


def _check_metadata(meta: dict) -> None:
    if not isinstance(meta, dict):
        raise ArtifactError("manifest must be a JSON object")
    if type(meta.get("schema_version")) is not int or meta["schema_version"] != 1:
        raise ArtifactError("unsupported schema_version; expected 1")
    for key in ("artifact_id", "task_id"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ArtifactError(f"{key} must be a nonempty string")
    formats = {
        "return_window": {"log_returns", "simple_returns"},
        "option_panel": {"raw_quote", "processed_iv"},
    }
    if meta.get("artifact_type") not in tuple(formats):
        raise ArtifactError("artifact_type must be return_window or option_panel")
    if meta.get("representation") not in tuple(formats[meta["artifact_type"]]):
        raise ArtifactError("representation does not match artifact_type")
    if meta.get("measure") not in ("P", "Q"):
        raise ArtifactError("measure must be P or Q")
    if meta.get("data_origin") not in ("synthetic", "market", "mixed"):
        raise ArtifactError("data_origin must be synthetic, market or mixed")
    if meta.get("purpose") not in (
        "parser_test", "training", "validation", "conditioning", "evaluation"
    ):
        raise ArtifactError("unsupported purpose")
    for key in ("conventions", "provenance"):
        if not isinstance(meta.get(key), dict) or not meta[key]:
            raise ArtifactError(f"{key} must be a nonempty object")
    truth = meta.get("truth")
    if truth is not None and not isinstance(truth, dict):
        raise ArtifactError("truth must be an object or null")
    if meta["data_origin"] == "market" and truth is not None:
        raise ArtifactError("market observations cannot have synthetic truth labels")

    columns = meta.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ArtifactError("columns must be a nonempty list")
    names, ids, observations = set(), 0, 0
    for col in columns:
        if not isinstance(col, dict):
            raise ArtifactError("each column must be an object")
        name = col.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactError("column names must be nonempty and unique")
        names.add(name)
        if col.get("dtype") not in ("str", "float", "bool"):
            raise ArtifactError(f"unsupported dtype for {name}")
        if col.get("role") not in ("id", "observation", "context", "auxiliary"):
            raise ArtifactError(f"unsupported role for {name}")
        if type(col.get("nullable", False)) is not bool:
            raise ArtifactError(f"nullable must be boolean for {name}")
        if col["dtype"] == "float" and (
            not isinstance(col.get("unit"), str) or not col["unit"].strip()
        ):
            raise ArtifactError(f"float column {name} needs an explicit unit")
        if col["role"] == "id":
            ids += 1
            if col["dtype"] != "str" or col.get("nullable", False):
                raise ArtifactError("row ID must be a non-nullable string")
        if col["role"] == "observation":
            observations += 1
            if col["dtype"] != "float":
                raise ArtifactError("observation columns must be float")
    if ids != 1 or observations == 0:
        raise ArtifactError("need exactly one ID column and at least one observation")


def _cell(text: str, col: dict):
    if text == "":
        if col.get("nullable", False):
            return None
        raise ArtifactError(f"missing required value in {col['name']}")
    if col["dtype"] == "str":
        return text
    if col["dtype"] == "bool":
        if text not in ("true", "false"):
            raise ArtifactError(f"{col['name']} must be true or false")
        return text == "true"
    try:
        value = float(text)
    except ValueError as exc:
        raise ArtifactError(f"invalid number in {col['name']}: {text!r}") from exc
    if not math.isfinite(value):
        raise ArtifactError(f"non-finite number in {col['name']}")
    return value


def _read_rows(text: str, meta: dict) -> tuple[dict, ...]:
    columns = meta["columns"]
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    if next(reader, None) != [col["name"] for col in columns]:
        raise ArtifactError("CSV header/order does not match declared columns")
    id_name = next(col["name"] for col in columns if col["role"] == "id")
    rows, seen = [], set()
    for line, cells in enumerate(reader, start=2):
        if len(cells) != len(columns):
            raise ArtifactError(f"CSV row {line} has the wrong number of fields")
        row = {col["name"]: _cell(cell, col) for cell, col in zip(cells, columns)}
        if row[id_name] in seen:
            raise ArtifactError(f"duplicate row ID: {row[id_name]!r}")
        seen.add(row[id_name])
        rows.append(row)
    if not rows:
        raise ArtifactError("observation table is empty")
    return tuple(rows)


def _reject_constant(value):
    raise ArtifactError(f"non-finite JSON value: {value}")


def load_artifact(directory: str | Path) -> LocalArtifact:
    """Read one local manifest.json + observations.csv; never download data."""
    directory = Path(directory)
    try:
        meta = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
        _check_metadata(meta)
        table = meta.get("table")
        if not isinstance(table, dict) or type(table.get("n_rows")) is not int:
            raise ArtifactError("table needs n_rows and sha256")
        payload = (directory / "observations.csv").read_bytes()
        if hashlib.sha256(payload).hexdigest() != table.get("sha256"):
            raise ArtifactError("observations.csv checksum mismatch")
        rows = _read_rows(payload.decode("utf-8"), meta)
        if len(rows) != table["n_rows"]:
            raise ArtifactError("row count does not match manifest")
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ArtifactError(f"cannot load {directory}: {exc}") from exc
    return LocalArtifact(meta, rows)


def write_artifact(directory: str | Path, metadata: dict, rows) -> LocalArtifact:
    """Write a small fixture to a new directory, publishing the manifest last.

    Existing directories are never overwritten. An interrupted write may leave
    an incomplete directory, which the reader rejects. Callers can inspect and
    remove that directory before retrying; no restart system is implied here.
    """
    meta = json.loads(json.dumps(metadata, allow_nan=False))
    _check_metadata(meta)
    names = [col["name"] for col in meta["columns"]]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(names)
    for row in rows:
        if set(row) != set(names):
            raise ArtifactError("row keys must match declared columns")
        writer.writerow([
            "" if row[name] is None else
            str(row[name]).lower() if isinstance(row[name], bool) else row[name]
            for name in names
        ])
    payload = stream.getvalue().encode("utf-8")
    parsed = _read_rows(payload.decode("utf-8"), meta)
    meta["table"] = {"n_rows": len(parsed), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "observations.csv").write_bytes(payload)
    (directory / "manifest.json").write_text(manifest, encoding="utf-8")
    return LocalArtifact(meta, parsed)
