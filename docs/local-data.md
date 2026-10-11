# Small local observation fixtures

This implements the local D0–D1 foundation, bounded D1-R return preparation, and [single-premium HKDE panels](hkde-local-panels.md) from the [data plan](https://github.com/fmuia/sbi4finance/pull/4). The library helpers use only the standard library and remain small, provisional interfaces that we can change as the tutorial develops. There is no new task registry, provider abstraction, bank format or neural dependency.

One directory holds one observed example: a whole return history or an option panel. It contains `manifest.json` and `observations.csv`. A row is a day/interval or a contract, not an independent labelled training example. Larger simulation banks remain the simulator workstream's responsibility.

## Use

```python
from sbi4finance.datasets import load_artifact, write_artifact

data = load_artifact("tests/fixtures/local/returns")
print(data.observations)  # {'return': (0.0, -0.02, None)}
print(data.context)       # only explicitly declared context columns
print(data.truth)         # optional synthetic labels, separate from inputs

# Write to a NEW directory. Existing directories are never overwritten.
copy = write_artifact("/tmp/my-return-fixture", data.metadata, data.rows)
```

`data.rows` preserves all typed columns and row order for inspection/round trips. **Do not feed all numeric columns to an encoder.** `data.observations` and `data.context` select only their declared roles; auxiliary fields, IDs and manifest metadata are excluded. These dictionaries are not yet BayesFlow tensors or a model-approved conditioning object. Fixed conventions/context remain in the manifest for an eventual task-specific adapter.

## Manifest conventions

The three [parser fixtures](../tests/fixtures/local/README.md) are complete examples. The small v1 convention requires:

- `schema_version: 1`, `artifact_id`, and a descriptive `task_id` (not a registered task).
- `artifact_type`: `return_window` or `option_panel`; `representation`: `log_returns`/`simple_returns` or `raw_quote`/`processed_iv`, respectively.
- `measure`: P or Q; `data_origin`: synthetic, market or mixed; `purpose`: parser_test, training, validation, conditioning or evaluation. These describe the artifact, not evidence that a model is validated.
- Nonempty `conventions` and `provenance` objects. Record return/IV units, time conventions, conditioning boundary and context/source assumptions here; this reader preserves them without interpreting every financial convention.
- An ordered `columns` list. Each entry has `name`, `dtype` (`str`, `float`, `bool`) and `role` (`id`, `observation`, `context`, `auxiliary`). Float columns need a `unit`. There is exactly one unique, non-null string ID column. Observation columns are numeric.
- Optional `nullable: true` for unknown values. Empty CSV cells mean `None`, including unknown strings; numeric zero stays zero. Booleans are `true`/`false`. All numeric values must be finite. Blank and empty-string values are not separate concepts in this initial format.
- Optional `truth`, an object with explicitly synthetic labels, or null/absent. Market-origin records cannot carry this object. Vendor estimates belong in auxiliary fields or metadata, not synthetic truth. Mixed origins require component provenance supplied by the caller.
- `table`, filled by the writer: row count and SHA-256 of the exact CSV bytes. This catches accidental changes/incomplete publication, not tampering or semantic mistakes in a manifest.

The reader checks version/representation, declared columns, types, IDs, row count and content hash. It returns `ArtifactError` for invalid/incomplete stored artifacts. Extra CSV columns are rejected unless declared; adding a diagnostic column cannot silently make it an observation. The manifest itself is a scientific declaration: a caller who mislabels hidden truth as context can still create leakage. Task/adapter review must check that declaration.

The writer validates before creating a new directory, writes the table, and publishes the manifest last. A failed write can leave an incomplete directory; the loader rejects it. Inspect/remove that directory before retrying. Automatic overwrite, recovery, migration and remote storage are intentionally absent.

## Prepare a daily return window

`prepare_return_window` takes local records and an explicit schedule. It returns an in-memory `LocalArtifact`; use the same writer to save it. Here is a complete synthetic preparation example (illustrative levels, not a validated simulation):

```python
from sbi4finance.datasets import prepare_return_window, write_artifact

prices = [
    {"timestamp": "2000-01-06", "value": 100.0},
    {"timestamp": "2000-01-07", "value": 100.0},
    {"timestamp": "2000-01-10", "value": 102.0},
    {"timestamp": "2000-01-11", "value": 99.0},  # future, excluded
]
window = prepare_return_window(
    prices,
    metadata={
        "artifact_id": "example-window", "task_id": "parser-only/returns",
        "measure": "P", "data_origin": "synthetic", "purpose": "parser_test",
        "provenance": {"source": "hand-constructed example", "series_id": "example"},
        "conventions": {
            "currency": "USD", "adjustment_policy": "none; hypothetical levels",
            "calendar": "explicit hypothetical session dates",
            "time_basis": "one trading step per supplied interval, 252 steps/year",
        },
    },
    source_kind="prices", source_unit="currency", return_kind="log",
    boundaries=["2000-01-06", "2000-01-07", "2000-01-10"],
    dt_years=[1 / 252, 1 / 252], cutoff="2000-01-10",
)
print(window.observations)  # zero followed by approximately 0.0198026
print(window.context)       # only dt_years
print(window.metadata["preparation"])  # selection counts and source hash
# Optional: write_artifact("/tmp/prepared-return-window", window.metadata, window.rows)
```

For supplied returns, use rows with `interval_start`, `interval_end`, and `value`; set `source_kind="log_returns"` or `"simple_returns"`, and `source_unit="decimal"` or `"percent"`. The return type and unit are explicit, never guessed from magnitude. The requested `return_kind` is `log` or `simple`; output is always decimal. Price conversion uses log price differences or price ratios minus one. Supplied simple/log returns are converted with `log1p`/`expm1` only when the requested output differs.

The preparation rules are deliberately narrow:

- `boundaries` contains N+1 strictly increasing `YYYY-MM-DD` dates for N returns. `dt_years` contains N positive year fractions. The last boundary must equal the explicit cutoff. Intraday timestamps and inferred calendars are deferred.
- The caller supplies the expected schedule, currency, adjustment policy and time convention. Weekend/holiday gaps are valid if declared by the schedule. A missing expected observation fails; there is no forward filling, zero filling, inferred trading calendar or automatic corporate-action correction. A wrong supplied schedule is still a wrong scientific assumption.
- Source endpoint dates must be unique and ordered. Selected rows must exactly match the requested schedule, including the start of each supplied return. Out-of-order input is rejected rather than silently sorted. Values before/after the requested window are excluded before numerical conversion; changing future values or extra diagnostic columns cannot change conditioning.
- Selected values must be finite and complete. Prices must be positive; simple returns must exceed -1 in this first positive-price scope. Zero returns remain valid. There is no per-window standardization or volatility estimation.
- Only returns and year fractions become observations/context. Dates, the all-true validity flag and selected source values remain auxiliary. Raw source files stay with the caller. The report records row counts and a hash of the selected source values/dates/units; it is not a checksum of the entire source file or a model-design fingerprint.
- `provenance.series_id` groups related windows from the same series; cutoff and window start identify the selection. Overlapping windows are not independent training/test examples. The simulator workstream still owns grouped splitting. Synthetic state labels must refer to this window's cutoff; the helper does not compute or retime them.

The same preparation works on declared synthetic or market records. Market windows remain unlabelled. This enables data preparation without accounts or downloads; it does not establish an empirical forecasting result.

## Status and limits

| Increment | Status | What remains |
|---|---|---|
| `1/5-data-plan-local-observation-fixtures` ([PR #5](https://github.com/fmuia/sbi4finance/pull/5)) | Local D0–D1 IO implemented | Model-specific contracts remain with their scientific tasks |
| `2/5-data-plan-return-window-preparation` ([PR #6](https://github.com/fmuia/sbi4finance/pull/6)) | D1-R daily preparation implemented | Durable return-checkpoint equivalence remains D2 |
| `3/5-data-plan-hkde-panel-handoff` ([PR #16](https://github.com/fmuia/sbi4finance/pull/16)) | [Single premiums and exact HKDE task checks](hkde-local-panels.md); real-checkpoint regression | Broader option normalization and other checkpoint interfaces |
| `4/5-data-plan-synthetic-bank-integration` ([PR #17](https://github.com/fmuia/sbi4finance/pull/17)) | [Simulator bank → local panel → checkpoint](hkde-local-panels.md#load-a-generated-bank-panel) | Scientific acceptance remains pending |
| `5/5-data-plan-offline-teaching-example` ([PR #18](https://github.com/fmuia/sbi4finance/pull/18)) | [Cached synthetic walkthrough](hkde-offline-teaching.md), independently executable without market files or weights | Provisional demonstration; accepted participant checkpoint still pending |
| D2 / D4 | Current HKDE engineering path implemented | Final scientific acceptance awaits a checkpoint passing the simulator plan's gates |
| D3 / D5 | Optional, not implemented | Market application / held-out empirical evaluation |

Implementation PRs use sequential `N/5-data-plan-…` titles linked to PR #4. Optional market branches do not block a selected synthetic workflow.

Successful parsing does **not** certify financial conventions, grid compatibility, numerical pricing accuracy or model adequacy. Return preparation checks only the explicit rules above. It does not validate the source's adjustment policy, infer curves, simulate paths, check checkpoints or train a network. Those belong to the relevant scientific workflow.

The current GBM reference uses decimal **log** returns and a positive year-fraction vector, with a `[batch, time, 1]` array convention. This local preparation preserves those numerical quantities when `return_kind="log"`; it is not yet a neural adapter or an integration test. Simple returns cannot be sent unchanged to that GBM interface. Missing-value parser fixtures also cannot condition a model that requires complete observations.

Fixtures here are hand-constructed parser examples, not market data or scientifically validated simulation outputs. Missing-value examples do not imply that a trained network supports masks. No real observations or account access are needed for these checks.

Run the checks with `pixi run data-check`, or `PYTHONPATH=src python -m unittest discover -s tests`. Tests use temporary directories and no provider access. The existing MMAR notebook, data cache and checkpoint remain separate.
