# Small local observation fixtures

This is the first D0–D1 implementation from the [data plan](https://github.com/fmuia/sbi4finance/pull/4): a standard-library-only CSV reader/writer with explicit metadata and column roles. It is deliberately a small, provisional interface that we can change as the tutorial develops. There is no new task registry, provider abstraction, bank format or neural dependency.

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

## What this slice does not establish

Successful parsing does **not** certify calendar handling, adjustment policy, price/IV conventions, grid compatibility, numerical pricing accuracy or model adequacy. It does not perform price-to-return conversion, financial filtering, curve inference, forward simulation, checkpoint checks or network training. Those belong to D1-R/D1-O, D2 and the simulator workstream as they become useful.

Fixtures here are hand-constructed parser examples, not market data or scientifically validated simulation outputs. Missing-value examples do not imply that a trained network supports masks. No real observations or account access are needed for these checks.

Run the focused checks with `pixi run data-check`, or `PYTHONPATH=src python -m unittest discover -s tests -p 'test_local_data.py'`. Tests use temporary directories and no provider access. The existing MMAR notebook, data cache and checkpoint remain separate.
