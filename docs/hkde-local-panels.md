# Local HKDE premium panels

This is **3/5-data-plan**, following [PR #6](https://github.com/fmuia/sbi4finance/pull/6) and [plan PR #4](https://github.com/fmuia/sbi4finance/pull/4). It implements the single-premium part of D1-O and a bounded D2 interoperability check against [simulator PR #15](https://github.com/fmuia/sbi4finance/pull/15). The scientific implementation remains in that workstream.

One observed example is nine **noisy European put premiums**, ordered by maturities 0.25, 0.5, 1 year and then strikes 90, 100, 110. The current checkpoint infers long-run variance and jump intensity, with spot 100, rate 0.02, dividend yield 0, noise SD 0.25 and the remaining model settings fixed. These premiums are neither IVs nor bid/ask pairs. The synthetic Gaussian observation law can produce negative quotes; we preserve them rather than change that law by clipping.

## Prepare and inspect

The data helpers use the standard library only. Here is a complete parser example with deliberately illustrative numbers:

```python
import json
from pathlib import Path
from sbi4finance.datasets import prepare_hkde_panel, hkde_panel_prices

# A copied task specification for parser tests, not generated model prices.
source_task = json.loads(Path("tests/fixtures/local/hkde_task.json").read_text())
rows = [
    {"contract_id": f"c{i}", **contract, "price": float(i) - 0.1}
    for i, contract in enumerate(source_task["context"]["contracts"])
]
panel = prepare_hkde_panel(
    rows, task=source_task,
    metadata={
        "artifact_id": "illustrative-panel", "data_origin": "synthetic",
        "purpose": "parser_test",
        "provenance": {"source": "hand-constructed parser example"},
        "conventions": {
            "quote_form": "single_premium", "price_unit": "currency",
            "exercise": "European", "settlement": "cash",
            "premium_basis": "per_underlying_unit",
        },
    },
)
prices = hkde_panel_prices(panel, checkpoint_task=source_task)
assert prices[0] == -0.1
```

Use the existing `write_artifact`/`load_artifact` to store this panel as CSV/JSON. Preparation copies only `contract_id`, `strike`, `maturity`, `option_type`, `price` and optional `quote_time`. Unknown quote times stay null; timestamps may also be absent from a stored inference-only artifact. Synthetic truth belongs in the optional manifest `truth` object. Extra source columns are not automatically copied or used for inference. Separately declared auxiliary columns may accompany a stored panel but are excluded from conditioning.

`task` must describe the **source experiment**. Do not attach a checkpoint's desired task to incompatible market quotes. Source contracts, representation, exercise, units and context must be independently correct; file validation cannot verify the honesty of a declaration. Existing conflicting metadata is rejected. This helper does not acquire or clean market quotes, infer curves, create bid/ask spreads, rescale currencies, convert IVs or interpolate strikes/expiries.

`hkde_panel_prices(panel, checkpoint_task=...)` compares the complete source task with the actual saved checkpoint task, checks the exact ordered coordinates and declared column roles/units, and returns only the nine prices. Missing/extra/permuted contracts and extra observation/context channels fail. It also rejects changes to spot, rates, quote noise, prior, fixed parameters or numerical settings. A price-column name and a nine-element shape are not sufficient evidence of compatibility.

After these checks, reuse the simulator's own adapter:

```python
import numpy as np
from sbi4finance import hkde_inference

# checkpoint_task is read from the selected checkpoint's task.json.
observations = np.asarray([
    hkde_panel_prices(panel, checkpoint_task=checkpoint_task)
], dtype=np.float64)[:, :, None]  # [1, 9, 1]
conditions = hkde_inference.conditioning_data(observations, checkpoint_task)
```

No new neural adapter or normalization is implemented here. The simulator adapter owns scientific task validation and mapping into BayesFlow. Truth, clean prices, numerical diagnostics and held-out target quotes never become input channels.

## Run the real checkpoint regression

The opt-in script uses **actual committed validation panels and weights**, not the parser example. It checks saved task/checkpoint/adapter identities, writes and reloads two panels, removes truth/auxiliary fields, compares direct versus loaded conditions and 32 same-seed posterior draws per case, and rejects incompatible panels. It creates temporary fixtures and can save a small new report file. It never trains, regenerates a bank or evaluates the reserved test split.

Once the data and simulator changes coexist in one checkout:

```bash
pixi run hkde-data-check
```

While the PRs remain separate, use the simulator checkout's pinned environment and explicitly identify its source root. Run from this data checkout; replace the example path with your checkout:

```bash
/path/to/simulator/.pixi/envs/default/bin/python scripts/check_hkde_data.py \
  --simulator-root /path/to/simulator \
  --out /tmp/hkde-data-check-new.json
```

The optional checkout bridge exists only in this review script; library code has no workspace paths. No files or code are copied from the simulator branch. With a fresh checkout, install its pinned environment first. `--out` never overwrites an existing report.

The preparation unit tests have no simulator dependency; `pixi run data-check` or `PYTHONPATH=src python -m unittest discover -s tests` runs the checkout's unittest suite. The fixture task pins the published PR #15 contract; it is not a provider fixture or a second model registry.

The [recorded integration run](results/hkde-data-handoff.json) passed against simulator commit `729ac26`: both validation cases retained exact conditions and seeded posterior draws, truth/auxiliary fields were removable, and all seven incompatible variants were rejected. The record includes source and checkpoint hashes.

## Evidence boundary

The current HKDE checkpoint remains **provisional**: its original downstream price-quantile error is approximately 0.0439 currency versus a 0.03 gate. An exact round trip proves that storage and preparation preserve its behavior; it does not repair or override that failed scientific check. The regression report records interoperability and scientific acceptance separately.

This completes the current single-premium preparation and checkpoint-handoff checks. General premium/IV market normalization, return-checkpoint compatibility, bank-wide integration and final D2/D4 acceptance remain separate work. No real data are needed for this synthetic path. Real-market adequacy and held-out empirical evaluation still require appropriately prepared real observations.
