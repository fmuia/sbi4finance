# Offline HKDE data walkthrough

This is **5/5-data-plan-offline-teaching-example**, following
[bank integration PR #17](https://github.com/fmuia/sbi4finance/pull/17) and
[plan PR #4](https://github.com/fmuia/sbi4finance/pull/4).
It completes the engineering path for the current restricted HKDE example.
**Final D2/D4 scientific acceptance remains pending:** the upstream checkpoint
fails its downstream price-quantile gate and is not the accepted participant
pricing checkpoint. This is an executable provisional example for development
and review, with that failure shown in the notebook.

Continue from this introduction to the [corrected HKDE lesson](hkde-teaching-integration.md).
It loads the same local panel and compares the original raw result with cached
weighted correction and the numerical reference. The original bundle below is
unchanged. Corrected pricing improves, but the separate corrected test still
fails theta calibration. An [opt-in provisional live demo](hkde-provisional-demo.md)
now exercises the fixed validation case; formal participant acceptance stays pending.

## Run without market data or weights

After installing the repository's pinned environment:

```bash
pixi run hkde-data-walkthrough
pixi run hkde-teaching-check
```

The [notebook](../notebooks/hkde_data_walkthrough.py) uses only local files,
NumPy, Matplotlib and marimo. It does not need the simulator checkout, a bank,
checkpoint weights, a GPU, provider account or network connection. It loads a
prepared panel and **replays cached posterior/price draws**; it does not perform
new inference or train a model. Missing/corrupt files fail, with no download or
retraining fallback. Editing observed prices invalidates the attached cache;
new observations require a fresh compatible inference run.

The check executes all notebook cells from a fresh temporary directory containing
only the local data helpers, notebook and synthetic bundle. Socket connection,
DNS and datagram operations are blocked. Market files, weights and simulator
code are absent. This is an execution check, not another statistical acceptance
test. Environment installation itself requires the usual package access.

The [recorded isolated run](results/hkde-teaching-offline.json) completed all
eight notebook cells in about eight seconds on the review machine. It records
source/bundle hashes and retains `scientific_validation_passed: false` separately
from successful execution. All 31 data tests and the marimo structure check pass.

## What is being fitted?

One observation is nine noisy **European put premiums**, with maturities
0.25/0.5/1 year and strikes 90/100/110, in that order. Known context is spot 100,
rate 0.02, dividend yield 0 and independent Gaussian quote-error SD 0.25 currency.
Only long-run variance `theta` and jump intensity are inferred; all remaining
HKDE parameters and initial variance are fixed. The complete prior/context is
available in the notebook, not guessed from array dimensions.

For each candidate parameter pair, the model produces nine prices. A Gaussian
observation likelihood compares them with the panel, and the prior completes
the posterior. The current simulator uses direct numerical Fourier quadrature,
not the original paper's PROJ implementation. No return series, IV inversion,
bid/ask construction or American exercise calculation is involved.

The live bank integration in increment 4 calls the original BayesFlow adapter
and checkpoint. This cached walkthrough displays the previously audited joint
draws and their prices. Keeping both parameter columns paired preserves their
posterior dependence when propagating them into a **held-out European put**, with
strike 95 and maturity 0.75 years. That contract is absent from conditioning.
The resulting distribution is uncertainty in a risk-neutral **model expectation**,
not realized payoff risk or tomorrow's option-chain forecast.

## Bundle and evidence

The approximately 100 KB bundle at `data/synthetic/hkde-walkthrough` contains:

| File | Role |
|---|---|
| `panel/observations.csv` and `panel/manifest.json` | Prepared validation panel; truth is separate metadata used only for evaluation |
| `task.json` | Exact saved checkpoint task, checked against the panel |
| `posterior.npz` | 4,096 paired parameter draws, their held-out model prices, and the exact conditioning prices |
| `bundle.json` | File/source hashes, numerical-reference quantiles and published provisional status |

The source is **synthetic `validation:257`** from simulator
[PR #15](https://github.com/fmuia/sbi4finance/pull/15), commit `729ac26`.
This case deliberately shows the largest published downstream discrepancy:

| Price quantile | Numerical posterior | Neural candidate |
|---|---:|---:|
| 5% | 4.69582 | 4.65195 |
| 50% | 4.85595 | 4.82286 |
| 95% | 5.00902 | 4.99152 |

The 0.04387 currency discrepancy exceeds the fixed 0.03 gate. The numerical
reference is itself a refined approximation. Neither loading nor caching fixes
the neural error. The reserved test split is not used for inference or metrics.

The builder checks artifact linkage and scientific source hashes, selects the
observation through the simulator's bank reader, verifies its equality to the
cached audit case, and reprices three paired draws with the original pricer.
This checks extraction and pairing, not the full statistical experiment again.
Recorded training/evaluation driver hashes and current checkout hashes are
retained separately: later driver edits added provenance checks. Core inference
and pricing sources match their records.

## Rebuild or exercise the live boundary

Rebuilding needs the simulator workstream's generated bank and published audit
artifacts. The full bank and model weights are not copied into this bundle.
Once the two workstreams coexist in one checkout:

```bash
pixi run python scripts/prepare_hkde_teaching.py \
  --bank /path/to/original-hkde-bank --out /tmp/new-hkde-teaching-bundle
pixi run hkde-bank-data-check \
  --bank /path/to/original-hkde-bank --example-id validation:257
```

Before merging, use the simulator checkout's pinned Python and add
`--simulator-root /path/to/simulator` to either script. The bank's identity must
match the published training record to rebuild this particular cache. The builder
refuses existing output directories. It does not silently regenerate a bank.
For independent bank generation instructions, see simulator PR #15; the default
`hkde-bank-data-check` can also generate its own tiny smoke bank.

For an optional pricing exercise with both source trees available, reuse the
scientific function on matched rows:

```python
from sbi4finance.datasets.hkde_teaching import load_teaching_example
from sbi4finance.hkde_reference import price_parameters

panel, report, arrays = load_teaching_example("data/synthetic/hkde-walkthrough")
prices = price_parameters(
    panel.metadata["scientific_task"], arrays["parameter_draws"][:3],
    contracts=[report["target_contract"]],
)
```

A fresh single-case neural call is a new Monte Carlo realization. The audit
sampled 20 cases together with batch size 8; using the same seed on one case need
not reproduce its cached draws. Increment 4 instead checks exact equality of
direct and transported inputs using identical calls within one run.

## What requires real data?

| Question | Synthetic data sufficient? | Real data needed |
|---|---|---|
| Generate training examples and train the inverse map | Yes | None |
| Check numerical prices, recovery, posterior calibration and storage compatibility | Yes, under the chosen generative assumptions | None |
| Demonstrate parameter uncertainty in another model price | Yes | None |
| Fit an actual market snapshot and examine residuals | No | Synchronized premiums, contract/exercise conventions and compatible spot/forward/rate/dividend context |
| Assess pricing performance on withheld market contracts or across dates | No | Appropriate withheld quotes/dates and an explicit empirical comparison protocol |

Market parameters are not known synthetic truth. Vendor IVs, fitted parameters
and later outcomes cannot be promoted to such labels. Arbitrary chains cannot
be attached to this fixed checkpoint by relabelling them: new quote designs,
context, noise laws or American contracts require compatible training/inference.
Physical-measure return forecasts are a separate task with different data and
validation. Optional market preparation and durable return-checkpoint integration
remain outside these two HKDE increments; their absence does not block the
synthetic walkthrough.
