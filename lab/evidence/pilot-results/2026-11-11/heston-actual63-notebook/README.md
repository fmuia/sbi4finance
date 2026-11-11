# Actual63 Heston development notebook: run record

This record supports [`lab/notebooks/heston_spx_calibration.py`](../../../../notebooks/heston_spx_calibration.py). That notebook is the lab counterpart of the tutorial lesson `tutorial/notebooks/slot4_heston_bayesflow.py`, which keeps its own four-parameter configuration. It presents the configuration of PR #85: 63 actual SPXW quotes, spread-scaled noise with an inferred multiplier c, and the native FlowMatching estimator trained for 300 epochs on 300,000 simulations (`300k-300-final300`). How to obtain the bundle is in the [training-data guide](../../../../plans/training-data-guide.md#current-actual63-heston-development-package-11-november-2026); how to run or render the notebook is in its last section. No checkpoint, model or tutorial asset is promoted, and nothing here is a new acceptance claim.

## Why the PyMC references were regenerated

The notebook compares the neural posterior with an MCMC reference on the three predeclared observations: `synthetic-broad`, `synthetic-market-like` and `market`. The recorded references are summarized in `revised-grid/recovery-bank-independent/references/<case>/posterior.json` as quantiles and means. Their chains (`chain_payload_path` in each `run.json`) stayed on the producer's machine and are not in the Drive bundle.

[`scripts/heston_actual63_notebook_references.py`](../../../../scripts/heston_actual63_notebook_references.py) reruns the **unchanged canonical worker** `run_heston_pr87_grid_reference_recovery.run_reference`. The seed (261112), sampler (DEMetropolisZ, four chains), data-only Powell initialization, 2k/2k then 6k/12k allocation, likelihood, Black-control-variate pricer, convergence gates, pointwise precision gates and resource caps are all identical. The frozen panel binds its design, source and preparation protocol by absolute paths on the producer's machine. `prepare` writes [`reference-panel.json`](reference-panel.json), a copy that changes only those three path fields after checking every bound SHA-256. The original panel, `824b7650…`, and every other recorded file are read, never modified. [`reference-relocation.json`](reference-relocation.json) records the mapping. [`reference-protocol.json`](reference-protocol.json) is the new protocol identity; the worker validates it again before and after sampling.

## Contents

| File | Role |
| --- | --- |
| `reference-panel.json`, `reference-relocation.json`, `reference-protocol.json` | Relocated observation-only panel, its receipt, and the frozen protocol for this run |
| `references/<case>/run.json`, `diagnostics.json`, `posterior.json`, `progress.json` | Per-case run record written by the canonical worker |
| `reference-comparison.json` | Regenerated versus recorded quantiles and means, chain identities, truths of the synthetic cases |
| `reference-display-draws.npz` | 4 chains × 500 draws per case (every 24th retained draw) and the paired variance products, for the notebook's plots |

The full chains (4 × 12,000 × 6 per case, plus products and sampler statistics) stay outside Git, as AGENTS.md requires. Their SHA-256 identities are in each `run.json`. To recreate them, run from `lab/` with the locked `core-reference` environment:

```sh
pixi run --locked -e core-reference python scripts/heston_actual63_notebook_references.py prepare \
  --out <new evidence dir> --payload-root <new payload dir outside Git>
pixi run --locked -e core-reference python scripts/run_heston_pr87_grid_reference_recovery.py \
  --panel <new evidence dir>/reference-panel.json --protocol <new evidence dir>/reference-protocol.json \
  --out <new evidence dir>/references/<case> --payload-out <new payload dir>/<case> --case-id <case>
pixi run --locked -e core-reference python scripts/heston_actual63_notebook_references.py summarize --out <new evidence dir>
```

The second command runs once per case. The protocol declares at most two concurrent cases.

## Results

All three regenerated references are ready. Each followed the recorded escalation: the 2k/2k attempt failed the MCMC diagnostics, the 6k/12k attempt passed, and every pointwise precision gate passed. Likelihood-evaluation counts equal the recorded ones. The chains are not byte-identical to the producer's, because the platforms differ, so their SHA-256 identities differ from the recorded ones. Agreement with the recorded quantiles is within Monte Carlo error:

| Case | Wall time | Attempts (likelihood evaluations) | Max \|mean difference\| (regenerated SDs) | 90% width ratio, regenerated / recorded |
| --- | ---: | --- | ---: | ---: |
| `synthetic-broad` | 846 s | 2k/2k failed (34,094); 6k/12k passed (146,094) | 0.070 | 0.977–1.018 |
| `synthetic-market-like` | 1579 s | 2k/2k failed (34,087); 6k/12k passed (146,087) | 0.020 | 0.982–1.017 |
| `market` | 1608 s | 2k/2k failed (34,113); 6k/12k passed (146,113) | 0.018 | 1.001–1.010 |

The full per-parameter comparison is in `reference-comparison.json`. The market reference again places c against its prior bound: median 1.9995. In the notebook, the market quotes sit 13.4 half-spreads (RMS) from the PyMC posterior-mean prices, with the largest misfits in the deep put wing.

## Validation of the notebook

- `marimo check` is clean.
- A complete `marimo export html` without the bundle executes every cell. Sections 4 (Infer) and 5 (Check) then show the PyMC reference alone, and the cells that resample or retrain the estimator stay off.
- An in-process run replaced the loader's two entry points with stand-ins: an untrained network of the production architecture behind the production raw-quote wrapper, and a small simulated bank. It executed every bundle-dependent cell, including the button-gated live training, retraining and 512-case resampling. Further runs forced the data-set selector to the stressed and market cases, with and without the stand-in, and simulated a box selection on the recovery plot, so that every branch of the Section 4 to 6 text runs. The stand-in bank uses the 512 held-out truths of the saved evaluation, which the notebook checks before resampling them. This exercises code paths only; its numbers are meaningless and were not kept.
- The live training cell, run on its own, simulates 3,300 data sets in 21 s, trains for 40 epochs in 12 s and samples the held-out check in 25 s on four CPU cores.
- `pixi run --locked -e core-reference release-check` passes.

The pretrained-estimator cells have so far run here only with stand-ins: the Drive archive is restricted to named accounts and could not be fetched into the cloud container. The first render against the real bundle should be recorded here, with its revision and environment.
