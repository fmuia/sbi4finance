# Heston European-observation pilot: reflection gate

The Heston pilot is complete. **Do not promote this task or its neural checkpoints into the lessons yet.** The controlled information comparison is useful, but the market fit remains sensitive to parameter support and discrepancy assumptions, and the neural posterior fails conditional reference agreement. No rough-Bergomi jobs or lesson changes were performed.

## Results by category

| Category | Result | Evidence |
| --- | --- | --- |
| European contracts and data | Frozen delayed intraday SPX feed; SPXW only, actual quotes, roots and timestamps preserved. No individual bid/ask update timestamps; synchronization is unverified. | `source.json`, `designs/*.json` |
| Quote selection | Six expiries, 54 selected OTM quotes; short arm keeps the first 27. All put-heavy targets supported. Symmetric +2 targets at June/September expiries are unsupported and visibly use nearest available actual quotes. | `designs/*.json` |
| Observation law | Gaussian SD = half-spread is a declared convention. Compared spread-only, inferred IV discrepancy, half/double spread multipliers and double discrepancy-prior ceiling. This does not estimate an empirical market error law. | `summary.json` |
| Pricing | Prior/corner, independent QuantLib and retained-posterior checks support the price accuracy used here. Failed numerical controls were retained and investigated, as described below. | `precision*.json`, `quantlib*.json`, `posterior_precision*.json` |
| Reference inference | All 11 declared cases pass R-hat <=1.01 and bulk/tail ESS >=400, for both physical parameters and the two variance products. Four chains, 6,000 warmup +12,000 retained draws per chain per case. | `summary.json` |
| Information effect | Forward-variance 90% interval expands **6.49x** after removing later smile quotes with the baseline prior; near-term control expands **1.52x**. Widening vol-of-vol support changes these to **3.11x / 1.32x**. | `summary.json`, `heston-reflection.png` |
| Carry and clock | Two-SE coherent carry stresses change the affected MAP strike by at most **0.30% full /0.99% short**. Individual price changes reach 1.25–1.40 half-spreads before refitting. Clock stresses are much smaller. This is optimization sensitivity, not posterior integration. | `carry-clock-sensitivity.json` |
| Neural inference | Paired 20,000 training surfaces; separate full/short spline flows at 2,000 and 10,000 updates. Save/reload reproduces seeded samples exactly. **Both budgets fail market-conditional reference agreement.** Prior-average synthetic coverage conceals the conditional error. | `neural.json` |
| Release checks | Lab suite: 590 tests passed. All 16 final targeted pilot tests, including subsequently added product/design-binding and heldout checks, also passed. Manifest, claims and runtime release checks passed. | `validation.json` |

## What the experiment conditions on

This is a comparison of a **full selected panel**, not the entire SPXW chain, with a shorter-maturity subset. Each expiry has a separate parity anchor panel: every third paired near-spot strike is reserved, and those call/put symbols are excluded from calibration and scoring. Spread-weighted parity estimates forwards and discounts. Both arms retain these anchors at **all six expiries**.

The short arm also conditions on physical strikes selected using the observed ATM IV and on frozen market vegas used for discrepancy weighting. Later smile *levels in the likelihood* are withheld, but this is not complete removal of all later option information or a test of market-dependent missingness. The forward curve here is a carry curve, not a supplied Heston variance curve; all five variance-dynamics parameters are inferred.

The source record uses snapshot generation time, 20:28:37 UTC. The selected designs explicitly override valuation time to **20:13:37 UTC**, assuming a 15-minute feed delay. Exact individual quote times are unavailable. ACT/365F elapsed time ends at 16:00 New York, with early-close expiries excluded. Clock sensitivity retains the uncertainty about this convention.

The prior is independent uniform over v0/theta [.01,.16], kappa [.1,10], vol-of-vol [.1,1.5], rho [-.95,-.1], plus tau [0,.02] for discrepancy cases. Tau is in decimal IV units, multiplied by each frozen observed Black vega; this is a local weighting convention. The wider prior changes vol-of-vol alone to [.1,3]. All task identities are separate from the historical four-target lesson with kappa fixed at 2.

## Scientific interpretation

The baseline put-heavy full-panel posterior puts vol-of-vol near its upper bound (median 1.479; 90% interval 1.420–1.499). Allowing it to reach 3 moves the full-panel median to about 1.80 and the short-panel median to about 2.12. The affected short-panel variance-strike median changes from .06777 to .05127. Posterior width and location therefore depend materially on the prior support. Kappa remains weakly constrained toward its upper support in the wider-prior short arm.

The spread-only full-panel posterior has RMS price residual about **12.3 half-spreads**. Adding independent discrepancy does not remove the pattern: the baseline full-panel median-price residual RMS is about **13.4 half-spreads**, with clear strike/maturity structure. Its inferred tau median is .00663, or **0.663 IV percentage points**, much larger than the measurement-error convention. Halving/doubling spreads or doubling tau's prior ceiling has little effect on this median. This is consistent with substantial mismatch under the chosen model/conventions; it does not establish the cause or validate independent scalar discrepancy. Quote timing/quality and conditional carry assumptions remain possible contributors.

The symmetric candidate infers a different posterior and lower tau on its own selected prices. Its fit residual cannot establish a design winner because the observed contracts differ. A common disjoint quote panel is reported separately in `heldout.json`; its predictions remain conditional on fixed carry, selection and observed-vega weighting. Cross-sectional predictive interval fractions are descriptive, not repeated-market calibration or SBC.

On the 30 common excluded quotes, full-panel RMS residual is **12.95 half-spreads for put-heavy /16.32 for symmetric**; the short-panel values are **25.65 /25.79**. Conditional 90% interval fractions are 29/30 versus 14/30 for full, and 29/30 versus 11/30 for short. Larger fitted discrepancy contributes to the higher fractions; these figures do not establish a calibrated winner or adequate fit. This panel was selected after the pilot and is development evidence.

The affected product is the analytic Heston expected-average-variance strike between 29 January and 30 September 2027; the control runs from valuation to 30 November 2026. Values are annualized variance, not volatility or model-risk reserves. Intervals exclude integrated carry uncertainty and wider model/continuation uncertainty.

Neural fits use the baseline prior only. At 10,000 updates, full-panel neural parameter interval widths remain approximately **1.75–5.94 times** the converged reference widths. The short-panel discrepancy also remains material. The fixed 128-case synthetic development panel and market references were reused when increasing training from 2,000 to 10,000 updates; there is no fresh acceptance study and no neural fit for the wider prior. Quantile/mean comparison thresholds are point criteria; borderline coordinate failures should not be overinterpreted independently of Monte Carlo uncertainty. The overall conditional failures are large.

## Numerical controls and retained failures

Pricing refinement uses **measurement SD alone**, never tau. Sixty-six prior/corner points per declared domain were checked, plus independent QuantLib implementations and deterministic retained draws from every chain. Pointwise price accuracy and likelihood stability are reported separately: far-off-fit prior points can have larger log-likelihood shifts even when price differences are tiny.

- The baseline put-heavy panel passes the strengthened price refinement audit. Independent QuantLib differences are below .00014 measurement SD on the tested baseline panels.
- The wider vol-of-vol prior fails one extreme corner with the original final 8192-node/cutoff-4800 controls. Adding **16384 nodes/cutoff 9600** resolves it. Both final wide-prior MCMC cases use this extended ladder. Their retained parameter and sampler arrays match the earlier completed runs under the shorter controls; archive ordering/hashes can differ. Both earlier audits and failed controls remain visible.
- One symmetric-panel corner has a failed doubled audit rung despite a valid production price. Additional node/cutoff refinements and independent QuantLib on all 54 contracts establish agreement well below the .02-SD gate. `precision_symmetric.json` retains the failure; `precision_symmetric_corner.json` records its resolution.
- Retained-draw audits reprice beyond accepted controls and evaluate the actual normalized likelihood/mask. These are local numerical sensitivity checks, not a global theorem about the target or an independently refined-posterior acceptance study.

## Reproduction and payloads

Run commands from `lab/` in the locked `core-reference` environment. The optional QuantLib 1.44 cross-check was installed separately under `/private/tmp/sbi4f-quantlib`; it does not alter the project lock. Its script documents installation and exact horizon normalization.

```bash
pixi install -e core-reference --locked
pixi run -e core-reference python scripts/run_heston_observation_pilot.py design \
  --raw ../local-artifacts/heston-observations-20261109/raw.json --out /tmp/heston-designs
PYTENSOR_FLAGS=base_compiledir=/tmp/sbi4f-pytensor \
pixi run -e core-reference python scripts/run_heston_observation_pilot.py reference \
  --design evidence/pilot-results/2026-11-09/heston-observations/designs/put_heavy.json \
  --out /tmp/heston-reference-full
```

Use `--arm short`, `--spread-only`, `--spread-multiplier .5`/`2`, `--tau-max .04`, the symmetric design, or `--vol-of-vol-max 3 --extended-quadrature` for the recorded variants. Output directories must be new. `summarize` can analyze saved chains and refuses a mismatched design hash. The numeric, carry, heldout and neural audit scripts are in `lab/scripts/*heston*pilo*.py`; each provides CLI help. `summarize_heston_observation_pilot.py` rebuilds `summary.json` and the figure; `compare_heston_pilot_neural.py` rebuilds the neural comparison with payload bindings checked.

Raw source, banks, native models and posterior arrays remain under the narrowly ignored `local-artifacts/heston-observations-20261109/`, with SHA-256 identities in the committed records. Native reference arrays preserve chain/draw axes, parameter names, sampler statistics, settings and exact package versions. The public Cboe endpoint changes over time; the committed derived quotes permit replay of this fixed-panel inference without the raw download. The raw snapshot is needed to regenerate quote selection and the common heldout panel.

Initial attempts using PyTensor's default home-directory cache failed before sampling. The completed v2 baseline runs initially encountered a missing optional NetCDF writer after retaining chains; they were analyzed from intact NPZ traces without rerunning or discarding samples. Final scripts persist the structured arrays directly. These infrastructure failures did not change the financial likelihood.

## Reflection decision

Keep the European SPXW ingestion, actual-quote representation, explicit spread inputs and stricter numerical controls. Before production training or a Bergomi extension, agree the parameter support, address structured discrepancy/model adequacy and decide how much carry and quote-selection uncertainty the teaching experiment should represent. Only then select an inference architecture/training budget and run fresh conditional and synthetic acceptance checks. Wider intervals alone are not an acceptance criterion.
