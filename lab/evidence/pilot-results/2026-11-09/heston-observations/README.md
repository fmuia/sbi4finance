# Heston European-observation pilot: reflection gate

The Heston pilot is complete. **Do not promote this task or its neural checkpoints into the lessons yet.** The controlled information comparison is useful, but the market fit remains sensitive to parameter support and discrepancy assumptions, and the neural posterior fails conditional reference agreement. No rough-Bergomi jobs or lesson changes were performed.

## Full-panel training continuation

After reflection, the user requested continuation. We completed the controlled data/optimization comparison on the full 54-quote panel. It uses a distinct wider-support target (vol-of-vol up to 3), the same observation law, and fresh tau/c/quote-error draws per batch. A separate kappa-ceiling check supports retaining kappa <=10: raising it to 20 changed parameter means by at most .049 reference SD and product widths by under 2%, with chain-aware uncertainty. See `kappa-prior.json`.

One new bank contains 100,000 clean training surfaces, 5,000 validation surfaces and 2,000 reserved final surfaces. The 20,000-row arm is an exact prefix of the same training pool. Bank generation took 267 seconds. Two fresh training trajectories use the same 50,000-update learning-rate schedule, with predeclared checkpoints at 10,000 and 50,000 updates. Each full trajectory took about 875 seconds, including compilation, probes and checkpoint checks; they ran concurrently. The architecture remains a two-layer 128-unit spline conditioner, batch 128, seed 9011.

| Training surfaces | Updates | Physical-coordinate validation NLL | Market tau posterior mean | Market conditional verdict |
| ---: | ---: | ---: | ---: | --- |
| 20,000 | 10,000 | -15.867 | .00771 | Fail |
| 20,000 | 50,000 | -17.915 | .00215 | Fail |
| 100,000 | 10,000 | -15.922 | .00911 | Fail |
| 100,000 | 50,000 | -17.979 | .00184 | Fail |

These scores use the same 512 saved validation observations and physical truths, with adapter and learned-standardization Jacobians included (`training-study/physical-density-score.json`). Lower is better. The paired 100k-minus-20k differences are -.0559 +/- .0498 SE at 10k updates and -.0646 +/- .0352 SE at 50k; both approximate 95% intervals include zero. The small data-volume advantage is therefore unresolved on this panel. The plotted monitoring losses are the raw objective in each checkpoint's standardized/logit coordinates, useful for tracking optimization but not a substitute for common-coordinate scoring.

The matched reference tau mean is **.00609**. At 50,000 updates its neural mean error is **-6.15 /-6.63 reference SD** for the 20k/100k arms; MC-aware intervals exclude the allowed +/-0.3 margin by a large amount. Tau interval widths are still **2.03 /1.79 times** the reference width. Several physical-parameter intervals instead become too narrow, roughly .59–.89 of reference widths. More training therefore improves the density objective and changes uncertainty substantially, but does not monotonically improve every conditional posterior metric.

The engineering checks use the same 512 validation-pool observations with new independent tau and error draws, 512 neural posterior draws per observation, and 4,096 draws for the market and each of six synthetic conditional observations. Actual observation/truth hashes are paired across all four checkpoints. Six independent synthetic references cover broad-prior, market-like and low-discrepancy/boundary observations. **All six references converge**, and 96 retained-point numerical checks pass (worst normalized likelihood shift .000871). The slowest stress case preserves two negative observations from the untruncated Gaussian law. All four checkpoints fail the full conditional criteria on the market and on each of these six synthetic cases; individual coordinates can still agree well.

The failure is not confined to the market snapshot. Broad-prior synthetic case 0, whose true tau is .00160, still has tau width ratios **6.89 /6.18** at 50,000 updates. On market-like synthetic case 6 the ratios are **2.88 /2.60**. Average development coverage can conceal these conditional failures: the 100k/50k tau 90% interval covers about **95.3%** of the 512 truths, and rank tests reject tau calibration. This does not identify a unique cause, but it argues against assuming that a larger physical bank alone will solve the problem.

The 2,000-case acceptance pool remains unopened. No capacity comparison, seed confirmation, short-panel retraining or model publication was performed. A training plateau had **not** been established at 50k: the 100k run's fixed validation probe improved by .141 NLL from 40k to 50k, with essentially unchanged train/validation gap. The user's suggestion to train longer was therefore the appropriate next experiment. The physical-coordinate 10k-to-50k improvement was about 2.06 nats per observation.

### Additional optimization from 50k

We continued only the 100k-surface model toward 100k total updates, with fixed 75k/100k checkpoints. The target, architecture, bank, batch size, seed, Adam moments, standardization, absolute data/noise sequence and original 50k cosine schedule were preserved; its learning rate stays at the existing 5e-5 floor. Original Keras archives omitted active dropout RNG states, so the continuation reconstructs those streams from the original initialization and verified counter increments. A bounded independent resumed-versus-uninterrupted run matches all learned weights, Adam and standardization arrays exactly. New checkpoints preserve the non-trainable state in a bound sidecar. The existing evaluator and sidecar-restoring loader also produce identical seeded inference samples. See `resume-continuity-proof.json` and `continuation-inference-compatibility.json` under `training-study/`.

| Updates | Physical-coordinate validation NLL | Market tau mean | Tau mean error, reference SD | Tau 90% width / reference |
| ---: | ---: | ---: | ---: | ---: |
| 50,000 | -17.9794 | .001842 | -6.629 | 1.794 |
| 75,000 | -18.0726 | .001432 | -7.268 | 1.527 |
| 100,000 | Not reached | — | — | — |

The paired 75k-minus-50k physical NLL change is **-.09324 nats per observation**, SE .01985, approximate 95% interval [-.13215,-.05432], on the same 512 saved development observations. The fixed validation probe also improves, from -4.8958 at 50k to -4.9730 at 75k. **Additional optimization improves the average density objective but does not resolve conditional posterior accuracy.** The reference tau mean remains .006092; its neural mean moves further away. Physical-parameter widths also narrow further: kappa .593 to .540, vol-of-vol .648 to .594 and rho .710 to .650 of reference widths. Collective agreement still fails on the market and each of the six completed synthetic reference observations. Individual coordinates can agree; the other six predeclared synthetic cases remain unrun.

The run stopped when the **88,000-update monitoring check detected a non-finite loss**. The last recorded finite point is 87k. The guard checks the current batch loss and both fixed probe losses together, and raises before saving the offending values; no 88k model or optimizer state survives. The artifacts therefore do not identify which quantity failed, when it first failed after the last check, or whether weights became non-finite. A read-only audit confirms finite 75k learned/Adam state and finite inputs for the exact 88k batch, including finite forward losses at the saved 75k weights; this does not reproduce the later state or resolve the cause (`training-study/continuation-failure-audit.json`). The independently validated 75k checkpoint remains usable for development; no 100k checkpoint or score exists. Preserve this failed attempt before any diagnostic replay or optimizer change.

The [continuation figure](training-study/optimization-continuation.png), `training-study/continuation-results.json`, `continuation-density-score.json`, `continuation-protocol.json`, `continuation-evaluation-jobs.json` and `continuation-validation.json` record this extension. Its frozen producer is commit **197d9c3**. The full updated lab suite passes **618 tests**. A plateau or a capacity limitation is still not established. The next implementation step would diagnose the non-finite loss with more detailed failure capture; further optimization or conditioning/capacity comparisons need a separately recorded intervention. Market-discrepancy adequacy remains a separate scientific question. Stop for reflection before another training campaign or any Bergomi work.

The [comparison figure](training-study/training-comparison.png), `training-study/results.json`, `protocol.json`, `reference-panel.json`, `reference-precision.json`, `code-review.json` and `validation.json` record methods, identities, limitations and outcomes. Native banks/models/draws stay in the ignored `local-artifacts/heston-observations-20261109/training-study/` tree. The producer is frozen at commit **76cef0d**. Direct CLI evaluation initially failed before loading anything because of an import path; the bootstrap-only fix and a real CLI regression test are recorded, and all four evaluations then completed. Full lab validation passed **613 tests**, followed by all **nine** final evaluator regression tests after that fix. No posterior draws were clipped or dropped; tiny native-float32 endpoint excursions are counted separately from material prior violations.

Reproduce a trajectory from `lab/` with `python scripts/train_heston_pilot_campaign.py --design <fixed-design> --bank <new-bank> --out <new-run> --bank-size 100000 --validation-rows 5000 --test-rows 2000 --train-rows 20000 --updates 50000 --schedule-updates 50000` (use 100000 for the second arm). The CLI evaluator, synthetic-reference runner and summary script are `evaluate_heston_pilot_campaign.py`, `run_heston_pilot_training_reference.py` and `summarize_heston_training_study.py`. Every output directory must be new; preserve producer/task bindings when reusing an existing bank.

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

The initial pilot's neural fits use the baseline prior only; the distinct wider-support training study is reported above. At 10,000 updates, initial full-panel neural parameter interval widths remain approximately **1.75–5.94 times** the converged reference widths. The short-panel discrepancy also remains material. The fixed 128-case synthetic development panel and market references were reused when increasing training from 2,000 to 10,000 updates; there is no fresh acceptance study. Quantile/mean comparison thresholds are point criteria; borderline coordinate failures should not be overinterpreted independently of Monte Carlo uncertainty. The overall conditional failures are large.

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
