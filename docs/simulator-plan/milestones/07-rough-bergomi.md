# M7 Rough Bergomi, starting with option calibration

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** accepted one-factor steps [M6.1–M6.3](06-bergomi-factors.md) and the local bank/loader pipeline in [M2.3](02-banks-and-cluster.md). Optional M6.4 two-factor Bergomi, HKDE, and M5 are not prerequisites. Read [models and numerics](../notes/02-models-and-numerics.md) and [inference and pricing](../notes/03-inference-and-pricing.md).

## Purpose

**Implementation update:** increment 9 delivers the Volterra joint Gaussian
reference, FFT hybrid simulator/pricer and small shared Q bank. See [math and
evidence](../../rough-bergomi.md), [measured bank](../../results/rough-bergomi-option-bank.json)
and notebook 08's rough sections. M7.1–M7.2 numerical infrastructure and M7.3a
bank delivery are runnable; their scientific evidence is bounded to the recorded
fixtures/screens. M7.3b now follows the frozen [restricted posterior protocol](../../rough-inference-protocol.md); scientific acceptance remains separate from delivery. The maximum 128-to-512-step bank-grid difference is
0.0427 currency with 0.0248 combined SE: the screen passes but does not prove
bias below 0.025. No posterior or scalar-state rough continuation is claimed.

**Coupled-grid follow-up:** the [frozen paired study](../../bergomi-coupled-refinement.md) leaves the 128-step comparison unresolved at H=0.1 and eta=1.2: maximum 128-to-1,024 envelope 0.03839 exceeds 0.025. All 512-to-1,024 comparisons resolve within the budget (maximum 0.02011). This supports 512 steps as a candidate for a new explicitly numerical pilot; it does not prove continuum accuracy or change existing bank identities. The bounded numerical study is complete. M7.3b now has a declared finite-Monte-Carlo observation law, independent-bank reference and bounded training/audit pipeline. Follow its [run record](../../rough-joint-inference.md) for measured status.

Add a power-law-memory forward-variance model with an independently checked driver and controlled numerical pricing. The initial inference task is Q calibration from option panels, which avoids pretending that current variance alone solves conditional forecasting from historical returns.

## Tasks

**Original M7.3b outcome (12 October):** the fixed-ID cluster run delivered
both banks (6,146 panels; 98 tasks; 34.28 minutes wall span). The independent
reference passes 19/20 panels; `validation:260` has ESS 600.55 below 800 and
largest weight 0.005392 above 0.005. All mean/block/target-precision checks pass.
The original stop rule prevented neural training, SBC and automatic bank
expansion. A separately authorized [bounded follow-up](../../rough-reference-expansion.md)
reuses the original 4,096 reference rows and adds 4,096 new rows. It preserves
the training bank, cases and gates, and records reference and any subsequent
neural outcomes separately. Its expanded reference passes all 20 cases and all
eight gates; the original single flow candidate completed training and exact
reload and passed all seven neural validation gates, including 512-case SBC.
Conditional mean RMSE is 0.1114 reference SD versus 1.4726 for prior-only draws;
maximum target-mean error plus twice MCSE is 0.02526 currency against 0.06.
M7.3b's bounded validation and didactic deliverables are complete for this
finite numerical task. These results do not certify continuum or market accuracy.
The final test stays reserved. Notebook09 reviews
both studies; a reference pass alone does not accept a neural posterior for M8.

1. Specify the rough kernel, its normalization, roughness parameter, leverage construction, forward-variance curve, and supports. Identify precisely which Gaussian field is being simulated; do not interchange fractional Brownian and Volterra conventions by name.
2. Build a small-grid covariance-based Gaussian reference including dependence with the asset driver. Use it to check a production hybrid/convolution scheme over representative and difficult parameter settings.
3. Validate normalization and implement MC European pricing with explicit grids and budgets. Verify the estimator admits its reported error assessment: a finite price need not have finite payoff variance. Bounded put-payoff estimation plus put-call parity is a candidate within a verified martingale domain. Keep sampling error, contract dependence and discretization bias explicit.
4. Generate a small option-panel bank with Q parameters and any declared curve coefficients as targets. Synthetic admissible curves, contracts, rates, and quote observations suffice for S; a real surface is optional M input. Reuse bank/adapter contracts and train a pilot posterior for a restricted task. Assess time-grid/kernel refinement, conditional cases and price-sensitive posterior functionals before increasing volume; unresolved reference accuracy limits acceptance for teaching.
5. If pursuing a finite-factor approximation, version its kernel, weights, factor count, and error targets as numerical settings for the specified rough model. Compare it against the reference; do not equate it with an independently estimated two-factor Bergomi model.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M7.1 Gaussian driver reference | Explicit kernel/field convention, supported parameter domain and small-grid joint covariance fixtures | Check Gaussian covariance and asset-driver dependence independently before production optimization. |
| M7.2 Reusable simulation and pricing | Hybrid/convolution generator, variance normalization, European pricer and numerical error report | Compare with M7.1, refine grids and justify the price estimator's moments/error bars before accepting a reusable training bank or scaling it. Small explicitly provisional pilots can inform these choices. |
| M7.3a Restricted bank | Q task, grouped pilot bank, numerical metadata and verified load/regeneration | Reuse the shared panel helper and M2.3 loading. Record fidelity limits; settle the selected price-label budget before scaling. This is available in increment 9. |
| M7.3b Restricted posterior | Pilot checkpoint and reproducible conditional/reference validation report | The frozen H/eta pilot uses 512 steps, a separate prior-importance reference and a single flow candidate. Follow the substeps below; working code or marginal SBC alone does not accept the posterior for M8. |
| M7.4 Optional numerical factor approximation | Versioned approximate kernel/weights and accuracy/cost comparison | Compare with the accepted rough reference at declared error targets. Do not present this approximation as the independently specified M6.4 model. |

### M7.3b bounded posterior subplan

The [frozen protocol](../../rough-inference-protocol.md) records all counts,
seeds, cases and gates before simulation. The [run record](../../rough-joint-inference.md)
separates implementation, reference adequacy, neural acceptance and review execution.

| Substep | Output | Acceptance boundary |
|---|---|---|
| M7.3b.1 Observation contract | Fixed Q H/eta prior, nine quotes, known curve and context; finite 512-step/262,144-path numerical observation law | Pricing randomness remains explicit. No claim of a continuum posterior or P forecast. |
| M7.3b.2 Independent banks and reference | 1,024 training bases, separate selection/audit/test; original 4,096-row reference and one authorized extension to 8,192; fixed-ID cluster chunks assemble to the existing schema | Check precision, weights and independent reference blocks. Preserve both reports. The extension permits no further automatic doubling. |
| M7.3b.3 Single joint flow | Eight quote augmentations per training base; one 80-epoch coupling-flow budget and exact checkpoint reload | Train only after the matching reference passes. Keep augmentations grouped and reserve audit cases from selection. |
| M7.3b.4 Inference and price audit | Conditional/joint comparisons, prior-only control, marginal SBC and two excluded-contract posterior means | Use independent target randomness and report sampling uncertainty. No price-quantile claim; preserve failed gates and reserved final test. |
| M7.3b.5 Didactic review | Notebook09, reproducible graphs, actual costs, pass/fail outcomes and next-analysis note | Verify offline with all market fixtures absent; cached review does not silently train or launch cluster jobs. |

Complete M7.1–M7.3b in order. M7.4 is optional and may follow the accepted numerical reference without delaying the reference-based posterior pilot. Flat `rough_bergomi.py` and the shared `option_panel.py`/`option_bank.py` now implement the numerical and bank responsibilities. Later inference can reuse these without creating a hidden dependency on M4 or M5.

## Didactic review deliverable

The rough-driver and numerical-pricing sections are delivered in `notebooks/08_bergomi_models.py`, following the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone). They explain the Volterra kernel, normalization and memory requirement with a small Gaussian-reference example and cached numerical evidence. The [coupled-grid follow-up](../../bergomi-coupled-refinement.md) adds a live pairing example and measured comparisons. The separate `notebooks/09_rough_inference.py` supplies the M7.3b review: a live importance-weighting example with an analytic answer, cached reference/conditional/SBC/mean-price evidence, prior controls and explicit missing-result states. Numerical review does not supply conditional-inference evidence. Exercises distinguish a finite-factor approximation from a two-factor model and explain why current variance alone cannot initialize a P rough-memory forecast.

Accept the sections after offline headless execution/export reproduces the bounded reference demonstration and renders the separate numerical/posterior gates. Large covariance studies, pricing banks and training remain opt-in; a failed inference gate stays visible even if simulation and notebook execution pass.

## Deliverables

Gaussian-driver fixtures, a checked production simulator/pricer, one Q calibration configuration, a pilot bank/posterior, an accuracy/cost report covering roughness and leverage regimes, and verified M7 review sections. Separate simulator acceptance from posterior acceptance; the latter includes a reproducible validation procedure and its reference limitations.

## Acceptance and tests

Check Gaussian covariances, asset-driver dependence, variance expectations, deterministic-volatility limits, pricing identities and refinement stability. Require a justified estimator/error model before reporting MC precision; see the [moment discussion](../notes/03-inference-and-pricing.md). Set tolerances before full-bank production. Inspect difficult prior regions and conditional inference cases, not only average error or marginal SBC. Compare numerical results against scientific budgets rather than seeded trajectory equality alone.

## Decision gate and deferred scope

Accept a production bank only after the driver and pricing gates pass, independently of market acquisition. If posterior reference accuracy is unresolved, accept numerical infrastructure separately and defer an inference teaching claim. Q calibration is neither a P forecast nor a future-chain forecast. A P forecast requires declared P dynamics and inferred or conditionally reconstructed memory; true simulated memory labels do not solve that inference problem. Defer such forecasts, joint P/Q mapping, rough Heston and arbitrary path-dependent products until separately validated.
