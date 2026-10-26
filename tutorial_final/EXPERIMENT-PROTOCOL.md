# Protocol for the numerical experiment campaign

Proposed on 26 October 2026. Read the [human overview](NUMERICAL-EXPERIMENT-PLAN.md) first. This protocol defines the decisions needed to turn that plan into reproducible runs. New criteria below are proposals to freeze before execution; they do not change historical acceptance records.

**Scope:** complete the full-surface Heston and joint rough workflows, applying the numerical/statistical checks to the retained claims. All sparse-chain support is a separate downstream unit, beginning with specification; no sparse adapter, checkpoint, mask training, market example or reserve-coverage study gates this release. The wider architecture benchmark and direct rough MCMC remain optional extensions. Infrastructure unit tests do not substitute for numerical and inferential checks.

## Tasks and common implementation

The primary programme has two configurations: full-surface Heston with four targets and fixed mean reversion, and full-surface rough Bergomi with six joint targets, `H, eta, rho, short_variance, long_variance, decay`. Record each prior, observation design, quote-noise model, inferred coordinates, numerical algorithm and products. The primary rough task receives quotes, not a supplied curve. The current supplied-curve and seven-level rough pipelines are historical/optional comparisons, not implementations of the new task.

The core input contract requires all 63 declared contract prices, finite values, valid positive noise metadata and the supported design/order. Explicitly reject incomplete or unsupported inputs rather than silently filling prices or dropping contracts. For set inputs, preserve complete quote records when reordering. Market gridding/interpolation is a labelled observation/preprocessing approximation with source and coverage diagnostics; reading a smooth surface at 63 locations does not create 63 independent observed quotes. Use the declared noise assumptions consistently in training, references and audits. No generic missing-value or raw-chain handling is promised by the core.

The [sparse follow-up specification](SPARSE-QUOTE-CASE-PLAN.md) records unresolved observation, selection, training and evaluation decisions. Start that unit after the core tutorial is complete unless explicitly reprioritized. Preserve compatible full-grid clean prices and historical EWG results without building sparse adapters or checkpoints now.

### Primary rough curve family and literature scope

The agreed simplified tutorial family is `xi_0(t) = long_variance + (short_variance - long_variance) exp(-decay t)`, giving six joint targets. The two levels are variance quantities, not volatility percentages. Freeze positive finite bounds/priors for the levels and decay, the rough-parameter prior, horizon and noise model before substantial generation. Positivity follows for every nonnegative t from positive endpoint levels and decay. The curve can rise, fall or be flat, but cannot have an interior hump. Use the actual declared coefficient prior in reference MCMC and proposal densities, including transformation Jacobians. Do not reuse the old seven-level curve prior or infer a new coefficient prior by projecting old curves.

The three coefficients are inferred within rough Bergomi, not copied from a fitted Heston model. Here decay governs the initial mean term structure; the rough variance process retains its lognormal rough dynamics. This is not a nested Heston process or rough Heston. With nearly equal endpoint levels, decay can remain weakly identified. Report posterior curve bands/products and coefficient dependence; do not exclude flat curves solely to manufacture coefficient recovery. Heston's current `kappa=2` restriction remains explicit: a controlled cross-model mean-curve comparison must match the relevant values/decay rather than attributing all differences to roughness.

[Rømer (2022), Section 3.3](https://doi.org/10.1080/14697688.2022.2081592) uses Heston-derived curves for rough-pricer training, with perturbations and other shapes; that is not evidence for a three-coefficient restriction in his market calibration. [Baschetti, Bormetti and Rossi, Section 4.3](https://arxiv.org/html/2306.11061v2#S4.SS3) jointly fit `xi_0(t) = b0 + b1 exp(-t/tau1) + b2 (t/tau2) exp(-t/tau2)`. The baseline sets the hump term to zero and removes its unused timescale. It is a literature-grounded teaching restriction rather than a universal standard. For the five-coefficient alternative, require positive timescales and positivity over the entire declared pricing/product horizon; positive endpoint values alone do not exclude a negative interior dip.

The completed [bounded price-fit challenge](../docs/curve-family-screen.md) retains a constructed hump case and its point-product discrepancies. The scope decision accepts the monotone restriction for the core lesson; the challenge does not require automatic promotion to eight targets or a second bank. Joint posterior/product checks within the chosen family remain required. A wider-family posterior and uncertainty comparison becomes necessary if a later lesson claims robustness to hump-shaped curves or general market adequacy. Such a comparison must use the same data/error conventions, family-compatible numerical pricing and the numerical/reference uncertainty and product tolerances below. Narrower intervals alone do not establish superiority.

Implement the chosen curve consistently on the pricing time grid. For the baseline, `W(T) = long_variance*T + (short_variance-long_variance)*(1-exp(-decay*T))/decay`. Interval averages are `(W(T_i)-W(T_{i-1}))/(T_i-T_{i-1})`, not midpoint samples. Seven such averages preserve integrated expected variance at those endpoints, but the resulting piecewise-flat process is still an approximation to the smooth input curve and can change option prices. Prefer evaluation on the simulator's finer time cells; whichever numerical convention is selected must agree in simulation, numerical references and control-variate expected integrals, with time-grid refinement checked. Record the curve formula, conversion rule and extrapolation through every product horizon.

### What the forward-variance curve input means

For instantaneous variance `v_t`, the initial curve is `xi_0(t) = E^Q[v_t]`, under the pricing measure. It is an input function in rough Bergomi; in Heston its form is implied by the model parameters, `theta + (v_0 - theta) exp(-kappa t)`. It is distinct from a historical-volatility forecast or an at-the-money implied-volatility curve. [Bayer, Friz and Gatheral](https://realvol.com/RVOL2Paper.pdf)

With continuously monitored variance-swap strikes `K_var(T)` quoted in variance units, total expected variance is `w(T) = T K_var(T) = integral_0^T xi_0(s) ds`. Estimate piecewise-constant forward levels using `(w(T_i) - w(T_{i-1})) / (T_i - T_{i-1})`. If strikes are quoted as volatility percentages, square the decimal volatility first. Alternatively replicate the log contract from a strip of European out-of-the-money puts/calls across strikes at each expiry, with consistent forwards, discounts and time conventions. This is the variance-replication principle behind the [Cboe methodology](https://cdn.cboe.com/resources/indices/Cboe_Volatility_Index_Mathematics_Methodology.pdf); finite strikes, tail treatment, market noise and exercise conventions limit the approximation.

The existing [`chain.build_surface`](../src/sbi4finance/calibration/chain.py) integrates cleaned finite option strips, retains expiries meeting its wing-coverage rule, linearly interpolates total variance to seven design maturities, and takes first differences using [`surface.forward_variances`](../src/sbi4finance/calibration/surface.py). This optional conditional pipeline does not fit a positivity-constrained total-variance curve or repair nonpositive levels; invalid levels are rejected downstream. Differencing can amplify estimation noise. Check positivity, wing truncation, interpolation sensitivity and exercise/forward assumptions before using its estimate as fixed context. These seven numerical intervals are not seven primary inferred coefficients in the new baseline.

Synthetic conditioning on a supplied true curve is a controlled problem. Market conditioning on a curve estimated from the same option chain is a plug-in approximation, potentially using more strikes than the design grid; the curve is a correlated, noisy derived summary rather than an independent exact measurement. Do not add it as another independent likelihood factor or interpret conditional posterior widths as including curve-estimation uncertainty. Joint curve inference is the corresponding uncertainty-aware task. Treat genuinely separate variance-swap quotes through their declared observation model if used.

Use a common application lifecycle: prepare observation, infer, diagnose, correct or stop, then propagate joint draws. The training lifecycle selects a validation checkpoint, reloads it and verifies the restored estimator. Application-specific prices, curve priors, adapters and audit methods remain components. Extending the shared core should preserve the historical SVJ/HKDE contracts and their source-bound evidence.

Each immutable run records code revision, task and bank hashes, data IDs, observation/noise and masking rules, architecture, training seed, optimizer schedule, selected epoch, inference settings and full environment/device versions. Resume identity includes those fields and the full training-data/augmentation identity. A nonempty directory without a valid matching resume record is rejected. No setup step deletes an unrelated run.

## BayesFlow migration checks

Pin BayesFlow **2.1.0** and resolve a separate CPU/GPU lock. Version 2.0.x serialized networks are incompatible; retain the original environment for archived checkpoints. The release tag is `v2.1.0`, at commit `d4421e7e68cfa4c7abe9483aedb416b65321d98f`. [Release notes](https://github.com/bayesflow-org/bayesflow/releases/tag/v2.1.0)

The four candidate APIs are:

```python
bf.networks.CouplingFlow(transform="affine")
bf.networks.CouplingFlow(transform="spline")
bf.networks.FlowMatching()
bf.experimental.MeanFlow()
```

Use the estimator API (`workflow.estimator`, `load_estimator`) and network-specific sampling. The current project sampler unconditionally applies FlowMatching integration settings; replace that assumption before comparing families.

Both continuous families place data at time zero in 2.1, but their training-time sampling formulas and default alpha values differ. Do not copy the old Euler grid or pass a single common alpha blindly. MeanFlow starts with `density_head=False`; learned density heads are an optional later experiment. [FlowMatching](https://github.com/bayesflow-org/bayesflow/blob/v2.1.0/bayesflow/networks/inference/flow_matching/flow_matching.py), [MeanFlow](https://github.com/bayesflow-org/bayesflow/blob/v2.1.0/bayesflow/experimental/mean_flow/mean_flow.py)

Before using a configuration in a screen or teaching fit, test that actual family/encoder/task combination on tiny data. Do not implement or validate a 36-combination Cartesian product as a prerequisite. Verify the applicable properties:

- A finite training update, correct support and outputs shaped `[batch, draws, targets]`; token input axes must not become parameter-sample axes accidentally.
- Adapter transformations and density Jacobians on a tractable toy problem.
- Checkpoint selection, same-device seeded save/reload and compatible/incompatible resume handling.
- Permutation invariance of complete quote records for set encoders; rejection of missing/nonfinite prices and unsupported designs for core inputs. Mask/padding equivalence is a downstream sparse requirement if that representation is selected.
- Actual JAX GPU placement, recorded driver/CUDA/backend versions and bounded memory use.

Seeded reproducibility is checked within a recorded device/environment; CPU and GPU draws need not be bit-identical. Synchronize asynchronous backend work for timings and report compilation/loading separately from warm execution.

## Data and holdouts

Reuse compatible Heston training/validation data. Use the original split sizes, Heston 50k/5k/5k and rough 100k/5k/7k, as initial allocations rather than evidence of sufficiency. The six-target rough family/prior/numerical definition requires a new task, collection and price generation before expansion. Old rough curves, prices, IDs and checkpoints do not become samples from this new task by relabelling or projection. Reuse the storage machinery and code where valid; preserve old data as historical or optional separate comparisons. Store all six targets and derived curve values with explicit roles; the primary joint estimator conditions on complete quotes/design/noise only, never its generating curve. Every noise augmentation stays in its base split; missingness augmentation is deferred.

Use the [incremental-bank workflow](../docs/incremental-calibration-banks.md) for nested 50k/200k/1M Heston and 100k/300k/1M new-task rough snapshots. Preserve clean prices, covariance/error records and stable IDs; expand selected-baseline learning curves when justified rather than fitting every candidate at each size.

A collection freezes the task, root seed, generation sources/runtime and seed protocol independently of training size. New tranches use only missing IDs; fixed validation/test prefixes never migrate into training. Adopt compatible old shards after checking identity, bytes, arrays and numerical budgets. Snapshot publication checks exact nonoverlapping coverage and collisions in the actual reduced pricing seeds. Record the immutable snapshot manifest digest in every run and retain the referenced shards. A task, root or numerical implementation change requires a distinct collection; do not merge independent roots behind ambiguous local IDs.

Learning curves use validation/development data, not final test data. Evaluate conditional-reference accuracy, calibration, difficult parameter/curve strata and product quantities with adequate optimization at each size. A flat learning curve with failed accuracy gates calls for diagnosis, not an automatic declaration of sufficiency. Dataset growth requires a new training directory and declared schedule; same-run recovery binds the original snapshot and settings. Warm-start training across snapshots is outside this implementation.

Create a data-use ledger before screens:

| Role | Permitted use |
| --- | --- |
| Training | Fit parameters and training-only normalization. |
| Validation and development references | Select family, encoder, capacity, solver, training budget and the production checkpoint/seed rule. |
| Frozen acceptance | Evaluate the preselected production configuration and its seed replicas once; no model or sampler selection afterward. |
| Market cases | Demonstrate conditioning and model checks; provide no known parameter truth. |

Existing inspected Heston and old rough-pilot test results are historical/development evidence. Generate and freeze new rough holdouts for the selected family/prior/numerical task. If any proposed acceptance pool influenced choices, generate a fresh supplemental bank with a different root seed. Additional acceptance simulations for an unchanged task do not require regenerating its training bank.

Prepare development reference panels separately from the final library. Predeclare typical, nearly flat/weakly identified and boundary cases rather than choosing attractive examples after seeing neural errors. All core panels use the complete declared design. New-task holdout identities must not be inherited from old rough results as if their prior and numerical model were unchanged. A later sparse study needs its own development/acceptance ledger and must not treat already inspected core outcomes as untouched selection evidence.

Final tutorial evaluation uses the two selected full-surface configurations and their three training-seed replicas. Use 2,000 untouched latent surfaces per model for SBC/coverage on the complete design, plus a fixed conditional-reference collection and disjoint audit panels. These checks are relatively inexpensive compared with rough generation. No sparse acceptance campaign is included. Runner-up evaluation is optional. Record IDs, noise seeds, checkpoints and metrics/bands in existing metadata before opening acceptance results; no new registry system is required. Do not use acceptance outputs to choose the model. A changed candidate or criterion needs a declared fresh evaluation, with earlier outcomes retained.

## Numerical precision

Freeze whether each task concerns ideal expected model prices or the finite numerical simulator used in the bank. Specify quadrature, time grids, adaptive path rules and all pricing randomness.

Heston generation uses the checked Fourier quadrature ladder and its existing refinement budget of 0.1 quote-noise standard deviations. For rough generation, retain the paths ladder and the 0.25 target for Monte Carlo standard error divided by quote noise. The existing final-rung allowance of 0.5 is an exception to report and review, not evidence that every surface met 0.25.

Assess time-grid bias at representative prior points, difficult low-Hurst/high-vol-of-vol corners and development posterior locations. The current mini-pilot has a small median shift but a maximum around one quote-noise standard deviation. Freeze the precision-control cases before production. Proposed refinement limits on those cases are posterior-mean movement at most 0.1 refined-reference SD, 90% interval-width movement at most 10%, and product-quantile movement at most one quarter of the product-error tolerance below. Measure Monte Carlo uncertainty tightly enough to resolve those limits. If a case exceeds them, refine the numerical model or explicitly restrict the claim to the frozen discretization before generating a production bank. Record which of these targets the scientific owner accepts.

Use full price-error covariance when quotes share paths. Coupled random numbers help measure numerical differences; independent pilot and final streams are needed where an estimator relies on independent precision selection. Keep unpriced surfaces/proposals visible. Do not replace failed seeds or treat a surface with unpriced proposals as a reliable audit.

Before relying on the rough likelihood audit, validate its error/covariance treatment on a Heston control with a refined Fourier reference and on a tractable noisy-price toy. Covariance subtraction is exact only under its stated distributional assumptions; finite pricing errors and estimated covariance require sensitivity checks.

## Training comparisons

Use these observation encoders:

| Encoder | Initial design |
| --- | --- |
| Direct vector | Ordered complete prices and declared noise metadata; no learned compression or generating-curve context in the primary joint task. |
| DeepSet | Quote/design/noise tokens; an initial 32-dimensional summary. |
| SetTransformer | The same tokens and initial summary size, allowing cross-contract interactions. |

Start with the same small summary for the six-target rough problem; investigate a larger summary only if development diagnostics expose a compression problem. A small CNN on the 7 × 9 grid is optional. Recurrent/time-series networks are outside this option-surface benchmark.

Each token contains a forward-unit price, log-moneyness, maturity, option type and quote-noise/spread information for one of the 63 observed design contracts. Include noise itself when it varies; a noise-scaled price alone does not identify the observation law. Avoid requiring implied volatility for synthetic noisy quotes where inversion is undefined. Keep supplied curve context separate from inferred targets. Set encoders are compared on complete sets; they do not establish arbitrary-mask, variable-length or raw-chain support.

The current EWG network's 27-point proposal layout and actual 61-quote audit are distinct historical objects. Preserve their labels; selecting or replacing the sparse market case belongs to the downstream unit and is not required here.

After the first checked core examples, use this bounded comparison:

| Stage | Configurations | Fits |
| --- | --- | ---: |
| Heston family screen | 4 posterior families with the direct vector | 4 short fits |
| Encoder comparison | DeepSet and SetTransformer with one fixed family on one representative task; reuse its direct-vector baseline | 2 additional short fits |
| Rough transfer screen | 2 promising configurations on an intermediate bank | 2 fits |
| Confirmation | 1 selected configuration × 3 seeds × full-surface Heston and full-surface joint rough | 6 teaching fits |

This budgets at most eight screen fits plus six confirmations, or fourteen posterior fits before selected-baseline learning curves; reuse identical runs where possible. Forward-price emulator training is additional. The curve-family adequacy pilot is separately bounded development work, not a second full posterior campaign. Each screen uses one declared seed and fixed tuning budget; confirmation quantifies seed variability. Sparse fits belong to the separate downstream unit. Supplied-curve rough fits, a full family × encoder × task matrix, runner-up seed campaigns and capacity sweeps are optional research, not table-completion requirements.

Use small nested subsets for software and short architecture screens. Measure larger data-sufficiency curves only for selected credible baselines, using the expandable-bank sizes in the central plan; one million is capacity, not an automatic fit requirement. Select each run's checkpoint by its own validation objective. Compare families using common external diagnostics, not unlike training losses. Report parameter counts, data exposure, convergence and compute effort; identical epoch counts alone do not establish fairness.

Fix the production checkpoint/seed selection rule using development results before final acceptance. Preserve all confirmation seeds. Do not select the best-looking acceptance replicate.

Evaluate MeanFlow at 1/2/4/8 steps from one trained checkpoint. Check FlowMatching solver tolerance/step refinement. Direct neural-density proposals are optional research: FlowMatching density evaluation needs validated trace/solver handling, and MeanFlow needs validated map Jacobians and round trips before its density is used as an importance denominator. These mechanisms are not needed for sample-based teaching or the existing fitted-proposal audit.

The existing audit fits a Student-t proposal to neural draws and mixes it with the prior. Because its proposal density is known, it works for every family without requiring network density. Use it first and report its diagnostics/failures; a fitted Student-t can lose multimodal structure. Add another proposal only if an observed failure requires it. [Current correction](../src/sbi4finance/calibration/correction.py)

## Reference posteriors

**Prefer PyMC for standard references and emulator-MCMC where it fits naturally**, following [the repository policy](../docs/mcmc-policy.md). Specialized teaching, particle/pseudo-marginal or disproportionately costly integration cases can use a documented custom or other-framework sampler. State the method, reason, target and limitations; the validation requirements below apply independently of framework. Resolve and lock the actual execution environment, and use a step method appropriate to the target rather than assuming all models support NUTS. Historical chains retain their original sampler labels and provenance. PyMC adoption can be deferred without requiring wholesale migration or blocking other experiments.

### Heston

Create development panels first. After selection is frozen, create a 50-case synthetic reference library plus SPY, with refined Fourier pricing and four dispersed chains per case. Starts must not all come from the neural candidate.

Proposed reference requirements are rank-normalized split R-hat at most 1.01, bulk/tail ESS at least 400 for parameters and important products, and stability under price refinement. Record Monte Carlo uncertainty in the reference summaries. A nonconverged chain is inconclusive, not a neural-method failure or a successful reference.

### Rough Bergomi

Use a reference ladder:

1. Train a separate forward price emulator on clean training labels. Validate it globally and in development posterior regions using independent, higher-precision pricing. Compare emulator-MCMC on about 50 final panels. Shared bank errors mean agreement is a cross-check, not ground truth.
2. Run independent Monte Carlo audits on 50 final panels; extend to 200 only after inspecting reliability and cost under a frozen expansion rule. Report ESS, largest weight, Pareto k-hat, unpriced proposals and approximation sensitivity. Weight concentration alone does not establish conditional accuracy.
3. Optional research: pilot direct Monte Carlo-based MCMC on one declared case. Measure likelihood variability, mixing, ESS and path/grid dependence. Extend to 3–5 fixed anchor cases only if justified by the pilot and the intended claim. This is outside the tutorial critical path unless a direct-MCMC agreement claim is included; that claim requires completed, converged anchors.

Keep the likelihood targets distinct:

| Target | Interpretation |
| --- | --- |
| Gaussian noise around ideal expected prices | Requires control of numerical price and likelihood approximation. |
| Gaussian noise around prices with a fixed simulation seed | A deterministic numerical approximation; different seed targets must not be pooled as chains of one posterior. |
| The finite simulator, averaged over its pricing randomness | A legitimate numerical generative model that can match bank generation, including its adaptive pricing rules. |

For the last target, with `y = P_hat(theta, U) + noise`, `N(y; P_hat(theta, U), Sigma)` is nonnegative and unbiased for that simulator-marginal likelihood. A valid pseudo-marginal chain retains its current likelihood/randomness on rejection. Every implementation must preserve that auxiliary state; a black-box operation returning a fresh random log likelihood is not a sufficient implementation. A validated specialized sampler can remain outside PyMC. This does not make it exact for ideal expected prices. [Andrieu and Roberts](https://arxiv.org/abs/0903.5480)

The current covariance-subtraction estimator `N(y; P_hat, V - S)` has the claimed ideal-price expectation under Gaussian pricing error with known covariance. Estimated covariance and finite Monte Carlo errors do not establish general unbiasedness. Label this approximation and test it; do not call the rough audit an unconditional exact reference.

## Metrics and decisions

Register tolerances before final evaluation. For product comparisons define `W = q95 - q05` of the reference posterior product price. Register a positive absolute tolerance `epsilon` in each product's units; deterministic controls in any optional supplied-curve benchmark need that absolute rule. In the primary joint rough task, variance-swap prices vary with the inferred curve and are not deterministic controls. Define product tolerance `T = max(0.10 * W, epsilon)` and require numerical/reference uncertainty at most `T / 2`. Refinement tests use at most `T / 4`. The following are proposed starting rules, preserving historical conditional tolerances where applicable:

| Question | Measurement and proposed rule |
| --- | --- |
| Does the posterior learn from data? | Prior-to-posterior contraction and recovery, including joint quantities. SBC alone is insufficient. |
| Is it calibrated under the declared simulator? | SBC with a predeclared simultaneous band and coverage intervals; account for the collection of targets/configurations being claimed. Report all seeds. |
| Does it agree conditionally? | Mean error below 0.3 reference SD and 90% interval-width error below 25%, with reference uncertainty reported. Show every predeclared panel. |
| Do product conclusions agree? | Absolute q10/q90 errors at most the declared product tolerance T, with the uncertainty rule above. Register the products, units and absolute tolerances; zero-width controls use epsilon. |
| Is the audit usable? | Existing proposal-pricing/reliability rules, ESS and both fixed and sample-size-sensitive k-hat diagnostics; failed or unpriced cases remain in the report. |
| Is amortization worthwhile? | Total cost at 100/1,000/10,000 compatible cases against classical-proposal importance sampling and emulator-MCMC, at matched accuracy. |
| Is the lesson dependable? | Fresh CPU participant execution, correct cached/live labels, bounded controls and no writes to tracked reference checkpoints. |

A blanket claim that the raw network agrees across the final reference collection requires that the stated criteria hold across that collection. Otherwise report its scope and failure rate, and assess the corrected result separately. Do not let correction conceal a poor raw amortizer. Numerical audits, model adequacy and empirical market performance remain different claims.

Keep a bounded prior/noise and curve/product-extrapolation sensitivity check on complete inputs. Successful prior-predictive calibration does not establish robustness to misspecified curve or observation laws. Historical EWG-conditioned truths describe local sensitivity, not broad market validation. Missing-quote comparisons, sparse market selection and the five-method population reserve-coverage programme are deferred; their scope and acceptance criteria must be specified separately.

Measure cold load/compilation, warm draws, density evaluation when used, audits, products, memory and total training cost. Price-emulator MCMC and classical-proposal importance sampling are required competitors. The bank/training cost is amortized over explicitly compatible tasks; a changed prior/design/noise law may require new validation or training.

## Submission and release

Use immutable Oscar checkouts and actual account/partition availability. CPU arrays generate price surfaces and reference work; GPU arrays index task/encoder/family/seed. Record per-job resources and avoid accidental multiplication of array workers and library threads. Enforce memory and concurrency limits after the pilot.

Submit only the agreed phase allocation. The historical 930 CPU-hour estimate included the roughly 360-hour seven-level bank and earlier baseline studies; it is not the new six-target campaign's budget. Re-estimate generation, selected posterior/emulator fits and the bounded family check from the new-task pilot, retaining CPU work and elapsed time as separate quantities. Direct reference anchors remain optional additional work.

Update environment/API/test commands in [CLUSTER-RUNS.md](CLUSTER-RUNS.md) before use. The existing scripts are starting points, not validated BayesFlow 2.1 commands. Refresh the [status](../docs/course/status.md), [claims](CLAIMS.md) and accepted-artifact manifest together after results are reviewed.

Release one selected full-surface checkpoint per model, seed-replica results, the completed comparison report, scoped reference plots, failures, participant exercises and cached fallbacks. A broad architecture benchmark can finish after the tutorial release; one reliable selected configuration per model is enough for the live workflow. Full rough-accuracy claims require their corresponding studies. Sparse-chain results and population reserve-coverage claims are excluded from this release; a market illustration alone does not establish coverage. Core completion opens the specification stage of the separate sparse unit, not an automatic sparse training campaign.
