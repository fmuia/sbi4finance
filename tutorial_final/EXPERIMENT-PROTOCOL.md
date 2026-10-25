# Protocol for the numerical experiment campaign

Proposed on 26 October 2026. Read the [human overview](NUMERICAL-EXPERIMENT-PLAN.md) first. This protocol defines the decisions needed to turn that plan into reproducible runs. New criteria below are proposals to freeze before execution; they do not change historical acceptance records.

## Tasks and common implementation

Keep Heston, liquid rough Bergomi and thin rough Bergomi as distinct tasks. Record the complete prior, observation design, quote-noise model, supplied context, inferred coordinates, numerical pricing algorithm and downstream quantities for each. The liquid rough task conditions on seven curve levels. The thin task infers them; its baseline receives exactly the same observed information as the neural method.

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

Before full runs, test all 4 families × 3 encoders × 3 tasks on tiny data. These 36 software checks are not production fits. Verify:

- A finite training update, correct support and outputs shaped `[batch, draws, targets]`; token input axes must not become parameter-sample axes accidentally.
- Adapter transformations and density Jacobians on a tractable toy problem.
- Checkpoint selection, same-device seeded save/reload and compatible/incompatible resume handling.
- Set permutation invariance, padded/unpadded equivalence and exclusion of masked values throughout pooling and attention.
- Actual JAX GPU placement, recorded driver/CUDA/backend versions and bounded memory use.

Seeded reproducibility is checked within a recorded device/environment; CPU and GPU draws need not be bit-identical. Synchronize asynchronous backend work for timings and report compilation/loading separately from warm execution.

## Data and holdouts

Reuse compatible training and validation data. The existing designs are Heston 50k/5k/5k and rough 100k/5k/7k train/validation/test surfaces. Preserve clean prices, covariance/error records, context and stable surface IDs. Every noise or mask augmentation of one base surface stays in its original split.

Create a data-use ledger before screens:

| Role | Permitted use |
| --- | --- |
| Training | Fit parameters and training-only normalization. |
| Validation and development references | Select family, encoder, capacity, solver, training budget and the production checkpoint/seed rule. |
| Frozen acceptance | Evaluate the preselected production configuration and its seed replicas once; no model or sampler selection afterward. |
| Market cases | Demonstrate conditioning and model checks; provide no known parameter truth. |

Existing inspected Heston and pilot test results are historical/development evidence. Freeze untouched full-rough IDs if available. If any proposed acceptance pool influenced choices, generate a fresh supplemental bank with a different root seed. Additional acceptance simulations do not require regenerating the training bank.

Prepare development reference panels separately from the final reference library. Predeclare typical, weakly identified and boundary cases rather than choosing attractive examples after seeing neural errors. The same untouched rough surfaces can support paired liquid/thin comparisons; report that dependence instead of counting them as independent experiments.

Final tutorial acceptance evaluates the production configuration chosen on development data and all three of its seed replicas. Include 2,000 synthetic observations per retained task for SBC/coverage, a fixed conditional-reference collection, and disjoint audit panels. The architecture benchmark can evaluate the runner-up configurations under a separately registered comparison, without using acceptance to choose the deployed model. Register IDs, noise seeds, candidate/checkpoint identities, metrics and simultaneous calibration bands before opening acceptance results. Do not reuse the same test outputs to tune architecture choices. A changed candidate or criterion needs a new declared evaluation, with earlier outcomes retained.

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
| Direct vector | Ordered complete prices, with explicit curve context where known; no learned compression. |
| DeepSet | Quote/design/noise tokens; an initial 32-dimensional summary. |
| SetTransformer | The same tokens and initial summary size, allowing cross-contract interactions. |

For ten-target thin inference, compare 32 and 64 summary dimensions during development if compression appears limiting. A small CNN on the 7 × 9 grid is optional. Recurrent/time-series networks are outside this option-surface benchmark.

Each token contains a forward-unit price, log-moneyness, maturity, option type, quote-noise/spread information and mask. Include noise itself when it varies; a noise-scaled price alone does not identify the observation law. Avoid requiring implied volatility for synthetic noisy quotes where inversion is undefined. Keep supplied curve context separate from inferred targets.

Begin with subsets/masks of the declared contract grid. Generalization to arbitrary raw chains requires simulation designs covering those contracts. The EWG network's 27-point proposal layout and the actual 61-quote audit are distinct objects; document both.

The proposed capped screen is:

| Stage | Configurations | Fits |
| --- | --- | ---: |
| Encoder pilots | 3 encoders × Heston/liquid rough with one fixed posterior family | 6 small fits |
| Thin encoder pilots, if retained | Direct vector and best set encoder | 2 small fits |
| Heston family screen | 4 posterior families × direct vector/best set encoder | 8 fits |
| Rough family screen | 4 families with the selected encoder on an intermediate bank | 4 fits |
| Confirmation | 2 shortlisted configurations × 3 seeds × each of 3 retained tasks | 18 full fits |

This is at most 38 scheduled fits before optional learning curves; reuse identical configurations rather than rerunning duplicates. Each screen uses one declared seed and a fixed tuning budget. Confirmation quantifies seed variability. One selected configuration per task with three seeds is the tutorial-release requirement (nine full fits if all tasks remain); the other nine confirmation fits and uncompleted screens belong to the benchmark. If thin inference is removed from release, reduce its rows explicitly. Add a small targeted encoder/family interaction check only if development results justify it.

Use deterministic nested learning-curve subsets: Heston 5k/20k/50k and rough 10k/30k/100k. Select each run's checkpoint by its own validation objective. Compare different families using common external diagnostics, not the magnitude of unlike training losses. Report parameter counts, data exposure, convergence, GPU-hours and hyperparameter-search effort; identical epoch counts alone do not establish fairness.

Fix the production checkpoint/seed selection rule using development results before final acceptance. Preserve all confirmation seeds. Do not select the best-looking acceptance replicate.

Evaluate MeanFlow at 1/2/4/8 steps from one trained checkpoint. Check FlowMatching solver tolerance/step refinement. For direct FlowMatching density evaluations in these small target dimensions, start with exact Jacobian traces where supported. MeanFlow's default uses map Jacobian log-determinants; validate its round trips and sampling/density consistency before using its density as an importance denominator.

The existing audit fits a Student-t proposal to neural draws and mixes it with the prior. Because its proposal density is known, it works for every family without requiring the network density. Compare direct neural-density proposals separately; a fitted Student-t can lose multimodal structure. [Current correction](../src/sbi4finance/calibration/correction.py)

## Reference posteriors

### Heston

Create development panels first. After selection is frozen, create a 50-case synthetic reference library plus SPY, with refined Fourier pricing and four dispersed chains per case. Starts must not all come from the neural candidate.

Proposed reference requirements are rank-normalized split R-hat at most 1.01, bulk/tail ESS at least 400 for parameters and important products, and stability under price refinement. Record Monte Carlo uncertainty in the reference summaries. A nonconverged chain is inconclusive, not a neural-method failure or a successful reference.

### Rough Bergomi

Use a reference ladder:

1. Train a separate forward price emulator on clean training labels. Validate it globally and in development posterior regions using independent, higher-precision pricing. Compare emulator-MCMC on about 50 final panels. Shared bank errors mean agreement is a cross-check, not ground truth.
2. Run independent Monte Carlo audits on 50 final panels; extend to 200 only after inspecting reliability and cost under a frozen expansion rule. Report ESS, largest weight, Pareto k-hat, unpriced proposals and approximation sensitivity. Weight concentration alone does not establish conditional accuracy.
3. Pilot direct Monte Carlo-based MCMC on one declared case. Measure likelihood variability, mixing, ESS and path/grid dependence. If feasible, extend to 3–5 fixed anchor cases including typical, weak-identification and market examples. This is reference research until a direct-MCMC agreement claim is included in the release; that claim requires completed, converged anchors.

Keep the likelihood targets distinct:

| Target | Interpretation |
| --- | --- |
| Gaussian noise around ideal expected prices | Requires control of numerical price and likelihood approximation. |
| Gaussian noise around prices with a fixed simulation seed | A deterministic numerical approximation; different seed targets must not be pooled as chains of one posterior. |
| The finite simulator, averaged over its pricing randomness | A legitimate numerical generative model that can match bank generation, including its adaptive pricing rules. |

For the last target, with `y = P_hat(theta, U) + noise`, `N(y; P_hat(theta, U), Sigma)` is nonnegative and unbiased for that simulator-marginal likelihood. A valid pseudo-marginal chain retains its current likelihood/randomness on rejection. This does not make it exact for ideal expected prices. [Andrieu and Roberts](https://arxiv.org/abs/0903.5480)

The current covariance-subtraction estimator `N(y; P_hat, V - S)` has the claimed ideal-price expectation under Gaussian pricing error with known covariance. Estimated covariance and finite Monte Carlo errors do not establish general unbiasedness. Label this approximation and test it; do not call the rough audit an unconditional exact reference.

## Metrics and decisions

Register tolerances before final evaluation. For product comparisons define `W = q95 - q05` of the reference posterior product price. Register a positive absolute tolerance `epsilon` for each product in its own units before production; deterministic controls such as a variance swap with a supplied curve need that absolute rule. Define product tolerance `T = max(0.10 * W, epsilon)` and require numerical/reference uncertainty at most `T / 2`. Refinement tests use at most `T / 4`. The following are proposed starting rules, preserving the project's historical conditional tolerances where applicable:

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

For thin-chain coverage, retain the existing five-method comparison, wrong-extrapolation arms and noise sensitivity at 0.5/1/2. A prior-predictive success under the same prior is expected in a correctly specified Bayesian experiment; it does not establish robustness when the unquoted curve law is wrong. EWG-conditioned truths provide a local sensitivity study, not broad market validation.

Measure cold load/compilation, warm draws, density evaluation when used, audits, products, memory and total training cost. Price-emulator MCMC and classical-proposal importance sampling are required competitors. The bank/training cost is amortized over explicitly compatible tasks; a changed prior/design/noise law may require new validation or training.

## Submission and release

Use immutable Oscar checkouts and actual account/partition availability. CPU arrays generate price surfaces and reference work; GPU arrays index task/encoder/family/seed. Record per-job resources and avoid accidental multiplication of array workers and library threads. Enforce memory and concurrency limits after the pilot.

Submit only the agreed phase allocation. The existing 930 CPU-hour estimate includes the roughly 360-hour bank and the baseline studies; it excludes this wider architecture campaign and direct reference anchors. Re-estimate from Oscar throughput, retaining CPU work and elapsed time as separate quantities.

Update environment/API/test commands in [CLUSTER-RUNS.md](CLUSTER-RUNS.md) before use. The existing scripts are starting points, not validated BayesFlow 2.1 commands. Refresh the [status](../docs/course/status.md), [claims](CLAIMS.md) and accepted-artifact manifest together after results are reviewed.

Release one selected checkpoint per retained task, the complete comparison report, scoped reference plots, failures, participant exercises and cached fallbacks. A broad architecture benchmark can finish after the tutorial release; one reliable selected configuration is enough for the live workflow. If EWG reserve-coverage or full rough-accuracy claims are retained, their corresponding studies remain required.
