# Parameter inference from Heston and rough Bergomi paths

**30 October 2026. Status: overnight implementation and bounded pilot authorized; submission receipt is recorded below when available.**

Build a counterpart to option-surface calibration: infer fixed stochastic-volatility parameters from discrete observed returns, and display how their posterior changes as the observed history grows. Keep simulated volatility hidden from the estimator. Start with synthetic physical-measure models; market-data inference and time-varying parameters are later tasks.

After the user requested more training data, the initial allocation is 100,000 independent training paths per task, with sharded generation, float32 neural inputs after physical scaling, a 12 GiB conservative host-memory guard and peak-RSS accounting. The first deliverable is three reproducible, bounded development fits with diagnostics at several history lengths. Architecture sweeps and tutorial publication follow only when this pilot establishes a useful observation regime and reliable inference.

## What the existing proof of concept teaches

The reference repository is [AlexanderFengler/sbi_fin_pocs](https://github.com/AlexanderFengler/sbi_fin_pocs), inspected at commit `89dd2afe94988d998864d801580c4d4d35625cbc`. Its latest GitHub `main` was cloned into the local workspace on 30 October. No reference code or historical results were changed.

Its [Heston experiment](https://github.com/AlexanderFengler/sbi_fin_pocs/blob/89dd2afe94988d998864d801580c4d4d35625cbc/heston-sbi/heston_sbi.py) already estimates six parameters from 252 daily log returns. It uses a TimeSeriesTransformer with an 18-dimensional summary, FlowMatching, 20,000 training histories, 300 validation histories and 300 new diagnostic histories. Latent variance is excluded from conditions. This is useful evidence for the application design, with no Bergomi implementation or growing-prefix training.

Map its names explicitly: `theta` is our `kappa` (mean-reversion speed); `omega` is long-run variance; `xi` is volatility of variance; `nu0` is initial variance. Use descriptive names in the new task manifests and figures.

The [recorded metrics](https://github.com/AlexanderFengler/sbi_fin_pocs/blob/89dd2afe94988d998864d801580c4d4d35625cbc/heston-sbi/metrics.csv) report contractions of 69.1% for drift, 71.7% for long-run variance, 0% for mean-reversion speed, 47.8% for vol-of-vol, 3.8% for leverage and 98.5% for initial variance. These values have not been independently reproduced: the checkout has a checkpoint and summaries, but no saved observation/truth/draw panel or locked BayesFlow environment.

Do not carry the report's stronger interpretations into the new lesson:

- Zero variance contraction does not imply posterior equality to the prior or structural nonidentifiability. Leverage need not require option observations. Weak finite-history information and approximation error must be distinguished with references.
- The prior couples parameters: vol-of-vol has a Feller-dependent ceiling and initial variance is conditional on long-run variance. Information can transfer through these relationships. The adapter enforces marginal bounds but does not enforce the joint Feller support, so a checkpoint replay must count out-of-support draws.
- Marginal SBC and aggregate coverage do not establish conditional or joint posterior accuracy, or validity on real data. The report's directional SBC-ECDF interpretation is also unreliable.
- The simulator has unseeded per-call RNGs. Its returned latent values are clipped pre-return variances, not signed terminal continuation states. The documented RNG argument is absent.
- The training script executes on import and requires missing local diagnostic helpers. It is a reference for design, not an executable campaign to submit unchanged.

Preserve the old result as historical evidence. A separate replay is worthwhile if its runtime and diagnostic inputs can be reconstructed; reproducing unsupported old percentages is not a prerequisite for a new seeded pilot.

## Reuse within sbi4finance

Develop under `lab/`, preserving tutorial independence. This plan is based on the integrated course/tutorial/lab tree at `76c4967`; at the time of the fetch, GitHub `main` was still `dd001a3`. The new branch must follow the final layout integration before code is merged.

The layout retains `src/sbi4finance/heston.py` and `rough_bergomi.py`, but removes the older Heston path-inference pipeline. Recover only the needed functions and tests from `dd001a3`: `heston_bank.py`, `heston_inference.py`, `heston_reference.py`, `heston_forecast.py`, their generation/training/reference scripts, and their task definitions. Rehouse these in a focused `src/sbi4finance/path_inference/` package rather than restoring every historical workflow.

The old restricted Heston task supplies a valuable control: infer long-run variance and signed terminal auxiliary state with the other parameters fixed. It already has a recurrent encoder, reversible state transform, and leverage-aware particle/quadrature reference. Its old acceptance status does not transfer to the new tasks.

Use the existing locked `core`, `core-gpu` and `core-reference` environments (BayesFlow 2.1.0), and `cluster/core_cpu.sbatch` / `cluster/core_gpu.sbatch`. Do not use archived default/BayesFlow 2.0.14 environments for new fits. Existing option-bank collection and readiness commands are task-specific; reuse their identity and integrity conventions, not an assumption that they accept return paths unchanged.

## First task definitions

These are proposed teaching domains to check in the small numerical/prior-predictive pilot, not domains certified by the POC. Freeze them and their numerical laws before generating the training banks. A changed prior, grid or initialization creates a new task and bank identity.

| Setting | Heston paths | Rough Bergomi paths |
| --- | --- | --- |
| Measure and observations | Synthetic P model; ordered log returns only | Synthetic P model; ordered log returns only |
| Fixed quantities | Drift 0.05/year; initial variance 0.04 | Drift 0.05/year; deterministic variance-scale function `xi0(t)=0.04` |
| Inferred parameters | `kappa`, `long_variance`, `vol_of_vol`, `rho` | `hurst`, `eta`, `rho` |
| Candidate prior | Independent: kappa log-uniform [0.25, 5]/year; long variance log-uniform [0.01, 0.09]; vol-of-vol uniform [0.05, 0.6]; rho uniform [-0.95, -0.05] | Independent: H uniform [0.03, 0.4]; eta uniform [0.2, 2.5]; rho uniform [-0.95, -0.05] |
| Maximum observed history | 1,260 daily returns, five trading years | 1,008 returns, four observations/trading day for one year |
| Training/display prefixes | 63, 126, 252, 504, 1,260 observations | 63, 126, 252, 504, 1,008 observations |
| Elapsed years at prefixes | 0.25, 0.5, 1, 2, 5 | 0.0625, 0.125, 0.25, 0.5, 1 |
| Initialization | Known initial variance; preserve signed numerical state | Start the Volterra history at time zero; no arbitrary-window restart |

The joint rough task additionally infers `short_variance`, `long_variance` and `decay` with `xi0(t) = long_variance + (short_variance - long_variance) exp(-decay*t)`. The two variance levels have independent log-uniform [0.01, 0.09] priors and decay has a log-uniform [0.25, 8]/year prior; H, eta and rho retain the known-curve priors. This is a six-parameter synthetic P task with an unknown mean-variance profile, not an observed Q forward curve. Near-flat profiles can leave decay weakly identified. The known flat curve remains the control.

The Heston prior intentionally imposes no Feller coupling. The process and declared simulator must support the entire domain, including boundary-reaching cases; never discard difficult parameters or redraw a path to improve its appearance. Report how often numerical auxiliary states become negative. Fixing drift and initial variance is an explicit simplification. Releasing Heston initial variance or drift is a later extension; the joint rough task releases its variance-scale profile immediately.

Write the rough P law explicitly: log-price drift `(mu - V/2)dt`, correlated price and variance drivers, and `V_t = xi0(t) exp(eta Y_t - eta^2 t^(2H)/2)` for the declared normalized continuous Volterra driver. The existing hybrid discretization instead compensates with `eta^2 Var(Y_hybrid,t)/2`, preserving the mean variance under its own Gaussian law. Declare and preserve that numerical compensator in the frozen task; substituting the continuum compensator changes the discrete model. Existing simulator APIs are explicitly Q-only. Reuse their numerical kernels under a named and tested P task; setting a pricing rate equal to a drift is not sufficient provenance.

## Numerical and data contract

1. **Small deterministic panel first.** Benchmark 256 independent trajectories per model, including predeclared prior-edge cases. Measure throughput, memory, finite outputs, prior-predictive return scales, truncation behavior and sensitivity of observation-derived summaries. Retain failed IDs.
2. **Freeze the numerical target.** The independent 32-case panel found material one-step effects and many exact daily drift returns. The primary Heston bank therefore uses eight internal steps per daily observation; the original one-step law remains a separate restricted control. The coupled 1/4/8 comparison remains diagnostic rather than a continuous-model accuracy certificate. A substepped target needs matching latent transitions in its reference; do not reuse a one-step likelihood. Rough checks compare working and refined hybrid grids, use the independent covariance reference on grids of at most 128 steps, and retain the joint price-driver correlation. Exact discrete-model correctness and approximation to the continuous model are separate claims.
3. **Avoid a silent numerical task change.** If the small panel suggests material grid effects, specify and cost the refined law before the 100k generation. Later paired fine-grid observations are a transfer/robustness panel, not draws from the original calibration task.
4. **Stable trajectory identities.** Derive parameter and innovation streams from task identity, split-qualified trajectory ID and root seed. Proposed new roots: 2026103001 for Heston and 2026103002 for known-curve rough and 2026103003 for joint-curve rough. Separate training-prefix and evaluation randomness. Shard boundaries, worker count and restart order must not change an existing row.
5. **Split before extracting prefixes.** Per model, propose 100,000 training, 2,000 validation and 2,000 reserved test maximum-length trajectories. All prefixes of a trajectory stay in its split. Prefixes are repeated views, not additional independent simulations. Keep the test split closed through development.
6. **Persist the sufficient provenance.** Store raw returns, physical parameters, the time grid, IDs, task/source/runtime hashes and checksums. Latent paths or selected latent states belong to diagnostic roles and must never enter parameter conditions. Include initialization, RNG definition and exact discretization in the task fingerprint.
7. **Frequency control.** Aggregate each group of four rough intraday returns into one daily return on a paired panel. This keeps horizon and trajectory fixed when assessing observation frequency; the five-year Heston experiment separately assesses duration.

## Training and sequential application

Use one modest causal baseline per task: GRU64, summary dimension 32, and a six-layer spline CouplingFlow with eight bins and 64-by-64 subnetworks. This is a proposed baseline, not a claim that it outperforms the POC's transformer. Keep a transformer/FlowMatching comparison for a measured representation or posterior-approximation problem.

The architecture supports every integer history length from 1 through the task maximum from the start. Draw random prefix lengths independently for training examples, pad to the fixed maximum and pass an explicit boolean `summary_mask` into the causal GRU. Padding and later observations cannot influence the summary. Supply observed times/intervals and prefix duration as known design; exclude padded values from normalization. Validate arbitrary odd lengths as well as the five diagnostic landmarks. The maximum horizon remains part of the trained domain.

Train for at most 60 epochs per initial fit, batch size 128, with a 50-minute GPU wall cap and a 45-minute soft stop at epoch boundaries. Randomize complete trajectories within the training split. Record optimizer steps and exposure by length. Select the checkpoint using the fixed validation prefix policy, retaining final and selected epochs. A soft stop saves a usable partial development fit with an explicit stop reason.

Use fixed, training-only global normalization; preserve return magnitude across histories. Apply physical-support parameter transforms before neural dtype casting, and check inverse transforms and sampling support. Never standardize each path to unit variance. Save optimizer/scheduler/RNG state if claiming exact resume; otherwise restart a new fit rather than calling weight loading a continuation.

At any observed history length in the trained range evaluate `q(parameters | all returns observed so far, time design)` using the original prior. Five landmarks organize the main accuracy report; arbitrary held-out lengths and prefix-invariance tests check the variable-length implementation. Finite diagnostics do not certify every conditional posterior or extrapolation beyond the maximum horizon. A causal hidden-state cache is an optional engineering optimization after parity with full-prefix inference is demonstrated. Do not multiply overlapping prefix posteriors or feed the previous posterior into an unchanged estimator. Posterior movement does not make the parameters time-varying, and realized intervals need not narrow monotonically.

First infer parameters only. The old restricted theta/state task remains the filtering control. A joint current-volatility target and posterior predictive forecasts can be added after the parameter baseline is checked. For rough Bergomi, current variance alone is not a sufficient continuation state.

## Independent references and validation

Restore the restricted Heston particle/quadrature control first. Check the observation law, leverage dependence and signed numerical state exactly. Refine particle counts and quadrature and repeat independent seeds before treating it as a reference.

For the four-parameter Heston task, begin with two case/horizon targets as a reference feasibility screen. Prefer a PyMC latent-path formulation where the selected numerical law has a valid differentiable density. Full-truncation zero-variance return atoms must not be replaced by arbitrary positive noise in that density. If this prevents a natural PyMC formulation, use a documented particle-MCMC exception under the [MCMC policy](mcmc-policy.md), extending the existing leverage-aware filter to the parameter vector. Retain the accepted likelihood estimate and its random state on rejection; a fresh noisy log likelihood passed to ordinary MCMC changes the method.

The first reference jobs measure likelihood noise, mixing, runtime and feasibility. They do not promise converged posteriors inside a small wall cap. For conditional accuracy comparisons require four chains, rank R-hat at most 1.01, bulk/tail ESS at least 400, and checked particle/grid precision. An incomplete or unstable reference yields an inconclusive comparison. Cost a larger panel of 12 prespecified cases at one- and five-year prefixes only after this screen.

Rough Bergomi has no conventional finite-dimensional HMM representation in its unapproximated form. Scope a bounded Gaussian-driver latent-path reference on short histories, or an explicitly validated Markovian approximation. Do not claim conditional accuracy for long histories from short-grid agreement alone. Reference feasibility and observation-design checks can progress alongside the Heston fit.

Development evaluation uses independent validation trajectories at **every supported length**, reporting:

- Marginal SBC/rank ECDFs and 50/80/90/95% coverage with sampling uncertainty; count independent trajectories, not prefixes, as cases. State any multiple-comparison screen.
- Physical-coordinate posterior/prior variance and interval-width ratios, without clipping expansions to zero. Use a separate large seeded prior sample; include recovery correlations and posterior-mean squared error against the prior-mean predictor.
- Prior-only and wrong-case pairing controls, using the same time design. A calibrated prior-only estimator is not a useful inference result.
- Prior-support violations, nonfinite outputs, joint dependence/derived-quantity checks and conditional comparisons where references pass. Do not reinterpret broad posteriors as structural nonidentifiability without further evidence.
- A small prior-predictive/representation screen for rough H, eta and rho before a full fit. Weak signal triggers review of the observation regime and numerical target, not an automatic larger encoder.

After development checks, freeze the model and evaluation recipe before opening the 2,000-trajectory test split. Any later changes use new development evidence rather than relabeling inspected tests.

## Oscar implementation order and first allocation

The following implemented entry points drive the new workstream:

- `scripts/path_bank.py`: initialize, benchmark, generate/check shards and freeze a snapshot for either task.
- `scripts/check_path_cluster.py`: all-length forward/backward checks, actual GPU assertion, transforms and checkpoint reload.
- `scripts/train_path_posterior.py` and `evaluate_path_posterior.py`: explicit task/bank/checkpoint identities and length-stratified outputs.
- `scripts/path_reference.py`: restricted-control checks and bounded reference feasibility runs, recording the actual sampler.

Reuse generic Slurm wrappers and a fresh campaign configuration. Do not edit frozen historical launch controllers. Submit from an immutable checkout of the new code, using an external scratch run root and explicit Pixi path. Existing installed environments may be reused only after checking byte-identical lock/runtime identity. Do not generate banks or train networks on a login node.

The user authorized the bounded overnight tranche on 30 October. Keep the previously proposed **24 allocated CPU-hour and 3 GPU-hour ceiling**, including training CPUs and failures. With the added joint task, the reservation below totals **19.8334 CPU-hours and 2.75 GPU-hours**, including a five-minute CPU report job. Unused time is not an automatic retry allowance.

| Stage | Requested allocations | Maximum CPU-h / GPU-h | Dependency and result |
| --- | --- | --- | --- |
| P0 CPU smoke | 1 job, 2 CPUs, 8 GiB, 15 min | 0.5 / 0 | New path tests and deterministic contracts |
| P0 GPU smoke | 1 job, 1 GPU, 4 CPUs, 24 GiB, 15 min | 1 / 0.25 | Masked recurrent gradients, variable-length invariance and reload |
| P1 numerical pilot | 1 job, 2 CPUs, 8 GiB, 30 min | 1 / 0 | P0 CPU; record Heston and rough grid behavior |
| P1 throughput | 1/model, 1 CPU, 8 GiB, 15 min | 0.75 / 0 | P0 CPU; benchmark all three tasks |
| P2 banks | 1/model, 1 CPU, 8 GiB, 30 min | 1.5 / 0 | Its benchmark and numerical execution; 104k paths/model |
| P3 fits | 1/model, 1 GPU, 4 CPUs, 24 GiB, 50 min | 10 / 2.5 | P0 GPU plus corresponding checked bank; soft stop before wall cap |
| P4 restricted reference control | 1 job, 2 CPUs, 8 GiB, 60 min | 2 / 0 | Original restricted task, independent of new four-parameter observations |
| P5 evaluation | 1/model, 2 CPUs, 8 GiB, 30 min | 3 / 0 | Corresponding fit; validation only, arbitrary lengths plus landmarks |
| Final report | 1 CPU, 8 GiB, 5 min | 0.0834 / 0 | After all jobs complete or cancel; summarize artifacts and accounting |

Heston and rough lanes can progress independently once their own task checks pass. References can run alongside training. A failed dependency blocks its dependent submissions; it must not consume the remainder through repeated automatic resubmission. Maintain a ledger of requested exposure, live jobs and completed `sacct` CPU/GPU accounting before each submission. If measured generation or reference costs exceed a reservation, revise the tranche before scheduling it.

The P4 restricted control keeps its original task identity, with four predeclared histories for bounded particle/quadrature refinement. It must not be evaluated as the reference for the new four-parameter observations. A multidimensional-reference feasibility screen follows after measured restricted-control cost; it is not included in the first submissions.

Use live association discovery to select account/partition/QOS; the templates' defaults are not proof of entitlement. Show exact commands and resource overrides before submitting. The checked environment is `core` for CPU generation/evaluation, `core-gpu` for training and `core-reference` where PyMC is actually used.

Read-only discovery on 30 October confirmed CPU account `carney-mjfrank-condo2`, partition `batch`, QOS `carney-condo2` (per-user cap 208 cores), and the corresponding `gpu-he` / `carney-gcondo2` lane (160 cores, 75 GPUs). These are allocation caps, not available idle capacity. The connected user's queue was empty at that observation. Recheck before submission; initial noninteractive SSH failed until the user renewed the Duo login. The verified existing runtime has lock SHA-256 `3e33b82d45dfdd696c5b12b22bca0943a5fca09804860fab13e3fc9126c60aae`; the source-tree lock is `c814f74feb11cb8712a1f722243beb4b558f0edba1f31865248924709c22bc42`. All resolved numerical and CUDA packages match; six peripheral packages differ. Reuse the existing interpreter with explicit new-source `PYTHONPATH`, and record the actual runtime lock separately. Do not install into or mutate the historical environment.

## Completion and subsequent scale

The tranche is complete when it delivers three checked task definitions, reproducible banks and reloadable pilot fits, a validation report by horizon, a numerical/throughput record, reference feasibility outcomes and actual accounting. An unresolved scientific check remains visible and can justify a narrower teaching claim; it does not trigger an unbounded certification campaign.

The first figures should reveal one return path progressively, with parameter posterior bands and joint snapshots at the diagnostic landmarks and selected intermediate times. Include representative weak cases, not only strong recovery. A separate ensemble figure reports calibration/informativeness across independent trajectories. Later filtered-volatility plots must distinguish filtering from retrospective smoothing, and forecasts must preserve parameter/state/history dependence.

Scale to a larger bank or a second architecture only when the pilot identifies a specific learning limitation. Expand the reference panel separately after measured cost and convergence checks. Publish small cached teaching cases, models and regeneration commands through the existing lab-to-tutorial tools only after the intended claims are supported. Keep large banks, full draws and model payloads outside Git.

## Overnight submission and launch correction

The first immutable source was `8a3f896`. CPU job 8109953 exposed a launcher working-directory error before generation or training: `SLURM_SUBMIT_DIR` referred to the remote login directory. Dependent jobs canceled and no banks were generated. The corrected launcher both submits from the source directory and exports it explicitly; dedicated mocked-submission tests enforce this. The initial condo GPU job was canceled while blocked by the shared group GPU quota. The replacement uses the discovered `default/gpu-debug` lane for readiness and `default/gpu` with QOS `norm-gpu` for training. Preserve the v1 receipt and logs.
