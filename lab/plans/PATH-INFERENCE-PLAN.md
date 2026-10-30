# Parameter inference from Heston and rough Bergomi paths

**30 October 2026 completed follow-up status:** all baseline work and all thirteen corrected follow-up jobs completed successfully; the Oscar queue is empty. The short-history refinements improved their targeted diagnostics, but both failed the full prespecified development screen, so the original baselines remain retained. The independent Heston reference still misses convergence gates. Reserved tests remain closed; no further runs were submitted during this audit.

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

The corrected v2 graph uses source `ea4d400` and jobs 8110042 through 8110058. Oscar CPU readiness passed all 30 tests; the full-production-shape GPU check and 100k banks then progressed. An additional eight-chain Heston PMMH feasibility screen uses the same eight-substep law, two validation prefixes (252 and 1,260 days) and four chains each, with 512 and 2,048 particles respectively. The supplemental launcher reconciles actual completed spending, outstanding reservations and 240 seconds of cleanup per job before admission to the same 24 CPU-h / 3 GPU-h ceiling. Rejected PMMH updates retain the accepted particle likelihood estimate and seed. The output remains a feasibility screen until cross-chain mixing and likelihood precision pass.

The active main graph is jobs **8110042–8110058**. Supplemental PMMH software/chain jobs are **8110221–8110229**. At admission of those nine CPU jobs, the reconciled conservative exposure, including completed launch failures, remaining wall reservations and cleanup allowances, was **19.3003 CPU-hours / 2.8158 GPU-hours**, below the 24 / 3 ceiling. Receipts, source archives and GPU readiness evidence are copied to the workspace external artifact directory `local-artifacts/path-inference-20261030/`. Banks have 100,000 training, 2,000 validation and 2,000 reserved test trajectories each. Validation and per-chain reference outputs are separate development evidence; the reserved test posteriors remain unevaluated.


## Afternoon result audit on 30 October

All banks retain 100k training / 2k validation / 2k reserved test paths. Rough-known completed 52 epochs (selected 47), rough-joint 54 (selected 53); both stopped deliberately at the 45-minute training budget and passed exact native reload checks. Slurm peak host RSS stayed below 4 GB for each rough fit.

On the 256-history validation panel, one-year posterior-mean MSE relative to the prior-mean predictor is H/eta/rho = 0.349/0.137/0.203 for known curve and 0.358/0.141/0.271 for joint curve. Joint short/long/decay ratios are 0.265/0.840/0.939: long level and especially decay remain much less informative. One-year nominal 90% marginal coverage ranges from 88.7% to 93.8% across the two tasks. These are development results on reused checkpoint-selection data, not conditional-reference acceptance.

The joint one-return control needs correction: nominal 90% coverage is 82.8% for short variance and 85.5% for eta. Eta should retain its independent prior at this horizon. Arbitrary-length implementation and mask tests passed, but calibration across all history lengths is not established.

Heston job 8110048 was held overnight by Slurm after a user-environment retrieval failure, before any estimator was written. It was released during this audit and verified running on gpu3103 with initial epochs saved. Its original evaluation 8110049 and report 8110058 dependencies remain intact.

The one-year PMMH screen completed four chains with 1,000 retained draws but fails reference gates: rank R-hat 1.031–1.104, bulk ESS 29–135 and 5%/95% tail ESS 30–205. The five-year jobs all timed out at 20 minutes; each retained checkpoint has 500 tuning + 300 retained iterations. Their rank R-hat 1.093–1.203 and bulk ESS 14–45 also fail. Preserve the partial traces. Improve proposal preconditioning/mixing and measure long-history likelihood noise before allocating more reference sampling; completing the original short chain alone would not certify convergence.

The external status bundle contains verified summaries, saved diagnostic draws, `status-summary.json`, `metrics-summary.json` and `status-progress.png`. The reserved test posteriors remain unevaluated. The immediate work is to finish Heston, address joint short-prefix calibration with its exact one-return control, and improve the independent references.


## Fixed follow-up runs defined on 30 October

The recovered Heston baseline completed all 60 epochs and selected epoch 50. Both rough baselines remain at their recorded epochs 47 and 53. The original models and all 100k banks remain frozen; this follow-up changes neither the prior nor the observation law.

| Candidate | Parent weights | Additional training | Prefix policy | Optimizer and endpoint |
| --- | --- | --- | --- | --- |
| Heston | Selected baseline epoch 50 | At most 8 epochs on the existing 100k paths; 420-second soft stop and 10-minute Slurm cap | 50% uniform integer lengths, 50% floor of log-uniform lengths, covering every length 1–1,260 | Fresh Adam at 0.0001, batch 128; last completed epoch |
| Joint rough | Selected baseline epoch 53 | Same bounds on the existing 100k paths | Same mixture, covering every length 1–1,008 | Same reset and fixed endpoint |

These are weights-only warm starts with new optimizer and prefix RNG states, not exact resumes. Task, bank and parent-model identities are checked. Known-curve rough is retained as a control without another fit. This is one combined optimization intervention per affected task, not an ablation isolating prefix weighting from additional updates or the new learning rate. Final model bytes and seeded reload are checked; minimum validation loss is informational rather than the candidate-selection rule.

Both baseline and candidate are evaluated with 512 posterior draws on the same 512 validation trajectories at lengths 1, 2, 4, 8, 17, 32, 63, 126, 252, 504, maximum-minus-seven and maximum. Rows 0–255 are the reused checkpoint-development cohort. Rows 256–511 are the predeclared primary validation cohort for the comparison; they remain validation data, not the reserved test set. Per-case means and interval bounds are saved with identities/checksums so all comparisons can pair trajectories.

The development screens are fixed before submission:

- At the five original landmarks, each target's upper pointwise 95% paired trajectory-bootstrap bound on the MSE increase, normalized by prior-mean risk, must be at most `max(0.01, 0.05 * baseline normalized MSE)`. Bootstrap complete trajectories; the horizons are repeated views of those trajectories.
- At those landmarks, the candidate's absolute 90% coverage error may not increase by more than 0.02 for any target; report this as a descriptive pointwise guard, not a simultaneous calibration certificate.
- For joint rough at one return, use independent scalar log-coordinate quadrature for short variance and exact independent priors for the other five targets. With 2,048 neural draws per case, require mean RMSE at most 0.10 reference SD and mean absolute error of exact probability mass inside neural 90% intervals at most 0.04 for every target. Also report casewise tails and quadrature refinement. Bind each exact check to its own model/task hashes.
- Heston uses eight internal steps, so its first daily return is parameter-dependent. Apply a simulation-based short-history screen rather than an exact-prior assertion: mean absolute coverage error across lengths 1–32 must improve, and each target's length-one 90% coverage must lie in [0.85, 0.95]. This is a development screen, not conditional posterior certification.
- Finite-output, prior-support and native reload checks must pass. Failed or inconclusive screens retain the original baseline. No report automatically publishes a model or opens reserved tests.

The reference work proceeds independently. Freeze a one-year proposal covariance from the old logit traces with 10% diagonal shrinkage and a relative eigenvalue floor. Start four new phases from their respective accepted parameter/particle-likelihood states, preserving the accepted likelihood estimate and seed, with fresh phase RNGs. Each has 500 tuning and 5,000 retained iterations, 512 particles and a 15-minute CPU cap. The covariance is fixed; scalar adaptation ends after tuning. Check rank R-hat at most 1.01 and bulk and 5%/95% tail ESS at least 400. A single Heston case comparison against both neural checkpoints runs only if those convergence gates pass; it remains a development comparison.

For five years, first measure likelihood noise and timing at the four saved partial-chain states with 512/1,024/2,048 particles and eight independent repeats. This job starts no new chains. The measurement will determine whether the next expense should address particle noise, proposal geometry or longer sampling.

The follow-up launcher checks actual old spending, outstanding requests, earlier failures and 240 seconds of cleanup per allocated job against the existing 24 CPU-hour / 3 GPU-hour ceiling. At the measured old allocation totals, the 20-job follow-up reserves approximately **21.44 cumulative CPU-hours and 2.99083 cumulative GPU-hours**, including cleanup. Its two GPU fits are the only new GPU allocations. A new immutable source/run directory, dependency gates, partial-submission receipt and final report preserve all prior artifacts. No automatic retry or architecture sweep is authorized by this tranche.


The follow-up was submitted as **8142698–8142717**, from immutable source commit `df52aadcb84e7010a27c8216a0938017fc2ba063` at `/oscar/scratch/afengler/sbi4finance-path-followup-20261030-v1/source/lab`. The separate run root is `/oscar/scratch/afengler/sbi4finance-path-followup-20261030-v1/run`. The source archive SHA-256 is `f012bfde822f8c796817b8a0f8d37173f54aeefa90e6ec970d3a21ed3309cf85`; its receipt and submission receipt are mirrored in `local-artifacts/path-inference-20261030/followup-20261030/` outside Git.

The training jobs are **8142706 (Heston)** and **8142710 (joint rough)**. Software gate 8142698 precedes both. Proposal preparation 8142699 precedes the four one-year reference chains 8142700–8142703; five-year likelihood-noise measurements use 8142704. Paired evaluations and the exact joint one-return checks feed comparisons 8142708 and 8142714. Reference reduction 8142715 gates the conditional Heston comparison 8142716; final report 8142717 records all artifacts even after failed dependencies. Admission reserved **21.43778 cumulative CPU-hours / 2.99083 cumulative GPU-hours**, including the conservative cleanup allowance. Submission is not evidence that a scientific screen has passed.


The first follow-up software gate passed all **56 tests**, but the new 11-digit neural seeds exceeded Keras's uint32 range. Fits 8142706/8142710 exited before optimizer updates, after 48/34 allocated seconds; their downstream neural jobs canceled. Baseline evaluation and the exact baseline check failed for the same seed-domain reason. The existing 100k banks and baseline checkpoints were unchanged. Fix backend initialization/sampling with an explicit recorded nonnegative-integer modulo-2^32 mapping, preserving every previously valid uint32 seed; physical simulator and prefix-generator seeds retain their full values. Regression tests use the actual large launch seed, including warm start and repeatable sampling. Evaluation now also checks the exact bank-manifest hash.

A manually reviewed repair reuses the original proposal, four PMMH chains, completed noise panel and reference reducer. It adds no replacement reference sampling. To include the two failed GPU allocations and their cleanup allowances within the original ceiling, reduce each corrected GPU fit to a **five-minute hard cap / 180-second soft stop**, retaining the maximum of eight additional epochs and final-epoch rule. Reconcile all original follow-up jobs as well as the earlier campaign before admitting the 13-job repair. No historical receipt or immutable source is overwritten. The pending original reducer 8142715 was changed from `afterok` to `afterany` for its four chains, so partial/failing references yield an explicit failed-gate report; this resource-neutral override is recorded in its submission receipt.

Five-year likelihood-noise measurements at the four saved states completed with no collapsed estimates in eight repeats per particle count. Log-likelihood SD ranges were 1.57–5.66 (512 particles), 1.01–2.55 (1,024), and 0.89–1.29 (2,048); median filter time was about 0.33/0.65/1.31 seconds respectively. These small pointwise panels suggest avoiding a particle-count reduction for the next five-year reference. They do not establish global precision or a converged reference; geometry and required chain length still determine the next reference allocation.


The corrected run uses immutable source **`43a332b0c6a248d6f7b1231fac9a32ee99593a75`** at `/oscar/scratch/afengler/sbi4finance-path-followup-20261030-v2/source/lab`, archive SHA-256 `46c98d7c24ca22b4132bbc1b81841e2e1ab46e14977a23f6db48ccc55140dd5b`. All **61 path checks passed locally and on Oscar**. Its complete graph is jobs **8144165–8144175, 8144235 and 8144236**. Heston fit **8144167** and joint-rough fit **8144171** have started on GPUs; both restored their exact frozen parent weights and entered training. Initial admission reserved **21.2175 cumulative CPU-hours / 2.98028 cumulative GPU-hours**, including failures, all outstanding jobs and cleanup. At completion of the last two submissions, the refreshed conservative total was **20.81083 / 2.98028**. These are reservation/accounting figures, not measured neural execution times.

Slurm had already purged the completed original reference/noise IDs from its live controller when the new conditional/report jobs were submitted. The first eleven corrected jobs remained valid. After verifying the external jobs' `COMPLETED` / `0:0` accounting and their output files, submit only the two missing jobs with those satisfied dependencies omitted. The submission receipt preserves the initial error and manual-completion commands/provenance; it now records all thirteen jobs as submitted. The local launcher preflights external dependencies through accounting before submission and refuses to resubmit an existing partial receipt. Its fourteen targeted tests pass. Frozen numerical source files and existing jobs are unchanged.

The four new one-year PMMH phases each completed **500 tuning + 5,000 retained iterations**. Preconditioning improved mixing, but the reference remains unaccepted:

| Parameter | Rank R-hat | Bulk ESS | 5%/95% tail ESS | Fixed convergence gate |
| --- | ---: | ---: | ---: | --- |
| kappa | 1.0114 | 410 | 514 | Fail R-hat |
| long variance | 1.0082 | 541 | 575 | Pass |
| vol of vol | 1.0061 | 592 | 619 | Pass |
| rho | 1.0070 | 364 | 224 | Fail ESS |

The conditional comparison must therefore remain inconclusive. For the next separately costed reference tranche, inspect the saved traces and likelihood-noise panels, then favor additional one-year retained sampling from validated extended states to address kappa agreement and rho tail exploration. Keep the numerical target, particle count and convergence thresholds fixed unless an explicit diagnostic motivates a change. Do not expand to multiple reference histories or start more five-year chains in this tranche. After the two current neural refinements finish, apply the predeclared primary-cohort and exact one-return marginal screens before defining larger training runs. Marginal gates do not certify the exact one-return product copula or general conditional accuracy.

The completed reference/noise evidence, raw chains and original startup logs have been copied locally and checksum-verified as `local-artifacts/path-inference-20261030/followup-20261030/reference-audit.tar.gz` (SHA-256 `3177750a5f32b32123da77376471f7c14a2fdea4fe375669ab35b9d38ed14871`). The corrected source, test log and receipts live under its `repair-v2/` sibling. The reserved 2,000-history test splits remain closed; baselines remain frozen.


Both corrected fits have now finished their intended soft stops. Heston completed **3 extra epochs / 2,346 fresh-optimizer steps** in 211.23 trainer seconds; joint rough completed **4 extra epochs / 3,128 steps** in 214.87 trainer seconds. Each final endpoint equals its selected-model hash and passed exact same-seed reload. Heston model SHA-256 is `b4378d8ed55b96bffa8babd7a8fbf98d5b8d525d81bd0495144550ce273131ba`; joint rough is `c46597c03b4569cdf4a0e5eb26c9cff859b63b4dbc089cf91eb589b1f18b5793`. Paired candidate evaluations, the candidate joint one-return reference check, guarded conditional reporting and the final report have all completed. Execution and checkpoint checks passed; the statistical decisions are recorded below.


## Completed follow-up evaluation

All thirteen corrected jobs ended `COMPLETED` with exit code `0:0`. On the prespecified primary cohort of validation rows 256–511 (256 independent histories, with repeated prefixes), both candidates have `retain_original_baseline_needs_further_work` status. Their finite-output, prior-support and native-reload checks passed. No model was promoted and the 2,000-history reserved test split for each task remains closed. Known-curve rough retains its original baseline as the control.

| Primary validation diagnostic | Baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| Heston mean absolute 90% coverage error across lengths 1, 2, 4, 8, 17, 32 and all four targets | 2.29 percentage points | 1.71 percentage points | Short-history screen passes; length-one coverage 90.2–92.6% |
| Joint rough one-return eta posterior-mean RMSE in exact-reference SD units | 0.182 | 0.064 | Passes 0.10 mean-error limit |
| Joint rough one-return short-variance posterior-mean RMSE in exact-reference SD units | 0.388 | 0.242 | Still fails 0.10 mean-error limit |
| Joint rough one-return short-variance mean absolute exact probability-mass error in neural 90% intervals | 6.95 percentage points | 2.69 percentage points | Now passes 4-percentage-point limit |

The other five joint one-return mean-error checks and all six interval-mass checks pass. These are marginal controls, not joint-copula certification or evidence of accuracy at every supported history length.

Heston still fails to establish the prespecified longer-history nondegradation bound for vol-of-vol at 252 and 504 days. The upper pointwise bootstrap limits were 0.016062 versus a 0.015932 margin and 0.011050 versus a 0.010000 margin, respectively. At 1,260 days, kappa coverage moves from 90.23% to 92.58%; the increase in absolute nominal-coverage error exceeds the allowed 2 percentage points. Heston's short-history pass therefore does not justify replacement of its baseline.

Joint rough fails longer-history nondegradation bounds for short variance at 252/504/1,008 observations, rho at 504/1,008, and H at 1,008. All its longer-history coverage guards pass. The point MSE changes are smaller than their prespecified margins; the upper uncertainty bounds prevent establishing nondegradation. These failed development screens must not be described as proof of material degradation. Several checks reuse the same histories across lengths, and the bootstrap intervals are pointwise rather than a simultaneous accuracy certificate.

The one-year PMMH convergence result remains unchanged: kappa rank R-hat 1.0114 exceeds 1.01 and rho bulk/tail ESS 364/224 falls below 400. The guarded conditional Heston report is explicitly `inconclusive_reference`, and neither neural model was sampled in that comparison. No threshold was relaxed.

At the final 30 October 22:00 UTC reconciliation, cumulative usage plus the declared 240-second cleanup allowance per allocated job was **19.11028 CPU-hours / 2.95417 GPU-hours**, below the original 24 / 3 ceiling, including startup failures. The two successful refinement jobs used 252/254 allocated GPU seconds. Their peak host RSS was about **4.45 / 4.00 GiB** (Heston / joint rough); this is host-memory accounting, not device-memory peak. There are no queued or running jobs in this campaign.

Next work should be separately specified and costed: a longer fixed training comparison that balances short-prefix calibration against longer-history performance, with particular attention to the exact one-return short-variance bias; and additional one-year reference sampling after inspecting kappa/rho traces and likelihood-noise diagnostics. The current bank size, physical simulator law and all-length architecture can be retained. This tranche does not show that bank size or model capacity is the limiting factor. Do not open reserved tests, publish the candidates, expand the reference case panel or start another five-year chain until the respective development/reference gates are met.

The final report is archived locally as `local-artifacts/path-inference-20261030/followup-20261030/repair-v2/followup-summary.json` and matches Oscar SHA-256 `0136db009d0714764d4919662cd59592071042e0ebd044d17a446f49099cf794`. Its embedded comparisons correspond to Heston comparison SHA-256 `d406e4ea4f2ca9836893b5cc05eff7312ffa3e8f4ff90e2d47e622d7c3541206` and joint-rough comparison SHA-256 `997dfc05cbc67b5a184b8078c37e2735911708eecbed524d3712306e65b2cc5a`.
