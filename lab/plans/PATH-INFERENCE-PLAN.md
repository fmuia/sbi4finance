# Parameter inference from Heston and rough Bergomi paths

**31 October 2026 completed scale-up status:** all 29 unattended jobs completed successfully in 2 hours 7 minutes. Three million-history banks and six full-budget fits are complete, using harmonized daily observations and inferred drift. The larger bank gives selected improvements rather than a universal win; marginal-calibration concerns and independent conditional validation remain. Historical models and reserved tests are preserved.

Build a counterpart to option-surface calibration: infer fixed stochastic-volatility parameters from discrete observed returns, and display how their posterior changes as the observed history grows. Keep simulated volatility hidden from the estimator. Start with synthetic physical-measure models; market-data inference and time-varying parameters are later tasks.

## Teaching design agreed on 31 October

Workshop clarity and harmony across examples govern the scale-up. The primary Heston and Bergomi examples must use the same observable time grid, calendar duration, supported prefix lengths and display checkpoints. Keep the data interface (ordered log returns), physical scaling, summary-network family, inference/training workflow, comparable data counts and optimization budget, and diagnostic/plot presentation consistent. Participants should be able to switch the stochastic model and its named parameters while following the same exercise. Model-specific parameter meanings, priors, latent dynamics and internal numerical discretizations remain explicit; equal observation intervals do not require identical simulation kernels or identical posterior concentration.

Select and validate one common observation design before freezing the next classroom banks. Extend duration in common for both models when teaching the effect of accumulating history. The earlier proposal of ten-year daily Heston against three-year intraday Bergomi is superseded as the primary teaching design. The 30 October banks remain historical pilot evidence and controls, not a harmonized model comparison. Their unequal cadence was a heuristic for parameter recovery, not an established workshop requirement.

Introduce the known-curve Bergomi example alongside Heston through the shared workflow. Joint-curve estimation remains a planned extension, with its added unknowns explained as a separate learning objective. Frequency changes are an optional subsequent lesson: first show the harmonized example, then compare observation schedules on paired histories at fixed duration and underlying numerical law. Form daily returns by summing the matching intraday log returns; keep this extension out of the core exercise until the shared example is established.

The user chose pre-generated training banks for the scale-up. Implement bounded-memory offline input and validated bank-extension contracts; online-generated batches are outside this workstream's current scope. Compare bank sizes at matched optimizer-update counts on the common teaching design. Numerical and prior-predictive checks support that design and must not silently replace it with separately optimized per-model frequencies or horizons. This design update starts no new Oscar jobs.

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

**Current source ownership, 2 November 2026:** edit shared physical-measure task
and simulation code in `tutorial/src/sbi4f_tutorial/trajectories/simulation.py`
and masking/native inference in `trajectories/inference.py`; the Heston physical
kernel lives in `heston_options/simulator.py`. Lab workers import these same
implementations and retain disk-backed banks, production training/evaluation,
references and publication. Keep the complete checkout and use the locked lab
environment. See the [current editing map](../../tutorial/README.md#author-or-change-a-lesson).
The original integration basis below records how the initial pipeline was built;
it does not override this ownership or the later completed-run status.

**Original integration basis:** development started under `lab/`, preserving tutorial independence, from the integrated course/tutorial/lab tree at `76c4967`; at the time of the fetch, GitHub `main` was still `dd001a3`. The initial branch followed the final layout integration before merge.

The layout retains `src/sbi4finance/heston.py` and `rough_bergomi.py`, but removes the older Heston path-inference pipeline. Recover only the needed functions and tests from `dd001a3`: `heston_bank.py`, `heston_inference.py`, `heston_reference.py`, `heston_forecast.py`, their generation/training/reference scripts, and their task definitions. Rehouse these in a focused `src/sbi4finance/path_inference/` package rather than restoring every historical workflow.

The old restricted Heston task supplies a valuable control: infer long-run variance and signed terminal auxiliary state with the other parameters fixed. It already has a recurrent encoder, reversible state transform, and leverage-aware particle/quadrature reference. Its old acceptance status does not transfer to the new tasks.

Use the existing locked `core`, `core-gpu` and `core-reference` environments (BayesFlow 2.1.0), and `cluster/core_cpu.sbatch` / `cluster/core_gpu.sbatch`. Do not use archived default/BayesFlow 2.0.14 environments for new fits. Existing option-bank collection and readiness commands are task-specific; reuse their identity and integrity conventions, not an assumption that they accept return paths unchanged.

## Initial 30 October pilot task definitions

These are the frozen initial research-pilot tasks, not domains certified by the POC or a harmonized classroom comparison. Retain their definitions and evidence unchanged. The next teaching tasks follow the common observation design above. A changed prior, grid or initialization creates a new task and bank identity.

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
7. **Frequency control.** First establish the harmonized Heston/Bergomi observation design. A subsequent optional frequency lesson should compare paired histories at fixed calendar duration and numerical law for each model. Aggregate groups of intraday log returns to obtain the matching daily observations. The original five-year daily Heston and one-year intraday rough banks do not isolate a model effect or a frequency effect.

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


## Data-generation economics and larger-bank design

The cost comparison is reproduced in external `local-artifacts/path-inference-20261030/generation-economics-20261030.json`. For option banks, sum `generation_cpu_seconds` over the canonical parts of each published snapshot exactly once. This is successful-process CPU time around generation, compressed saving and hashing, excluding failed attempts and allocation overhead. For trajectory banks, use complete single-CPU Slurm generation-job time (including final verification and imports). Each bank has 100k training examples; option banks include 10k/12k additional validation/test examples, while each trajectory bank includes 4k. Different tasks, priors and numerical grids mean this is a comparison of current pipelines rather than an intrinsic model speed ratio.

| Model | Option-generation process CPU-hours | Whole trajectory-generation job, one CPU | Option native stored GB | Trajectory stored GB |
| --- | ---: | ---: | ---: | ---: |
| Heston | 1.229 | 100 seconds | 0.128 | 1.069 |
| Rough known curve | 332.396 | 81 seconds | 1.060 | 0.858 |
| Rough joint curve | 2,459.184 | 71 seconds | 1.159 | 0.861 |

GB is decimal. Native option payloads already use compressed NPZ files; their complete distributed archives are 0.123/0.900/0.934 GB respectively. The trajectory shards currently use uncompressed NPZ. The rough option phases' allocated costs, including historical failures/overhead, were about 338.64/2,574.38 CPU-hours; neither CPU-hours column represents parallel elapsed time.

An option-surface example estimates 63 expected prices under Q, then adds the declared quote noise. Rough pricing shares paths across three maturity blocks and uses adaptive 32,768–2,097,152-path estimates per block, with antithetic pairs, conditional pricing and controls. The actual known-curve block grids are 80/504/504 steps; joint uses 1,280/2,016/2,016. Heston is the important exception: it uses adaptive Fourier quadrature, not Monte Carlo. Only price estimates, observations, labels, contexts and numerical diagnostics are kept; the millions of internal pricing trajectories are discarded.

A trajectory example draws parameters and one joint asset/latent-volatility path under the declared synthetic P law, then saves returns, parameters and IDs. Intrinsic path randomness is the intended observation law; averaging it away is unnecessary. Heston has 1,260 daily returns over five years with eight internal steps per return. Rough has 1,008 returns over one year at four observations per trading day and uses FFT hybrid simulation. Latent variances and normal innovations remain excluded from conditions and stored banks.

Raw-return storage is approximately `8 * N * T` bytes, with labels/IDs/headers/heldouts additional. One million current-length training histories require about 10.08 GB Heston or 8.064 GB rough in return payloads. One million ten-year daily Heston histories require 20.16 GB; one million five-year rough histories at four observations/day require 40.32 GB. At current horizons, linear-in-N extrapolation suggests approximately 10–20 CPU-minutes per million-history bank on comparable Oscar hardware, subject to file-system and verification overhead. Small local warmed generation-only checks found Heston 2,520/5,040 steps cost 1.85/3.71 times its 1,260-step baseline, while rough 2,016/5,040 steps cost 1.45/5.23 times its 1,008-step baseline. These are local scaling checks, not new Oscar throughput guarantees; FFT padding and numerical workload make strict linear-in-T forecasts inappropriate. The benchmark is saved in `generation-scaling-local-20261030.json`.

Generation is therefore cheap enough to support a much larger trajectory experiment. The next limits are loading, training and validation. The current eager trainer's conservative host estimates are about 43.5 GiB for one million Heston histories or 35.0 GiB for one million rough histories at current lengths, exceeding its 12 GiB guard. Stream shards and construct tokens per batch before million-history fits. The GRU currently scans padded maximum length, including masked steps; bucketed padding is a candidate optimization that needs parity and throughput checks. Bank expansion also needs a validated superset/checkpoint contract: the existing writer does not append and warm starts currently require an identical bank manifest. Increasing maximum history length changes task identity and RNG streams, so existing prefixes must not be assumed automatically preserved.

The original fits were 40–45 minutes on one GPU, with 52–60 epochs and roughly 5–6 million prefix presentations. The recent 3–4 minute runs were weights-only refinements. At unchanged epochs, 10 times as many examples costs roughly 10 times as much training, and longer maximum histories add sequence work. A first learning-curve comparison should instead match updates: 100k×50, 250k×20, 500k×10 and 1M×5 epochs each present about five million prefixes. This isolates more distinct data at similar optimization cost; runtime still needs measurement with streaming.

Revised next design after the user's 31 October teaching clarification: retain the current benchmark banks as historical controls, select a common observable grid/duration/prefix design for the primary Heston and Bergomi lessons, add bounded-memory offline input and extension support, then compare up to one million training histories at fixed update count on that common design. Longer-duration teaching examples should likewise use a common observation schedule and calendar duration. More training cases improve the amortizer; longer observed histories can increase statistical information. Slow curve decay has up to a four-year time constant; near-flat curves can remain weakly informative. Longer rough horizons also increase log-variance dispersion, requiring new numerical/prior-predictive checks. Frequency changes follow only as a separate paired extension after the harmonized example.

The cost assessment identified online generation as technically plausible, but the user explicitly selected pre-generated offline banks. Do not add online producers to this scale-up. Keep large banks on Oscar/artifact storage and transfer models and compact teaching cases for local use. No expanded Oscar jobs were started during this cost assessment or teaching-design revision.


## Authorized unattended offline campaign on 31 October

The user approved inferred drift and offline scale-up, and required thorough tests before submission and only necessary run dependencies. This is a new bounded tranche, separate from the completed 24 CPU-hour / 3 GPU-hour pilot allowance.

The common design is **1,260 daily log returns over five trading years**, with the same history checkpoints and observed-data interface for Heston, known-curve rough and joint-curve rough. All three infer annual proportional price drift with independent `Uniform(-0.10, 0.20)` prior. Heston has five targets, known-curve rough four and joint rough seven. The core lesson uses Heston and known-curve rough; joint curve is the extension. Heston keeps eight internal steps/day; rough uses four internal steps/day over its entire zero-origin Volterra path and sums four log returns into each daily observation. None of those latent states or innovations is observed. These are new named task identities; original fixed-drift banks and checkpoints remain unchanged. Legacy PMMH/scalar one-return reference helpers explicitly reject the new task rather than silently reusing an invalid reference.

Per model, generate **1,000,000 training / 4,000 validation / 4,000 reserved-test histories** once. Compare the first 100k training IDs with all 1M IDs from that same bank, using fresh weights and the same per-model initialization seed, recurrent GRU64/summary32, spline CouplingFlow6, batch128, Adam0.0005 and uniform/log prefix mixture. The planned 50/5 epochs give five million training views each, with 39,100/39,065 optimizer steps due to different final partial batches. Final completed epoch is selected; best validation loss does not select between 50 versus five opportunities. Each fit has a 3,300-second soft stop / 70-minute hard cap. Actual exposure is reported if either cap prevents the planned comparison.

The trainer reads pre-generated immutable shards through a two-shard LRU and constructs batch tokens on demand. It never simulates during training or loads the whole million-row split. Data-order/prefix randomness is deterministic by seed/epoch/batch. It records selected-ID and source-shard hashes, actual cache/memory statistics, completed training views, optimizer steps, task/runtime/source/model identities, and seeded native reload checks. The current 12 GiB conservative host guard is retained. Dataset integrity is checked once per accessed shard and later accesses verify immutable file statistics. GPU jobs request 24 GiB host memory.

Each model has an independent CPU pilot and production-shape GPU readiness job. Its bank depends only on its own pilot (finite/numerical execution, complete small-bank write/read tests, storage availability and a conservative forecast below the 75-minute generation limit). GPU readiness exercises the actual streamed dataset at batch128/T1260, backward updates, future-mask invariance and checkpoint reload. Bank generation does not depend on GPU readiness or the separate software-diagnostic job. Each fit depends only on its own complete bank check and own GPU readiness; each evaluation depends only on its own fit. Per-model comparisons run after either evaluation terminates and write explicit inconclusive reports for missing/invalid inputs. The final report runs after all jobs terminate, including failures. No model lane depends on another model lane. There are no automatic retries/requeues or local-laptop controllers.

The graph has **29 jobs** with conservative worst-case reservations of **42.25 allocated CPU-hours / 7.85 GPU-hours**, including 240 seconds of cleanup per job/core, bounded by 48 / 8 for this new tranche. Actual use is reconciled in the final report. Current Oscar association/quota checks confirm the condo GPU lane is eligible and scratch usage is about 27.8 GiB against a 512 GiB soft quota; expected banks require roughly 31 GB. The shared default GPU QOS permits only two GPUs per user, so the selected execution lane is `carney-mjfrank-condo2/gpu-he/carney-gcondo2` requesting one `l40s` GPU per allocation. Readiness and all numerical tests guard the reused locked runtime. Queue timing still affects completion; jobs continue independently while the user is away.

Evaluation uses the same 1,024 validation histories and 512 posterior draws at lengths 1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756 and 1,260. Rows 0–255 remain the loss-monitoring cohort; rows 256–1,023 are the primary new-validation cohort. Reports include paired risk/coverage differences, prior/wrong-case controls from the evaluator, actual training exposure, drift diagnostics and predeclared validation-case-0 posterior evolution figures. They do not automatically publish or accept a model. The 4,000 reserved tests per task remain closed to posterior evaluation. No new conditional-reference result is claimed for the inferred-drift daily tasks.

Local preflight passed **90 path-workflow tests**, including legacy arrays, new task/aggregation/drift contracts, large seed handling, bounded-cache integrity, subset identities, independent submission dependencies, durable partial receipts and missing-artifact reports. Production-network streamed fitting/reload was also checked across two epochs for all 5/4/7-target teaching tasks. A real CLI train/evaluate cycle on the joint task at T1260 passed, as did a production batch128 streamed readiness run and per-model small-bank generation/write/verify pilots. Manifest/claims checks remain current. Verification logs and pilot artifacts are stored under `local-artifacts/path-scaleup-20261031/`; no expanded bank or training allocation was submitted before these checks.


## Completed harmonized offline scale-up

The immutable numerical source is `64851349b7a3adfa58239181cd7f24da00ab411e`, archive SHA-256 `85721abde273cc3127fad8ef7432fb0a79a986ca580b5fe03fbe1db7a5d33637`. Source and run roots are `/oscar/scratch/afengler/sbi4finance-path-scaleup-20261031-v1/source/lab` and its sibling `run/`. All 29 allocations in the submission receipt ended `COMPLETED` / `0:0`, without retries, failed allocations or timeouts. The campaign ran from 30 October 19:50:18 to 21:57:22 Oscar local time (23:50:18 to 31 October 01:57:22 UTC), **127.07 elapsed minutes**. No jobs remain running.

Each task has exactly 1M training / 4k validation / 4k reserved-test histories. Native payload sizes are 10.372 GB Heston, 10.364 GB known-curve rough and 10.388 GB joint rough: **31.124 GB total**. Whole bank-generation allocations took 927/1,567/1,558 seconds on one CPU (15.45/26.12/25.97 minutes), including imports, saving, hashing and final verification. Independent bounded bank checks also passed.

All six models completed their full planned budgets: 100k fits 50 epochs and 1M fits five epochs. Each saw exactly **five million training presentations**, with 39,100 versus 39,065 optimizer steps due to final partial batches. All seeded native reload checks passed, model hashes bind to the evaluations/comparisons, and the first 100k training trajectories are the nested prefix of the same million-row bank. Training took about **33.2–34.4 minutes per fit**. There were no missing artifacts or plot errors; all three paired comparisons report complete development results. Reserved tests were not opened for posterior evaluation, and no model was automatically promoted.

Final terminal accounting is **16.20389 allocated CPU-hours / 3.445 GPU-hours**. Including the declared cleanup allowance, totals are **20.60389 CPU-hours / 4.045 GPU-hours**, below the new-tranche 48 / 8 ceilings. The reporter's own running snapshot was slightly more conservative than its final six-second allocation. Trainer `ru_maxrss` peaks were about 2.07–2.23 GiB, while Slurm's `.batch MaxRSS` for the 1M fits was 12.26/12.12/11.41 GiB (Heston/known/joint), below the 24 GiB requests. These are different memory-accounting measurements; no cause for their difference is asserted. The streaming input cache itself stayed bounded.

The primary comparison uses 768 previously unused validation histories (rows 256–1,023), with the same histories revisited across lengths. Findings at matched training exposure:

- Heston: mostly similar posterior-mean risk. Vol-of-vol improves modestly at three/five years (five-year normalized MSE 0.07950 to 0.07456), while the five-year kappa nondegradation uncertainty bound and vol-of-vol relative coverage guard fail. This does not establish a blanket advantage for the 1M fit.
- Known-curve rough: no broad improvement from 1M at this fixed update count. For example, one-year eta normalized MSE rises from 0.18969 to 0.20237, with paired pointwise change interval [0.00327, 0.02186]. Several risk nondegradation bounds fail. More unique trajectories alone are not sufficient to improve this task at the tested optimization budget.
- Joint rough: the clearest repeated benefit is H, with normalized MSE 0.509 to 0.479 at two years, 0.426 to 0.395 at three and 0.346 to 0.311 at five (about 10% relative improvement at five years). The corresponding paired pointwise change intervals are negative. Other targets change less; five-year long-variance/decay normalized risks remain about 0.714/0.963 in the 1M fit. The three-year rho risk bound and 126-day short-variance relative coverage guard fail.

Drift learning primarily benefits from observing a longer history. In the 1M fits, posterior-mean MSE divided by the prior-mean predictor's MSE is approximately 1 at one day, then 0.587/0.689/0.603 at one year and **0.245/0.224/0.196 at five years** for Heston/known/joint. The larger bank itself gives no clear drift improvement at those horizons. On the full 1,024-case development panel, average five-year drift 90% interval widths are about 42%/44%/39% of prior width. This is marginal development evidence, not a conditional-reference certificate.

Across the primary histories, 1M-fit nominal 90% marginal coverage ranges over tested targets/lengths are 88.3–92.2% Heston, 88.3–93.1% known rough and 86.2–91.9% joint rough. Joint long variance has particularly low coverage at some short/intermediate prefixes. These pointwise marginal diagnostics and repeated-history comparisons do not certify every history length, the joint posterior or real-data validity. New inferred-drift daily tasks still lack an accepted independent conditional reference. The next priorities are targeted calibration/reference checks and curation of the shared workshop workflow, not automatic bank expansion.

Final raw reports and terminal accounting have been copied to `local-artifacts/path-scaleup-20261031/`, with compact audit `final-audit.json`. `scaleup-summary.json` matches Oscar SHA-256 `eb4d9874c172d48d1437fc2051e12025be580191f259b8568b096361c44a7c5b`. Predeclared validation-case-0 posterior-evolution figures were generated; they are illustrations, not selected accuracy evidence. The original immutable source/report are unchanged. A local future-reporter fix adds the trainer's `completed_training_views` field to its high-level exposure display; the current comparison already recorded the correct [5M,5M] exposure, so scientific results are unaffected.

## Authorized finite workshop closeout

The user authorized the six-step closeout after the completed scale-up. Preserve all six native checkpoints, exact task/training records, development panels and reports. The 270,621,445-byte archive `completed-models-and-diagnostics.tar.gz` has SHA-256 `21d198732ad51184f346249b02637daab3f67dd403750a0a0be96306da73b059`; all 234 individual payload hashes were verified after download to `local-artifacts/path-workshop-closeout-20261031/`. Large native bank shards remain separately identified by their original manifests. No historical bytes are replaced.

Freeze the common 1M/five-epoch final endpoint as a **provisional development candidate** for each model, retaining the 100k comparison and every known development failure. This maintains one workshop training recipe; it is not a claim of universal larger-bank superiority. No additional neural training is justified without a diagnosed correctable mechanism. A hash-bound canonical selection receipt fixes models/tasks/banks/source/runtime, the 12 original evaluation lengths, 512 draws, seed 20261101001, untouched validation rows 1024:4000 (2,976 cases) and reserved tests 0:4000. Neither recipe nor thresholds may change after confirmation.

The declared screens require zero nonfinite/support/identity failures; 90% coverage Clopper–Pearson intervals with family alpha .025 Bonferroni-corrected across all 192 model/length/target cells; and bounded paired-loss empirical-Bernstein risk intervals versus the analytic prior mean, with a separate family alpha .025. Heston drift uses a one-sided minimum-coverage criterion because correct central intervals for an atomic posterior can legitimately overcover; other coordinates use two-sided nominal compatibility. The two families together have at most .05 error **per assessment phase**, not jointly across both phases. Compatibility does not establish calibration equivalence. Require no demonstrated risk degradation and at least one informative non-drift target at five years per model. Broad/unresolved cells remain explicit. These screens concern marginal performance across independent histories and do not certify conditional, joint, all-length or real-data accuracy. Statistical failures preserve their failed status. An explicit review may permit once-only final characterization of the unchanged development candidates with narrower claims; execution/identity failures cannot be overridden. Durable exclusive receipts prevent repeated test openings, including after a worker failure.

`cluster/submit_path_closeout.py` submits three independent CPU-only confirmation jobs (2 CPUs/12 GiB/30 minutes each), then three independent final-characterization jobs (2 CPUs/12 GiB/40 minutes each) after review. No model depends on another lane. Both evaluation phases together reserve at most **7.8 CPU-hours including cleanup, zero GPU-hours**. The broader closeout ceiling is **12 CPU-hours/zero GPU-hours**, including capped reference feasibility and execution checks, without automatic retries. Reference work and notebook curation proceed in parallel. Heston's inferred-drift target has parameter-dependent zero-integrated-variance return atoms; the old fixed-drift particle method cannot simply add drift. Rough's old single-fine-return quadrature also does not apply to four-step daily aggregates. New short-history references must match the actual finite-grid target, report independent-seed precision/importance ESS and close as inconclusive if unreliable.

The standalone lesson uses four predeclared validation cases per model, the same daily grid and history controls, cached posterior views by default and explicit optional CPU inference. Only observed returns and known time design enter inference; generating parameters are plotting overlays. Tutorial assets enter through the existing acceptance/manifest tools. Full tests, notebook execution, source parity and artifact checks precede integration and publication; scientific scope remains explicit even when software checks pass.

The actual closeout worker source is `49e3e3aae626cc88b4ea69481f4bb26879b1e144`, archive SHA-256 `268d851e996951ea3734ca4903a950a1e1822270a48dd120ef528aea95eba280`, under `/oscar/scratch/afengler/sbi4finance-path-workshop-closeout-20261031-v1/source/lab`. The canonical selected-model receipt is `d83ba139a61c138672a2b9f62c285bbd382abbf52f5dd327dd478c1e9ef6a1ba`. Confirmation jobs 8159378/8159379/8159380 and final jobs 8159935/8159937/8159940 all completed `0:0`. Following the explicit frozen-development review, each reserved split was opened once. Source setup initially encountered the system Python's older tar API before any selection/submission; extraction was completed with the verified Python 3.12 runtime. The first source archive is preserved as unused staging, not numerical run provenance.

Bounded reference feasibility has completed locally on inspected validation cases 0/1, with independent prior/latent importance integration and analytic integration of independent price noise and uniform drift. This is NumPy/SciPy numerical integration, not a PyMC result. All six one-day controls passed fixed ESS/independent-run precision screens; all four 21-day rough controls were inconclusive through importance-weight degeneracy. Heston multiday results are prior atom-support diagnostics only. All six one-day neural/reference comparisons have largest marginal mean discrepancy 0.1354 reference SD, neural 90% interval reference mass 0.8847–0.9278 and SD ratios 0.9529–1.0729. These inspected, two-case marginal comparisons do not confer general acceptance. Failed multiday reference checks are not evidence of neural failure, and no extra remote reference allocation was spent.

Heston has a structural numerical limitation: signed full-truncation Euler can remain at zero effective variance for every substep in a day, producing a return exactly equal to drift times the daily interval. With inferred drift the exact conditional posterior can therefore have point masses that the continuous spline flow cannot represent exactly. In the prespecified first 32 development histories, five have 18 returns numerically compatible with their generating drift/day; these are compatibility counts, not posterior atom probabilities or a population estimate. Further fitting cannot remove this support mismatch. Existing Heston files remain a **numerical-development prototype**, with a visible lesson warning and no endorsed calibrated-posterior claim. A newly versioned simulator with validated positive integrated variance, or atom-aware inference, belongs to a separate follow-up; do not silently project states, add observation noise or treat neural float32 rounding as a measurement model. Known-curve rough opens by default, and joint-curve rough remains an explicit extension.

Native participant models total 6,354,318 compressed bytes (SHA-256 `261c36550da52948aedcc2680d51bcb756db485626a6a2b8ee918fa9d6fff4cf`). The four initial archive files have locally verified hashes and Drive-provider upload-completion flags in the user-selected existing shared Drive folder, subfolder `path-workshop-2026-10-31`; no sharing permissions changed. Local CPU timing on the recorded Apple M5 Max machine separates three cold loads/model (medians 2.44–3.04 seconds) from nine warmed repetitions/length (512 draws in approximately 64–75 ms). Prefixes are padded to 1,260 internally; this is not a claim that inference cost scales with revealed length, or a single-thread benchmark. The fully self-contained `lesson-offline.html` and PNGs work without external frontend assets; Marimo HTML files are labelled static previews.

### Completed final characterization

The closeout evaluations ran for 40 minutes 16 seconds elapsed, from 30 October 23:38:14 to 31 October 00:18:30 Oscar local time. Six successful allocations consumed **3.41111 CPU-hours**, or **4.21111 including cleanup**, below the 8 CPU-hour evaluation ceiling and broader 12 CPU-hour closeout ceiling. No GPU was allocated. Peak batch memory was 4.305 GiB against 12 GiB requests. No jobs remain active, no retries occurred and no additional fits were run.

Untouched confirmation used 2,976 histories per model. All execution/risk checks passed; known rough passed its marginal compatibility screen, while Heston and joint rough failed. The sealed explicit review retained all failures and allowed only descriptive tests of the unchanged candidates. Each final panel then used exactly **4,000 reserved histories, 512 draws and the same twelve lengths**, with the fixed seed and model/task/bank identities. The final receipt is `b59ddf1501712352bfb76204673e772e6316792de985692dab4cf9072db6cfa0`. All final execution and risk checks passed; **all three final marginal calibration screens failed**. Known rough's confirmation pass did not reproduce in every final cell. No model was promoted.

| Model | Final nominal 90% coverage range | Failed declared cells (days: target, coverage) | Five-year drift MSE / prior-mean baseline MSE |
|---|---|---|---:|
| Heston prototype | 87.75–92.475% | 252: kappa 87.75%; 504: rho 92.30%; 756: rho 92.20%, vol-of-vol 91.975%; 1260: vol-of-vol 92.475% | 0.24521 |
| Known-curve rough | 89.00–92.00% | 1260: drift 92.00% | 0.20651 |
| Joint-curve rough | 87.575–91.10% | 252: rho 87.575% | 0.22104 |

These are independent-trajectory marginal screens with repeated prefixes, not conditional or joint certificates. Overcoverage and undercoverage retain their direction. Risk passes indicate useful estimation relative to the prior baseline under the declared simultaneous bounds; weak joint long-variance/decay cells remain broad or unresolved. Five-year drift MSE is approximately 75–79% below the prior-mean baseline. Future tuning must use a new held-out cohort; these reserved tests have now been consumed.

The 92,104,580-byte final audit archive has SHA-256 `a29fd9bdf1cc9d44aa4a9bcfe0f30125dd4c59e759c8c1bb1cfb649ef4e9a180` and is checksum-verified outside Oscar scratch, alongside all previous models and the new reference audit. The standalone tutorial is packaged with all three native models and verified independently of the checkout: 208 initial payload hashes, 29 tutorial tests and actual seeded 137-day CPU inference for all models. Full lab tests passed 476 tests with three optional PyMC-environment skips; seven notebooks exported 183 cells without genuine errors. Source parity, manifest/claims, Marimo and diff checks passed. Final aggregate reports/figures enter tutorial assets only through the existing acceptance and manifest tools. Source integration is prepared as [draft PR #63](https://github.com/fmuia/sbi4finance/pull/63).

## Oscar preservation closure — 1 November 2026

The [preservation record](../evidence/pilot-results/2026-11-01/oscar-preservation/README.md)
reconciles 41 original run roots, 1,650 JSON receipts and all known allocations.
All three million-history trajectory banks are now available in native form
off Oscar: 30 resumable archive parts, 29.172 GB compressed / 31.124 GB native,
with all 23,631 extracted shards verified one at a time. Historical unique
results, including old fixed-drift banks, repaired models, intermediate option
studies and failed/partial references, are archived and logically hash-verified.
Source/config identities and explicit exclusion proofs are retained.

Original Gmail copy/readback proofs are retained. The remaining five bank
parts were copied and hash-verified through the verified Brown-account view;
fresh Brown provider metadata confirms completion under the same Google parent
and workshop identities. Old cold payloads were not re-read or freshly hashed.
The Gmail-path denial remains unexplained; sharing permissions were unchanged.
An independent cloud download was not performed. Remote originals
and existing versioned workshop archives are unchanged. No active allocation
or unpreserved unique output was identified in the saved accounting scope.
The last successful queue observation remains bound; the latest read-only retry
failed with expired SSH authentication. The live queue was not freshly rechecked,
and no jobs were submitted after the pilot. The seven packaging jobs
and one isolated numerical pilot used 784 allocated CPU-seconds and no GPU;
historical accounting remains separate. Use the [training guide](training-data-guide.md#trajectory-banks-daily-histories-with-inferred-drift)
for extraction and streaming training. This closes preservation work; numerical
accuracy, conditional references and fresh acceptance remain the follow-ups below.

## Bounded numerical diagnosis follow-up — 1 November 2026

This develops [application subplan E](../../tutorial/curriculum/TUTORIAL-REVIEW.md#e-resolve-trajectory-numerical-and-inference-limitations)
against merged baseline `7751e4c`. The current implementation tranche is diagnosis
and specification; this section submits no jobs and does not extend an earlier
allocation. Notebook refinement and preservation proceed independently. Equal
application status requires the same evidence discipline for trajectory and option
inference; it does not turn a working interface into scientific acceptance.

### E1. Freeze the evidence and locate the failure

Use the existing training/validation outputs for development. The inspected
4,000-history final panels are now historical characterization, never a fresh
selection or acceptance cohort. The [closeout summary](../evidence/path-inference/2026-10-31/closeout-summary.json)
binds task/model hashes and failed cells. Preserve all six 100k/1M fits and the
old numerical laws before modifying a candidate.

| Lane | Recorded issue | First diagnosis on development data |
|---|---|---|
| Heston | Several marginal failures and an atom-bearing numerical return law | Separate zero-effective-variance episodes from continuous-return cases; inspect interval shape and boundary behavior without treating compatibility counts as posterior atom weights. |
| Known-curve rough | Five-year drift **overcoverage** of 92% | Inspect drift bias, posterior variance and interval tails by horizon and prior region. A wider interval is not the remedy for overcoverage. |
| Joint-curve rough | One-year rho **undercoverage** of 87.575%; decay can remain broad | Inspect rho bias/width, eta/rho and curve trade-offs, prior boundaries and nearly flat curves. Broad decay alone is not evidence of an inaccurate posterior or structural nonidentifiability. |

For all lanes, compare posterior-mean risk with the analytic prior-mean baseline,
coverage at several nominal levels, rank diagnostics, support/nonfinite counts
and saved within-case dependence. Retain the direction and uncertainty of each
effect. Stratify by parameter quantiles and curve orientation using declared
development groups; truth labels are diagnostic overlays, never estimator inputs.
Check actual prefix-exposure histograms, endpoint behavior, loss history and
selected/final checkpoint identity before attributing a failure to data volume.
The matched scale-up used five million presentations for either bank size;
five epochs on one million rows is not evidence of less optimization by itself.
Use a fixed higher-draw replication only on development cases if needed to
separate finite posterior-sample noise from estimator error.

**Bound:** one source-bound diagnosis table and a fixed case panel per lane,
with an initial proposed ceiling of four allocated CPU-hours and zero GPUs.
No architecture search or repeated acceptance testing belongs to E1.

The [first development diagnosis](../evidence/path-inference/2026-11-01/development-diagnosis/README.md)
is complete: 192 comparisons bind 27 input files and use 768 already-inspected
validation histories as their primary cohort. Known-curve eta shows negative
long-history mean error; the joint results do not isolate a correction. The
record names the next data/reference diagnostics and justifies no production
fit, broader architecture search or additional bank generation.

### E2. Choose Heston's numerical target before a refit

The first 32 inspected development histories contained five histories with 18
daily returns numerically compatible with generating drift/day. This is a
forensic panel, not a population atom-rate estimate. The issue is structural:
under signed full-truncation Euler, eight zero-effective-variance substeps can
produce a deterministic daily return. A continuous spline density cannot exactly
represent its resulting drift atoms. More training cannot remove that mismatch.

Compare at most two explicitly specified directions:

1. **A new Heston numerical law with strictly positive daily integrated variance.**
   Select a documented construction preserving variance dynamics, leverage and
   the joint price/integrated-variance law over the whole declared prior box.
   Exact variance endpoints alone do not make a joint price transition exact.
   Show why the independent price-noise variance is positive; do not substitute
   an arbitrary floor or added measurement noise. Retain the daily grid,
   inferred proportional drift and parameter priors unless a separate modelling
   decision changes them. A finer version of the existing signed scheme can
   reduce observed episodes but does not establish absence of atoms.
2. **The unchanged v2 observation law with atom-aware inference.** This requires
   deriving and representing both drift atoms and continuous posterior mass,
   with reference comparisons of their weights. It is a different estimator
   contract, not another fit of the existing continuous flow. If that work is
   disproportionate, retain v2 as a development control while specifying the
   first direction; do not silently drop difficult histories.

Before selecting a law, use a prespecified panel of 64 parameter settings spanning
the prior and low-variance/high-vol-of-vol regions, with four innovations each
(256 histories/law), at the daily maximum horizon. Compare the working internal
grid and two refinements, using shared Brownian increments where a valid coupling
exists; otherwise use replicated weak-distribution comparisons. Record integrated
variance, return and realized-variance summaries, boundary events, variance-mean
and price-drift/martingale checks, finite outputs, runtime and memory. Distinguish
Monte Carlo uncertainty from numerical differences. Require analytical support
arguments and checked leverage, not merely zero observed atom episodes.

Choose by fidelity to the intended Heston process, numerical stability across
the prior, feasible matched references and measured generation cost. Convenience
for a neural density alone is insufficient. A changed law gets a new immutable
task ID/version, source hash, bank and catalogue entry; preserve v2 bytes and
results. Publish neither a numerical-accuracy nor an exact-transition claim
without the corresponding checks.

#### E2 first-candidate pilot: projected Euler

**Analysis/prototype only: no production target has been adopted.** Test the
documented absorption fix, described here as projected Euler:
\[
V_{j+1}=\max\{V_j+\kappa(L-V_j)h+\sigma\sqrt{V_jh}Z_j^v,0\}.
\]
Keep the Gaussian log-price increment
\((\mu-V_j/2)h+\sqrt{V_jh}(\rho Z_j^v+\sqrt{1-\rho^2}Z_j^\perp)\),
including its shared variance driver and independent price noise. Zero is not
absorbing: its next value is \(\kappa Lh>0\). Thus no two consecutive variance
states are both zero; eight left-point substeps contain at least four positive
states. In real arithmetic, daily \(I_h=h\sum_jV_j>0\) and conditional price-noise
variance \((1-\rho^2)I_h>0\) over the declared prior. This argument gives no
uniform positive lower bound on \(I_h\); finite-precision behavior still needs
checking. The adapted Gaussian price increment preserves proportional price
drift. The [Lord–Koekkoek–van Dijk comparison](https://papers.tinbergen.nl/06046.pdf)
documents absorption's boundary bias and generally better option-price accuracy
of full truncation: removing this support mismatch does not certify accuracy.

[Andersen's QE scheme](https://www.maths.univ-evry.fr/pages_perso/crepey/Finance/andersen%20vol%20sto.pdf)
is not an automatic remedy: its high-variance-ratio branch has a zero endpoint
atom, and the joint price approximation can have zero conditional variance when
both endpoints are zero. Exact endpoints alone are insufficient.
[Broadie–Kaya](https://doi.org/10.1287/opre.1050.0247) instead samples conditional
integrated variance and the joint price law; numerical inversion and its cost
need separate checks before treating it as a reference or production generator.

The [first numerical pilot](../evidence/path-inference/2026-11-01/numerical-pilot/README.md)
completed on Oscar as job 8206381 at source `5eeefc4`, using the 64-setting,
four-innovation panel and coupled 8/16/32 grids. All recorded values were finite.
The projected law had zero daily integrated-variance atoms in this panel; the
signed control had 11,412 / 6,691 / 3,342 zero days respectively. These deliberately
selected settings do not estimate a population atom rate. The allocation used
13 CPU-seconds and 203.3 MiB peak memory, with no GPU. Source, results and logs
are preserved locally and provider-confirmed uploaded.

Next, specify a bounded fixed-parameter precision check against continuous-time
moments and characteristic functions before deciding on a production law. Retain
the coupled-grid sensitivity and explicit boundary-bias concern. Four paths per
setting cannot certify weak accuracy; the completed support screen neither
selects projected Euler nor justifies a production bank regeneration/refit.

### E3. Reuse data only when the target is unchanged

| Proposed change | Reuse | Required new work |
|---|---|---|
| Rough prefix exposure, optimization or density capacity; unchanged v2 task | Existing immutable one-million-history offline bank and its canonical 100k subset | New fit identity, matched development evaluation and fresh acceptance cohort. |
| Heston numerical law, observation noise, prior or observation design changes | Old parameters/appropriate innovations as a controlled comparison panel; old bank as historical evidence | New target identity and newly simulated observations; no manifest relabelling or inherited reference. |
| Atom-aware estimator for unchanged Heston v2 law | Existing v2 bank, including difficult histories | New model/loader contract and mixed-posterior references; fresh acceptance cohort. |

For rough, select one mechanism per lane from E1 and change only that mechanism
in the first experiment. Keep the original model as the frozen baseline. Do not
both enlarge the bank and change capacity/optimization and then attribute the
effect to one component. Compare at matched presentations and report optimizer
steps, actual prefix frequencies, validation use, wall time and peak memory.

**First refit bounds, proposed for a separately admitted E2/E3 tranche:** at most
one candidate per model, one declared seed, batch 128, at most five million
training presentations and a hard maximum of 60 GPU-minutes/model (three GPU-hours
total). Keep the endpoint/checkpoint-selection rule fixed before execution.
Production-shape readiness and a measured pilot must show the planned work fits
before submission; if the cap stops a fit, record actual exposure and an
incomplete comparison rather than silently increasing it. Cost generation and
references separately from measured pilots. A broader seed replication or
second candidate needs a new bounded tranche. No first-pass one-million-row
Heston regeneration is automatic: verify a small bank and decide its necessary
size from the diagnosed question first. Use bounded offline shard streaming;
online producers remain outside this design.

### E4. Stage conditional references and a fresh final cohort

The first identity-contract implementation adds
`evaluate_path_posterior.py --bank TRAINING_BANK --cohort-bank DEVELOPMENT_BANK`
for development/validation only. It retains the exact native training-manifest guard, checks the same task
with a distinct root seed and canonical IDs, and reads only the selected cohort
rows/shards. Exclusive per-output opening receipts bind both manifests and the
model; a durable validation-exposure marker permits paired development comparisons
without describing the exposed data as fresh acceptance. **A versioned fresh
confirmation/acceptance selection-and-review schema remains pending.** The new
route rejects reserved tests and the old confirmation/final guards; the default
route keeps those original guards unchanged. No fresh cohort was generated and
no acceptance result follows from this software change.

Keep the existing one-day validation cases 0/1 as development controls. Their six
stable marginal comparisons do not validate multiday or joint inference. The
four 21-day rough importance references were inconclusive through weight
degeneracy; allocating more samples without an ESS/precision forecast is not a
validated remedy. First derive a reference for the actual chosen numerical law,
then try a bounded intermediate horizon (four days), followed by 21 days. A
one-year feasibility attempt follows only if shorter checks pass and its measured
cost fits a separately declared cap. Do not promise a five-year latent-path
reference in this tranche. Record actual NumPy/SciPy, particle or PyMC provenance,
target density/Jacobians, independent-run precision and convergence failures.

For a retained mixed Heston target, references and SBC must handle atoms/ties
explicitly. Central intervals can legitimately contain more than nominal mass;
do not force them toward 90% by shrinking a correct mixed posterior. For a new
continuous target, use the declared continuous-posterior checks. In neither case
does optimizer or chain completion establish reference accuracy.

Before new acceptance evaluation:

1. Freeze the selected model bytes, task, training recipe, numerical/reference
   scope, assessed quantities, thresholds and draw seeds. Select using development
   evidence only; keep the inspected final panels labelled consumed.
2. Generate 4,000 genuinely fresh maximum-length histories/model from a new
   recorded root seed, with disjoint trajectory identities. Keep all prefixes
   of a trajectory together. Preserve explicit generation/source receipts.
3. Add an explicit evaluation-cohort identity to the evaluator. Its current
   guard requires the training bank's exact manifest; a fresh cohort necessarily
   has another manifest. Validate both identities, exact task agreement and
   disjoint IDs. Never rewrite the frozen training record or relax its existing
   identity guard to make a new cohort appear original.
4. Evaluate 512 draws/history at the twelve existing reported lengths plus two
   fixed non-landmark lengths 137 and 997. Count all assessed model/length/target
   cells in the declared simultaneous family; do not retain the old family size
   after adding lengths. Retain execution/support checks, marginal coverage and
   prior-relative risk screens. Report paired baseline/candidate comparisons only
   on a genuinely shared target and cohort. An old-law Heston model applied to
   new-law data is a transfer experiment, not a matched-target baseline.
5. Open each fresh acceptance cohort once, after the freeze, with durable
   exclusive receipts. Preserve failed outcomes and actual resource use. A failure
   returns the model to development; further tuning consumes a newly generated
   cohort, not a rerun until the same panel passes. Marginal compatibility is not
   conditional/joint certification or proof of accuracy at every integer length.

**Delivery gate:** a compact diagnosis and chosen-target record precedes any
production generation/refit. New numerical results reach the lesson only through
the existing exporter, catalogue, manifest and runtime-boundary checks, with labels
matching the checks actually passed. A new task version may need a separately
reviewed runtime/loader extension; do not overwrite the v2 tutorial assets to
bypass its strict task contract. Until then retain all current candidate and
prototype labels while continuing the application-level presentation work.
