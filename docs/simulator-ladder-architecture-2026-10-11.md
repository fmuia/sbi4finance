# Simulator ladders and pre-simulated inference datasets

11 October 2026. Design proposal, not an implemented API or a curriculum decision. This note builds on the local HKDE data specification and SVJ integration assessment. It proposes shared infrastructure that supports either application without conflating historical-return inference with option calibration.

The expanded [implementation plan and mathematical guide](simulator-plan/README.md) now provide milestone subplans and explicitly distinguish reference pricing, learned pricing surrogates and direct posterior-price inference. Use that roadmap for the implementation review; this note preserves the initial architecture discussion.

## Recommendation

Build a small simulator layer inside `sbi4finance`, with two model families, explicit observation/inference tasks, and one reproducible dataset-generation pipeline. Start with NumPy reference calculations and Numba for suitable CPU kernels. Add Cython only where measurements justify the build and maintenance cost. Keep BayesFlow, Slurm, and storage outside the scientific kernels.

The useful product is a reproducible mapping from a declared experiment to training examples, not simply a collection of fast path generators. Model complexity, numerical fidelity, and inference difficulty should be separately configurable and separately documented.

Develop the contracts and a small working bank now. Integrate Francesco's exact SVJ implementation when available, preserving its discretization, initialization, and parameterization before attempting replacements. This extends the earlier recommendation to defer duplicate SVJ work: shared infrastructure and analytic baselines can proceed independently of his scientific implementation.

## 1. Model ladders

### Heston and jumps

| Stage | Model | Why include it? |
|---|---|---|
| A0 | GBM / deterministic variance | Exact grid increments; analytic pricing and simple inference checks |
| A1 | Heston | Hidden variance, leverage, parameter–state inference, and a characteristic-function pricing reference |
| A2a | Heston with Gaussian log-price jumps | A precisely specified Bates/SVJ variant; jump identification and predictive tail risk |
| A2b | Heston with double-exponential log-price jumps: HKDE | Asymmetric jump alternative, sharing the variance process and much of the simulation machinery |
| A3, deferred | Joint price/variance jumps, such as a specified SVCJ model | Additional mechanism only when a concrete inference question requires it |

A2a and A2b are siblings. HKDE is already an SVJ model in the broad sense. Use descriptive internal identifiers such as `heston_gaussian_jumps` and `heston_double_exponential_jumps`; the name `svj` alone does not specify the jump law. The HKDE paper explicitly uses a Heston backbone and compares Gaussian and double-exponential jump choices under the risk-neutral measure. [HKDE paper](https://arxiv.org/html/2502.13824v1)

Share the variance step and explicit price-jump components, but register a small set of supported combinations instead of accepting every possible combination of components. Constant-variance Gaussian/double-exponential jump limits are useful verification cases without requiring full additional teaching notebooks.

### Bergomi and rough volatility

| Stage | Model | Why include it? |
|---|---|---|
| B0 | Deterministic forward-variance curve | Curve conventions and exact integrated-variance checks; can reuse A0 machinery |
| B1 | One-factor exponential-kernel Bergomi | Finite-dimensional volatility state and lognormal forward variance |
| B2 | Two-factor Bergomi | Fast and slow volatility responses and factor dependence |
| B3 | Rough Bergomi with a power-law kernel | Roughness, memory, and more demanding simulation/calibration |

This is a progression in ideas, not a claim that every model is obtained by setting one parameter of the next. A multifactor approximation to a specified rough kernel belongs to the **numerical fidelity** ladder. Its factor count and approximation weights are numerical settings unless the scientific model explicitly treats them as unknown parameters. A two-factor Bergomi model estimated in its own right is a different specification.

The one-factor exponential kernel admits an OU state; rough Bergomi does not have a finite-dimensional Markov representation of its forward-variance curve. Standard Bergomi formulations are under Q. A returns task under P needs declared dynamics and initialization; changing the asset drift alone does not establish a P-to-Q mapping or identify risk premia. [Forward-variance model definitions, equations 1.1–1.4](https://arxiv.org/html/2202.10413v2)

Build B1 before committing to the more demanding B3 implementation. B2 is useful but need not block a rough-model prototype. Rough Heston is a distinct family, not another name for rough Bergomi; leave it outside the initial implementation.

## 2. What defines a simulator setting?

Use a few immutable specifications, with explicit validation at the Python boundary:

| Specification | Owns | Example |
|---|---|---|
| Model | Equations, parameter schema and units, measure, jump law, state representation | Heston plus Gaussian price jumps under P |
| Observation | Observable channels, sampling times/contracts, known context, measurement/missingness law | 252 daily log returns and their elapsed times |
| Inference task | Prior, free/fixed parameters, initialization law, target selection | Joint posterior of static parameters and terminal variance |
| Numerics | Integration scheme/grid, rough-kernel approximation, precision, pricing tolerances and MC budget | Four internal steps per observed interval |
| Execution | Example IDs, batch/shard sizes, workers, backend, output location | CPU array, one independent shard per task |

For Bergomi, explicitly classify the initial forward-variance curve as known context, an uncertain measured input with an observation model, or an unknown parameterized curve. A flat curve is a useful first experiment, not an implicit default for general calibration. Likewise declare whether Heston's initial variance is fixed, inferred, drawn from a conditional stationary law, or obtained by burn-in.

The internal integration grid is distinct from the observation grid. A missing quote or return does not remove elapsed time. Varying sequence length, noise scale, contract design, or mask patterns requires a training task that supports those variations and an adapter/encoder that uses the relevant context.

A change of probability measure is substantive. Shared kernels can implement separately specified P and Q processes, but a posterior trained on historical returns cannot automatically be used for derivative valuation.

## 3. One training example and one observation

| Task | One complete inference example | Target | Internal numerical replication |
|---|---|---|---|
| Returns | One ordered return window, times, mask, and declared known inputs | Parameters and, when required, terminal state | Substeps and random increments used to generate that window |
| Returns plus a volatility proxy | Return window and noisy proxy measurements | Parameters and latent states/functionals | Intraday paths or measurement-noise draws |
| Option calibration | One quote panel, contract coordinates, curves/forwards and mask | Q parameters and any unknown initial variance/curve coefficients | Many MC paths if a numerical expectation is needed |

Daily returns are not independent training examples for a posterior conditioned on a whole history. Likewise, 10,000 paths used to calculate one option surface yield **one surface example**, not 10,000 independently observed surfaces.

For returns, the path generator and observation operator together define the data distribution. For options, the observation generator normally evaluates a model price and then applies a declared quote-error model. Heston/HKDE should use a validated Fourier/PROJ engine where suitable; a path simulator remains useful for cross-checks and payoffs. Rough Bergomi may require MC pricing. Algorithm and variance-reduction choices should precede compiler optimization. [Rough Bergomi variance reduction](https://arxiv.org/abs/1708.02563)

Separate market quote error, Monte Carlo estimation error, and discretization bias. More accurate prices should approach the same declared observation model. Training on noisy numerical prices otherwise changes the effective likelihood; record MC uncertainty and demonstrate that it is controlled relative to the intended observation noise. Prices from shared paths have correlated numerical errors.

An illustrative batch contract is:

```text
example_id       [B]                  immutable identity
parameters       [B, P]               raw physical values, declared order
observations     [B, T, C]             returns task
                 [B, M, C]            quotes task
mask             same observation axes, with declared semantics
context          task-specific named arrays: times, contracts, curves, etc.
target_state     task-specific arrays, only if inferred
diagnostics      numerical flags, error estimates; excluded from conditioning
```

Each bank has one declared schema. Use fixed lengths/grids first, then explicit length buckets or padded arrays with masks. Preserve axes even when B, T, or C is one. Never pad missing observations with economically meaningful zero returns without a mask. Parameter/state truth must be separated from network inputs; only explicitly observed context reaches the encoder. Store full latent trajectories for selected reference cases or tasks that actually infer them, rather than automatically for every example.

The interface permits multiple inference objectives later, but the first bank should serve joint posterior inference. A per-transition learned likelihood is a different task with different conditioning and latent-state integration; the existing SSMS likelihood-data layout should not define the finance bank.

## 4. Forecast continuation is a model capability

Use a model-specific continuation state, not a universal `terminal_variance` interface:

| Model | State needed for conditional continuation |
|---|---|
| Heston and independent price-jump extensions | Current variance; current price when outputs depend on price level |
| Finite-factor Bergomi | All factor states, curve specification, and time |
| Rough Bergomi | A sufficient memory/conditional forward-variance representation for the chosen discretization |
| Finite-factor approximation of a rough model | All retained approximation factors and their kernel specification |

Path-dependent contracts additionally require their accumulated payoff state. For rough returns inference, estimating only `(parameters, current variance)` generally does not solve conditional forecasting. One must infer or conditionally reconstruct the relevant memory, or explicitly use a finite-factor approximation. Returning a true memory array during training is not a substitute for estimating it from observed data.

Preserve joint posterior draws when continuing paths. Independently pairing marginal parameter and state samples changes the predictive distribution. Keep predictive-mixture VaR/ES separate from a posterior distribution of conditional VaR/ES; they answer different questions.

## 5. Package shape and backend choice

Start inside the current package; a separate published finance-simulator package can be extracted once multiple consumers require it.

```text
src/sbi4finance/
  simulators/
    specs.py, registry.py
    gbm.py, heston.py, jumps.py, bergomi.py, rough_bergomi.py
  observations/
    returns.py, option_quotes.py
  pricing/
    fourier.py, monte_carlo.py
  tasks/
    spec.py, generate.py
  datasets/
    shards.py, manifest.py
  adapters/
    bayesflow.py
configs/tasks/
scripts/generate_bank.py
scripts/verify_bank.py
cluster/simulate.sbatch
```

This is a proposed layout, not a requirement to create all modules immediately. Keep the existing MMAR implementation/checkpoint intact. A small registry should advertise capabilities such as path simulation, continuation, and characteristic-function pricing. Unsupported combinations fail explicitly. Use functions and typed data containers; avoid a deep inheritance framework or general-purpose plugin system at this stage.

Conceptual boundaries:

```python
simulate_paths(parameters, *, model, initial_state, grid, numerics, streams)
price_contracts(parameters, *, model, market, contracts, numerics, streams)
continue_paths(parameters, *, model, state, grid, numerics, streams)
generate_examples(task, *, example_ids)
```

The scientific kernels accept parameter values rather than drawing a hidden prior. `generate_examples` composes prior/context/initial-state draws with the appropriate forward and observation operators. Inferred or known initialization is resolved by the task, with exactly matching conventions in reference inference.

**NumPy + Numba is the initial implementation choice.** Keep inspectable float64 reference calculations and use Numba for time-stepping and independent path loops where benchmarks justify it. Keep schemas, validation, file I/O, priors, and orchestration in ordinary Python. Training can use JAX while simulation runs on CPU; the simulator does not need the neural network's backend.

Cython becomes worthwhile for a persistent measured bottleneck, an existing reusable C/Cython kernel, or distribution requirements favoring compiled extensions. It brings compilation, wheels and potentially OpenMP dependencies. Cython's `nogil`/`prange` machinery is useful for parallel numeric kernels, but releasing the GIL alone is not an algorithmic speed-up. [Cython documentation](https://docs.cython.org/en/latest/src/userguide/nogil.html)

For rough models, convolution, memory layout and variance reduction can dominate interpreter overhead. Use a small-grid Gaussian-covariance reference and assess a hybrid scheme with appropriate convolution for production. A finite-factor route is a separately versioned approximation. [Hybrid scheme](https://arxiv.org/abs/1507.03004), [authors' reference implementation](https://github.com/rmcrkd/rough_bergomi)

Benchmark generation and serialization end to end, including cold compilation, steady-state examples/second, peak memory, numerical error, and dollars/core-hours for a specified bank. No Numba-versus-Cython speed claim is established by this design.

## 6. What to borrow from ssm-simulators / HSSMSpine

The useful pattern is the separation of a model catalog, explicit parameter/output contracts, efficient numeric kernels, and training-data generation. SSMS already provides C/Cython kernels and a distinct data-generation layer. Its observation schema work also makes the distinction between stochastic observations and auxiliary state explicit. [SSMS repository](https://github.com/lnccbrown/ssm-simulators)

Local inspection found the main SSMS checkout on `codex/structured-observation-contract` at `20e718e`; cached `origin/main` was `cca9887`. Those local references differ, so proposed feature-branch interfaces should not be mistaken for the installed/public API. Cached main contains observation-schema utilities, while the familiar simulator interface still produces its legacy RT/choice output.

Adopt stable dimensions, model registration, backend conformance tests and a clear producer/consumer boundary. Adapt the data unit to financial time series and quote panels. Do not import RT/choice conventions, KDE/LAN-specific training rows, or automatic assumptions of exchangeable trials. Keep finance independent of HSSM runtime packages.

Also strengthen seed identity beyond the inspected SSMS single-/multi-thread behavior: the documented NumPy/GSL paths can produce different streams when thread count changes. Dataset identity should be tied to examples, not thread assignment. The goal is reproducible scientific examples within a pinned backend, not universal bitwise equality across compilers/backends.

## 7. Pre-simulated banks and the cluster

The core pipeline should be:

```text
versioned task + example IDs
          -> CPU generation array
          -> individually validated immutable shards
          -> verified bank manifest
          -> GPU/CPU training
          -> checkpoint tied to the manifest
```

Use the same generation CLI locally and in a Slurm array. The scheduler only assigns predetermined work; it does not define the data distribution.

Each bank should retain the canonical task specification/hash, source revision (and any recorded patch), environment-lock hash, schema, parameter order/units, RNG algorithm, root seed, ID ranges, split policy, numerical settings, diagnostics, checksums and creation metadata. Checkpoints additionally record architecture, adapter/transforms, training-only normalization, and bank identity.

An initial layout:

```text
banks/<bank-id>/
  spec.json
  manifest.json
  train/shard-000000/
    example_id.npy
    parameters.npy
    observations.npy
    target_state.npy           # when the task needs it
    ... named context/mask arrays ...
    metadata.json
  validation/...
  test/...
```

Start with many-example `.npy` shards and JSON metadata: straightforward NumPy memory mapping, no extra storage service. Choose shard sizes from measured I/O and memory, initially on the order of tens to hundreds of MB. Small NPZ banks are convenient for pilots but do not provide normal array memory mapping. Consider HDF5/Zarr only when compression or access patterns justify them; one writer per shard remains the simpler concurrency contract.

For scale, one million 252-day one-channel float32 return windows occupy about 1.0 GB before parameters, masks and metadata. Saving thousands of pricing paths per surface multiplies storage unnecessarily; save the surface, required labels and numerical diagnostics instead.

Seed each example and purpose using `SeedSequence` with stable numeric identifiers, for example `(purpose_id, example_id, split_id, task_id, root_seed)`. Use separate streams for prior, context, initialization, process and quote error, plus explicit replicate IDs where needed. Derive IDs deterministically, never with Python's process-randomized `hash()`. Shard boundaries, job IDs, worker count and retries must not redefine an example. [NumPy parallel RNG guidance](https://numpy.org/doc/stable/reference/random/parallel.html)

First use independent process/job streams with serial RNG consumption per example, or generate random blocks outside parallel numeric kernels. Numba currently documents that its support for NumPy `Generator` is not thread-safe; do not share one mutable generator across a `prange` loop. A faster kernel RNG must satisfy the same example-identity contract. [Numba RNG support](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#generator-objects)

Workers write temporary shards, validate them and publish by atomic rename on the same filesystem. A retry skips only shards with matching specification and checksums. A final verifier checks expected IDs, duplicates, shapes, failed examples and split disjointness, then freezes the manifest. Record numerical failures against their original IDs; silently redrawing until success creates an undocumented selected prior.

Split by independent base experiment before augmentation, repeated quote-noise draws, or overlapping windows. Common random numbers used for method comparisons belong in explicitly grouped reference cases; do not share them accidentally across train/test. Maintain both held-out cases from the training law and separate higher-accuracy fixtures for numerical-bias checks.

Generate banks on cluster data storage, then train there. Transfer small fixtures, diagnostics and checkpoints back to the laptop; there is no need to shuttle the entire bank through the local connection. Cap array concurrency and coordinate process/Numba/BLAS threads to the allocation. Validate the bank before starting dependent training jobs.

## 8. Fit to the current repository

The inspected local scaffold contains an in-memory MMAR training path: `scripts/train_univariate.py:53–57` generates arrays and calls `fit_offline`. `cluster/train.sbatch` is a one-GPU MMAR job. `cluster/discover_cluster.py:105–114` already anticipates CPU simulation versus GPU training, but no simulation array, shard writer, manifest verifier or large-bank loader is implemented.

The local BayesFlow checkout is tag `v2.0.14`, matching the scaffold pin. Its `DiskDataset` treats each file as one complete example and stacks files (`bayesflow/datasets/disk_dataset.py:107–115`). Passing a multi-example shard directly to `fit_disk` would add the wrong batch dimension. Small banks can use the existing `fit_offline` path. Larger banks need a thin shard-aware `keras.utils.PyDataset`, with bounded caching and row batching, then the usual BayesFlow adapter. Use the public `approximator.fit(dataset=...)` interface rather than the private workflow `_fit` method; preserve build, compile, normalization and checkpoint behavior in that bridge.

The cluster connection was not exercised, and no successful generation run or personal discovery file was found in the inspected scaffold. Current remote refs could not be refreshed in this session; the previous upstream audit remains the last verified upstream state. This proposal is grounded in the local code, not a claim that Francesco's artifacts are still absent from every current remote branch.

## 9. Acceptance and implementation sequence

**First increment: contracts plus one complete bank.** Implement GBM, a returns observation task, explicit targets, deterministic example IDs, shard write/read/verify, and a small BayesFlow training round trip. Demonstrate identical example identity after resharding and retrying. Provide a CPU simulation array using the same CLI. This delivers the cluster building block without waiting for the SVJ bundle.

**Second increment: Heston and Francesco's SVJ.** Reproduce his exact simulator, reference case and checkpoint before changing kernels. Validate variance initialization, leverage, jump counts, jump compensation and agreement of the simulator/filter/forecast transition. For new continuous-time implementations, compare a declared baseline scheme with refinement and an appropriate stronger reference. Do not impose the Feller condition merely to hide a scheme's inability to handle accessible zero variance. [Full-truncation study](https://repub.eur.nl/pub/18571)

**Third increment: HKDE and an option-panel task.** Add the double-exponential jump law and validated pricing adapter. Check compensation and moment support (with positive upward-jump probability, the upward exponential rate must exceed one for the exponential moment used in risk-neutral compensation). Match numerical prices against independent references. This tests that the architecture handles both paths and pricing-based observations without forcing identical inference targets.

**Fourth increment: Bergomi.** Validate one-factor behavior and complete forecast state, then introduce a rough reference and production scheme. Test Gaussian-driver covariance, correlation structure, forward-variance normalization and step/kernel refinement. Treat increasing approximation-factor count as numerical refinement when approximating a fixed kernel.

Across these increments, verify analytic limits, moments/covariances, declared supports, price identities where applicable, and state continuation. Test martingale expectations only on parameter domains where the specified model should be a true martingale. Compare backends statistically and against numerical error tolerances, not merely matching one seeded trajectory. Validate the prior's difficult regions as well as representative parameter points.

SBC, posterior/reference comparisons and predictive-risk diagnostics remain separate inference acceptance tests; a validated simulator does not certify a learned posterior. Conversely, a fast training loss decrease does not validate a simulator. More complex models enter the library when a concrete observation and inference task needs them, rather than because they complete a catalog.
