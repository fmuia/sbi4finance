# Software and dataset contracts

The software should express the experiment defined in [foundations](01-foundations.md), [models](02-models-and-numerics.md), and [inference and pricing](03-inference-and-pricing.md). This note maps those mathematical objects to modules and stored data. The APIs and layouts are proposals to implement through the [milestones](../README.md), not existing functionality.

## Responsibilities and dependencies

Keep mathematical definitions independent of where computation runs and which network consumes its output:

```text
src/sbi4finance/
  simulators/       process laws, numerical kernels, continuation states
  observations/     return measurements, quote errors, masks and contexts
  pricing/          reference prices and numerical diagnostics
  tasks/            priors, initialization, target selection and composition
  datasets/         example identity, shards, manifests and verification
  inference/        network construction, posterior sampling and references
  surrogates/       learned parameter-to-price maps
  adapters/         mapping raw scientific arrays into BayesFlow inputs
configs/tasks/      versioned experiment specifications
scripts/            local generation, verification, training and evaluation
cluster/            execution wrappers around those same commands
```

Create modules as they acquire a concrete implementation. The numerical kernels need not import BayesFlow or Slurm. A network consumes arrays with a reviewed meaning; it need not know which CPU array job generated them. The reference pricer and learned surrogate remain separate capabilities even if they eventually accept similar inputs.

The initial registry is a small table of supported models and capabilities. A model may support paths without Fourier pricing, or pricing without the conditional continuation required by a returns task. Unsupported combinations must fail explicitly. A general plugin system and arbitrary composition of every variance/jump/payoff component would create obligations beyond the initial task.

## Five specifications

| Specification | Required meaning | Examples of decisions |
|---|---|---|
| Model | Dynamics, parameter schema, state representation and measure | Gaussian versus double-exponential jumps; P versus Q |
| Observation | Measured channels, times/contracts, noise, masks and known inputs | Log returns; price versus IV error; fixed strike panel |
| Inference task | Prior, initialization, inferred/fixed quantities and derived outputs | Joint parameters and continuation state; fixed or inferred curve |
| Numerical | Scheme, internal grid, precision, tolerances and pricing budget | Full truncation; rough hybrid scheme; MC replicates |
| Execution | Backend, resources, batch sizes, shard plan and storage | One process per CPU task; data-volume location |

Model parameters use raw named units and a fixed declared order. Transforms used by a network are recorded separately. Time convention, adjustment of observed prices, rates, forwards, dividends and curves are part of the task's scientific context, not undocumented preprocessing.

The [S/M/E delivery boundary](../README.md#synthetic-delivery-and-market-evidence) adds provenance and evidence fields to these specifications without replacing the common schema. S uses generated observations and may use hypothetical curves, contracts and forward-variance inputs. Optional M applies a compatible trained network to actual observations; optional E introduces separately held-out market outcomes/quotes. A provider or market fixture is not a dependency of S.

The model and numerical specification together define the law actually simulated. Keep a semantic task identifier, a numerical-generator identifier and a bank manifest hash. Changing shard size can change the file inventory while leaving the example identities and values unchanged. Changing an integrator changes the numerical-generator identity even when it approximates the same continuous model. Changing priors, targets or observation noise changes the statistical experiment.

Although execution selects a backend, the numerical-generator fingerprint must include that backend, the RNG implementation, precision and relevant implementation versions. Worker assignment, shard size and storage location are excluded from example identity. Moving the same implementation between workers should reproduce examples; switching from NumPy to a different RNG or compiled kernel requires separately recorded identity and compatibility evidence.

An initial-state law can depend on the drawn parameters. It must be resolved before simulating a path and used consistently in reference inference. For full-truncation Euler, distinguish nonnegative effective variance from the possibly negative retained auxiliary state. Record the state required to continue the discrete transition exactly. A generic field named `state` can contain named arrays; it need not always be one positive scalar.

## Kernel and experiment boundaries

Conceptual function contracts are:

```python
simulate_paths(parameters, *, model, initial_state, grid, numerics, streams)
continue_paths(parameters, *, model, state, grid, numerics, streams)
price_contracts(parameters, *, model, market, contracts, numerics, streams)
generate_examples(task, *, example_ids)
```

These names show responsibility rather than freeze a final Python signature. The first three receive explicit parameter values. They never silently sample a global prior. The last function owns experiment generation: sample parameters/context/initial state, invoke the appropriate process or pricer, produce observations, extract targets, and collect diagnostics.

The conceptual `market` argument means pricing context, not necessarily acquired market data. It may contain a declared synthetic spot, curves and other known inputs. Preserve component provenance when mixing synthetic observations with market-derived context; that mixture alone is not an application to actual market observations.

For a return example, generate one latent process realization, aggregate at observation times and apply any declared measurement mechanism. For an option example, evaluate a set of pricing expectations and apply a quote-error model. Numerical pricing paths remain internal. For a surrogate example, parameters and contracts are inputs and the reference price is the label. A direct price-posterior example pairs a noisy observed panel with a reference target price generated from the same underlying parameters.

A reference price estimates a conditional expectation with a recorded numerical error budget; a single simulated payoff is not a price label. A future simulated return can be a predictive evaluation outcome, but is never part of the historical conditioning window.

Continuation must consume the complete state, matched to its posterior parameter draw and the correct time origin. Exact or statistically controlled agreement between split and uninterrupted simulations is a useful test. For rough models, a path-generation API alone does not supply an inference algorithm for the hidden memory.

## Raw arrays and network inputs

Each bank uses one declared schema. A batch of $B$ **labeled synthetic training or validation examples** has the following conceptual structure:

| Field | Shape or content | Network role |
|---|---|---|
| `example_id` | $[B]$, stable integer identity | Provenance only |
| `parameters` | $[B,d_\vartheta]$, raw parameter units | Selected inference targets or surrogate inputs |
| `observations` | $[B,n,d_y]$ for a return window, or $[B,m,d_y]$ for a quote panel | Observed conditioning |
| `context` | Named times, contract coordinates, curves and known inputs | Only declared observed conditioning |
| `mask` | Explicit validity of elements/channels | Conditioning and batching semantics |
| `target_state` | Named state arrays required by the task | Joint inference targets |
| `diagnostics` | Solver flags, pricing standard errors, refinement metadata | Validation only unless explicitly modeled as observed context |

The distinction between diagnostics and context is deliberate. An error estimate calculated using the true parameters during simulation may reveal information unavailable for real observations. Feeding it into the network would change the inference task. The same applies to true terminal variance, latent jumps and internal Brownian increments.

An optional market-conditioning fixture reuses the observation/context/mask definitions and provenance fields, but omits unavailable `parameters` and `target_state` labels. It must not manufacture zeros or other placeholders that could be mistaken for truth. Market volatility proxies and vendor estimates are observations or derived quantities with their own definition, not true latent states. The manifest declares which labels exist and their origin.

Future evaluation outcomes and excluded target quote values belong to separate evaluation records linked by example/origin/contract identity. They cannot enter encoders, normalization, smoothing, curve/noise estimation or other fitting steps for the evaluated conditioning set. Provenance and evidence metadata also do not become network inputs merely because they are stored alongside arrays.

Preserve singleton axes. A single example should not unexpectedly lose its batch dimension. Fix sequence length and contract grid in the first task versions. Later, bucket lengths or use padding with masks; a mask-aware training design is needed before claiming support for arbitrary missingness. If missingness depends on the latent process, that mechanism is part of the observation model.

Store raw scientifically meaningful arrays. Apply support transforms, normalizations and encoder mappings in a versioned adapter. Estimate normalizations using training data only. A checkpoint must specify its parameter ordering, target shape, adapter, architecture, supported context and training bank; a file of weights alone is insufficient.

## Examples of task definitions

These examples show the fields to settle, without inventing production priors or budgets.

Both tasks can use fully synthetic observations and context. Their market-application versions must match the trained support, observation convention and adapter; loading an actual asset/date normally changes conditioning inputs, not network weights.

| Field | Returns task | Options task |
|---|---|---|
| Model | Declared Heston/Gaussian-jump law under P | Declared HKDE law under Q |
| Observation | Ordered decimal log returns at explicit year fractions | European prices on explicit contracts with known curves |
| Initialization | Fixed, inferred or sampled variance/state law | Known or inferred current variance |
| Inferred quantities | Selected static parameters and sufficient current state | Selected Q parameters, any uncertain initialization/noise inputs |
| Prior | Reviewed distribution over those quantities | Reviewed distribution respecting moments required by pricing |
| Numerics | Recorded transition, substeps and state convention | Validated pricing engine and grid/tolerances |
| Financial output | Conditional future loss distribution | Posterior distribution of model prices for target contracts |

The numerical grid is not a neural target. A fixed numerical setting can remain metadata. A deliberate model that conditions on fidelity or learns corrections is a separate task, with its own validation.

## Randomness and example identity

Use a fixed RNG algorithm and derive independent streams from stable identifiers, such as `(purpose_id, example_id, split_id, task_id, root_seed)`. Purposes include parameter sampling, context, initialization, process shocks, quote errors and evaluation futures. Use an explicit stable encoding for task identities rather than Python's randomized `hash()`.

NumPy's `SeedSequence` supports combining integer identities into reproducible streams with extremely small collision risk. The design requirement here is stronger than simply giving each worker a seed: an example must retain its stream when it moves to another worker or shard. [NumPy RNG guidance](https://numpy.org/doc/stable/reference/random/parallel.html)

First keep serial stream consumption within an example and parallelize independent examples through jobs/processes, or provide pre-generated random blocks to a parallel kernel. Numba's supported NumPy `Generator` interface is documented as not thread-safe, so one mutable generator must not be shared across parallel iterations. [Numba RNG support](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#generator-objects)

Record reproducibility within a pinned backend/environment. Different compilers, numerical methods or reduction orders may not be bitwise identical. Compare those implementations through scientific tolerances and distributional checks. For a refinement experiment, specify how coarse and fine randomness is coupled; reusing a seed alone does not guarantee matched Brownian paths.

## Shards and restartable generation

An initial bank layout is:

```text
banks/<bank-id>/
  spec.json
  manifest.json
  train/shard-000000/
    example_id.npy
    parameters.npy
    observations.npy
    target_state.npy           # if needed
    ... context and mask arrays ...
    metadata.json
  validation/...
  test/...
```

A shard holds many examples. Plain `.npy` arrays permit memory mapping without a storage dependency; JSON describes their meaning and provenance. Shard sizes are chosen from measured memory and I/O behavior. NPZ is convenient for small fixtures but does not provide ordinary per-array memory mapping. The library need not adopt a database or object-storage service to produce its first bank.

The manifest records schema and model versions, canonical task specification, source revision or recorded patch, lockfile hash, numerical settings, seeds, ID ranges, split groups, checksums and diagnostics. Save both the intended number of examples and the actual status of every requested ID.

Extend that manifest and its linked experiment/result record with the following meanings. Field names are proposed; the information is required:

| Additional information | Meaning and boundary |
|---|---|
| Data origin | Synthetic, market or documented mixture, with provenance for observation channels and context separately |
| Purpose | Training, tuning, validation/test, conditioning or evaluation, independently of data origin; separate artifacts/roles where needed |
| Task and measure | Returns under P, quote calibration under Q, pricing regression or direct price posterior, linked to the existing task specification |
| Conditioning boundary | Observation time range and information cutoff, or included contract IDs and market timestamp; preprocessing lineage and excluded evaluation selections |
| Available truth labels | Which parameter/state, synthetic future or reference-price labels exist and how they were generated; absent for unknown real parameter/state truth |
| Evidence scope | S, M or E on the reported experiment/result, with supported claims and evaluation design; a market file does not automatically establish empirical evidence |
| Source/parent identity | Source fixture or generating bank, parent surface/path and split group, transformations and checksums |

A synthetic reference-price label is an approximation to the declared model's expectation, not a known fair market value. Keep its numerical diagnostics. If an optional fixture uses actual curves with generated observations, label both components; the required S fixture must also be runnable with generated context and no market artifacts.

A worker writes to a temporary location, validates it, then publishes by atomic rename on the same filesystem. A retry accepts a completed shard only if its specification and checksums match; otherwise it reports or regenerates the same IDs. An incomplete temporary shard never becomes training data by directory globbing.

The final verifier rejects missing or duplicate IDs, inconsistent shapes, corrupted arrays, incompatible specs and leakage across splits. Numerical failures retain their original parameters and reasons in diagnostic records. They are not silently replaced by new prior draws. Fix the method or formally revise the supported experiment before using a selected bank.

Split independent base experiments before creating repeated quote-noise versions, overlapping windows or other related augmentations. For a price surrogate, related contracts at one underlying parameter draw should be grouped so validation does not merely repeat near-identical training cases. Maintain additional parameter-region holdouts when testing extrapolation is part of the scientific question.

Optional E uses a separate chronological/contract holdout design. Account for dependence between overlapping return windows and forecast horizons; many chronological origins/dates are needed for broad performance claims. Excluded quote values cannot influence panel smoothing or other calibration preprocessing. One actual panel supports a case study, not a general backtest or recovery of unknown market parameters.

## Cluster workflow and training

The proposed CPU array selects predefined example-ID ranges and invokes the same generation command used locally. The laptop connection submits and monitors work; large banks are generated and read on the cluster data volume. Transfer only the small fixtures, diagnostics and checkpoints required locally.

Generation and training consume synthetic task specifications, not provider credentials or acquired quotes. After artifacts are prepared, the required participant S notebook runs offline from synthetic fixtures/checkpoints, including its downstream forecast/pricing exercise. An acceptance rehearsal removes all market fixtures and disables provider/network access. Optional M/E cells must report missing inputs explicitly and never relabel synthetic data as real.

Verify cluster access, allocation, storage and environment before a smoke job. Cap concurrency and total threads: processes, Numba, BLAS and FFT libraries can otherwise multiply the requested CPU count. A separate training job starts only from a verified bank. A pilot measures runtime, memory, write/read throughput and failure rates before choosing full-scale resources.

For a rough workload estimate, suppose $N$ examples cost $t_e$ seconds each on one core. Ideal compute time on $C$ independent cores is $Nt_e/C$, before startup, imbalance, I/O and scheduling. At a purely illustrative 10 ms per example, one million examples require about 2.8 core-hours; at one second they require about 278 core-hours. Actual pricing tasks must be benchmarked, because a surface may contain many internal MC paths.

One million 252-day float32 return windows require $10^6\times252\times4=1.008\times10^9$ bytes for that observation channel alone, about 1.0 decimal GB. Targets, masks, context and metadata add storage. Retaining every high-resolution path for every price label would usually dominate this budget.

The inspected scaffold's [trainer](https://github.com/fmuia/sbi4finance/blob/9a3abf86902ec72aa859bf060b3ec05edd42581d/scripts/train_univariate.py) generates train/validation dictionaries in RAM and calls `fit_offline`. Its [cluster discovery](https://github.com/fmuia/sbi4finance/blob/9a3abf86902ec72aa859bf060b3ec05edd42581d/cluster/discover_cluster.py) already distinguishes CPU generation and GPU training. M2 supplies the missing array, manifest and loader components.

In the inspected BayesFlow 2.0.14 source, [DiskDataset](https://github.com/bayesflow-org/bayesflow/blob/v2.0.14/bayesflow/datasets/disk_dataset.py) loads each file as one example and stacks files. Passing a multi-example shard directly would add an incorrect batch axis. Use a thin shard-aware `keras.utils.PyDataset` with bounded caching and row batching, passed to the public [approximator dataset interface](https://github.com/bayesflow-org/bayesflow/blob/v2.0.14/bayesflow/approximators/approximator.py). Small banks can retain the existing in-memory path. The bridge must also preserve training-only normalization, compile/build behavior and checkpoint reloads.

## What to reuse from SSMS

Borrow the separation of model catalogs, Python validation, efficient kernels and generation pipelines from [ssm-simulators](https://github.com/lnccbrown/ssm-simulators). Its domain-specific observation and likelihood-training conventions should not define finance examples. A return history is a dependent sequence; a pricing panel represents expectation-based measurements.

Use the HSSMSpine pipeline as a design reference for generation/training separation and data-to-checkpoint lineage. Keep the finance package independently installable. Cython is an available implementation route, but should be justified by measurements or a concrete reusable kernel rather than adopted merely because SSMS uses it.

## A validation sheet for every task

Before its large bank is generated, record:

| Field | What must be specified |
|---|---|
| Supported experiment | Model, measure, prior, context domain, targets, initialization and numerics |
| Origin and scope | Observation/context provenance, artifact purpose, available truth labels, S/M/E claims, and the conditioning cutoff or included/excluded contracts |
| Reference cases | Ordinary, boundary and difficult regions; independent generation seeds |
| Numerical tests | Moments/covariances, limits, state continuation, refinement and pricing identities as applicable |
| Statistical tests | Conditional references, joint dependence, SBC and predictive functionals |
| Error scales | Absolute/relative price tolerances, MC uncertainty with justified moment/dependence assumptions, posterior sampling error and the intended quote-noise scale |
| Failure decision | Repair, narrow explicitly supported scope, or defer the task |
| Resource report | Cold/warm compute, serialization, memory, training and deployment costs |
| Optional empirical assessment | Held-out later outcomes or contemporaneous excluded quotes, dates/origins, leakage controls, dependence and claim limits; not an S prerequisite |
| Offline delivery | Full chosen S workflow succeeds with every real-market fixture absent and provider access disabled; optional M/E output is clearly labeled |

Use both matched-discretization reference cases to assess inference and higher-accuracy cases to assess numerical bias. Separate training, tuning and final evaluation using independent base experiments and grouped derivatives. Numerical correctness, conditional inference validity and empirical model assessment answer different questions. Dataset integrity or synthetic calibration cannot establish real-market model adequacy; market acquisition cannot replace synthetic truth-based checks or certify an implementation.

Implementation begins with [M0](../milestones/00-contracts.md) and [M1](../milestones/01-gbm-reference.md); [M2](../milestones/02-banks-and-cluster.md) turns the accepted example into a reproducible cluster workflow.
