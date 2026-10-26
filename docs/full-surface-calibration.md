# Implementing the complete-surface tutorial

The first implementation covers **complete-surface Heston and joint rough Bergomi** on the declared 63-contract design. Sparse inputs are rejected. Sparse-chain specification and implementation remain a [separate downstream unit](../tutorial_final/SPARSE-QUOTE-CASE-PLAN.md).

The tutorial workflow is observation → posterior → diagnostics/reference comparison → product uncertainty. This PR builds the generation, training, reload, sample-based evaluation and first PyMC reference pieces. It does not yet deliver scientifically accepted checkpoints or participant notebooks.

Actual Oscar readiness, development training and bounded scientific pilots are recorded in the [26 October development report](oscar-pilot-2026-10-26.md). Read that report before choosing a large rough-bank configuration: the candidate grid and curve family have unresolved limitations.

## Work completed in parallel

| Piece | Implementation |
| --- | --- |
| New joint rough task | `rough_joint_calibration_v1`, with six ordered targets: H, eta, rho, short variance, long variance and curve decay. |
| Curve/pricing consistency | Positive exponential initial curve evaluated at each simulator grid node; expected-integrated-variance controls use the same discrete left-cell sum. Analytic integrals provide curve summaries and a refinement check. Stored seven-interval averages are auxiliary, never supplied to inference. |
| Training/inference | BayesFlow 2.1 factory for affine coupling, spline coupling, FlowMatching and MeanFlow; complete-vector, DeepSet and SetTransformer inputs. Fresh quote noise, validation checkpoint selection, protected recovery and checked reload. |
| Environments | Isolated `core`, `core-gpu` and `core-reference`; archived default/GPU/collaboration environments retain BayesFlow 2.0.14. |
| First classical reference | Deterministic Fourier Heston likelihood through PyMC's built-in DEMetropolisZ, preserving chains, sampler statistics and numeric diagnostics. No artificial gradients or stochastic likelihood calls. |

Set encoders here compare representations of complete surfaces. They do not establish support for changing masks, arbitrary raw chains or new contracts. The primary rough estimator conditions on quotes and declared design/noise only; generating curve values and coefficients are not extra inputs.

## Run the small instructor checks

Install the CPU environment and check actual backend availability:

```sh
pixi install --locked -e core
pixi run --locked -e core core-check
```

Generate a tiny Heston bank in new output directories:

```sh
pixi run --locked -e core python scripts/calibration_bank.py config \
  --model heston --train 16 --validation 4 --test 4 --seed 271026 \
  --out /tmp/heston-core-config.json
pixi run --locked -e core python scripts/calibration_bank.py local \
  --config /tmp/heston-core-config.json --chunk 8 --workers 1 \
  --out /tmp/heston-core-bank
```

Train a software smoke estimator, then evaluate on development data:

```sh
pixi run --locked -e core python scripts/train_full_surface.py \
  --bank /tmp/heston-core-bank --out /tmp/heston-core-smoke \
  --family spline --encoder vector --widths 16,16 \
  --epochs 2 --batch-size 8 --chunk-epochs 1 --seed 0
pixi run --locked -e core python scripts/evaluate_full_surface.py \
  --bank /tmp/heston-core-bank --checkpoint /tmp/heston-core-smoke \
  --out /tmp/heston-core-development --split validation --surfaces 4 --draws 32
```

Those sizes check execution, not inference quality. Rerun the identical training command to exercise recovery; changing the snapshot, family, encoder, schedule, source or environment requires a new output directory. Recovery preserves a committed state but does not promise bitwise equivalence to uninterrupted training. Evaluation defaults to validation; final test data remain reserved for a frozen acceptance campaign.

The checked-in [joint rough candidate config](../configs/tasks/calibration_rough_joint_pilot.json) contains only 8/2/2 surfaces. Use `--model rough-joint` to create another small configuration, or use that config with the same bank commands. Its independent-uniform level bounds are 0.01–0.16 in annualized variance units and decay is 0.25–8 per year; these are **development candidates**, not a scientifically approved prior. The full numerical ladder is retained, so difficult surfaces can cost more than a nominal price call. Do not multiply its counts into a large campaign yet.

For generation that grows later, use the existing [incremental collection/snapshot workflow](incremental-calibration-banks.md). Loading an old compatible Heston bank for training remains supported. Changed generator source hashes intentionally prevent silently adopting old shards into a new generator collection; do not bypass that check. Old seven-level rough data/checkpoints are not data for the new joint task.

## PyMC reference route

```sh
pixi install --locked -e core-reference
pixi run --locked -e core-reference python scripts/calibration_reference.py \
  --bank /tmp/heston-core-bank --split validation --index 0 \
  --out /tmp/heston-core-reference --draws 2000 --tune 2000 --chains 4 \
  --cores 4 --mp-context spawn --checkpoint-every 100 --seed 0
```

This uses all observed contracts and the task's declared independent Gaussian quote noise and uniform prior. Fourier precision failures abort rather than being mistaken for zero posterior density. DEMetropolisZ is gradient-free; PyMC applies the bounded-variable transforms and their Jacobians. Starts are independent prior draws, not neural posterior draws. Request at least four CPUs for the example above; the command enforces allocation limits. Spawn/forkserver supports native parallel chains without forking an initialized JAX process. Pilot mixing and cost before generating a reference collection.

Outputs include `chains.npz` with explicit chain/draw/parameter axes and sampler statistics, `reference.json` with observation/task/data/source/environment identities, and `diagnostics.json` with R-hat/ESS/MCSE. Atomic files in `partial/` retain physical draws and tuning flags during sampling; they are not resumable sampler/adaptation/RNG state. Passing statistical diagnostics alone does not establish scientific acceptance: numerical refinement, product precision and reference uncertainty must also pass. Short or incomplete chains are explicitly inconclusive. The rough emulator-MCMC route remains to be implemented and independently checked; this wrapper must not receive a fresh random Monte Carlo log likelihood.

On Oscar, install `core-gpu` for training. Run `core-check --require-gpu` **inside an allocated GPU job**; the new [GPU launcher](../cluster/core_gpu.sbatch) does this automatically. The [CPU launcher](../cluster/core_cpu.sbatch) explicitly selects `core` or `core-reference`. Both accept an absolute Pixi executable and resource overrides. The [readiness command](../scripts/check_core_cluster.py) tests actual generation, shared-storage operations and interrupted training/reload. CPU/GPU readiness now passes on Oscar, with the scope and evidence linked above. Historical training wrappers still select archived environments and should not be used for these candidates.

## Scientific work still required

1. Complete the bounded exponential-versus-published-hump curve check and agree the curve/parameter prior and product horizon before substantial rough generation.
2. Measure time-grid bias and Monte Carlo error across representative and difficult prior/posterior regions on Oscar; freeze the numerical target and throughput budget. A few surfaces passing MC standard-error checks do not establish discretization accuracy.
3. Expand immutable banks as justified by development learning curves, train selected configurations, and build converged conditional references and independently checked rough price-emulator comparisons.
4. Evaluate the frozen candidates on untouched acceptance data, propagate posterior draws into the selected product, and package/rehearse the full-surface notebooks with cached fallbacks.

The [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) and [scientific protocol](../tutorial_final/EXPERIMENT-PROTOCOL.md) remain the source of sequencing and acceptance proposals. Bounded development runs are documented in the pilot report; no large rough bank or final acceptance campaign is approved by those runs.
