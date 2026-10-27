# Implementing the complete-surface tutorial

The tutorial backbone has three routes on the declared complete 63-contract design:

| Route | Unknowns | Information supplied to inference |
| --- | --- | --- |
| Heston | Initial variance, long-run variance, vol-of-vol and rho; mean reversion is fixed. | Quotes and the declared design/noise. |
| Rough Bergomi with a known curve | H, eta and rho. | Quotes, design/noise and the seven declared forward-variance levels. |
| Joint rough Bergomi | H, eta, rho, short variance, long variance and curve decay. | Quotes and design/noise; the generating curve is not supplied. |

In the first rough lesson, the curve is **known and held fixed for each inference problem**. It can vary across training examples, as in Francesco's supplied-curve task. This is followed by the joint lesson, where curve uncertainty becomes part of the posterior. Sparse inputs are rejected; sparse-chain specification and implementation remain a [separate downstream unit](../tutorial_final/SPARSE-QUOTE-CASE-PLAN.md).

The current reproduction uses a seven-interval known curve; the joint extension uses an analytic monotone curve and finer grids. Assess each estimator against its own numerical target. A difference between their posterior widths does not isolate the effect of making the curve unknown; that would require a matched curve specification, observation and numerical target.

The shared workflow is observation → posterior → diagnostics/reference comparison → product uncertainty. The deliverable for each route is a reusable bank and checked estimator that a live tutorial can load. This PR builds the generation, training, reload, sample-based evaluation and first PyMC reference pieces; the new development checkpoints have not yet met the tutorial acceptance requirements.

Actual Oscar readiness, development training and bounded scientific pilots are recorded in the [26 October development report](oscar-pilot-2026-10-26.md). The [reviewed unit](reviewed-unit-2026-10-27.md) completed the Heston references and rough cost pilot; the [follow-up](followup-results-2026-10-27.md) checked empirical pricing errors and two further Heston fits. The [latest execution](native-default-and-bank-start.md) records the native FlowMatching sampling check and the authorized start of the first 32k joint-rough development bank. The monotone exponential curve and working numerical policy support development, with limits retained; no new estimator is accepted as a teaching checkpoint.

Heston reproduction and the known-curve rough backbone now take priority. The [28 October Heston audit](heston-replication-audit.md) found that minimum stochastic validation loss selected a weaker checkpoint: final epoch 500 passes all four rank screens and all three cached conditional screens, with no retraining. It remains a post-hoc candidate requiring a frozen selection rule and fresh acceptance/product checks. The trainer now imports BayesFlow before resetting its seed so cold and already-imported sessions initialize identically. Known-curve rough has a checked 32k bank, selected posterior/pricer and passing four-target rank screen; its three approximate-emulator comparisons expose conditional failures requiring independent pricing checks. Joint recovery preserved the existing collection's **9,925 training rows** and old failures; its remaining bank is pending. No new estimator is accepted yet. The [work order](tutorial-backbone-work-order.md) gives current results, paths and focused next steps.

## What a delivered backbone contains

Each route needs the same small set of artifacts:

1. An immutable bank snapshot with unique split-qualified IDs, clean numerical prices, the observation/noise law, numerical diagnostics and fixed holdouts.
2. A selected, reloadable BayesFlow 2.1 estimator, its exact training/sampling recipe and the snapshot identity used for fitting.
3. A comparison report covering calibration, conditional reference agreement, product uncertainty and measured generation/training/inference cost, with failures and numerical limits visible.
4. Small cached notebook fixtures bound to the checked artifacts, plus instructor commands to regenerate the expensive results. Participant use must not require cluster generation or retraining.

Existing bank, training and report formats supply these pieces. Software smoke checks and accumulating more rows are intermediate steps; the estimator and notebook artifacts must also be delivered.

## Relationship to the existing rough examples

This increment builds on substantial completed simulation and inference work. It is not the repository's first rough bank or trained network:

| Existing route | Completed work and scope |
| --- | --- |
| [Earlier main example](rough-reference-expansion.md) | Nine puts, inferred H/eta, fixed rho and known flat curve; 1,024 training bases with eight quote augmentations, an 8,192-row reference bank, and an 80-epoch trained spline checkpoint. All development gates passed; [one held-out spread gate failed](rough-heldout-test.md), so it remains provisional. |
| [Course-v2 rough pilot](course/status.md) | 2,200 surfaces, 1,600 for training; infer H/eta/rho from 63 quotes given seven curve levels. Actual neural evaluation, MC error/covariance checks, adaptive path generation, variance reduction, grid checks, audits and products exist. Its SBC criterion was partly unmet; the full 112k-bank programme is explicitly pending. |
| Known-curve backbone | Reuse the course pilot's three-target task as the first complete-surface rough lesson, with the curve supplied for each case. Recover compatible data/checkpoints and reproduce its evidence before claiming an improvement. |
| Current joint task | Infer H/eta/rho and three analytic curve coefficients from complete quotes, without supplying the generating curve. This changes the task and needs compatible prices and a new fitted estimator; old prices cannot be relabelled or projected into that distribution. |

Francesco's [runbook, R2](../tutorial_final/CLUSTER-RUNS.md), calls the bank a **discretized model** and reports a grid study without a pass/fail threshold. That study already found a resolved five-day wing shift of about one quote SD. This is not a declaration that his whole course targets a separate finite-simulator problem: its exposition also discusses model prices and numerical approximation. The [27 October plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) extends the same approximate-price workflow with a bounded check of posterior/product sensitivity to refinement. Match the observation/noise and reference conventions, reuse existing components and retain historical checks with their original scope. Numerical difficulty alone does not justify narrowing the financial domain.

The [committed course Heston estimator](../tutorial_final/checkpoints/heston/training.json) was trained on **50k unique surfaces** and has the same task as the new Heston route. The new 20k bank was a development prefix, not an adequate replacement deliverable. The first reproduction uses **at least 50k training surfaces**; the reusable bank target is **100k training surfaces**, with nested 50k/100k snapshots that retain the existing 20k prefix and unchanged holdouts. Recover compatible original assets where available and extend only missing IDs through the frozen compatible producer. Training size alone does not establish estimator quality.

Keep the archived BayesFlow 2.0.14 checkpoint as a control while reproducing its training recipe in the new workflow. The [original Heston evidence](course/status.md) includes 2,000-case SBC, 95% coverage of 93.5–95.2%, and conditional mean/90%-width agreement on 40 of 50 observations. The recent three-case comparison is a development diagnosis, not the full matching benchmark. Compare old and new on shared, task-matched observations where recoverable, then retain fresh final acceptance data. The original raw network also had conditional failures; report raw and corrected/reference results separately.

The course rough evaluation records a checkpoint hash, but that model and both raw course banks are not tracked. Recover those artifacts if available before rerunning their old tasks; the recorded ephemeral paths are not verified live Oscar locations. A reproduction comparison must preserve the prior, curve contexts, quote law and numerical target. If a new rough estimator uses a finer grid, compare it with references for that grid and report the grid effect separately; agreement with the old coarse-grid posterior is not a pricing-accuracy criterion.

The [performance pass](oscar-pilot-2026-10-26.md#pricing-performance-pass) measures optimizations on Oscar without changing the pricing target. Reproduce the focused comparisons with `pixi run --locked -e core python scripts/benchmark_rough_conditional.py --out /tmp/rough-performance` and, with BLAS/OpenMP threads set to one, `pixi run --locked -e core python tests/test_hkde_maturity_reuse.py --benchmark`. Use a new output directory for each rough benchmark. These commands run pricing checks, not participant notebooks.

## Shared implementation

| Piece | Implementation |
| --- | --- |
| Known-curve rough task | Reuse `rough_calibration_v1`, with H, eta and rho as targets and seven positive curve levels as known context. The first BayesFlow 2.1 baseline uses vector conditions. |
| New joint rough task | `rough_joint_calibration_v1`, with six ordered targets: H, eta, rho, short variance, long variance and curve decay. |
| Curve/pricing consistency | Positive exponential initial curve evaluated at each simulator grid node; expected-integrated-variance controls use the same discrete left-cell sum. Analytic integrals provide curve summaries and a refinement check. Stored seven-interval averages are auxiliary, never supplied to inference. |
| Training/inference | BayesFlow 2.1 factory for affine coupling, spline coupling, FlowMatching and MeanFlow; complete-vector, DeepSet and SetTransformer inputs. Fresh quote noise, validation checkpoint selection, protected recovery and checked reload. |
| Environments | Isolated `core`, `core-gpu` and `core-reference`; archived default/GPU/collaboration environments retain BayesFlow 2.0.14. |
| First classical reference | Deterministic Fourier Heston likelihood through PyMC's built-in DEMetropolisZ, preserving chains, sampler statistics and numeric diagnostics. No artificial gradients or stochastic likelihood calls. |

The known-curve vector baseline supplies log curve levels alongside the scaled quotes. Its four-quantity calibration screen is `evaluation.json["diagnostics_with_derived_targets"]["sbc"]`, covering H, eta, rho and rho×eta; top-level `sbc` remains parameter-only for compatibility. The joint route ignores stored auxiliary curve values and uses quotes/design/noise only. That distinction must survive training, reload, sampling and evaluation. Set encoders compare representations of complete surfaces; they do not establish support for changing masks, arbitrary raw chains or new contracts. Broader family/encoder comparisons follow delivered baselines.

## FlowMatching sampling defaults

New FlowMatching workflows use the native solver shipped with the pinned BayesFlow 2.1 environment: **adaptive Tsit5**. Omit `--flow-steps` in `scripts/train_full_surface.py` to retain that default. A positive count, such as `--flow-steps 100`, explicitly selects the earlier fixed-step Euler convention. New training reports record the resolved integration settings.

This solver reads out the trained posterior network. It transports Gaussian noise toward parameter draws, conditioned on the observed quotes. Its flow time is an artificial interpolation coordinate, not option maturity or the financial simulator's time grid; posterior sampling makes no option-pricing calls. Changing this solver can alter sampling accuracy and runtime, but cannot by itself repair an inaccurate learned posterior.

Loading an existing checkpoint preserves its recorded sampler, including explicit Euler counts. The loader checks the serialized integration settings against its metadata rather than silently upgrading a historical run. The native-solver check on the saved model uses a recorded call-time override and leaves the original checkpoint and comparison artifacts intact. Archived BayesFlow 2.0.14 estimators continue to use their original environment and sampling convention.

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

The checked-in [joint rough pilot config](../configs/tasks/calibration_rough_joint_pilot.json) contains only 8/2/2 surfaces on the older pilot grids. Use `--model rough-joint` to create another small configuration, or use that config for software checks. Its independent-uniform level bounds are 0.01–0.16 in annualized variance units and decay is 0.25–8 per year; these are development prior choices, not an empirically established market distribution. The full numerical ladder is retained, so difficult surfaces can cost more than a nominal price call.

The running bank uses the separate [32k development configuration](../configs/calibration_rough_joint_development_20k.json): 20k training, 5k validation and 7k reserved test surfaces, root seed 271200, and working grids 1280/2016/2016. Its [execution record](native-default-and-bank-start.md) identifies the existing collection and staged submissions. Grow that collection through the incremental workflow; resizing the small pilot or creating another root would not extend this bank. Reserved test outcomes remain unopened for model selection.

For generation that grows later, use the existing [incremental collection/snapshot workflow](incremental-calibration-banks.md). Loading an old compatible Heston bank for training remains supported. Changed generator source hashes intentionally prevent silently adopting old shards into a new generator collection; do not bypass that check. Old seven-level rough data/checkpoints are not data for the new joint task.

## Use the known-curve rough route

Use the existing `--model rough` bank configuration and the shared trainer with `--encoder vector`; no new generator is needed. The stored `context_values` supply the known curve during training and batched evaluation. The original pilot configuration remains a reproduction target with its recorded numerical scope, rather than a newly accepted teaching bank.

When sampling from a selected known-curve checkpoint, supply one curve row for each observed quote row:

```python
task, workflow, record = full_surface.load_checkpoint(checkpoint_directory)
draws = full_surface.sample(
    workflow, task, observed_quotes, 1000,
    context_values=known_curve_levels,  # [batch, 7]
    seed=0,
)
```

Quote arrays have shape `[batch, 63]` or `[batch, 63, 1]`. Curve levels are positive annualized variances (0.04 corresponds to 20% volatility); their logarithms are taken internally. For the joint checkpoint, curve uncertainty is inferred and the stored auxiliary levels do not enter the conditions. Known-curve training/evaluation and exact seeded reload have software checks; a scientifically accepted known-curve checkpoint still needs the work order's numerical runs.

## PyMC Heston reference route

```sh
pixi install --locked -e core-reference
pixi run --locked -e core-reference python scripts/calibration_reference.py \
  --bank /tmp/heston-core-bank --split validation --index 0 \
  --out /tmp/heston-core-reference --draws 2000 --tune 2000 --chains 4 \
  --cores 4 --mp-context spawn --checkpoint-every 100 --seed 0
```

This uses all observed contracts and the task's declared independent Gaussian quote noise and uniform prior. Fourier precision failures abort rather than being mistaken for zero posterior density. DEMetropolisZ is gradient-free; PyMC applies the bounded-variable transforms and their Jacobians. Default starts are independent prior draws. Request at least four CPUs for the example above; the command enforces allocation limits. Spawn/forkserver supports native parallel chains without forking an initialized JAX process. Pilot mixing and cost before generating a reference collection.

For the observed mixing problem, an opt-in bounded follow-up adds `--initialization data-fit --fit-starts 4 --fit-budget 2000` and uses a new output directory. It fits the transformed-space posterior from dispersed data-only starts, including the Uniform/logit Jacobian, then uses regularized local curvature for an explicit PyMC `MultivariateNormalProposal` inside DEMetropolisZ. The optimizer's finite logit search box restricts initialization only; sampling retains the full original prior. Fit budgets/results, separated supported fit locations, proposal covariance, seeds and dispersed chain starts are recorded. No neural estimate supplies initialization. This is a proposal/initialization change, not a new posterior target or a proof of global mode coverage.

Outputs include `chains.npz` with explicit chain/draw/parameter axes and sampler statistics, `reference.json` with observation/task/data/source/environment identities, and `diagnostics.json` with R-hat/ESS/MCSE. Analytic one-year variance-swap and six-to-twelve-month forward-variance rates receive separate MCMC diagnostics. They are continuously monitored variance quantities, not volatility-swap prices or checks of leverage/vol-of-vol. The guarded comparison script describes their posterior means/quantiles without inventing absolute acceptance tolerances; parameter convergence does not substitute for product convergence. Atomic files in `partial/` retain physical draws and tuning flags during sampling; they are not resumable sampler/adaptation/RNG state. Passing statistical diagnostics alone does not establish scientific acceptance: numerical refinement, product precision and reference uncertainty must also pass. Short or incomplete chains are explicitly inconclusive. This wrapper must not receive a fresh random Monte Carlo log likelihood.

## Forward price emulation and rough PyMC inference

Train a forward emulator using the same bank's **clean numerical prices**, rather than noisy observations. This requires no new bank generation:

```sh
pixi run --locked -e core python scripts/train_full_surface_pricer.py \
  --bank /path/to/checked-rough-snapshot --out /tmp/rough-pricer \
  --widths 256,256,256,256 --epochs 300 --batch-size 512 --seed 0
```

The command reads training and validation only. It writes `checkpoint/pricer.npz`, `task.json` and `pricer.json`, binding the exported NumPy predictor to the task and bank. Epoch selection uses mean squared price error in **quote-noise units**, with exact-erf GELU matching Keras. It checks export differences and exact saved/reloaded predictions. Reports show per-quote errors and the norm across all 63 contracts; errors can accumulate in a likelihood. Label MCSE is reported separately and does not measure grid bias. Old unmarked NumPy pricers retain their original tanh activation convention.

For a known-curve task, inputs are the three model parameters in logit coordinates plus seven supplied log curve levels. A joint pricer uses its six parameters and excludes auxiliary curve labels. Both outputs are the 63 numerical prices divided by their fixed quote standard deviations.

Use that fixed, task-bound emulator in the existing PyMC command:

```sh
pixi run --locked -e core-reference python scripts/calibration_reference.py \
  --bank /path/to/checked-rough-snapshot --pricer /tmp/rough-pricer/checkpoint \
  --split validation --index 0 --out /tmp/rough-emulator-reference \
  --draws 2000 --tune 2000 --chains 4 --cores 4 --mp-context spawn \
  --initialization data-fit --fit-starts 4 --fit-budget 2000 --seed 0
```

PyMC supplies the physical independent-Uniform prior and bounded transforms once; the emulator callback supplies only the deterministic Gaussian quote likelihood. The known curve is copied from the selected observation and held fixed; the joint route receives no generating curve. DEMetropolisZ, data-only initialization, chain/draw storage and partial traces are shared with Heston. Rough diagnostics additionally check rho×eta. Record the fixed emulator hashes and approximation scope alongside convergence results.

Compare a completed reference with a checkpoint through the same conditional-comparison command:

```sh
pixi run --locked -e core python scripts/compare_full_surface_reference.py \
  --checkpoint /path/to/rough-posterior-checkpoint \
  --reference /tmp/rough-emulator-reference --out /tmp/rough-emulator-comparison \
  --draws 10000 --seed 0
```

The comparison rechecks the fixed emulator files and task/observation/curve identities, and requires passing parameter and rho×eta MCMC diagnostics. The recorded emulator files must remain accessible at their bound paths. It reports approximate-target agreement, including a separate rho×eta comparison; it does not apply Heston variance products or Fourier precision reports to rough models. Inconclusive references do not produce a neural-accuracy verdict.

This is a posterior for the **fitted emulator**. Amortized training adds fresh quote noise to stored Monte Carlo price estimates; those labels also contain Monte Carlo randomness and covariance. The emulator likelihood instead centers the declared Gaussian quote law on the fitted mean prices and omits that extra pricing randomness. Even a perfect mean emulator need not reproduce the finite-Monte-Carlo simulator posterior. A fair model-price comparison therefore needs posterior-region checks showing the pricing randomness is negligible at the required accuracy, or an explicitly justified treatment of it.

A converged chain or agreement with an amortized estimator trained on the same prices does not independently validate that pricing target. Held-out interpolation errors, independent price/refinement checks in relevant posterior regions, and product uncertainty remain scientific delivery requirements; shared grid bias can cancel in emulator validation errors. The new route has software checks; a full rough reference campaign and accepted tutorial artifacts remain pending. More detail on the financial workflow is in the [forward-pricing explanation](course/forward-pricing-emulation.md).

On Oscar, install `core-gpu` for training. Run `core-check --require-gpu` **inside an allocated GPU job**; the new [GPU launcher](../cluster/core_gpu.sbatch) does this automatically. The [CPU launcher](../cluster/core_cpu.sbatch) explicitly selects `core` or `core-reference`. Both accept an absolute Pixi executable and resource overrides. The [readiness command](../scripts/check_core_cluster.py) tests actual generation, shared-storage operations and interrupted training/reload. CPU/GPU readiness now passes on Oscar, with the scope and evidence linked above. Historical training wrappers still select archived environments and should not be used for these candidates.

## Work needed to deliver the tutorials

1. **Restore the Heston baseline.** Extend the compatible bank to nested 50k/100k training snapshots, preserving its existing prefix and holdouts. Reproduce the original training recipe at 50k, diagnose the current accuracy gap and compare with the original broader calibration/conditional evidence before calling the new estimator a replacement.
2. **Deliver known-curve rough inference.** Reuse the three-target course task, recover its assets where available, and provide a compatible bank plus selected BayesFlow 2.1 estimator. Match or improve the original calibration, conditional and product evidence using task-matched references. The original rho/rho×eta calibration failures remain issues to resolve, not acceptance standards to copy.
3. **Complete the joint extension.** Preserve and check the authorized 32k collection within its existing generation ceiling, then fit one joint posterior and one forward price emulator under separately costed training limits. Carry out the predeclared numerical/posterior/product panel, including difficult cases and independently checked emulator/reference errors. The reported MCSE policy remains empirical; completion does not certify absolute pricing bias or continuous-model accuracy.
4. **Bind the results to the notebooks.** Deliver the bank/checkpoint/recipe/evidence/cache set for each route, evaluate frozen candidates on untouched acceptance data and rehearse the participant workflow. Add data or architecture/summary comparisons where they answer a measured accuracy question; keep sparse-chain work downstream.

The [central experiment plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) and [scientific protocol](../tutorial_final/EXPERIMENT-PROTOCOL.md) remain the source of sequencing and acceptance proposals. The [latest execution](native-default-and-bank-start.md) records the separately authorized first bank and its 1,500-allocated-CPU-hour generation ceiling. This is development generation, not approval of final scientific claims, wider architecture screens or sparse-chain work.
