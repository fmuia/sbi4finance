# Heston noise scale investigation and collaborator handover

## Investigation brief

Investigate why the amortized posterior barely learns the shared noise multiplier **c**, although the five Heston parameters recover strongly. Separate **statistical information about c** from **failure of this neural approximation to use it**. Non-identifiability is a hypothesis to test, not the current conclusion: joint MCMC substantially contracts c in two synthetic examples, both generated at c=1. Recovery across different generating c values is the next important check.

The completed source/evidence snapshot is commit `a2dd7d673dd121fb0196fc16991aa5153ab6273c`, branch `feat/heston-spread-baseline-20261109`, [PR #85](https://github.com/fmuia/sbi4finance/pull/85). This handover adds a portable copy of the existing development inputs; it does not add training or MCMC results. All paths and commands below are relative to **`lab/`**, unless explicitly prefixed with `tutorial/` or described as repository-relative.

| Evidence | What it establishes |
| --- | --- |
| Original 20k-bank, 60k-update network | Median central-90% c width 1.336, versus prior width 1.35; c recovery RMSE .4452, versus .4371 for always returning the prior median |
| 100k-bank, 500 extra passes, batch 128 | 50 million extra examples; final c width 1.34853, truth–median correlation −.04614, RMSE .44705 |
| Same larger bank/exposure, batch 512 | Final c width 1.34545, truth–median correlation −.03848, RMSE .44788 |
| Joint MCMC, broad synthetic, true c=1 | c 5%/median/95%: .9381 / 1.0974 / 1.3152 |
| Joint MCMC, market-like synthetic, true c=1 | c 5%/median/95%: .9702 / 1.1406 / 1.3645 |
| Actual-market joint MCMC, unknown c | Concentrates at upper prior bound 2; model/prior adequacy diagnostic, not known-truth recovery |

The larger-bank runs improved validation loss and physical-parameter recovery without learning c. Good pooled marginal SBC can coexist with ignoring observations: returning the prior is a marginally calibrated procedure under prior-predictive sampling. The two existing synthetic MCMC examples show available scale information locally; they do not establish recovery throughout the prior. Read [the original report](report.html), [the larger-bank report](capacity-study/report.html) and [its exact comparison summary](capacity-study/summary.json).

## Statistical model and parameter order

For one surface with 54 fixed contracts,

`Y_i = P_i(phi) + c * h_i * Z_i`, with independent `Z_i ~ Normal(0,1)` conditional on c.

`h_i` is the frozen observed price half-spread, `(ask_i-bid_i)/2`. One dimensionless c is shared by all quotes in a surface. c varies independently between simulated surfaces. There are no learned per-quote scales, additional tau/vega noise terms, changing contract locations or changing spread profiles.

| Column | Parameter | Independent Uniform prior in physical coordinates |
| --- | --- | --- |
| 0 | `initial_variance` (v0) | [.01, .16] |
| 1 | `kappa` | [.1, 10] |
| 2 | `theta` | [.01, .16] |
| 3 | `vol_of_vol` | [.1, 3] |
| 4 | `rho` | [−.95, −.1] |
| 5 | `c` | [.5, 2] |

The normalized log likelihood is

`log L = -.5*sum(((y-P(phi))/(c*h))**2) - sum(log(c*h)) - 54/2*log(2*pi)`.

Keep the `-54*log(c)` normalization. Omitting it changes inference about c. The joint posterior infers all six parameters; an oracle that fixes phi at its generating truth answers the different question `p(c | phi_truth,y)`.

The neural input is the ordered 54-vector `Y/h`, followed by BayesFlow feature standardization. c is a target only. Generating physical parameters, c, clean prices and realized noise are not inference conditions. Neural `sample()` accepts **raw prices**, so do not divide by h before calling it.

## Implementations to inspect

| Purpose | File and callable, relative to lab/ |
| --- | --- |
| Task, prior support, normalized likelihood, simulation | `src/sbi4finance/calibration/heston_spread.py`: `HestonSpreadPilot`, `log_likelihood`, `simulate` |
| Known-phi c oracle | Same module: `conditional_c_log_density`, `conditional_c_mode`; accept residuals divided by h. Instance methods also accept physical parameters and observations |
| Adaptive checked pricing | `src/sbi4finance/calibration/heston_pilot.py`: `HestonPilot.prices`, `_checked_prices`, `_rung_prices` |
| Authoritative Fourier pricer | Repository-relative `tutorial/src/sbi4f_tutorial/heston_options/model.py`: `characteristic_function`, `european_price`, `_european_price` |
| Joint task, augmentation, flow, posterior sampling | `scripts/train_heston_spread_scale.py`: `make_task`, `validate_task`, `draw_c`, `augmented_batch`, `build_workflow`, `sample` |
| Input scaling | `scripts/train_heston_spread_baseline.py`: `condition_data` |
| Paired row order and fresh c/noise by bank pass | `scripts/continue_heston_spread_scale.py`: `epoch_order`, `epoch_data` |
| Corrected continuation and latest checkpoint loader | `scripts/continue_heston_spread_scale_precision.py`: `host_adapt_training`, `load_checkpoint`, `run_continuation` |
| Exact 512-dataset population | `scripts/evaluate_heston_spread_scale.py`: `population_observations` |
| Latest evaluations and recovery metrics | `scripts/evaluate_heston_spread_scale_precision.py`: `evaluate`; `scripts/evaluate_heston_spread_scale_continuation.py`: `recovery_summary` |
| Data-only MCMC initialization | `src/sbi4finance/calibration/reference.py`: `data_fit_initialization`; implementation in `reference_initialization.py` |
| Native PyMC sampler and diagnostics | `src/sbi4finance/calibration/reference.py`: `sample_deterministic_reference`, `chain_diagnostics` |
| Existing reference runner and numerical checks | `scripts/run_heston_spread_reference.py`: `run_reference`, `load_case`, `diagnose`, `price_audit` |
| Original fixed/joint-c matrix and known-phi oracle | `scripts/run_heston_spread_scale_reference.py`: `job_spec`, `fixed_phi_oracle` |

The flow is a six-block spline coupling network with two 128-unit hidden layers per conditioner. Effective dropout is **zero**. The constructor's `.05` is superseded by the saved zero-dropout control. Both corrected larger-bank branches restore model, Adam and normalization state from the same six-target 60k parent. They restart at 5e-5, cosine-decay by examples to 5e-6 over 400 passes, then hold that floor through 500 passes. Batch 128 gets 391,000 additional optimizer updates; batch 512 gets 98,000. This is an equal-example comparison, not equal optimizer updates or an isolated test of bank size.

Learning curves are in `training/learning-curve.json` for the original fit and `capacity-study/precision-replay/training/batch-{128,512}/learning-curve.json` for the corrected continuations. Their adjacent `run.json`, `initial-probes.json`, `training.json` and the frozen [continuation protocol](capacity-study/precision-replay/protocol.json) record the construction, initial loss, completion and schedules. Read c recovery alongside aggregate loss; improving joint density loss alone has not indicated learning of c here.

## Environment and portable recovery inputs

Check out `docs/heston-c-collaborator-handover-20261110` for this handover and its portable inputs. It includes the completed Heston source from PR #85 and the sibling `tutorial/` package. Install the locked **core-reference** environment from `lab/`; the default historical BayesFlow 2.0 environment is unsuitable for these models.

The handover branch supplies the README, portable inputs/manifest and saved neural recovery quantiles directly. If using the separately shared handover ZIP, unpack that overlay at the repository root. Source code and the locked environment come from the checkout; model weights are separate local artifacts.

```sh
cd /path/to/sbi4finance/lab
pixi install --locked -e core-reference
export PYTHONPATH=.:src:../tutorial/src
export KERAS_BACKEND=jax
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
c_probe_cache="$(mktemp -d "${TMPDIR:-/tmp}/heston-c-probe.XXXXXX")"
export PYTENSOR_FLAGS="base_compiledir=$c_probe_cache,compiledir=$c_probe_cache/compiled"
export NUMBA_CACHE_DIR="$c_probe_cache/numba-native"
export MPLCONFIGDIR="$c_probe_cache/matplotlib"
```

The lock pins Python 3.12, BayesFlow 2.1.0, Keras 3.15.1, JAX .11.1, NumPy 2.5.3, SciPy 1.18.1 and PyMC 6.3.2. Explicit writable PyTensor **base and compiled** paths avoid the startup failures recorded in the original campaign.

[recovery-inputs.json](recovery-inputs.json) contains the complete task/geometry, parameter order, products, source identities and payload checksum. [recovery-inputs.npz](recovery-inputs.npz) is a 443 KB export of the **same 512 development observations** used by the original and both final neural evaluations:

| Array | Shape | Use |
| --- | --- | --- |
| `quotes` | (512, 54), float64 | Raw observed prices, supplied to either inference method |
| `truth` | (512, 6), float64 | Generating values for selection and recovery reporting only |
| `clean_prices` | (512, 54), float64 | Known-phi oracle and numerical diagnostics only |
| `row_id` | (512,) | Stable validation-row identifiers |

The export reads only the first 512 `validation_parameters` and `validation_clean` members of the preserved bank, and reproduces the original c/noise namespaces. It opens no reserved test arrays. It makes no pricing calls and creates no new observation realizations. The whole bank and model payloads are unnecessary for the MCMC API below.

Load and verify the inputs; the following Python blocks can be run consecutively in one script or notebook under `.pixi/envs/core-reference/bin/python`:

```python
from pathlib import Path
import hashlib
import json
import numpy as np
from scripts import train_heston_spread_scale as neural
from scripts.run_heston_spread_reference import values_digest

E = Path("evidence/pilot-results/2026-11-09/heston-spread-baseline/noise-scale")
spec = json.loads((E / "recovery-inputs.json").read_text())
payload = E / spec["payload"]
assert hashlib.sha256(payload.read_bytes()).hexdigest() == spec["payload_sha256"]
task = spec["task"]
pilot = neural.validate_task(task)
with np.load(payload, allow_pickle=False) as data:
    quotes, truth, clean, row_id = [data[k].copy() for k in
                                  ("quotes", "truth", "clean_prices", "row_id")]
assert values_digest(quotes) == spec["observations_sha256"]
assert values_digest(truth) == spec["truth_sha256"]
assert values_digest(clean) == spec["clean_prices_sha256"]
assert pilot.names == spec["parameter_names"]
# Eight c-stratified cases, selected before looking at MCMC outcomes.
selected = np.argsort(truth[:, 5], kind="stable")[[32, 96, 160, 224, 288, 352, 416, 480]]
print([(int(i), str(row_id[i]), float(truth[i, 5])) for i in selected])
```

These are reused development data, not an untouched acceptance set. Different c values have different physical parameters and noise realizations here. A separate paired experiment at fixed phi would answer an additional question.

## Run joint MCMC on a selected observation

The direct API below uses only the portable task and raw observed prices. It avoids the absolute paths embedded in the historical campaign panels. It initializes from the likelihood and prior, never from truth or the neural posterior. Call `fit_row(int(selected[0]))` for one case; call the same function on the remaining selected rows for the small recovery panel. Give any rerun a new output root.

```python
from sbi4finance.calibration import reference
from scripts.run_heston_spread_reference import Budget, diagnose, price_audit

def fit_row(row, root=Path("../local-artifacts/heston-c-collaborator-v1"),
            tune=2000, draws=2000, seconds=900):
    out = root / f"row-{row:04d}"
    out.mkdir(parents=True, exist_ok=False)
    seed = 271110 + int(row)
    y = quotes[row].copy()
    budget = Budget(seconds)
    with budget:
        def likelihood(point):
            budget.check()
            return pilot.log_likelihood(point, observations=y)
        def callback(**kwargs):
            budget.check()
        init = reference.data_fit_initialization(
            likelihood, pilot.bounds, 4, seed, fit_starts=4, fit_budget=2000)
        idata = reference.sample_deterministic_reference(
            likelihood, pilot.bounds, pilot.names, draws=draws, tune=tune,
            chains=4, cores=1, seed=seed, progressbar=False,
            initialization=init, callback=callback)
        chains = np.asarray(idata["posterior"]["parameters"].transpose(
            "chain", "draw", "parameter").values)
        diagnostics = diagnose(chains, pilot.names, spec["products"])
        checks = [price_audit(pilot, chains[c, d], y)[1]
                  for c in range(4) for d in (0, draws-1)]
        ready = (diagnostics["mcmc_diagnostics_passed"] and
                 all(p["pointwise_gates_passed"] for p in checks))
        stats = {"sample_stats_" + name: value.transpose("chain", "draw").values
                 for name, value in idata["sample_stats"].data_vars.items()
                 if set(value.dims) == {"chain", "draw"}}
        np.savez_compressed(out / "chains.npz", parameters=chains, **stats)
        record = dict(row=int(row), row_id=str(row_id[row]), seed=seed,
                      tune=tune, draws=draws, chains=4, cores=1,
                      budget_seconds=seconds, sampler="PyMC DEMetropolisZ",
                      input_sha256=spec["payload_sha256"], task=task,
                      observations_sha256=values_digest(y),
                      reference_ready=bool(ready), initialization=init,
                      diagnostics=diagnostics, numerical_checks=checks)
        (out / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    return out

# Execute explicitly when ready to sample:
# first_result = fit_row(int(selected[0]))
```

This is a starting diagnostic run, with four sequential native PyMC **DEMetropolisZ** chains and independent physical-coordinate Uniform priors. PyMC supplies bounded transforms and their Jacobians. It is not NUTS: the deterministic Fourier likelihood is wrapped as a black-box Op without a validated gradient. In the completed two-case references, 2k tuning/2k retained draws per chain were insufficient and the declared second attempt used 6k/12k. If the first run is inconclusive, declare a new output root and explicit budget before requesting that larger run; do not combine unconverged attempts.

Retain `(chain, draw, parameter)` structure. Readiness requires R-hat <=1.01 and bulk/tail ESS >=400 for **all six parameters and both variance products**, plus sampled pricing refinement. The price-difference tolerance is `.02*(.5*h)`, independent of inferred c; absolute log-likelihood refinement change must be <=.1. A timeout or failed convergence is inconclusive, not evidence of non-identifiability. The compact example saves completed traces and diagnostics; the production runner additionally records versions/source identities and partial traces on failure.

For the two original synthetic cases, the existing supervised runner can also be called directly on the original machine:

```sh
.pixi/envs/core-reference/bin/python scripts/run_heston_spread_reference.py run \
  --panel evidence/pilot-results/2026-11-09/heston-spread-baseline/references/panel.json \
  --case synthetic-broad --infer-c --seed 261112 \
  --budget-seconds 900 --fit-budget 2000 \
  --out ../local-artifacts/heston-c-original-broad-v1/summary \
  --payload-out ../local-artifacts/heston-c-original-broad-v1/payload
```

Use `synthetic-market-like` for the second observation. `--infer-c` is essential; `--c 1` alone fixes c. Both output directories must be new. This historical CLI verifies absolute declaration/design paths in sibling checkouts and will need those original files staged on another machine. Do not edit its frozen manifests or disable hash guards. The portable API above is the simpler route to a new, separately identified investigation.

## Compare MCMC recovery with the saved neural results

No model loading is needed for comparison on these 512 observations. For each batch, read

`capacity-study/precision-replay/evaluations/batch-{128,512}/pass-0500/recovery.npz`

under E. Keys are `truth` (512,6), `row_id` (512,) and `quantiles` (3,512,6), ordered **5%, median, 95%**. Assert the truths and row identifiers match the portable inputs before subsetting. For each ready MCMC trace, compute

```python
record = json.loads((first_result / "run.json").read_text())
assert record["reference_ready"], "Keep an unconverged/failed case inconclusive"
with np.load(first_result / "chains.npz", allow_pickle=False) as data:
    chains = data["parameters"]
mcmc_quantiles = np.quantile(chains.reshape(-1, 6), [.05, .5, .95], axis=0)
```

Make six generating-value versus posterior-median panels, with `y=x`, and plot central-90% intervals for **every selected case**. Offset the two neural series and MCMC slightly for readability, retaining the original true values for metrics. Keep failed/capped cases visible as inconclusive rather than dropping them silently. Report c RMSE, interval widths and truth inclusion for this subset, and show per-case diagnostics. Eight selected cases are an illustration, not a population SBC or precise coverage estimate. A known-phi oracle can be overlaid separately using

```python
row = int(selected[0])
residual = (quotes[row] - clean[row]) / np.asarray(task["records"]["half_spread"])
from sbi4finance.calibration.heston_spread import conditional_c_log_density
c_grid = np.linspace(.5, 2., 1001)
oracle_density = np.exp(conditional_c_log_density(c_grid, residual))
```

The oracle uses clean prices only for this diagnostic. Compare it with the full six-parameter MCMC marginal; do not feed clean prices or truth to either inference method.

## Payloads and the next diagnostic decisions

Completed joint reference summaries are in `recovery-v2-references/{synthetic-broad-c-joint,synthetic-market-like-c-joint,market-c-joint}/{run,diagnostics,posterior}.json`. Full chain paths and SHA-256s are in each run record. They remain ignored local payloads.

On the original machine, let `P = Path("../local-artifacts/heston-spread-baseline-20261109/noise-scale")`. The original network is `P/"training/checkpoints/update-060000"`; the corrected final models are `P/f"capacity-study/precision-replay/batch-{batch}/checkpoints/pass-0500"`. To sample the latter:

```python
from scripts import continue_heston_spread_scale_precision as continuation
P = Path("../local-artifacts/heston-spread-baseline-20261109/noise-scale")
workflow, metadata = continuation.load_checkpoint(
    P / "capacity-study/precision-replay/batch-128/checkpoints/pass-0500")
draws = neural.sample(workflow, metadata["task"], quotes[selected[:2]],
                      count=512, seed=261112)  # (2,512,6), raw-price inputs
```

Use this **precision** loader for the latest checkpoints; the original trainer loader expects another format. Transfer `checkpoint.json`, `model.keras` and `state.npz` together for exact-state loading. Latest model SHA-256s are `d59fe17e717670bc840adc4dcaed4fac76437f4ed6950572e06295ddafa58805` (128) and `771d3d883e8f54de1962d0bfa2d822a9157ee7f43bead8fe1fc6364756b08697` (512). The checkpoints and full clean bank are outside Git, with no published download URL recorded for this study. Native loaders also validate absolute bank/protocol/source bindings; relocation needs a separately recorded approach. Cloning the branch alone does not make those loaders portable. The supplied recovery quantiles and input export suffice for the requested matched MCMC comparison.

Two implementation findings should guide further investigation:

- **Fixed numerical bug:** original continuations both stopped at a valid c=1.9999999587495068 because float32 bounded-target preprocessing rounded to the endpoint. The corrected `host_adapt_training` evaluates `log(x-lower)-log(upper-x)` in NumPy float64, then casts. No clipping, redraws or prior changes were used. Both corrected runs restarted from the same original parent and completed. Use `precision-replay` evidence, not the failed or preliminary continuation outputs.
- **Untested conditioning hypothesis:** [the training-only covariance check](capacity-study/diagnostics/input-conditioning.json) finds 98.23% of clean variance in three directions, noise dominating 33/54 smaller directions, and expected standardized noisy covariance condition number about 2.14 million. Input casting error is far smaller than the noise. These are global prior-predictive geometry diagnostics; they neither establish local non-identifiability nor prove that whitening fixes learning. No whitening experiment has been trained.

First establish whether converged MCMC recovers c on the same observations across its prior range. Then inspect whether the network responds when c/noise changes at fixed phi, and audit the target transform, input standardization and c-dependent learning signal. Only after those checks choose a new representation, optimizer or training experiment. Return the selected row IDs, exact targets/seeds, all chain diagnostics, six recovery panels, c marginal comparisons and a clear distinction between model information and approximation error. The revised grid and Bergomi are separate workstreams.

For focused implementation checks, run the existing tests rather than changing the frozen producers in place:

```sh
.pixi/envs/core-reference/bin/python -m unittest discover -s tests -p test_heston_spread.py -v
.pixi/envs/core-reference/bin/python -m unittest discover -s tests -p test_heston_spread_scale_training.py -v
.pixi/envs/core-reference/bin/python -m unittest discover -s tests -p test_heston_spread_scale_precision.py -v
```

The first covers likelihood/simulation and the conditional-c oracle; the others cover the six-target training contract and precision-safe state restoration. Code changes need new source/protocol identities and the repository's lab test/release checks. Keep the completed runs immutable.

The original study record follows; its historical targets, results and protocols remain unchanged.

## Original fixed and inferred scale study

**Completed development study, 10 November 2026; no accepted neural checkpoint.** This study examines the noise-scale step of the Heston plan on the **existing 54 actual SPXW contracts**. It follows the user-authorized scale investigation in [PR #85](https://github.com/fmuia/sbi4finance/pull/85). The [offline HTML report](report.html) shows the results, recovery/SBC plots and executed network/rate code. The frozen [study protocol](protocol.json), original [reference protocol](reference-protocol.json) and [training protocol](training-protocol.json) were persisted before their respective runs. The explicit [cache recovery addendum](reference-recovery-v2.json) preserves the original targets and budgets.

The subsequent user-authorized [larger-bank training comparison](capacity-study/README.md) is complete under separate protocols and output identities. Both corrected continuations use the same six-target posterior, 100k physical surfaces, batches of 128/512 and 500 additional complete bank passes. Loss improves but c remains close to its prior, with final central-90% widths 1.349/1.345 against prior 1.35. The [new report](capacity-study/report.html) retains the numerical repair, all eight evaluations and an exploratory input-conditioning diagnostic. The results and stop below describe this original study and remain unchanged.

The five physical Heston priors, contract geometry, carry, ACT/365F clock and observed half-spread profile remain unchanged. No grid labels, spread-profile variation, additional tau/vega component, flexible-input training or Bergomi work belongs to this study. The earlier fixed-c neural failures and capped boundary reference remain under their original identities.

## Results and reflection

All seven new reference targets pass their declared convergence and sampled pricing checks. Fixed-scale sensitivity behaves as expected on the same observed prices:

| Synthetic case | Physical 90% widths at c=.5 / c=1 | Widths at c=2 / c=1 | Joint reference c central-90% interval | 60k neural c width / joint reference width |
| --- | --- | --- | --- | --- |
| Broad | .477–.494 | 1.93–1.99 | [.938,1.315] | 3.48 |
| Market-like | .495–.514 | 1.97–2.10 | [.970,1.364] | 3.33 |

The fixed-scale ranges span the five physical parameters. They are descriptive changes between different assumed targets, not accuracy scores. The joint references establish useful scale information in both synthetic cases. The current neural approximation misses much of it: at 60k its median c interval width is 1.336 versus the Uniform prior's 1.35, and its posterior-median recovery RMSE is .4452 versus .4371 for the constant prior-median predictor.

All six marginal population SBC screens pass at 60k. Nevertheless, the matched conditional panels fail at every predeclared endpoint (10k,50k,60k). At 60k, broad-case c has .94 reference-SD mean error and 3.48× width; market-like c has 1.03 SD error and 3.33× width. These are definite failures under the predeclared Monte Carlo-aware criteria. Strong physical-parameter recovery and reasonable pooled ECDFs coexist with these conditional errors. Neural ignorance does not establish that c is unidentifiable.

The separate real-market reference concentrates at the illustration prior's upper bound: c mean 1.99904, SD .00097, central-90% interval [1.99710,1.99995], with all sampled mass above 1.95. The network also fails that matched target. Boundary pressure is a warning about residual-scale support, model/data assumptions and fit under this truncated working model; it does not identify pure quote noise or justify retrospectively widening the prior.

The single six-target fit completed 60k finite updates in 480.48 seconds, with exact seeded checkpoint reload at all three endpoints. All 20k physical training rows were seen; reserved arrays remained unopened. The [neural exports](neural-evaluations/060000/evaluation.json) preserve the original population results and conditional samples. The [final comparisons](evaluations/060000/evaluation.json) consume those same samples after reference completion; no additional neural sampling or SBC recomputation occurs.

Across the seven references, maximum R-hat is 1.00893 and minimum bulk/tail ESS is 421.36. The largest sampled price-refinement change is approximately 1.03e-5 of the minimum supported measurement SD, and maximum absolute log-likelihood change is 6.81e-5. These sampled checks support numerical accuracy where tested, not a prior-wide certificate. Charged reference CPU, including failed zero-draw startup work and the symbolic cache preflight, is 2,947.86 of 6,300 seconds; the original wall deadline was preserved. [Recovery records](recovery-v2-campaign.json) retain all costs and failures; [verification](verification.json) records software and compute checks.

**Stop for reflection here.** The scale study is complete; neither the old fixed-c estimator nor the new joint estimator is accepted. A further neural improvement study must keep its target and comparisons explicit. Revised-grid labels, flexible inputs and Bergomi remain unstarted.

## Two inference questions on the same observations

Use one conditional Gaussian error law:

`Y_i = P_i(phi) + c * h_i * Z_i`, with independent standard-Normal `Z_i` conditional on c.

Here `h_i = (ask_i - bid_i)/2` is a frozen price half-spread and c is one positive dimensionless multiplier shared by all 54 quotes in a surface. It is not one absolute price SD applied equally to all quotes. Equating h to a Gaussian SD at c=1 remains a working convention.

| Question | What is supplied or inferred? | What is held fixed? |
| --- | --- | --- |
| Known-scale sensitivity | Assume c=.5,1,2 separately; infer the same five physical parameters | Exactly the same observed price vector y, h, geometry, physical prior and numerical gate |
| Unknown-scale inference | Infer the five physical parameters and c jointly, with independent physical-coordinate `c ~ Uniform(.5,2)` | The same y, h and geometry; the generating c is hidden |

Changing the assumed fixed c does **not** regenerate the observed prices. The two synthetic reference observations were generated at c=1; treating those same observations under c=.5 or2 is deliberate likelihood sensitivity, not a fresh correctly specified calibration experiment at those scales. With independent Uniform physical priors, changing a global fixed c preserves the physical MAP mathematically, because it multiplies the same weighted squared residual objective by a positive constant. Posterior means and interval widths can nevertheless change with nonlinear geometry and prior boundaries.

Joint inference is different from averaging the three fixed-scale posterior plots. It satisfies

`p(phi | y) = integral p(phi | y,c) * p(c | y) dc`,

where the scale weights are learned from the data under the declared prior. They are not equal fixed weights. The normalized likelihood includes `-54*log(c) - R(phi)/(2*c*c)`, where `R = sum(((y-P(phi))/h)^2)`, plus c-independent constants. Omitting the normalization changes the target.

The existing five-target c=1 network is not applied at other assumed scales or relabelled as a six-target network. Its failures do not disappear if a different joint target proves easier. Five- and six-target training losses are also not directly comparable accuracy scores.

## Matched reference panel and fixed-physical oracle

| Saved observation | Fixed c=.5 | Fixed c=1 | Fixed c=2 | Joint unknown c |
| --- | --- | --- | --- | --- |
| Broad synthetic, generating c=1 | New reference | Reuse ready reference | New reference | New six-target reference |
| Market-like synthetic, generating c=1 | New reference | Reuse ready reference | New reference | New six-target reference |
| Actual market | Deferred | Reuse ready application reference | Deferred | New application reference |

This is **seven new targets**, not seven new observations. Exact observation, design, task, source and selected-diagnostic/payload hashes bind every comparison. The existing boundary reference exhausted its cap and remains inconclusive; this matrix does not retry it or replace its historical outcome.

The reference protocol also records `p(c | phi_truth,y)` for the two saved synthetic observations, using their saved refined clean prices. At fixed physical parameters its density is proportional to `c^(-54)*exp(-R/(2*c*c))` on [.5,2], and its physical-coordinate mode is `sqrt(R/54)` clipped to that interval. This is a useful normalization and finite-sample check. It is **not** the six-dimensional posterior, its marginal `p(c|y)`, or evidence that physical parameters are known at inference. Its center need not equal the generating c in each noise realization.

No paired generating-scale .6/1/1.8 reference observations are added in this bounded matrix. The two conditional synthetic cases therefore do not establish conditional neural accuracy away from generating c=1. Broader scale calibration is assessed separately through the declared prior-predictive development population.

Each new reference uses four sequential native PyMC chains on one CPU, starting with 2k tuning/2k retained draws per chain. Only the one already declared 6k/12k escalation is available when initial chain diagnostics fail; numerical failure does not authorize escalation. The inner reference budget is 900 CPU/wall seconds and the outer process cap is 950 wall seconds including overhead. The controller permits at most two concurrent reference processes, 3,900 seconds for the whole reference campaign and 6,300 summed **inner** CPU seconds. A cap or unresolved reference leaves the comparison inconclusive; there is no automatic retry or budget extension.

## One new six-target estimator

The output is exactly `initial_variance, kappa, theta, vol_of_vol, rho, c`: five physical parameters plus one scale. Reuse the immutable 20k physical-row clean-price bank and current spline family, six coupling blocks, 128-by-128 conditioners, batch 128 and effective dropout zero. Different six-dimensional weights from the old five-dimensional model are unavoidable; no cross-dimension identity claim is made.

For each training exposure, draw one c independently from Uniform(.5,2), then apply the same Gaussian quote-error shape to `clean_price + c*h*Z`. Keep the c RNG namespace separate from the preserved physical-row/quote-error schedule. The input contains only the 54 ordered `Y/h` values. The generating c, `c*h`, clean prices and physical truth are not inference conditions. The new c target is not an old tau column reused under another name.

The single fit has predeclared 10k/50k/60k endpoints. Its schedule matches the current controlled recipe: the original 10k cosine schedule, then 5e-5 through step 49,999, 5e-6 for steps 50,000–54,999 and 5e-7 for steps 55,000–59,999. Retain Adam state across rate changes. The fit/reload wall bound is 900 seconds, with no automatic extension; the source-bound training protocol supplies the executable identity and limits. This document does not authorize another seed, capacity change or fit after a failure.

The population evaluation uses the first 512 declared validation physical rows, fresh independent c values from its frozen namespace 3, and the paired Gaussian-error seed. It evaluates 512 posterior draws per surface. Rank/coverage checks concern the full declared six-parameter prior-predictive population. There is no uniform-rank assertion conditional on a fixed physical truth or on truth-c bins. All reserved test members remain unopened.

## Acceptance scope and interpretation

- Pricing retains the strictest measurement gate `.02 * (.5*h)`, independently of assumed/inferred c. Larger inferred scatter cannot relax numerical accuracy. Reference readiness needs parameter/product convergence and sampled pointwise refinement checks; it is not a prior-wide numerical certificate.
- Conditional comparisons retain chain/draw structure and Monte Carlo-aware means within .3 reference SD and central-90% width ratios [.75,1.25]. The declared family has 144 checks: two metrics, eight quantities (six parameters and two analytic variance products), three cases and three endpoints. Missing, failed or capped references are inconclusive.
- Report all endpoints, physical/c recovery, prior-predictive ranks/coverage, paired parameter dependence and the two variance products. Passing population calibration alone cannot establish conditional accuracy or accept a checkpoint. The two synthetic conditional cases are development evidence, not untouched acceptance data.
- The market case has no generating physical truth or scale. It can test neural agreement with the specified joint reference, while market-model adequacy remains separate. Report c marginals, prior comparison, mass below .55/above 1.95 and physical–c correlations descriptively.
- A posterior pressing against c=2 can indicate inadequate residual-scale support, mean-model error or data/prior conditions under this truncated working model. It is not an estimate of pure measurement noise, proof of neural error, or permission to widen the prior retrospectively. Persistent strike/maturity residual structure remains relevant even with an inferred global scale.
- Stop with an explicit development outcome and reflection. This study does not promote tutorial assets, certify the old c=1 estimator, accept a revised grid or authorize Bergomi production.

## Grid reminder: unchanged in this study

| Stage | Locations and coordinates | Scope |
| --- | --- | --- |
| Original published Heston | 63 contracts: seven nominal maturities `[5,10,21,42,63,126,252]/252`, nine symmetric targets `[-2,-1.5,-1,-.5,0,.5,1,1.5,2]` at fixed 20% | Historical normalized complete-grid task, with kappa fixed |
| Current pilot | 54 actual contracts: six expiries, selected once near ATM-scaled `[-4,-3,-2,-1.5,-1,-.5,0,.5,1]`, then frozen | Already put-heavy; 21.03–324.99 calendar days with exact ACT/365F. These are the contracts used here |
| Audited E candidate | 63 actual nearest contracts at seven expiry buckets; fixed 16% targets `[-5,-3.5,-2.25,-1.25,-.5,0,.5,1,2]` | Data audit only: 62/63 targets supported, no one-year PM expiry. Explicit final-call +1.5 declaration supports the same physical contracts but does not meet the original endpoint/+2 criteria |

The candidate extends short-expiry and wing coverage relative to the current 54; it is not the first asymmetric design. Reference-volatility coordinates differ: original ±2 at 20% equals ±2.5 at 16%, so E does not extend the original positive wing at a matched maturity. See the [raw coverage audit](../grid-coverage/README.md). No actual contract, half-spread, carry estimate or maturity changes during this noise-scale study; new-grid labels and inference remain a separate later stage.
