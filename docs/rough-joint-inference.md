# Rough-Bergomi: a restricted joint posterior pilot

This M7.3b experiment adds inference to the existing rough simulator and option
bank. The [protocol](rough-inference-protocol.md) was frozen before generation.
It infers **H and eta jointly from nine synthetic Q option quotes**, with all
other context fixed. The experiment is a numerical-model study, not a market
calibration or a continuation forecast from historical returns.

## Review and reproduce

Open `pixi run review-rough-inference`, or
[`notebooks/09_rough_inference.py`](../notebooks/09_rough_inference.py).
The review explains the new functionality, runs a small importance-weighting
example and displays cached scientific evidence. No default cell generates a
large bank, trains, contacts Oscar or acquires market observations.

The frozen [training-bank configuration](../configs/tasks/rough_inference_bank.json)
and [reference-bank configuration](../configs/tasks/rough_reference_bank.json)
use the same 512-step task and separate root seeds. Both preserve the existing
`option_bank_v1` schema. Arrays are compact; this increment adds fixed-ID CPU
chunks for an expensive generator, not a new storage or orchestration framework.

To generate locally, use the existing `scripts/generate_option_bank.py` with
each config. For parallel generation, `scripts/generate_option_bank_shard.py`
accepts a config, split, start and stop ID. An optional JSON list of
`{config, split, start, stop, group}` supplies the SLURM array; paths to configs
are relative to that list. `cluster/option_bank_array.sbatch` runs the same
generator for one record. The assembler requires exact disjoint coverage,
matching configuration, source hashes, generation versions and array checksums.
It rejects incomplete or failed chunks; it never silently redraws them.

```bash
PYTHONPATH=src python scripts/assemble_option_bank.py \
  --config configs/tasks/rough_inference_bank.json \
  --shards /scratch/run/shards/pilot --out /scratch/run/pilot-bank
PYTHONPATH=src python scripts/assemble_option_bank.py \
  --config configs/tasks/rough_reference_bank.json \
  --shards /scratch/run/shards/reference --out /scratch/run/reference-bank
PYTHONPATH=src python scripts/check_rough_reference.py \
  --bank /scratch/run/pilot-bank --reference-bank /scratch/run/reference-bank \
  --out /scratch/run/reference
PYTHONPATH=src KERAS_BACKEND=jax python scripts/train_rough.py \
  --bank /scratch/run/pilot-bank --reference /scratch/run/reference \
  --out /scratch/run/checkpoint
PYTHONPATH=src KERAS_BACKEND=jax python scripts/evaluate_rough.py \
  --bank /scratch/run/pilot-bank --reference /scratch/run/reference \
  --checkpoint /scratch/run/checkpoint --out /scratch/run/evaluation
```

Use new output directories. The trainer requires a matching passed reference;
the evaluator requires the frozen checkpoint and matching artifacts. The
reserved final test is not a candidate-selection set.

The actual [pilot bank](results/rough-inference/banks/pilot/manifest.json) and
[independent reference bank](results/rough-inference/banks/reference/manifest.json)
are committed with this review (about 6 MB together). To reproduce without Oscar,
substitute `docs/results/rough-inference/banks/pilot` and
`docs/results/rough-inference/banks/reference` for the input bank paths above.
The generation step is unnecessary when reusing these exact artifacts. Final-test
arrays are included for reproducibility, but remain excluded from inference and
model selection.

## Measured bank generation

The [cluster record](results/rough-inference/cluster.json) records the actual
fixed-ID run, verified transfer, configuration hashes and independent local replay.
All 98 tasks completed without redraws or retries. The 6,146 generated panels
comprise 2,048 pilot rows and 4,098 reference rows; the latter include two unused
compatibility rows that preserve the common train/validation/test bank schema.

| Measurement | Observed value |
|---|---:|
| Wall span, first start to last completion | 34.28 minutes |
| Allocated CPU time (not measured utilization) | 31.68 CPU hours |
| Maximum concurrent tasks | 96 |
| Maximum reported task RSS | 212.78 MiB, against 4 GiB requested |
| Largest marginal price MCSE | 0.018722 currency, below 0.025 |
| Largest local replay difference | 8.88e-15 |

These are generation costs for this experiment, separate from posterior
inference and target pricing. The MCSE check addresses sampling precision, not
time-grid bias. Local replay verifies both banks' first training row; artifact
integrity checks cover all saved arrays. A separate
[tiny CLI smoke](results/rough-inference/software-smoke.json) exercises bank →
reference → train/reload → audit with reduced numerical settings and permissive
test gates. It verifies software wiring only and supplies no scientific result
for the frozen experiment.

## What the comparison means

The observation law includes finite Monte Carlo pricing randomness. The
independent reference therefore weights clean numerical prices from independent
prior draws, rather than pretending those prices are deterministic expectations.
Its weighted H/eta rows approximate the posterior for that numerical law.
Effective sample size and block agreement measure finite-bank precision; they
do not certify global support or continuum accuracy.

The reference bank can itself be reused across compatible panels, so it also
spreads simulation cost over repeated queries. This restricted two-parameter
experiment does not establish a neural speed advantage over that alternative.
Report bank generation, reference weighting, training, posterior sampling and
downstream pricing separately before making an amortization claim.

The planned training would use eight new quote errors per expensive base panel, creating
8,192 training observations from **1,024 independent parameter/pricing bases**.
Selection and audit remain independent, unaugmented base panels. The adapter
uses noisy observations alone; it does not expose true parameters, clean prices,
Monte Carlo errors or future outcomes to the network.

Three questions are kept distinct:

- **Does the network learn from a particular panel?** Compare with both the
  conditional reference and an unchanged prior baseline, including H/eta
  dependence. Weak learning of H alone does not mean the whole joint posterior
  is merely the prior.
- **Is it calibrated on average under the experiment?** Marginal SBC uses 512
  independent audit cases. An observation-ignoring prior can pass marginal SBC,
  which is why this check cannot replace conditional comparisons.
- **Does approximation error change a financial output?** Compare mean prices
  of two excluded puts. Fresh target-pricing random numbers prevent conditioning
  on the random fluctuations used to construct the input prices. The comparison
  includes sampling uncertainty and concerns the finite-grid mean price, not
  price-distribution quantiles or a continuum error bound.

## Measured status

**The bank is delivered; the reference gate stops this run before training.**
The actual [reference audit](results/rough-inference/reference.json) passes
19 of 20 panels. `validation:260` fails the two weight-concentration checks.
All eight gates, including the six that pass, remain at their frozen values:

| Reference check (worst across 20 cases) | Measured | Required | Result |
|---|---:|---:|---|
| Effective sample size | 600.55 | ≥800 | Fail |
| Largest normalized weight | 0.005392 | ≤0.005 | Fail |
| Mean MCSE / posterior SD | 0.03664 | ≤0.05 | Pass |
| Two-block mean difference / posterior SD | 0.06107 | ≤0.20 | Pass |
| Two-block relative SD difference | 0.05090 | ≤0.20 | Pass |
| Two-block correlation difference | 0.02171 | ≤0.20 | Pass |
| Target-mean MCSE, currency | 0.003576 | ≤0.02 | Pass |
| Two-block target-mean difference, currency | 0.004332 | ≤0.06 | Pass |

The [diagnostic graph](results/rough-inference/plots/reference-precision.png)
shows how the single failed panel compares with the others. Importance weights
concentrate on a smaller part of the prior bank for that observation; 4,096
simulated rows therefore do not mean 4,096 equally useful posterior draws.
An ESS near 601 does **not** prove the reference is wrong, and the agreeing
block/mean-price checks are encouraging. It falls short of this experiment's
declared evidence requirement, which remains the acceptance rule.

The reference run took **400.58 seconds locally**, including 376.64 seconds for
fresh target pricing and 23.89 seconds for loading and stress-panel generation.
These target draws are independent of the Monte Carlo randomness that generated
the conditioning quotes. Their sampling uncertainty does not bound discretization
bias. The frozen protocol and all raw reference arrays are retained.

**No neural checkpoint was trained for this frozen run; no neural conditional
audit or SBC result exists.** The implemented trainer refuses the failed
reference. The reduced CLI smoke tests the complete software route, not these
scientific gates. The final test remains unevaluated. This outcome is a
reference-precision limitation, not evidence that rough inference failed to
learn or merely reproduced the prior.

The bounded next inference experiment should change the reference proposal or
budget, keeping this result visible and specifying its new checks before running.
A targeted reference proposal may be cheaper than regenerating the whole bank;
it needs correct importance densities and explicit support checks. Existing pilot
training rows can be reused, while already inspected audit cases remain development
cases. No automatic bank expansion, threshold change or tuning sweep is part of
this increment. M5.2's learned pricing surrogate and M8's selected participant
release remain separate work.

The [offline review verification](results/rough-inference/notebook-verification.json)
passes strict marimo checks and a fresh-process HTML export: 21 cells, zero
errors, 8.51 seconds, with external connections blocked and no market fixtures,
checkpoint or raw bank present. The live analytic example and cached diagnostic
figures were visually inspected. The integrated software suite passes 142 tests;
standalone PR #14, without the HKDE inference increment, passes 122. The
[run status](results/rough-inference/run-status.json) also records an actual
trainer invocation rejecting the failed reference before creating a checkpoint.

Reference adequacy, neural acceptance and offline tutorial execution are
separate gates. The earlier Heston and HKDE failures and their reserved tests
remain unchanged. This numerical Q experiment makes no market-performance or
P rough-memory forecast claim and does not choose the participant application.
