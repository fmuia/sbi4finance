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

Training uses eight new quote errors per expensive base panel. This creates
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

Generation and audit outcomes are recorded here when the frozen run completes.
Reference adequacy, neural acceptance and offline tutorial execution are
separate gates. The earlier Heston and HKDE failures and their reserved tests
remain unchanged. A working Bergomi bank does not resolve those experiments or
choose the participant tutorial's application.
