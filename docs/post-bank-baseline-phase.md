# First larger rough baselines: prepared post-bank phase

This phase is **prepared, not submitted**. It runs one posterior and one forward price emulator for each completed rough bank. It does not evaluate reserved tests, run new MCMC, execute notebooks, choose a winning architecture or promote scientific acceptance.

Each route starts independently, only after its bank controller publishes a successful `completed_checked_snapshot` record with the exact snapshot hash, split counts, producer and collection identities. A snapshot directory alone is not readiness.

| Route | Required immutable bank | Training / validation / test rows |
| --- | --- | --- |
| Known curve | `/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-50k` | 50,000 / 5,000 / 7,000 |
| Joint curve | `/oscar/scratch/afengler/sbi4finance-native-bank-20261027/snapshot-20k` | 20,000 / 5,000 / 7,000 |

Generation success records are `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/{known,joint}/final.json`. Known-curve inference observes its supplied curve; joint inference excludes the generating curve. The numerical target and producer of each bank remain unchanged.

## Fixed recipe

Each route receives one four-CPU, one-L40S, 32-GiB allocation with a 60-minute wall limit and no requeue:

1. Train FlowMatching with the direct vector encoder, four 256-unit blocks, 500 epochs, batch size 256, 50-epoch recovery chunks, seed 0 and fresh quote noise. Use the existing recorded AdamW schedule and BayesFlow 2.1's native adaptive sampler. **Predeclare final-epoch selection** with `--checkpoint-selection final`.
2. Fit the deterministic forward-price emulator with four 256-unit blocks, at most 300 epochs, batch size 256, learning rate .001, validation patience 30 and seed 0. Its ordinary validation selection is retained: these fixed-label regression losses differ from stochastic FlowMatching validation losses.
3. Evaluate the posterior on exactly `validation:0..1999`, 1,000 draws per observation, batch size 100 and seeds `0 + batch start`. Keep every 100-case draw batch before continuing; retain ordinary parameter and derived rho×eta diagnostics. Validation is development evidence, not acceptance.

The posterior trainer still saves the validation-best checkpoint and final recovery checkpoint separately. Final selection only changes which committed model is published as `model.keras`; metadata retains both epochs and hashes. Default trainer behavior remains validation-loss selection for existing commands. A run cannot change its declared policy during recovery.

The two allocations' full wall limits plus 240-second cleanup reserves total **30,720 CPU-seconds / 7,680 GPU-seconds**, within separate pooled ceilings of **9 allocated CPU-hours / 2.25 GPU-hours**. Internal per-stage maxima are 1,500 seconds for posterior training, 1,200 for pricer training and 1,700 for validation, each clipped to the remaining 3,550-second whole-run allowance. A stage timeout is retained as a failure, not automatically resubmitted or repaired with a new seed.

## Durable operator

The one-off [operator](../cluster/submit_full_surface_backbones_20261028.py) reuses the existing training/evaluation scripts and has only two routes. Initialize it **after** the revised trainer is committed and a clean immutable checkout plus pinned GPU environment are available on Oscar:

```bash
python cluster/submit_full_surface_backbones_20261028.py initialize \
  --root /oscar/scratch/afengler/sbi4finance-post-bank-baselines-20261028 \
  --repo /path/to/clean/frozen/repo --revision EXACT_COMMIT \
  --python /path/to/existing/core-gpu/bin/python
```

Then launch `watch --root /oscar/scratch/afengler/sbi4finance-post-bank-baselines-20261028` under the usual persistent Oscar operator session or `nohup`, retaining its PID/log. It checks readiness once per minute, records each attempted submission **before** contacting Slurm, and never retries an uncertain response. A filesystem lock prevents two operators from issuing the same jobs. A failed bank, source change or unknown submission stops that route; the other route remains independent. Watching is bounded to 24 hours and performs metadata operations only.

The phase saves `plan.json`, `execution.json`, per-route `run.json`, training/pricer/validation outputs and raw logs. Accounting uses each top-level allocation once. Completed training output is still provisional until independent price, posterior-reference and product checks support the relevant claims. In particular the existing rough pricing-emulator errors and bank-pricing-covariance distinction do not disappear merely because more data are available.

Actual tiny BF tests verify the default selection policy and interrupted/resumed final selection with an intentionally different validation-best epoch. Pure metadata tests verify that incomplete/wrong-producer banks do not release work and ambiguous scheduler timeouts are durably recorded without retries. No Slurm jobs were submitted while preparing those tests.
