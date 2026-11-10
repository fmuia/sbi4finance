# Bounded learning-rate comparison

This is one finite optimization comparison from the exact zero-dropout 50k
checkpoint. It retains the same 20k clean-bank rows, five physical priors,
fixed `c=1` observation model, ordered 54-quote design, seed and absolute
row/noise steps. Adam's moments, iteration, standardizers and auxiliary RNG
arrays were restored exactly. Only the learning-rate schedule changed:
absolute steps 50000–54999 use `5e-6`, and 55000–59999 use `5e-7`.

The [protocol](../annealing-protocol.json) was persisted before the single
training run. Both 55k and 60k checkpoints and all evaluation outcomes are
retained. No checkpoint is accepted, and no further fit or extension was
launched.

| Endpoint | Broad synthetic | Market-like synthetic | Boundary synthetic | Market application |
|---|---|---|---|---|
| 55k | fail | fail | inconclusive | fail |
| 60k | fail | fail | inconclusive | fail |

The 60k development SBC rank/DKW check passes for all five coordinates, with
90% interval coverage of 0.916–0.934 on the fixed 512 development rows. Both
synthetic conditional comparisons still fail. Pooled calibration therefore
does not establish conditional posterior fidelity. The boundary reference
exhausted its predeclared budget and remains inconclusive. The market
application is assessed separately and also fails neural/reference agreement.

At 60k, the broad-case physical 90% width ratios are 1.13–1.41. Its largest
absolute mean discrepancy is 0.78 reference SD. In the market-like case,
`kappa`, `theta` and the affected expected-variance product pass their
individual checks; `vol_of_vol` and `rho` still fail. Some means improve and
others worsen relative to 50k. These results support optimization sensitivity
without identifying a data or network-capacity limit.

The run completed without a nonfinite event in **99.86 seconds**, including
restoration, training, checkpoint checks and both evaluations, within the
300-second total cap. The final 60k fixed-probe timestamp is 75.04 seconds;
this precedes final checkpoint/reload and evaluation work and is not a
separate measured fit-only duration. Validation probe loss changes from
−16.1048 at the parent to −17.2289 at 60k. Lower loss is not an acceptance
criterion.

All 84 trainable, 46 nontrainable, 169 optimizer and two metric arrays are
checked on endpoint reload, along with the serialized learning-rate schedule
and exact seeded posterior samples. Parent checkpoint, bank, panel,
reference and producer hashes remained unchanged. No reserved arrays or
pricing calls were used. Three focused annealing tests passed before launch.

The JSON files here are exact copies of the local run's records, with their
source and destination hashes in [manifest.json](manifest.json).
[checkpoint-payloads.json](checkpoint-payloads.json) binds both endpoint
metadata, model and full-state sidecars to their evaluations. The large model
and state payloads remain under
`local-artifacts/heston-spread-baseline-20261109/training-fixed-c1-dropout-zero-annealed/`.
