# Final integrity and training handover

The native tail campaign retries only the two retained ranges: validation
4925:4950 and test 0:25. Its new snapshot lives under the tail campaign root.
The metadata bridge then submits the unchanged tested `bank_delivery_stage_v2.py`
final stage once. That stage creates the original `snapshot-20k`, verifies all
1,278 preserved parts unchanged, and checks the complete 32,000-row bank.

The original numerical checks remain in force: exact prior, joint derived curve,
and quote-noise replay; reported Monte Carlo standard errors at most 0.5 quote
standard deviations; exceptions above the nominal 0.25 allowance only at the
last permitted path-count rung; and complete split coverage. This does not
establish continuous-price accuracy or scientific acceptance.

Both snapshots are small manifests referencing the same immutable parts. There
is no duplicate pricing payload and no additional financial simulation in this
bridge. The original historical seed preflight is reused byte for byte. The
retained 1,278-part inventory is bound separately and verified by the finalizer.

The bridge waits for successful terminal accounting of both native workers,
the native snapshot allocation, and its independent integrity allocation. It
then publishes the same completed receipt at:

- `tail-recovery-20261028/integrity-bridge/completion.json`, for the versioned
  100k extension gate;
- `joint/final.json`, for the existing joint20k training waiter.

These paths are below
`/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028`.
Neither destination may already exist. Failed or ambiguous submissions remain
recorded, and the bridge never resubmits them. Temporary read-only accounting
failures wait within its finite 14-hour lifetime.

The integrity allocation is one CPU, 8 GiB, 60 minutes, with a 240-second cleanup
reserve and no requeue. The conservative combined maximum, including all prior
allocations, the two replacement workers and both finalizers, is 3,178,549 CPU
seconds. It remains within the original 5,400,000-second (1,500 CPU-hour) cap.
The completion receipt recomputes actual cumulative cost from settled Slurm
allocations; historical timeout costs remain included.

The prepared source and plans are `complete_joint_tail.py`,
`integrity-stage-plan.json`, and `integrity-bridge-plan.json`. They require a
separate reviewed deployment/arm step; preparing these files does not start a
job. The remote stage directory must receive exact copies of the original
`joint/preflight.json`, original `jobs-20k.json`, and retained 1,278-part
`preserved-parts.json` at the paths and hashes listed in the bridge plan.

The reviewed bridge was armed at 13:57:27 UTC on 2026-10-28, with PID 3795047
on `login010`. The first retained receipt shows it waiting for the native
recovery and an empty error log; this is not a completed bank or an integrity
job submission. See `integrity-running-receipt.json` for the launch attempt,
confirmed launch and first heartbeat. Source SHA-256 is
`8b84e573238daf4adcc937388a11876b8213a9c17ad41ff09470474101c4b338`;
the bridge plan is
`4106b4079a4df4ea07ce342237fb1546c3c060b92d2c721254b2397a0a517dbd`.

Four local synthetic metadata regressions pass: terminal-accounting release,
preserved-part identity rejection, ambiguous submission without retry, and
delayed accounting followed by successful completion with one finalizer
attempt. No numerical bank, model, notebook, or reserved-test outcome was run
for these tests.
