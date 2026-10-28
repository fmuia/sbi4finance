# Generate or extend a bank with one launch command

The reusable launcher takes one campaign JSON and calls the existing collection, shard and snapshot code. It uses the same operator for a new task or an append; no new Python wrapper is needed for each experiment. The original generation phases retain their frozen operators and ledgers. The known-curve 100k append now uses a separately frozen copy of this launcher; see the [live recovery record](run-recovery-2026-10-28.md).

On Oscar, use the locked numerical Python belonging to the frozen producer:

```bash
/absolute/path/to/locked/python scripts/run_bank_campaign.py \
  --config /absolute/path/to/campaign.json --detach
```

The command returns a PID and paths to `launch.json`, `operator.log` and `execution.json`. The child starts a new session and survives terminal/assistant exit. A launch receipt means the process started; check the execution ledger for preparation or generation success. Do not start another campaign against the same collection while one is active.

Copy and edit [the example JSON](../configs/campaigns/rough_known_append.example.json). The placeholder paths deliberately do not name an active collection. Parameters include the target training size, chunk size, maximum concurrent workers, each worker's wall/memory limit and the campaign CPU-hour ceiling. The final freeze job's full wall and cleanup allowance are reserved throughout generation. `prior_cpu_seconds` is zero for a new phase budget; specify existing spending only when the ceiling is cumulative. `python` defaults to the Python running the launcher; normally leave it unset.

`producer_repo` can point to the older compatible frozen producer when the launcher itself comes from a newer checkout. This avoids modifying the numerical code of an existing collection. Optional `expected_git_revision` and `expected_runtime` provide extra declared startup assertions. Exact generation source hashes and package versions are always recorded by the native collection and checked by native workers.

Optional `exclude_nodes` is a list of Slurm node names, for example `["node2339"]`. Use it when measured cluster failures or severe slowdowns justify excluding a node; it does not change simulation identities or seeds.

## Reuse data or change the experiment

- **Same task, root, generator and runtime:** point at the existing collection and increase `train`. Native planning verifies completed data and emits only missing ranges. Validation/test counts and bytes stay fixed. Use new campaign and snapshot directories.
- **Changed grid, prior, quote law or generation code:** make a new task config and collection path. Keep the old collection. Choose a distinct root for a genuinely new campaign; creating another collection with the same task/root/IDs would repeat experiments.
- **A different worker count or chunk size:** these do not enter a surface's random identity. The identity remains task/root/split/ID/purpose.

The example is an **append** to an existing compatible known-curve collection. Do not use its task/root in another fresh collection and call that new data. For a genuinely new task, copy the numerical config, edit the scientific settings and root deliberately, then point the campaign at that file and an unused collection path. Existing Heston, known-curve rough and joint rough configs can all use this launcher.

## Checks without repeated historical reconstruction

Native planning and snapshot publication always audit actual uint64 pricing seeds across the target collection. This is distinct from checking every PCG64 state in every historical campaign.

The seed policy is explicit:

- `native_collection` performs the native target-collection audit. The ledger states that no cross-collection certificate was requested. It makes no claim of a global history audit.
- `certificate` additionally requires a previously validated cross-collection seed certificate at `seed_certificate`, with its SHA-256 in `seed_certificate_sha256`. Its format is `calibration_cross_collection_seed_certificate_v1`, with `passed: true`, exact `collection_id`, `rng_protocol`, `task_fingerprint`, `root_seed`, and `covered_splits` for train/validation/test. The requested counts must be inside that audited domain. A certificate covering 50k cannot silently certify 100k. A compatible verified certificate is reused; no historical scan is hidden in the launch.

If a wider cross-collection audit is required, create/extend an explicit saved seed inventory or certificate separately. The current one-off historical report must not be relabelled as covering new ranges. Full initialized-state independence is not proved by distinct root numbers or a uint64 collision check.

## Budget and failure behavior

Only enough array tasks to fit both the concurrency and **completed cost + all outstanding full wall/cleanup exposure + final freeze reserve** limits are submitted. Missing accounting records retain their full reservation. Array offsets avoid a large global index exceeding Slurm's configured array size. The thread count is one per worker; GPU resources are not requested.

Each attempt is written before `sbatch`. A timeout or ambiguous response stops the campaign and is never retried automatically. Failed tasks retain their original identities; no easier seed is substituted. Successful data remain reusable. An exhausted budget produces a stopped ledger instead of raising the cap. An existing campaign directory is not restarted automatically: reconcile its recorded job IDs before planning another attempt. Snapshot publication uses native integrity/coverage checks, not a custom price replay.

The launcher is a small bounded process, not a service, scheduler or database. Its tests use tiny native collections with mocked prices and a fake scheduler: append/holdout preservation, changed-task rejection, budget/exposure, unknown submission, array offsets, certificate coverage and detached receipt. No numerical bank or cluster job was run to test this interface.

For the lower-level commands and artifact identities, see [incremental banks](incremental-calibration-banks.md). Training starts from a checked immutable snapshot in a separate run directory; it is not part of this launcher.
