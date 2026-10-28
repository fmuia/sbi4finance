# Rough banks: active append-only delivery

Both banks have actual pricing workers running. These are generation jobs, not merely metadata checks or plans.

| Route | Current phase target | First actual pricing allocation | Initial worker state |
| --- | --- | --- | --- |
| Known curve per case | Extend training 20k → 50k; retain 5k validation and 7k test | `8032877`, new map indices 0–63 | 64 running |
| Joint curve inference | Finish original 20k training / 5k validation / 7k test from 9,925 checked training rows | `8032878`, original map indices 397–516 | 120 running |

The [first worker-state query](first-worker-states.json) confirms the running counts. They are a dated observation, not a live dashboard. The detached controller refills missing ranges until checked snapshots exist or an explicit stop condition is reached. No new training, posterior comparisons or notebook execution is part of this controller.

**Later 100k direction:** the user subsequently requested 100k training surfaces for each rough route. Separate follow-on phases are being prepared; they do not change this phase's caps or readiness gates. The prepared `bank_delivery_controller_v3.py` / `borrow_joint_slots.py` slot-transfer prototype was **never deployed or armed** and is superseded by those follow-ons. Do not run it while the known100k phase needs its worker slots. The active current-generation controller remains **v2**.

## Durable process and limits

- Oscar root: `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028`.
- Reviewed controller PID at launch: **2709727**, source [bank_delivery_controller_v2.py](bank_delivery_controller_v2.py); receipt [controller-launch-v2.json](controller-launch-v2.json).
- Active remote journal: `execution-v2.json`; [the committed copy](execution-v2.json) is a point-in-time receipt. Each scheduler submission is recorded before `sbatch`; ambiguous responses stop rather than resubmit.
- Known phase: **120 additional allocated CPU-hours**, one CPU/8 GiB per 25-row shard, 40-minute limits, at most 64 nonterminal workers.
- Joint phase: **1,500 cumulative allocated CPU-hours**, including the previous **1,020,299 CPU-seconds**; one CPU/8 GiB, three-hour limits, at most 120 nonterminal workers. The previously problematic node2338 is excluded.
- Combined maximum **184 nonterminal bank workers**, one numerical library thread per worker. Full queued/running time limits plus 240-second cleanup exposure and the two-hour per-route check reserve are charged before each wave. No GPU.

Terminal failure, changed source/map identities, unknown submission, unavailable accounting or exhausted allowance stops expansion for review. Successful parts and failed original ranges are retained. There are no automatic replacement seeds, retries, larger budgets or changed financial tasks.

## Immutable producers and preserved data

The [reviewed v2 plan](plan-v2.json) retains both producers and collections:

| Route | Producer | Collection identity | Root |
| --- | --- | --- | ---: |
| Known | `616352035be4b9d271151090aa5d2d9539da8b1c` | `da963f7294aee242a80ebf9b47d914294b439aef8b6ee8f7c6d1899766e546f1` | 271300 |
| Joint | `c773ceb26e7273612b15c69fd63715d131778083` | `817de3664968de0c605ae4c73e2c99c26465344294af2fc5412f39fffa4baa50` | 271200 |

Metadata jobs **8032794** and **8032795** passed before pricing. The [known preflight](known-preflight.json) checked all 1,280 existing parts and audited new train IDs 20000–49999 against retained historical declarations, including the full planned joint and Heston banks. The [joint preflight](joint-preflight.json) checked all 397 existing parts, the original job map and original seed certificate. Both compare actual financial-module hashes and Python/NumPy/SciPy versions with the collection, not only Git revision names. Holdout bytes and existing successful surface identities are preserved.

The known bank keeps its original **80/504/504** finite-simulator target and empirical precision rule. The measured emulator and grid errors remain documented; larger data do not remove them by themselves. The joint bank keeps **1280/2016/2016** and the original broad prior/curve task. No incompatible prices are mixed into either collection.

## Readiness for training

Expected immutable snapshots:

- Known: `/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-50k`.
- Joint: `/oscar/scratch/afengler/sbi4finance-native-bank-20261027/snapshot-20k`.

A snapshot path alone is not readiness. The route's `known/final.json` or `joint/final.json` must have `status="completed_checked_snapshot"`, the matching `snapshot_manifest_sha256`, `split_counts`, collection/task identity, producer hashes and runtime. Final integrity freezes through the existing native command, confirms preserved part hashes, and replays prior/context/noise identities. Test data are generated and checked for numerical/ID integrity only, not evaluated or used for model selection. A separately approved training phase can consume each route when its own check passes; it need not wait for the other bank.

## Caught wrapper correction

Before any pricing submission, review caught a final-integrity wrapper bug: joint curve auxiliaries must use `tasks.derived_context(task, theta)`, while known curves use the independent purpose-3 `draw_context` stream. The first controller was stopped with only its two metadata submissions recorded. Old plan/source/receipts are retained. The reviewed [amendment](amendment-v2.json) versions the finalizer/controller, reuses the same passed metadata receipts and lets each bank advance after its own gate. It does not change a simulator, seed, prior, grid, cap or existing payload.

[The tiny regression](finalizer-checks.json) reproduces the old joint failure and then passes real native freeze/replay on mocked-price known and joint banks after the fix. [Operator checks](operator-v2-checks.json) verify independent gates, prospective accounting, attempt-before-submit, frozen submission directory and duplicate/failure rejection without scheduler calls.

Use the existing [collection instructions](../../../incremental-calibration-banks.md) for later on-demand growth: choose a new target and allocation, `plan` only missing IDs with the unchanged producer/root/holdouts, launch bounded workers, and `freeze` a new immutable snapshot. Current jobs remain active independently of this chat; do not launch a second controller against their missing ranges.
