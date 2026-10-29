# Two remaining joint holdout ranges

The morning recovery completed 307 of its 309 pricing allocations. Two allocations, `8046699_997` and `8046699_1000`, reached their three-hour limits on node2346. The other two slow allocations on that node eventually succeeded. Read-only process samples showed unusually high system CPU and minor page-fault counts; this supports a node-associated slowdown, without establishing a hardware diagnosis.

The checked native collection planner found exactly two missing 25-row ranges:

| Original map index | Split and range | New array task |
| --- | --- | --- |
| 997 | validation `[4925, 4950)` | `8051758_0` |
| 1000 | test `[0, 25)` | `8051758_1` |

The new native job hashes match the original job hashes. All **1,278 successful parts** remain in place: 20,000 training, 4,975 validation and 6,975 test surfaces. No successful range, parameter vector or seed is replaced. Reserved test surfaces are generated for completeness; their generation does not authorize their use for model selection.

The unchanged [generic launcher](../../../../../scripts/run_bank_campaign.py) started a separately recorded campaign after explicit review. Array **8051758** was accepted at 13:50:44 UTC, 28 October 2026. Both workers were observed running on node1804 and node1807, avoiding node2346 and the four previously excluded nodes. Each has one CPU, 8 GiB memory and the original three-hour wall limit; concurrency is two. The detached operator PID is 191281.

All previous successful and failed allocations are charged: **3,141,589 allocated CPU-seconds (872.663611 hours)** before this continuation. The original cumulative **1,500 CPU-hour ceiling** remains unchanged. Even reserving both full worker walls plus cleanup, a two-hour generic freeze and a separate two-hour integrity allowance gives 882.930278 hours. Actual metadata and worker usage will be reconciled before the final receipt.

The source/config directory is `/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/tail-recovery-source-20261028`. The campaign ledger and logs are under the adjacent `tail-recovery-20261028` directory. The generic launcher first creates `tail-recovery-20261028/native-snapshot-20k`; the existing tested integrity stage then creates and checks the original `sbi4finance-native-bank-20261027/snapshot-20k` target. Completion and downstream readiness require the explicit chained recovery receipt, with all failed journals preserved.

Evidence: [terminal reconciliation](reconciliation.json), [retained inventory](preserved-parts.json), [reviewed config](campaign.json), [budget and identity checks](checks.json), [deployed hashes](deployment.json), [actual detached config and native jobs](launch-receipt.json), [submission ledger](execution.json), [running nodes](running-states.psv). The source config and detached config differ only by the generic launcher's `expected_operator_sha256` field, added before detaching.

These receipts document a launch, not a finished bank. No training, numerical settings, priors or scientific acceptance claims change in this continuation.
