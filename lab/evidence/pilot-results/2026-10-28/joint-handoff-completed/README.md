# Joint 20k bank: integrity and automatic handoff complete

The original joint bank is **checked and complete: 20,000 training / 5,000 validation / 7,000 test** surfaces. Snapshot manifest SHA-256: `c0de030bafef6de3f728d75788cd7feddd672b476795b57c42ecd8cbb0843ce9`.

Both replacement ranges completed successfully, followed by native snapshot publication and the unchanged full integrity finalizer. All four allocations exited `COMPLETED/0:0`. The finalizer verified all 1,278 preserved parts unchanged, full 1,280-part coverage, exact prior/context/noise replay, and the recorded Monte Carlo precision conditions. There were no nominal-precision exceptions. These checks do not establish convergence to continuous-time model prices.

The complete original phase, both recoveries and finalizers consumed **873.895 allocated CPU-hours**, including all failed attempts, within the original 1,500-hour ceiling. Native and original-path snapshots refer to the same immutable parts; prices are not duplicated.

The existing metadata operators then released both downstream jobs automatically. At the observed handoff, joint baseline **8053365** was running on a GPU and joint-100k array **8053366** had **200 running workers**, none on the five excluded nodes. The observer made no launches or mutations. Training has since completed; consult the current collaborator overview for its validation result and newer bank counts.

[Observed terminal/start summary](handoff-summary.json) · [Completion/readiness record](completion.json) · [Native recovery ledger](generic_execution.json) · [Integrity ledger](integrity_execution.json) · [Checked manifest summary](checked-manifest-summary.json).

JSON wrappers retain original remote paths, original-file hashes and parsed records. The manifest summary omits its repeated part references while recording their count and the full original manifest hash. Exact large manifests and data remain on Oscar. The completion and original readiness files were verified byte-identical at publication. Reserved test data were used for generator integrity only, not inference evaluation.
