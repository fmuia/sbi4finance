# Final-epoch 100k Heston development panel

Job **8032002** completed the already-saved final-epoch 500 checkpoint on the same **2,000 validation observations**, with **1,000 draws per observation**, batch size 100 and seeds `0, 100, …, 1900`. BayesFlow's native adaptive Tsit5 policy was unchanged. The checkpoint was examined after observing selected-checkpoint failures, so it is explicitly a **post-hoc development candidate**, not independent acceptance.

All four rank KS statistics are below the simultaneous DKW band **0.0356201271**. The first column below is initial variance, followed by theta, vol-of-vol and rho in the machine report. [The complete evaluation](results/final100k-validation/evaluation.json) includes all IDs, pointwise Wilson intervals and the post-hoc flag.

| Parameter | Final-500 rank KS | 50% coverage | 80% coverage | 90% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial variance | 0.0138308 | 50.60% | 81.25% | 91.15% | 95.80% |
| Theta | 0.0282422 | 52.25% | 81.60% | 90.80% | 95.65% |
| Vol-of-vol | 0.0256307 | 52.35% | 82.10% | 90.85% | 95.10% |
| Rho | 0.0211828 | 49.40% | 81.00% | 91.10% | 95.70% |

For comparison on these observations, initial-variance KS was **0.02215 for the archive**, **0.05084 for the selected epoch-402 checkpoint**, and **0.01383 for the final checkpoint**. Other parameters are mixed: the final theta/vol-of-vol KS values are slightly larger than the archive's 0.02757/0.02189, while its rho KS is smaller than the archive's 0.02387. Both the final candidate and archive pass the four-parameter screen. This does not establish universal superiority of the new model.

The pointwise, unadjusted Wilson intervals exclude nominal coverage for theta and vol-of-vol at 50%, and vol-of-vol at 80%. Those deviations remain visible; a passing simultaneous rank screen is not a claim of perfect calibration. The same validation pool was used by the original epoch-selection procedure, and the final candidate was investigated after seeing development failures. Untouched acceptance, broader conditional and product checks remain outstanding.

## Identity and scope

The final checkpoint SHA-256 is `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`. The original epoch-402 checkpoint (`7f037dee0fde78cd5a33f1107e13031ff01fdcb5481dab76fc6c8f7d07598f39`) and its reports were not overwritten. The checked 100k-bank manifest remains `4b5929b0d0b23698c340eec6defabfd629fe23eca2ed78fe9bd46e4de35c244e`.

The [run journal](results/final100k-validation/run.json) explicitly records `posthoc_development_candidate=true`, `scientific_acceptance=false` and `reserved_test_evaluated=false`. It also commits each 100-case batch before moving on. All 20 batches and the combined draw archive remain on Oscar; only their SHA-256 hashes and sizes are retained in [remote-artifacts.json](remote-artifacts.json). All **404 frozen source/input hashes** were rechecked. [inputs.json](inputs.json) and the [driver](evaluate_heston_final_audit_20261027.py) preserve the exact computation.

There was **no refitting, new bank generation, financial pricing, MCMC or reserved-test evaluation** in this unit. These are different saved weights from the same completed 500-epoch training run.

## Accounting and preservation

The allocation completed in **921 seconds with four CPUs and one GPU**, using **3,684 allocated CPU-seconds / 921 GPU-seconds**. Sampling and report computation took 903.91 seconds. Actual use is below the separate **2 CPU-hour / 0.5 GPU-hour ceilings**, which reserved the full 25-minute limit plus cleanup. Count only the top-level allocation once; [accounting.psv](accounting.psv) also preserves job-step records for inspection.

[execution.json](execution.json) is the normalized terminal ledger; [execution-submitted.json](execution-submitted.json) preserves the original submission. The [Slurm log](logs/final-audit-8032002.out), [summary](summary.json) and [downloaded-files.sha256](downloaded-files.sha256) retain the results and byte identities. The local and atomically updated Oscar ledgers agree exactly. Model, combined draws and partial-batch payloads were not transferred.
