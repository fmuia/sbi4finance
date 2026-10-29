Evaluation-only continuation **8061867 completed successfully** at 18:07:56 UTC on 28 October 2026, with scheduler state COMPLETED and exit code 0:0. It used 139 seconds on gpu3102, four CPUs and one L40s GPU. Both saved-checkpoint evaluations completed without retraining. Together with the original four evaluations, the pilot now has **6/6 trained and 6/6 evaluated models**. The original partial-screen failure remains preserved.

The original pilot trained **all six models successfully** for 200 epochs and passed their final-checkpoint reload checks. Four evaluations completed. DeepSet and SetTransformer evaluation subprocesses exceeded the original driver's 65-second limit; the driver then marked the screen partial and exited with code 1. Slurm job 8059371 ended FAILED at 17:57:31 UTC after 1,231 seconds. That failure and all original outputs remain unchanged.

| Completed original variant | Marginal rank checks passed | Empirical coverage of nominal 90% intervals, across four targets |
| --- | ---: | --- |
| Flow matching, vector | 4/4 | 84–89% |
| Affine coupling, vector | 3/4 | 90.5–98% |
| Spline coupling, vector | 4/4 | 88–92.5% |
| MeanFlow, vector | 4/4 | 80.5–86% |

These are preliminary measurements on 200 validation cases. In particular, the loose marginal rank screen does not override MeanFlow's visible undercoverage. No architecture winner or scientific acceptance is declared. The existing development task is four-parameter Heston represented through zero-jump HKDE with kappa fixed at 2; results should not be generalized to the primary five-parameter task.

The completed continuation used the identical validation cases, draws and sampling seed:

| Continuation variant | Marginal rank checks passed | Nominal 90% interval coverage: initial variance, theta, vol-of-vol, rho |
| --- | ---: | --- |
| Flow matching, DeepSet | 4/4 | 92.5%, 85.5%, 90.0%, 88.5% |
| Flow matching, SetTransformer | 4/4 | 91.5%, 85.0%, 90.5%, 87.0% |

Both pass the same rank bound of 0.11264, while theta coverage remains below nominal. The preliminary screen establishes that all six variants can train and sample; it does not establish that every variant is sufficiently calibrated. The two evaluator-reported runtimes were 44.39 and 45.55 seconds; full CLI startup and loading also consume the allocation and were included in the continuation deadline.

The continuation evaluates only the two saved final-200 checkpoints, using the same first 200 validation cases, 256 draws, seed 0, batch size 50 and BayesFlow native adaptive sampler. Each evaluator gets at most 240 seconds within one ten-minute allocation. No model is retrained, no solver setting is changed and no original output is overwritten. It cannot submit another job or automatically retry an ambiguous submission.

Fresh accounting charged the original failed allocation 4,924 CPU-seconds and 1,231 GPU-seconds. The new four-CPU, one-L40s-GPU, 32-GiB job reserved at most 3,360 CPU-seconds and 840 GPU-seconds including cleanup. Its actual completed use was 556 CPU-seconds and 139 GPU-seconds, bringing cumulative pilot use to 5,480/1,370 within the original 15,360/3,840 ceilings. No allowance was increased.

`original/` preserves the exact original plan, execution record, partial `run.json` and four completed evaluations. `checkpoint-identities.json` and `preflight.json` bind all six saved models and training records. `prior-accounting.json` preserves terminal failed-allocation accounting. `plan.json`, the launch receipts and `queue-verification.json` describe the continuation. The final `run.json` and two encoder subdirectories preserve the exact completed evaluation metadata; `terminal-verification.json` binds their hashes and terminal accounting. The original and continuation results each have a compact summary JSON. `sha256sums.txt` inventories this evidence. No raw samples or model binaries were copied, no reserved test data were used and no artifacts were promoted.
