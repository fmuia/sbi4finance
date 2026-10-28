Evaluation-only continuation **8061867** was accepted at 18:05:36 UTC on 28 October 2026. At 18:06:23 UTC it was RUNNING on gpu3102, with the saved DeepSet checkpoint being evaluated. This snapshot does not claim completed continuation results.

The original pilot trained **all six models successfully** for 200 epochs and passed their final-checkpoint reload checks. Four evaluations completed. DeepSet and SetTransformer evaluation subprocesses exceeded the original driver's 65-second limit; the driver then marked the screen partial and exited with code 1. Slurm job 8059371 ended FAILED at 17:57:31 UTC after 1,231 seconds. That failure and all original outputs remain unchanged.

| Completed original variant | Marginal rank checks passed | Empirical coverage of nominal 90% intervals, across four targets |
| --- | ---: | --- |
| Flow matching, vector | 4/4 | 84–89% |
| Affine coupling, vector | 3/4 | 90.5–98% |
| Spline coupling, vector | 4/4 | 88–92.5% |
| MeanFlow, vector | 4/4 | 80.5–86% |

These are preliminary measurements on 200 validation cases. In particular, the loose marginal rank screen does not override MeanFlow's visible undercoverage. No architecture winner or scientific acceptance is declared. The existing development task is four-parameter Heston represented through zero-jump HKDE with kappa fixed at 2; results should not be generalized to the primary five-parameter task.

The continuation evaluates only the two saved final-200 checkpoints, using the same first 200 validation cases, 256 draws, seed 0, batch size 50 and BayesFlow native adaptive sampler. Each evaluator gets at most 240 seconds within one ten-minute allocation. No model is retrained, no solver setting is changed and no original output is overwritten. It cannot submit another job or automatically retry an ambiguous submission.

Fresh accounting charged the original failed allocation 4,924 CPU-seconds and 1,231 GPU-seconds. The new four-CPU, one-L40s-GPU, 32-GiB job reserves at most 3,360 CPU-seconds and 840 GPU-seconds including cleanup. Conservative cumulative exposure is therefore 8,284/2,071, within the original 15,360/3,840 pilot ceilings; no allowance was increased.

`original/` preserves the exact original plan, execution record, partial `run.json` and four completed evaluations. `checkpoint-identities.json` and `preflight.json` bind all six saved models and training records. `prior-accounting.json` preserves terminal failed-allocation accounting. `plan.json`, the launch receipts and `queue-verification.json` describe the new continuation; `original-results-summary.json` provides the existing partial results in compact form. `sha256sums.txt` inventories this evidence. No raw samples or model binaries were copied, no reserved test data were used and no artifacts were promoted.
