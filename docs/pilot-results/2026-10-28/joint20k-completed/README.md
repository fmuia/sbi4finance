# Joint-curve rough Bergomi: completed 20k baseline

Oscar job **8053365 completed successfully, exit 0:0**, including posterior fitting (185 seconds), pricing-emulator fitting (41 seconds) and 2,000-case validation (848 seconds). Allocation cost was **4,352 CPU-seconds / 1,088 GPU-seconds**. Both routes in the original intermediate-training phase are now complete.

The six-parameter posterior uses the checked 20k / 5k / 7k snapshot, final epoch 500, and exact same-seed reload verification. The validation-best epoch 475 is retained separately. Model, task, bank and evaluation identities agree across the attached records.

**All seven simultaneous rank screens pass**, including rho × eta computed downstream from joint draws: maximum KS 0.02805 versus the declared 0.03753 bound on 2,000 validation cases with 1,000 draws each.

Coverage still needs review. Examples include:

| Target | Nominal coverage | Observed coverage |
| --- | --- | --- |
| eta | 95% | 94.00% |
| short variance | 95% | 96.35% |
| decay | 50% | 46.75% |

Short-variance 80% and 90% intervals also overcover (82.3% and 92.0%). The full report retains every coverage level and pointwise Wilson interval. These exploratory pointwise checks are not adjusted for multiple comparisons; they are qualifications alongside the passing simultaneous rank screen, not a completed scientific acceptance decision.

The pricer's validation-label RMSE is **0.101 quote-noise standard deviations**, but its worst individual quote error is **3.888** and worst surface RMS is **1.522** in the same units. Average error alone is insufficient to trust an emulator-based likelihood. These comparisons use finite-grid bank labels; independent-price, conditional PyMC/reference, grid-refinement and downstream-product checks remain unfinished.

Reserved test outcomes were not evaluated. Scientific acceptance remains false. Large checkpoints and full draw panels remain on Oscar.

[Terminal summary](summary.json) · [Worker completion](run.json) · [Posterior training](posterior-training.json) · [Pricer report](pricer-run.json) · [Full validation report](evaluation.json) · [Official accounting](terminal-accounting.psv) · [Both routes completed](execution-terminal.json).
