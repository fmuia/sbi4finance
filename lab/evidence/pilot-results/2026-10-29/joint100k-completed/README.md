The joint rough Bergomi 100k fit **completed successfully** as Slurm job 8080084 on gpu3006. It ended at 01:35:58 UTC on 29 October 2026 after 1,713 seconds, using four CPUs and one GPU, with state COMPLETED and exit code 0:0. The posterior completed 500 epochs with predeclared final-epoch selection; the pricer ran 300 epochs and selected epoch 297. Validation completed on 2,000 cases with 1,000 draws each. The bank is the checked 100k/5k/7k snapshot. No scientific acceptance or artifact promotion is claimed.

The 20k comparison uses the same task, collection, root seed, source, runtime, validation IDs, sampling seed and draw count. Actual manifest and payload hashes were verified for the original 20,000 training rows and all 5,000 validation rows, and their references agree across both snapshots. No examples were regenerated for this comparison.

| Measurement | Joint 20k | Joint 100k |
| --- | ---: | ---: |
| Six-primary-parameter marginal rank screen | 6/6 pass | 6/6 pass |
| Seven-target rank screen including rho × eta | 7/7 pass | 7/7 pass |
| Largest discrepancy in seven-target family | 0.02805 | 0.02897 |
| Short-variance coverage of nominal 90% intervals | 92.00% | 90.15% |
| Range of 90% coverage across six primary parameters | 89.70–92.00% | 89.15–90.15% |
| Pricer validation RMSE, quote-noise SD | 0.10118 | 0.08076 |
| Worst absolute quote error, quote-noise SD | 3.88847 | 1.04655 |
| Pricer epochs run / selected | 300 / 295 | 300 / 297 |
| Whole allocation, GPU minutes | 18.13 | 28.55 |

The larger fit brings short-variance coverage closer to nominal and lowers pricer label RMSE by 20.2%. Most other posterior diagnostics change modestly; the maximum rank discrepancy is slightly higher and remains below the same 0.03753 simultaneous bound. This does not establish uniform posterior improvement, full joint-posterior accuracy or continuous-price correctness.

Both posterior fits use one seed and 500 epochs, but their optimizer exposure differs: 39,500 versus 195,500 gradient updates, approximately 4.95 times as many. This is not a replicated or fixed-compute learning curve. Pricer patience changed explicitly from 30 to 300; **both joint pricers actually ran the full 300 epochs**, so there was no observed early-stop truncation in either fit. The patience change should not be described as an active early-stopping difference here. Existing downstream reference or emulator acceptance cannot automatically transfer to the new model hashes. Reserved test data remain unused.

The 100k posterior stage took 674.70 seconds, pricer 134.27 seconds and validation 888.59 seconds. Its allocation consumed 6,852 CPU-seconds and 1,713 GPU-seconds. Cumulative use of the original training pool is 22,484 CPU-seconds and 5,621 GPU-seconds, below its 32,400/8,100 ceilings. These timings include data loading, training and the existing validation workflow; the comparison does not rerun inference or pricing.

Completed identities are bank `eb72588b9b7665e0e416cb5d3a1bcf8cd78a1c97f2b43f18ea8246c749a871d8`, posterior `19c24d64290651f592d3c13d7d0a1616d0d8ae080d487d0298aee8f97779498b` and pricer `be699c31a7950ca3ebd9117275b4f8cb1a8355c96f7ca1e18bcfa36e0d252301`. Actual model files matched their run records.

`joint100k/` and `joint20k/` preserve the exact phase, training, pricer and evaluation JSONs. `comparison-summary.json` contains per-target rank/coverage values and uncertainty intervals. `data-model-and-accounting-proof.json` records verified data/model hashes and raw terminal allocation accounting. `sha256sums.txt` inventories this compact metadata evidence. No model binaries or posterior-draw arrays were copied, no new jobs were submitted and no fitting, selection, MCMC, tests or promotion were performed for this collection.
