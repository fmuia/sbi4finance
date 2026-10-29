The known-curve rough Bergomi fit on 100,000 training surfaces completed successfully as Slurm job **8059257** on gpu3001. Scheduler accounting reports COMPLETED with exit code 0:0, 1,478 seconds, four CPUs and one L40s GPU. The allocation ended at 17:09:36 UTC on 28 October 2026; the detached watcher reconciled it at 17:09:51 UTC. The posterior, price emulator and validation stages all completed. These artifacts remain development evidence without scientific acceptance or cache promotion.

The comparison below uses the same first 2,000 validation cases, 1,000 posterior draws per case, sampling seed, architecture and BayesFlow default adaptive sampler. All 5,000 validation surfaces have identical part hashes across the two bank snapshots; the original 50,000 training surfaces are also unchanged. Both fits use seed 0 and a predeclared final epoch of 500.

| Measurement | 50,000 training surfaces | 100,000 training surfaces |
| --- | ---: | ---: |
| Simultaneous marginal rank screen, including rho × eta | 4/4 pass | 4/4 pass |
| Largest rank discrepancy in that family | 0.02376 | 0.02199 |
| H: empirical coverage of 90% intervals | 89.30% | 89.45% |
| Eta: empirical coverage of 90% intervals | 89.45% | 89.40% |
| Rho: empirical coverage of 90% intervals | 89.85% | 89.85% |
| Rho × eta: empirical coverage of 90% intervals | 89.70% | 89.90% |
| Pricer RMSE, in quote-noise standard deviations | 0.08287 | 0.09843 |
| Pricer epochs run / selected | 300 / 298 | 116 / 86 |
| Whole allocation, GPU minutes | 20.15 | 24.63 |

The larger bank yields slightly smaller primary-parameter rank discrepancies and little change in coverage or recovery correlation. The derived rho × eta rank discrepancy increases slightly, while remaining within the same simultaneous bound of 0.03562. This single comparison does not demonstrate a material posterior improvement. The price emulator has 18.8% worse validation label RMSE under the unchanged early-stopping recipe. That is evidence that additional data alone does not guarantee a better optimized emulator; it does not establish that more data is harmful. Both label noise and emulator error contribute to the reported price discrepancies.

This is one seed, with nearly twice as many posterior gradient updates at the same 500 epochs: 98,000 versus 195,500. It is therefore not a replicated or fixed-compute learning curve. New posterior and pricer hashes also mean that earlier PyMC comparisons for the 50k artifacts do not automatically apply to the 100k artifacts. Reserved test data remain unused. Continuous-price accuracy, joint posterior agreement and numerical adequacy remain separate questions.

Actual completed identities are posterior `39f1a81986a38994dc6eacc520ad95254b34346abc02e758c08779c7c5916fc4`, pricer `9f0f695b0c369f6eaedb0c5559a46c254459047bd806fc14982377416a61a546`, and bank manifest `f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313`. The original training pool has now consumed 15,100 CPU-seconds and 3,775 GPU-seconds, within its original 32,400/8,100 ceilings.

Exact metadata bytes are stored under `known100k/` and `baseline50k/`: phase plans and execution journals, worker run records, posterior training records, pricer records and validation evaluations. `comparison-summary.json` contains the table values and uncertainty intervals; `bank-and-model-identity-proof.json` verifies the shared data and actual model files. `terminal-inventory.json` includes raw terminal accounting and the independent architecture pilot's still-pending state at 17:20:45 UTC. No new jobs were submitted for this collection step. `sha256sums.txt` inventories the evidence.
