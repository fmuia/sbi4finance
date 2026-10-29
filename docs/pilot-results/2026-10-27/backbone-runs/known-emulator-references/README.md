These are **approximate forward-emulator PyMC references**, not independent model-price truth or accepted tutorial results. They use three fixed development observations (`validation:0`, `validation:1`, `validation:2`) from the known-curve rough Bergomi bank. Each case holds its recorded seven curve levels fixed.

PyMC 6.3.2's native DEMetropolisZ sampled four chains per case, each with 4,000 tuning draws and 8,000 retained draws. Initialization uses data-only multistart fits and dispersed starts; no neural posterior supplies starts or proposal geometry. Individual optimizer budget failures remain visible in the initialization records. A finite multistart search does not establish global mode coverage.

All parameter and derived `rho × eta` diagnostics passed the declared `R-hat ≤ 1.01`, bulk ESS ≥ 400 and tail ESS ≥ 400 screen:

| Observation | Worker elapsed | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS |
| --- | ---: | ---: | ---: | ---: |
| `validation:0` | 71 s | 1.002151 | 2,947 | 3,203 |
| `validation:1` | 65 s | 1.002203 | 2,817 | 3,542 |
| `validation:2` | 61 s | 1.002294 | 2,786 | 3,311 |

The likelihood uses a fixed fitted emulator mean and Gaussian quote noise. The amortized estimator trains on numerical Monte Carlo bank prices plus fresh quote noise; the reference does not propagate emulator bias or bank Monte Carlo randomness/covariance. Convergence and emulator agreement therefore do not establish independent pricing accuracy, like-for-like simulator-posterior agreement, or scientific acceptance. Independent pricing/refinement and posterior checks remain pending.

The sampler/pricer source is commit `845c458b849c472271527cccf815ea6714f02206`. All ten originally frozen inputs were verified unchanged before collection and again before the terminal ledger update. Array job `8030892` completed all three workers with exit code `0:0`: **788 allocated CPU-seconds (0.2189 CPU-hours), zero GPU**. Accounting counts each array element once, excluding duplicate `.batch`/`.extern` rows. The separate six CPU-hour pilot ceiling and metadata reserve remain in place.

Original records, diagnostics, initialization extracts, raw logs and accounting are retained here. `remote-artifacts.json` records SHA256 and bytes for chains, partial draws and pricer inputs; their payloads remain on Oscar. `execution-submitted.json` preserves the original ledger; `execution.json` matches the terminal ledger atomically copied back to Oscar. `downloaded-files.sha256` verifies the local evidence files.
