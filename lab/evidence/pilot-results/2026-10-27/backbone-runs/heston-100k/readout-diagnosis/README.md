The remaining Heston initial-variance discrepancies are **not repaired by tighter FlowMatching readout tolerances on these three observations**. Native BayesFlow Tsit5 remains the tutorial default. No training, pricing, MCMC, new data or reserved-test evaluation occurred in this diagnostic.

The frozen 100k checkpoint was sampled on the same three existing observations with 10,000 draws and seeds `271026`, `271027`, `271028`. Native Tsit5 (`atol=1e-6`, `rtol=1e-4`) was paired with a call-time 10× tighter readout (`atol=1e-7`, `rtol=1e-5`). Native draws reproduced the earlier stored arrays bit-for-bit in all cases; the saved sampler policy was unchanged.

| Case | Initial-variance mean error, native → tighter (reference SD) | 90% width/reference width, native → tighter |
| --- | ---: | ---: |
| 0 | 0.482159 → 0.482835 | 1.136448 → 1.136679 |
| 1 | 0.067142 → 0.067170 | 1.088906 → 1.089742 |
| 2 | 0.652036 → 0.651986 | 1.355472 → 1.356108 |

Across all cases and parameters, the largest paired mean shift was **0.000676 reference SD**, and the largest relative interval-width change was **0.0768%**. The same one of three cases passed the existing proposed parameter rules under either readout. These remain development checks, not scientific acceptance.

One batch of three reference-posterior mean parameter vectors and their observed quotes passed the actual saved adapter/standardizer audit. Quote/noise scaling matched manually exactly; the largest ordered-logit difference was `2.38e-6`. Manual standardization differed by at most `2.38e-7`, and forward/inverse normalization passed the declared float32 `atol=rtol=2e-6` checks. Stored moments were unchanged after sampling. This audit supports consistency at these inputs; it does not exclude every training-data or optimization issue. The evidence points toward the learned approximation/optimization as the next diagnostic focus.

Job `8030926` completed with exit code `0:0` in **96 seconds: 384 allocated CPU-seconds and 96 GPU-seconds**. Only the top-level Slurm row is counted. The recorded pool totals are 17,660 CPU-seconds and 3,860 GPU-seconds, within the unchanged six CPU-hour / 1.25 GPU-hour ceiling. Source is `845c458b849c472271527cccf815ea6714f02206`; the prospectively bounded driver SHA is `2160278be2b167ca7551a76ceb8278514fec97d82e294cc98259c17b34825f6d`. All 58 frozen input/source/environment/library/driver hashes were verified unchanged before collection and the atomic terminal-ledger update.

Original reports, raw logs, accounting and the exact driver are retained here. `remote-arrays.json` records SHA256 and bytes for all seven arrays, whose payloads remain on Oscar. The original submission ledger is preserved; local `execution.json` matches Oscar's updated diagnostic ledger. `downloaded-files.sha256` verifies the local evidence files.
