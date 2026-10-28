# Known-curve rough Bergomi: completed 50k baseline

Oscar job **8046588 completed successfully, exit 0:0**. Posterior fitting took 309 seconds, pricing-emulator fitting 79 seconds, and validation 818 seconds. The allocation charged **4,836 CPU-seconds / 1,209 GPU-seconds** to the original pooled training allowance; the joint route remains waiting for its checked bank.

The posterior uses 50,000 training surfaces, final-epoch selection at epoch 500, and exact same-seed reload verification. The validation-best epoch 477 is retained separately. The bank, model and evaluation hashes agree across the attached records.

All four simultaneous rank screens pass on **2,000 validation cases**, with 1,000 joint posterior draws per case. Targets are H, eta, rho and the downstream product rho × eta. KS statistics are 0.02276, 0.02376, 0.02194 and 0.01562, below the declared 0.03562 threshold.

**Qualification:** nominal 95% intervals cover 94.55%, 94.45%, 94.85% and **93.85%** respectively. The product's pointwise Wilson interval is 92.71%–94.82%, so retain its undercoverage despite the passing rank screen. This is development evidence, not proof of full joint calibration or conditional reference agreement.

The pricer's validation-label RMSE is 0.08287 quote-noise standard deviations. This measures agreement with the bank's noisy numerical labels; it does not replace independent-price or simulator-grid checks. Reserved test outcomes were not evaluated and scientific acceptance remains false.

[Worker completion](run.json) · [Posterior recipe and selection](posterior-training.json) · [Pricer report](pricer-run.json) · [Full validation report](evaluation.json) · [Terminal scheduler accounting](terminal-accounting.psv) · [Reconciled watcher receipt](execution-terminal.json).
