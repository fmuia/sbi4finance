# Claims

Every number said on stage or posted, with the study that produces it and its
status. A claim is on screen only when its status is **verified**. Values are
filled from the study outputs; nothing here is typed by hand.

| # | Claim | Study | Output | Value | Status |
|---|---|---|---|---|---|
| C1 | The SPY ATM skew follows a power law in T with exponent H − ½ ≈ −0.4 | S8 | `results/calibration/spy/surface.json` | — | blocked: no chain source reachable (see the plan, §15) |
| C2 | The surface identifies ρη and H; η alone has a posterior ridge truncated at ρ = −1 | S2, S6 | `results/calibration/rough/s2/validation.json` | — | pending: rough bank on Oscar |
| C3 | The variance swap's posterior range is zero; the volatility swap's and the forward-start's are not; the forward-start reserve is X% of its price | S7 | `results/calibration/rough/s7/products.json` | — | pending: rough bank |
| C4 | Multi-start least squares lands on N distinct optima along the ridge; the regularized fit picks one by its penalty | S6 | `results/calibration/rough/s6/classical.json` | — | pending: rough bank |
| C5 | The amortized posterior agrees with the exact Heston posterior to within 0.3 posterior sd and 25% in width | S3 | `results/calibration/heston/s3/heston-check.json` | — | running |
| C6 | The amortized posterior and MCMC on the neural pricer agree within the same criterion | S4 | `results/calibration/*/s4/pricer-route.json` | — | pending |
| C7 | SBC passes on 2,000 held-out surfaces; the audit flags Y% of validation surfaces and passes the real one | S2, S5 | `results/calibration/*/s2/validation.json` | — | running (Heston), pending (rough) |
| C8 | One exact Monte Carlo likelihood evaluation costs A CPU-s; the exact posterior B CPU-hours; the network C s; the audit D s | S9 | `results/calibration/costs/costs.json` | — | pending |
| C9 | Heston and rough Bergomi calibrated to the same smile price the forward-start Z% apart | S7 | `results/calibration/*/s7/products.json` | — | pending |
