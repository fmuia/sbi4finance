# Manifest

What `tutorial_final/` holds and where it came from. Copied artifacts carry
their source commit and SHA-256; the library is imported, not copied.

## Library imported by the slot-5 material

| Module | Role |
|---|---|
| `src/sbi4finance/calibration/surface.py` | Black–Scholes in forward units, implied volatility, the design grid and noise, smile statistics, replication |
| `src/sbi4finance/calibration/pricing.py` | Rough Bergomi surface pricer (hybrid scheme, conditional estimator, antithetic paths, control variates, blocks); Heston Fourier surface |
| `src/sbi4finance/calibration/tasks.py` | The calibration task family, its validation, context prior and pricing dispatch |
| `src/sbi4finance/calibration/bank.py` | Calibration banks, shards and assembly |
| `src/sbi4finance/calibration/inference.py` | Adapter and FlowMatching workflow |
| `src/sbi4finance/calibration/correction.py` | The audit: proposals, likelihood, weights, k-hat, reference chains |
| `src/sbi4finance/calibration/validation.py` | SBC, coverage, contraction, recovery, agreement |
| `src/sbi4finance/calibration/pricer_net.py` | The neural pricer and MCMC on it |
| `src/sbi4finance/calibration/classical.py` | Least squares, Gauss–Newton, multi-start, Tikhonov |
| `src/sbi4finance/calibration/products.py` | Product prices under posterior draws |
| `src/sbi4finance/calibration/chain.py` | A real chain to the design grid |
| `src/sbi4finance/calibration/plots.py` | Figures |

Reused unchanged from the archive, by import: `rough_bergomi.simulate_rough` and
its hybrid driver, `bergomi.conditional_put_payoffs` and `curve_values`,
`hkde.european_price` and `characteristic_function`, `gbm_bank` helpers,
`svj.plots`, `svj.pmmh.rhat_ess`. Their source hashes are pinned by archived
evidence and are not edited.

## Bank configurations

| File | Model | Splits (train / validation / test) | Seed |
|---|---|---|---|
| `configs/tasks/calibration_heston_bank.json` | Heston (HKDE, jumps off) | 50,000 / 5,000 / 5,000 | 2610 |
| `configs/tasks/calibration_rough_bank.json` | rough Bergomi | 100,000 / 5,000 / 7,000 | 2611 |
| `configs/tasks/calibration_rough_pilot.json` | rough Bergomi, pilot | 1,600 / 200 / 400 | 2612 |

## Copied artifacts

None yet. Results enter `results/` and checkpoints `checkpoints/` when their
studies are accepted, each with its SHA-256 here.
