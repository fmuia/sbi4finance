# Studies

Each study writes one directory of results; every directory must be new. The one exception is `scripts/calibration_bank.py local --resume`, which keeps the finished shards of an interrupted local bank build. The
numbers in `tutorial_final/CLAIMS.md` and in the [design document](../../docs/design/calibration-design-document.md)
come from these outputs. Library code is in `src/sbi4finance/calibration/`.

| ID | Script | What it produces |
|---|---|---|
| S1 | `scripts/calibration_bank.py` (`config`, `job-map`, `shard`, `assemble`, `local`), `cluster/calibration_bank_array.sbatch` | Calibration banks: the option_bank_v1 arrays plus the forward-variance context and the paths used per block |
| S1 | `bank_quality.py` | A bank's numerical error per quote against its budget, the paths-ladder rungs used, and the generation cost |
| S2 | `train_calibration.py`, `evaluate_calibration.py` | The amortized posterior; SBC, coverage, recovery and contraction on held-out surfaces |
| S3 | `heston_check.py` | The Heston network against the exact posterior: importance reweighting with the Fourier likelihood and random-walk chains |
| S4 | `pricer_route.py` | The neural pricer (deep calibration's forward map), its held-out error, and MCMC on it against the amortized posterior |
| S5 | `audit.py` (Stage B) | The audit of rough-Bergomi surfaces with the Monte Carlo likelihood |
| S6 | `classical_route.py` | Multi-start least squares, Gauss–Newton intervals and their coverage, distinct optima, Tikhonov |
| S7 | `products_study.py` | Product prices under the posterior, ranges, reserves and their drivers |
| S8 | `scripts/build_surface.py` (Stage C) | The real smile on the design grid, from a chain file |
| S9 | `costs.py` | CPU-seconds per route on one machine |
| S2-1, S2-2 | `smile_facts.py` (`example`, `prior-predictive`) | The example surfaces at the centre of the prior, and the prior predictive check of four smile statistics |
| – | `manifest.py` | Regenerates the artifact table of `tutorial_final/MANIFEST.md` from the files in `results/` and `checkpoints/` |

## Order and test-split use

Every study uses its own range of test surfaces, so no surface serves two checks:

| Study | Test rows |
|---|---|
| S2 | 0–1999 |
| S3 | 2000–2049 (Heston) |
| S4 | held-out pricer error on 0–2049; chains on 2050–2099 |
| S6 | 2100–2299 |
| S7 | 2300 onwards |
| S5 | 2500 onwards (rough) |

## Commands

Heston (Stage A), on one machine:

```bash
python scripts/calibration_bank.py local --config configs/tasks/calibration_heston_bank.json --chunk 500 --workers 4 --out BANK
python tutorial_final/studies/train_calibration.py --bank BANK --out RUN --epochs 40
python tutorial_final/studies/evaluate_calibration.py --bank BANK --checkpoint RUN --out S2
python tutorial_final/studies/heston_check.py --bank BANK --checkpoint RUN --out S3
python tutorial_final/studies/pricer_route.py --bank BANK --checkpoint RUN --out S4
python tutorial_final/studies/classical_route.py --bank BANK --checkpoint RUN --out S6
python tutorial_final/studies/products_study.py --bank BANK --checkpoint RUN --out S7
```

Rough Bergomi (Stage B), bank on Oscar:

```bash
python scripts/calibration_bank.py job-map --config configs/tasks/calibration_rough_bank.json --chunk 50 --out jobs.json
sbatch --array=0-2239%200 [account/partition flags] cluster/calibration_bank_array.sbatch "$(which python)" jobs.json /scratch/$USER/rough-shards
python scripts/calibration_bank.py assemble --config configs/tasks/calibration_rough_bank.json --shards /scratch/$USER/rough-shards --out /scratch/$USER/rough-bank
```

then the same study commands with `--pricer S4/pricer` for S6 and S7.
