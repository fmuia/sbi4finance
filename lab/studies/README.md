# Studies

Each study writes one directory of results; every directory must be new. The one exception is `scripts/calibration_bank.py local --resume`, which keeps the finished shards of an interrupted local bank build. The
numbers in `tutorial/CLAIMS.md` and in the [course](../../course/README.md)
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
| S7 | `across_models.py` | The same rough-Bergomi smiles under Heston: the Heston posterior (the network corrected with the Fourier likelihood), both least-squares fits with their chi-square, and the products under both posteriors |
| S8 | `scripts/build_surface.py` (Stage C) | The real smile on the design grid, from a chain file |
| S9 | `costs.py` | CPU-seconds per route on one machine |
| S1-G | `grid_bias.py` | The bank's time-grid bias: each surface priced on the bank's grid and on one twice as fine |
| S2-T | `train_calibration.py --thin-layout`, `evaluate_thin.py` | The thin-chain network on a real chain's layout and noise (EWG), which infers the forward-variance curve; its SBC at a noise scale |
| S8 | `real_smile.py` | SPY under the liquid network: posterior, audit, predictive check, products |
| S10 | `thin_coverage.py` (`run`, `summarize`), `cluster/thin_coverage_array.sbatch` | Breach rates of every method's one-sided 90% bounds on simulated chains with the thin layout; with `--truths`, truths drawn from S11's posterior (S10-E) |
| S11 | `thin_real.py` | A real thin chain (EWG): the audited posterior, products, classical fits, the predictive check, at several noise scales |
| S2-1, S2-2 | `smile_facts.py` (`example`, `prior-predictive`) | The example surfaces at the centre of the prior, and the prior predictive check of four smile statistics |
| – | `manifest.py` | Refreshes the copies of quick-check outputs, task files and figures in `tutorial/assets/`, then regenerates the artifact table of `tutorial/MANIFEST.md` from every file there |
| – | `copy_to_tutorial.py` | Refreshes the tutorial's copies of lab code in `tutorial/src/sbi4f_tutorial/` (`pixi run copy-to-tutorial`; `--add MODULE.NAME` copies a new function, `--check` only reports the copies that differ) |

## Order and test-split use

Every study uses its own range of test surfaces, so no surface serves two checks:

| Study | Heston and full rough bank: test rows | Rough pilot bank (400 test surfaces): test rows |
|---|---|---|
| S2 | 0–1999 | 0–249 |
| S3 | 2000–2049 (Heston) | – |
| S4 | held-out pricer error on 0–2049; chains on 2050–2099 | – |
| S5 | 2500–2699 (rough) | 250–289 |
| S6 | 2100–2299 | 293–300 |
| S7 | 2300–2499 | 290–292 |
| S10 | 3000–3999 (rough) | – |
| S2-T | 4000–5999 (rough) | – |
| S1-G | 6000–6049 (rough) | – |

S4's pricer error on rows 0–2049 overlaps S2's rows; it is a different estimator, so no result is biased. The
prior predictive check (`smile_facts.py`) also reads rows 0–999, for description only.

## Commands

The runs still missing on the full bank, with their exact commands, costs and acceptance checks, are in
[`plans/CLUSTER-RUNS.md`](../plans/CLUSTER-RUNS.md). The commands below are the ones behind the accepted results; each run
directory's JSON records its settings again. The pilot's S5 (22 October) priced all proposals of a surface on one
seed and must be rerun with the current `audit.py`.

Heston (Stage A), on one machine (4 cores):

```bash
python scripts/calibration_bank.py local --config configs/tasks/calibration_heston_bank.json --chunk 500 --workers 4 --out BANK
python studies/bank_quality.py --bank BANK --shards BANK-shards --out S1
python studies/smile_facts.py prior-predictive --bank BANK --out PRIOR-PREDICTIVE
python studies/train_calibration.py --bank BANK --out RUN --epochs 500 --batch-size 256 \
    --fresh-noise --seed 0 --chunk-epochs 25 --time-alpha -0.5
python studies/heston_check.py --bank BANK --checkpoint RUN --out S3 --first 2000 --surfaces 50 \
    --proposals 2048 --chain-surfaces 4 --chains 4 --steps 6000 --burn 1000 --workers 4
python studies/evaluate_calibration.py --bank BANK --checkpoint RUN --out S2 --surfaces 2000 --draws 1000
python studies/pricer_route.py --bank BANK --checkpoint RUN --out S4
python studies/classical_route.py --bank BANK --checkpoint RUN --out S6 --surfaces 200
python studies/products_study.py --bank BANK --checkpoint RUN --out S7 --surfaces 3 --draws 400
python studies/costs.py --out COSTS --heston-checkpoint RUN --heston-bank BANK --heston-check S3/heston-check.json
```

Rough Bergomi pilot (2,200 surfaces, `configs/tasks/calibration_rough_pilot.json`), on the same machine:

```bash
python scripts/calibration_bank.py local --config configs/tasks/calibration_rough_pilot.json --chunk 25 --workers 2 --out PILOT --resume
python studies/bank_quality.py --bank PILOT --shards PILOT-shards --out S1
python studies/smile_facts.py prior-predictive --bank PILOT --out PRIOR-PREDICTIVE --split test --surfaces 400
python studies/train_calibration.py --bank PILOT --out RUN --epochs 1000 --batch-size 128 \
    --fresh-noise --seed 0 --chunk-epochs 100 --time-alpha -0.5
python studies/evaluate_calibration.py --bank PILOT --checkpoint RUN --out S2 --surfaces 250 --draws 1000
python studies/audit.py --bank PILOT --checkpoint RUN --out S5 --first 250 --surfaces 40 --proposals 256 \
    --paths 8192 --workers 4
python studies/products_study.py --bank PILOT --checkpoint RUN --out S7 --first 290 --surfaces 3 --draws 400 \
    --starts 4 --least-squares-paths 8192 --workers 4
python studies/classical_route.py --bank PILOT --checkpoint RUN --out S6 --first 293 --surfaces 8 --starts 10 \
    --least-squares-paths 8192 --landscape-points 21 --workers 4
python studies/across_models.py --rough-bank PILOT --rough-products S7 --heston-checkpoint HESTON-RUN \
    --out S7-ACROSS
```

`accept.py` then copies a finished run into `tutorial/assets/results/calibration/` (`--keep-surfaces 20`
trims the stored draws), and `claims.py` and `manifest.py` regenerate the claims table and
the artifact table.

Rough Bergomi (Stage B), bank on Oscar (112,000 surfaces in 2,240 single-core jobs of 50):

```bash
python scripts/calibration_bank.py job-map --config configs/tasks/calibration_rough_bank.json --chunk 50 --out jobs.json
sbatch --array=0-2239%200 [account/partition flags] cluster/calibration_bank_array.sbatch "$(which python)" jobs.json /scratch/$USER/rough-shards
python scripts/calibration_bank.py assemble --config configs/tasks/calibration_rough_bank.json --shards /scratch/$USER/rough-shards --out /scratch/$USER/rough-bank
```

then the Heston study commands with the full bank's test rows, and `--pricer S4/pricer` for S6 and S7.
