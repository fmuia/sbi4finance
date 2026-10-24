# Cluster runs still missing

24 October 2026. For whoever runs Oscar jobs next. Every run below is missing: nothing in
`tutorial_final/results/` comes from the full bank yet. The plan behind the runs is
[`docs/course/v2-plan.md`](../docs/course/v2-plan.md); the scripts are described in
[`studies/README.md`](studies/README.md).

All costs are in CPU-hours on this project's 2-core container; Oscar's cores may be up to about 30% faster or
slower. Every script that takes `--out` needs a new directory, except the S10 array, which appends to its own.

**For the weekend of 24–25 October.** Run R0 to R4 first, in order. If an acceptance check fails, stop and report
it to Francesco Muia with the run's JSON; do not change the code in the run checkout. Every new or changed Python
command below was run end to end on a 68-surface test bank, with networks trained for three epochs. The numbers of
those runs mean nothing, but the commands, outputs and summaries work. The `sbatch` wrappers could not be tested
off Oscar.

## The runs at a glance

| Run | Study | What it produces | Cost | Needs |
| --- | --- | --- | --- | --- |
| R0 | – | Oscar environment check: the 200-surface mini-pilot | 0.6 CPU-h | – |
| R1 | S1 | The full rough-Bergomi bank: 112,000 surfaces | about 360 CPU-h | R0 |
| R2 | S1, S1-G | The bank's quality, its prior predictive check, its time-grid bias | about 2 CPU-h | R1 |
| R3 | S2 | The liquid network (the curve given) and its SBC on 2,000 surfaces | 1 GPU job, 1 CPU-h | R1 |
| R4 | S2-T | The thin network on EWG's layout (the curve inferred) and its SBC at three noise scales | 1 GPU job, 2 CPU-h | R1 |
| R5 | S5 | The audit of the liquid network on 200 surfaces | about 190 CPU-h | R3 |
| R6 | S8 | SPY's posterior, audit, predictive check and products | about 2 CPU-h | R3 |
| R7 | S11 | EWG's posterior, products, classical fits and predictive check, at three noise scales | about 3 CPU-h | R4 |
| R8 | S10 | Breach rates of every method's reserves on 1,000 simulated chains with EWG's layout | about 160 CPU-h | R4 |
| R9 | S10-E | The same on 300 chains whose truths come from EWG's posterior | about 50 CPU-h | R7 |
| R10 | S4, S6, S7, S9 | The neural-pricer route, the classical route, products, costs | under 20 CPU-h | R3 |

The total is about 790 CPU-hours and two GPU jobs. R3 and R4 can run together, and so can R5 to R8.

## Before anything

1. **Lanes.** Follow `.claude/skills/cluster-setup/SKILL.md`: open the ssh control socket, run
   `pixi run python cluster/discover_cluster.py --ssh-host oscar`, and pass the account and partition you hold
   with `--account` and `--partition` on every `sbatch` line below. The `#SBATCH` defaults in `cluster/` are the
   lab's.
2. **One frozen checkout.** Clone the repository under the lab's data volume, check out the commit that holds this
   file, and run `pixi install` and `pixi install -e gpu`. Never edit that checkout while jobs run. Shards refuse
   to run if the config file changed, assembly refuses shards whose source files differ, and each shard records
   the git commit it ran on (`run.json`).
3. **Checks.**
   ```bash
   pixi run python -c "import json; from sbi4finance.calibration import bank; \
       print(bank.task_fingerprint(json.load(open('configs/tasks/calibration_rough_bank.json'))))"
   # must print d338d49829889af5c95f89c02e840d293422a55221e8bb65cb999ff8db73b45a
   PYTHONPATH=src pixi run python -m pytest -q tests/test_calibration.py tests/test_thin.py
   ```
4. **Paths used below.** `export RUNS=/oscar/data/<lab>/$USER/sbi4finance-runs` and `BANK=$RUNS/rough-bank`.
   Run every command from the checkout's root.

## R0. Environment check: the mini-pilot

```bash
sbatch --cpus-per-task=16 --time=01:00:00 cluster/study.sbatch python scripts/calibration_bank.py local \
    --config configs/tasks/calibration_rough_minipilot.json --chunk 25 --workers 16 --out $RUNS/minipilot
```

Accept when it finishes with about 11.5 CPU-seconds per surface (the last line prints it; the container measured
11.54). If Oscar's figure differs, scale every cost below by the same ratio. On the container the mini-pilot
passed gate G0′: no quote above the Monte Carlo budget, and SPY's four smile statistics at ranks 0.28, 0.25, 0.79
and 0.83 of the prior predictive.

## R1. S1: the full bank

```bash
pixi run python scripts/calibration_bank.py job-map --config configs/tasks/calibration_rough_bank.json --chunk 50 \
    --out $RUNS/jobs.json
sbatch --array=0-2239%200 cluster/calibration_bank_array.sbatch "$(pixi run which python)" $RUNS/jobs.json \
    $RUNS/rough-shards
```

- 2,240 single-core tasks of 50 surfaces, about 10 minutes each on average; the slowest about an hour (limit 2
  hours).
- If `scontrol show config | grep MaxArraySize` is below 2,240, submit in parts with the offset argument, for
  example `--array=0-999` with offset 0, then `--array=0-999` with offset 1000, then `--array=0-239` with offset 2000.
- A requeued or resubmitted task keeps a finished shard and redoes a partial one.
- When every task has finished, assemble in a job with enough memory (about 7 GB in use):
  ```bash
  sbatch --cpus-per-task=2 --mem=32G --time=02:00:00 cluster/study.sbatch python scripts/calibration_bank.py \
      assemble --config configs/tasks/calibration_rough_bank.json --shards $RUNS/rough-shards --out $BANK
  ```

Accept when assembly succeeds. It checks exact coverage of every split, the configuration, the source hashes and
every array.

## R2. S1 and S1-G: checks of the bank

```bash
sbatch --mem=48G --time=02:00:00 cluster/study.sbatch python tutorial_final/studies/bank_quality.py \
    --bank $BANK --shards $RUNS/rough-shards --out $RUNS/s1
sbatch --mem=16G --time=01:00:00 cluster/study.sbatch python tutorial_final/studies/smile_facts.py \
    prior-predictive --bank $BANK --out $RUNS/s1-prior-predictive --split test --surfaces 1000
sbatch --time=02:00:00 cluster/study.sbatch python tutorial_final/studies/grid_bias.py --bank $BANK \
    --out $RUNS/s1g --first 6000 --surfaces 50 --workers 16
```

Accept when these hold:

- `bank-quality.json`: `share_of_surfaces_with_a_quote_above_budget` is 0. The 99% quantile of
  `mc_covariance_largest_eigenvalue_over_noise_variance` is below 0.5 (the mini-pilot's maximum was 0.24).
- `prior-predictive.json`: SPY's four statistics lie inside the central 95% of the clean distribution.

`grid-bias.json` has no threshold. It states how far the bank's discretized model is from a grid twice as fine,
in units of the quote noise. A review found shifts up to about 0.5 of the noise sd at H = 0.10. The course reports
the number.

## R3. S2: the liquid network

```bash
sbatch cluster/train_calibration.sbatch --bank $BANK --out $RUNS/s2-train --epochs 300 --batch-size 256 \
    --fresh-noise --chunk-epochs 25 --time-alpha -0.5 --seed 0
sbatch --mem=16G --time=02:00:00 cluster/study.sbatch python tutorial_final/studies/evaluate_calibration.py \
    --bank $BANK --checkpoint $RUNS/s2-train --out $RUNS/s2 --surfaces 2000 --draws 1000
```

- Look at the validation loss in `training.json`. If it still falls over the last tenth of the epochs, train
  again with `--epochs 600`.
- Accept (gate G2) when `validation.json` has every target of the SBC inside the simultaneous band: H, η, ρ and
  ρη, on test rows 0 to 1999.

## R4. S2-T: the thin network on EWG's layout

```bash
sbatch cluster/train_calibration.sbatch --bank $BANK --out $RUNS/s2t-train --thin-layout configs/thin/ewg_layout.json \
    --widths 512,512,512,512 --epochs 300 --batch-size 256 --chunk-epochs 25 --time-alpha -0.5 --seed 0
for s in 1 0.5 2; do
  sbatch --mem=16G --time=02:00:00 cluster/study.sbatch python tutorial_final/studies/evaluate_thin.py \
      --bank $BANK --checkpoint $RUNS/s2t-train --out $RUNS/s2t-scale-$s --first 4000 --surfaces 2000 --scale $s
done
```

The network infers H, η, ρ and the seven forward-variance levels from EWG's 27 layout quotes. Its noise is EWG's
own half-spreads, times a scale s drawn in training from 0.5 to 2. Accept (gate G2-T) when, at s = 1, all eleven
quantities of the SBC (the ten targets and ρη) lie inside the band. Report s = 0.5 and s = 2 as they come.

## R5. S5: the audit of the liquid network

```bash
sbatch --cpus-per-task=32 --mem=32G --time=12:00:00 cluster/study.sbatch python tutorial_final/studies/audit.py \
    --bank $BANK --checkpoint $RUNS/s2-train --out $RUNS/s5 --first 2500 --surfaces 200 --proposals 512 \
    --paths 32768 --workers 32
```

- Each proposal is priced on its own seed. The pilot's audit priced all of a surface's proposals on one seed,
  which biases the reweighted posterior.
- An interrupted run resumes from `s5.partial.jsonl`: resubmit the same command after deleting the empty `s5`
  directory.
- Start with `--surfaces 50`, about 45 CPU-hours, and look at `unpriced_proposals_total` and the cost per surface
  before running the rest.
- Accept when fewer than 1% of proposals are unpriced. The flag rate and the network's error on the reliable audits
  are the results.

## R6. S8: SPY

```bash
sbatch --time=04:00:00 cluster/study.sbatch python tutorial_final/studies/real_smile.py \
    --checkpoint $RUNS/s2-train --surface tutorial_final/results/calibration/spy/surface.json --out $RUNS/s8 \
    --workers 16
```

Accept (gate G4) when the audit's k-hat is at most 0.7. Otherwise rerun with `--proposals 4096`. The predictive
check is a result, not a gate: SPY's grid values are spline readings that the likelihood treats as independent
quotes.

## R7. S11: EWG's own chain

```bash
sbatch --time=04:00:00 cluster/study.sbatch python tutorial_final/studies/thin_real.py \
    --checkpoint $RUNS/s2t-train --observation tutorial_final/checks/thin_chains/EWG_observation.json \
    --out $RUNS/s11 --scales 1,0.5,2 --workers 16
```

Accept when k-hat at s = 1 is at most 0.5 after at most four rounds. Otherwise rerun with `--proposals 4096`.
The data are EWG's 61 cleaned quotes of 22 October 2026 at their own strikes and trading days, with the carry
forward and NYSE trading days (`checks/thin_chains/build_layout.py`).

## R8. S10: coverage on simulated thin chains

```bash
sbatch --array=0-4 cluster/thin_coverage_array.sbatch $BANK $RUNS/s2t-train $RUNS/s10 3000 100
# after a look at the first 500 chains:
sbatch --array=5-9 cluster/thin_coverage_array.sbatch $BANK $RUNS/s2t-train $RUNS/s10 3000 100
pixi run python tutorial_final/studies/thin_coverage.py summarize --out $RUNS/s10
```

- About 9.5 CPU-minutes a chain, so one task of 100 chains takes about an hour on 16 cores.
- Methods compared: the network alone, the audited posterior, Gauss–Newton with the curve flat or power-law beyond
  the last quote, and H fixed at 0.10.
- Accept when at least 90% of chains have a reliable audit (`reliable_chains`), and when the product seed check
  moves the bounds by less than a tenth of their width.

## R9. S10-E: coverage near EWG

```bash
for i in 0 1 2; do
  sbatch --cpus-per-task=16 --time=04:00:00 cluster/study.sbatch python tutorial_final/studies/thin_coverage.py run \
      --truths $RUNS/s11/draws.npz --checkpoint $RUNS/s2t-train --out $RUNS/s10e --start $((100*i)) \
      --stop $((100*i+100)) --workers 16
done
pixi run python tutorial_final/studies/thin_coverage.py summarize --out $RUNS/s10e
```

The truths are drawn from EWG's audited posterior, not from the prior. The posterior's coverage is then not
guaranteed by construction, which makes R9 the robustness check of R8. Accept as for R8.

## R10. S4, S6, S7, S9 on the full bank

Use the commands of `studies/README.md` (section "Rough Bergomi (Stage B)") with the full bank's test rows, and
`--pricer` from S4 for S6 and S7. The table there gives each study's rows.

## After the runs

Copy each accepted run into the repository and regenerate the tables:

```bash
python tutorial_final/studies/accept.py --checkpoint $RUNS/s2-train --to checkpoints/rough
python tutorial_final/studies/accept.py --checkpoint $RUNS/s2t-train --to checkpoints/rough-thin-ewg
python tutorial_final/studies/accept.py --run $RUNS/s2 --to results/calibration/rough/s2 --keep-surfaces 20
# likewise: s1, s1-prior-predictive, s1g, s2t-scale-1 (and 0.5, 2), s5, s10, s10e
#   -> results/calibration/rough/<name>; s8 -> results/calibration/spy/s8; s11 -> results/calibration/ewg/s11
python tutorial_final/studies/claims.py && python tutorial_final/studies/manifest.py
```

The bank and the shards stay on Oscar. Never commit raw quote files (`data/SOURCES.md`).

## What these runs fix

A review on 24 October found the following errors in earlier runs and scripts. All are fixed in the code above.

1. **The front-month case gave every method SPY's curve to one year**, read from the full chain. The same held
   for the 15-quote and three-times-noise checks. S10 and S11 give no method the curve.
2. **The pilot's audit priced all proposals on one seed**, and it was not unbiased. Every audit now prices each
   proposal on its own seed and uses the full Monte Carlo covariance.
3. **Classical fits started at the network's median.** They now start from the quotes and the prior box.
4. **The thin chain's noise used the vega at 20% volatility**, which understates the put wing's half-spread
   threefold. It now uses each quote's own vega.
5. **The forward came from put–call parity on American options**, which moves EWG's put and call volatilities apart
   by up to 1.3 half-spreads. The forward is now the carry forward.
6. **Maturities counted holidays as trading days**, which made EWG's April expiry 126 days instead of 120. NYSE
   holidays are now removed.
7. **The truth's products were priced on another seed than the methods'.** They now share the seed.
8. **A failed proposal dropped its whole surface from the audit.** It is now priced again with more paths, or
   counted.
9. **Shards were not safe to rerun, and the config was not frozen.** Both are fixed.
