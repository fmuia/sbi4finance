# Frozen Heston candidate check — 28 October 2026

Job **8032275 completed successfully**: all four marginal rank screens pass on the untouched 2,000-case panel, and all twelve cached analytic product-quantile comparisons are within the declared tolerance including Monte Carlo uncertainty. This is a passing check for one frozen candidate, not full tutorial scientific acceptance.

The candidate is the existing 100k run's final epoch 500, SHA-256 `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`. It was chosen using development diagnostics, then frozen before opening this test panel. The [declaration](manifest.json) fixes the model, task, bank, sampling policy, exact IDs, random seeds, metrics and product rules. Its SHA-256 is `6db210193e45b7d6af634ed77a5de17e5fbf7102c22298d18c13a3e1a141f525`.

## Held-out calibration result

| Parameter | Rank KS; band .0356201 | 50% coverage | 80% coverage | 90% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial variance | .0263922 | 52.25% | 81.30% | 90.80% | 95.20% |
| Long-run variance | .0178110 | 52.30% | 80.50% | 89.95% | 94.40% |
| Vol-of-vol | .0174502 | 50.30% | 80.30% | 90.00% | 94.55% |
| Rho | .0184105 | 50.60% | 80.15% | 90.20% | 94.80% |

All four rank statistics meet the predeclared simultaneous band. Initial- and long-run-variance 50% coverage have pointwise, unadjusted Wilson intervals just above nominal 50%; those deviations remain visible. The [full report](results/acceptance/evaluation.json) retains every coverage interval, contraction and recovery metric. Passing marginal screens does not establish joint or universal conditional accuracy. The historical archive was not evaluated on this new test panel, so this is not a paired archive-versus-candidate acceptance comparison.

## Data and decision scope

- Evaluate only `test:0` through `test:1999` from the checked new-root-271028 snapshot, with 1,000 draws per case, batch size 100, seeds `280028 + batch start`, and BayesFlow's native adaptive Tsit5 policy.
- The primary gate is the four-parameter marginal rank screen at family alpha .05, with simultaneous DKW bound .0356201271. Report central 50/80/90/95% coverage with pointwise Wilson intervals, contraction and recovery; those descriptive metrics are not additional undeclared gates.
- A metadata-only preflight inspected **23 retained Heston run/reference journals** and found no recorded prior test performance evaluation in those new-root runs. Generation/integrity checks did access test arrays. Historical old-root test outcomes remain separate. The preflight opened no test arrays; the exact journal paths and hashes are retained in the declaration.
- This evaluates **one frozen candidate**, not the later three-seed robustness study, independent conditional-library acceptance, joint calibration, or market validity. Retain the outcome regardless of pass or failure; do not use it to choose a different checkpoint.

## Analytic product check

In the same allocation, reuse the existing 10,000 final-model draws and four-chain, 8,000-draw-per-chain PyMC references for each of the three previously inspected development cases. No new neural conditional sampling, pricing or MCMC is needed.

Evaluate the analytic one-year variance-swap rate and six-to-twelve-month forward-variance rate. Both are annualized variance quantities. For each product's q10 and q90, use `T = max(.10 * reference 90% width, .0001)`. Compare errors with chain-aware ArviZ reference quantile MCSE and neural quantile MCSE. An approximate pointwise 95% uncertainty upper bound must be inside `T`, and its half-width at most `T/2`, to call the comparison within tolerance. A resolved excess and an inconclusive result remain distinct.

These products test uncertainty in initial and long-run variance. They do not test leverage, vol-of-vol or path-dependent pricing. In particular this job does not use the existing path-product forward-start implementation.

All **12/12** quantile comparisons pass the registered pointwise rule. The largest uncertainty-adjusted absolute quantile error is **60.1% of its tolerance**, for case 0's forward-variance q10. The [complete product report](results/products/products.json) retains both quantiles, means, reference diagnostics, tolerances and separate neural/reference MCSEs for every case/product pair.

| Development case | Product | Largest `(absolute error + uncertainty) / tolerance`, across q10/q90 |
| --- | --- | ---: |
| 0 | One-year variance-swap rate | .578 |
| 0 | Six-to-twelve-month forward variance | .601 |
| 1 | One-year variance-swap rate | .463 |
| 1 | Six-to-twelve-month forward variance | .404 |
| 2 | One-year variance-swap rate | .374 |
| 2 | Six-to-twelve-month forward variance | .275 |

## Execution and preservation

The [prospective ledger](execution-prospective.json) and declaration existed before the single scheduler submission; [the submitted ledger](execution-submitted.json) preserves its response. The hard cap is 25 minutes, four CPUs, one L40S GPU, 24 GiB, no requeue or automatic retry. Including a 240-second cleanup reserve, maximum exposure is **6,960 CPU-seconds / 1,740 GPU-seconds**, below separate **2 CPU-hour / .5 GPU-hour** ceilings.

The [driver](check_heston_candidate_20261028.py) commits every 100-case batch before proceeding. Products use the existing reference Python environment as a subprocess in the same allocation. The [metadata-only preparation/submission script](prepare_heston_candidate_20261028.py) is retained too. All 31 frozen source/input hashes are checked before evaluation and afterward. Large model/bank/draw files remain on Oscar.

Synthetic checks passed for analytic integrals, ArviZ chain-aware quantile MCSE and within-tolerance/resolved-excess/inconclusive classification. No notebooks were executed, and no model training or new bank generation is part of this unit.

Actual Slurm use was **975 seconds × four CPUs = 3,900 allocated CPU-seconds (1.0833 hours)** and **975 GPU-seconds (.2708 hours)**, below the separate limits. Sampling took 938.58 seconds; allocation accounting includes setup and the analytic checks. Count only the top-level allocation once; [accounting.psv](accounting.psv) also retains job-step records.

[execution.json](execution.json) reconciles the terminal state and actual resources. [The raw log](logs/slurm-8032275.out), [terminal summary](summary.json), [collector source](collect_heston_candidate_20261028.py) and [remote artifact inventory](remote-artifacts.json) retain the exact outcome. All 20 partial batches and the combined draw payload remain on Oscar; their 21 SHA-256 hashes and byte counts are retained without transferring the arrays. The original candidate and all prior selected-checkpoint failures remain unchanged.
